"""TEP planning-task replay, safety screening and physical goal metrics."""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from data.data_gen import TEPDataGenerator
from data.data_process import NormalizationStats
from data.data_validation import classify_trajectory_outcome


def replay_tep_generator(
    data_config: Mapping,
    record: Mapping,
    actions: np.ndarray,
    stop_step: int,
) -> tuple[TEPDataGenerator, np.ndarray]:
    """Reconstruct a recorded simulator state before a planning task."""
    generator = TEPDataGenerator({
        **data_config,
        "seed": int(record["seed"]),
        "operating_mode": int(record["mode"]),
        "warmup_hours": float(record["warmup_hours"]),
        "action_events": [],
        "mode_events": [],
        "disturbance_events": [],
    })
    observation = generator.get_observation()
    for step in range(int(stop_step)):
        transition = generator.step(actions[step])
        observation = np.asarray(transition["next_observation"]).copy()
        if not transition["alive"]:
            raise RuntimeError(
                f"{record['trajectory_id']} shut down during replay"
            )
    return generator, observation
def execute_action_sequence(
    generator: TEPDataGenerator,
    actions: np.ndarray,
) -> dict[str, np.ndarray | bool]:
    """Execute a one-second action sequence for oracle and hold baselines."""
    observations = [generator.get_observation()]
    alive = True
    for action in np.asarray(actions):
        transition = generator.step(action)
        observations.append(
            np.asarray(transition["next_observation"]).copy()
        )
        alive = bool(transition["alive"])
        if not alive:
            break
    return {
        "observations": np.asarray(observations),
        "alive": alive,
    }
def standardized_xmeas_mse(
    observation: np.ndarray,
    goal: np.ndarray,
    stats: NormalizationStats,
) -> float:
    """Measure goal error over the 41 standardized process measurements."""
    difference = np.asarray(observation)[:41] - np.asarray(goal)[:41]
    return float(np.mean(np.square(difference / stats.observation_std[:41])))
def prepare_tep_planning_task(
    data_config: Mapping,
    record: Mapping,
    observations: np.ndarray,
    actions: np.ndarray,
    start: int,
    goal: int,
    stats: NormalizationStats,
    *,
    success_threshold: float,
    replay_threshold: float,
    require_nontrivial: bool,
) -> dict:
    """Validate safety, replay fidelity, oracle reachability and task difficulty."""
    outcome, _ = classify_trajectory_outcome(
        observations[goal : goal + 1],
        data_config.get("safety_limits", {}),
        shutdown=False,
        terminated_early=False,
        boundary_margin_fraction=float(
            data_config.get("boundary_margin_fraction", 0.05)
        ),
    )
    if outcome == "limit_exceeded":
        return {"valid": False, "reason": "unsafe_goal"}
    planner_generator, replayed = replay_tep_generator(
        data_config, record, actions, start
    )
    replay_error = standardized_xmeas_mse(replayed, observations[start], stats)
    if replay_error > replay_threshold:
        return {"valid": False, "reason": "replay_mismatch"}
    oracle_generator = copy.deepcopy(planner_generator)
    oracle = execute_action_sequence(oracle_generator, actions[start:goal])
    oracle_error = standardized_xmeas_mse(
        oracle["observations"][-1], observations[goal], stats
    )
    if not oracle["alive"] or oracle_error > replay_threshold:
        return {"valid": False, "reason": "oracle_unreachable"}
    _, oracle_violations = classify_trajectory_outcome(
        oracle["observations"], data_config.get("safety_limits", {}),
        shutdown=False, terminated_early=False,
    )
    if any(oracle_violations.values()):
        return {"valid": False, "reason": "unsafe_reference"}
    hold_generator = copy.deepcopy(planner_generator)
    hold = np.broadcast_to(actions[max(start - 1, 0)], (goal - start, actions.shape[1]))
    no_action = execute_action_sequence(hold_generator, hold)
    no_action_error = standardized_xmeas_mse(
        no_action["observations"][-1], observations[goal], stats
    )
    _, hold_violations = classify_trajectory_outcome(
        no_action["observations"], data_config.get("safety_limits", {}),
        shutdown=not no_action["alive"], terminated_early=not no_action["alive"],
    )
    hold_safe = no_action["alive"] and not any(hold_violations.values())
    if require_nontrivial and hold_safe and no_action_error <= success_threshold:
        return {"valid": False, "reason": "trivial_goal"}
    return {
        "valid": True, "reason": "accepted",
        "generator": planner_generator, "replay_mse": replay_error,
        "oracle_mse": oracle_error,
        "hold_generator": hold_generator, "hold_result": no_action,
    }
