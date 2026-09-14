"""Parallel held-out planning evaluation for the TEP world model."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from data.data_process import load_manifest
from data.data_validation import (action_change_counts, matches_action_window,
                                  model_step_seconds, validate_action_sampling)
from engine.planning import (
    build_cem_planner,
    prepare_tep_planning_task,
    run_tep_mpc,
    standardized_xmeas_mse,
)

_WORKER: dict = {}


def evaluate_planning_tasks(
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    config: dict,
    *,
    split: str | None = None,
    device: str = "auto",
    show_progress: bool = True,
    planner_overrides: dict | None = None,
    num_workers: int | None = None,
) -> dict:
    """Evaluate deterministic planning tasks concurrently across available GPUs."""
    settings = config.get("planning", {})
    split = split or settings.get("split", "validation")
    task_count = settings.get("num_tasks")
    task_count = None if task_count is None else int(task_count)
    workers = int(settings.get("num_workers", 1) if num_workers is None else num_workers)
    if workers < 1 or (task_count is not None and task_count < 1):
        raise ValueError("planning worker and task counts must be positive")
    devices = _resolve_devices(device, workers)
    candidates, pool_info = _build_candidates(dataset_dir, config, split)
    if not candidates:
        raise ValueError("no valid held-out action trajectory is available")

    init_args = (
        str(checkpoint_path), str(dataset_dir), config,
        planner_overrides or {}, devices,
    )
    rows, cases = [], []
    runtime_rejections = Counter()
    evaluated = 0
    progress = tqdm(
        total=len(candidates), desc="CEM-MPC planning evaluation", unit="task",
        disable=not show_progress,
    )
    executor = None
    if len(devices) == 1:
        _initialize_worker(*init_args, fixed_device=devices[0])
    else:
        executor = ProcessPoolExecutor(
            max_workers=len(devices), mp_context=mp.get_context("spawn"),
            initializer=_initialize_worker, initargs=init_args,
        )
    try:
        cursor = 0
        while cursor < len(candidates) and (
            task_count is None or len(rows) < task_count
        ):
            remaining = len(candidates) if task_count is None else task_count - len(rows)
            batch_size = len(candidates) if task_count is None else remaining + len(devices)
            batch = candidates[cursor : cursor + batch_size]
            results = map(_evaluate_candidate, batch) if executor is None else executor.map(
                _evaluate_candidate, batch, chunksize=1
            )
            for result in results:
                evaluated += 1
                progress.update(1)
                if result["reason"]:
                    runtime_rejections[result["reason"]] += 1
                    continue
                if task_count is not None and len(rows) >= task_count:
                    continue
                rows.append(result["row"])
                cases.append(result["case"])
            cursor += len(batch)
    finally:
        progress.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if not rows:
        raise ValueError(f"no planning task passed screening: {dict(runtime_rejections)}")

    per_task = pd.DataFrame(rows)
    threshold = float(settings.get("success_mse_threshold", 0.10))
    summary = _planning_summary(per_task, threshold)
    pool = [case for case in cases if case["success"]] or cases
    typical = min(pool, key=lambda case: abs(
        case["improvement"] - np.median([item["improvement"] for item in pool])
    ))
    rejections = Counter(pool_info["rejections"])
    rejections.update(runtime_rejections)
    return {
        "summary": summary, "per_task": per_task,
        "typical_case": typical, "cases": cases, "split": split,
        "devices": devices,
        "screening": {
            "candidate_count": pool_info["candidate_count"],
            "evaluated_candidate_count": evaluated,
            "accepted_count": len(rows), "rejections": dict(rejections),
        },
    }


def _build_candidates(dataset_dir, config, split):
    """Build a deterministic ordered candidate pool without running simulation."""
    settings = config.get("planning", {})
    scenarios = set(settings.get(
        "scenario_types", config.get("evaluation", {}).get(
            "scenario_types", ["single_action", "multi_action"]
        ),
    ))
    allowed = set(settings.get("allowed_outcomes", ["normal", "near_boundary", "recovered"]))
    records = [
        record for record in load_manifest(Path(dataset_dir), split)
        if record["scenario_type"] in scenarios
        and record.get("actual_action_event_steps")
        and not record.get("active_idvs", [])
    ]
    rng = np.random.default_rng(int(settings.get("seed", 42)))
    rng.shuffle(records)
    history = int(config["model"]["history_size"])
    stride = model_step_seconds(config["data"])
    control_steps = int(settings["goal_offset_steps"])
    if control_steps < 1 or control_steps != settings["goal_offset_steps"]:
        raise ValueError("goal_offset_steps must be a positive integer")
    candidates, rejections = [], Counter()
    candidate_count = 0
    for record in records:
        if int(record["model_step_seconds"]) != stride:
            raise ValueError("dataset model step differs from configuration")
        starts = record["actual_action_event_steps"]
        candidate_count += len(starts)
        if record.get("outcome") not in allowed:
            rejections["disallowed_outcome"] += len(starts)
            continue
        if any(start % stride for start in starts):
            raise ValueError("planning action events must align with the model time grid")
        valid = [start for start in starts if
                 start - (history - 1) * stride >= 0 and
                 start + control_steps * stride < int(record["transitions"]) + 1]
        rejections["insufficient_context"] += len(starts) - len(valid)
        for start in rng.permutation(valid):
            task_id = f"{record['trajectory_id']}:{int(start)}"
            digest = hashlib.sha256(task_id.encode()).digest()
            task_seed = int(settings.get("seed", 42)) + int.from_bytes(digest[:4], "little")
            candidates.append((record, int(start), task_seed))
    return candidates, {"candidate_count": candidate_count, "rejections": rejections}


def _initialize_worker(
    checkpoint_path, dataset_dir, config, overrides, devices, fixed_device=None,
):
    """Load one model per process and bind that process to one evaluation device."""
    if fixed_device is None:
        identity = mp.current_process()._identity
        index = (identity[0] - 1) % len(devices) if identity else 0
        device = devices[index]
    else:
        device = fixed_device
    _WORKER.clear()
    _WORKER.update(
        root=Path(dataset_dir), config=config,
        planner=build_cem_planner(checkpoint_path, config, device=device, **overrides),
    )


def _evaluate_candidate(candidate):
    """Screen and execute one task using the process-local model and simulator."""
    record, start, task_seed = candidate
    root, config, planner = _WORKER["root"], _WORKER["config"], _WORKER["planner"]
    settings = config.get("planning", {})
    sim_stride = model_step_seconds(config["data"])
    control_steps = int(settings["goal_offset_steps"])
    goal = start + control_steps * sim_stride
    history = int(planner.model.history_size)
    indices = start - np.arange(history - 1, -1, -1) * sim_stride
    with np.load(root / record["file"], allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    validate_action_sampling(actions, sim_stride)
    previous = actions[max(start - 1, 0)]
    reference_actions = actions[start:goal:sim_stride]
    counts = action_change_counts(reference_actions, previous)
    if not matches_action_window(counts, record["scenario_type"]):
        return {"reason": "insufficient_action_changes", "row": None, "case": None}
    preceding = np.concatenate([previous[None], reference_actions[:-1]], axis=0)
    action_range = planner.physical_upper - planner.physical_lower
    tolerance = 1e-6 + 1e-6 * action_range
    reference_with_initial = np.concatenate([previous[None], reference_actions])
    if (
        np.any(reference_with_initial < planner.physical_lower - tolerance)
        or np.any(reference_with_initial > planner.physical_upper + tolerance)
        or np.any(counts > planner.max_changed_actions)
        or np.any(np.abs(reference_actions - preceding) >
                  planner.max_action_delta_fraction * action_range + tolerance)
    ):
        return {"reason": "reference_action_constraints", "row": None, "case": None}
    audit = prepare_tep_planning_task(
        config["data"], record, observations, actions, start, goal, planner.stats,
        success_threshold=float(settings.get("success_mse_threshold", 0.10)),
        replay_threshold=float(settings.get("replay_mse_threshold", 1e-4)),
        require_nontrivial=bool(settings.get("require_nontrivial_goal", True)),
    )
    if not audit["valid"]:
        return {"reason": audit["reason"], "row": None, "case": None}
    generator = audit.pop("generator")
    planner.generator.manual_seed(task_seed)
    result = run_tep_mpc(
        generator, planner, observations[indices], actions[indices[:-1]],
        observations[goal], control_steps=control_steps,
        simulator_steps_per_control=sim_stride,
        replan_every=int(settings.get("replan_every", 1)),
    )
    reference_indices = start + np.arange(control_steps + 1) * sim_stride
    initial = standardized_xmeas_mse(observations[start], observations[goal], planner.stats)
    final = standardized_xmeas_mse(result["observations"][-1], observations[goal], planner.stats)
    threshold = float(settings.get("success_mse_threshold", 0.10))
    row = {
        "trajectory_id": record["trajectory_id"],
        "task_id": f"{record['trajectory_id']}:{start}", "split": record["split"],
        "mode": int(record["mode"]), "start_step": start,
        "replay_xmeas_mse": audit["replay_mse"], "oracle_xmeas_mse": audit["oracle_mse"],
        "no_action_xmeas_mse": audit["no_action_mse"],
        "no_action_alive": bool(audit["no_action_alive"]),
        "no_action_soft_limit_violated": bool(audit["no_action_soft_limit_violated"]),
        "no_action_success": bool(audit["no_action_success"]),
        "reference_action_event_count": int(np.count_nonzero(counts)),
        "reference_multi_sp_event_count": int(np.count_nonzero(counts >= 2)),
        "planned_action_event_count": int(np.count_nonzero(action_change_counts(result["actions"], previous))),
        "soft_limit_violated": bool(result["soft_limit_violated"]),
        "control_steps": control_steps,
        "planning_horizon": planner.horizon,
        "elapsed_minutes": float(result["time_minutes"][-1]),
        "scenario_type": record["scenario_type"], "initial_xmeas_mse": initial,
        "final_xmeas_mse": final, "improvement": 1.0 - final / max(initial, 1e-12),
        "success": bool(result["alive"] and not result["soft_limit_violated"] and final <= threshold),
        "alive": bool(result["alive"]),
    }
    case = {
        **row, "time_minutes": np.arange(control_steps + 1) * sim_stride / 60,
        "planned_time_minutes": result["time_minutes"],
        "limit_violations": result["limit_violations"],
        "reference_observations": observations[reference_indices],
        "planned_observations": result["observations"],
        "reference_actions": actions[reference_indices[:-1]],
        "planned_actions": result["actions"], "goal_observation": observations[goal],
        "normalization": planner.stats,
        "action_sp_numbers": tuple(config["data"]["action_sp_numbers"]),
    }
    return {"reason": None, "row": row, "case": case}


def _resolve_devices(device: str, workers: int) -> list[str]:
    """Resolve one process per visible GPU, or the requested CPU worker count."""
    if device in {"auto", "cuda"} and torch.cuda.is_available():
        count = min(workers, torch.cuda.device_count())
        return [f"cuda:{index}" for index in range(count)]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.startswith("cuda:"):
        return [device]
    return ["cpu"] * workers


def _planning_summary(per_task: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate overall and scenario-specific planning success metrics."""
    groups = [("overall", per_task), *per_task.groupby("scenario_type")]
    return pd.DataFrame([{
        "scenario_type": name, "task_count": len(group),
        "trajectory_count": group["trajectory_id"].nunique(),
        "success_rate": group["success"].mean(),
        "shutdown_rate": 1.0 - group["alive"].mean(),
        "soft_limit_violation_rate": group["soft_limit_violated"].mean(),
        "no_action_success_rate": group["no_action_success"].mean(),
        "no_action_shutdown_rate": 1.0 - group["no_action_alive"].mean(),
        "no_action_soft_limit_violation_rate": group["no_action_soft_limit_violated"].mean(),
        "reference_action_event_count": group["reference_action_event_count"].mean(),
        "planned_action_event_count": group["planned_action_event_count"].mean(),
        "initial_xmeas_mse": group["initial_xmeas_mse"].mean(),
        "final_xmeas_mse": group["final_xmeas_mse"].mean(),
        "oracle_xmeas_mse": group["oracle_xmeas_mse"].mean(),
        "no_action_xmeas_mse": group["no_action_xmeas_mse"].mean(),
        "mean_improvement": group["improvement"].mean(),
        "success_mse_threshold": threshold,
    } for name, group in groups])
