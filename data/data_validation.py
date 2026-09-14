"""Validation and dataset-level audits for generated TEP trajectories."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from data.data_gen import Trajectory

SAMPLE_INTERVAL_HOURS = 1.0 / 3600.0
DEFAULT_ACTION_SP_NUMBERS = (5, 7, 8, 11, 13, 14, 15, 17, 18, 19, 20)


def model_step_seconds(data_config: Mapping[str, Any]) -> int:
    """Convert the configured model interval to an exact number of 1-second steps."""
    seconds = float(data_config["model_step_minutes"]) * 60
    if (
        not np.isfinite(seconds) or seconds < 1
        or not np.isclose(seconds, round(seconds), rtol=0, atol=1e-8)
    ):
        raise ValueError("model_step_minutes must be positive and aligned to whole seconds")
    return round(seconds)


def validate_action_sampling(actions: np.ndarray, stride: int) -> None:
    """Require each complete model transition to have one held SP action."""
    if stride < 1:
        raise ValueError("model step must be positive")
    complete = len(actions) // stride * stride
    blocks = actions[:complete].reshape(-1, stride, actions.shape[-1])
    if not np.all(blocks == blocks[:, :1]):
        raise ValueError(
            "SP actions change within a model step; regenerate trajectories "
            "on the configured model time grid"
        )


def action_change_counts(actions: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Count actual SP changes at each event, including previous -> first action."""
    actions = np.asarray(actions)
    preceding = np.concatenate([np.asarray(previous)[None], actions[:-1]], axis=0)
    return (~np.isclose(actions, preceding, rtol=1e-6, atol=1e-6)).sum(axis=-1)


def matches_action_window(counts: np.ndarray, scenario_type: str) -> bool:
    """Require repeated events and the requested single/multiple-SP pattern."""
    if np.count_nonzero(counts) < 2:
        return False
    if scenario_type == "single_action":
        return bool(np.all(counts <= 1))
    if scenario_type == "multi_action":
        return bool(np.any(counts >= 2))
    return True


def validate_trajectory(
    trajectory: Trajectory,
    *,
    expected_duration_hours: float | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """执行轨迹入库前的维度、数值、时序和完整性质量检查。"""
    steps = len(trajectory.actions)
    expected_shapes = {
        "time": (steps + 1,),
        "observations": (steps + 1, 53),
        "setpoints": (steps, 20),
        "disturbances": (steps, 20),
        "operating_modes": (steps,),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = getattr(trajectory, name).shape
        if actual_shape != expected_shape:
            raise ValueError(f"{name} has shape {actual_shape}, expected {expected_shape}")

    action_dim = len(trajectory.metadata["action_sp_numbers"])
    if trajectory.actions.shape != (steps, action_dim):
        raise ValueError("actions have an invalid shape")
    for name in ("time", "observations", "actions", "setpoints"):
        if not np.all(np.isfinite(getattr(trajectory, name))):
            raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.isin(trajectory.disturbances, (0, 1))):
        raise ValueError("disturbances must be binary")
    if not np.all(np.isin(trajectory.operating_modes, range(1, 7))):
        raise ValueError("operating modes must be between 1 and 6")
    if steps == 0 or not np.isclose(trajectory.time[0], 0.0):
        raise ValueError("trajectory must contain at least one transition from t=0")
    if not np.allclose(np.diff(trajectory.time), SAMPLE_INTERVAL_HOURS):
        raise ValueError("trajectory must use continuous 1-second sampling")
    if require_complete and trajectory.terminated_early:
        raise ValueError("trajectory terminated before the requested duration")
    if (
        expected_duration_hours is not None
        and not trajectory.terminated_early
        and not np.isclose(
            trajectory.time[-1], expected_duration_hours, atol=0.5 / 3600.0
        )
    ):
        raise ValueError("trajectory duration does not match the request")

    return {
        "transitions": steps,
        "duration_hours": float(trajectory.time[-1]),
        "observation_dim": 53,
        "action_dim": action_dim,
        "trajectory_type": trajectory.trajectory_type,
        "outcome": trajectory.outcome,
        "shutdown": trajectory.shutdown,
        "terminated_early": trajectory.terminated_early,
        "accepted": True,
    }


def classify_trajectory_outcome(
    observations: np.ndarray,
    safety_limits: Mapping[str, Sequence[float | None]],
    *,
    shutdown: bool,
    terminated_early: bool,
    boundary_margin_fraction: float = 0.05,
) -> tuple[str, dict[str, int]]:
    """依据安全边界和停车状态标注轨迹结局并返回超限详情。"""
    if not isinstance(safety_limits, Mapping):
        raise TypeError("safety_limits must be a mapping")
    if not 0.0 <= boundary_margin_fraction < 0.5:
        raise ValueError("boundary_margin_fraction must be in [0, 0.5)")

    outside_any = np.zeros(len(observations), dtype=bool)
    near_any = np.zeros(len(observations), dtype=bool)
    violations: dict[str, int] = {}
    for key, bounds in safety_limits.items():
        number = _parse_xmeas_number(key)
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ValueError(f"{key} safety limit must be [lower, upper]")
        lower = -np.inf if bounds[0] is None else float(bounds[0])
        upper = np.inf if bounds[1] is None else float(bounds[1])
        if lower >= upper:
            raise ValueError(f"{key} safety lower bound must be below upper bound")

        values = observations[:, number - 1]
        outside = (values < lower) | (values > upper)
        outside_any |= outside
        violations[f"XMEAS{number}"] = int(outside.sum())
        if np.isfinite(lower) and np.isfinite(upper):
            margin = boundary_margin_fraction * (upper - lower)
            near_any |= (~outside) & (
                (values <= lower + margin) | (values >= upper - margin)
            )

    if shutdown:
        outcome = "shutdown"
    elif terminated_early:
        outcome = "terminated"
    elif outside_any[-1]:
        outcome = "limit_exceeded"
    elif outside_any.any():
        outcome = "recovered"
    elif near_any.any():
        outcome = "near_boundary"
    else:
        outcome = "normal"
    return outcome, violations


def summarize_action_coverage(
    trajectories: Sequence[Trajectory],
    expected_sp_numbers: Sequence[int] = DEFAULT_ACTION_SP_NUMBERS,
) -> dict[str, Any]:
    """在数据集层面统计每个动作SP的变化次数和总体覆盖率。"""
    expected = tuple(int(number) for number in expected_sp_numbers)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_sp_numbers must be non-empty and unique")
    counts = {number: 0 for number in expected}
    for trajectory in trajectories:
        for number in trajectory.metadata.get("changed_action_sp_numbers", ()):
            if number in counts:
                counts[number] += 1
    covered = [number for number, count in counts.items() if count]
    missing = [number for number, count in counts.items() if not count]
    return {
        "trajectory_count": len(trajectories),
        "counts": counts,
        "covered_sp_numbers": covered,
        "missing_sp_numbers": missing,
        "coverage_ratio": len(covered) / len(expected),
    }


def audit_dataset(
    dataset_dir: str | Path,
    expected_sp_numbers: Sequence[int] = DEFAULT_ACTION_SP_NUMBERS,
) -> dict[str, Any]:
    """Audit manifest integrity, split coverage and paired counterfactual groups."""
    root = Path(dataset_dir)
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing dataset manifest: {manifest}")
    payload = manifest.read_bytes()
    records = [
        json.loads(line) for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [str(record["trajectory_id"]) for record in records]
    duplicate_ids = sorted(
        key for key, count in Counter(identifiers).items() if count > 1
    )
    missing_files = sorted(
        record["file"] for record in records
        if not (root / record["file"]).is_file()
    )
    expected = tuple(map(int, expected_sp_numbers))
    splits = sorted({str(record["split"]) for record in records})
    action_counts = {
        split: Counter({number: 0 for number in expected}) for split in splits
    }
    single_action_counts = {
        split: Counter({number: 0 for number in expected}) for split in splits
    }
    counterfactual_groups = defaultdict(list)
    group_splits = defaultdict(set)
    for record in records:
        split = str(record["split"])
        group_splits[str(record["group_id"])].add(split)
        changed = set(map(int, record.get("changed_action_sp_numbers", ())))
        action_counts[split].update(changed)
        if record["scenario_type"] == "single_action":
            single_action_counts[split].update(changed)
        if record.get("counterfactual_role"):
            counterfactual_groups[str(record["group_id"])].append(record)
    pair_errors = []
    leaking_groups = sorted(group for group, values in group_splits.items() if len(values) > 1)
    for group_id, pair in counterfactual_groups.items():
        roles = {record["counterfactual_role"] for record in pair}
        pair_splits = {record["split"] for record in pair}
        if roles != {"reference", "intervention"} or len(pair_splits) != 1:
            pair_errors.append(group_id)
    missing_actions = {
        split: [number for number, count in counts.items() if count == 0]
        for split, counts in action_counts.items()
    }
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    manifest_hash = hashlib.sha256(payload).hexdigest()
    fingerprint_matches = summary.get("manifest_sha256") in (None, manifest_hash)
    scenarios = sorted({str(record["scenario_type"]) for record in records})
    scenario_counts_by_split = {
        split: dict(Counter(
            record["scenario_type"] for record in records
            if record["split"] == split
        )) for split in splits
    }
    missing_scenarios = {
        split: sorted(set(scenarios) - set(counts))
        for split, counts in scenario_counts_by_split.items()
    }
    mode_switch_targets = Counter(
        str(event["mode"]) for record in records
        for event in record.get("mode_event_schedule", ())
    )
    initialization_attempts = Counter(
        int(record.get("initialization_attempts", 1)) for record in records
    )
    return {
        "accepted": not duplicate_ids and not missing_files
                    and not pair_errors and not leaking_groups and fingerprint_matches
                    and not any(missing_actions.values())
                    and not any(missing_scenarios.values()),
        "trajectory_count": len(records),
        "scenario_counts": dict(Counter(r["scenario_type"] for r in records)),
        "outcome_counts": dict(Counter(r["outcome"] for r in records)),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "mode_counts": dict(Counter(str(r["mode"]) for r in records)),
        "scenario_counts_by_split": scenario_counts_by_split,
        "missing_scenarios_by_split": missing_scenarios,
        "outcome_counts_by_scenario": {
            scenario: dict(Counter(
                r["outcome"] for r in records if r["scenario_type"] == scenario
            )) for scenario in scenarios},
        "mode_switch_target_counts": dict(mode_switch_targets),
        "initialization_attempt_counts": dict(initialization_attempts),
        "action_counts_by_split": {k: dict(v) for k, v in action_counts.items()},
        "single_action_counts_by_split": {
            k: dict(v) for k, v in single_action_counts.items()},
        "missing_actions_by_split": missing_actions,
        "duplicate_trajectory_ids": duplicate_ids,
        "missing_files": missing_files,
        "counterfactual_pair_errors": sorted(pair_errors),
        "cross_split_groups": leaking_groups,
        "manifest_sha256": manifest_hash,
        "fingerprint_matches": fingerprint_matches,
    }


def _parse_xmeas_number(value: int | str) -> int:
    """将整数或XMEAS7形式的标识统一解析为测量变量编号。"""
    if isinstance(value, str):
        value = value.upper().removeprefix("XMEAS")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid XMEAS identifier: {value!r}") from error
    if number not in range(1, 42):
        raise ValueError("XMEAS number must be between 1 and 41")
    return number
