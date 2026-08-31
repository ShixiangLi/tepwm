"""Validation and dataset-level audits for generated TEP trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from data.data_gen import Trajectory

SAMPLE_INTERVAL_HOURS = 1.0 / 3600.0
DEFAULT_ACTION_SP_NUMBERS = (5, 7, 8, 11, 13, 14, 15, 17, 18, 19, 20)


def validate_trajectory(
    trajectory: Trajectory,
    *,
    expected_duration_hours: float | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Run the structural quality gate used before dataset admission."""
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
    """Label a trajectory as normal, boundary, exceeded, recovered or shutdown."""
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
    """Summarize action coverage across a dataset rather than per trajectory."""
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


def _parse_xmeas_number(value: int | str) -> int:
    if isinstance(value, str):
        value = value.upper().removeprefix("XMEAS")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid XMEAS identifier: {value!r}") from error
    if number not in range(1, 42):
        raise ValueError("XMEAS number must be between 1 and 41")
    return number
