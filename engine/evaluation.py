"""Latent one-step and autoregressive rollout evaluation for TEP-LeWM."""
from __future__ import annotations
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from data.data_process import NormalizationStats, load_manifest
from engine.planning import (
    build_cem_planner, prepare_tep_planning_task, run_tep_mpc,
    standardized_xmeas_mse,
)
from engine.training import load_checkpoint
def predict_latent_trajectory(
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    *,
    split: str = "test",
    scenario_type: str = "multi_action",
    trajectory_id: str | None = None,
    start_step: int | None = None,
    horizon: int = 30,
    sample_stride_steps: int = 60,
    device: str = "auto",
) -> dict:
    """Roll out one representative trajectory and return predicted/true latents."""
    if horizon < 1 or sample_stride_steps < 1:
        raise ValueError("horizon and sample_stride_steps must be positive")
    device_obj = _resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device_obj)
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    root = Path(dataset_dir)
    records = load_manifest(root, split)
    candidates = [
        record for record in records
        if record["scenario_type"] == scenario_type
        and (trajectory_id is None or record["trajectory_id"] == trajectory_id)
    ]
    if not candidates:
        raise ValueError("no trajectory matches the requested split and scenario")
    record = next(
        (item for item in candidates if item.get("action_event_schedule")),
        candidates[0],
    )
    with np.load(root / record["file"], allow_pickle=False) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    history_size = int(model.history_size)
    maximum_start = len(actions) - (
        history_size + horizon - 1
    ) * sample_stride_steps
    if maximum_start < 0:
        raise ValueError("trajectory is too short for the requested rollout")
    if start_step is None:
        events = record.get("action_event_schedule", [])
        event_step = round(float(events[0]["time_hours"]) * 3600) if events else 0
        start_step = event_step - (history_size - 1) * sample_stride_steps
        start_step = round(start_step / sample_stride_steps) * sample_stride_steps
        start_step = min(max(start_step, 0), maximum_start)
    if start_step < 0 or start_step > maximum_start:
        raise ValueError("start_step is outside the valid rollout range")
    history_indices = start_step + np.arange(history_size) * sample_stride_steps
    future_indices = start_step + (
        history_size - 1 + np.arange(horizon)
    ) * sample_stride_steps
    target_indices = future_indices + sample_stride_steps
    observation_history = stats.normalize_observations(observations[history_indices])
    action_history = stats.normalize_actions(actions[history_indices[:-1]])
    future_actions = stats.normalize_actions(actions[future_indices])
    targets = stats.normalize_observations(observations[target_indices])
    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32, device=device_obj)[None]
    model.eval()
    with torch.inference_mode():
        initial = model.encode_observations(tensor(observation_history[-1:]))[0]
        predicted = model.rollout(
            tensor(observation_history), tensor(action_history), tensor(future_actions)
        )[0]
        actual = model.encode_observations(tensor(targets))[0]
    predicted = torch.cat([initial, predicted], dim=0).cpu().numpy()
    actual = torch.cat([initial, actual], dim=0).cpu().numpy()
    return {
        "trajectory_id": record["trajectory_id"],
        "scenario_type": record["scenario_type"],
        "start_step": int(start_step),
        "time_minutes": np.arange(horizon + 1) * sample_stride_steps / 60.0,
        "predicted_embeddings": predicted,
        "actual_embeddings": actual,
        "latent_mse": np.square(predicted - actual).mean(axis=-1),
    }
def evaluate_latent_prediction(
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    *,
    split: str = "test",
    scenario_types: Sequence[str] = ("single_action", "multi_action"),
    horizons: Sequence[int] = (1, 5, 10, 30),
    sample_stride_steps: int = 60,
    window_stride_steps: int = 60,
    batch_size: int = 256,
    max_windows: int | None = None,
    seed: int = 42,
    device: str = "auto",
    show_progress: bool = True,
) -> dict:
    """Evaluate latent MSE at selected rollout horizons on held-out trajectories."""
    device_obj = _resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device_obj)
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    history_size = int(model.history_size)
    horizons = tuple(sorted(set(map(int, horizons))))
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must contain positive integers")
    if min(sample_stride_steps, window_stride_steps, batch_size) < 1:
        raise ValueError("stride and batch parameters must be positive")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive when provided")
    windows = _collect_windows(
        dataset_dir, split, set(scenario_types), history_size, max(horizons),
        sample_stride_steps, window_stride_steps,
    )
    if max_windows is not None and len(windows["trajectory_id"]) > max_windows:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(
            len(windows["trajectory_id"]), size=int(max_windows), replace=False
        ))
        windows = {key: value[selected] for key, value in windows.items()}
    errors = []
    model.eval()
    with torch.inference_mode():
        starts = range(0, len(windows["trajectory_id"]), batch_size)
        for start in tqdm(
            starts, total=len(starts), desc="Latent rollout evaluation",
            unit="batch", disable=not show_progress,
        ):
            stop = start + batch_size
            observations = torch.as_tensor(
                stats.normalize_observations(windows["observation_history"][start:stop]),
                dtype=torch.float32, device=device_obj,
            )
            action_history = torch.as_tensor(
                stats.normalize_actions(windows["action_history"][start:stop]),
                dtype=torch.float32, device=device_obj,
            )
            future_actions = torch.as_tensor(
                stats.normalize_actions(windows["future_actions"][start:stop]),
                dtype=torch.float32, device=device_obj,
            )
            targets = torch.as_tensor(
                stats.normalize_observations(windows["targets"][start:stop]),
                dtype=torch.float32, device=device_obj,
            )
            predicted = model.rollout(observations, action_history, future_actions)
            target_embeddings = model.encode_observations(targets)
            errors.append(
                (predicted - target_embeddings).square().mean(dim=-1).cpu().numpy()
            )
    errors = np.concatenate(errors)
    rows = []
    for index in range(len(errors)):
        for horizon in horizons:
            rows.append({
                "trajectory_id": windows["trajectory_id"][index],
                "scenario_type": windows["scenario_type"][index],
                "start_step": int(windows["start_step"][index]),
                "horizon": horizon,
                "latent_mse": float(errors[index, horizon - 1]),
            })
    per_window = pd.DataFrame(rows)
    summary = _summarize(per_window)
    return {
        "summary": summary,
        "per_window": per_window,
        "window_count": len(windows["trajectory_id"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device_obj),
    }
def evaluate_planning_tasks(
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    config: dict,
    *,
    split: str | None = None,
    device: str = "auto",
    show_progress: bool = True,
    planner_overrides: dict | None = None,
) -> dict:
    """Run CEM-MPC on held-out reachable goals and report XMEAS success."""
    settings = config.get("planning", {})
    split = split or settings.get("split", "validation")
    scenarios = set(settings.get(
        "scenario_types", config.get("evaluation", {}).get(
            "scenario_types", ["single_action", "multi_action"]
        ),
    ))
    task_count = settings.get("num_tasks")
    task_count = None if task_count is None else int(task_count)
    goal_minutes = int(settings.get("goal_offset_minutes", 30))
    steps_per_control = int(settings.get("simulator_steps_per_control", 60))
    replan_every = int(settings.get("replan_every", 1))
    success_threshold = float(settings.get("success_mse_threshold", 0.10))
    if (task_count is not None and task_count < 1) or goal_minutes < 1 or success_threshold <= 0:
        raise ValueError("invalid planning evaluation settings")
    allowed_outcomes = set(settings.get(
        "allowed_outcomes", ["normal", "near_boundary", "recovered"]
    ))
    replay_threshold = float(settings.get("replay_mse_threshold", 1e-4))
    require_nontrivial = bool(settings.get("require_nontrivial_goal", True))
    root = Path(dataset_dir)
    records = [
        record for record in load_manifest(root, split)
        if record["scenario_type"] in scenarios
        and record.get("action_event_schedule")
        and not record.get("active_idvs", [])
    ]
    rng = np.random.default_rng(int(settings.get("seed", 42)))
    rng.shuffle(records)
    candidate_count = sum(len({round(float(e["time_hours"]) * 3600)
        for e in r["action_event_schedule"]}) for r in records)
    rejections = Counter()
    if not records:
        raise ValueError("no valid held-out action trajectory is available")
    planner = build_cem_planner(
        checkpoint_path, config, device=device, **(planner_overrides or {}))
    rows, cases = [], []
    history_size = int(planner.model.history_size)
    sample_stride = int(config["training"].get("sample_stride_steps", 60))
    control_steps = round(goal_minutes * 60 / steps_per_control)
    for record in tqdm(records, desc="CEM-MPC planning evaluation",
                       unit="trajectory", disable=not show_progress):
        if task_count is not None and len(rows) >= task_count:
            break
        scheduled_starts = sorted({round(float(e["time_hours"]) * 3600)
                                   for e in record["action_event_schedule"]})
        if record.get("outcome") not in allowed_outcomes:
            rejections["disallowed_outcome"] += len(scheduled_starts)
            continue
        with np.load(root / record["file"], allow_pickle=False) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
        starts = [start for start in scheduled_starts if
                  start - (history_size - 1) * sample_stride >= 0 and
                  start + control_steps * steps_per_control < len(observations)]
        rejections["insufficient_context"] += len(scheduled_starts) - len(starts)
        for start in rng.permutation(starts):
            if task_count is not None and len(rows) >= task_count:
                break
            start = int(start)
            goal = start + control_steps * steps_per_control
            history_indices = start - np.arange(
                history_size - 1, -1, -1) * sample_stride
            audit = prepare_tep_planning_task(
                config["data"], record, observations, actions, start, goal,
                planner.stats, success_threshold=success_threshold,
                replay_threshold=replay_threshold,
                require_nontrivial=require_nontrivial,
            )
            if not audit["valid"]:
                rejections[audit["reason"]] += 1
                continue
            generator = audit.pop("generator")
            result = run_tep_mpc(
                generator, planner, observations[history_indices],
                actions[history_indices[:-1]], observations[goal],
                control_steps=control_steps,
                simulator_steps_per_control=steps_per_control,
                replan_every=replan_every,
            )
            reference_indices = (
                start + np.arange(control_steps + 1) * steps_per_control)
            reference_action_indices = reference_indices[:-1]
            initial_error = standardized_xmeas_mse(
                observations[start], observations[goal], planner.stats
            )
            final_error = standardized_xmeas_mse(
                result["observations"][-1], observations[goal], planner.stats
            )
            improvement = 1.0 - final_error / max(initial_error, 1e-12)
            success = bool(result["alive"] and final_error <= success_threshold)
            row = {
                "trajectory_id": record["trajectory_id"],
                "task_id": f"{record['trajectory_id']}:{start}",
                "split": split,
                "mode": int(record["mode"]),
                "start_step": start,
                "replay_xmeas_mse": audit["replay_mse"],
                "oracle_xmeas_mse": audit["oracle_mse"],
                "no_action_xmeas_mse": audit["no_action_mse"],
                "scenario_type": record["scenario_type"],
                "initial_xmeas_mse": initial_error,
                "final_xmeas_mse": final_error,
                "improvement": improvement,
                "success": success,
                "alive": bool(result["alive"]),
            }
            rows.append(row)
            cases.append({
                **row,
                "time_minutes": np.arange(control_steps + 1) * steps_per_control / 60,
                "reference_observations": observations[reference_indices],
                "planned_observations": result["observations"],
                "reference_actions": actions[reference_action_indices],
                "planned_actions": result["actions"],
                "goal_observation": observations[goal],
                "normalization": planner.stats,
                "action_sp_numbers": tuple(config["data"]["action_sp_numbers"]),
            })
    if not rows:
        raise ValueError("no trajectory is long enough for the configured planning task")
    per_task = pd.DataFrame(rows)
    summary = _planning_summary(per_task, success_threshold)
    pool = [case for case in cases if case["success"]] or cases
    typical = min(pool, key=lambda case: abs(
        case["improvement"] - np.median([item["improvement"] for item in pool])
    ))
    return {
        "summary": summary, "per_task": per_task,
        "typical_case": typical, "cases": cases,
        "split": split,
        "screening": {"candidate_count": candidate_count,
                      "screened_count": len(rows) + sum(rejections.values()),
                      "accepted_count": len(rows),
                      "rejections": dict(rejections)},
    }
def _planning_summary(per_task: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate overall and scenario-specific planning success metrics."""
    groups = [("overall", per_task), *per_task.groupby("scenario_type")]
    return pd.DataFrame([{
        "scenario_type": name,
        "task_count": len(group),
        "trajectory_count": group["trajectory_id"].nunique(),
        "success_rate": group["success"].mean(),
        "shutdown_rate": 1.0 - group["alive"].mean(),
        "initial_xmeas_mse": group["initial_xmeas_mse"].mean(),
        "final_xmeas_mse": group["final_xmeas_mse"].mean(),
        "oracle_xmeas_mse": group["oracle_xmeas_mse"].mean(),
        "no_action_xmeas_mse": group["no_action_xmeas_mse"].mean(),
        "mean_improvement": group["improvement"].mean(),
        "success_mse_threshold": threshold,
    } for name, group in groups])
def _collect_windows(
    dataset_dir, split, scenario_types, history_size, horizon,
    sample_stride, window_stride,
) -> dict[str, np.ndarray]:
    """Load aligned H-state context and future action/target windows."""
    root = Path(dataset_dir)
    records = [
        record for record in load_manifest(root, split)
        if record["scenario_type"] in scenario_types
    ]
    collected = {
        "trajectory_id": [], "scenario_type": [], "start_step": [],
        "observation_history": [], "action_history": [],
        "future_actions": [], "targets": [],
    }
    required_steps = (history_size + horizon - 1) * sample_stride
    for record in records:
        with np.load(root / record["file"], allow_pickle=False) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
        last_start = len(actions) - required_steps
        for start in range(0, last_start + 1, window_stride):
            history_indices = start + np.arange(history_size) * sample_stride
            action_indices = history_indices[:-1]
            future_indices = start + (
                history_size - 1 + np.arange(horizon)
            ) * sample_stride
            target_indices = future_indices + sample_stride
            collected["trajectory_id"].append(record["trajectory_id"])
            collected["scenario_type"].append(record["scenario_type"])
            collected["start_step"].append(start)
            collected["observation_history"].append(observations[history_indices])
            collected["action_history"].append(actions[action_indices])
            collected["future_actions"].append(actions[future_indices])
            collected["targets"].append(observations[target_indices])
    if not collected["trajectory_id"]:
        raise ValueError("no evaluation windows match the requested split and scenarios")
    return {key: np.asarray(value) for key, value in collected.items()}
def _summarize(per_window: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trajectory-window errors by scenario and horizon."""
    metrics = {"latent_mse": ["mean", "median", "std", "count"]}
    by_scenario = per_window.groupby(
        ["scenario_type", "horizon"], as_index=False
    ).agg(metrics)
    overall = per_window.groupby("horizon", as_index=False).agg(metrics)
    overall.insert(0, "scenario_type", "overall")
    summary = pd.concat([overall, by_scenario], ignore_index=True)
    summary.columns = [
        "scenario_type", "horizon", "latent_mse_mean", "latent_mse_median",
        "latent_mse_std", "window_count",
    ]
    return summary
def _resolve_device(value: str) -> torch.device:
    """Resolve an auto/cpu/cuda evaluation device."""
    if value == "auto": value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)
