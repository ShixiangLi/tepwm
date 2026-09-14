"""Configuration-driven batch generation for TEP world-model datasets."""
from __future__ import annotations
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
import numpy as np
from tqdm.auto import tqdm
from data.data_gen import TEPDataGenerator, Trajectory
from data.data_validation import (action_change_counts, matches_action_window,
                                  model_step_seconds, validate_action_sampling, validate_trajectory)
SCENARIO_TYPES = ("fixed_sp", "single_action", "multi_action", "disturbance",
                  "boundary_action", "mode_switch")
def build_generation_plan(
    data_config: Mapping[str, Any],
    num_trajectories: int | None = None,
    counterfactual_pairs: int | None = None,
) -> list[dict[str, Any]]:
    """根据配置构建可复现的轨迹计划，但不实际运行TEP。"""
    data = dict(data_config)
    generation = dict(data.get("generation", {}))
    if not generation:
        raise ValueError("data config must contain a generation mapping")
    count = int(generation.get("num_trajectories", 300)
                if num_trajectories is None else num_trajectories)
    pair_count = int(generation.get("counterfactual_pairs", 0)
                     if counterfactual_pairs is None else counterfactual_pairs)
    if count <= 0 or pair_count < 0:
        raise ValueError("trajectory count must be positive and pair count non-negative")
    seed = int(generation.get("seed", data.get("seed", 42)))
    step_seconds = model_step_seconds(data)
    rng = np.random.default_rng(seed)
    action_sps = tuple(map(int, data["action_sp_numbers"]))
    modes = tuple(map(int, generation.get("operating_modes", range(1, 7))))
    base = {key: value for key, value in data.items() if key not in {"generation", "output_dir"}}
    nominal = _load_nominal_actions(base, modes, action_sps, generation["action"]["bounds"])
    if not data.get("safety_limits") and generation["scenario_proportions"]["boundary_action"]:
        raise ValueError("boundary_action requires explicit experimental safety_limits")
    scenario_types = _allocate_scenario_types(count, generation, rng)
    plans = []
    action_pairs = [(mode, sp) for mode in modes for sp in action_sps]
    action_indices = Counter()
    for index, scenario_type in enumerate(scenario_types):
        mode = modes[index % len(modes)]
        required_sp = None
        if scenario_type in {"single_action", "multi_action", "boundary_action"}:
            pair = action_pairs[action_indices[scenario_type] % len(action_pairs)]
            mode, required_sp = pair
            action_indices[scenario_type] += 1
        duration = _sample_minutes(generation["duration_minutes"][scenario_type], rng, step_seconds)
        config = _common_config(base, generation, mode, seed + index, duration, rng)
        if scenario_type in {"single_action", "multi_action", "boundary_action"}:
            config["base_scenario_type"] = (
                "multi_action" if scenario_type == "boundary_action" else scenario_type)
            config["action_events"] = _sample_action_events(
                generation, nominal[mode], action_sps, duration,
                "multi_action" if scenario_type == "boundary_action" else scenario_type,
                required_sp, rng, step_seconds=step_seconds,
                force_boundary=scenario_type == "boundary_action",
            )
        elif scenario_type == "disturbance":
            config["disturbance_events"] = _sample_disturbances(generation, duration, rng)
            base_type = str(rng.choice(("single_action", "multi_action")))
            config["base_scenario_type"] = base_type
            config["action_events"] = _sample_action_events(
                generation, nominal[mode], action_sps, duration, base_type,
                int(rng.choice(action_sps)), rng, step_seconds=step_seconds,
            )
        elif scenario_type == "mode_switch":
            config["mode_events"] = _sample_mode_events(
                generation, mode, duration, rng, step_seconds)
        group_id = f"group_{index:06d}"
        plans.append(
            _make_plan(
                f"traj_{index:06d}", group_id, scenario_type,
                _split_group(group_id, seed, generation["split_ratios"]), config,
            )
        )
    offset = len(plans)
    for pair_index in range(pair_count):
        mode = modes[(count + pair_index) % len(modes)]
        pair_seed = seed + count + pair_index
        required_sp = action_sps[pair_index % len(action_sps)]
        duration = _sample_minutes(generation["duration_minutes"]["single_action"], rng, step_seconds)
        common = _common_config(base, generation, mode, pair_seed, duration, rng)
        events = _sample_action_events(
            generation, nominal[mode], action_sps, duration, "single_action",
            required_sp, rng, step_seconds=step_seconds, allow_restore=False,
        )
        group_id = f"counterfactual_{pair_index:05d}"
        split = _split_group(group_id, seed, generation["split_ratios"])
        for role, action_events in (("reference", []), ("intervention", events)):
            index = offset + 2 * pair_index + int(role == "intervention")
            config = {**common, "action_events": action_events}
            plans.append(
                _make_plan(
                    f"traj_{index:06d}", group_id, f"counterfactual_{role}",
                    split, config, role,
                )
            )
    return plans
def generate_dataset(
    data_config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    num_trajectories: int | None = None,
    counterfactual_pairs: int | None = None,
    show_progress: bool = True,
    num_workers: int | None = None,
    compress: bool | None = None,
    overwrite: bool | None = None,
) -> dict[str, Any]:
    """执行全部计划、验证轨迹，并保存到配置指定的数据集目录。"""
    data = dict(data_config)
    generation = data["generation"]
    workers = int(generation.get("num_workers", 1) if num_workers is None else num_workers)
    compressed = bool(generation.get("compress", True) if compress is None else compress)
    replace = bool(generation.get("overwrite", False) if overwrite is None else overwrite)
    if workers < 1:
        raise ValueError("num_workers must be positive")
    plans = build_generation_plan(data, num_trajectories, counterfactual_pairs)
    target = Path(output_dir) if output_dir is not None else Path(data.get(
        "output_dir", "datasets")) / str(generation.get("dataset_name", "tep_world_model"))
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        if not replace or not target.is_dir():
            raise FileExistsError(f"dataset directory is not empty: {target}")
        resolved = target.resolve()
        if resolved == Path.cwd().resolve() or resolved == Path(resolved.anchor):
            raise ValueError(f"refusing to replace unsafe dataset path: {resolved}")
        shutil.rmtree(resolved)
    trajectory_dir = target / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(plan, trajectory_dir, target, compressed) for plan in plans]
    progress = {"total": len(tasks), "desc": "Generating TEP trajectories",
                "unit": "trajectory", "disable": not show_progress}
    if workers == 1:
        generated = list(tqdm(map(_generate_one, tasks), **progress))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            generated = list(tqdm(pool.map(_generate_one, tasks), **progress))
    records, pair_initials = [], {}
    for record, initial in generated:
        if initial is not None:
            previous = pair_initials.get(record["group_id"])
            if previous is not None and not np.allclose(previous, initial):
                raise RuntimeError(f"counterfactual pair mismatch: {record['group_id']}")
            pair_initials[record["group_id"]] = initial
        records.append(record)
    manifest_path = target / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                                     for record in records), encoding="utf-8")
    summary = summarize_dataset(records, data["action_sp_numbers"])
    summary["model_step_seconds"] = model_step_seconds(data)
    summary["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary["config_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    summary_path = target / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return {"output_dir": target, "manifest_path": manifest_path,
            "summary_path": summary_path, "records": records, "summary": summary}
def _generate_one(task):
    """生成、验证并保存一条轨迹，供串行或多进程执行。"""
    source_plan, trajectory_dir, root, compressed = task
    plan = {**source_plan, "config": dict(source_plan["config"])}
    retries = int(plan["config"].get("warmup_retries", 5))
    initial_seed = int(plan["config"]["seed"])
    for attempt in range(retries + 1):
        try:
            trajectory = TEPDataGenerator(
                plan["config"], initialize=False).generate_trajectory()
            break
        except RuntimeError as error:
            if "warmup" not in str(error).lower() or attempt == retries:
                raise RuntimeError(
                    f"{plan['trajectory_id']} failed initialization") from error
            plan["config"]["seed"] = initial_seed + (attempt + 1) * 1_000_003
    plan["initialization_attempts"] = attempt + 1
    if trajectory.terminated_early and not trajectory.shutdown:
        raise RuntimeError(f"{plan['trajectory_id']} terminated without TEP shutdown")
    quality = validate_trajectory(trajectory, expected_duration_hours=plan["config"]["duration_hours"],
                                  require_complete=not trajectory.shutdown)
    validate_action_sampling(trajectory.actions, model_step_seconds(plan["config"]))
    file_path = trajectory_dir / f"{plan['trajectory_id']}.npz"
    record = _trajectory_record(plan, trajectory, quality, file_path, root)
    save_trajectory(file_path, trajectory, record, compressed)
    initial = trajectory.observations[0].copy() if plan["counterfactual_role"] else None
    return record, initial
def save_trajectory(path: str | Path, trajectory: Trajectory, metadata: Mapping[str, Any], compressed: bool = True) -> None:
    """将训练数组和元数据保存为不含pickle、可选压缩的NPZ文件。"""
    save = np.savez_compressed if compressed else np.savez
    save(
        Path(path),
        time=trajectory.time,
        observations=trajectory.observations.astype(np.float32),
        actions=trajectory.actions.astype(np.float32),
        setpoints=trajectory.setpoints.astype(np.float32),
        disturbances=trajectory.disturbances,
        operating_modes=trajectory.operating_modes,
        metadata=np.asarray(json.dumps(dict(metadata), ensure_ascii=False)),
    )
def summarize_dataset(records: Sequence[Mapping[str, Any]], action_sp_numbers: Sequence[int]) -> dict[str, Any]:
    """汇总场景、划分、动作、工况和干扰的全局覆盖情况。"""
    count = lambda key: dict(Counter(str(record[key]) for record in records))
    action_counts = Counter({str(sp): 0 for sp in action_sp_numbers})
    idv_counts = Counter()
    for record in records:
        action_counts.update(map(str, record["changed_action_sp_numbers"]))
        idv_counts.update(map(str, record["active_idvs"]))
    missing = [int(sp) for sp, value in action_counts.items() if value == 0]
    return {
        "trajectory_count": len(records),
        "total_transitions": sum(record["transitions"] for record in records),
        "total_duration_hours": sum(record["actual_duration_hours"] for record in records),
        "scenario_counts": count("scenario_type"),
        "outcome_counts": count("outcome"),
        "action_pattern_incomplete_count": sum(not r["action_pattern_complete"] for r in records),
        "boundary_attempt_count": sum(r["scenario_type"] == "boundary_action" for r in records),
        "boundary_observed_unsafe_count": sum(
            r["scenario_type"] == "boundary_action" and r["observed_unsafe"] for r in records),
        "mean_action_event_count": float(np.mean([r["action_event_count"] for r in records])),
        "mean_action_interval_minutes": float(np.mean([
            gap / 60 for r in records for gap in np.diff(r["actual_action_event_steps"])
        ])) if any(r["action_event_count"] > 1 for r in records) else None,
        "split_counts": count("split"),
        "mode_counts": count("mode"),
        "action_counts": dict(action_counts),
        "missing_action_sp_numbers": missing,
        "action_coverage_ratio": 1.0 - len(missing) / len(action_counts),
        "idv_counts": dict(idv_counts),
    }
def _sample_action_events(
    generation: Mapping[str, Any],
    nominal: Mapping[int, float],
    action_sps: Sequence[int],
    duration: float,
    scenario_type: str,
    required_sp: int | None,
    rng: np.random.Generator,
    *,
    step_seconds: int,
    allow_restore: bool = True,
    force_boundary: bool = False,
) -> list[dict[str, Any]]:
    """Sample repeated SP events with short bursts and long settling intervals."""
    config = generation["action"]
    if not (1 <= int(config["sp_per_event"][0]) <= int(config["sp_per_event"][1]) <= len(action_sps)):
        raise ValueError("invalid sp_per_event range")
    if scenario_type == "multi_action" and int(config["sp_per_event"][1]) < 2:
        raise ValueError("multi_action requires at least two SPs per event")
    if not 0 < float(config["step_fraction"][0]) <= float(config["step_fraction"][1]) <= 1:
        raise ValueError("step_fraction must lie in (0, 1]")
    probability = float(config["short_hold_probability"])
    if not 0 <= probability <= 1:
        raise ValueError("short_hold_probability must lie in [0, 1]")
    for key in ("short_hold_steps", "long_hold_steps"):
        if any(int(v) != v or v < 1 for v in config[key]):
            raise ValueError(f"{key} must contain positive model-step counts")
    latest = duration - float(config["final_context_minutes"])
    first = config["first_event_minutes"]
    # Reserve room for two actual events, separated by a short hold.
    latest_first = latest - int(config["short_hold_steps"][1]) * step_seconds / 60
    event_time = _sample_minutes([first[0], min(first[1], latest_first)], rng, step_seconds)
    current, events = dict(nominal), []
    targets = generation["boundary_action"]["targets"] if force_boundary else {}
    if force_boundary and not targets:
        raise ValueError("boundary_action requires nonempty SP targets")
    for name, target in targets.items():
        sp = int(name.removeprefix("SP"))
        if sp not in action_sps or not config["bounds"][name][0] <= target <= config["bounds"][name][1]:
            raise ValueError(f"invalid boundary target: {name}")
    if len(targets) > int(config["sp_per_event"][1]):
        raise ValueError("boundary targets exceed the simultaneous-SP limit")
    while event_time <= latest:
        sp_count = 1 if scenario_type == "single_action" else _randint(config["sp_per_event"], rng)
        if (not events or force_boundary) and scenario_type == "multi_action":
            sp_count = max(2, sp_count)
        selected = list(map(int, rng.choice(action_sps, sp_count, replace=False)))
        if not events and required_sp is not None and required_sp not in selected:
            selected[0] = required_sp
        if force_boundary:
            target_sps = [int(name.removeprefix("SP")) for name in targets]
            selected = target_sps + [sp for sp in selected if sp not in target_sps]
            selected = selected[:max(len(target_sps), sp_count)]
        values = {}
        for sp in selected:
            name = f"SP{sp}"
            lower, upper = map(float, config["bounds"][name])
            limit = float(config["step_fraction"][1]) * (upper - lower)
            if name in targets:
                value = current[sp] + np.clip(float(targets[name]) - current[sp], -limit, limit)
            elif allow_restore and rng.random() < float(config["restore_probability"]):
                value = current[sp] + np.clip(nominal[sp] - current[sp], -limit, limit)
                if np.isclose(value, current[sp]):
                    value = _sample_action_value(config, sp, current[sp], rng)
            else:
                value = _sample_action_value(config, sp, current[sp], rng)
            if not np.isclose(value, current[sp], rtol=1e-6, atol=1e-6):
                values[name] = float(value)
                current[sp] = float(value)
        if values:
            events.append({"time_hours": event_time / 60.0, "values": values})
        if force_boundary and len(events) >= 2 and all(
            np.isclose(current[int(name.removeprefix("SP"))], target)
            for name, target in targets.items()
        ):
            break  # Hold the stress target for the remainder of the trajectory.
        short = force_boundary or len(events) == 1 or rng.random() < probability
        key = "short_hold_steps" if short else "long_hold_steps"
        event_time += _randint(config[key], rng) * step_seconds / 60.0
    return events

def _sample_action_value(config: Mapping[str, Any], sp: int, current: float,
                         rng: np.random.Generator) -> float:
    """Sample a bounded step relative to the immediately preceding SP value."""
    lower, upper = map(float, config["bounds"][f"SP{sp}"])
    delta = rng.uniform(*map(float, config["step_fraction"])) * (upper - lower)
    direction = float(rng.choice((-1.0, 1.0)))
    value = float(np.clip(current + direction * delta, lower, upper))
    if np.isclose(value, current):
        value = float(np.clip(current - direction * delta, lower, upper))
    return value

def _sample_disturbances(
    generation: Mapping[str, Any], duration: float, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """采样一个或多个IDV的开启与关闭事件。"""
    config = generation["disturbance"]
    idvs = rng.choice(
        config["idvs"], _randint(config["idv_count"], rng), replace=False
    )
    latest_off = duration - int(config.get("final_context_minutes", 30))
    events = []
    for offset, idv in enumerate(idvs):
        start = _randint(config["start_minutes"], rng) + 5 * offset
        stop = min(start + _randint(config["active_minutes"], rng), latest_off)
        if stop <= start:
            raise ValueError("disturbance timing does not fit trajectory duration")
        events += [
            {"time_hours": start / 60.0, "idv": int(idv), "value": 1},
            {"time_hours": stop / 60.0, "idv": int(idv), "value": 0},
        ]
    return sorted(events, key=lambda event: event["time_hours"])
def _sample_mode_events(generation, current_mode, duration, rng, step_seconds):
    """Sample one operating-mode transition from a stable initial mode."""
    config = generation["mode_switch"]
    targets = [int(mode) for mode in config["target_modes"]
               if int(mode) != int(current_mode)]
    if not targets:
        raise ValueError("mode_switch requires a target different from the initial mode")
    latest = duration - int(config.get("final_context_minutes", 60))
    bounds = config["event_minutes"]
    event_time = _sample_minutes([bounds[0], min(bounds[1], latest)], rng, step_seconds)
    return [{"time_hours": event_time / 60.0, "mode": int(rng.choice(targets))}]
def _common_config(base, generation, mode, seed, duration, rng) -> dict[str, Any]:
    """组装所有场景共享的种子、工况、预热和时长配置。"""
    warmup = _randint(generation["warmup_minutes"], rng)
    return {
        **base,
        "seed": int(seed),
        "operating_mode": int(mode),
        "warmup_hours": warmup / 60.0,
        "duration_hours": duration / 60.0,
        "mode_events": [],
        "action_events": [],
        "disturbance_events": [],
    }
def _load_nominal_actions(base, modes, action_sps, bounds) -> dict[int, dict[int, float]]:
    """读取各初始工况下的名义SP动作，供动作采样与恢复使用。"""
    nominal = {}
    for mode in modes:
        generator = TEPDataGenerator(
            {**base, "operating_mode": mode, "warmup_hours": 0.0,
             "mode_events": [], "action_events": [], "disturbance_events": []}
        )
        nominal[mode] = dict(zip(action_sps, generator.get_action()))
        for sp, value in nominal[mode].items():
            lower, upper = map(float, bounds[f"SP{sp}"])
            if not lower <= value <= upper:
                raise ValueError(f"mode {mode} nominal SP{sp}={value} is outside action bounds")
    return nominal
def _allocate_scenario_types(count, generation, rng) -> list[str]:
    """按目标比例分配场景类型，并随机打乱生成顺序。"""
    proportions = generation["scenario_proportions"]
    if set(proportions) != set(SCENARIO_TYPES) or not np.isclose(sum(proportions.values()), 1):
        raise ValueError("scenario_proportions must define all scenario types and sum to 1")
    raw = {name: count * float(proportions[name]) for name in SCENARIO_TYPES}
    allocated = {name: int(value) for name, value in raw.items()}
    order = sorted(raw, key=lambda name: raw[name] - allocated[name], reverse=True)
    for name in order[: count - sum(allocated.values())]:
        allocated[name] += 1
    result = [name for name in SCENARIO_TYPES for _ in range(allocated[name])]
    rng.shuffle(result)
    return result
def _make_plan(trajectory_id, group_id, scenario_type, split, config, role=None):
    """创建字段统一的单条轨迹计划记录。"""
    return {
        "trajectory_id": trajectory_id,
        "group_id": group_id,
        "scenario_type": scenario_type,
        "split": split,
        "counterfactual_role": role,
        "config": config,
    }
def _split_group(group_id, seed, ratios) -> str:
    """按组ID进行确定性数据划分，防止反事实对跨集合泄漏。"""
    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("split_ratios must sum to 1")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode()).digest()
    value, cumulative = int.from_bytes(digest[:8], "big") / 2**64, 0.0
    for split, ratio in ratios.items():
        cumulative += float(ratio)
        if value < cumulative:
            return str(split)
    return str(next(reversed(ratios)))
def _trajectory_record(plan, trajectory, quality, file_path, root) -> dict[str, Any]:
    """合并计划、质量结果和文件位置，形成manifest记录。"""
    config = plan["config"]
    counts = action_change_counts(trajectory.actions, trajectory.metadata["initial_action"])
    event_steps = np.flatnonzero(counts)
    base_type = config.get("base_scenario_type", plan["scenario_type"])
    return {
        **{key: value for key, value in plan.items() if key != "config"},
        "base_scenario_type": base_type,
        "actual_action_event_steps": event_steps.tolist(),
        "action_event_count": int(len(event_steps)),
        "multi_sp_event_count": int(np.count_nonzero(counts >= 2)),
        "action_pattern_complete": bool(matches_action_window(counts, base_type))
            if base_type in {"single_action", "multi_action"} else True,
        "observed_unsafe": bool(trajectory.shutdown or any(trajectory.metadata["limit_violations"].values())),
        "file": file_path.relative_to(root).as_posix(),
        "seed": config["seed"],
        "model_step_seconds": model_step_seconds(config),
        "mode": config["operating_mode"],
        "warmup_hours": config["warmup_hours"],
        "requested_duration_hours": config["duration_hours"],
        "actual_duration_hours": quality["duration_hours"],
        "transitions": quality["transitions"],
        "trajectory_type": quality["trajectory_type"],
        "outcome": quality["outcome"],
        "shutdown": quality["shutdown"],
        "terminated_early": quality["terminated_early"],
        "changed_action_sp_numbers": list(trajectory.metadata["changed_action_sp_numbers"]),
        "active_idvs": (np.flatnonzero(trajectory.disturbances.max(axis=0)) + 1).tolist(),
        "action_event_schedule": config["action_events"],
        "disturbance_event_schedule": config["disturbance_events"],
        "mode_event_schedule": config["mode_events"],
        "limit_violations": trajectory.metadata["limit_violations"],
    }
def _randint(bounds, rng) -> int:
    """在含上下界的整数区间内均匀采样。"""
    lower, upper = map(int, bounds)
    if lower > upper:
        raise ValueError(f"invalid integer range: {bounds}")
    return int(rng.integers(lower, upper + 1))


def _sample_minutes(bounds, rng, step_seconds: int) -> float:
    """Sample a model-grid time inside an inclusive range expressed in minutes."""
    lower = int(np.ceil(float(bounds[0]) * 60 / step_seconds))
    upper = int(np.floor(float(bounds[1]) * 60 / step_seconds))
    if lower > upper:
        raise ValueError(f"time range {bounds} contains no model-step boundary")
    return _randint((lower, upper), rng) * step_seconds / 60.0
