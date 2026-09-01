"""Configuration-driven batch generation for TEP world-model datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from data.data_gen import TEPDataGenerator, Trajectory
from data.data_validation import validate_trajectory

SCENARIO_TYPES = ("fixed_sp", "single_action", "multi_action", "disturbance")


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
    count = int(
        generation.get("num_trajectories", 300)
        if num_trajectories is None
        else num_trajectories
    )
    pair_count = int(
        generation.get("counterfactual_pairs", 0)
        if counterfactual_pairs is None
        else counterfactual_pairs
    )
    if count <= 0 or pair_count < 0:
        raise ValueError("trajectory count must be positive and pair count non-negative")

    seed = int(generation.get("seed", data.get("seed", 42)))
    rng = np.random.default_rng(seed)
    action_sps = tuple(map(int, data["action_sp_numbers"]))
    modes = tuple(map(int, generation.get("operating_modes", range(1, 7))))
    base = {key: value for key, value in data.items() if key not in {"generation", "output_dir"}}
    nominal = _load_nominal_actions(base, modes, action_sps)
    scenario_types = _allocate_scenario_types(count, generation, rng)

    plans = []
    action_index = 0
    for index, scenario_type in enumerate(scenario_types):
        mode = modes[index % len(modes)]
        required_sp = None
        if scenario_type in {"single_action", "multi_action"}:
            required_sp = action_sps[action_index % len(action_sps)]
            action_index += 1
        duration = _randint(generation["duration_minutes"][scenario_type], rng)
        config = _common_config(base, generation, mode, seed + index, duration, rng)
        if scenario_type in {"single_action", "multi_action"}:
            config["action_events"] = _sample_action_events(
                generation, nominal[mode], action_sps, duration, scenario_type,
                required_sp, rng,
            )
        elif scenario_type == "disturbance":
            config["disturbance_events"] = _sample_disturbances(generation, duration, rng)
            if rng.random() < float(
                generation["disturbance"].get("action_overlay_probability", 0.5)
            ):
                config["action_events"] = _sample_action_events(
                    generation, nominal[mode], action_sps, duration, "multi_action",
                    int(rng.choice(action_sps)), rng,
                )
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
        required_sp = action_sps[action_index % len(action_sps)]
        action_index += 1
        duration = _randint(generation["duration_minutes"]["single_action"], rng)
        common = _common_config(base, generation, mode, pair_seed, duration, rng)
        events = _sample_action_events(
            generation, nominal[mode], action_sps, duration, "single_action",
            required_sp, rng, allow_restore=False,
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
) -> dict[str, Any]:
    """执行全部计划，验证并保存一个不可覆盖的数据集版本。"""
    data = dict(data_config)
    generation = data["generation"]
    workers = int(generation.get("num_workers", 1) if num_workers is None else num_workers)
    compressed = bool(generation.get("compress", True) if compress is None else compress)
    if workers < 1:
        raise ValueError("num_workers must be positive")
    plans = build_generation_plan(data, num_trajectories, counterfactual_pairs)
    target = Path(output_dir) if output_dir is not None else Path(data.get(
        "output_dir", "datasets")) / str(generation.get("dataset_name", "tep_world_model"))
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"dataset directory is not empty: {target}")
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
    summary_path = target / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    return {"output_dir": target, "manifest_path": manifest_path,
            "summary_path": summary_path, "records": records, "summary": summary}


def _generate_one(task):
    """生成、验证并保存一条轨迹，供串行或多进程执行。"""
    plan, trajectory_dir, root, compressed = task
    trajectory = TEPDataGenerator(plan["config"], initialize=False).generate_trajectory()
    if trajectory.terminated_early and not trajectory.shutdown:
        raise RuntimeError(f"{plan['trajectory_id']} terminated without TEP shutdown")
    quality = validate_trajectory(trajectory, expected_duration_hours=plan["config"]["duration_hours"],
                                  require_complete=not trajectory.shutdown)
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
    duration: int,
    scenario_type: str,
    required_sp: int | None,
    rng: np.random.Generator,
    *,
    allow_restore: bool = True,
) -> list[dict[str, Any]]:
    """为单动作或多动作场景采样带保持间隔的SP事件序列。"""
    config = generation["action"]
    event_count = 1 if scenario_type == "single_action" else _randint(
        config["multi_event_count"], rng
    )
    latest = duration - int(config.get("final_context_minutes", 30))
    first = config["first_event_minutes"]
    event_time = _randint([first[0], min(first[1], latest)], rng)
    current, changed, events = dict(nominal), set(), []
    for event_index in range(event_count):
        if event_time > latest:
            break
        sp_count = 1 if scenario_type == "single_action" else _randint(
            config["sp_per_event"], rng
        )
        selected = list(rng.choice(action_sps, min(sp_count, len(action_sps)), replace=False))
        if event_index == 0 and required_sp is not None and required_sp not in selected:
            selected[0] = required_sp
        values = {}
        for sp in map(int, selected):
            values[f"SP{sp}"] = _sample_action_value(config, sp, current[sp], rng)
            current[sp] = values[f"SP{sp}"]
            changed.add(sp)
        events.append({"time_hours": event_time / 60.0, "values": values})
        event_time += _randint(config["hold_minutes"], rng)
    if (
        allow_restore and changed and event_time <= latest
        and rng.random() < float(config.get("restore_probability", 0.2))
    ):
        sp = int(rng.choice(sorted(changed)))
        events.append({"time_hours": event_time / 60.0, "values": {f"SP{sp}": nominal[sp]}})
    return events


def _sample_action_value(config: Mapping[str, Any], sp: int, current: float, rng: np.random.Generator) -> float:
    """在给定SP边界内采样显著阶跃，并按概率采样边界值。"""
    bounds = config["bounds"].get(f"SP{sp}")
    if not isinstance(bounds, Sequence) or len(bounds) != 2:
        raise ValueError(f"missing generation action bounds for SP{sp}")
    lower, upper = map(float, bounds)
    if rng.random() < float(config.get("boundary_probability", 0.1)):
        band = float(config.get("boundary_band_fraction", 0.05))
        edge = float(rng.choice((lower, upper)))
        inward = band * (upper - lower) * rng.random()
        return round(edge + inward if edge == lower else edge - inward, 6)
    delta = rng.uniform(*map(float, config["step_fraction"])) * (upper - lower)
    direction = float(rng.choice((-1.0, 1.0)))
    value = float(np.clip(current + direction * delta, lower, upper))
    if np.isclose(value, current):
        value = float(np.clip(current - direction * delta, lower, upper))
    return round(value, 6)


def _sample_disturbances(
    generation: Mapping[str, Any], duration: int, rng: np.random.Generator
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


def _load_nominal_actions(base, modes, action_sps) -> dict[int, dict[int, float]]:
    """读取各初始工况下的名义SP动作，供动作采样与恢复使用。"""
    nominal = {}
    for mode in modes:
        generator = TEPDataGenerator(
            {**base, "operating_mode": mode, "warmup_hours": 0.0,
             "mode_events": [], "action_events": [], "disturbance_events": []}
        )
        nominal[mode] = dict(zip(action_sps, generator.get_action()))
    return nominal


def _allocate_scenario_types(count, generation, rng) -> list[str]:
    """按目标比例分配场景类型，并随机打乱生成顺序。"""
    proportions = generation["scenario_proportions"]
    if set(proportions) != set(SCENARIO_TYPES) or not np.isclose(sum(proportions.values()), 1):
        raise ValueError("scenario_proportions must define four types and sum to 1")
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
    return {
        **{key: value for key, value in plan.items() if key != "config"},
        "file": file_path.relative_to(root).as_posix(),
        "seed": config["seed"],
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
        "limit_violations": trajectory.metadata["limit_violations"],
    }


def _randint(bounds, rng) -> int:
    """在含上下界的整数区间内均匀采样。"""
    lower, upper = map(int, bounds)
    if lower > upper:
        raise ValueError(f"invalid integer range: {bounds}")
    return int(rng.integers(lower, upper + 1))
