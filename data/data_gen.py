"""TEP trajectory generation for world-model training and evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tep import ControlMode, TEPSimulator

from data.data_validation import (
    DEFAULT_ACTION_SP_NUMBERS,
    SAMPLE_INTERVAL_HOURS,
    classify_trajectory_outcome,
    model_step_seconds,
)

# SP12 is implemented by controller 22 in the TEP decentralized controller.
_SP_TO_CONTROLLER = {number: f"ctrl{number}" for number in range(1, 21)}
_SP_TO_CONTROLLER[12] = "ctrl22"


@dataclass
class Trajectory:
    """One causally aligned trajectory: observation[t], action[t], observation[t+1]."""

    time: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    setpoints: np.ndarray
    disturbances: np.ndarray
    operating_modes: np.ndarray
    shutdown: bool
    terminated_early: bool
    metadata: dict[str, Any]

    @property
    def measurements(self) -> np.ndarray:
        """返回观测中的41维过程测量量XMEAS。"""
        return self.observations[:, :41]

    @property
    def manipulated_vars(self) -> np.ndarray:
        """返回观测中的12维实际操纵量XMV。"""
        return self.observations[:, 41:]

    @property
    def trajectory_type(self) -> str:
        """返回轨迹类型标签，如fixed_sp、action或disturbance。"""
        return str(self.metadata["trajectory_type"])

    @property
    def outcome(self) -> str:
        """返回轨迹结局标签，如normal、boundary、violation或shutdown。"""
        return str(self.metadata["outcome"])


def load_data_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """读取YAML配置并返回其中的data配置段。"""
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise TypeError("config must contain a 'data' mapping")
    return data_config


class TEPDataGenerator:
    """Generate fixed-SP, action and disturbance trajectories from timed events."""

    dt_hours = SAMPLE_INTERVAL_HOURS

    def __init__(self, config: Mapping[str, Any], *, initialize: bool = True):
        """保存生成配置、校验动作空间，并按需创建仿真器。"""
        self.config = dict(config)
        self.seed = int(self.config.get("seed", 42))
        self.backend = str(self.config.get("backend", "python"))
        self.action_sp_numbers = tuple(
            int(number)
            for number in self.config.get(
                "action_sp_numbers", DEFAULT_ACTION_SP_NUMBERS
            )
        )
        if not self.action_sp_numbers or len(set(self.action_sp_numbers)) != len(
            self.action_sp_numbers
        ):
            raise ValueError("action_sp_numbers must be non-empty and unique")
        if any(number < 1 or number > 20 for number in self.action_sp_numbers):
            raise ValueError("action SP numbers must be between 1 and 20")
        self.simulator: TEPSimulator | None = None
        if initialize:
            self.reset()

    @property
    def controller(self):
        """返回当前闭环控制器，并阻止在初始化前访问。"""
        if self.simulator is None:
            raise RuntimeError("generator is not initialized")
        return self.simulator.controller

    def reset(
        self,
        seed: int | None = None,
        operating_mode: int | None = None,
        warmup_hours: float | None = None,
    ) -> np.ndarray:
        """重建仿真器、设置初始工况并执行不记录数据的预热。"""
        self.seed = self.seed if seed is None else int(seed)
        mode = int(
            self.config.get("operating_mode", 1)
            if operating_mode is None
            else operating_mode
        )
        warmup = float(
            self.config.get("warmup_hours", 0.0)
            if warmup_hours is None
            else warmup_hours
        )
        if mode not in range(1, 7):
            raise ValueError("operating_mode must be between 1 and 6")
        if warmup < 0:
            raise ValueError("warmup_hours must be non-negative")

        self.simulator = TEPSimulator(
            random_seed=self.seed,
            control_mode=ControlMode.CLOSED_LOOP,
            backend=self.backend,
        )
        # 仿真器对所有工况都从同一套内置基准物理状态开始。
        self.simulator.initialize()
        # 工况切换只更新控制器SP，不会替换反应器等内部物理状态。
        self.switch_operating_mode(mode)
        # 预热既消除初始瞬态，也验证该起点能否安全到达目标工况。
        warmup_steps = round(warmup / self.dt_hours)
        if warmup_steps and not self.simulator.step(warmup_steps):
            raise RuntimeError("TEP shut down during warmup")
        return self._observation()

    def switch_operating_mode(
        self, mode: int, preserve_actions: bool = False
    ) -> np.ndarray:
        """保持内部物理状态不变，仅切换工况对应的控制器SP。"""
        previous_action = self.get_action() if preserve_actions else None
        self.controller.set_mode(int(mode))
        self._sync_all_setpoints()
        if previous_action is not None:
            self.set_action(previous_action)
        return self.get_action()

    def get_action(self) -> np.ndarray:
        """按配置顺序读取世界模型使用的外部SP动作向量。"""
        return np.asarray(
            [
                getattr(self.controller, _SP_TO_CONTROLLER[number]).setpoint
                for number in self.action_sp_numbers
            ],
            dtype=np.float64,
        )

    def get_observation(self) -> np.ndarray:
        """Return the current 53-dimensional observable process state."""
        return self._observation().copy()

    def set_action(self, action: Sequence[float]) -> np.ndarray:
        """一次设置完整SP动作向量，并校验维度与数值有效性。"""
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        expected_shape = (len(self.action_sp_numbers),)
        if values.shape != expected_shape:
            raise ValueError(f"action must have shape {expected_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("action must contain only finite values")
        for number, value in zip(self.action_sp_numbers, values):
            self._set_setpoint(number, float(value))
        return self.get_action()

    def update_action(self, changes: Mapping[int | str, float]) -> np.ndarray:
        """只修改指定SP，其余动作分量保持当前值。"""
        if not isinstance(changes, Mapping) or not changes:
            raise ValueError("action changes must be a non-empty mapping")
        action = self.get_action()
        positions = {
            number: index for index, number in enumerate(self.action_sp_numbers)
        }
        for key, value in changes.items():
            number = self._parse_sp_number(key)
            if number not in positions:
                raise ValueError(f"SP{number} is not in the configured action space")
            action[positions[number]] = float(value)
        return self.set_action(action)

    def set_disturbance(self, idv_index: int, value: int = 1) -> None:
        """打开或关闭一个一基编号的TEP干扰变量IDV。"""
        if int(idv_index) not in range(1, 21):
            raise ValueError("idv_index must be between 1 and 20")
        if int(value) not in (0, 1):
            raise ValueError("disturbance value must be 0 or 1")
        self.simulator.set_disturbance(int(idv_index), int(value))

    def clear_disturbances(self) -> None:
        """关闭当前仿真器中的全部干扰。"""
        self.simulator.clear_disturbances()

    def step(self, action: Sequence[float] | None = None) -> dict[str, Any]:
        """可选地施加动作，并推进一个因果对齐的状态转移。"""
        observation = self._observation()
        if action is not None:
            self.set_action(action)
        applied_action = self.get_action()
        setpoints = np.asarray(self.controller.setpoints, dtype=np.float64).copy()
        disturbances = self.simulator.get_disturbances().copy()
        mode = int(self.controller.mode)
        alive = bool(self.simulator.step())
        return {
            "observation": observation,
            "action": applied_action,
            "next_observation": self._observation(),
            "setpoints": setpoints,
            "disturbances": disturbances,
            "operating_mode": mode,
            "alive": alive,
        }

    def generate_trajectory(
        self, config: Mapping[str, Any] | None = None
    ) -> Trajectory:
        """按时间事件执行仿真，生成一条可变时长的完整轨迹。"""
        trajectory_config = {**self.config, **dict(config or {})}
        duration = float(trajectory_config.get("duration_hours", 1.0))
        if duration <= 0:
            raise ValueError("duration_hours must be positive")

        total_steps = round(duration / self.dt_hours)
        if total_steps < 1 or not np.isclose(
            duration, total_steps * self.dt_hours, atol=1e-10
        ):
            raise ValueError("duration_hours must be aligned to 1-second sampling")

        self.reset(
            seed=int(trajectory_config.get("seed", self.seed)),
            operating_mode=int(trajectory_config.get("operating_mode", 1)),
            warmup_hours=float(trajectory_config.get("warmup_hours", 0.0)),
        )
        start_time = float(self.simulator.time)
        initial_action = self.get_action().copy()
        step_seconds = model_step_seconds(trajectory_config)
        mode_events = self._group_events(
            trajectory_config.get("mode_events", []), duration, "mode", step_seconds
        )
        action_events = self._group_events(
            trajectory_config.get("action_events", []), duration, "action", step_seconds
        )
        disturbance_events = self._group_events(
            trajectory_config.get("disturbance_events", []), duration, "disturbance", step_seconds
        )

        time = [0.0]
        observations = [self._observation()]
        transitions: dict[str, list[np.ndarray | int]] = {
            "actions": [],
            "setpoints": [],
            "disturbances": [],
            "modes": [],
        }
        for step_index in range(total_steps):
            for event in mode_events.get(step_index, []):
                self.switch_operating_mode(
                    int(event["mode"]), bool(event.get("preserve_actions", False))
                )
            for event in disturbance_events.get(step_index, []):
                self.set_disturbance(int(event["idv"]), int(event.get("value", 1)))
            for event in action_events.get(step_index, []):
                self.update_action(event["values"])

            transition = self.step()
            transitions["actions"].append(transition["action"])
            transitions["setpoints"].append(transition["setpoints"])
            transitions["disturbances"].append(transition["disturbances"])
            transitions["modes"].append(transition["operating_mode"])
            observations.append(transition["next_observation"])
            time.append(float(self.simulator.time) - start_time)
            if not transition["alive"]:
                break

        actions = np.asarray(transitions["actions"], dtype=np.float64)
        observation_array = np.asarray(observations, dtype=np.float64)
        terminated_early = len(actions) < total_steps
        shutdown = bool(self.simulator.is_shutdown())
        outcome, violations = classify_trajectory_outcome(
            observation_array,
            trajectory_config.get("safety_limits", {}),
            shutdown=shutdown,
            terminated_early=terminated_early,
            boundary_margin_fraction=float(
                trajectory_config.get("boundary_margin_fraction", 0.05)
            ),
        )
        changed = np.any(~np.isclose(actions, initial_action), axis=0)
        trajectory_type = (
            "disturbance"
            if disturbance_events
            else "action"
            if action_events or mode_events
            else "fixed_sp"
        )

        return Trajectory(
            time=np.asarray(time, dtype=np.float64),
            observations=observation_array,
            actions=actions,
            setpoints=np.asarray(transitions["setpoints"], dtype=np.float64),
            disturbances=np.asarray(transitions["disturbances"], dtype=np.int8),
            operating_modes=np.asarray(transitions["modes"], dtype=np.int8),
            shutdown=shutdown,
            terminated_early=terminated_early,
            metadata={
                "seed": self.seed,
                "backend": self.backend,
                "trajectory_type": trajectory_type,
                "outcome": outcome,
                "limit_violations": violations,
                "initial_mode": int(trajectory_config.get("operating_mode", 1)),
                "warmup_hours": float(trajectory_config.get("warmup_hours", 0.0)),
                "requested_duration_hours": duration,
                "action_sp_numbers": self.action_sp_numbers,
                "initial_action": initial_action.tolist(),
                "changed_action_sp_numbers": tuple(
                    number
                    for number, is_changed in zip(self.action_sp_numbers, changed)
                    if is_changed
                ),
            },
        )

    def _observation(self) -> np.ndarray:
        """拼接XMEAS和XMV，形成世界模型的53维观测状态。"""
        return np.concatenate(
            [self.simulator.get_measurements(), self.simulator.get_manipulated_vars()]
        ).astype(np.float64, copy=False)

    def _sync_all_setpoints(self) -> None:
        """将控制器SP数组同步到各个实际PI控制回路。"""
        for number, controller_name in _SP_TO_CONTROLLER.items():
            getattr(self.controller, controller_name).setpoint = float(
                self.controller.setpoints[number - 1]
            )

    def _set_setpoint(self, number: int, value: float) -> None:
        """同时更新一个SP的统一存储值与对应控制回路。"""
        self.controller.setpoints[number - 1] = value
        getattr(self.controller, _SP_TO_CONTROLLER[number]).setpoint = value

    def _group_events(
        self,
        events: Sequence[Mapping[str, Any]],
        duration_hours: float,
        event_name: str,
        step_seconds: int,
    ) -> dict[int, list[Mapping[str, Any]]]:
        """校验事件时间，并按一秒采样步索引对事件分组。"""
        grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for event in events:
            if not isinstance(event, Mapping) or "time_hours" not in event:
                raise TypeError(f"each {event_name} event needs a time_hours field")
            event_time = float(event["time_hours"])
            step_index = round(event_time / self.dt_hours)
            if event_time < 0 or not np.isclose(
                event_time, step_index * self.dt_hours, atol=1e-10
            ):
                raise ValueError("event time must be non-negative and aligned to 1 second")
            if event_time >= duration_hours:
                raise ValueError("event time must be earlier than trajectory duration")
            if event_name in {"action", "mode"} and step_index % step_seconds:
                raise ValueError("SP and mode events must align with model_step_minutes")
            grouped[step_index].append(event)
        return grouped

    @staticmethod
    def _parse_sp_number(value: int | str) -> int:
        """将整数或SP18形式的标识统一解析为SP编号。"""
        if isinstance(value, str):
            value = value.upper().removeprefix("SP")
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid SP identifier: {value!r}") from error
