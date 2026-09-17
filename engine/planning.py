"""LeWM-style latent CEM planning and receding-horizon TEP execution."""
from __future__ import annotations
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
import numpy as np
import torch
from data.data_validation import classify_trajectory_outcome, model_step_seconds
from data.data_process import NormalizationStats
from engine.planning_tasks import standardized_xmeas_mse
from engine.training import load_checkpoint
class CEMPlanner:
    """Search one bounded SP vector held until the predicted goal time."""
    def __init__(
        self,
        model: torch.nn.Module,
        normalization: NormalizationStats,
        action_bounds: Sequence[Sequence[float]],
        *,
        num_samples: int = 300,
        num_elites: int = 30,
        iterations: int = 30,
        initial_std: float = 1.0,
        max_changed_actions: int = 3,
        max_action_delta_fraction: float = 0.30,
        seed: int = 42,
    ):
        """Configure CEM in the same normalized action space used for training."""
        if not (num_samples > 1 and 0 < num_elites <= num_samples):
            raise ValueError("invalid CEM sample count or elite count")
        if iterations < 1 or not np.isfinite(initial_std) or initial_std <= 0:
            raise ValueError("invalid CEM sampling settings")
        if not 1 <= max_changed_actions <= model.action_dim:
            raise ValueError("max_changed_actions is outside the action dimension")
        if max_action_delta_fraction <= 0:
            raise ValueError("invalid action trust-region settings")
        bounds = np.asarray(action_bounds, dtype=np.float32)
        if bounds.shape != (model.action_dim, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("action_bounds must have shape (action_dim, 2)")
        self.model = model.eval()
        self.stats = normalization
        self.device = next(model.parameters()).device
        self.physical_lower = bounds[:, 0]
        self.physical_upper = bounds[:, 1]
        normalized = (
            bounds - normalization.action_mean[:, None]
        ) / normalization.action_std[:, None]
        self.lower = torch.as_tensor(normalized[:, 0], device=self.device)
        self.upper = torch.as_tensor(normalized[:, 1], device=self.device)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.iterations = int(iterations)
        self.initial_std = float(initial_std)
        self.max_changed_actions = int(max_changed_actions)
        self.max_action_delta_fraction = float(max_action_delta_fraction)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
    @torch.inference_mode()
    def plan(
        self,
        observation_history,
        action_history,
        goal_observation,
        *,
        horizon: int,
        initial_action=None,
    ) -> dict[str, np.ndarray | float]:
        """Return the final CEM mean projected onto the TEP action constraints."""
        if int(horizon) != horizon or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        observations, past_actions = self._prepare_context(
            observation_history, action_history
        )
        goal = np.asarray(goal_observation, dtype=np.float32).reshape(-1)
        if goal.shape != (self.model.observation_dim,):
            raise ValueError("goal_observation has the wrong dimension")
        if initial_action is not None:
            current = np.asarray(initial_action, dtype=np.float32).reshape(-1)
        elif len(past_actions):
            current = past_actions[-1]
        else:
            current = 0.5 * (self.physical_lower + self.physical_upper)
        if current.shape != (self.model.action_dim,):
            raise ValueError("initial_action has the wrong dimension")
        if np.any(current < self.physical_lower - 1e-6) or np.any(current > self.physical_upper + 1e-6):
            raise ValueError("current SP action is outside the shared action bounds")
        planning_horizon = int(horizon)
        observations = self._observation_tensor(observations)[None]
        past_actions = self._action_tensor(past_actions)[None]
        goal_tensor = self._observation_tensor(goal)[None]
        current_tensor = self._action_tensor(current)
        mean = torch.zeros(planning_horizon, self.model.action_dim, device=self.device)
        action_range = self.upper - self.lower
        std = torch.full_like(mean, self.initial_std)
        for _ in range(self.iterations):
            # Preserve the validated sampling layout; only the first SP is free.
            noise = torch.randn(
                self.num_samples, planning_horizon, self.model.action_dim,
                generator=self.generator, device=self.device,
            )
            candidates = mean[None] + std[None] * noise
            candidates[0] = mean
            candidates = self._constrain_candidates(
                candidates, current_tensor, action_range
            )
            predicted = self.model.rollout(
                observations, past_actions, candidates[None]
            )
            costs = self.model.goal_cost(predicted, goal_tensor)[0]
            elite_indices = torch.topk(
                costs, self.num_elites, largest=False
            ).indices
            elites = candidates[elite_indices]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0, unbiased=False)
        # Sparse action constraints are non-convex: an elite mean also needs projection.
        planned_actions = self._constrain_candidates(
            mean[None], current_tensor, action_range
        )[0]
        predicted = self.model.rollout(
            observations, past_actions, planned_actions[None, None]
        )
        latent_cost = float(self.model.goal_cost(predicted, goal_tensor)[0, 0])
        physical_actions = (
            planned_actions.cpu().numpy() * self.stats.action_std
            + self.stats.action_mean
        )
        return {
            "actions": physical_actions,
            "latent_cost": latent_cost,
            "predicted_embeddings": predicted[0, 0].cpu().numpy(),
        }
    def _constrain_candidates(
        self, candidates: torch.Tensor, current: torch.Tensor,
        action_range: torch.Tensor,
    ) -> torch.Tensor:
        """Constrain one SP change, then hold it over the entire predicted horizon."""
        limit = self.max_action_delta_fraction * action_range
        proposal = torch.maximum(torch.minimum(candidates[:, 0], self.upper), self.lower)
        delta = torch.maximum(torch.minimum(proposal - current, limit), -limit)
        scores = delta.abs() / action_range
        selected = scores.topk(self.max_changed_actions, dim=-1).indices
        mask = torch.zeros_like(scores).scatter_(1, selected, 1.0)
        held = current + delta * mask
        return held[:, None].expand_as(candidates).contiguous()
    def _prepare_context(self, observations, actions) -> tuple[np.ndarray, np.ndarray]:
        """Validate and truncate a causally aligned raw state/action history."""
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.model.observation_dim:
            raise ValueError("observation_history has the wrong shape")
        if actions.ndim != 2 or actions.shape[1] != self.model.action_dim:
            raise ValueError("action_history has the wrong shape")
        if len(observations) < 1 or len(actions) != len(observations) - 1:
            raise ValueError("action_history must have one fewer step than observations")
        steps = min(len(observations), self.model.history_size)
        observations = observations[-steps:]
        actions = actions[-(steps - 1):] if steps > 1 else actions[:0]
        return observations, actions
    def _observation_tensor(self, values) -> torch.Tensor:
        values = self.stats.normalize_observations(np.asarray(values))
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)
    def _action_tensor(self, values) -> torch.Tensor:
        values = self.stats.normalize_actions(np.asarray(values))
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)
def build_cem_planner(
    checkpoint_path: str | Path,
    config: Mapping,
    *,
    device: str = "auto",
    **overrides,
) -> CEMPlanner:
    """Load a checkpoint and construct a planner from project configuration."""
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    if model_step_seconds(config["data"]) != int(
        checkpoint["training_config"]["model_step_seconds"]
    ):
        raise ValueError("configured model step differs from checkpoint; retrain the model")
    data = config["data"]
    bounds_config = data["generation"]["action"]["bounds"]
    bounds = [bounds_config[f"SP{number}"] for number in data["action_sp_numbers"]]
    planner_keys = {
        "num_samples", "num_elites", "iterations", "initial_std", "seed",
    }
    options = {
        key: value for key, value in config.get("planning", {}).items()
        if key in planner_keys
    }
    action_config = data["generation"]["action"]
    options["max_changed_actions"] = int(action_config["sp_per_event"][1])
    options["max_action_delta_fraction"] = float(action_config["step_fraction"][1])
    options.update(overrides)
    return CEMPlanner(model, stats, bounds, **options)
def run_tep_mpc(
    generator,
    planner: CEMPlanner,
    observation_history,
    action_history,
    goal_observation,
    *,
    goal_offset_steps: int,
    control_steps: int,
    simulator_steps_per_control: int,
    replan_every: int = 5,
    success_threshold: float = 0.10,
) -> dict[str, np.ndarray | bool]:
    """Replan to the remaining goal time and hold SP between feedback updates."""
    counts = (goal_offset_steps, control_steps, simulator_steps_per_control, replan_every)
    if any(int(value) != value or value < 1 for value in counts):
        raise ValueError("control step counts must be positive integers")
    if control_steps < goal_offset_steps or success_threshold < 0:
        raise ValueError("budget must cover the goal time and threshold must be nonnegative")
    observations, actions = planner._prepare_context(
        observation_history, action_history
    )
    executed_observations = [observations[-1].copy()]
    executed_actions, latent_costs, planning_horizons = [], [], []
    elapsed_seconds = [0]
    alive = True
    success = False
    safety_limits = generator.config.get("safety_limits", {})
    _, initial_violations = classify_trajectory_outcome(
        observations[-1:], safety_limits, shutdown=False, terminated_early=False,
    )
    violations = Counter(initial_violations)
    completed_steps = 0
    while completed_steps < control_steps:
        # After the target time, use one-step feedback within the remaining budget.
        horizon = max(1, goal_offset_steps - completed_steps)
        planning_horizons.append(horizon)
        result = planner.plan(
            observations, actions, goal_observation,
            horizon=horizon, initial_action=generator.get_action(),
        )
        planned_actions = np.asarray(result["actions"])
        latent_costs.append(float(result["latent_cost"]))
        execution_steps = min(replan_every, horizon, control_steps - completed_steps)
        for offset in range(execution_steps):
            action = planned_actions[offset]
            step_observations = []
            for second in range(simulator_steps_per_control):
                transition = generator.step(action if second == 0 else None)
                step_observations.append(np.asarray(transition["next_observation"]).copy())
                if not transition["alive"]:
                    break
            elapsed_seconds.append(elapsed_seconds[-1] + len(step_observations))
            next_observation = step_observations[-1]
            executed_actions.append(action.copy())
            executed_observations.append(next_observation)
            observations = np.concatenate(
                [observations, next_observation[None]], axis=0
            )
            actions = np.concatenate([actions, action[None]], axis=0)
            observations, actions = planner._prepare_context(observations, actions)
            completed_steps += 1
            alive = bool(transition["alive"])
            _, step_violations = classify_trajectory_outcome(
                np.asarray(step_observations), safety_limits,
                shutdown=not alive, terminated_early=not alive,
            )
            violations.update(step_violations)
            success = bool(alive and not any(violations.values()) and
                           standardized_xmeas_mse(next_observation, goal_observation,
                                                 planner.stats) <= success_threshold)
            if not alive or success:
                break
        if not alive or success:
            break
    return {
        "observations": np.asarray(executed_observations),
        "actions": np.asarray(executed_actions),
        "latent_costs": np.asarray(latent_costs),
        "planning_horizons": np.asarray(planning_horizons),
        "time_minutes": np.asarray(elapsed_seconds) / 60,
        "limit_violations": dict(violations),
        "soft_limit_violated": any(violations.values()),
        "alive": alive,
        "success": success,
    }
