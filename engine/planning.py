"""LeWM-style latent CEM planning and receding-horizon TEP execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from data.data_process import NormalizationStats
from engine.training import load_checkpoint


class CEMPlanner:
    """Search bounded SP sequences using terminal latent distance to the goal."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalization: NormalizationStats,
        action_bounds: Sequence[Sequence[float]],
        *,
        horizon: int = 10,
        num_samples: int = 512,
        num_elites: int = 64,
        iterations: int = 5,
        momentum: float = 0.1,
        min_std_fraction: float = 0.01,
        initial_std_fraction: float = 0.15,
        action_block_size: int = 10,
        max_changed_actions: int = 3,
        max_action_delta_fraction: float = 0.20,
        action_change_weight: float = 0.05,
        seed: int = 42,
    ):
        """Configure CEM in the same normalized action space used for training."""
        if not (horizon > 0 and num_samples > 1 and 0 < num_elites <= num_samples):
            raise ValueError("invalid CEM horizon, sample count or elite count")
        if iterations < 1 or not 0 <= momentum < 1 or min_std_fraction <= 0:
            raise ValueError("invalid CEM iteration or smoothing parameters")
        if not 0 < initial_std_fraction <= 1 or action_block_size < 1:
            raise ValueError("invalid CEM sampling or action-block settings")
        if not 1 <= max_changed_actions <= model.action_dim:
            raise ValueError("max_changed_actions is outside the action dimension")
        if max_action_delta_fraction <= 0 or action_change_weight < 0:
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
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.num_elites = int(num_elites)
        self.iterations = int(iterations)
        self.momentum = float(momentum)
        self.min_std_fraction = float(min_std_fraction)
        self.initial_std_fraction = float(initial_std_fraction)
        self.action_block_size = int(action_block_size)
        self.max_changed_actions = int(max_changed_actions)
        self.max_action_delta_fraction = float(max_action_delta_fraction)
        self.action_change_weight = float(action_change_weight)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    @torch.inference_mode()
    def plan(
        self,
        observation_history,
        action_history,
        goal_observation,
        *,
        initial_action=None,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray | float]:
        """Return the lowest terminal-latent-cost physical SP action sequence."""
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

        planning_horizon = self.horizon if horizon is None else int(horizon)
        if planning_horizon < 1 or planning_horizon > self.horizon:
            raise ValueError("horizon must be between 1 and the configured horizon")
        observations = self._observation_tensor(observations)[None]
        past_actions = self._action_tensor(past_actions)[None]
        goal_tensor = self._observation_tensor(goal)[None]
        current_tensor = self._action_tensor(current)
        mean = current_tensor.expand(planning_horizon, -1).clone()
        action_range = self.upper - self.lower
        std = (self.initial_std_fraction * action_range).expand(
            planning_horizon, -1
        ).clone()
        minimum_std = self.min_std_fraction * action_range
        best_cost, best_actions = float("inf"), None

        for _ in range(self.iterations):
            noise = torch.randn(
                self.num_samples, planning_horizon, self.model.action_dim,
                generator=self.generator, device=self.device,
            )
            candidates = mean[None] + std[None] * noise
            candidates = torch.maximum(
                torch.minimum(candidates, self.upper), self.lower
            )
            candidates = self._constrain_candidates(
                candidates, current_tensor, action_range
            )
            predicted = self.model.rollout(
                observations, past_actions, candidates[None]
            )
            costs = self.model.goal_cost(predicted, goal_tensor)[0]
            action_change = (candidates[:, 0] - current_tensor) / action_range
            costs = costs + self.action_change_weight * action_change.square().mean(-1)
            elite_indices = torch.topk(
                costs, self.num_elites, largest=False
            ).indices
            elites = candidates[elite_indices]
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, unbiased=False)
            mean = self.momentum * mean + (1 - self.momentum) * elite_mean
            std = self.momentum * std + (1 - self.momentum) * elite_std
            std = torch.maximum(std, minimum_std)
            iteration_cost, iteration_index = costs.min(dim=0)
            if float(iteration_cost) < best_cost:
                best_cost = float(iteration_cost)
                best_actions = candidates[int(iteration_index)].clone()

        assert best_actions is not None
        predicted = self.model.rollout(
            observations, past_actions, best_actions[None, None]
        )[0, 0]
        physical_actions = (
            best_actions.cpu().numpy() * self.stats.action_std
            + self.stats.action_mean
        )
        return {
            "actions": physical_actions,
            "latent_cost": best_cost,
            "predicted_embeddings": predicted.cpu().numpy(),
        }

    def _constrain_candidates(
        self, candidates: torch.Tensor, current: torch.Tensor,
        action_range: torch.Tensor,
    ) -> torch.Tensor:
        """Project CEM samples onto sparse, held and locally bounded SP changes."""
        limit = self.max_action_delta_fraction * action_range
        candidates = torch.maximum(
            torch.minimum(candidates, current + limit), current - limit
        )
        for start in range(0, candidates.size(1), self.action_block_size):
            stop = min(start + self.action_block_size, candidates.size(1))
            candidates[:, start:stop] = candidates[:, start:start + 1]
        delta = candidates - current
        scores = delta.abs().amax(dim=1)
        changed = scores.topk(self.max_changed_actions, dim=-1).indices
        mask = torch.zeros_like(scores).scatter_(1, changed, 1.0)
        candidates = current + delta * mask[:, None]
        return torch.maximum(torch.minimum(candidates, self.upper), self.lower)

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
    device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    device = "cpu" if device == "auto" else device
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    data = config["data"]
    bounds_config = data["generation"]["action"]["bounds"]
    bounds = [bounds_config[f"SP{number}"] for number in data["action_sp_numbers"]]
    planner_keys = {
        "horizon", "num_samples", "num_elites", "iterations", "momentum",
        "min_std_fraction", "initial_std_fraction", "action_block_size",
        "max_changed_actions", "max_action_delta_fraction",
        "action_change_weight", "seed",
    }
    options = {
        key: value for key, value in config.get("planning", {}).items()
        if key in planner_keys
    }
    options.update(overrides)
    return CEMPlanner(model, stats, bounds, **options)


def run_tep_mpc(
    generator,
    planner: CEMPlanner,
    observation_history,
    action_history,
    goal_observation,
    *,
    control_steps: int,
    simulator_steps_per_control: int = 60,
    replan_every: int = 1,
) -> dict[str, np.ndarray | bool]:
    """Periodically replan with CEM and execute the resulting actions in TEP."""
    if min(control_steps, simulator_steps_per_control, replan_every) < 1:
        raise ValueError("control step counts must be positive")
    observations, actions = planner._prepare_context(
        observation_history, action_history
    )
    executed_observations = [observations[-1].copy()]
    executed_actions, latent_costs = [], []
    alive = True
    completed_steps = 0
    while completed_steps < control_steps:
        horizon = min(planner.horizon, control_steps - completed_steps)
        result = planner.plan(
            observations, actions, goal_observation, horizon=horizon
        )
        planned_actions = np.asarray(result["actions"])
        latent_costs.append(float(result["latent_cost"]))
        execution_steps = min(replan_every, horizon)
        for offset in range(execution_steps):
            action = planned_actions[offset]
            transition = generator.step(action)
            for _ in range(simulator_steps_per_control - 1):
                if not transition["alive"]:
                    break
                transition = generator.step()
            next_observation = np.asarray(transition["next_observation"]).copy()
            executed_actions.append(action.copy())
            executed_observations.append(next_observation)
            observations = np.concatenate(
                [observations, next_observation[None]], axis=0
            )
            actions = np.concatenate([actions, action[None]], axis=0)
            observations, actions = planner._prepare_context(observations, actions)
            completed_steps += 1
            alive = bool(transition["alive"])
            if not alive:
                break
        if not alive:
            break
    return {
        "observations": np.asarray(executed_observations),
        "actions": np.asarray(executed_actions),
        "latent_costs": np.asarray(latent_costs),
        "alive": alive,
    }
