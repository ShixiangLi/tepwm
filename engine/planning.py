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
        seed: int = 42,
    ):
        """Configure CEM in the same normalized action space used for training."""
        if not (horizon > 0 and num_samples > 1 and 0 < num_elites <= num_samples):
            raise ValueError("invalid CEM horizon, sample count or elite count")
        if iterations < 1 or not 0 <= momentum < 1 or min_std_fraction <= 0:
            raise ValueError("invalid CEM iteration or smoothing parameters")
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
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    @torch.inference_mode()
    def plan(
        self,
        observation_history,
        action_history,
        goal_observation,
        *,
        initial_action=None,
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

        observations = self._observation_tensor(observations)[None]
        past_actions = self._action_tensor(past_actions)[None]
        goal_tensor = self._observation_tensor(goal)[None]
        current_tensor = self._action_tensor(current)
        mean = current_tensor.expand(self.horizon, -1).clone()
        action_range = self.upper - self.lower
        std = (0.5 * action_range).expand(self.horizon, -1).clone()
        minimum_std = self.min_std_fraction * action_range
        best_cost, best_actions = float("inf"), None

        for _ in range(self.iterations):
            noise = torch.randn(
                self.num_samples, self.horizon, self.model.action_dim,
                generator=self.generator, device=self.device,
            )
            candidates = mean[None] + std[None] * noise
            candidates = torch.maximum(
                torch.minimum(candidates, self.upper), self.lower
            )
            predicted = self.model.rollout(
                observations, past_actions, candidates[None]
            )
            costs = self.model.goal_cost(predicted, goal_tensor)[0]
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
        "min_std_fraction", "seed",
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
) -> dict[str, np.ndarray | bool]:
    """Replan each control step and execute the first CEM action in TEP."""
    if control_steps < 1 or simulator_steps_per_control < 1:
        raise ValueError("control step counts must be positive")
    observations, actions = planner._prepare_context(
        observation_history, action_history
    )
    executed_observations = [observations[-1].copy()]
    executed_actions, latent_costs = [], []
    alive = True
    for _ in range(control_steps):
        result = planner.plan(observations, actions, goal_observation)
        action = np.asarray(result["actions"])[0]
        transition = generator.step(action)
        for _ in range(simulator_steps_per_control - 1):
            if not transition["alive"]:
                break
            transition = generator.step()
        next_observation = np.asarray(transition["next_observation"]).copy()
        executed_actions.append(action.copy())
        executed_observations.append(next_observation)
        latent_costs.append(float(result["latent_cost"]))
        observations = np.concatenate([observations, next_observation[None]], axis=0)
        actions = np.concatenate([actions, action[None]], axis=0)
        observations, actions = planner._prepare_context(observations, actions)
        alive = bool(transition["alive"])
        if not alive:
            break
    return {
        "observations": np.asarray(executed_observations),
        "actions": np.asarray(executed_actions),
        "latent_costs": np.asarray(latent_costs),
        "alive": alive,
    }
