"""TEP-specific LeWM joint-embedding predictive world model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from models.module import MLP, ARPredictor, Embedder


class TEPLeWM(nn.Module):
    """Encode TEP states and predict their next latent embeddings from SP actions."""

    def __init__(
        self,
        observation_dim: int = 53,
        action_dim: int = 11,
        latent_dim: int = 128,
        state_hidden_dim: int = 256,
        projector_hidden_dim: int = 512,
        history_size: int = 8,
        predictor_depth: int = 4,
        predictor_heads: int = 8,
        predictor_dim_head: int = 16,
        predictor_mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        """Assemble the TEP encoder, LeWM predictor and projection heads."""
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.history_size = history_size
        self.state_encoder = MLP(
            observation_dim, state_hidden_dim, latent_dim, norm="layer"
        )
        self.projector = MLP(
            latent_dim, projector_hidden_dim, latent_dim, norm="batch"
        )
        self.action_encoder = Embedder(action_dim, latent_dim)
        self.predictor = ARPredictor(
            num_frames=history_size,
            input_dim=latent_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            dropout=dropout,
        )
        self.pred_projector = MLP(
            latent_dim, projector_hidden_dim, latent_dim, norm="batch"
        )

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Map normalized (B,T,53) observations into LeWM latent embeddings."""
        if observations.ndim != 3 or observations.size(-1) != self.observation_dim:
            raise ValueError("observations must have shape (B,T,observation_dim)")
        batch, steps, _ = observations.shape
        flat = observations.float().reshape(batch * steps, -1)
        encoded = self.projector(self.state_encoder(flat))
        return encoded.reshape(batch, steps, self.latent_dim)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Map normalized (B,T,11) SP actions into conditioning embeddings."""
        if actions.ndim != 3 or actions.size(-1) != self.action_dim:
            raise ValueError("actions must have shape (B,T,action_dim)")
        return self.action_encoder(actions)

    def predict(
        self, embeddings: torch.Tensor, action_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Predict the next latent embedding at each causal context position."""
        predicted = self.predictor(embeddings, action_embeddings)
        batch, steps, _ = predicted.shape
        predicted = self.pred_projector(predicted.reshape(batch * steps, -1))
        return predicted.reshape(batch, steps, self.latent_dim)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        future_actions: torch.Tensor | None = None,
        rollout_targets: torch.Tensor | None = None,
        rollout_horizons: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute teacher-forced and optional autoregressive latent predictions."""
        if observations.size(1) != actions.size(1) + 1:
            raise ValueError("one more observation than action is required")
        embeddings = self.encode_observations(observations)
        predicted = self.predict(
            embeddings[:, :-1], self.encode_actions(actions)
        )
        output = {
            "embeddings": embeddings,
            "predicted_embeddings": predicted,
            "target_embeddings": embeddings[:, 1:],
        }
        supplied = (future_actions is not None, rollout_targets is not None)
        if any(supplied) and not all(supplied):
            raise ValueError("future actions and rollout targets must be supplied together")
        if future_actions is None:
            return output
        horizons = tuple(sorted(set(map(int, rollout_horizons or (1,)))))
        if not horizons or horizons[0] < 1:
            raise ValueError("rollout horizons must contain positive integers")
        maximum = horizons[-1]
        if maximum > min(future_actions.size(1), rollout_targets.size(1)):
            raise ValueError("rollout horizon exceeds the supplied future sequence")
        rollout = self.rollout(
            observations[:, :-1], actions[:, :-1], future_actions[:, :maximum]
        )
        target_rollout = self.encode_observations(rollout_targets[:, :maximum])
        selected = torch.as_tensor(
            [horizon - 1 for horizon in horizons], device=rollout.device
        )
        output["rollout_predicted_embeddings"] = rollout.index_select(1, selected)
        output["rollout_target_embeddings"] = target_rollout.index_select(1, selected)
        return output

    def rollout(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressively roll out latent states for one or many action candidates."""
        if future_actions.ndim == 4:
            batch, samples, horizon, action_dim = future_actions.shape
            history = observation_history[:, None].expand(-1, samples, -1, -1)
            past = action_history[:, None].expand(-1, samples, -1, -1)
            result = self._rollout_flat(
                history.reshape(batch * samples, history.size(2), -1),
                past.reshape(batch * samples, past.size(2), -1),
                future_actions.reshape(batch * samples, horizon, action_dim),
            )
            return result.reshape(batch, samples, horizon, self.latent_dim)
        return self._rollout_flat(observation_history, action_history, future_actions)

    def _rollout_flat(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Run latent rollout for flattened (batch,candidate) dimensions."""
        history = observation_history.size(1)
        if history > self.history_size or action_history.size(1) != history - 1:
            raise ValueError("action history must align with observation history")
        embeddings = self.encode_observations(observation_history)
        past_actions = action_history
        predictions = []
        for step in range(future_actions.size(1)):
            action = future_actions[:, step : step + 1]
            context_actions = torch.cat([past_actions, action], dim=1)
            context_embeddings = embeddings[:, -history:]
            predicted = self.predict(
                context_embeddings, self.encode_actions(context_actions[:, -history:])
            )[:, -1:]
            predictions.append(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            past_actions = torch.cat([past_actions, action], dim=1)
            if history > 1:
                past_actions = past_actions[:, -(history - 1) :]
            else:
                past_actions = past_actions[:, :0]
        return torch.cat(predictions, dim=1)

    def goal_cost(
        self, predicted_embeddings: torch.Tensor, goal_observations: torch.Tensor
    ) -> torch.Tensor:
        """Return final-step latent MSE cost for model-predictive planning."""
        goal = self.encode_observations(goal_observations[:, None])[:, 0]
        final_prediction = predicted_embeddings[..., -1, :]
        while goal.ndim < final_prediction.ndim:
            goal = goal.unsqueeze(1)
        return (final_prediction - goal).square().mean(dim=-1)


def build_tep_lewm(config: Mapping[str, Any]) -> TEPLeWM:
    """Construct the TEP LeWM model from the project model configuration."""
    return TEPLeWM(**dict(config))
