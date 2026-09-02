"""One-step, multi-step and SIGReg objectives for TEP latent dynamics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.module import SIGReg


class LeWMLoss(nn.Module):
    """Combine teacher-forced and rollout prediction losses with SIGReg."""

    def __init__(
        self, sigreg_weight: float = 0.09, rollout_weight: float = 1.0,
        knots: int = 17, num_proj: int = 1024,
    ):
        """Configure prediction weights and the SIGReg approximation."""
        super().__init__()
        if min(sigreg_weight, rollout_weight) < 0:
            raise ValueError("loss weights must be non-negative")
        self.sigreg_weight = sigreg_weight
        self.rollout_weight = rollout_weight
        self.sigreg = SIGReg(knots=knots, num_proj=num_proj)

    def forward(self, output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return total, one-step, rollout and regularization losses."""
        prediction = F.mse_loss(
            output["predicted_embeddings"], output["target_embeddings"]
        )
        rollout = prediction.new_zeros(())
        if "rollout_predicted_embeddings" in output:
            rollout = F.mse_loss(
                output["rollout_predicted_embeddings"],
                output["rollout_target_embeddings"],
            )
        regularization = self.sigreg(output["embeddings"].transpose(0, 1))
        total = (
            prediction + self.rollout_weight * rollout
            + self.sigreg_weight * regularization
        )
        return {
            "loss": total,
            "prediction_loss": prediction,
            "rollout_loss": rollout,
            "sigreg_loss": regularization,
        }
