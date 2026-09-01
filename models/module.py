"""LeWM neural components adapted for low-dimensional TEP states.

The Transformer, action embedder and SIGReg follow lucas-maes/le-wm (MIT),
Copyright (c) 2026 Lucas Maes, with dependency-free tensor reshaping.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive LayerNorm shift and scale."""
    return x * (1 + scale) + shift


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer used by LeWM to prevent collapse."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        """Initialize Gaussian-test knots and random projection count."""
        super().__init__()
        if knots < 2 or num_proj < 1:
            raise ValueError("knots must be >= 2 and num_proj must be positive")
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Measure deviation from an isotropic Gaussian; input is (T, B, D)."""
        projections = torch.randn(
            embeddings.size(-1), self.num_proj, device=embeddings.device
        )
        projections = projections / projections.norm(p=2, dim=0).clamp_min(1e-12)
        projected = (embeddings @ projections).unsqueeze(-1) * self.t
        error = (
            (projected.cos().mean(-3) - self.phi).square()
            + projected.sin().mean(-3).square()
        )
        return ((error @ self.weights) * embeddings.size(-2)).mean()


class FeedForward(nn.Module):
    """Pre-normalized Transformer feed-forward sublayer."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        """Build the normalized two-layer feed-forward network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the last feature dimension independently at each time step."""
        return self.net(x)


class Attention(nn.Module):
    """Causal multi-head scaled dot-product attention."""

    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ):
        """Configure head geometry and attention projections."""
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if inner_dim != dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Attend over a (B,T,D) sequence with an optional causal mask."""
        batch, steps, _ = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = (
            value.view(batch, steps, self.heads, self.dim_head).transpose(1, 2)
            for value in qkv
        )
        dropout = self.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout, is_causal=causal
        )
        output = output.transpose(1, 2).reshape(batch, steps, -1)
        return self.to_out(output)


class ConditionalBlock(nn.Module):
    """LeWM Transformer block with zero-initialized action-conditioned AdaLN."""

    def __init__(
        self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0
    ):
        """Build one zero-initialized action-conditioned Transformer block."""
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Condition attention and MLP residuals on aligned action embeddings."""
        values = self.modulation(condition).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = values
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        return x + gate_m * self.mlp(modulate(self.norm2(x), shift_m, scale_m))


class ConditionalTransformer(nn.Module):
    """Stack action-conditioned LeWM blocks over a fixed causal history."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        """Stack conditional blocks between input and output projections."""
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.condition_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Predict contextualized state embeddings conditioned on actions."""
        x, condition = self.input_proj(x), self.condition_proj(condition)
        for block in self.layers:
            x = block(x, condition)
        return self.output_proj(self.norm(x))


class Embedder(nn.Module):
    """LeWM action embedder applied independently along the time axis."""

    def __init__(
        self, input_dim: int, emb_dim: int, smoothed_dim: int | None = None,
        mlp_scale: int = 4,
    ):
        """Build pointwise smoothing and nonlinear action projection layers."""
        super().__init__()
        smoothed_dim = smoothed_dim or input_dim
        self.smooth = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a (B,T,A) action sequence to (B,T,D) embeddings."""
        return self.embed(self.smooth(x.float().transpose(1, 2)).transpose(1, 2))


class MLP(nn.Module):
    """Two-layer projection MLP with configurable normalization."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int,
        norm: str = "layer",
    ):
        """Build a normalized two-layer feature projector."""
        super().__init__()
        norms = {"layer": nn.LayerNorm, "batch": nn.BatchNorm1d, "none": None}
        if norm not in norms:
            raise ValueError(f"unknown normalization: {norm}")
        norm_layer = nn.Identity() if norms[norm] is None else norms[norm](hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), norm_layer, nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the last dimension of a 2D tensor."""
        return self.net(x)


class ARPredictor(nn.Module):
    """LeWM autoregressive next-embedding predictor."""

    def __init__(
        self, *, num_frames: int, input_dim: int, hidden_dim: int, depth: int,
        heads: int, mlp_dim: int, output_dim: int | None = None,
        dim_head: int = 64, dropout: float = 0.0, emb_dropout: float = 0.0,
    ):
        """Initialize positional embeddings and the conditional Transformer."""
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = ConditionalTransformer(
            input_dim, hidden_dim, output_dim or input_dim, depth, heads,
            dim_head, mlp_dim, dropout,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Predict the next embedding at every causal position in the context."""
        steps = x.size(1)
        if steps > self.pos_embedding.size(1):
            raise ValueError("sequence exceeds predictor num_frames")
        return self.transformer(
            self.dropout(x + self.pos_embedding[:, :steps]), condition
        )
