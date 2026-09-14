"""Visualization helpers for world-model training diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_history(
    history: str | Path | pd.DataFrame,
    *,
    sigreg_weight: float = 0.09,
    search_roots: Sequence[str | Path] = (),
    figsize: tuple[float, float] = (14, 4.5),
):
    """Plot total and component losses and return the best validation result."""
    frame = _load_history(history, search_roots)
    required = {
        "epoch", "train_loss", "validation_loss",
        "train_prediction_loss", "validation_prediction_loss",
        "train_sigreg_loss", "validation_sigreg_loss",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"training history is missing columns: {sorted(missing)}")
    if sigreg_weight < 0:
        raise ValueError("sigreg_weight must be non-negative")

    selection_column = "validation_prediction_loss"
    best = frame.loc[frame[selection_column].idxmin()]
    summary = {
        "best_epoch": int(best["epoch"]),
        "checkpoint_metric": selection_column,
        "best_checkpoint_metric": float(best[selection_column]),
        "best_validation_loss": float(best["validation_loss"]),
        "best_validation_prediction_loss": float(best["validation_prediction_loss"]),
    }
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    _line(axes[0], frame, "train_loss", "Train")
    _line(axes[0], frame, "validation_loss", "Validation")
    axes[0].axvline(
        summary["best_epoch"], color="#D55E00", linestyle="--", alpha=0.8,
        label=f"Best epoch: {summary['best_epoch']}",
    )
    axes[0].set(title="Total Training Objective", xlabel="Epoch", ylabel="Loss")

    components = (
        ("train_prediction_loss", "Train prediction", "-"),
        ("validation_prediction_loss", "Validation prediction", "--"),
        ("train_sigreg_loss", "Train weighted SIGReg", "-"),
        ("validation_sigreg_loss", "Validation weighted SIGReg", "--"),
    )
    colors = {"prediction": "#0072B2", "sigreg": "#009E73"}
    for column, label, style in components:
        color = next(value for key, value in colors.items() if key in column)
        values = frame[column] * sigreg_weight if "sigreg" in column else frame[column]
        axes[1].plot(frame["epoch"], values, label=label, linestyle=style, color=color)
    axes[1].set(title="Loss Components", xlabel="Epoch", ylabel="Contribution")
    for axis in axes:
        axis.legend(frameon=True)
        axis.grid(True, alpha=0.25)
        sns.despine(ax=axis)
    figure.tight_layout()
    return summary, figure, axes


def _load_history(
    history: str | Path | pd.DataFrame, search_roots: Sequence[str | Path]
) -> pd.DataFrame:
    """Load a history table, resolving either a JSON file or output directory."""
    if isinstance(history, pd.DataFrame):
        return history.copy()
    path = Path(history)
    if path.suffix.lower() != ".json":
        path /= "history.json"
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(Path(root) / path for root in search_roots)
    resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f"training history not found; checked: {candidates}")
    return pd.read_json(resolved)


def _line(axis, frame: pd.DataFrame, column: str, label: str) -> None:
    """Draw one consistently styled loss curve."""
    sns.lineplot(
        data=frame, x="epoch", y=column, label=label, linewidth=2, ax=axis
    )
