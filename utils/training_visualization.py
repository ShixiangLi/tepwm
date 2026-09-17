"""Visualization helpers for world-model training diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_training_history(
    history: str | Path | pd.DataFrame,
    *,
    sigreg_weight: float = 0.09,
    checkpoint_epoch: int | None = None,
    figsize: tuple[float, float] = (14, 4.5),
):
    """Distinguish the minimum validation loss from the saved checkpoint epoch."""
    frame = _load_history(history)
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
    minimum = frame.loc[frame[selection_column].idxmin()]
    summary = {
        "minimum_validation_epoch": int(minimum["epoch"]),
        "minimum_validation_prediction_loss": float(minimum[selection_column]),
        "checkpoint_metric": selection_column,
    }
    if checkpoint_epoch is not None:
        selected = frame.loc[frame["epoch"] == checkpoint_epoch]
        if len(selected) != 1:
            raise ValueError("checkpoint_epoch must identify one row in training history")
        summary.update(
            checkpoint_epoch=int(checkpoint_epoch),
            checkpoint_validation_prediction_loss=float(selected.iloc[0][selection_column]),
        )
    figure, axes = plt.subplots(1, 2, figsize=figsize)
    _line(axes[0], frame, "train_loss", "Train")
    _line(axes[0], frame, "validation_loss", "Validation")
    axes[0].axvline(
        minimum["epoch"], color="#009E73", linestyle=":", alpha=0.8,
        label=f"Minimum validation loss: {int(minimum['epoch'])}",
    )
    if checkpoint_epoch is not None:
        axes[0].axvline(
            checkpoint_epoch, color="#D55E00", linestyle="--", alpha=0.8,
            label=f"Saved checkpoint: {checkpoint_epoch}",
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


def _load_history(history: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load an explicit history file, output directory or in-memory table."""
    if isinstance(history, pd.DataFrame):
        return history.copy()
    path = Path(history)
    if path.suffix.lower() != ".json":
        path /= "history.json"
    return pd.read_json(path)


def _line(axis, frame: pd.DataFrame, column: str, label: str) -> None:
    """Draw one consistently styled loss curve."""
    sns.lineplot(
        data=frame, x="epoch", y=column, label=label, linewidth=2, ax=axis
    )
