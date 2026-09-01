"""Visualization helpers for world-model evaluation results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_latent_trajectory_comparison(
    result: dict,
    *,
    figsize: tuple[float, float] = (14, 4.5),
):
    """Compare one predicted and actual latent trajectory in a shared PCA plane."""
    predicted = np.asarray(result["predicted_embeddings"], dtype=float)
    actual = np.asarray(result["actual_embeddings"], dtype=float)
    time = np.asarray(result["time_minutes"], dtype=float)
    error = np.asarray(result["latent_mse"], dtype=float)
    if predicted.shape != actual.shape or predicted.ndim != 2:
        raise ValueError("predicted and actual embeddings must share a 2-D shape")
    if len(time) != len(predicted) or len(error) != len(predicted):
        raise ValueError("time and error must align with the latent trajectories")

    combined = np.concatenate([actual, predicted], axis=0)
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:2].T
    actual_2d, predicted_2d = np.split(projected, 2)

    figure, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(
        actual_2d[:, 0], actual_2d[:, 1], marker="o", markersize=3,
        linewidth=2, color="#0072B2", label="Actual",
    )
    axes[0].plot(
        predicted_2d[:, 0], predicted_2d[:, 1], marker="o", markersize=3,
        linewidth=2, color="#D55E00", label="Predicted",
    )
    axes[0].scatter(
        actual_2d[0, 0], actual_2d[0, 1], s=90, marker="*", color="#009E73",
        label="Start", zorder=5,
    )
    axes[0].set(
        title="Actual vs Predicted Latent Trajectory",
        xlabel="Shared PCA component 1",
        ylabel="Shared PCA component 2",
    )
    sns.lineplot(x=time, y=error, linewidth=2, color="#CC79A7", ax=axes[1])
    axes[1].fill_between(time, 0, error, color="#CC79A7", alpha=0.15)
    axes[1].set(
        title="Latent Prediction Error over Time",
        xlabel="Rollout horizon (minutes)",
        ylabel="Latent MSE",
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
        sns.despine(ax=axis)
    axes[0].legend(frameon=True)
    figure.tight_layout()
    return figure, axes


def plot_latent_prediction_evaluation(
    result: dict,
    *,
    figsize: tuple[float, float] = (14, 4.5),
):
    """Plot horizon-wise latent MSE means and per-window distributions."""
    summary = result["summary"]
    per_window = result["per_window"]
    if not isinstance(summary, pd.DataFrame) or not isinstance(per_window, pd.DataFrame):
        raise TypeError("evaluation result must contain summary and per_window tables")

    figure, axes = plt.subplots(1, 2, figsize=figsize)
    sns.lineplot(
        data=summary,
        x="horizon",
        y="latent_mse_mean",
        hue="scenario_type",
        marker="o",
        linewidth=2,
        ax=axes[0],
    )
    axes[0].set(
        title="Latent Rollout Error by Horizon",
        xlabel="Prediction horizon (minutes)",
        ylabel="Mean latent MSE",
    )
    sns.boxplot(
        data=per_window,
        x="horizon",
        y="latent_mse",
        hue="scenario_type",
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set(
        title="Per-window Latent Error Distribution",
        xlabel="Prediction horizon (minutes)",
        ylabel="Latent MSE",
    )
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=True)
        sns.despine(ax=axis)
    figure.tight_layout()
    return figure, axes


def plot_planning_case(
    case: dict,
    *,
    key_xmeas: tuple[int, ...] = (7, 8, 9, 15),
    figsize: tuple[float, float] = (15, 9),
):
    """Compare one MPC execution with its reachable reference trajectory."""
    reference = np.asarray(case["reference_observations"], dtype=float)
    planned = np.asarray(case["planned_observations"], dtype=float)
    reference_actions = np.asarray(case["reference_actions"], dtype=float)
    planned_actions = np.asarray(case["planned_actions"], dtype=float)
    goal = np.asarray(case["goal_observation"], dtype=float)
    stats = case["normalization"]
    indices = np.asarray(key_xmeas, dtype=int) - 1
    if np.any((indices < 0) | (indices >= 41)):
        raise ValueError("key_xmeas numbers must be between 1 and 41")

    reference_z = stats.normalize_observations(reference)
    planned_z = stats.normalize_observations(planned)
    goal_z = stats.normalize_observations(goal)
    reference_error = np.square(reference_z[:, :41] - goal_z[:41]).mean(axis=1)
    planned_error = np.square(planned_z[:, :41] - goal_z[:41]).mean(axis=1)
    reference_time = np.asarray(case["time_minutes"], dtype=float)
    planned_time = reference_time[:len(planned)]

    figure, axes = plt.subplots(2, 2, figsize=figsize)
    sns.lineplot(x=reference_time, y=reference_error, ax=axes[0, 0],
                 color="#0072B2", linewidth=2.2, label="Reference")
    sns.lineplot(x=planned_time, y=planned_error, ax=axes[0, 0],
                 color="#D55E00", linewidth=2.2, label="CEM-MPC")
    axes[0, 0].set(
        title="Standardized XMEAS Distance to Goal",
        xlabel="Time after planning start (min)", ylabel="XMEAS MSE",
    )

    colors = sns.color_palette("colorblind", len(indices))
    for index, color in zip(indices, colors):
        label = f"XMEAS{index + 1}"
        axes[0, 1].plot(reference_time, reference_z[:, index], "--",
                        color=color, linewidth=1.8, alpha=0.75)
        axes[0, 1].plot(planned_time, planned_z[:, index], color=color,
                        linewidth=2.0, label=label)
        axes[0, 1].scatter(reference_time[-1], goal_z[index], color=color, s=32)
    axes[0, 1].set(
        title="Key XMEAS: Solid MPC / Dashed Reference",
        xlabel="Time after planning start (min)", ylabel="Standardized value",
    )
    axes[0, 1].legend(ncol=2, frameon=True)

    action_mean, action_std = stats.action_mean, stats.action_std
    reference_action_z = (reference_actions - action_mean) / action_std
    planned_action_z = (planned_actions - action_mean) / action_std
    limit = max(
        1.0, np.nanpercentile(np.abs(reference_action_z), 98),
        np.nanpercentile(np.abs(planned_action_z), 98),
    )
    labels = [f"SP{number}" for number in case["action_sp_numbers"]]
    for axis, values, title in (
        (axes[1, 0], reference_action_z, "Reference Actions"),
        (axes[1, 1], planned_action_z, "CEM-MPC Actions"),
    ):
        sns.heatmap(
            values.T, cmap="vlag", center=0, vmin=-limit, vmax=limit,
            yticklabels=labels, xticklabels=5, cbar_kws={"label": "Action z-score"},
            ax=axis,
        )
        axis.set(title=title, xlabel="Control step (min)", ylabel="")
    for axis in axes[0]:
        axis.grid(True, alpha=0.25)
        sns.despine(ax=axis)
    status = "success" if case["success"] else "failure"
    figure.suptitle(
        f"{case['trajectory_id']} ({case['scenario_type']}): {status}, "
        f"improvement={case['improvement']:.1%}", fontsize=14,
    )
    figure.tight_layout()
    return figure, axes
