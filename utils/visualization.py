"""Reusable visualization helpers for TEP experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _to_2d_float_array(values) -> np.ndarray:
    """Convert array-like values to a non-empty 2-D float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("values must be a non-empty 2-D array")
    return array


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    """Standardize each variable while keeping constant variables finite."""
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    std[~np.isfinite(std) | (std < 1e-12)] = 1.0
    return (values - mean) / std


def _tick_positions(size: int, max_ticks: int) -> np.ndarray:
    """Return evenly spaced integer positions without duplicates."""
    count = min(size, max_ticks)
    return np.unique(np.linspace(0, size - 1, count, dtype=int))


def plot_variable_heatmap(
    time,
    values,
    *,
    labels: Sequence[str] | None = None,
    title: str = "Variables",
    standardize: bool = True,
    clip: float = 3.0,
    max_y_ticks: int = 15,
    max_time_points: int = 1000,
    figsize: tuple[float, float] = (14, 5),
):
    """Plot all variables in a time series as one heatmap.

    Standardization is enabled by default because TEP variables use different
    physical units. Constant variables are displayed as zero after scaling.
    """
    array = _to_2d_float_array(values)
    time_array = np.asarray(time, dtype=float).reshape(-1)
    if len(time_array) != array.shape[0]:
        raise ValueError("time length must match the number of samples")
    if not np.all(np.isfinite(time_array)):
        raise ValueError("time must contain only finite values")
    if labels is not None and len(labels) != array.shape[1]:
        raise ValueError("labels length must match the number of variables")

    sample_indices = _tick_positions(array.shape[0], max_time_points)
    plot_values = (
        _standardize_columns(array)[sample_indices]
        if standardize
        else array[sample_indices]
    )
    color_limit = clip if standardize else None

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.heatmap(
        plot_values.T,
        ax=ax,
        cmap="RdBu_r" if standardize else "viridis",
        center=0.0 if standardize else None,
        vmin=-color_limit if color_limit is not None else None,
        vmax=color_limit,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            "label": "Standardized value (z-score)" if standardize else "Value",
            "shrink": 0.85,
        },
    )

    x_positions = _tick_positions(len(sample_indices), 7)
    ax.set_xticks(x_positions + 0.5)
    ax.set_xticklabels([f"{time_array[sample_indices[i]]:.2f}" for i in x_positions])

    positions = _tick_positions(array.shape[1], max_y_ticks)
    tick_labels = (
        labels if labels is not None else [str(i + 1) for i in range(array.shape[1])]
    )
    ax.set_yticks(positions + 0.5)
    ax.set_yticklabels([tick_labels[i] for i in positions], rotation=0)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Variable")
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=12)
    return fig, ax


def plot_key_variable_lines(
    time,
    values,
    indices: Sequence[int],
    *,
    labels: Sequence[str] | None = None,
    title: str = "Key variables",
    max_time_points: int = 2000,
    figsize_per_row: tuple[float, float] = (14, 3.2),
):
    """Plot selected variables as small-multiple line charts in physical units."""
    array = _to_2d_float_array(values)
    time_array = np.asarray(time, dtype=float).reshape(-1)
    selected = np.asarray(indices, dtype=int).reshape(-1)
    if len(time_array) != array.shape[0]:
        raise ValueError("time length must match the number of samples")
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= array.shape[1]):
        raise ValueError("indices must reference existing variables")
    if labels is not None and len(labels) != selected.size:
        raise ValueError("labels length must match indices length")

    sample_indices = _tick_positions(array.shape[0], max_time_points)
    plot_time = time_array[sample_indices]
    n_columns = min(2, selected.size)
    n_rows = math.ceil(selected.size / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    colors = sns.color_palette("colorblind", selected.size)

    for plot_index, (ax, variable_index, color) in enumerate(
        zip(axes, selected, colors)
    ):
        sns.lineplot(
            x=plot_time,
            y=array[sample_indices, variable_index],
            ax=ax,
            color=color,
            linewidth=1.6,
            errorbar=None,
        )
        variable_label = (
            labels[plot_index]
            if labels is not None
            else f"Variable {variable_index + 1}"
        )
        ax.set_title(variable_label, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        sns.despine(ax=ax)

    for ax in axes[selected.size :]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    return fig, axes[: selected.size]


def plot_vector_bar(
    values,
    *,
    labels: Sequence[str] | None = None,
    title: str = "Values",
    log_scale: bool = False,
    figsize: tuple[float, float] = (14, 4),
):
    """Plot a one-dimensional variable group as a bar chart."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if labels is not None and len(labels) != array.size:
        raise ValueError("labels length must match values length")
    if log_scale and np.any(array <= 0):
        raise ValueError("log-scale values must all be positive")

    x = np.arange(array.size)
    tick_labels = labels if labels is not None else [str(i + 1) for i in x]
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    colors = sns.color_palette("crest", array.size)
    ax.bar(x, array, color=colors, width=0.75, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, tick_labels, rotation=45, ha="right")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=12)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax)
    return fig, ax


def plot_trajectory_overview(
    trajectory,
    *,
    action_sp_numbers: Sequence[int],
    title: str = "TEP trajectory",
    measurement_indices: Sequence[int] = (6, 7, 8, 17),
    figsize: tuple[float, float] = (14, 10),
):
    """Plot key measurements, SP actions, mode and disturbances together."""
    measurements = _to_2d_float_array(trajectory.measurements)
    actions = _to_2d_float_array(trajectory.actions)
    time = np.asarray(trajectory.time, dtype=float)
    action_time = time[:-1]
    if len(action_sp_numbers) != actions.shape[1]:
        raise ValueError("action_sp_numbers length must match the action dimension")

    selected = np.asarray(measurement_indices, dtype=int)
    if np.any(selected < 0) or np.any(selected >= measurements.shape[1]):
        raise ValueError("measurement_indices reference missing measurements")

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    colors = sns.color_palette("colorblind", len(selected))
    selected_measurements = _standardize_columns(measurements[:, selected])
    for index, (variable_index, color) in enumerate(zip(selected, colors)):
        sns.lineplot(
            x=time,
            y=selected_measurements[:, index],
            ax=axes[0],
            color=color,
            label=f"XMEAS({variable_index + 1})",
            linewidth=1.5,
            errorbar=None,
        )
    axes[0].set_ylabel("Standardized value")
    axes[0].set_title("Key process measurements")

    reference = actions[0]
    denominator = np.where(np.abs(reference) < 1e-12, 1.0, np.abs(reference))
    relative_actions = 100.0 * (actions - reference) / denominator
    action_colors = sns.color_palette("husl", actions.shape[1])
    for index, (number, color) in enumerate(zip(action_sp_numbers, action_colors)):
        axes[1].step(
            action_time,
            relative_actions[:, index],
            where="post",
            color=color,
            label=f"SP{number}",
            linewidth=1.25,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Change from initial (%)")
    axes[1].set_title("External SP actions")
    axes[1].legend(ncol=6, fontsize=8, loc="upper left")

    modes = np.asarray(trajectory.operating_modes)
    active_disturbances = np.asarray(trajectory.disturbances).sum(axis=1)
    axes[2].step(action_time, modes, where="post", color="#4C72B0", label="Mode")
    axes[2].set_ylabel("Operating mode")
    axes[2].set_yticks(sorted(set(modes.tolist())))
    disturbance_axis = axes[2].twinx()
    disturbance_axis.fill_between(
        action_time,
        active_disturbances,
        step="post",
        color="#C44E52",
        alpha=0.25,
        label="Active IDVs",
    )
    disturbance_axis.set_ylabel("Active disturbances")
    axes[2].set_title("Operating mode and disturbances")

    for axis in axes:
        axis.set_xlabel("Time (hours)")
        axis.grid(True, alpha=0.25)
        sns.despine(ax=axis)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    return fig, axes


def plot_counterfactual_comparison(
    reference,
    intervention,
    variables: Mapping[str, int],
    *,
    reference_label: str = "Reference",
    intervention_label: str = "Intervention",
    intervention_time_hours: float | None = None,
    final_window_minutes: float = 15.0,
    initial_tolerance: float = 1e-10,
    max_time_points: int = 1000,
    title: str = "Paired counterfactual trajectories",
):
    """Compare a strictly paired reference and intervention trajectory.

    Variable indices are zero-based XMEAS column indices. The returned table
    contains final-window means and intervention-minus-reference differences.
    """
    reference_time = np.asarray(reference.time, dtype=float)
    intervention_time = np.asarray(intervention.time, dtype=float)
    if reference_time.shape != intervention_time.shape or not np.allclose(
        reference_time, intervention_time
    ):
        raise ValueError("counterfactual trajectories must share the same time axis")

    reference_observations = _to_2d_float_array(reference.observations)
    intervention_observations = _to_2d_float_array(intervention.observations)
    if reference_observations.shape != intervention_observations.shape:
        raise ValueError("counterfactual observations must have the same shape")
    initial_difference = float(
        np.max(np.abs(reference_observations[0] - intervention_observations[0]))
    )
    if initial_difference > initial_tolerance:
        raise ValueError(
            f"initial observations differ by {initial_difference:.3e}, "
            f"above tolerance {initial_tolerance:.3e}"
        )

    if not isinstance(variables, Mapping) or not variables:
        raise ValueError("variables must be a non-empty label-to-index mapping")
    labels = list(variables)
    indices = np.asarray(list(variables.values()), dtype=int)
    reference_values = _to_2d_float_array(reference.measurements)
    intervention_values = _to_2d_float_array(intervention.measurements)
    if np.any(indices < 0) or np.any(indices >= reference_values.shape[1]):
        raise ValueError("variable indices reference missing measurements")

    if len(reference_time) < 2:
        raise ValueError("counterfactual trajectories need at least two samples")
    dt_hours = float(np.median(np.diff(reference_time)))
    window_steps = round((float(final_window_minutes) / 60.0) / dt_hours)
    if window_steps <= 0 or window_steps > len(reference_time):
        raise ValueError("final_window_minutes is incompatible with trajectory length")

    rows = []
    for label, index in zip(labels, indices):
        reference_final = reference_values[-window_steps:, index].mean()
        intervention_final = intervention_values[-window_steps:, index].mean()
        rows.append(
            {
                "variable": label,
                "reference final": reference_final,
                "intervention final": intervention_final,
                "counterfactual difference": intervention_final - reference_final,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.attrs["initial_observation_max_difference"] = initial_difference

    sample_indices = _tick_positions(len(reference_time), max_time_points)
    n_columns = min(2, len(indices))
    n_rows = math.ceil(len(indices) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(14, 3.2 * n_rows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    colors = ("#526D82", "#D95F59")
    for axis, label, index in zip(axes, labels, indices):
        axis.plot(
            reference_time[sample_indices],
            reference_values[sample_indices, index],
            color=colors[0],
            label=reference_label,
            linewidth=1.6,
        )
        axis.plot(
            reference_time[sample_indices],
            intervention_values[sample_indices, index],
            color=colors[1],
            label=intervention_label,
            linewidth=1.6,
        )
        if intervention_time_hours is not None:
            axis.axvline(
                intervention_time_hours,
                color="#333333",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )
        axis.set_title(label, fontsize=11, fontweight="semibold")
        axis.set_xlabel("Time (hours)")
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        sns.despine(ax=axis)
    for axis in axes[len(indices) :]:
        axis.set_visible(False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return comparison, fig, axes[: len(indices)]
