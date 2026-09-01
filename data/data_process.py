"""Convert generated TEP trajectories into normalized causal training windows."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NormalizationStats:
    """Training-split z-score statistics for observations and actions."""

    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def normalize_observations(self, values: np.ndarray) -> np.ndarray:
        """Normalize TEP observations with training-only statistics."""
        return (values - self.observation_mean) / self.observation_std

    def normalize_actions(self, values: np.ndarray) -> np.ndarray:
        """Normalize SP actions with training-only statistics."""
        return (values - self.action_mean) / self.action_std

    def to_dict(self) -> dict[str, list[float]]:
        """Convert statistics to a checkpoint-safe plain mapping."""
        return {
            "observation_mean": self.observation_mean.tolist(),
            "observation_std": self.observation_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> NormalizationStats:
        """Restore normalization statistics from a checkpoint mapping."""
        return cls(**{key: np.asarray(value, dtype=np.float32) for key, value in values.items()})


def load_manifest(dataset_dir: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    """Read dataset records and optionally retain one predefined group split."""
    root = Path(dataset_dir)
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if split is not None:
        records = [record for record in records if record["split"] == split]
    if not records:
        raise ValueError(f"no trajectories found for split={split!r}")
    for record in records:
        if not (root / record["file"]).is_file():
            raise FileNotFoundError(f"trajectory file not found: {record['file']}")
    return records


def compute_normalization(
    dataset_dir: str | Path,
    *,
    split: str = "train",
    sample_stride_steps: int = 60,
    minimum_std: float = 1e-6,
) -> NormalizationStats:
    """Stream training trajectories once to estimate observation/action z-scores."""
    if sample_stride_steps < 1 or minimum_std <= 0:
        raise ValueError("sample_stride_steps and minimum_std must be positive")
    root = Path(dataset_dir)
    records = load_manifest(root, split)
    totals: dict[str, dict[str, Any]] = {}
    for record in records:
        with np.load(root / record["file"], allow_pickle=False) as data:
            arrays = {
                "observation": data["observations"][::sample_stride_steps],
                "action": data["actions"][::sample_stride_steps],
            }
        for name, array in arrays.items():
            values = np.asarray(array, dtype=np.float64)
            values = values[np.isfinite(values).all(axis=1)]
            if not len(values):
                continue
            stats = totals.setdefault(
                name,
                {"count": 0, "sum": np.zeros(values.shape[1]), "square": np.zeros(values.shape[1])},
            )
            stats["count"] += len(values)
            stats["sum"] += values.sum(axis=0)
            stats["square"] += np.square(values).sum(axis=0)
    means, stds = {}, {}
    for name, values in totals.items():
        mean = values["sum"] / values["count"]
        variance = values["square"] / values["count"] - np.square(mean)
        means[name] = mean.astype(np.float32)
        stds[name] = np.sqrt(np.maximum(variance, minimum_std**2)).astype(np.float32)
    if set(means) != {"observation", "action"}:
        raise ValueError("normalization data are incomplete")
    return NormalizationStats(
        means["observation"], stds["observation"],
        means["action"], stds["action"],
    )


class TEPWindowDataset(Dataset):
    """Expose causal H-action/H+1-state windows from trajectory NPZ files."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        stats: NormalizationStats,
        *,
        history_size: int = 8,
        sample_stride_steps: int = 60,
        window_stride_steps: int = 60,
        preload: bool = False,
        cache_size: int = 4,
    ):
        """Index valid windows and optionally preload their normalized arrays."""
        if min(history_size, sample_stride_steps, window_stride_steps, cache_size) < 1:
            raise ValueError("window and cache parameters must be positive")
        self.root = Path(dataset_dir)
        self.records = load_manifest(self.root, split)
        self.stats = stats
        self.history_size = history_size
        self.sample_stride_steps = sample_stride_steps
        self.cache_size = cache_size
        self.index: list[tuple[int, int]] = []
        required_steps = history_size * sample_stride_steps
        for file_index, record in enumerate(self.records):
            last_start = int(record["transitions"]) - required_steps
            self.index.extend(
                (file_index, start)
                for start in range(0, last_start + 1, window_stride_steps)
            )
        if not self.index:
            raise ValueError(f"no valid {split} windows for the requested history")
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._preloaded: tuple[np.ndarray, np.ndarray] | None = None
        if preload:
            self._preloaded = self._preload_windows()
            self._cache.clear()

    def __len__(self) -> int:
        """Return the number of valid causal windows."""
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        """Return one normalized observation/action training window."""
        if self._preloaded is None:
            observations, actions = self._window(item)
        else:
            observations, actions = (array[item] for array in self._preloaded)
        return {
            "observations": torch.from_numpy(observations),
            "actions": torch.from_numpy(actions),
        }

    def describe(self) -> dict[str, Any]:
        """Summarize trajectory count, window count and effective time spacing."""
        return {
            "trajectory_count": len(self.records),
            "window_count": len(self),
            "history_size": self.history_size,
            "sample_stride_seconds": self.sample_stride_steps,
        }

    def _read_trajectory(self, file_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Load one trajectory with a small per-process least-recently-used cache."""
        if file_index in self._cache:
            self._cache.move_to_end(file_index)
            return self._cache[file_index]
        path = self.root / self.records[file_index]["file"]
        with np.load(path, allow_pickle=False) as data:
            arrays = (
                np.asarray(data["observations"], dtype=np.float32),
                np.asarray(data["actions"], dtype=np.float32),
            )
        self._cache[file_index] = arrays
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arrays

    def _window(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        """Slice and normalize one causally aligned window."""
        file_index, start = self.index[item]
        observations, actions = self._read_trajectory(file_index)
        indices = start + np.arange(self.history_size + 1) * self.sample_stride_steps
        obs = self.stats.normalize_observations(observations[indices])
        act = self.stats.normalize_actions(actions[indices[:-1]])
        return obs.astype(np.float32), act.astype(np.float32)

    def _preload_windows(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialize normalized windows once for fast randomized GPU training."""
        observations = np.empty(
            (len(self), self.history_size + 1, len(self.stats.observation_mean)),
            dtype=np.float32,
        )
        actions = np.empty(
            (len(self), self.history_size, len(self.stats.action_mean)),
            dtype=np.float32,
        )
        for item in range(len(self)):
            observations[item], actions[item] = self._window(item)
        return observations, actions
