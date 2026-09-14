"""Plain-PyTorch training loop for the TEP-adapted LeWM."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from data.data_process import NormalizationStats, TEPWindowDataset, compute_normalization
from data.data_validation import model_step_seconds
from models.losses import LeWMLoss
from models.model import build_tep_lewm


def train_world_model(
    config: dict[str, Any],
    *,
    dataset_dir: str | Path | None = None,
    device: str | None = None,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Train TEP-LeWM on one device or under torchrun DistributedDataParallel."""
    training = dict(config["training"])
    model_config = dict(config["model"])
    dataset_path = Path(dataset_dir or training["dataset_dir"])
    summary_path = dataset_path / "summary.json"
    dataset_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    stride = model_step_seconds(config["data"])
    if dataset_summary.get("model_step_seconds") != stride:
        raise ValueError("dataset model step does not match config; regenerate the dataset")
    epochs = int(max_epochs or training.get("epochs", 100))
    device_obj, rank, world_size, distributed, initialized_here = _setup_distributed(
        device or training.get("device", "auto")
    )
    is_main = rank == 0
    seed = int(training.get("seed", 42))
    _set_seed(seed + rank)
    if device_obj.type == "cuda":
        torch.set_float32_matmul_precision("high")

    stats = _distributed_normalization(
        dataset_path, str(training.get("train_split", "train")), stride,
        device_obj, rank, distributed,
    )
    dataset_options = {
        "stats": stats,
        "history_size": int(model_config["history_size"]),
        "sample_stride_steps": stride,
        "preload": bool(training.get("preload", True)),
    }
    train_set = TEPWindowDataset(
        dataset_path, str(training.get("train_split", "train")), **dataset_options
    )
    val_set = TEPWindowDataset(
        dataset_path, str(training.get("validation_split", "validation")),
        **dataset_options,
    )
    per_device_batch = int(training.get("batch_size", 256))
    loader_options = {
        "batch_size": per_device_batch,
        "num_workers": int(training.get("loader_workers", 0)),
        "pin_memory": device_obj.type == "cuda",
    }
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    ) if distributed else None
    validation_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True
    ) if distributed else None
    train_loader = DataLoader(
        train_set, sampler=train_sampler, shuffle=train_sampler is None,
        drop_last=True, **loader_options,
    )
    val_loader = DataLoader(
        val_set, sampler=validation_sampler, shuffle=False,
        drop_last=False, **loader_options,
    )

    model = build_tep_lewm(model_config).to(device_obj)
    if distributed:
        device_ids = [device_obj.index] if device_obj.type == "cuda" else None
        model = DistributedDataParallel(model, device_ids=device_ids)
    criterion = LeWMLoss(
        sigreg_weight=float(training.get("sigreg_weight", 0.09)),
        knots=int(training.get("sigreg_knots", 17)),
        num_proj=int(training.get("sigreg_projections", 1024)),
    ).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 5e-5)),
        weight_decay=float(training.get("weight_decay", 1e-3)),
    )
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    optimizer_steps = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = max(1, epochs * optimizer_steps)
    warmup_steps = int(training.get("warmup_epochs", 5)) * optimizer_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_multiplier(step, warmup_steps, total_steps)
    )

    output_dir = Path(training.get("output_dir", "checkpoints/tep_lewm"))
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    checkpoint_training = {
        **training,
        "model_step_seconds": stride,
        "world_size": world_size,
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation_steps": accumulation_steps,
        "global_batch_size": per_device_batch * world_size * accumulation_steps,
        "dataset_manifest_sha256": dataset_summary.get("manifest_sha256"),
        "dataset_config_sha256": dataset_summary.get("config_sha256"),
        "dataset_trajectory_count": dataset_summary.get("trajectory_count"),
        "dataset_total_transitions": dataset_summary.get("total_transitions"),
    }
    history, best_validation, stale_epochs = [], float("inf"), 0
    checkpoint_metric = "prediction_loss"
    patience = int(training.get("early_stopping_patience", 0))
    minimum_delta = float(training.get("early_stopping_min_delta", 0.0))
    if patience < 0 or minimum_delta < 0:
        raise ValueError("early stopping patience and minimum delta must be non-negative")
    checkpoint_training["checkpoint_metric"] = checkpoint_metric
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            model, criterion, train_loader, device_obj,
            optimizer=optimizer, scheduler=scheduler,
            gradient_clip=float(training.get("gradient_clip", 1.0)),
            gradient_accumulation_steps=accumulation_steps,
            description=f"Epoch {epoch}/{epochs} train",
            show_progress=is_main,
        )
        validation_metrics = _run_epoch(
            model, criterion, val_loader, device_obj,
            description=f"Epoch {epoch}/{epochs} validation",
            show_progress=is_main,
        )
        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_metrics)
        current_validation = validation_metrics[checkpoint_metric]
        improved = current_validation < best_validation - minimum_delta
        stale_epochs = 0 if improved else stale_epochs + 1
        if improved:
            best_validation = current_validation
        if is_main:
            checkpoint = _checkpoint(
                _unwrap_model(model), optimizer, epoch, model_config,
                checkpoint_training, stats.to_dict(), history,
            )
            torch.save(checkpoint, output_dir / "last.pt")
            if improved:
                torch.save(checkpoint, output_dir / "best.pt")
            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
        if patience and stale_epochs >= patience:
            if is_main:
                print(f"Early stopping after {stale_epochs} epochs without improvement")
            break
    result = {
        "model": _unwrap_model(model),
        "normalization": stats,
        "history": history,
        "train_dataset": train_set.describe(),
        "validation_dataset": val_set.describe(),
        "best_checkpoint": output_dir / "best.pt",
        "last_checkpoint": output_dir / "last.pt",
        "device": str(device_obj),
        "rank": rank,
        "world_size": world_size,
        "per_device_batch_size": per_device_batch,
        "gradient_accumulation_steps": accumulation_steps,
        "global_batch_size": per_device_batch * world_size * accumulation_steps,
        "checkpoint_metric": checkpoint_metric,
        "best_validation_metric": best_validation,
    }
    if distributed:
        dist.barrier()
    if initialized_here:
        dist.destroy_process_group()
    return result


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Restore a trained TEP LeWM and return its complete checkpoint metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_tep_lewm(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def _run_epoch(
    model: torch.nn.Module,
    criterion: LeWMLoss,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    gradient_clip: float = 1.0,
    gradient_accumulation_steps: int = 1,
    description: str = "",
    show_progress: bool = True,
) -> dict[str, float]:
    """Run an epoch and aggregate sample-weighted metrics across all ranks."""
    is_training = optimizer is not None
    model.train(is_training)
    totals = {
        "loss": 0.0, "prediction_loss": 0.0,
        "sigreg_loss": 0.0,
    }
    sample_count = 0
    progress = tqdm(
        loader, desc=description, unit="batch", leave=False, disable=not show_progress
    )
    for batch_index, batch in enumerate(progress):
        observations = batch["observations"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        batch_size = len(observations)
        if is_training and batch_index % gradient_accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
            ):
                output = model(observations, actions)
                losses = criterion(output)
            if is_training:
                group_start = batch_index // gradient_accumulation_steps * gradient_accumulation_steps
                group_size = min(gradient_accumulation_steps, len(loader) - group_start)
                (losses["loss"] / group_size).backward()
                if batch_index + 1 == group_start + group_size:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
        for name in totals:
            totals[name] += float(losses[name].detach()) * batch_size
        sample_count += batch_size
        if show_progress:
            progress.set_postfix(loss=f"{totals['loss'] / sample_count:.4f}")
    if dist.is_initialized():
        packed = torch.tensor(
            [*(totals[name] for name in totals), sample_count],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        totals = {name: float(packed[index]) for index, name in enumerate(totals)}
        sample_count = int(packed[-1].item())
    if not sample_count:
        raise ValueError("data loader produced no batches")
    return {name: value / sample_count for name, value in totals.items()}


def _setup_distributed(
    requested_device: str,
) -> tuple[torch.device, int, int, bool, bool]:
    """Initialize torchrun state and bind each process to its local GPU."""
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = dist.is_initialized() or env_world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _resolve_device(requested_device)
    if distributed and device.type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    initialized_here = distributed and not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    return device, rank, world_size, distributed, initialized_here


def _distributed_normalization(
    dataset_path: Path,
    split: str,
    stride: int,
    device: torch.device,
    rank: int,
    distributed: bool,
) -> NormalizationStats:
    """Compute training statistics once, then broadcast them to every rank."""
    stats = compute_normalization(
        dataset_path, split=split, sample_stride_steps=stride
    ) if rank == 0 else None
    if distributed:
        payload = [stats.to_dict() if stats is not None else None]
        dist.broadcast_object_list(payload, src=0, device=device)
        stats = NormalizationStats.from_dict(payload[0])
    assert stats is not None
    return stats


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model so DDP-free checkpoints remain portable."""
    return model.module if isinstance(model, DistributedDataParallel) else model


def _resolve_device(value: str) -> torch.device:
    """Resolve auto/cpu/cuda device selection and reject unavailable CUDA."""
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible data order and weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    """Compute the LeWM linear-warmup cosine-decay learning-rate multiplier."""
    if warmup_steps and step < warmup_steps:
        return max(1e-8, step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _checkpoint(
    model, optimizer, epoch, model_config, training_config, normalization, history
) -> dict[str, Any]:
    """Build a portable state-dict checkpoint with preprocessing metadata."""
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_config,
        "training_config": training_config,
        "normalization": normalization,
        "history": history,
    }
