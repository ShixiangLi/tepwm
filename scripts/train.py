"""Command-line entry point for TEP LeWM training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.training import train_world_model


def parse_args() -> argparse.Namespace:
    """Parse optional config, dataset, device and epoch overrides."""
    parser = argparse.ArgumentParser(description="Train TEP-adapted LeWorldModel")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def main() -> None:
    """Load project configuration, run training and print checkpoint locations."""
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    dataset_dir = args.dataset_dir or Path(config["training"]["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    result = train_world_model(
        config,
        dataset_dir=dataset_dir,
        device=args.device,
        max_epochs=args.epochs,
    )
    if result["rank"] == 0:
        print(f"devices: {result['world_size']} x {result['device']}")
        print(f"global batch size: {result['global_batch_size']}")
        print(f"train dataset: {result['train_dataset']}")
        print(f"validation dataset: {result['validation_dataset']}")
        print(f"best checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
