"""Command-line interface for assignment experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import make_dataloaders
from .engine import evaluate_model, save_test_artifacts, train_model
from .models import ResNet18
from .models.vit import DEFAULT_VIT_CHECKPOINT, build_vit
from .reporting import plot_training_curves
from .utils import choose_device, save_json, seed_everything


def build_model(config: dict[str, Any]) -> nn.Module:
    if config["model"] == "resnet":
        return ResNet18(num_classes=10)
    return build_vit(
        mode=config["vit_mode"],
        num_classes=10,
        checkpoint=config["vit_checkpoint"],
    )


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("CINIC-10"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--max-eval-batches", type=int, default=None, help=argparse.SUPPRESS)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train, validate, and test a model")
    add_shared_arguments(train)
    train.add_argument("--model", choices=("resnet", "vit"), required=True)
    train.add_argument("--vit-mode", choices=("full", "head"), default="full")
    train.add_argument("--vit-checkpoint", default=DEFAULT_VIT_CHECKPOINT)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--validation-interval", type=int, default=None)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--beta1", type=float, default=0.9)
    train.add_argument("--beta2", type=float, default=0.999)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--log-interval", type=int, default=100)
    train.add_argument("--seed", type=int, default=677)
    train.add_argument("--augment", action="store_true")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--failed-examples", type=int, default=12)
    train.add_argument("--max-train-batches", type=int, default=None, help=argparse.SUPPRESS)
    train.add_argument("--output-root", type=Path, default=Path("outputs"))
    train.add_argument("--run-name", default=None)
    train.add_argument(
        "--force",
        action="store_true",
        help="delete and recreate an existing non-empty run directory",
    )

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved checkpoint")
    add_shared_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=None)
    evaluate.add_argument("--failed-examples", type=int, default=12)
    return parser


def _validated_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def run_train(args: argparse.Namespace) -> None:
    epochs = args.epochs if args.epochs is not None else (20 if args.model == "resnet" else 3)
    validation_interval = (
        args.validation_interval
        if args.validation_interval is not None
        else (5 if args.model == "resnet" else 1)
    )
    for name, value in (
        ("epochs", epochs),
        ("validation_interval", validation_interval),
        ("batch_size", args.batch_size),
        ("log_interval", args.log_interval),
    ):
        _validated_positive(name, value)

    run_name = args.run_name or f"{args.model}_{args.vit_mode if args.model == 'vit' else 'main'}_{datetime.now():%Y%m%d_%H%M%S}"
    output_root = args.output_root.resolve()
    output_dir = (output_root / run_name).resolve()
    if output_dir == output_root or output_root not in output_dir.parents:
        raise ValueError("run-name must identify a directory inside output-root")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise FileExistsError(
                f"Refusing to overwrite non-empty run directory: {output_dir}. "
                "Pass --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {
        "model": args.model,
        "vit_mode": args.vit_mode,
        "vit_checkpoint": args.vit_checkpoint,
        "data_dir": str(args.data_dir.resolve()),
        "epochs": epochs,
        "validation_interval": validation_interval,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "beta1": args.beta1,
        "beta2": args.beta2,
        "weight_decay": args.weight_decay,
        "log_interval": args.log_interval,
        "seed": args.seed,
        "augment": args.augment,
        "amp": not args.no_amp,
        "failed_examples": args.failed_examples,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
    }
    save_json(output_dir / "config.json", config)
    seed_everything(args.seed)
    device = choose_device(args.device)
    data = make_dataloaders(
        args.data_dir,
        args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config)
    summary = train_model(
        model=model,
        train_loader=data.train,
        valid_loader=data.valid,
        test_loader=data.test,
        class_names=data.class_names,
        device=device,
        output_dir=output_dir,
        config=config,
    )
    plot_training_curves(output_dir / "metrics.jsonl", output_dir / "training_curves.png")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {output_dir.resolve()}")


def run_evaluate(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    data = make_dataloaders(
        args.data_dir,
        config["model"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if data.class_names != checkpoint["class_names"]:
        raise ValueError("Checkpoint class ordering does not match this dataset")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    result = evaluate_model(
        model,
        data.test,
        nn.CrossEntropyLoss(),
        device,
        max_batches=args.max_eval_batches,
        collect_failures=args.failed_examples,
    )
    output_dir = args.output_dir or args.checkpoint.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = save_test_artifacts(result, data.class_names, output_dir)
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {output_dir.resolve()}")


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "train":
        run_train(args)
    else:
        run_evaluate(args)

