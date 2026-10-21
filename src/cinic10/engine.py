"""Training and evaluation loops shared by ResNet and ViT experiments."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .reporting import (
    confusion_matrix,
    save_confusion_matrix,
    save_failed_examples,
    save_per_class_accuracy,
)
from .utils import append_jsonl, parameter_counts, save_json


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    targets: list[int]
    predictions: list[int]
    failures: list[tuple[torch.Tensor, int, int]]


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
    collect_failures: int = 0,
) -> EvaluationResult:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    targets: list[int] = []
    predictions: list[int] = []
    failure_buckets: dict[int, list[tuple[torch.Tensor, int, int]]] = {}
    failure_limit_per_class: int | None = None

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images_device = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(images_device)
            loss = criterion(logits, labels_device)
            predicted = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            correct += (predicted == labels_device).sum().item()
            targets.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())

            if collect_failures:
                if failure_limit_per_class is None:
                    failure_limit_per_class = max(
                        1, math.ceil(collect_failures / logits.shape[1])
                    )
                wrong_indices = (predicted != labels_device).nonzero(as_tuple=False).flatten().cpu()
                for index in wrong_indices:
                    item = int(index.item())
                    target = int(labels[item])
                    bucket = failure_buckets.setdefault(target, [])
                    if len(bucket) < failure_limit_per_class:
                        bucket.append((images[item].clone(), target, int(predicted[item])))

    if total_examples == 0:
        raise RuntimeError("Evaluation dataloader produced no examples")
    failures: list[tuple[torch.Tensor, int, int]] = []
    if collect_failures:
        max_bucket_size = max((len(bucket) for bucket in failure_buckets.values()), default=0)
        for item_index in range(max_bucket_size):
            for target in sorted(failure_buckets):
                bucket = failure_buckets[target]
                if item_index < len(bucket):
                    failures.append(bucket[item_index])
                    if len(failures) == collect_failures:
                        break
            if len(failures) == collect_failures:
                break

    return EvaluationResult(
        loss=total_loss / total_examples,
        accuracy=correct / total_examples,
        targets=targets,
        predictions=predictions,
        failures=failures,
    )


def save_test_artifacts(
    result: EvaluationResult,
    class_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    matrix = confusion_matrix(result.targets, result.predictions, len(class_names))
    save_confusion_matrix(matrix, class_names, output_dir / "confusion_matrix.png")
    save_per_class_accuracy(matrix, class_names, output_dir / "per_class_accuracy.csv")
    save_failed_examples(result.failures, class_names, output_dir / "failed_examples.png")
    return {
        "loss": result.loss,
        "accuracy": result.accuracy,
        "examples": len(result.targets),
        "confusion_matrix": matrix.tolist(),
    }


def train_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metrics_path = output_dir / "metrics.jsonl"
    best_accuracy = -1.0
    best_epoch = 0
    global_step = 0
    interval_loss = 0.0
    interval_examples = 0
    started = time.perf_counter()

    print(f"Device: {device}; parameters: {parameter_counts(model)}")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config['epochs']}", leave=False)
        for batch_index, (images, labels) in enumerate(progress):
            if config["max_train_batches"] is not None and batch_index >= config["max_train_batches"]:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            interval_loss += loss.item() * labels.size(0)
            interval_examples += labels.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if global_step % config["log_interval"] == 0:
                average_loss = interval_loss / interval_examples
                record = {
                    "kind": "train_step",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": average_loss,
                }
                append_jsonl(metrics_path, record)
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={average_loss:.4f}"
                )
                interval_loss = 0.0
                interval_examples = 0

        if interval_examples:
            append_jsonl(
                metrics_path,
                {
                    "kind": "train_step",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": interval_loss / interval_examples,
                },
            )
            interval_loss = 0.0
            interval_examples = 0

        validate_now = epoch % config["validation_interval"] == 0 or epoch == config["epochs"]
        if validate_now:
            validation = evaluate_model(
                model,
                valid_loader,
                criterion,
                device,
                max_batches=config["max_eval_batches"],
            )
            append_jsonl(
                metrics_path,
                {
                    "kind": "validation",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": validation.loss,
                    "accuracy": validation.accuracy,
                },
            )
            print(
                f"epoch={epoch} validation_loss={validation.loss:.4f} "
                f"validation_accuracy={100 * validation.accuracy:.2f}%"
            )
            if validation.accuracy > best_accuracy:
                best_accuracy = validation.accuracy
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "class_names": class_names,
                        "epoch": epoch,
                        "validation_accuracy": validation.accuracy,
                    },
                    output_dir / "best.pt",
                )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "class_names": class_names,
            "epoch": config["epochs"],
        },
        output_dir / "final.pt",
    )

    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_result = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        max_batches=config["max_eval_batches"],
        collect_failures=config["failed_examples"],
    )
    test_summary = save_test_artifacts(test_result, class_names, output_dir)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "duration_seconds": time.perf_counter() - started,
        "parameters": parameter_counts(model),
        "test": test_summary,
    }
    save_json(output_dir / "summary.json", summary)
    return summary

