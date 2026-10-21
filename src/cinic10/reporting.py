"""Metrics and report-ready visualizations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CINIC10_MEAN, CINIC10_STD


def confusion_matrix(
    targets: Iterable[int], predictions: Iterable[int], num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def per_class_accuracy(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1)
    return np.divide(
        matrix.diagonal(),
        totals,
        out=np.zeros(matrix.shape[0], dtype=np.float64),
        where=totals != 0,
    )


def plot_training_curves(metrics_path: Path, output_path: Path) -> None:
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train = [record for record in records if record["kind"] == "train_step"]
    valid = [record for record in records if record["kind"] == "validation"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if train:
        axes[0].plot(
            [record["global_step"] for record in train],
            [record["loss"] for record in train],
            linewidth=1.5,
        )
    axes[0].set(title="Training loss", xlabel="Optimizer step", ylabel="Cross-entropy loss")
    axes[0].grid(alpha=0.25)

    if valid:
        axes[1].plot(
            [record["epoch"] for record in valid],
            [100 * record["accuracy"] for record in valid],
            marker="o",
        )
    axes[1].set(title="Validation accuracy", xlabel="Epoch", ylabel="Top-1 accuracy (%)")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_confusion_matrix(
    matrix: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    size = max(7, 0.75 * len(class_names))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="Test confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_per_class_accuracy(
    matrix: np.ndarray, class_names: list[str], output_path: Path
) -> None:
    accuracies = per_class_accuracy(matrix)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "correct", "total", "accuracy", "accuracy_percent"])
        for index, name in enumerate(class_names):
            correct = int(matrix[index, index])
            total = int(matrix[index].sum())
            writer.writerow([name, correct, total, f"{accuracies[index]:.6f}", f"{100 * accuracies[index]:.2f}"])


def save_failed_examples(
    failures: list[tuple[torch.Tensor, int, int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    if not failures:
        figure, axis = plt.subplots(figsize=(6, 3))
        axis.text(
            0.5,
            0.5,
            "No misclassified examples were found.",
            ha="center",
            va="center",
            fontsize=12,
        )
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(output_path, dpi=180)
        plt.close(figure)
        return
    columns = min(4, len(failures))
    rows = math.ceil(len(failures) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.1 * rows), squeeze=False)
    mean = torch.tensor(CINIC10_MEAN).view(3, 1, 1)
    std = torch.tensor(CINIC10_STD).view(3, 1, 1)
    for axis, failure in zip(axes.flat, failures):
        image, target, prediction = failure
        display = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axis.imshow(display)
        axis.set_title(f"True: {class_names[target]}\nPred: {class_names[prediction]}", fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(failures) :]:
        axis.axis("off")
    figure.suptitle("Representative failed test cases")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

