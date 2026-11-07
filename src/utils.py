"""Reproducibility, metrics, complexity and plotting helpers."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, input_shape=(1, 3, 32, 32)) -> int:
    macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal macs
        batch, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        kernel_ops = kernel_h * kernel_w * module.in_channels // module.groups
        macs += batch * out_channels * out_h * out_w * kernel_ops

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal macs
        macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.zeros(*input_shape, device=device))
    model.train(training)
    for hook in hooks:
        hook.remove()
    return int(macs)


def confusion_matrix(targets: list[int], predictions: list[int], classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(targets), np.asarray(predictions)), 1)
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict[str, float]:
    diagonal = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted != 0)
    recall = np.divide(diagonal, actual, out=np.zeros_like(diagonal), where=actual != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(diagonal.sum() / matrix.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv(row: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_history(rows: list[dict], path: str | Path) -> None:
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [100 * row["train_accuracy"] for row in rows], label="Train")
    axes[1].plot(epochs, [100 * row["val_accuracy"] for row in rows], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, path: str | Path) -> None:
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=np.float64),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Normalized confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

