"""Create report-ready plots and per-class recall tables from saved outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.labels import CLASS_NAMES
from src.models import VARIANTS


root = Path("outputs")
figure_root = root / "figures"
figure_root.mkdir(parents=True, exist_ok=True)

metrics = []
for variant in VARIANTS:
    path = root / variant / "metrics.json"
    if path.exists():
        metrics.append(json.loads(path.read_text(encoding="utf-8")))

names = [row["variant"] for row in metrics]
x = np.arange(len(names))
width = 0.36
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(x - width / 2, [100 * row["test_accuracy"] for row in metrics], width, label="Accuracy")
ax.bar(x + width / 2, [100 * row["macro_f1"] for row in metrics], width, label="Macro-F1")
ax.set_ylabel("Score (%)")
ax.set_ylim(90, 100)
ax.set_xticks(x, names, rotation=20)
ax.set_title("GTSRB test performance")
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(figure_root / "test_performance.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(names, [row["latency_ms_batch1"] for row in metrics], color="#4c78a8")
axes[0].set(ylabel="Latency (ms/image)", title="Batch-1 inference latency")
axes[0].tick_params(axis="x", rotation=25)
axes[0].grid(axis="y", alpha=0.25)
axes[1].bar(names, [row["training_seconds"] / 60 for row in metrics], color="#f58518")
axes[1].set(ylabel="Training time (min)", title="Training time")
axes[1].tick_params(axis="x", rotation=25)
axes[1].grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(figure_root / "efficiency.png", dpi=200)
plt.close(fig)

robustness = {}
for variant in ("baseline", "combined"):
    path = root / variant / "robustness.json"
    robustness[variant] = json.loads(path.read_text(encoding="utf-8"))
conditions = ["clean", "blur", "dark", "noise"]
fig, ax = plt.subplots(figsize=(8.5, 4.5))
for offset, variant in zip((-width / 2, width / 2), ("baseline", "combined")):
    ax.bar(
        np.arange(len(conditions)) + offset,
        [100 * robustness[variant][condition] for condition in conditions],
        width,
        label=variant,
    )
ax.set_xticks(np.arange(len(conditions)), conditions)
ax.set(ylabel="Accuracy (%)", title="Robustness under image corruption", ylim=(75, 100))
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(figure_root / "robustness.png", dpi=200)
plt.close(fig)

for variant in ("baseline", "combined"):
    matrix = np.load(root / variant / "confusion_matrix.npy")
    recall = np.divide(
        np.diag(matrix), matrix.sum(axis=1), out=np.zeros(43), where=matrix.sum(axis=1) != 0
    )
    rows = [
        {
            "class_id": index,
            "class_name": CLASS_NAMES[index],
            "support": int(matrix[index].sum()),
            "correct": int(matrix[index, index]),
            "recall": float(recall[index]),
        }
        for index in range(43)
    ]
    with (root / variant / "per_class_recall.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    worst = sorted(rows, key=lambda row: row["recall"])[:5]
    print(variant, "worst classes:")
    for row in worst:
        print(
            f"  {row['class_id']:2d} {row['class_name']}: "
            f"{100*row['recall']:.2f}% ({row['correct']}/{row['support']})"
        )

print(f"Figures saved in {figure_root}")

