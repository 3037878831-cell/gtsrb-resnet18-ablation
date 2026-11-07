"""Aggregate completed metrics.json files into CSV and Markdown tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.models import VARIANTS


output_root = Path("outputs")
rows = []
for variant in VARIANTS:
    path = output_root / variant / "metrics.json"
    if path.exists():
        rows.append(json.loads(path.read_text(encoding="utf-8")))

if not rows:
    raise SystemExit("No completed metrics.json files found.")

fields = [
    "variant", "best_epoch", "best_val_accuracy", "test_accuracy", "macro_precision",
    "macro_recall", "macro_f1", "parameters", "macs", "model_size_mb",
    "latency_ms_batch1", "training_seconds",
]
with (output_root / "experiment_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

header = "| Model | Params | MACs (M) | Val acc. (%) | Test acc. (%) | Macro-F1 (%) | Latency (ms) |"
separator = "|---|---:|---:|---:|---:|---:|---:|"
lines = [header, separator]
for row in rows:
    lines.append(
        f"| {row['variant']} | {row['parameters']:,} | {row['macs']/1e6:.2f} | "
        f"{100*row['best_val_accuracy']:.3f} | {100*row['test_accuracy']:.3f} | "
        f"{100*row['macro_f1']:.3f} | {row['latency_ms_batch1']:.3f} |"
    )
(output_root / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

