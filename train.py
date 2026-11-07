"""Train and evaluate one GTSRB ResNet-18 variant."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.data import build_loaders
from src.models import VARIANTS, build_model
from src.utils import (
    append_csv,
    classification_metrics,
    confusion_matrix,
    count_macs,
    parameter_count,
    plot_confusion,
    plot_history,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="baseline")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, limit=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0
    predictions: list[int] = []
    targets: list[int] = []
    progress = tqdm(loader, leave=False, desc="train" if training else "eval")
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, (images, labels) in enumerate(progress, start=1):
            if limit and batch_index > limit:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.size(0)
            predicted = logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            correct += (predicted == labels).sum().item()
            seen += batch_size
            predictions.extend(predicted.detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
            progress.set_postfix(loss=f"{total_loss / seen:.4f}", acc=f"{correct / seen:.4f}")
    return total_loss / seen, correct / seen, targets, predictions


@torch.inference_mode()
def benchmark_latency(model, device, image_size: int, warmup=50, iterations=200):
    model.eval()
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / iterations


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.output_root / args.variant
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    if history_path.exists():
        history_path.unlink()

    train_loader, val_loader, test_loader = build_loaders(
        args.data_root,
        args.output_root / "split_seed2026.json",
        args.batch_size,
        args.workers,
        args.seed,
        args.image_size,
    )
    model = build_model(args.variant).to(device)
    params = parameter_count(model)
    macs = count_macs(model, (1, 3, args.image_size, args.image_size))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config = vars(args).copy()
    config.update(
        {
            "data_root": str(args.data_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "parameters": params,
            "macs": macs,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
        }
    )
    save_json(config, run_dir / "config.json")
    print(
        f"variant={args.variant} device={device} params={params:,} MACs={macs / 1e6:.2f}M "
        f"samples={len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}"
    )

    best_accuracy = -1.0
    stale_epochs = 0
    history = []
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            args.limit_train_batches,
        )
        val_loss, val_accuracy, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            limit=args.limit_eval_batches,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        append_csv(row, history_path)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_accuracy:.4%} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4%} time={row['epoch_seconds']:.1f}s"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "variant": args.variant,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "config": config,
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= args.early_stopping:
            print(f"Early stopping after {epoch} epochs")
            break

    training_seconds = time.perf_counter() - training_start
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_accuracy, targets, predictions = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        limit=args.limit_eval_batches,
    )
    matrix = confusion_matrix(targets, predictions, 43)
    metrics = classification_metrics(matrix)
    metrics.update(
        {
            "variant": args.variant,
            "best_epoch": checkpoint["epoch"],
            "best_val_accuracy": checkpoint["val_accuracy"],
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "parameters": params,
            "macs": macs,
            "model_size_mb": (run_dir / "best.pt").stat().st_size / (1024**2),
            "latency_ms_batch1": benchmark_latency(model, device, args.image_size),
            "training_seconds": training_seconds,
            "evaluated_test_samples": len(targets),
        }
    )
    save_json(metrics, run_dir / "metrics.json")
    npy_path = run_dir / "confusion_matrix.npy"
    import numpy as np
    np.save(npy_path, matrix)
    plot_history(history, run_dir / "training_curves.png")
    plot_confusion(matrix, run_dir / "confusion_matrix.png")
    print("FINAL", metrics)


if __name__ == "__main__":
    main()

