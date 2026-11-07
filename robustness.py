"""Evaluate a trained model under reproducible image corruptions."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data import MEAN, STD
from src.models import build_model


class CorruptedTestSet(Dataset):
    def __init__(self, pickle_path: Path, corruption: str, image_size: int, seed: int = 2026):
        with pickle_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.features = payload["features"]
        self.labels = payload["labels"]
        self.corruption = corruption
        self.image_size = image_size
        self.seed = seed
        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.features[index]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if self.corruption == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
        elif self.corruption == "dark":
            image = ImageEnhance.Brightness(image).enhance(0.45)
        elif self.corruption == "noise":
            rng = np.random.default_rng(self.seed + index)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = np.clip(array + rng.normal(0, 0.08, array.shape), 0, 1)
            image = Image.fromarray(np.uint8(array * 255))
        elif self.corruption != "clean":
            raise ValueError(self.corruption)
        return self.to_tensor(image), int(self.labels[index])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/combined/best.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/traffic-signs-data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device):
    correct = 0
    total = 0
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predicted = model(images).argmax(1)
        correct += (predicted == labels).sum().item()
        total += labels.numel()
    return correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("config", {}).get("image_size", 32))
    model = build_model(checkpoint["variant"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    results = {}
    for corruption in ("clean", "blur", "dark", "noise"):
        dataset = CorruptedTestSet(args.data_root / "test.p", corruption, image_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        results[corruption] = evaluate(model, loader, device)
        print(f"{corruption}: {results[corruption]:.4%}")
    clean = results["clean"]
    results["mean_corrupted_accuracy"] = float(
        np.mean([results[name] for name in ("blur", "dark", "noise")])
    )
    results["mean_accuracy_drop"] = clean - results["mean_corrupted_accuracy"]
    output = args.checkpoint.parent / "robustness.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

