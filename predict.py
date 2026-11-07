"""Run Top-k prediction on one traffic-sign image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.data import make_transforms
from src.labels import CLASS_NAMES
from src.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/combined/best.pt"))
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    variant = checkpoint["variant"]
    image_size = int(checkpoint.get("config", {}).get("image_size", 32))
    model = build_model(variant).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _, transform = make_transforms(image_size)
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    values, indices = probabilities.topk(min(args.top_k, len(CLASS_NAMES)))
    result = [
        {
            "class_id": int(index),
            "class_name": CLASS_NAMES[int(index)],
            "probability": float(value),
        }
        for value, index in zip(values.cpu(), indices.cpu())
    ]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

