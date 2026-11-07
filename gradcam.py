"""Generate a Grad-CAM overlay from the final ResNet stage."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data import make_transforms
from src.labels import CLASS_NAMES
from src.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/combined/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/gradcam.png"))
    parser.add_argument("--class-id", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    image_size = int(checkpoint.get("config", {}).get("image_size", 32))
    model = build_model(checkpoint["variant"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    activations = []
    gradients = []
    # norm2 is the final convolutional feature map before residual addition;
    # hooking here avoids in-place activation/backward-hook conflicts.
    target_layer = model.layer4[-1].norm2
    forward_handle = target_layer.register_forward_hook(
        lambda module, inputs, output: activations.append(output)
    )
    backward_handle = target_layer.register_full_backward_hook(
        lambda module, grad_input, grad_output: gradients.append(grad_output[0])
    )

    original = Image.open(args.image).convert("RGB")
    _, transform = make_transforms(image_size)
    tensor = transform(original).unsqueeze(0).to(device)
    logits = model(tensor)
    class_id = int(logits.argmax(1).item()) if args.class_id is None else args.class_id
    model.zero_grad(set_to_none=True)
    logits[0, class_id].backward()
    forward_handle.remove()
    backward_handle.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
    heatmap = torch.nn.functional.interpolate(
        heatmap, size=(image_size, image_size), mode="bilinear", align_corners=False
    )[0, 0]
    heatmap -= heatmap.min()
    heatmap /= heatmap.max().clamp_min(1e-8)
    heatmap = heatmap.detach().cpu().numpy()
    display_image = np.asarray(original.resize((image_size, image_size))) / 255.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(display_image)
    axes[0].set_title("Input")
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(display_image)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title(CLASS_NAMES[class_id], fontsize=8)
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"class_id={class_id} class_name={CLASS_NAMES[class_id]} output={args.output}")


if __name__ == "__main__":
    main()

