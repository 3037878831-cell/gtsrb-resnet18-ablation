"""Dataset construction and deterministic train/validation splitting."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import GTSRB


MEAN = (0.3337, 0.3064, 0.3171)
STD = (0.2672, 0.2564, 0.2629)


class TransformSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], transform) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, target = self.dataset[self.indices[item]]
        return self.transform(image), target


class ArrayDataset(Dataset):
    """Read the fixed Udacity GTSRB split stored as numpy arrays."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, transform) -> None:
        self.features = features
        self.labels = labels.astype(np.int64, copy=False)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.features[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])


def _load_pickle_split(path: Path):
    # Files come from the public Udacity self-driving course dataset mirror.
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["features"], payload["labels"]


def make_transforms(image_size: int = 32):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 8, image_size + 8)),
            transforms.RandomCrop(image_size),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_transform, eval_transform


def stratified_split(targets: list[int], val_ratio: float, seed: int):
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        val_count = max(1, round(len(class_indices) * val_ratio))
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _read_or_create_split(dataset: GTSRB, path: Path, val_ratio: float, seed: int):
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["train_indices"], payload["val_indices"]

    targets = [int(target) for _, target in dataset._samples]
    train_indices, val_indices = stratified_split(targets, val_ratio, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "val_ratio": val_ratio,
                "train_indices": train_indices,
                "val_indices": val_indices,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return train_indices, val_indices


def build_loaders(
    data_root: str | Path,
    split_file: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    image_size: int = 32,
    val_ratio: float = 0.1,
    download: bool = True,
):
    data_root = Path(data_root)
    train_transform, eval_transform = make_transforms(image_size)
    pickle_root = data_root / "traffic-signs-data"
    if all((pickle_root / name).exists() for name in ("train.p", "valid.p", "test.p")):
        train_x, train_y = _load_pickle_split(pickle_root / "train.p")
        val_x, val_y = _load_pickle_split(pickle_root / "valid.p")
        test_x, test_y = _load_pickle_split(pickle_root / "test.p")
        train_dataset = ArrayDataset(train_x, train_y, train_transform)
        val_dataset = ArrayDataset(val_x, val_y, eval_transform)
        test_dataset = ArrayDataset(test_x, test_y, eval_transform)
        split_path = Path(split_file)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(
            json.dumps(
                {
                    "source": "Udacity GTSRB fixed split",
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "test_samples": len(test_dataset),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        base_train = GTSRB(root=data_root, split="train", download=download, transform=None)
        test_dataset = GTSRB(root=data_root, split="test", download=download, transform=None)
        train_indices, val_indices = _read_or_create_split(
            base_train, Path(split_file), val_ratio, seed
        )
        train_dataset = TransformSubset(base_train, train_indices, train_transform)
        val_dataset = TransformSubset(base_train, val_indices, eval_transform)
        test_dataset.transform = eval_transform

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader

