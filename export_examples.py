"""Export deterministic test examples from the Udacity pickle file."""

import pickle
from pathlib import Path

from PIL import Image

from src.labels import CLASS_NAMES


source = Path("data/traffic-signs-data/test.p")
output = Path("outputs/examples")
output.mkdir(parents=True, exist_ok=True)
with source.open("rb") as handle:
    payload = pickle.load(handle)

requested_classes = [14, 25, 38]  # stop, road work, keep right
for class_id in requested_classes:
    index = next(i for i, label in enumerate(payload["labels"]) if int(label) == class_id)
    path = output / f"class_{class_id:02d}_index_{index}.png"
    Image.fromarray(payload["features"][index]).save(path)
    print(f"{path}: {CLASS_NAMES[class_id]}")

