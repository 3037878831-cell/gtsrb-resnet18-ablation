"""Print parameter and MAC comparisons without training."""

import torch

from src.models import VARIANTS, build_model
from src.utils import count_macs, parameter_count


print("variant,parameters,macs")
for name in VARIANTS:
    model = build_model(name)
    print(f"{name},{parameter_count(model)},{count_macs(model)}")

