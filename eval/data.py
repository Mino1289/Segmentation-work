"""ADE20K dataset and personal image loading for visualization.

Note: on ``merve/scene_parse_150``, only ``train`` and ``validation`` include
pixel annotations. The ``test`` split has images only (no ground truth).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import Dataset, load_dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSONAL_IMAGE_PATHS = [
    str(PROJECT_ROOT / "dataset" / "dog.jpg"),
    str(PROJECT_ROOT / "dataset" / "rue.jpg"),
]

_DATASET_CACHE: dict = {}


def load_ade20k_split(split: str = "test") -> Dataset:
    if split not in _DATASET_CACHE:
        raw = load_dataset("merve/scene_parse_150")
        if split not in raw:
            raise ValueError(f"Unknown split '{split}'. Available: {list(raw.keys())}")
        _DATASET_CACHE[split] = raw[split]
    return _DATASET_CACHE[split]


def _decode_mask(mask_img) -> np.ndarray:
    if mask_img is None:
        raise ValueError("Annotation mask is missing for this sample.")
    mask = np.array(mask_img, dtype=np.int32) - 1
    return mask


def get_ade20k_sample(
    index: int, split: str = "test"
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    dataset = load_ade20k_split(split)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} out of range for split '{split}' ({len(dataset)} samples)")

    item = dataset[index]
    image = item["image"].convert("RGB")
    mask_img = item.get("annotation")
    mask = _decode_mask(mask_img) if mask_img is not None else None
    return image, mask


def get_ade20k_sample_count(split: str = "test") -> int:
    return len(load_ade20k_split(split))


def list_test_indices(seed: int = 42, n: int = 10) -> List[int]:
    rng = np.random.default_rng(seed)
    count = get_ade20k_sample_count("test")
    n = min(n, count)
    return sorted(rng.choice(count, size=n, replace=False).tolist())


def load_personal_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Personal image not found: {path}")
    return Image.open(path).convert("RGB")


def get_available_personal_images() -> List[str]:
    return [p for p in PERSONAL_IMAGE_PATHS if os.path.isfile(p)]
