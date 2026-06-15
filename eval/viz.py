"""Visualization helpers for ADE20K segmentation."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

from eval.ade20k_classes import get_class_name, present_class_ids


def create_ade20k_palette(num_classes: int = 150, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, len(palette) - 1)
    return palette[mask]


def pil_to_numpy(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image)
    return image


def get_display_image(image: Union[Image.Image, np.ndarray], model_name: str) -> np.ndarray:
    """Apply the same resize+center-crop as model inference (without normalization)."""
    from eval.model_registry import get_display_transform

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    tensor = get_display_transform(model_name)(image)
    arr = np.array(tensor)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def get_display_mask(mask: np.ndarray, model_name: str) -> np.ndarray:
    """Apply the same resize+center-crop as model inference, preserving labels."""
    from eval.model_registry import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_name]
    if mask.ndim != 2:
        raise ValueError("Expected a 2D segmentation mask.")

    image = Image.fromarray(mask.astype(np.int32), mode="I")
    width, height = image.size
    short_side = min(width, height)
    scale = cfg.img_size / short_side
    resized_size = (int(round(width * scale)), int(round(height * scale)))

    image = image.resize(resized_size, Image.Resampling.NEAREST)

    left = max(0, (image.size[0] - cfg.img_size) // 2)
    top = max(0, (image.size[1] - cfg.img_size) // 2)
    image = image.crop((left, top, left + cfg.img_size, top + cfg.img_size))

    return np.array(image, dtype=np.int32)


def make_overlay(
    image: Union[Image.Image, np.ndarray],
    color_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    base = pil_to_numpy(image).astype(np.float32)
    overlay = 0.6 * base + 0.4 * color_mask.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def collect_present_classes(predictions: dict) -> List[int]:
    class_ids: set[int] = set()
    for mask in predictions.values():
        class_ids.update(present_class_ids(mask))
    return sorted(class_ids)


def add_class_legend(
    fig: plt.Figure,
    class_ids: Sequence[int],
    palette: np.ndarray,
    ncol: int = 6,
) -> float:
    """Add class color legend below the figure. Returns bottom margin used."""
    if not class_ids:
        return 0.04

    patches = [
        Patch(
            facecolor=palette[cid] / 255.0,
            edgecolor="0.3",
            label=f"{cid}: {get_class_name(cid)}",
        )
        for cid in class_ids
    ]

    n_rows = (len(patches) + ncol - 1) // ncol
    bottom = max(0.08, 0.04 + 0.04 * n_rows)
    fig.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        fontsize=8,
        frameon=False,
    )
    return bottom


def plot_segmentation_row(
    axes,
    image: Union[Image.Image, np.ndarray],
    pred_mask: np.ndarray,
    palette: np.ndarray,
    title: str = "",
    alpha: float = 0.5,
    model_name: Optional[str] = None,
) -> None:
    if model_name is not None:
        display = get_display_image(image, model_name)
    else:
        display = pil_to_numpy(image)
        if display.shape[:2] != pred_mask.shape:
            display = np.array(
                Image.fromarray(display).resize(
                    (pred_mask.shape[1], pred_mask.shape[0]), Image.BILINEAR
                )
            )

    color_mask = colorize_mask(pred_mask, palette)

    axes[0].imshow(display)
    axes[0].set_title(f"{title} — Image")
    axes[0].axis("off")

    axes[1].imshow(display)
    axes[1].imshow(color_mask, alpha=alpha)
    axes[1].set_title(f"{title} — Overlay")
    axes[1].axis("off")

    axes[2].imshow(color_mask)
    axes[2].set_title(f"{title} — Masque")
    axes[2].axis("off")


def plot_gt_row(
    axes,
    image: Union[Image.Image, np.ndarray],
    gt_mask: np.ndarray,
    palette: np.ndarray,
    title: str = "Ground truth",
    alpha: float = 0.5,
    model_name: Optional[str] = None,
) -> None:
    valid = gt_mask >= 0
    display_mask = np.zeros_like(gt_mask, dtype=np.int32)
    display_mask[valid] = gt_mask[valid]
    if model_name is not None:
        display_mask = get_display_mask(display_mask, model_name)
    plot_segmentation_row(
        axes,
        image,
        display_mask,
        palette,
        title=title,
        alpha=alpha,
        model_name=model_name,
    )


def plot_model_comparison(
    image: Union[Image.Image, np.ndarray],
    predictions: dict,
    model_names: Sequence[str],
    palette: Optional[np.ndarray] = None,
    display_names: Optional[dict] = None,
    figsize: tuple = (15, 4),
    alpha: float = 0.5,
    legend_ncol: int = 6,
):
    if palette is None:
        palette = create_ade20k_palette()
    if display_names is None:
        display_names = {name: name for name in model_names}

    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 3, figsize=(figsize[0], figsize[1] * n_models))
    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, name in enumerate(model_names):
        plot_segmentation_row(
            axes[row],
            image,
            predictions[name],
            palette,
            title=display_names.get(name, name),
            alpha=alpha,
            model_name=name,
        )

    class_ids = collect_present_classes(predictions)
    bottom = add_class_legend(fig, class_ids, palette, ncol=legend_ncol)
    plt.tight_layout(rect=[0, bottom, 1, 1])
    return fig


def plot_gt_and_predictions(
    image: Union[Image.Image, np.ndarray],
    gt_mask: Optional[np.ndarray],
    predictions: dict,
    model_names: Sequence[str],
    palette: Optional[np.ndarray] = None,
    display_names: Optional[dict] = None,
    figsize: tuple = (15, 4),
    alpha: float = 0.5,
    legend_ncol: int = 6,
    display_model_name: Optional[str] = None,
):
    if palette is None:
        palette = create_ade20k_palette()
    if display_names is None:
        display_names = {name: name for name in model_names}

    n_rows = len(model_names) + (1 if gt_mask is not None else 0)
    fig, axes = plt.subplots(n_rows, 3, figsize=(figsize[0], figsize[1] * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    row = 0
    align_name = display_model_name or (model_names[0] if model_names else None)
    if gt_mask is not None:
        plot_gt_row(
            axes[row],
            image,
            gt_mask,
            palette,
            title="Ground truth",
            alpha=alpha,
            model_name=align_name,
        )
        row += 1

    for name in model_names:
        plot_segmentation_row(
            axes[row],
            image,
            predictions[name],
            palette,
            title=display_names.get(name, name),
            alpha=alpha,
            model_name=name,
        )
        row += 1

    all_preds = dict(predictions)
    if gt_mask is not None:
        valid = gt_mask >= 0
        gt_display = np.zeros_like(gt_mask, dtype=np.int32)
        gt_display[valid] = gt_mask[valid]
        all_preds["__gt__"] = gt_display
    class_ids = collect_present_classes(all_preds)
    bottom = add_class_legend(fig, class_ids, palette, ncol=legend_ncol)
    plt.tight_layout(rect=[0, bottom, 1, 1])
    return fig
