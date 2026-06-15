"""SAM inference wrapper for the Gradio demo."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from eval.model_registry import resolve_checkpoint_path
from models.sam import SAM
from models.util import get_device, inference_context

DEFAULT_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"

_MASK_COLORS = np.array(
    [
        [255, 99, 71],
        [65, 105, 225],
        [50, 205, 50],
    ],
    dtype=np.uint8,
)


class SAMPredictor:
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = SAM()
        checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}\n"
                "Download sam_vit_h_4b8939.pth from Meta and place it in weights/"
            )
        self.model.load_official_weights(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)

    def predict(
        self,
        image: Image.Image,
        points: List[Tuple[float, float]],
        labels: List[int],
        multimask_output: bool = True,
    ) -> dict:
        if not points:
            raise ValueError("At least one point prompt is required.")

        tensor = self._pil_to_tensor(image).to(self.device)
        h, w = tensor.shape[-2:]

        point_coords = torch.tensor([points], dtype=torch.float32, device=self.device)
        point_labels = torch.tensor([labels], dtype=torch.int64, device=self.device)

        batched_input = [
            {
                "image": tensor,
                "point_coords": point_coords,
                "point_labels": point_labels,
                "original_size": (h, w),
            }
        ]

        with inference_context(self.device):
            outputs = self.model(batched_input, multimask_output=multimask_output)[0]
        masks = outputs["masks"].cpu().numpy()  # [N, 1, H, W]
        iou_preds = outputs["iou_predictions"].cpu().numpy()[0]

        best_idx = int(np.argmax(iou_preds))
        best_mask = masks[best_idx, 0].astype(bool)

        return {
            "best_mask": best_mask,
            "best_iou": float(iou_preds[best_idx]),
            "all_masks": [masks[i, 0].astype(bool) for i in range(len(iou_preds))],
            "all_ious": [float(v) for v in iou_preds],
        }


def mask_to_color_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 99, 71),
    alpha: float = 0.5,
) -> np.ndarray:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def mask_to_binary_image(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = 255
    return out


def render_candidate_gallery(
    image: Image.Image,
    masks: List[np.ndarray],
    ious: List[float],
) -> List[Tuple[np.ndarray, str]]:
    gallery = []
    for i, (mask, iou) in enumerate(zip(masks, ious)):
        color = _MASK_COLORS[i % len(_MASK_COLORS)]
        overlay = mask_to_color_overlay(image, mask, color=tuple(color))
        gallery.append((overlay, f"Masque {i + 1} — IoU prédit: {iou:.3f}"))
    return gallery
