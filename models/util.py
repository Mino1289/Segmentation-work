import torch
from torch import nn
from torch.nn import functional as F
import torchprofile
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import cv2
from contextlib import contextmanager
from typing import Iterator, Optional, Union


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class AdaptiveAvgPool2dSafe(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "mps":
            return F.adaptive_avg_pool2d(x.cpu(), self.output_size).to(x.device)
        return F.adaptive_avg_pool2d(x, self.output_size)


def get_inference_amp_dtype(device: Union[torch.device, str]) -> Optional[torch.dtype]:
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == "cuda":
        return torch.bfloat16
    if device_type == "xpu":
        return torch.float16
    return None


@contextmanager
def inference_context(device: Union[torch.device, str]) -> Iterator[None]:
    """Inference with torch.inference_mode and AMP (bf16 on CUDA, fp16 on XPU)."""
    device_type = device.type if isinstance(device, torch.device) else device
    amp_dtype = get_inference_amp_dtype(device_type)

    with torch.inference_mode():
        if amp_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                yield
        else:
            yield


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def num_parameters_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_flops(model: nn.Module, input_size: tuple) -> int:
    """Return MACs (multiply-accumulate operations) for one forward pass."""
    device = next(model.parameters()).device
    dummy = torch.randn(1, *input_size, device=device)
    return torchprofile.profile_macs(model, dummy)


def unit(v: int):
    units = ["", "k", "M", "G", "T", "P"]
    i = 0
    while v // 1024 > 1:
        i += 1
        v //= 1024
    dec = v / 1024
    return f"{v + dec:.3f}{units[i]}"


def load_state_dict_flexible(model, state_dict):
    """
    Load a state_dict into a model, handling potential '_orig_mod.' prefix
    from torch.compile and removing it if the current model is not compiled.
    """
    new_state_dict = {}
    is_model_compiled = hasattr(model, "_orig_mod") or any(
        k.startswith("_orig_mod.") for k in model.state_dict().keys()
    )

    for k, v in state_dict.items():
        if k.startswith("_orig_mod.") and not is_model_compiled:
            # Remove prefix
            name = k[10:]
        elif not k.startswith("_orig_mod.") and is_model_compiled:
            # Add prefix
            name = f"_orig_mod.{k}"
        else:
            name = k
        new_state_dict[name] = v

    return model.load_state_dict(new_state_dict)


def compute_pixel_accuracy(preds, targets, ignore_index=255):
    """Compute the overall proportion of correctly classified pixels.

    preds:   [B, H, W] or [H, W] (Tensor of predicted class IDs)
    targets: [B, H, W] or [H, W] (Tensor of ground-truth class IDs)
    """
    valid_mask = targets != ignore_index
    correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
    total = valid_mask.sum().item()
    return correct / total if total > 0 else 0.0


class SegmentationMetric:
    """Calcule l'Accuracy globale et le mIoU en accumulant la matrice de confusion."""

    def __init__(self, num_classes: int = 150):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1
    ):
        # Conversion en CPU et aplatissement
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()

        # Filtrage des pixels ignorés et des valeurs aberrantes
        valid_mask = (
            (targets != ignore_index) & (targets >= 0) & (targets < self.num_classes)
        )
        preds = preds[valid_mask]
        targets = targets[valid_mask]

        # Accumulation ultra-rapide via bincount (Matrice de confusion 1D reformée en 2D)
        self.hist += np.bincount(
            self.num_classes * targets + preds, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        # La diagonale contient les vrais positifs (Intersection)
        intersection = np.diag(self.hist)
        # L'union est la somme de la ligne + colonne - l'intersection
        union = self.hist.sum(axis=1) + self.hist.sum(axis=0) - intersection

        # Masque de sécurité pour éviter la division par zéro sur les classes absentes
        valid_classes = union > 0
        iou = np.zeros(self.num_classes)
        iou[valid_classes] = intersection[valid_classes] / union[valid_classes]

        mIoU = np.mean(iou[valid_classes]) if np.any(valid_classes) else 0.0
        pixel_acc = intersection.sum() / (self.hist.sum() + 1e-10)

        return pixel_acc, mIoU

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))


def _get_boundary(mask, dilation_pixels=2):
    """Extract the boundary of a binary mask using morphological operations.

    dilation_pixels: number of pixels to dilate/erode when computing boundary.
    """
    mask_np = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilation_pixels * 2 + 1, dilation_pixels * 2 + 1)
    )

    # Erode to obtain the inner area and subtract to get the boundary
    erosion = cv2.erode(mask_np, kernel, iterations=1)
    boundary = mask_np - erosion
    return boundary


def compute_boundary_iou(
    preds, targets, num_classes=150, dilation_pixels=2, ignore_index=-1
):
    """Compute IoU restricted to object boundaries (sensitive to fine detail).

    preds & targets: Tensors [B, H, W] or [H, W]
    """
    # Convert to CPU NumPy arrays for OpenCV boundary operations
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    valid_mask = targets_np != ignore_index
    preds_np[~valid_mask] = ignore_index

    # If input is 2D [H, W], add a batch dimension [1, H, W] to unify processing
    if preds_np.ndim == 2:
        preds_np = np.expand_dims(preds_np, axis=0)
        targets_np = np.expand_dims(targets_np, axis=0)

    num_images = preds_np.shape[0]
    biou_per_class = []

    for c in range(num_classes):
        total_intersection = 0
        total_union = 0
        class_present = False

        # Loop through each image in the batch
        for i in range(num_images):
            pred_c = preds_np[i] == c
            target_c = targets_np[i] == c

            if not np.any(target_c) and not np.any(pred_c):
                continue

            class_present = True

            # Extract boundary bands for the single 2D slice
            b_pred = _get_boundary(pred_c, dilation_pixels)
            b_target = _get_boundary(target_c, dilation_pixels)

            # Accumulate intersection & union for this class across the batch
            total_intersection += np.logical_and(b_pred, b_target).sum()
            total_union += np.logical_or(b_pred, b_target).sum()

        if not class_present or total_union == 0:
            continue

        biou_per_class.append(total_intersection / total_union)

    return np.mean(biou_per_class) if len(biou_per_class) > 0 else 0.0
