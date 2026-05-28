import torch
from torch import nn
from torch.nn import functional as F
import torchprofile
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import cv2


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model: nn.Module, input_size: tuple) -> int:
    return torchprofile.profile_macs(model, torch.randn(1, *input_size))

def unit(v: int):
    units = ["", "k", "M", "G", "T", "P"]
    i = 0
    while v // 1024 > 1:
        i += 1
        v //= 1024
    dec = v/1024 
    return f"{v+dec:.3f}{units[i]}"


def compute_pixel_accuracy(preds, targets, ignore_index=255):
    """Compute the overall proportion of correctly classified pixels.

    preds:   [B, H, W] or [H, W] (Tensor of predicted class IDs)
    targets: [B, H, W] or [H, W] (Tensor of ground-truth class IDs)
    """
    valid_mask = (targets != ignore_index)
    correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
    total = valid_mask.sum().item()
    return correct / total if total > 0 else 0.0


def compute_dice_score(preds, targets, num_classes=150, ignore_index=None):
    """Compute the mean macro Dice score (F1) across classes.
    """
    dice_per_class = []
    
    for c in range(num_classes):
        if c == ignore_index:
            continue
            
        pred_c = (preds == c)
        target_c = (targets == c)
        
        intersection = (pred_c & target_c).sum().item()
        cardinality = pred_c.sum().item() + target_c.sum().item()
        
        if cardinality == 0:
            # Class absent in both prediction and target: ignore
            continue
            
        dice = (2.0 * intersection) / cardinality
        dice_per_class.append(dice)
        
    return np.mean(dice_per_class) if len(dice_per_class) > 0 else 0.0

def compute_mIoU(preds, targets, num_classes=150, ignore_index=None):
    """Compute mean Intersection over Union (mIoU) across classes.
    """
    iou_per_class = []
    
    for c in range(num_classes):
        if c == ignore_index:
            continue
            
        pred_c = (preds == c)
        target_c = (targets == c)
        
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        
        if union == 0:
            # No ground truth and no prediction for this class
            continue
            
        iou = intersection / union
        iou_per_class.append(iou)
        
    return np.mean(iou_per_class) if len(iou_per_class) > 0 else 0.0

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

def compute_boundary_iou(preds, targets, num_classes=150, dilation_pixels=2):
    """Compute IoU restricted to object boundaries (sensitive to fine detail).

    preds & targets: Tensors [B, H, W] or [H, W]
    """
    # Convert to CPU NumPy arrays for OpenCV boundary operations
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    biou_per_class = []

    # If batch dimension present [B, H, W], iterating over images is safer for contours
    if preds_np.ndim == 3:
        # Simplified approach: either flatten or loop per image. Looping is safer.
        pass

    for c in range(num_classes):
        pred_c = (preds_np == c)
        target_c = (targets_np == c)

        if not np.any(target_c) and not np.any(pred_c):
            continue

        # Extract boundary bands
        b_pred = _get_boundary(pred_c, dilation_pixels)
        b_target = _get_boundary(target_c, dilation_pixels)

        # Intersection & union restricted to the extracted boundaries
        intersection = np.logical_and(b_pred, b_target).sum()
        union = np.logical_or(b_pred, b_target).sum()

        if union == 0:
            continue

        biou_per_class.append(intersection / union)

    return np.mean(biou_per_class) if len(biou_per_class) > 0 else 0.0

def compute_mask_ap(pred_logits, targets, num_classes=150):
    """Compute semantic Average Precision (AP) via area under the precision-recall curve.

    pred_logits: Tensor [B, num_classes, H, W] (raw model outputs before softmax)
    targets:     Tensor [B, H, W] (ground-truth class IDs)
    """
    # Apply softmax to obtain per-class probabilities [B, num_classes, H, W]
    probs = F.softmax(pred_logits, dim=1)

    probs_np = probs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    ap_per_class = []

    for c in range(num_classes):
        target_c = (targets_np == c).astype(int)

        if not np.any(target_c):
            # No ground-truth for this class in the batch
            continue

        # Extract predicted probabilities for class c
        prob_c = probs_np[:, c, :, :].flatten()
        target_c_flat = target_c.flatten()

        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(target_c_flat, prob_c)

        # Area under the curve (AUC) equals AP for this curve
        ap = auc(recall, precision)
        if not np.isnan(ap):
            ap_per_class.append(ap)

    return np.mean(ap_per_class) if len(ap_per_class) > 0 else 0.0
