import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors
from tqdm import tqdm
import numpy as np
import os
import csv
from datasets import load_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

from models.upernet import Backbone, UPerNet
from dataset.ade20k_dataset import ADE20KDataset
from models.util import (
    get_device,
    num_parameters,
    compute_flops,
    unit,
    load_state_dict_flexible,
    compute_pixel_accuracy,
    compute_mIoU,
    compute_boundary_iou,
)


class CombinedLoss(nn.Module):
    def __init__(self, ignore_index=-1, gamma=2.0, dice_weight=0.5, ce_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        """
        inputs: (B, C, H, W) - Unnormalized model logits
        targets: (B, H, W) - Ground truth (0 to 149), and -1 for ignored pixels
        """
        # --- 1. Optimized Focal / CrossEntropy Loss ---
        ce_loss = F.cross_entropy(
            inputs, targets, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce_loss)  # Probability of the correct class
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        # --- 2. Dynamically Filtered Multi-Class Dice Loss ---
        num_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)

        mask = (targets != self.ignore_index).unsqueeze(1)  # (B, 1, H, W)
        probs = probs * mask

        clean_targets = torch.clamp(targets, min=0)

        unique_classes = torch.unique(clean_targets)
        unique_classes = unique_classes[unique_classes != self.ignore_index]

        if len(unique_classes) > 0:
            probs_present = probs[:, unique_classes, ...]

            target_one_hot = F.one_hot(clean_targets, num_classes=num_classes)
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).to(dtype=probs.dtype)
            target_one_hot = target_one_hot[:, unique_classes, ...] * mask

            num_active_classes = len(unique_classes)
            probs_flat = probs_present.permute(1, 0, 2, 3).reshape(
                num_active_classes, -1
            )
            target_flat = target_one_hot.permute(1, 0, 2, 3).reshape(
                num_active_classes, -1
            )

            intersection = torch.sum(probs_flat * target_flat, dim=1)
            cardinality = torch.sum(probs_flat + target_flat, dim=1)

            dice_coef = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
            dice_loss = 1.0 - dice_coef.mean()
        else:
            dice_loss = 0.0

        # --- 3. Final Combination ---
        total_loss = (self.ce_weight * focal_loss) + (self.dice_weight * dice_loss)
        return total_loss


def plot_metrics(history, log_dir):
    epochs = [h["epoch"] for h in history]

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h["train_acc"] for h in history], label="Train Acc")
    val_epochs = [h["epoch"] for h in history if h["val_acc"] is not None]
    if val_epochs:
        plt.plot(
            val_epochs,
            [h["val_acc"] for h in history if h["val_acc"] is not None],
            label="Val Acc",
            marker="o",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "metrics_accuracy.png"))
    plt.close("all")

    # Plot mIoU
    if val_epochs:
        plt.figure(figsize=(6, 5))
        plt.plot(
            val_epochs,
            [h["val_mIoU"] for h in history if h["val_mIoU"] is not None],
            label="Val mIoU",
            marker="o",
            color="green",
        )
        plt.plot(
            val_epochs,
            [h["val_biou"] for h in history if h["val_biou"] is not None],
            label="Val Bnd IoU",
            marker="x",
            color="red",
        )
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("Validation IoU Metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "metrics_iou.png"))
        plt.close("all")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    raw_dataset = load_dataset("merve/scene_parse_150")

    SCALE_RANGE = (0.5, 2.0)
    train_transforms = v2.Compose(
        [
            v2.RandomResize(
                min_size=int(512 * SCALE_RANGE[0]),
                max_size=int(512 * SCALE_RANGE[1]),
                antialias=True,
            ),
            v2.RandomCrop(
                size=(512, 512),
                pad_if_needed=True,
                fill={tv_tensors.Image: 0, tv_tensors.Mask: -1},
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomPhotometricDistort(
                brightness=(0.5, 1.5),
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=(-0.1, 0.1),
                p=1.0,
            ),
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.5
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transforms = v2.Compose(
        [
            v2.Resize(size=512, antialias=True),
            v2.CenterCrop(size=(512, 512)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ADE20KDataset(raw_dataset["train"], transform=train_transforms)
    val_dataset = ADE20KDataset(
        raw_dataset["validation"], transform=val_test_transforms
    )
    test_dataset = ADE20KDataset(raw_dataset["test"], transform=val_test_transforms)

    BATCH_SIZE = 16
    NUM_WORKERS = 8
    PIN_MEMORY = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Use a batch size of 1 for testing/benchmarking
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    BACKBONE_LR = 1e-5
    UPERNET_LR = 2e-4
    WEIGHT_DECAY = 0.05

    NUMS_EPOCHS = 128

    weight_dir = "weights"
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = os.path.join(weight_dir, "upernet_convnextv2_base.pth")
    best_mIoU = 0.0

    device = get_device()
    print(f"Using device: {device}")
    backbone = Backbone(model_size="base", pretrained=True, drop_path_rate=0.4)
    model = UPerNet(backbone, channels=512, num_classes=150)
    model.eval()
    print(f"# parameters: {unit(num_parameters(model))}")
    print(
        f"# FLOPs @ 512x512: {unit(compute_flops(model, input_size=(3, 512, 512)))}\n"
    )

    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    loss_fn = CombinedLoss(ignore_index=-1)
    params = [
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.fpn.parameters(), "lr": UPERNET_LR},
        {"params": model.head.parameters(), "lr": UPERNET_LR},
    ]
    optimizer = AdamW(params, weight_decay=WEIGHT_DECAY, fused=True)

    warmup_epochs = 3
    scheduler1 = lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    scheduler2 = lr_scheduler.PolynomialLR(
        optimizer, total_iters=NUMS_EPOCHS - warmup_epochs, power=0.9
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs]
    )

    scaler = torch.amp.GradScaler(device.type)

    if device.type == "cuda":
        model = torch.compile(model)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "metrics.csv")
    history = []
    start_epoch = 0

    if os.path.exists(weight_path):
        print("Found saved model and optim !")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

        load_state_dict_flexible(model, checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_mIoU = checkpoint["best_mIoU"]
        print(
            f"Resume training at epoch {start_epoch + 1}. (best mIoU: {best_mIoU:.4f})"
        )

        for _ in range(start_epoch):
            scheduler.step()

        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                reader = csv.DictReader(f)
                history = list(reader)
                # Convert CSV strings back to appropriate types when needed
                for h in history:
                    h["epoch"] = int(h["epoch"])
                    h["train_loss"] = float(h["train_loss"])
                    h["train_acc"] = float(h["train_acc"])
                    h["val_acc"] = float(h["val_acc"]) if h["val_acc"] else None
                    h["val_mIoU"] = float(h["val_mIoU"]) if h["val_mIoU"] else None
                    h["val_biou"] = float(h["val_biou"]) if h["val_biou"] else None

    for epoch in range(start_epoch, NUMS_EPOCHS):
        # Training loop
        model.train()

        epoch_loss = 0.0
        pxl_acc = []

        optimizer.zero_grad(set_to_none=True)

        with tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{NUMS_EPOCHS}", unit="batch"
        ) as tepoch:
            for batch, (images, masks) in enumerate(tepoch):
                images = images.to(device)
                masks = masks.to(device)

                images = images.to(memory_format=torch.channels_last)
                masks = masks.long()

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                ):
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print("This batch contains NaN, ignoring it.", file=sys.stderr)
                        continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                loss_val = loss.item()
                epoch_loss += loss_val

                preds = torch.argmax(outputs, dim=1)
                pxl_acc.append(compute_pixel_accuracy(preds, masks, ignore_index=-1))

                tepoch.set_postfix(
                    loss=loss_val,
                    avg_loss=epoch_loss / (batch + 1),
                    pxl_acc=np.mean(pxl_acc),
                )
            scheduler.step()

        # Metrics for this epoch
        metrics = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_loader),
            "train_acc": np.mean(pxl_acc),
            "val_acc": None,
            "val_mIoU": None,
            "val_biou": None,
        }

        # Run validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()

            total_mIoU = []
            total_acc = []
            total_biou = []

            with torch.no_grad():
                with tqdm(val_loader, desc="Validation", unit="batch") as tval:
                    for images, masks in tval:
                        # Move data to GPU if available
                        images = images.to(device)
                        masks = masks.to(device)

                        # ADE20K labels are shifted in the dataset: background 0 -> ignore, classes 1-150 -> 0-149
                        masks = masks.long()

                        outputs = model(images)

                        preds = torch.argmax(outputs, dim=1)

                        total_acc.append(
                            compute_pixel_accuracy(preds, masks, ignore_index=-1)
                        )
                        total_mIoU.append(
                            compute_mIoU(preds, masks, num_classes=150, ignore_index=-1)
                        )
                        total_biou.append(
                            compute_boundary_iou(
                                preds, masks, num_classes=150, ignore_index=-1
                            )
                        )
                        tval.set_postfix(
                            {
                                "Pxl Acc": f"{np.mean(total_acc):.3f}",
                                "mIoU": f"{np.mean(total_mIoU):.3f}",
                                "Bnd IoU": f"{np.mean(total_biou):.3f}",
                            }
                        )

            avg_acc = np.mean(total_acc)
            avg_mIoU = np.mean(total_mIoU)
            avg_biou = np.mean(total_biou)

            metrics["val_acc"] = avg_acc
            metrics["val_mIoU"] = avg_mIoU
            metrics["val_biou"] = avg_biou

            if avg_mIoU > best_mIoU:
                best_mIoU = avg_mIoU

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_mIoU": best_mIoU,
                    },
                    weight_path,
                )

        history.append(metrics)

        # Save to CSV
        keys = history[0].keys()
        with open(log_file, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(history)

        # Plot metrics
        plot_metrics(history, log_dir)

    print("Training complete!")
