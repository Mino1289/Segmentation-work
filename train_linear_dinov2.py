import torch
from torch.optim import AdamW, lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors
from tqdm import tqdm
import numpy as np
import os
import csv
from datasets import load_dataset
import matplotlib

from models.dino_feature_extractor import DinoLinearADE20K, DinoV2FeatureExtractor
from models.dinov2 import DinoV2, DinoV2Config

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

from dataset.ade20k_dataset import ADE20KDataset
from models.util import (
    SegmentationMetric,
    get_device,
    num_parameters,
    compute_flops,
    unit,
    load_state_dict_flexible,
    compute_pixel_accuracy,
)


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
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("Validation IoU Metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "metrics_iou.png"))
        plt.close("all")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    raw_dataset = load_dataset("merve/scene_parse_150")

    IMG_SIZE = 518

    SCALE_RANGE = (0.5, 2.0)
    train_transforms = v2.Compose(
        [
            v2.RandomResize(
                min_size=int(IMG_SIZE * SCALE_RANGE[0]),
                max_size=int(IMG_SIZE * SCALE_RANGE[1]),
                antialias=True,
            ),
            v2.RandomCrop(
                size=(IMG_SIZE, IMG_SIZE),
                pad_if_needed=True,
                fill={tv_tensors.Image: 0, tv_tensors.Mask: -1},
            ),
            v2.RandomHorizontalFlip(p=0.5),
            # On baisse la probabilité à 0.5 et on adoucit les amplitudes pour ne pas déboussoler le ViT
            v2.RandomPhotometricDistort(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=(-0.05, 0.05),
                p=0.5,
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_test_transforms = v2.Compose(
        [
            v2.Resize(size=IMG_SIZE, antialias=True),
            v2.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = ADE20KDataset(raw_dataset["train"], transform=train_transforms)
    val_dataset = ADE20KDataset(
        raw_dataset["validation"], transform=val_test_transforms
    )
    test_dataset = ADE20KDataset(raw_dataset["test"], transform=val_test_transforms)

    BATCH_SIZE = 32
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

    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    NUMS_EPOCHS = 60

    weight_dir = "weights"
    os.makedirs(weight_dir, exist_ok=True)
    weight_path = os.path.join(weight_dir, "linear_dinov2_base.pth")
    best_mIoU = 0.0

    device = get_device()
    print(f"Using device: {device}")

    config = DinoV2Config("B")  # embed_dim = 768
    dino = DinoV2(config=config, pretrained=True)
    # freeze all dino parameters
    for param in dino.parameters():
        param.requires_grad = False

    feature_extractor = DinoV2FeatureExtractor(dino)
    model = DinoLinearADE20K(backbone=feature_extractor, num_classes=150)

    model.eval()

    print(f"# parameters: {unit(num_parameters(model))}")
    print(
        f"# FLOPs @ {IMG_SIZE}x{IMG_SIZE}: {unit(compute_flops(model, input_size=(3, IMG_SIZE, IMG_SIZE)))}\n"
    )

    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    loss_fn = CrossEntropyLoss(ignore_index=-1)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    warmup_epochs = 3
    scheduler1 = lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUMS_EPOCHS - warmup_epochs, eta_min=1e-6
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs]
    )

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

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

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
        }

        # Run validation every 5 epochs, and after 50% of the epochs to track final convergence
        if (epoch + 1) % 5 == 0 or epoch >= NUMS_EPOCHS * 0.5:
            model.eval()

            metric = SegmentationMetric(num_classes=150)

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.long().to(device)

                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)

                    metric.update(preds, masks, ignore_index=-1)

            avg_acc, avg_mIoU = metric.compute()

            metrics["val_acc"] = avg_acc
            metrics["val_mIoU"] = avg_mIoU
            print(f"Validation => Pxl Acc: {avg_acc:.4f} | mIoU: {avg_mIoU:.4f}")

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
