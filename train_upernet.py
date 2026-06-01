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
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset

from models.upernet import Backbone, UPerNet
from dataset.ade20k_dataset import ADE20KDataset
from models.util import (
    get_device,
    num_parameters,
    compute_flops,
    unit,
    compute_pixel_accuracy,
    compute_mIoU,
    compute_boundary_iou,
)


def plot_metrics(history, log_dir):
    epochs = [h["epoch"] for h in history]

    # Plotting Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plotting Accuracy
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
    plt.close()

    # Plotting mIoU
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
        plt.close()


if __name__ == "__main__":
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
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

    BATCH_SIZE = (
        16  # TODO: switch to accumulated gradients for larger batch size if OOM
    )
    NUM_WORKERS = 4  # Multi-processing CPU
    PIN_MEMORY = True  # might pressure VRAM but can speed up GPU data transfer

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

    # for testing/benchmarking, batch size of 1
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    BACKBONE_LR = 2e-5
    UPERNET_LR = 2e-4
    WEIGHT_DECAY = 0.05

    NUMS_EPOCHS = 128

    weight_path = "weights/upernet_convnextv2_base.pth"
    best_mIoU = 0.0

    device = get_device()
    print(f"Using device: {device}")
    backbone = Backbone(model_size="base", pretrained=True)
    model = UPerNet(backbone, channels=512, num_classes=150)
    model.eval()
    print(f"# parameters: {unit(num_parameters(model))}")
    print(
        f"# FLOPs @ 512x512: {unit(compute_flops(model, input_size=(3, 512, 512)))}\n"
    )

    model.to(device)

    loss_fn = CrossEntropyLoss(ignore_index=-1)
    params = [
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.fpn.parameters(), "lr": UPERNET_LR},
        {"params": model.head.parameters(), "lr": UPERNET_LR},
    ]
    optimizer = AdamW(params, weight_decay=0.05, fused=True)
    scheduler = lr_scheduler.PolynomialLR(optimizer, total_iters=NUMS_EPOCHS, power=0.9)

    if device.type == "cuda":
        model = torch.compile(model)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "metrics.csv")
    history = []

    for epoch in range(NUMS_EPOCHS):
        # training loop
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

                # ADE20K background is 0, classes are 1-150. We shift them to 0-149 and ignore background.
                masks = masks.long() - 1

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                ):
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)

                loss.backward()
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
            "val_biou": None,
        }

        # testing loop every 5 epochs
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

                        # Shift ADE20K labels: background 0 -> ignore, classes 1-150 -> 0-149
                        masks = masks - 1

                        outputs = model(images)

                        preds = torch.argmax(outputs, dim=1)

                        total_acc.append(
                            compute_pixel_accuracy(preds, masks, ignore_index=-1)
                        )
                        total_mIoU.append(
                            compute_mIoU(preds, masks, num_classes=150, ignore_index=-1)
                        )
                        total_biou.append(
                            compute_boundary_iou(preds, masks, num_classes=150)
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

    # end of training loop, best model is saved at weight_path
    # test it
    model.load_state_dict(torch.load(weight_path)["model_state_dict"])

    model.eval()

    print("\n====== GENERATING PREDICTIONS ON THE TEST SET ======")
    # The test set has no ground-truth masks, so we cannot compute metrics.
    # Save predictions instead for visualization or submission.
    os.makedirs("test_predictions", exist_ok=True)

    with torch.no_grad():
        with tqdm(test_loader, desc="Test Inference", unit="batch") as ttest:
            for i, (images, _) in enumerate(ttest):
                # Move input to GPU if available
                images = images.to(device)

                # Forward pass
                outputs = model(images)

                # Get predictions (argmax) as a NumPy array
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                # Save predicted image (batch size = 1 for test_loader)
                pred_img = Image.fromarray(preds[0].astype(np.uint8))
                pred_img.save(f"test_predictions/pred_test_{i:04d}.png")

    print("\nPredictions complete!")
    print("Predicted masks saved to the 'test_predictions/' folder.")
    print(f"Best validation mIoU: {best_mIoU:.4f}")
