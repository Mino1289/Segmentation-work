#!/usr/bin/env python3
"""Benchmark ADE20K semantic segmentation models on validation split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.ade20k_dataset import ADE20KDataset
from eval.model_registry import (
    MODEL_CONFIGS,
    SEMANTIC_MODEL_NAMES,
    build_model,
    get_val_transform,
    load_checkpoint,
    resolve_checkpoint_path,
)
from models.util import (
    SegmentationMetric,
    compute_boundary_iou,
    compute_flops,
    get_device,
    inference_context,
    num_parameters_total,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ADE20K segmentation models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SEMANTIC_MODEL_NAMES),
        choices=list(SEMANTIC_MODEL_NAMES),
        help="Models to evaluate",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["validation", "test"],
        help="Dataset split (default: validation)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output",
        default="results/benchmark_ade20k.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Override checkpoint path, e.g. dinov2_linear=weights/foo.pth",
    )
    return parser.parse_args()


def parse_checkpoint_overrides(overrides: list) -> dict:
    paths = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid checkpoint override '{item}'. Use MODEL=PATH.")
        name, path = item.split("=", 1)
        paths[name.strip()] = path.strip()
    return paths


def evaluate_model(model, loader, device, ignore_index: int = -1) -> dict:
    metric = SegmentationMetric(num_classes=150)
    boundary_ious = []

    for images, masks in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        masks = masks.long().to(device)

        with inference_context(device):
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

        metric.update(preds, masks, ignore_index=ignore_index)
        boundary_ious.append(
            compute_boundary_iou(preds, masks, ignore_index=ignore_index)
        )

    pixel_acc, miou = metric.compute()
    boundary_iou = (
        float(sum(boundary_ious) / len(boundary_ious)) if boundary_ious else 0.0
    )

    return {
        "pixel_acc": float(pixel_acc),
        "mIoU": float(miou),
        "boundary_iou": boundary_iou,
    }


def benchmark_model(
    model_name: str,
    loader,
    device: torch.device,
    checkpoint_path: str | None = None,
) -> dict:
    cfg = MODEL_CONFIGS[model_name]
    ckpt = resolve_checkpoint_path(checkpoint_path or cfg.default_checkpoint)

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Checkpoint missing for {model_name}: {ckpt}\n"
            "Train the model or pass --checkpoint {model}=path"
        )

    model = build_model(model_name, pretrained_backbone=True)
    load_checkpoint(model, ckpt)
    model.eval()

    # Profile on CPU: torchprofile is unreliable on XPU/MPS and FLOPs are device-agnostic.
    params = num_parameters_total(model)
    macs = compute_flops(model, (3, cfg.img_size, cfg.img_size))

    model.to(device)
    metrics = evaluate_model(model, loader, device, ignore_index=cfg.ignore_index)

    return {
        "model": model_name,
        "display_name": cfg.display_name,
        "pixel_acc": metrics["pixel_acc"],
        "mIoU": metrics["mIoU"],
        "boundary_iou": metrics["boundary_iou"],
        "params_M": params / 1e6,
        "gflops": macs / 1e9,
        "img_size": cfg.img_size,
        "checkpoint": ckpt,
    }


def export_results(results: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "model",
        "display_name",
        "pixel_acc",
        "mIoU",
        "boundary_iou",
        "params_M",
        "gflops",
        "img_size",
        "checkpoint",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    json_path = os.path.splitext(output_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


def print_table(results: list) -> None:
    header = f"{'Model':<18} {'Pxl Acc':>8} {'mIoU':>8} {'B-IoU':>8} {'Params(M)':>10} {'GFLOPs':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['display_name']:<18} "
            f"{r['pixel_acc']:>8.4f} "
            f"{r['mIoU']:>8.4f} "
            f"{r['boundary_iou']:>8.4f} "
            f"{r['params_M']:>10.2f} "
            f"{r['gflops']:>8.2f}"
        )


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_overrides = parse_checkpoint_overrides(args.checkpoint)

    raw_dataset = load_dataset("merve/scene_parse_150")
    split_data = raw_dataset[args.split]

    results = []
    errors = []

    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        transform = get_val_transform(model_name)
        dataset = ADE20KDataset(split_data, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type in ("cuda", "xpu"),
        )

        print(
            f"\nBenchmarking {cfg.display_name} on {args.split} ({len(dataset)} images)..."
        )
        try:
            result = benchmark_model(
                model_name,
                loader,
                device,
                checkpoint_path=checkpoint_overrides.get(model_name),
            )
            results.append(result)
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")
            errors.append(str(exc))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            errors.append(str(exc))

    if not results:
        print("\nNo models evaluated successfully.")
        if errors:
            for err in errors:
                print(f"  - {err}")
        sys.exit(1)

    print_table(results)
    export_results(results, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
