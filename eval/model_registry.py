"""Model registry for ADE20K semantic segmentation evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torchvision.transforms import v2

from models.dino_feature_extractor import (
    DinoLinearADE20K,
    DinoV2FeatureExtractor,
    DinoV3FeatureExtractor,
)
from models.dinov2 import DinoV2, DinoV2Config
from models.dinov3 import DinoV3, DinoV3Config
from models.upernet import Backbone, UPerNet
from models.util import get_device, load_state_dict_flexible, inference_context

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEMANTIC_MODEL_NAMES = ("dinov2_linear", "dinov3_linear", "upernet")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_checkpoint_path(path: str) -> str:
    """Resolve checkpoint path relative to project root (works from notebooks/)."""
    p = Path(path)
    if p.is_file():
        return str(p.resolve())
    project_path = PROJECT_ROOT / p
    if project_path.is_file():
        return str(project_path)
    return str(project_path)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    display_name: str
    img_size: int
    num_classes: int = 150
    ignore_index: int = -1
    default_checkpoint: str = ""


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "dinov2_linear": ModelConfig(
        name="dinov2_linear",
        display_name="DINOv2 linear",
        img_size=518,
        default_checkpoint="weights/linear_dinov2_base.pth",
    ),
    "dinov3_linear": ModelConfig(
        name="dinov3_linear",
        display_name="DINOv3 linear",
        img_size=512,
        default_checkpoint="weights/linear_dinov3_base.pth",
    ),
    "upernet": ModelConfig(
        name="upernet",
        display_name="ConvNeXtV2 + UPerNet",
        img_size=512,
        default_checkpoint="weights/upernet_convnextv2_base.pth",
    ),
}


def get_val_transform(model_name: str) -> v2.Compose:
    cfg = MODEL_CONFIGS[model_name]
    return v2.Compose(
        [
            v2.Resize(size=cfg.img_size, antialias=True),
            v2.CenterCrop(size=(cfg.img_size, cfg.img_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_inference_transform(model_name: str) -> v2.Compose:
    """Transform for PIL images (notebooks / single-image inference)."""
    cfg = MODEL_CONFIGS[model_name]
    return v2.Compose(
        [
            v2.Resize(size=cfg.img_size, antialias=True),
            v2.CenterCrop(size=(cfg.img_size, cfg.img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_display_transform(model_name: str) -> v2.Compose:
    """Resize/crop for display without normalization."""
    cfg = MODEL_CONFIGS[model_name]
    return v2.Compose(
        [
            v2.Resize(size=cfg.img_size, antialias=True),
            v2.CenterCrop(size=(cfg.img_size, cfg.img_size)),
            v2.ToImage(),
        ]
    )


def build_model(model_name: str, pretrained_backbone: bool = True) -> nn.Module:
    if model_name == "dinov2_linear":
        config = DinoV2Config("B")
        dino = DinoV2(config=config, pretrained=pretrained_backbone)
        for param in dino.parameters():
            param.requires_grad = False
        feature_extractor = DinoV2FeatureExtractor(dino)
        return DinoLinearADE20K(backbone=feature_extractor, num_classes=150)

    if model_name == "dinov3_linear":
        config = DinoV3Config("B")
        dino = DinoV3(config=config, pretrained=pretrained_backbone)
        for param in dino.parameters():
            param.requires_grad = False
        feature_extractor = DinoV3FeatureExtractor(dino)
        return DinoLinearADE20K(backbone=feature_extractor, num_classes=150)

    if model_name == "upernet":
        backbone = Backbone(model_size="base", pretrained=pretrained_backbone, drop_path_rate=0.4)
        return UPerNet(backbone, channels=512, num_classes=150)

    raise ValueError(
        f"Unknown model '{model_name}'. Choose from: {list(MODEL_CONFIGS.keys())}"
    )


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict_flexible(model, state_dict)


def load_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    pretrained_backbone: bool = True,
) -> nn.Module:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    if device is None:
        device = get_device()

    model = build_model(model_name, pretrained_backbone=pretrained_backbone)
    ckpt = checkpoint_path or MODEL_CONFIGS[model_name].default_checkpoint
    load_checkpoint(model, ckpt)
    model.to(device)
    model.eval()
    return model


def load_all_models(
    model_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    checkpoint_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, nn.Module]:
    names = model_names or list(MODEL_CONFIGS.keys())
    paths = checkpoint_paths or {}
    models: Dict[str, nn.Module] = {}
    for name in names:
        models[name] = load_model(
            name,
            checkpoint_path=paths.get(name),
            device=device,
        )
    return models


def predict_mask(
    model: nn.Module,
    image,
    model_name: str,
    device: Optional[torch.device] = None,
) -> "np.ndarray":
    """Run inference on a PIL image and return class indices [H, W]."""
    from PIL import Image

    if device is None:
        device = next(model.parameters()).device

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    transform = get_inference_transform(model_name)
    tensor = transform(image).unsqueeze(0).to(device)
    with inference_context(device):
        logits = model(tensor)
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
