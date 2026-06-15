import torch
from torch import nn
from torch.nn import functional as F

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from models.layers.image_encoder import ImageEncoder
from models.layers.prompt_encoder import PromptEncoder
from models.layers.mask_decoder import MaskDecoder


class SAMConfig:
    def __init__(
        self,
        encoder_size: str = "huge",
        img_size: int = 1024,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        mask_threshold: float = 0.0,
    ):
        self.encoder_size = encoder_size
        self.img_size = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.mask_threshold = mask_threshold


class SAM(nn.Module):
    mask_threshold: float = 0.0

    def __init__(
        self,
        config: SAMConfig = SAMConfig(),
    ):
        super().__init__()
        self.config = config
        self.encoder_size = config.encoder_size
        self.img_size = config.img_size
        self.mask_threshold = config.mask_threshold

        self.image_encoder = ImageEncoder(model_size=self.encoder_size)
        self.prompt_encoder = PromptEncoder()
        self.mask_decoder = MaskDecoder()

        self.register_buffer(
            "pixel_mean", torch.Tensor(self.config.pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(self.config.pixel_std).view(-1, 1, 1), False
        )

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device

    @torch.no_grad()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        return F.pad(x, (0, padw, 0, padh))

    # transforme les logits de sortie du décodeur en masques alignés sur l'image d'entrée
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]  # enlever le padding
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )
        image_embeddings = self.image_encoder(images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record and "point_labels" in image_record:
                points = (
                    image_record["point_coords"],
                    image_record["point_labels"],
                )
            else:
                points = None

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            low_res_masks, iou_pred = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            input_size = image_record["image"].shape[-2:]
            original_size = image_record.get("original_size", input_size)
            masks = self.postprocess_masks(low_res_masks, input_size, original_size)

            outputs.append(
                {
                    "masks": masks > self.mask_threshold,
                    "iou_predictions": iou_pred,
                    "low_res_logits": low_res_masks,
                }
            )

        return outputs

    def load_official_weights(
        self,
        checkpoint: Union[str, Path, Dict[str, torch.Tensor]],
    ) -> "SAM":
        """Load Meta SAM ViT-H (or compatible) checkpoint into this model."""
        if isinstance(checkpoint, (str, Path)):
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        else:
            state_dict = checkpoint

        new_state_dict = {}

        def map_key(src: str, dst: str) -> None:
            if src in state_dict:
                new_state_dict[dst] = state_dict[src]

        # 1. Correspondances directes et simples (sans suffixes spécifiques)
        direct_mappings = {
            "image_encoder.patch_embed.proj.weight": "image_encoder.sam_vit.patch_embed.proj.weight",
            "image_encoder.patch_embed.proj.bias": "image_encoder.sam_vit.patch_embed.proj.bias",
            "image_encoder.pos_embed": "image_encoder.sam_vit.pos_embed",
            "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix": "prompt_encoder.pe_layer.gauss_matrix",
            "prompt_encoder.not_a_point_embed.weight": "prompt_encoder.not_a_point_embed.weight",
            "prompt_encoder.no_mask_embed.weight": "prompt_encoder.no_mask_embed.weight",
            "mask_decoder.iou_token.weight": "mask_decoder.iou_token.weight",
            "mask_decoder.mask_tokens.weight": "mask_decoder.mask_tokens.weight",
            "mask_decoder.transformer.norm_final_attn.weight": "mask_decoder.norm_final.weight",
            "mask_decoder.transformer.norm_final_attn.bias": "mask_decoder.norm_final.bias",
        }
        for src, dst in direct_mappings.items():
            map_key(src, dst)

        # Suffixes standards (weight & bias) pour factoriser les modules linéaires/convs
        w_b = ["weight", "bias"]

        # 2. Image Encoder Neck & Blocks
        for i in range(4):
            for suf in w_b:
                map_key(
                    f"image_encoder.neck.{i}.{suf}", f"image_encoder.convs.{i}.{suf}"
                )

        for i in range(32):
            src_p, dst_p = (
                f"image_encoder.blocks.{i}",
                f"image_encoder.sam_vit.encoder.{i}",
            )
            for suf in w_b:
                map_key(f"{src_p}.norm1.{suf}", f"{dst_p}.norm1.{suf}")
                map_key(f"{src_p}.norm2.{suf}", f"{dst_p}.norm2.{suf}")
                map_key(f"{src_p}.attn.qkv.{suf}", f"{dst_p}.attn.qkv.{suf}")
                map_key(f"{src_p}.attn.proj.{suf}", f"{dst_p}.attn.proj.{suf}")
                map_key(f"{src_p}.mlp.lin1.{suf}", f"{dst_p}.mlp.0.{suf}")
                map_key(f"{src_p}.mlp.lin2.{suf}", f"{dst_p}.mlp.2.{suf}")
            map_key(f"{src_p}.attn.rel_pos_h", f"{dst_p}.attn.rel_pos_h")
            map_key(f"{src_p}.attn.rel_pos_w", f"{dst_p}.attn.rel_pos_w")

        # 3. Prompt Encoder (Points & Masks)
        pts = [
            state_dict[k]
            for i in range(4)
            if (k := f"prompt_encoder.point_embeddings.{i}.weight") in state_dict
        ]
        if len(pts) == 4:
            new_state_dict["prompt_encoder.point_embeddings.weight"] = torch.cat(
                pts, dim=0
            )

        for i in [0, 1, 3, 4, 6]:
            for suf in w_b:
                map_key(
                    f"prompt_encoder.mask_downscaling.{i}.{suf}",
                    f"prompt_encoder.mask_downscaling.{i}.{suf}",
                )

        # 4. Mask Decoder Heads & Upscaling
        # Predictor Head & Hypernetworks MLPs (Même pattern de mapping de couches: 0->0, 1->2, 2->4)
        layer_map = {0: 0, 1: 2, 2: 4}
        for src_l, dst_l in layer_map.items():
            for suf in w_b:
                map_key(
                    f"mask_decoder.iou_prediction_head.layers.{src_l}.{suf}",
                    f"mask_decoder.iou_prediction_head.{dst_l}.{suf}",
                )
                for i in range(4):
                    map_key(
                        f"mask_decoder.output_hypernetworks_mlps.{i}.layers.{src_l}.{suf}",
                        f"mask_decoder.output_hypernetworks_mlps.{i}.{dst_l}.{suf}",
                    )

        # Upscaling (0->0, 1->1, 3->3)
        for i in [0, 1, 3]:
            for suf in w_b:
                map_key(
                    f"mask_decoder.output_upscaling.{i}.{suf}",
                    f"mask_decoder.upscaling.{i}.{suf}",
                )

        # 5. Mask Decoder Transformer Layers & Final Attention
        projs = ["q_proj", "k_proj", "v_proj", "out_proj"]
        for i in range(2):
            src_p, dst_p = (
                f"mask_decoder.transformer.layers.{i}",
                f"mask_decoder.layers.{i}",
            )
            for suf in w_b:
                for n in ["1", "2", "3", "4"]:
                    map_key(f"{src_p}.norm{n}.{suf}", f"{dst_p}.norm{n}.{suf}")
                map_key(f"{src_p}.mlp.lin1.{suf}", f"{dst_p}.mlp.0.{suf}")
                map_key(f"{src_p}.mlp.lin2.{suf}", f"{dst_p}.mlp.2.{suf}")

            for proj in projs:
                for suf in w_b:
                    map_key(
                        f"{src_p}.self_attn.{proj}.{suf}",
                        f"{dst_p}.self_attn.{proj}.{suf}",
                    )
                    map_key(
                        f"{src_p}.cross_attn_token_to_image.{proj}.{suf}",
                        f"{dst_p}.cross_attn_tok_to_img.{proj}.{suf}",
                    )
                    map_key(
                        f"{src_p}.cross_attn_image_to_token.{proj}.{suf}",
                        f"{dst_p}.cross_attn_img_to_toks.{proj}.{suf}",
                    )

        # Final Attention
        for proj in projs:
            for suf in w_b:
                map_key(
                    f"mask_decoder.transformer.final_attn_token_to_image.{proj}.{suf}",
                    f"mask_decoder.final_attn.{proj}.{suf}",
                )

        self.load_state_dict(new_state_dict, strict=False)
        return self


if __name__ == "__main__":
    model = SAM()
    model.to("xpu")
    model.eval()
    dummy_input = [
        {
            "image": torch.randn(3, 1024, 1024),
            "point_coords": torch.tensor([[[100.0, 150.0], [200.0, 250.0]]]),
            "point_labels": torch.tensor([[1, 0]]),
            "original_size": (1024, 1024),
        }
    ]
    for elem in dummy_input:
        for key in elem:
            if isinstance(elem[key], torch.Tensor):
                elem[key] = elem[key].to("xpu")
    with torch.no_grad():
        output = model(dummy_input)
    for elem in output:
        print(
            f"Masks shape: {elem['masks'].shape}, "
            f"IoU pred shape: {elem['iou_predictions'].shape}"
        )
