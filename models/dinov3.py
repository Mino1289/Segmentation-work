import re

import torch
from torch import nn
import timm

from models.layers.dino_block import DinoBlock
from models.layers.patch_embedding import PatchEmbedding
from models.layers.rotary_embedding import RopePositionEmbedding2D


class DinoV3Config:
    def __init__(self, name: str = "S"):
        configs = {
            "S": {"embed_dim": 384, "num_heads": 6, "depth": 12},
            "B": {"embed_dim": 768, "num_heads": 12, "depth": 12},
            "L": {"embed_dim": 1024, "num_heads": 16, "depth": 24},
        }
        if name not in configs:
            raise ValueError(
                f"Taille '{name}' non reconnue. Choisissez parmi 'S', 'B', 'L'."
            )
        self.name = name
        self.embed_dim = configs[name]["embed_dim"]
        self.num_heads = configs[name]["num_heads"]
        self.depth = configs[name]["depth"]


class DinoV3(nn.Module):
    def __init__(
        self,
        config: DinoV3Config = DinoV3Config(name="S"),
        image_size: int = 512,
        patch_size: int = 16,
        num_reg_tokens: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()

        self.config = config

        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.depth = self.config.depth
        self.num_reg_tokens = num_reg_tokens
        self.num_storage_tokens = 1 + num_reg_tokens

        self.rope = RopePositionEmbedding2D(
            embed_dim=self.embed_dim, num_heads=self.num_heads
        )

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, emb_size=self.embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_reg_tokens, self.embed_dim))

        self.blocks = nn.ModuleList(
            [DinoBlock(self.embed_dim, self.num_heads) for _ in range(self.depth)]
        )

        self.layer_norm = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-6)

        if pretrained:
            timm_models = {
                "S": "vit_small_patch16_dinov3",
                "B": "vit_base_patch16_dinov3",
                "L": "vit_large_patch16_dinov3",
            }
            timm_model = timm.create_model(
                timm_models[self.config.name],
                pretrained=True,
                num_classes=0,
            )
            self.load_from_timm(timm_model)

    def forward(self, x: torch.Tensor, return_all_blocks: bool = False) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Patch embedding -> donne maintenant [B, H_p * W_p, embed_dim]
        x = self.patch_embed(x)
        b, h, w, c = x.shape

        x = x.reshape(B, -1, self.embed_dim)  # [B, N, embed_dim] avec N = H_p * W_p

        storage_tokens = torch.cat(
            [
                self.cls_token.expand(B, -1, -1),
                self.reg_token.expand(B, -1, -1),
            ],
            dim=1,
        )
        x = torch.cat([storage_tokens, x], dim=1)

        # 2. Calcul du nombre de patchs en hauteur et largeur
        hp = H // self.patch_embed.proj.kernel_size[0]
        wp = W // self.patch_embed.proj.kernel_size[1]

        # 3. RoPE 2D -> donne sin, cos de forme [hp * wp, head_dim]
        sin, cos = self.rope(hp, wp, device=x.device)

        # Stocker les activations si demandé
        intermediate_outputs = []
        total_blocks = len(self.blocks)
        target_layers = list(range(total_blocks - 4, total_blocks))

        # 4. Blocs de transformer avec RoPE
        for idx, block in enumerate(self.blocks):
            x = block(
                x,
                sin=sin,
                cos=cos,
                num_prefix_tokens=self.num_storage_tokens,
            )

            if return_all_blocks and idx in target_layers:
                intermediate_outputs.append(self.layer_norm(x))

        x = self.layer_norm(x)

        if return_all_blocks:
            return (
                intermediate_outputs  # On retourne directement les 4 derniers blocs !
            )

        # 5. Normalisation finale
        x = x[:, self.num_storage_tokens :, :]

        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x

    def _remap_timm_key(self, key: str) -> str | None:
        if key.startswith("head."):
            return None
        if key.startswith("norm."):
            return key.replace("norm.", "layer_norm.", 1)
        if match := re.match(r"blocks\.(\d+)\.gamma_1$", key):
            return f"blocks.{match.group(1)}.ls1.gamma"
        if match := re.match(r"blocks\.(\d+)\.gamma_2$", key):
            return f"blocks.{match.group(1)}.ls2.gamma"
        return key

    def load_from_timm(self, timm_model: nn.Module):
        state_dict_timm = timm_model.state_dict()
        state_dict_custom = self.state_dict()

        new_state_dict = {}
        loaded_keys = []
        ignored_keys = []
        mismatch_keys = []

        for key_timm, value_timm in state_dict_timm.items():
            key_custom = self._remap_timm_key(key_timm)
            if key_custom is None:
                ignored_keys.append(key_timm)
                continue

            if key_custom in state_dict_custom:
                value_custom = state_dict_custom[key_custom]
                if value_custom.shape == value_timm.shape:
                    new_state_dict[key_custom] = value_timm
                    loaded_keys.append(key_custom)
                else:
                    mismatch_keys.append(
                        f"{key_custom}: custom {value_custom.shape} != timm {value_timm.shape}"
                    )
            else:
                ignored_keys.append(key_timm)

        missing_in_custom, _ = self.load_state_dict(new_state_dict, strict=False)

        timm_has_qkv_bias = any(
            key.endswith("attn.qkv.bias") for key in state_dict_timm
        )
        if not timm_has_qkv_bias:
            with torch.no_grad():
                for block in self.blocks:
                    if block.attn.qkv.bias is not None:
                        block.attn.qkv.bias.zero_()

                custom_state = dict(self.named_parameters())
                for missing in missing_in_custom:
                    if missing.endswith(".bias") and missing in custom_state:
                        custom_state[missing].zero_()

        print(f" {len(loaded_keys)} tenseurs de poids chargés avec succès.")

        if mismatch_keys:
            print(f" {len(mismatch_keys)} erreurs de dimensions (ignorées) :")
            for mismatch in mismatch_keys[:5]:
                print(f"  - {mismatch}")

        if missing_in_custom:
            print(
                f" Clés manquantes dans ton modèle (non chargées) : {len(missing_in_custom)}"
            )
            for missing in missing_in_custom[:5]:
                print(f"  - {missing}")

        return self


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    size = "S"
    image_size = 512
    patch_size = 16
    timm_models = {
        "S": "vit_small_patch16_dinov3",
        "B": "vit_base_patch16_dinov3",
        "L": "vit_large_patch16_dinov3",
    }

    timm_model = (
        timm.create_model(timm_models[size], pretrained=True, num_classes=0)
        .eval()
        .to(device)
    )

    config = DinoV3Config(name=size)
    mymodel = DinoV3(
        config=config, image_size=image_size, patch_size=patch_size, pretrained=False
    )
    mymodel.load_from_timm(timm_model)
    mymodel = mymodel.eval().to(device)

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

        output = mymodel(dummy_input)
        output_timm_raw = timm_model.forward_features(dummy_input)

        batch_size, _, channels = output_timm_raw.shape
        grid_size = image_size // patch_size
        num_storage_tokens = mymodel.num_storage_tokens
        output_timm = output_timm_raw[:, num_storage_tokens:, :]
        output_timm = (
            output_timm.reshape(batch_size, grid_size, grid_size, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        patch_embed_custom = mymodel.patch_embed(dummy_input).reshape(
            batch_size, -1, mymodel.embed_dim
        )
        patch_embed_timm = timm_model.patch_embed(dummy_input).reshape(
            batch_size, -1, channels
        )
        patch_embed_diff = (patch_embed_custom - patch_embed_timm).abs().max().item()

        hp = wp = grid_size
        sin, cos = mymodel.rope(hp, wp, device=device)
        storage_tokens = torch.cat(
            [
                mymodel.cls_token.expand(batch_size, -1, -1),
                mymodel.reg_token.expand(batch_size, -1, -1),
            ],
            dim=1,
        )
        tokens_custom = torch.cat([storage_tokens, patch_embed_custom], dim=1)
        block0_custom = mymodel.blocks[0](
            tokens_custom,
            sin=sin,
            cos=cos,
            num_prefix_tokens=num_storage_tokens,
        )

        tokens_timm, rope = timm_model._pos_embed(timm_model.patch_embed(dummy_input))
        block0_timm = timm_model.blocks[0](tokens_timm, rope=rope)[
            :, num_storage_tokens:, :
        ]
        block0_diff = (
            (block0_custom[:, num_storage_tokens:, :] - block0_timm).abs().max().item()
        )

    print("MyModel Output shape :", output.shape)
    print("Timm Output shape    :", output_timm.shape)
    print("Patch embed max diff :", patch_embed_diff)
    print("Block 0 max diff     :", block0_diff)

    max_diff = (output - output_timm).abs().max().item()
    print("Max absolute diff    :", max_diff)
