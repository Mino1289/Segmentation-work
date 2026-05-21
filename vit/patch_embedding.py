import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        emb_size: int = 768,
    ):
        super(PatchEmbedding, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.linear_proj = nn.Linear(self.patch_dim, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Image size must match"

        # patches will be [Batch, Channels, num_patches_H, num_patches_W, patch_H, patch_W]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        # patches will be [B, N, C, patch_H x patch_W]
        patches = (
            patches.contiguous()
            .view(B, C, -1, self.patch_size, self.patch_size)
            # ".patchify()"
            .permute(0, 2, 1, 3, 4)
            .flatten(2)
        )

        x = self.linear_proj(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        return x
