import torch
from torch import nn
from typing import Tuple

from .two_way_transformer_layer import TwoWayTransformerLayer, MaskAttention
from .layer_norm2d import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        num_heads: int = 8,
        num_multimask_outputs: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.num_heads = num_heads
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, self.embed_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.embed_dim)

        self.iou_prediction_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_mask_tokens),
        )

        self.num_layers = 2
        self.layers = nn.ModuleList(
            [
                TwoWayTransformerLayer(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    skip_first_layer_pe=(i == 0),
                )
                for i in range(self.num_layers)
            ]
        )

        self.final_attn = MaskAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, internal_dim=128
        )
        self.norm_final = nn.LayerNorm(self.embed_dim)

        self.upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim, self.embed_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(self.embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.embed_dim // 4, self.embed_dim // 8, kernel_size=2, stride=2
            ),
            nn.GELU(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Linear(self.embed_dim, self.embed_dim // 8),
                )
                for _ in range(self.num_mask_tokens)
            ]
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        return masks[:, mask_slice], iou_pred[:, mask_slice]

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = image_embeddings.shape[0]

        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(bs, -1, -1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
        query_pe = tokens

        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        for layer in self.layers:
            tokens, image_embeddings = layer(
                queries=tokens,
                keys=image_embeddings,
                query_pe=query_pe,
                key_pe=image_pe,
            )

        q = torch.add(tokens, query_pe).permute(1, 0, 2)
        k = torch.add(image_embeddings, image_pe).permute(1, 0, 2)
        v = image_embeddings.permute(1, 0, 2)

        attn_out, _ = self.final_attn(q, k, v)
        tokens = tokens + attn_out.permute(1, 0, 2)
        tokens = self.norm_final(tokens)

        image_embeddings = image_embeddings.permute(0, 2, 1).reshape(
            bs,
            self.embed_dim,
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        )

        upscaled = self.upscaling(image_embeddings)
        iou_token_out = tokens[:, 0, :]
        mask_tokens_out = tokens[:, 1 : (1 + self.num_mask_tokens), :]

        iou_pred = self.iou_prediction_head(iou_token_out)

        hyper_in_list = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ]
        hypernetworks = torch.stack(hyper_in_list, dim=1)

        feat_map = upscaled.flatten(2).permute(0, 2, 1)
        kernels = hypernetworks.permute(0, 2, 1)
        masks_flattened = (feat_map @ kernels).permute(0, 2, 1)
        masks = masks_flattened.reshape(bs, self.num_mask_tokens, 256, 256)

        return masks, iou_pred
