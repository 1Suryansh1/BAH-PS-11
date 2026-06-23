import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )

    def forward(self, q, c, key_padding_mask=None):
        q_norm = self.norm1(q)
        attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        q = q + attn_out

        q_norm2 = self.norm2(q)
        cross_out, _ = self.cross_attn(q_norm2, c, c, key_padding_mask=key_padding_mask)
        q = q + cross_out

        q_norm3 = self.norm3(q)
        q = q + self.mlp(q_norm3)
        return q


class CrossModalPredictor(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 predictor_embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 max_patches: int = 256):
        super().__init__()
        self.predictor_proj = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, max_patches, predictor_embed_dim))
        self.max_patches = max_patches

        self.blocks = nn.ModuleList([
            PredictorBlock(predictor_embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj_out = nn.Linear(predictor_embed_dim, embed_dim)

        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.predictor_pos_embed, std=.02)

    def _get_pos_embed(self, n_patches):
        if n_patches <= self.max_patches:
            return self.predictor_pos_embed[:, :n_patches, :]
        return F.interpolate(
            self.predictor_pos_embed.permute(0, 2, 1),
            size=n_patches,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

    def forward(self, context_tokens: torch.Tensor, mask: torch.Tensor):
        B, N_vis, D = context_tokens.shape
        N_all = mask.shape[1]

        c = self.predictor_proj(context_tokens)

        pos_embed = self._get_pos_embed(N_all)

        mask_tokens = self.mask_token.expand(B, N_all, -1).clone()
        mask_tokens = mask_tokens + pos_embed.expand(B, -1, -1)

        queries = mask_tokens[mask].reshape(B, -1, mask_tokens.shape[-1])

        for block in self.blocks:
            queries = block(queries, c)

        queries = self.predictor_norm(queries)
        pred_tokens = self.predictor_proj_out(queries)
        return pred_tokens


class SameModalPredictor(CrossModalPredictor):
    pass
