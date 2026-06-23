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
    
    def forward(self, q, c):
        # q: query tokens (B, N_q, D)
        # c: context tokens for cross-attention (B, N_c, D)
        q_norm = self.norm1(q)
        attn_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        q = q + attn_out
        
        q_norm2 = self.norm2(q)
        cross_out, _ = self.cross_attn(q_norm2, c, c)
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
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        # Learnable positional embeddings
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, max_patches, predictor_embed_dim))
        
        self.blocks = nn.ModuleList([
            PredictorBlock(predictor_embed_dim, num_heads, mlp_ratio) 
            for _ in range(depth)
        ])
        
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)
        self.predictor_proj_out = nn.Linear(predictor_embed_dim, embed_dim)
        
        # Initialization
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.predictor_pos_embed, std=.02)

    def forward(self,
                context_tokens: torch.Tensor,   # (B, N_vis, D)
                mask: torch.Tensor):            # (B, N_all) bool — B's masked positions
        
        B, N_vis, D = context_tokens.shape
        
        # Project down context to serve as cross-attention keys/values
        c = self.predictor_proj(context_tokens) # (B, N_vis, proj_dim)
        
        # Create queries from mask tokens + pos embedding
        # We need to collect pos embeddings for the masked positions
        queries = []
        for i in range(B):
            masked_pos = torch.where(mask[i])[0]
            pos_emb = self.predictor_pos_embed[0, masked_pos, :]
            q_i = self.mask_token[0].expand(len(masked_pos), -1) + pos_emb
            queries.append(q_i)
        
        queries = torch.stack(queries, dim=0) # (B, N_masked, proj_dim)
        
        # Forward through blocks
        for block in self.blocks:
            queries = block(queries, c)
            
        queries = self.predictor_norm(queries)
        
        # Project back up
        pred_tokens = self.predictor_proj_out(queries) # (B, N_masked, D)
        
        return pred_tokens

class SameModalPredictor(CrossModalPredictor):
    """
    Simpler predictor for same-modal prediction.
    Inherits from CrossModalPredictor since architecture is identical.
    """
    pass
