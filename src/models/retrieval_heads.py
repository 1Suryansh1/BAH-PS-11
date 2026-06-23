import torch
import torch.nn as nn
import torch.nn.functional as F

class RetrievalProjectionHead(nn.Module):
    """
    Shared architecture for phi_cross and phi_uni.
    phi_cross: used for cross-modal retrieval
    phi_uni:   used for same-modal (unified) retrieval
    """
    def __init__(self,
                 input_dim: int = 768,      # CopFM output dim
                 hidden_dim: int = 2048,    # MLP hidden dimension
                 output_dim: int = 256):    # retrieval embedding dimension
        super().__init__()
        
        self.norm = nn.LayerNorm(input_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, token_sequence: torch.Tensor):
        """
        Args:
            token_sequence: (B, N_patches, 768) — ALL patch tokens from CopFM
        Returns:
            e: (B, output_dim) — L2-normalized embedding for retrieval
            r: (B, output_dim) — raw projection for SIGReg
        """
        pooled = token_sequence.mean(dim=1)          # (B, 768)
        x = self.norm(pooled)
        x = self.mlp(x)
        
        e = F.normalize(self.final_norm(x), dim=-1)  # retrieval embedding
        r = x                                        # raw for SIGReg
        
        return e, r

class DualRetrievalHead(nn.Module):
    """
    Wraps both phi_cross and phi_uni into one module.
    """
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=256):
        super().__init__()
        self.phi_cross = RetrievalProjectionHead(input_dim, hidden_dim, output_dim)
        self.phi_uni   = RetrievalProjectionHead(input_dim, hidden_dim, output_dim)
    
    def forward(self, tokens_a, tokens_b):
        """
        Returns:
            e_cross_a, r_cross_a
            e_cross_b, r_cross_b
            e_uni_a,   r_uni_a
            e_uni_b,   r_uni_b
        """
        e_ca, r_ca = self.phi_cross(tokens_a)
        e_cb, r_cb = self.phi_cross(tokens_b)
        
        e_ua, r_ua = self.phi_uni(tokens_a)
        e_ub, r_ub = self.phi_uni(tokens_b)
        
        return (e_ca, r_ca), (e_cb, r_cb), (e_ua, r_ua), (e_ub, r_ub)
