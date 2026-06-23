import torch

def random_block_mask(n_patches_h, n_patches_w, mask_ratio=0.50, device='cpu'):
    """
    Random token masking for JEPA-style training.
    
    Args:
        n_patches_h, n_patches_w: grid dimensions (e.g., 7x7 for 112x112 img, patch=16)
        mask_ratio: fraction of patches to mask (0.50 = 24-25 of 49 tokens)
    
    Returns:
        mask: (n_patches_h * n_patches_w,) bool tensor
              True = this patch is MASKED (hidden from context encoder)
    """
    n_total = n_patches_h * n_patches_w
    n_mask = int(n_total * mask_ratio)
    
    # Random shuffle of indices, first n_mask are masked
    indices = torch.randperm(n_total, device=device)
    mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    mask[indices[:n_mask]] = True
    return mask

def get_visible_tokens(tokens, mask):
    """
    Args:
        tokens: (B, N, D) patch token tensor from ViT encoder
        mask:   (B, N) bool tensor — True = masked
    Returns:
        visible_tokens: (B, N_visible, D) — only unmasked tokens
    """
    B, N, D = tokens.shape
    return tokens[~mask].view(B, -1, D)

def get_masked_tokens(tokens, mask):
    """
    Args:
        tokens: (B, N, D) patch token tensor from ViT encoder
        mask:   (B, N) bool tensor — True = masked
    Returns:
        masked_tokens: (B, N_masked, D) — only masked tokens
    """
    B, N, D = tokens.shape
    return tokens[mask].view(B, -1, D)
