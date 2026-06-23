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
    visible_tokens = []
    for i in range(B):
        # mask[i] is True for masked, so ~mask[i] is visible
        vis = tokens[i, ~mask[i], :]
        visible_tokens.append(vis)
    # Stack back to (B, N_visible, D) assuming same mask count per batch element
    return torch.stack(visible_tokens, dim=0)

def get_masked_tokens(tokens, mask):
    """
    Args:
        tokens: (B, N, D) patch token tensor from ViT encoder
        mask:   (B, N) bool tensor — True = masked
    Returns:
        masked_tokens: (B, N_masked, D) — only masked tokens
    """
    B, N, D = tokens.shape
    masked_tokens = []
    for i in range(B):
        tgt = tokens[i, mask[i], :]
        masked_tokens.append(tgt)
    # Stack back to (B, N_masked, D) assuming same mask count per batch element
    return torch.stack(masked_tokens, dim=0)
