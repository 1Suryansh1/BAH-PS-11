import torch
import torch.nn.functional as F

def info_nce_loss(embeddings_a: torch.Tensor, 
                  embeddings_b: torch.Tensor,
                  temperature: float = 0.07):
    """
    Symmetric InfoNCE (NT-Xent) loss for cross-modal alignment.
    """
    embeddings_a = F.normalize(embeddings_a, dim=-1)
    embeddings_b = F.normalize(embeddings_b, dim=-1)

    # Similarity matrix (B, B) — symmetric
    sim = (embeddings_a @ embeddings_b.T) / temperature
    
    B = embeddings_a.shape[0]
    # Diagonal = positive pairs
    labels = torch.arange(B, device=sim.device)
    
    # Cross-entropy in both directions
    loss_ab = F.cross_entropy(sim, labels)
    loss_ba = F.cross_entropy(sim.T, labels)
    
    return (loss_ab + loss_ba) / 2.0

def cosine_alignment_loss(embeddings_a: torch.Tensor,
                           embeddings_b: torch.Tensor):
    """
    Direct cosine similarity alignment between paired embeddings.
    """
    a = F.normalize(embeddings_a, dim=-1)
    b = F.normalize(embeddings_b, dim=-1)
    cos_sim = (a * b).sum(dim=-1)  # (B,)
    return (1 - cos_sim).mean()

def total_retrieval_loss(model_output, lambda_cross=1.0, lambda_uni=0.5):
    """
    CR-JEPA retrieval loss: cross-modal NCE + cosine alignment
    """
    l_cross = info_nce_loss(model_output['e_cross_a'], model_output['e_cross_b'])
    l_uni   = cosine_alignment_loss(model_output['e_uni_a'], model_output['e_uni_b'])
    return lambda_cross * l_cross + lambda_uni * l_uni
