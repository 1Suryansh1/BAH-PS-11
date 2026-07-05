import torch
import torch.nn.functional as F

def jepa_pred_loss(pred_tokens, target_tokens):
    """
    Smooth L1 (Huber) loss between predicted and target patch tokens.
    Target tokens are from the BACKBONE OUTPUT.
    """
    # CRITICAL: stop gradient on targets
    target_tokens = target_tokens.detach()
    
    # Normalize both
    pred_norm = F.normalize(pred_tokens, dim=-1)
    tgt_norm  = F.normalize(target_tokens, dim=-1)
    
    # L2 loss (MSE)
    loss = F.mse_loss(pred_norm, tgt_norm)
    return loss

def total_pred_loss(model_output):
    """
    Combines all 4 predictive routes.
    L_pred = L_aa + L_bb + L_ab + L_ba
    """
    l_aa = jepa_pred_loss(model_output['pred_aa'], model_output['tgt_a'])
    l_bb = jepa_pred_loss(model_output['pred_bb'], model_output['tgt_b'])
    l_ab = jepa_pred_loss(model_output['pred_ab'], model_output['tgt_b'])
    l_ba = jepa_pred_loss(model_output['pred_ba'], model_output['tgt_a'])
    return (l_aa + l_bb + l_ab + l_ba) / 4.0
