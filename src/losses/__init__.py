from .jepa_pred import total_pred_loss
from .sigreg import total_sigreg_loss, SIGRegLoss
from .nce import total_retrieval_loss
import torch

_sigreg_fn = None

def compute_total_loss(model_output: dict,
                       lambda_pred: float = 1.0,
                       lambda_retr: float = 1.0, 
                       lambda_sigreg: float = 0.01):
    """
    Total loss = lambda_pred * L_pred + lambda_retr * L_retr + lambda_sigreg * L_sigreg
    """
    global _sigreg_fn
    
    l_pred   = total_pred_loss(model_output)
    l_retr   = total_retrieval_loss(model_output)
    
    # Initialize SIGRegLoss once and put it on the correct device
    if _sigreg_fn is None:
        device = model_output.get('r_cross_a', torch.tensor([])).device
        _sigreg_fn = SIGRegLoss().to(device)
        
    l_sigreg = total_sigreg_loss(model_output, _sigreg_fn)
    
    total = lambda_pred * l_pred + lambda_retr * l_retr + lambda_sigreg * l_sigreg
    
    return {
        'total': total,
        'pred': l_pred.item(),
        'retr': l_retr.item(),
        'sigreg': l_sigreg.item()
    }
