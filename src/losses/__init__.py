from .jepa_pred import total_pred_loss
from .sigreg import total_sigreg_loss
from .nce import total_retrieval_loss

def compute_total_loss(model_output: dict,
                       lambda_pred: float = 1.0,
                       lambda_retr: float = 1.0, 
                       lambda_sigreg: float = 0.01):
    """
    Total loss = lambda_pred * L_pred + lambda_retr * L_retr + lambda_sigreg * L_sigreg
    """
    l_pred   = total_pred_loss(model_output)
    l_retr   = total_retrieval_loss(model_output)
    l_sigreg = total_sigreg_loss(model_output)
    
    total = lambda_pred * l_pred + lambda_retr * l_retr + lambda_sigreg * l_sigreg
    
    return {
        'total': total,
        'pred': l_pred.item(),
        'retr': l_retr.item(),
        'sigreg': l_sigreg.item()
    }
