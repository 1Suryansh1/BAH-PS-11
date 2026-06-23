import torch
from src.losses.sigreg import total_sigreg_loss

def test():
    # Mock model output
    B, D = 4, 768
    model_output = {
        'emb_cross': torch.randn(B, D, requires_grad=True),
        'emb_uni': torch.randn(B, D, requires_grad=True)
    }
    
    loss = total_sigreg_loss(model_output)
    print("SIGReg Loss:", loss.item())
    
    loss.backward()
    
    print("Cross gradient sum:", model_output['emb_cross'].grad.sum().item())
    print("Test passed successfully.")

if __name__ == "__main__":
    test()
