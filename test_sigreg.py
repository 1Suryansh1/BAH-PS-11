import torch
from src.losses.sigreg import total_sigreg_loss

def test():
    # Mock model output
    B, D = 4, 768
    model_output = {
        'r_cross_a': torch.randn(B, D, requires_grad=True),
        'r_cross_b': torch.randn(B, D, requires_grad=True),
        'r_uni_a': torch.randn(B, D, requires_grad=True),
        'r_uni_b': torch.randn(B, D, requires_grad=True)
    }
    
    loss = total_sigreg_loss(model_output)
    print("SIGReg Loss:", loss.item())
    
    loss.backward()
    
    grad_sum = (model_output['r_cross_a'].grad.sum().item() + 
                model_output['r_cross_b'].grad.sum().item())
    print("Cross gradient sum:", grad_sum)
    print("Test passed successfully.")

if __name__ == "__main__":
    test()
