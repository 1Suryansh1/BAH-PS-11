import torch
from src.losses.sigreg import total_sigreg_loss, SIGRegLoss


def test():
    B, D = 4, 768
    model_output = {
        'r_cross_a': torch.randn(B, D, requires_grad=True),
        'r_cross_b': torch.randn(B, D, requires_grad=True),
        'r_uni_a':   torch.randn(B, D, requires_grad=True),
        'r_uni_b':   torch.randn(B, D, requires_grad=True),
    }

    sigreg_fn = SIGRegLoss(num_slices=128)
    loss = total_sigreg_loss(model_output, sigreg_fn)
    print("SIGReg Loss:", loss.item())

    loss.backward()

    for key in ['r_cross_a', 'r_cross_b', 'r_uni_a', 'r_uni_b']:
        assert model_output[key].grad is not None, f"No gradient for {key}"
        assert not torch.isnan(model_output[key].grad).any(), f"NaN gradient in {key}"

    grad_sum = (model_output['r_cross_a'].grad.sum().item() +
                model_output['r_cross_b'].grad.sum().item())
    print("Cross gradient sum:", grad_sum)
    print("Test passed successfully.")


if __name__ == "__main__":
    test()
