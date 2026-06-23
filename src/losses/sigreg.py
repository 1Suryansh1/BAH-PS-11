import torch
import torch.nn as nn
from torch import distributed as dist


def all_reduce(x, op="SUM"):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x / dist.get_world_size()
    return x


class EppsPulley(nn.Module):
    def __init__(self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"):
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd for trapezoid integration"
        self.integration = integration
        self.n_points = n_points

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = (-0.5 * t.square()).exp()
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x):
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        cos_mean = cos_vals.mean(-3)
        sin_mean = sin_vals.mean(-3)

        cos_mean = all_reduce(cos_mean)
        sin_mean = all_reduce(sin_mean)

        err = (cos_mean - self.phi).square() + sin_mean.square()

        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        return (err @ self.weights) * N * world_size


class SlicingUnivariateTest(nn.Module):
    def __init__(self, univariate_test, num_slices: int, reduction: str = "mean", clip_value: float = None):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        with torch.no_grad():
            global_step_sync = all_reduce(self.global_step.clone(), op="SUM")
            seed = global_step_sync.item()
            g = self._get_generator(x.device, seed)
            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, device=x.device, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)

        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0

        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        return stats


class SIGRegLoss(nn.Module):
    def __init__(self, num_slices=128):
        super().__init__()
        self.test = SlicingUnivariateTest(
            univariate_test=EppsPulley(t_max=3.0, n_points=17),
            num_slices=num_slices,
            reduction='mean'
        )

    def forward(self, embeddings):
        if embeddings.shape[0] < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        mean = embeddings.mean(dim=0, keepdim=True)
        std = embeddings.std(dim=0, unbiased=False, keepdim=True) + 1e-6
        normalized_emb = (embeddings - mean) / std
        return self.test(normalized_emb)


def total_sigreg_loss(model_output: dict, sigreg_fn: SIGRegLoss):
    l1 = sigreg_fn(model_output['r_cross_a'])
    l2 = sigreg_fn(model_output['r_cross_b'])
    l3 = sigreg_fn(model_output['r_uni_a'])
    l4 = sigreg_fn(model_output['r_uni_b'])
    return (l1 + l2 + l3 + l4) / 4.0
