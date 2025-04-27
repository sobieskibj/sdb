import torch

from .base import BaseTimestepSampler


class InverseExponentialTimestepSampler(BaseTimestepSampler):
    """Samples timesteps using inverse function of exponential noise schedule."""

    def __init__(self, k: float, *args, **kwargs):
        super(InverseExponentialTimestepSampler, self).__init__()
        self.k = k
        self.exp_m1 = 1 - torch.tensor(-1.).exp()


    def inverse_f(self, t):
        return (-torch.log(1 - t * self.exp_m1))**(1 / self.k)


    def get_sample(self, n_samples):
        variances = torch.rand((n_samples,))
        return self.inverse_f(variances)


    def get_path(self, n_steps):
        variances = torch.linspace(1, 0, n_steps + 1).unsqueeze(1)
        timesteps = self.inverse_f(variances)
        dts = - timesteps.diff(dim=0)
        return timesteps, dts