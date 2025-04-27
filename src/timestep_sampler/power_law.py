import torch

from .base import BaseTimestepSampler


class PowerLawTimestepSampler(BaseTimestepSampler):
    """
    Samples timesteps according to a power law on [0, 1] range. The density function is:
    
    p(t) = \lambda^{-1} x^{\lambda^{-1} - 1},

    where \lambda represents the power and:
    - \lambda = 1 leads to uniform distribution
    - \lambda < 1 focuses the density near 0
    - \lambda > 1 focuses the density near 1
    """

    def __init__(self, power: float):
        super(PowerLawTimestepSampler, self).__init__()

        assert power > 0.
        self.power = power


    def get_sample(self, n_samples):
        ts = torch.rand((n_samples,))
        return ts ** (1 / self.power)


    def get_path(self, n_steps):
        quantiles = torch.linspace(1, 0, n_steps + 1).unsqueeze(1)
        timesteps = quantiles ** (1 / self.power)
        dts = - timesteps.diff(dim=0)
        return timesteps, dts