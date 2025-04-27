import torch

from .base import BaseTimestepSampler


class UniformDDBMTimestepSampler(BaseTimestepSampler):
    """Samples timesteps distributed uniformly across the [0, 1] range according to DDBM implementation."""

    def __init__(self, sigma_min: float, sigma_max: float, *args, **kwargs):
        super(UniformDDBMTimestepSampler, self).__init__()
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max


    def get_sample(self, n_samples):
        timesteps = torch.rand(n_samples) * (self.sigma_max - self.sigma_min) + self.sigma_min
        return timesteps


    def get_path(self, n_steps):
        timesteps = torch.linspace(self.sigma_max - 1e-4, self.sigma_min, n_steps + 1).unsqueeze(1)
        dts = - timesteps.diff(dim=0)
        return timesteps, dts