import torch

from .base import BaseTimestepSampler


class UniformTimestepSampler(BaseTimestepSampler):
    """Samples timesteps distributed uniformly across the [0, 1] range."""

    def __init__(self, epsilon: float, epsilon_ub: float, *args, **kwargs):
        super(UniformTimestepSampler, self).__init__()

        self.epsilon = epsilon
        self.epsilon_ub = epsilon_ub

    def get_sample(self, n_samples):
        return torch.rand((n_samples,))

    def get_path(self, n_steps):
        timesteps = torch.linspace(
            1 - self.epsilon_ub, self.epsilon, n_steps + 1
        ).unsqueeze(1)
        dts = -timesteps.diff(dim=0)
        return timesteps, dts

    def get_T(self, batch):
        return torch.ones_like(batch) - self.epsilon_ub
