import torch
import wandb
import matplotlib.pyplot as plt

from .base import BaseTimestepSampler


class PowerFunctionTimestepSampler(BaseTimestepSampler):
    """Samples timesteps distributed according to a power function in the [0, 1] range."""

    def __init__(self, power: float, epsilon: float, epsilon_ub: float, log_assets: bool, *args, **kwargs):
        super(PowerFunctionTimestepSampler, self).__init__()

        self.power = power
        self.epsilon = epsilon
        self.epsilon_ub = epsilon_ub
        if log_assets: self.log_plot()


    def get_sample(self, n_samples):
        """Samples random timesteps from [epsilon, 1 - epsilon_ub] with a power-law distribution."""
        uniform_steps = torch.rand(n_samples)
        scaled_steps = self.epsilon + (1 - self.epsilon - self.epsilon_ub) * uniform_steps.pow(self.power)
        return scaled_steps


    def get_path(self, n_steps):
        linspace = torch.linspace(1, 0, steps=n_steps + 1)  # uniform grid in [0,1]
        timesteps = self.epsilon + (1 - self.epsilon - self.epsilon_ub) * linspace.pow(self.power)
        timesteps = timesteps.unsqueeze(1)
        dts = - timesteps.diff(dim=0)
        return timesteps, dts
    
    
    def get_T(self, batch):
        return torch.ones_like(batch) - self.epsilon_ub
    

    def log_plot(self):
        timesteps = self.get_path(20)[0].numpy(force=True)
        fig = plt.figure()
        plt.plot(range(len(timesteps)), timesteps, marker='o')
        plt.grid()
        wandb.log({f'misc/sampling_path': wandb.Image(fig)})
        plt.close(fig)