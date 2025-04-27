import torch
import wandb
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseTimestepSampler


class InverseI2SBSmoothTimestepSampler(BaseTimestepSampler):


    def __init__(self, epsilon: float, epsilon_ub: float, alpha: float, log_assets: bool, *args, **kwargs):
        super(InverseI2SBSmoothTimestepSampler, self).__init__()

        self.epsilon = epsilon
        self.epsilon_ub = epsilon_ub
        self.alpha = alpha
        self.dummy_tensor = torch.tensor([], dtype=torch.float32)
        if log_assets: self.log_plot()


    def get_sample(self, n_samples):
        """Samples random timesteps from [epsilon, 1 - epsilon_ub]"""
        steps = torch.from_numpy(
            scipy.stats.beta.rvs(self.alpha, self.alpha, size=(n_samples,))).to(self.dummy_tensor.device).float()
        scaled_steps = self.epsilon + (1 - self.epsilon - self.epsilon_ub) * steps
        return scaled_steps


    def get_path(self, n_steps):
        linspace = np.linspace(1, 0, n_steps + 1)  # uniform grid in [0,1]
        timesteps = torch.from_numpy(
            scipy.stats.beta.ppf(linspace, self.alpha, self.alpha)).to(self.dummy_tensor.device).float()
        timesteps = self.epsilon + (1 - self.epsilon - self.epsilon_ub) * timesteps
        timesteps = timesteps.unsqueeze(1)
        dts = - timesteps.diff(dim=0)
        return timesteps, dts
    
    
    def get_T(self, batch):
        return torch.ones_like(batch) - self.epsilon_ub
    

    def log_plot(self):
        timesteps = self.get_path(1000)[0].numpy(force=True)
        fig = plt.figure()
        plt.hist(timesteps, bins=50)
        plt.grid()
        wandb.log({f'misc/sampling_path': wandb.Image(fig)})
        plt.close(fig)