import abc
import torch


class BaseTimestepSampler(abc.ABC, torch.nn.Module):
    """Base class for timestep samplers."""

    @abc.abstractmethod
    def get_sample(self, n_samples):
        ...


    @abc.abstractmethod
    def get_path(self, n_samples, n_steps):
        ...