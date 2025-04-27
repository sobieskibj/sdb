import abc
import torch


class BaseSolver(abc.ABC, torch.nn.Module):
    def __init__(self):
        super(BaseSolver, self).__init__()

    @abc.abstractmethod
    def sample(self): ...
