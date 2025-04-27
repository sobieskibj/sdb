import torch
import numpy as np

from .base import BaseNoiseSchedule


class ExponentialNoiseSchedule(BaseNoiseSchedule):


    def __init__(self, k: float):
        super(ExponentialNoiseSchedule, self).__init__()

        self.k = torch.tensor(k)
        self.exp_m1 = torch.tensor(-1.).exp()


    def forward(self, t):
        base_weight = (1 - (-t**self.k).exp()) / (1 - self.exp_m1)
        return (base_weight / t).sqrt()