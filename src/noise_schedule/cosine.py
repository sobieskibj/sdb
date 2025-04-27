import torch
import numpy as np

from .base import BaseNoiseSchedule


class CosineNoiseSchedule(BaseNoiseSchedule):


    def __init__(self, s: float, sqrt_f: bool, squared_weight: bool):
        super(CosineNoiseSchedule, self).__init__()

        self.s = torch.tensor(s)
        self.sqrt_f = sqrt_f
        self.squared_weight = squared_weight
        self.pi = torch.tensor(np.pi)
        self.f_0 = self.f(torch.tensor(0.))


    def f(self, t):
        f_base = ((self.pi / 2) * ((t + self.s) / (1 + self.s))).cos()
        return f_base ** 2 if not self.sqrt_f else f_base


    def forward(self, t):
        base_weight = 1. - (self.f(t) / self.f_0)
        return base_weight if not self.squared_weight else base_weight ** 2