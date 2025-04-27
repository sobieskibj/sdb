import torch
import numpy as np

from .base import BaseNoiseSchedule


class TrueCosineNoiseSchedule(BaseNoiseSchedule):


    def __init__(self, s: float, min_t: float):
        super(TrueCosineNoiseSchedule, self).__init__()

        self.s = torch.tensor(s)
        self.pi = torch.tensor(np.pi)
        self.f_0 = self.f(torch.tensor(min_t))


    def f(self, t):
        f_base = ((self.pi / 2) * ((t + self.s) / (1 + self.s))).cos() ** 2
        return f_base


    def forward(self, t):
        base_weight = 1. - (self.f(t) / self.f_0)
        return (base_weight / t).sqrt()