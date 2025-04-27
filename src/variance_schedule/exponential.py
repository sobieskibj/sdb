import torch

from .base import BaseVarianceSchedule


class ExponentialVarianceSchedule(BaseVarianceSchedule):


    def __init__(self, k: float):
        super(ExponentialVarianceSchedule, self).__init__()

        self.k = torch.tensor(k)
        self.exp_m1 = torch.tensor(-1.).exp()


    def sigma(self, t):
        return (1. - (- t ** self.k).exp()) / (1. - self.exp_m1)
    

    def dsigma(self, t):
        return (self.k * (t ** (self.k - 1.)) * (- t ** self.k).exp()) / (1. - self.exp_m1)