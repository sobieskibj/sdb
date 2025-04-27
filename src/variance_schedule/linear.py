import torch

from .base import BaseVarianceSchedule


class LinearVarianceSchedule(BaseVarianceSchedule):


    def __init__(self):
        super(LinearVarianceSchedule, self).__init__()

        torch.c = torch.tensor([1.])


    def sigma(self, t):
        return t
    

    def dsigma(self, t):
        return torch.c