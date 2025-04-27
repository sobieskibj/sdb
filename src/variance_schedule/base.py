import abc
import torch


class BaseVarianceSchedule(abc.ABC, torch.nn.Module):


    def __init__(self):
        super(BaseVarianceSchedule, self).__init__()


    @abc.abstractmethod
    def sigma(self, t):
        ...


    @abc.abstractmethod
    def dsigma(self, t):
        ...