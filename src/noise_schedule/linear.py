import torch

from .base import BaseNoiseSchedule


class LinearNoiseSchedule(BaseNoiseSchedule):


    def __init__(self):
        """
        Implements linear noise schedule. As PDB is implemented with the wiener process,
        it suffices to leave the variance as is.
        """
        super(LinearNoiseSchedule, self).__init__()
        self.constant = torch.tensor([1.])


    def forward(self, x):
        return self.constant