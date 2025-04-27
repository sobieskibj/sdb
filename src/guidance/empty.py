import torch

from .base import BaseGuidance


class EmptyGuidance(BaseGuidance):
    differentiable = False

    def __init__(self, zeta_prime: float, scale: float, covar_scale: bool):
        super(EmptyGuidance, self).__init__()

    def forward(self, *args, **kwargs):
        return 0.0
