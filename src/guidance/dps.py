import torch

from .base import BaseGuidance


class DPS(BaseGuidance):
    differentiable = True

    def __init__(self, zeta_prime: float, scale: float, covar_scale: bool = True):
        super(DPS, self).__init__()

        self.zeta_prime = zeta_prime
        self.covar_scale = covar_scale
        self.scale = scale

    def forward(self, x_t, t, output, y, pinvA_y, **kwargs):
        # ensure that measurement system is given
        assert "measurement_system" in kwargs.keys()
        measurement_system = kwargs["measurement_system"]

        # compute likelihood score
        residual_norm = torch.linalg.norm(
            y - measurement_system.mean_system_response(output)
        )
        grad = torch.autograd.grad(residual_norm, x_t)[0]

        # optionally scale by inverse covariance
        if self.covar_scale:
            grad = measurement_system.noise_covariance.invSigma(1.0) * grad

        # scale by step size
        return self.zeta_prime * grad


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]
