import torch

from .base import BaseGuidance


class PiGDM(BaseGuidance):
    differentiable = False

    def __init__(self, zeta_prime: float, scale: float, covar_scale: bool):
        super(PiGDM, self).__init__()

        self.scale = scale

    def forward(self, x_t, t, output, y, pinvA_y, **kwargs):
        assert {"measurement_system", "variance_schedule"} <= kwargs.keys()

        measurement_system = kwargs["measurement_system"]
        variance_schedule = kwargs["variance_schedule"]

        # Access operators
        A = measurement_system.mean_system_response  # forward op: A(x)
        AT = A.AT  # adjoint op: A^T(z)

        sigma_y = measurement_system.noise_covariance.var ** (1 / 2)  # scalar
        x0_pred = output  # predicted x0

        # Compute residual: y - A(x0_pred)
        A_x0 = A(x0_pred)
        residual = y - A_x0  # shape: [B, ...]

        # Compute diffusion variance
        sigma_t_sq = variance_schedule.sigma_sq(t)  # shape: [B]

        if sigma_y == 0:
            # Hard constraint: pseudo-inverse projection
            correction = measurement_system.pinvA(
                -residual
            )  # negative because residual = y - A(x₀)
        else:
            # PiGDM gradient: scaled A^T Σ⁻¹ (y - A(x₀))
            correction = AT(residual)  # shape: like x₀
            scale = sigma_t_sq / sigma_y**2

            # Reshape for broadcasting
            scale = scale.view(-1, *[1] * (correction.ndim - 1))
            correction = scale * correction

        return self.scale * correction


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]
