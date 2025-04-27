import torch

from .base import BaseGuidance


class DDNM(BaseGuidance):
    differentiable = False

    def __init__(self, zeta_prime: float, scale: float, covar_scale: bool):
        super(DDNM, self).__init__()

        self.scale = scale

    def forward(self, x_t, t, output, y, pinvA_y, **kwargs):
        # ensure that measurement system is given
        assert set(["measurement_system", "variance_schedule"]).issubset(kwargs.keys())
        measurement_system = kwargs["measurement_system"]
        variance_schedule = kwargs["variance_schedule"]

        # extract std of measurement system
        sigma_y = measurement_system.noise_covariance.var ** (1 / 2)

        x0_pred = output  # output is interpreted as x0 estimate

        # Retrieve sigma_t^2 from the variance schedule at time t
        sigma_t_sq = variance_schedule.sigma_sq(t)  # This returns sigma_t^2 directly

        # Compute sigma_t from sigma_t^2
        sigma_t = torch.sqrt(sigma_t_sq)

        # Get alpha_t
        alpha_t = variance_schedule.alpha(t)

        # Compute the threshold for lambda_t
        sigma_thresh_sq = (alpha_t * sigma_y) ** 2

        # Compute lambda_t
        lambda_t = torch.where(
            sigma_t_sq >= sigma_thresh_sq,
            torch.ones_like(t),
            sigma_t / (alpha_t * sigma_y),
        )

        # Compute A * x_0_pred (the measurement response)
        A_x0 = measurement_system.mean_system_response(x0_pred)

        # Compute the residual between the predicted measurement and actual observation
        residual = A_x0 - y

        # Compute the correction using the pseudo-inverse of the residual
        correction = measurement_system.pinvA(residual)

        # Apply scaling
        lambda_t = lambda_t.view(
            -1, *[1] * (correction.ndim - 1)
        )  # expand dims for broadcasting
        return self.scale * lambda_t * correction


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]
