import torch

from .base import BaseGuidance


import torch


import torch


import torch


class DDPG(BaseGuidance):
    differentiable = False

    def __init__(self, zeta_prime: float, scale: float, covar_scale: bool):
        """
        Initialize the DDPG guidance technique.
        :param zeta_prime: scaling factor for the guidance term (typically set based on model tuning).
        :param scale: scale factor for the guidance.
        :param covar_scale: whether to scale by the noise covariance.
        """
        super(DDPG, self).__init__()
        self.scale = scale

    def forward(self, x_t, t, output, y, pinvA_y, **kwargs):
        """
        Perform the guidance step for DDPG at time step t.
        :param x_t: Current noisy image (x_t).
        :param t: Current timestep.
        :param output: The predicted clean image, output of the neural network.
        :param y: The observed measurement.
        :param pinvA_y: The precomputed pseudo-inverse of the forward model applied to the measurement.
        :param kwargs: Additional arguments that include measurement system and variance schedule.
        :return: The corrected image after applying DDPG guidance.
        """
        # Ensure the required arguments are passed
        assert {"measurement_system", "variance_schedule"} <= kwargs.keys()

        # Extract the measurement system and variance schedule
        measurement_system = kwargs["measurement_system"]
        variance_schedule = kwargs["variance_schedule"]

        # Operators
        A = measurement_system.mean_system_response  # Forward op: A(x)
        AT = A.AT  # Adjoint op: A^T(z)
        sigma_y_sq = (
            measurement_system.noise_covariance.var
        )  # The noise covariance (variance)

        # Predicted x0 (clean image estimate)
        x0_pred = output  # Network predicts the clean image x0

        # Compute residual: (y - A(x0_pred)) over the batch
        residual = y - A(x0_pred)

        # Diffusion variance at timestep t: sigma_t^2 (shape: [B])
        sigma_t_sq = variance_schedule.sigma_sq(t)  # scalar or [B]

        # Compute the preconditioned residual:
        # Solve (A Sigma_t A^T + Sigma)^(-1) * residual, where:
        # Sigma_t = sigma_t_sq * I (diffusion covariance at timestep t)
        # Sigma = sigma_y_sq (measurement noise covariance)

        def solve_preconditioned(residual, sigma_t_sq, sigma_y_sq):
            result = []
            batch_size = residual.shape[0]  # [B, ...]
            for i in range(batch_size):
                r = residual[i].unsqueeze(0)  # [1, C, H, W]
                sig_sq = sigma_t_sq[i]  # [scalar]

                # Define the matrix-vector multiplication M(v) = A(A^T(v)) * sigma_t_sq + Sigma * v
                def M(v):
                    return A(AT(v)) * sig_sq + sigma_y_sq * v

                # Solve the system M * v = r for each sample in the batch
                # M(v) = r --> (A Sigma_t A^T + Sigma) * v = r
                M_residual = M(
                    r
                )  # This gives us M(v) = r without the need for explicit matrix construction
                v = torch.linalg.solve(M_residual, r)  # Solve the system directly

                result.append(v)
            return torch.stack(result).squeeze(1)

        # Step 2: Apply the full guidance formula
        preconditioned_residual = solve_preconditioned(residual, sigma_t_sq, sigma_y_sq)

        # Now compute the correction using the adjoint operator A^T
        correction = sigma_t_sq.view(-1, *[1] * (x0_pred.ndim - 1)) * AT(
            preconditioned_residual
        )

        # Apply scaling and return the result
        return self.scale * correction


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]
