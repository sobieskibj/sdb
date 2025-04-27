import enum
import torch

from .base import BaseScalarSDE
from guidance.base import BaseGuidance
from network.base import PredictionType, ConditioningType
from variance_schedule.base import BaseVarianceSchedule
from measurement_model.base import BaseLinearModelWithGaussianNoise


class ReverseType(enum.Enum):
    SDE = "SDE"
    PFODE = "PFODE"


class ScalarSDE(BaseScalarSDE):
    def __init__(
        self,
        measurement_system: BaseLinearModelWithGaussianNoise,
        variance_schedule: BaseVarianceSchedule,
        guidance: BaseGuidance,
        reverse_type: str,
        jump_start: bool = False,
    ):
        super(ScalarSDE, self).__init__()

        self.measurement_system = measurement_system
        self.variance_schedule = variance_schedule
        self.reverse_type = reverse_type
        self.guidance = guidance
        self.jump_start = jump_start

    def _sample_x_t_minus_dt_given_x_t(
        self, x_t, t, dt, y, pinvA_y, denoising_network, is_last=False
    ):
        # pick conditioning based on network info
        if (
            ConditioningType[denoising_network.condition_type]
            == ConditioningType.PSEUDOINVERSE_RECONSTRUCTION
        ):
            cond = pinvA_y

        elif (
            ConditioningType[denoising_network.condition_type]
            == ConditioningType.MEASUREMENT
        ):
            cond = y

        elif (
            ConditioningType[denoising_network.condition_type] == ConditioningType.NONE
        ):
            cond = None

        else:
            raise ValueError(
                f"Unrecognized ConditioningType: {denoising_network.condition_type}"
            )

        # predict with (un)conditional network
        output = denoising_network(x_t, cond, t)

        # compute drift part of the reverse equation
        coef_drift = self._map_network_output_to_reverse_drift(
            x_t, t, output, denoising_network.prediction_type
        )

        # sample according to reverse sde
        if ReverseType[self.reverse_type] == ReverseType.SDE:
            # sample noise only if its not the last step
            if is_last:
                coef_diff = 0.0

            else:
                noise = torch.randn_like(x_t)
                coef_diff = expand(self.G(t), noise) * noise

            # sample with unconditional process
            x_t_prime = (
                x_t - coef_drift * expand(dt, x_t) + coef_diff * expand(dt.sqrt(), x_t)
            )

        # sample according to probability flow ode
        elif ReverseType[self.reverse_type] == ReverseType.PFODE:
            # sample with unconditional process
            x_t_prime = x_t - coef_drift * expand(dt, x_t)

        else:
            raise ValueError(f"Reverse type {self.reverse_type} not recognized.")

        # correct with guidance term
        return x_t_prime - self.guidance(
            x_t,
            t,
            output,
            y,
            pinvA_y,
            **{
                "measurement_system": self.measurement_system,
                "variance_schedule": self.variance_schedule,
            },
        )

    def sample_x_t_minus_dt_given_x_t(
        self, x_t, t, dt, y, pinvA_y, denoising_network, is_last=False
    ):
        # for simplicity, we assume that output is the estimate of x_0
        assert PredictionType[denoising_network.prediction_type] == PredictionType.X_0

        if self.guidance.differentiable:
            # require grad to allow for guidance
            with torch.enable_grad():
                x_t.requires_grad_(True)
                return self._sample_x_t_minus_dt_given_x_t(
                    x_t, t, dt, y, pinvA_y, denoising_network, is_last
                )
        else:
            return self._sample_x_t_minus_dt_given_x_t(
                x_t, t, dt, y, pinvA_y, denoising_network, is_last
            )

    def sample_x_t_given_x_0(self, x, t):
        # sample noise
        noise = torch.randn_like(x)

        # get coeffs at timestep t
        H_t = expand(self.H(x, t), x)
        Sigma_sqrt_t = expand(self.Sigma_sqrt(t), x)

        # sample x_t
        return H_t * x + Sigma_sqrt_t * noise

    def sample_x_T(self, x, T):
        if not self.jump_start:
            return expand(self.Sigma_sqrt(T), x) * torch.randn_like(x)

        else:
            return self.sample_x_t_given_x_0(x, T)

    def F(self, x_t, t):
        return expand(self.variance_schedule.d_alpha_t_over_alpha_t(t), x_t) * x_t

    def G(self, t):
        return (
            self.variance_schedule.d_sigma_sq(t)
            - 2
            * self.variance_schedule.d_alpha_t_over_alpha_t(t)
            * self.variance_schedule.sigma_sq(t)
        ).sqrt()

    def H(self, x_0, t):
        return expand(self.variance_schedule.alpha(t), x_0) * x_0

    def Sigma_sqrt(self, t):
        return self.variance_schedule.sigma_sq(t).sqrt()

    def _map_network_output_to_reverse_drift(self, x_t, t, output, prediction_type):
        coef = 0.5 if ReverseType[self.reverse_type] == ReverseType.PFODE else 1.0

        if PredictionType[prediction_type] == PredictionType.MEAN:
            reverse_drift = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            forward_drift = self.F(x_t, t)
            residual = self.H(output, t) - x_t
            coef_residual = self.variance_schedule.d_sigma_sq_t_over_sigma_sq_t(
                t
            ) - 2 * self.variance_schedule.d_alpha_t_over_alpha_t(t)
            reverse_drift = (
                forward_drift - expand(coef * coef_residual, residual) * residual
            )

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            reverse_drift = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")

        return reverse_drift

    def fix_state(self, x_0, fabric, eval):
        self.measurement_system.fix_state(x_0, fabric, eval)


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]
