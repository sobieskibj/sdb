import enum
import torch

from .base import BaseLinearSDE
from network.base import PredictionType
from variance_schedule.base import BaseVarianceSchedule
from measurement_model.base import BaseLinearModelWithGaussianNoise


class ReverseType(enum.Enum):
    SDE = "SDE"
    PFODE = "PFODE"


class SDB(BaseLinearSDE):


    def __init__(
            self, 
            measurement_system: BaseLinearModelWithGaussianNoise,
            variance_schedule: BaseVarianceSchedule,
            reverse_type: str):
    
        super(SDB, self).__init__()

        self.measurement_system = measurement_system
        self.variance_schedule = variance_schedule
        self.reverse_type = reverse_type
        self.y_shape = None


    def sample_x_t_minus_dt_given_x_t(self, x_t, t, dt, cond, denoising_network, is_last=False):

        # predict with un/conditional network
        output = denoising_network(x_t, cond, t)

        # compute drift part of the reverse equation
        coef_drift = self._map_network_output_to_reverse_drift(x_t, t, output, denoising_network.prediction_type)

        # sample according to reverse sde
        if ReverseType[self.reverse_type] == ReverseType.SDE:

            # sample noise only if its not the last step
            if is_last:
                coef_diff = 0.
                
            else:
                # get noise for range space
                noise_x = torch.randn_like(x_t)

                # get measurement shape if not known yet
                if self.y_shape is None:
                    self.y_shape = self.measurement_system(x_t).shape[1:]

                # get noise for measurement space
                noise_y = torch.randn(x_t.shape[0], *self.y_shape)
                coef_diff = self.G_t(noise_x, noise_y, t)

            return x_t + coef_drift * expand(dt, x_t) + coef_diff * expand(dt.sqrt(), x_t)
        
        # sample according to probability flow ode
        elif ReverseType[self.reverse_type] == ReverseType.PFODE:

            return x_t + coef_drift * expand(dt, x_t)
        
        else:
            raise ValueError(f"Reverse type {self.reverse_type} not recognized.")
    

    def _map_network_output_to_reverse_drift(self, x_t, t, output, prediction_type):

        forward_drift = self.F_t(x_t, t)
        coef = 0.5 if ReverseType[self.reverse_type] == ReverseType.PFODE else 1.0

        if PredictionType[prediction_type] == PredictionType.MEAN:
            reverse_drift = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            
            residual = self.H_t(output, t) - x_t
            coef_sigma = self.variance_schedule.d_sigma_sq_t_over_sigma_sq_t(t)
            g_t_sq_score = expand(coef_sigma, residual) * residual - 2 * self.F_t(residual, t)
            reverse_drift = forward_drift - coef * g_t_sq_score

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            reverse_drift = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")
        
        return - reverse_drift


    def sample_x_t_given_x_0(self, x, t):
        
        # compute the measurement and its pseudoinverse reconstruction
        # NOTE: noise in the measurement is different from the above one
        y = self.measurement_system(x)
        pinvA_y = self.measurement_system.pinvA(y)
        if self.y_shape is None: self.y_shape = y.shape[1:]

        # sample white noise for null and range spaces
        noise_y = torch.randn_like(y)
        noise_x = torch.randn_like(x)

        # map x_0 to x_t
        x_t = self.H_t(x, t) + self.sqrt_Sigma_t(noise_y, noise_x, t)

        return x_t, pinvA_y, y


    def sample_x_T(self, pinvA_y, T):
        sigma_sq_T = self.variance_schedule.sigma_sq_T(T)
        noise = torch.randn_like(pinvA_y)
        return pinvA_y + expand(sigma_sq_T.sqrt(), noise) * (noise - self.measurement_system.pinvA_A(noise))


    def H_t(self, x, t):

        # compute identity and range coefficients
        coef_id = expand(self.variance_schedule.alpha(t), x)
        coef_range = 1 - coef_id

        # compute x mappings
        x_range = self.measurement_system.pinvA_A(x)
        x_id = x

        return coef_range * x_range + coef_id * x_id


    def sqrt_Sigma_t(self, noise_y, noise_x, t):

        coef_range = self.variance_schedule.sigma_sq_t_over_sigma_sq_T(t)
        coef_null = self.variance_schedule.sigma_sq(t)
        
        x_range = self.measurement_system.pinvA_Sigma_sqrt(noise_y)
        x_null = noise_x - self.measurement_system.pinvA_A(noise_x)

        return expand(coef_range.sqrt(), x_range) * x_range + expand(coef_null.sqrt(), x_null) * x_null


    def F_t(self, x, t):
        
        # compute coefficient
        coef = self.variance_schedule.d_alpha_t_over_alpha_t(t)

        # map x
        x = x - self.measurement_system.pinvA_A(x)

        return expand(coef, x) * x


    def G_t(self, noise_x, noise_y, t):
        
        # compute coefficients
        coef_range = self.variance_schedule.d_sigma_sq_t_over_sigma_sq_T(t)
        coef_null = self.variance_schedule.d_sigma_sq(t) - 2 * self.variance_schedule.sigma_sq(t) * \
            self.variance_schedule.d_alpha_t_over_alpha_t(t)
        
        # compute mappings of x
        noise_range = self.measurement_system.pinvA_Sigma_sqrt(noise_y)
        noise_null = noise_x - self.measurement_system.pinvA_A(noise_x)

        return expand(coef_range.sqrt(), noise_range) * noise_range + expand(coef_null.sqrt(), noise_null) * noise_null


    def _map_x_0_to_network_target(self, x_0, x_t, t, prediction_type):

        if PredictionType[prediction_type] == PredictionType.MEAN:
            target = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            target = x_0

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            target = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")

        return target
    

    def _map_network_output_to_x_0(self, x_t, t, output, prediction_type):

        if PredictionType[prediction_type] == PredictionType.MEAN:
            x_0 = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            x_0 = output

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            x_0 = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")
        
        return x_0


    def fix_state(self, x_0, fabric, eval):
        self.measurement_system.fix_state(x_0, fabric, eval)


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]