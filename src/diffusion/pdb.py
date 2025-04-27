import enum
import wandb
import torch

from .base import BaseDiffusion

from measurement_model.base import BaseLinearModelWithGaussianNoise
from noise_covariance.base import BaseNoiseCovariance
from network.base import PredictionType, ConditioningType
from timestep_sampler.base import BaseTimestepSampler
from variance_schedule.base import BaseVarianceSchedule
from utils.wandb import min_max_scale


class ReverseType(enum.Enum):
    SDE = "SDE"
    PFODE = "PFODE"


class PseudoinverseDiffusionBridge(BaseDiffusion):


    def __init__(
            self, 
            measurement_likelihood: BaseLinearModelWithGaussianNoise, 
            null_space_noise_covariance: BaseNoiseCovariance,
            timestep_sampler: BaseTimestepSampler,
            variance_schedule: BaseVarianceSchedule,
            n_sampling_timesteps: int,
            reverse_type: ReverseType,
            low_snr_trick: bool,
            low_snr_trick_thr: float,
            naive_last: bool,
        ):
        
        super(PseudoinverseDiffusionBridge, self).__init__()

        self.measurement_likelihood = measurement_likelihood
        self.null_space_noise_covariance = null_space_noise_covariance(measurement_likelihood=measurement_likelihood)
        # ensure that measurement likelihood in null space covariance points to the same (in memory)
        # measurement likelihood that is used by diffusion
        assert self.measurement_likelihood is self.null_space_noise_covariance.measurement_likelihood
        self.timestep_sampler = timestep_sampler
        self.variance_schedule = variance_schedule
        self.n_sampling_timesteps = n_sampling_timesteps
        self.reverse_type = reverse_type
        self.low_snr_trick = low_snr_trick
        self.low_snr_trick_thr = low_snr_trick_thr
        self.naive_last = naive_last


    def sample_x_t_given_x_0(self, x_0, t, y_shape = None):

        # if measurement shape is not provided, we extract it
        if y_shape is None:

            with torch.no_grad():        
            
                y_shape = self.measurement_likelihood.mean_system_response.A(x_0).shape

        # sample range space noise
        range_space_noise = self.sample_range_space_noise(y_shape)

        # optionally zero out null space before adding noise to ensure zero SNR at t over given threshold
        if self.low_snr_trick:
            x_0[t >= self.low_snr_trick_thr] = self.null_space_noise_covariance._project_to_range_space(x_0)[t >= self.low_snr_trick_thr]

        # sample null space noise
        null_space_noise = self.sample_null_space_noise(x_0)

        # combine noises
        noise = range_space_noise + null_space_noise

        # add both noises to x_0 and return
        scaled_noise = expand(self.variance_schedule.sigma(t).sqrt(), x_0) * noise

        return x_0 + scaled_noise
    

    def sample_x_t_minus_dt_given_x_t(self, x_t, t, dt, denoising_network, conditioning, last=False):
        
        # if measurement shape is not provided, we extract it
        with torch.no_grad():
            y_shape = self.measurement_likelihood.mean_system_response.A(x_t).shape

        # predict with un/conditional network
        output = denoising_network(x_t, conditioning, t)

        if last and self.naive_last:
            # returns the denoised estimate at the last step
            return output

        # compute drift part of the reverse equation
        drift = self._map_network_output_to_reverse_drift(x_t, output, t, dt, denoising_network.prediction_type)

        # sample according to reverse sde
        if ReverseType[self.reverse_type] == ReverseType.SDE:

            if not last:
                # add noise for each step except the last one

                # sample range space noise
                range_space_noise = self.sample_range_space_noise(y_shape)

                # sample null space noise
                null_space_noise = self.sample_null_space_noise(x_t)

                # combine noises
                noise = range_space_noise + null_space_noise

            else:

                noise = 0.

            # get time-dependent coefficient
            t_coef = expand((self.variance_schedule.dsigma(t) * dt).sqrt(), x_t)

            return x_t + drift + t_coef * noise
        
        # sample according to probability flow ode
        elif ReverseType[self.reverse_type] == ReverseType.PFODE:

            return x_t + (drift / 2.)
        
        else:
            raise ValueError(f"Reverse type {self.reverse_type} not recognized.")


    def _map_network_output_to_reverse_drift(self, x_t, output, t, dt, prediction_type):

        if PredictionType[prediction_type] == PredictionType.MEAN:
            drift = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            # \frac{d\sigma}{\sigma}(t) * dt * (\hat{x}_0 - x_t)
            t_coef = expand((self.variance_schedule.dsigma(t) / self.variance_schedule.sigma(t)) * dt, x_t)
            drift = t_coef * (output - x_t)

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            drift = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")
        
        return drift


    def sample_range_space_noise(self, y_shape):

        # sample white noise for range space noise        
        range_space_noise = torch.randn(y_shape)

        # scale measurement domain white noise to get correct covariance
        range_space_noise = self.measurement_likelihood.noise_covariance.sqrtSigma(range_space_noise)
        range_space_noise = self.measurement_likelihood.mean_system_response.pinvA(range_space_noise)

        return range_space_noise


    def sample_null_space_noise(self, x):

        # generate white noise for null space
        null_space_noise = torch.randn_like(x)

        # scale image domain white noise to get correct covariance
        null_space_noise = self.null_space_noise_covariance.sqrtSigma(null_space_noise)

        return null_space_noise


    def sample(self, fabric, x_0, denoising_network, n_timesteps, log=False, log_prefix=""):

        # if A depends on randomness, i.e., sampled masks, it is fixed here
        self.measurement_likelihood.fix_state(x_0, fabric, eval=True)

        # get path of timesteps and dts for sampling
        n_timesteps = self.n_sampling_timesteps if n_timesteps is None else n_timesteps
        timesteps, dts = self.timestep_sampler.get_path(n_timesteps)

        # extract sizes
        B, C, H, W = x_0.shape

        # obtain pseudoinverse reconstruction + null space noise

        # get measurement
        measurement = self.measurement_likelihood(x_0)

        # sample starting point
        pinvA_y = self.measurement_likelihood.mean_system_response.pinvA(measurement)
        null_space_noise = self.sample_null_space_noise(pinvA_y)
        x_T = pinvA_y + null_space_noise
        x_t = x_T.clone()

        # pick conditioning based on network info
        if ConditioningType[denoising_network.condition_type] == ConditioningType.PSEUDOINVERSE_RECONSTRUCTION:
            conditioning = pinvA_y
        
        elif ConditioningType[denoising_network.condition_type] == ConditioningType.MEASUREMENT:
            conditioning = measurement

        # make list to collect x_t trajectory
        if log:
            x_t_traj = [x_t[0]]

        for i, (t, dt) in enumerate(zip(timesteps, dts)):

            # get time increment and timestep
            t = t.repeat(B)
            dt = dt.repeat(B)

            # predict with the network
            with torch.no_grad():

                # sample at timestep t except the last one
                x_t = self.sample_x_t_minus_dt_given_x_t(x_t, t, dt, denoising_network, conditioning, last=(i == n_timesteps - 1))

                if log:
                    x_t_traj.append(x_t[0])

        if log:
            wandb.log({
                f"{log_prefix}images/eval/x_0": wandb.Image(min_max_scale(x_0)),
                f"{log_prefix}images/eval/x_0_hat": wandb.Image(min_max_scale(x_t)),
                f"{log_prefix}images/eval/y": wandb.Image(min_max_scale(measurement)),
                f"{log_prefix}images/eval/x_t_traj": wandb.Image(min_max_scale(torch.stack(x_t_traj))),
                f"{log_prefix}images/eval/pinvA_y": wandb.Image(min_max_scale(pinvA_y))})
        
        return x_t


    def _map_x_0_to_network_target(self, x_0, x_t, t, prediction_type):
            
        if PredictionType[prediction_type] == PredictionType.MEAN:
            target = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            target = x_0

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            # \frac{x_t - x_0}{t^{\frac{1}{2}}} is the added correlated noise
            target = (x_t - x_0) / expand(t.sqrt(), x_0)

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")
        
        return target


    def _map_network_output_to_x_0(self, x_t, output, t, prediction_type: PredictionType):

        if PredictionType[prediction_type] == PredictionType.MEAN:
            x_0_hat = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            x_0_hat = output

        elif PredictionType[prediction_type] == PredictionType.EPSILON:
            x_0_hat = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")
        
        return x_0_hat


    def training_step(self, fabric, batch_idx, x_0, loss_fn, denoising_network, log_assets):

        # if A depends on randomness, i.e., sampled masks, it is fixed here
        self.measurement_likelihood.fix_state(x_0, fabric, eval=False)

        # sample timesteps
        t = self.timestep_sampler.get_sample(x_0.shape[0])

        # sample x_t given x_0
        x_0_target = x_0.clone()
        x_t = self.sample_x_t_given_x_0(x_0, t)

        # get measurement and pseudoinverse rec
        measurement = self.measurement_likelihood(x_0_target)
        pinvA_y = self.measurement_likelihood.mean_system_response.pinvA(measurement)

        # pick conditioning based on network info
        if ConditioningType[denoising_network.condition_type] == ConditioningType.PSEUDOINVERSE_RECONSTRUCTION:
            conditioning = pinvA_y
        
        elif ConditioningType[denoising_network.condition_type] == ConditioningType.MEASUREMENT:
            conditioning = measurement

        else:
            raise ValueError(f"ConditioningType {denoising_network.condition_type} not recognized.")

        # predict with (un)conditional denoising network
        output = denoising_network(x_t, conditioning, t)

        # derive the network's target based on its prediction type
        target = self._map_x_0_to_network_target(x_0_target, x_t, t, denoising_network.prediction_type)

        # compute loss and return
        loss = loss_fn(target, output, t)

        # log images when the epoch begins
        if batch_idx == 0 and log_assets:
            wandb.log({
                f"images/train/x_0": wandb.Image(min_max_scale(x_0_target)),
                f"images/train/x_t": wandb.Image(min_max_scale(x_t)),
                f"images/train/x_0_hat": wandb.Image(min_max_scale(
                    self._map_network_output_to_x_0(x_t, output, t, denoising_network.prediction_type))),
                f"images/train/y": wandb.Image(min_max_scale(measurement)),
                f"images/train/pinvA_y": wandb.Image(min_max_scale(pinvA_y))})

        return loss


    @torch.no_grad()
    def validation_step(self, fabric, batch_idx, x, denoising_network, log_imgs, log_prefix):

        # sample solutions for given x
        log = batch_idx == 0 and log_imgs
        x_0_hat = self.sample(fabric, x, denoising_network, self.n_sampling_timesteps, log, log_prefix)

        return x_0_hat


def map_01_to_m1p1(x):
    return (x - 0.5) * 2


def normalize(x):
    x = x - x.flatten(1).min(1)[0][(...,) + (None,) * (x.ndim - 1)]
    x = x / x.flatten(1).max(1)[0][(...,) + (None,) * (x.ndim - 1)]
    return x


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]