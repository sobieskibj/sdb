"""Ported from https://github.com/alexzhou907/DDBM/tree/main and further adapted."""
import enum
import wandb
import torch
import numpy as np

from .base import BaseDiffusion
from utils.wandb import min_max_scale
from network.base import ConditioningType
from measurement_model.base import BaseLinearModelWithGaussianNoise
from timestep_sampler.base import BaseTimestepSampler


class EndpointType(enum.Enum):
    PSEUDOINVERSE_RECONSTRUCTION = "PSEUDOINVERSE_RECONSTRUCTION"
    MEASUREMENT = "MEASUREMENT"


class DDBM(BaseDiffusion):
    """
    Implements the VP version of DDBM.
    """

    def __init__(
            self, 
            measurement_likelihood: BaseLinearModelWithGaussianNoise, 
            endpoint_type: EndpointType,
            beta_d: float, 
            beta_min: float,
            sigma_min: float,
            sigma_max: float,
            guidance_scale: float,
            timestep_sampler: BaseTimestepSampler,
            n_sampling_timesteps: int,
        ):
        
        super(DDBM, self).__init__()

        self.measurement_likelihood = measurement_likelihood
        self.endpoint_type = endpoint_type

        self.beta_d = beta_d
        self.beta_min = beta_min
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.guidance_scale = guidance_scale

        self.timestep_sampler = timestep_sampler
        self.n_sampling_timesteps = n_sampling_timesteps

        self.set_coefs()


    def set_coefs(self):
        vp_snr_sqrt_reciprocal = lambda t: (np.e ** (0.5 * self.beta_d * (t ** 2) + self.beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (self.beta_min + self.beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)
        logs = lambda t: -0.25 * t ** 2 * (self.beta_d) - 0.5 * t * self.beta_min
        std =  lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        logsnr = lambda t :  - 2 * torch.log(vp_snr_sqrt_reciprocal(t))
        logsnr_T = logsnr(torch.as_tensor(self.sigma_max))
        logs_T = logs(torch.as_tensor(self.sigma_max))

        self.vp_snr_sqrt_reciprocal = vp_snr_sqrt_reciprocal
        self.vp_snr_sqrt_reciprocal_deriv = vp_snr_sqrt_reciprocal_deriv
        self.s = s
        self.s_deriv = s_deriv
        self.logs = logs
        self.std = std
        self.logsnr = logsnr
        self.logsnr_T = logsnr_T
        self.logs_T = logs_T


    def training_step(self, fabric, batch_idx, x_0, loss_fn, denoising_network, log_assets):

        # if A depends on randomness, i.e., sampled masks, it is fixed here
        self.measurement_likelihood.fix_state(x_0, fabric, eval=False)

        # get final state x_T which is either measurement or pinv
        measurement = self.measurement_likelihood(x_0)
        pinvA_y = self.measurement_likelihood.mean_system_response.pinvA(measurement)

        if EndpointType[self.endpoint_type] == EndpointType.MEASUREMENT:
            x_T = measurement
        
        elif EndpointType[self.endpoint_type] == EndpointType.PSEUDOINVERSE_RECONSTRUCTION:
            x_T = pinvA_y

        # pick conditioning based on network info
        if ConditioningType[denoising_network.condition_type] == ConditioningType.PSEUDOINVERSE_RECONSTRUCTION:
            conditioning = pinvA_y
        
        elif ConditioningType[denoising_network.condition_type] == ConditioningType.MEASUREMENT:
            conditioning = measurement

        else:
            raise ValueError(f"ConditioningType {denoising_network.condition_type} not recognized.")
        
        # sample x_t given x_0 and x_T
        x_t, t = self.sample_x_t_given_x_0_x_T(x_0, x_T)

        # predict with (un)conditional denoising network
        output = denoising_network(x_t, conditioning, t)

        # compute loss and return
        # DDBM predicts x_0 by default, hence we hardcode it here
        loss = loss_fn(x_0.detach(), output, t)

        # log images when the epoch begins
        if batch_idx == 0 and log_assets:
            wandb.log({
                f"images/train/x_0": wandb.Image(min_max_scale(x_0)),
                f"images/train/x_0_hat": wandb.Image(min_max_scale(output)),
                f"images/train/x_t": wandb.Image(min_max_scale(x_t)),
                f"images/train/y": wandb.Image(min_max_scale(measurement)),
                f"images/train/pinvA_y": wandb.Image(min_max_scale(pinvA_y))})

        return loss


    def sample_x_t_given_x_0_x_T(self, x_0, x_T):

        noise = torch.randn_like(x_0) 
        sigmas = self.timestep_sampler.get_sample(x_0.shape[0])

        def bridge_sample(x_0, x_T, t):

            t = expand(t, x_0)

            logsnr_t = vp_logsnr(t, self.beta_d, self.beta_min)
            logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
            logs_t = vp_logs(t, self.beta_d, self.beta_min)
            logs_T = vp_logs(self.sigma_max, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = - torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()
            
            samples = a_t * x_T + b_t * x_0 + std_t * noise

            return samples
        
        return bridge_sample(x_0, x_T, sigmas), sigmas


    @torch.no_grad()
    def validation_step(self, fabric, batch_idx, x, denoising_network, log_imgs, log_prefix):

        # sample solutions for given x
        log = batch_idx == 0 and log_imgs
        x_0_hat = self.sample(fabric, x, denoising_network, log, log_prefix)

        return x_0_hat
    

    def sample(self, fabric, x_0, denoising_network, log=False, log_prefix="", n_timesteps=None):

        # if A depends on randomness, i.e., sampled masks, it is fixed here
        self.measurement_likelihood.fix_state(x_0, fabric, eval=True)

        # get final state x_T which is either measurement or pinv
        measurement = self.measurement_likelihood(x_0)
        pinvA_y = self.measurement_likelihood.mean_system_response.pinvA(measurement)

        if EndpointType[self.endpoint_type] == EndpointType.MEASUREMENT:
            x_T = measurement
        
        elif EndpointType[self.endpoint_type] == EndpointType.PSEUDOINVERSE_RECONSTRUCTION:
            x_T = pinvA_y

        # pick conditioning based on network info
        if ConditioningType[denoising_network.condition_type] == ConditioningType.PSEUDOINVERSE_RECONSTRUCTION:
            conditioning = pinvA_y
        
        elif ConditioningType[denoising_network.condition_type] == ConditioningType.MEASUREMENT:
            conditioning = measurement

        # extract sizes
        B, C, H, W = x_0.shape

        # set xt as x_T and optionally add to trajectory log
        x_t = x_T.clone()
        if log: x_t_traj = [x_t[0]]

        # get timesteps path
        n_timesteps = self.n_sampling_timesteps if n_timesteps is None else n_timesteps
        timesteps, dts = self.timestep_sampler.get_path(n_timesteps)

        for i, (t, dt) in enumerate(zip(timesteps, dts)):
            
            # get time increment and timestep
            t = t.repeat(B)
            dt = dt.repeat(B)

            # predict with the network
            with torch.no_grad():

                # sample at timestep t, no noise is added at the last step
                x_t = self.sample_x_t_minus_dt_given_x_t(x_t, t, dt, denoising_network, conditioning, last=(i == n_timesteps - 1))

            if log: x_t_traj.append(x_t[0])

        if log:
            wandb.log({
                f"{log_prefix}images/eval/x_0": wandb.Image(min_max_scale(x_0)),
                f"{log_prefix}images/eval/x_0_hat": wandb.Image(min_max_scale(x_t)),
                f"{log_prefix}images/eval/y": wandb.Image(min_max_scale(measurement)),
                f"{log_prefix}images/eval/x_t_traj": wandb.Image(min_max_scale(torch.stack(x_t_traj))),
                f"{log_prefix}images/eval/pinvA_y": wandb.Image(min_max_scale(pinvA_y))})
        
        return x_t
    

    def sample_x_t_minus_dt_given_x_t(self, x_t, t, dt, denoising_network, x_T, last=False):
        std_t = expand(self.std(t), x_t)
        logsnr_t = expand(self.logsnr(t), x_t)
        logs_t = expand(self.logs(t), x_t)
        logs_T = expand(self.logs_T, x_t)
        s_t_deriv = expand(self.s_deriv(t), x_t)
        sigma_t = expand(self.vp_snr_sqrt_reciprocal(t), x_t)
        sigma_t_deriv = expand(self.vp_snr_sqrt_reciprocal_deriv(t), x_t)
        dt = expand(dt, x_t)

        denoised = denoising_network(x_t, x_T, t)

        a_t = (self.logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = - torch.expm1(self.logsnr_T - logsnr_t) * logs_t.exp()

        mu_t = a_t * x_T + b_t * denoised 
        
        grad_logq = - (x_t - mu_t) / std_t ** 2 / -torch.expm1(self.logsnr_T - logsnr_t)
        grad_logpxTlxt = - (x_t - torch.exp(logs_t - logs_T) * x_T) / std_t ** 2 / torch.expm1(logsnr_t - self.logsnr_T)

        f = s_t_deriv * (-logs_t).exp() * x_t
        g2 = 2 * (logs_t).exp() ** 2 * sigma_t * sigma_t_deriv 

        d = f - g2 * (grad_logq - self.guidance_scale * grad_logpxTlxt)

        g_dt_sq_eps = 0. if last else torch.randn_like(x_t) * (dt ** 0.5) * g2.sqrt()

        return x_t - d * dt + g_dt_sq_eps


    def _map_network_output_to_x_0(self):
        pass


    def _map_x_0_to_network_target(self):
        pass


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]


def vp_logsnr(t, beta_d, beta_min):
    t = torch.as_tensor(t)
    return - torch.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)


def vp_logs(t, beta_d, beta_min):
    t = torch.as_tensor(t)
    return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min