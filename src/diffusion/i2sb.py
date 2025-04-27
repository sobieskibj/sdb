"""Ported from https://github.com/NVlabs/I2SB/tree/master and further adapted."""
import enum
import wandb
import torch
from tqdm import tqdm

from .base import BaseDiffusion
from utils.wandb import min_max_scale
from network.base import ConditioningType
from measurement_model.base import BaseLinearModelWithGaussianNoise
from mean_system_response.mask import MaskMeanSystemResponse


class EndpointType(enum.Enum):
    PSEUDOINVERSE_RECONSTRUCTION = "PSEUDOINVERSE_RECONSTRUCTION"
    MEASUREMENT = "MEASUREMENT"


class I2SB(BaseDiffusion):


    def __init__(
            self, 
            measurement_likelihood: BaseLinearModelWithGaussianNoise, 
            endpoint_type: EndpointType,
            beta_max: float,
            interval: int,
            n_sampling_timesteps: int,
            ot_ode:  bool,
        ):
        
        super(I2SB, self).__init__()

        self.measurement_likelihood = measurement_likelihood
        self.endpoint_type = endpoint_type
        self.betas = get_beta_schedule(interval=interval, linear_end=beta_max / interval)
        self.interval = interval
        self.n_sampling_timesteps = n_sampling_timesteps
        self.ot_ode = ot_ode
        self.set_coefficients()
        

    def set_coefficients(self):

        # compute the required coefficients
        std_fwd = torch.sqrt(torch.cumsum(self.betas, dim=0))
        std_bwd = torch.sqrt(
            torch.flip(
                torch.cumsum(
                    torch.flip(self.betas, dims=(0,)), dim=0), 
                dims=(0,)))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = torch.sqrt(var)

        # save them as attributes
        self.std_fwd = std_fwd.float()
        self.std_bwd = std_bwd.float()
        self.std_sb  = std_sb.float()
        self.mu_x0 = mu_x0.float()
        self.mu_x1 = mu_x1.float()


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

        # obtain target for the network
        target = self.get_target(t, x_0, x_t)

        # compute loss and return
        loss = loss_fn(target, output, t)

        # log images when the epoch begins
        if batch_idx == 0 and log_assets:
            wandb.log({
                f"images/train/x_0": wandb.Image(min_max_scale(x_0)),
                f"images/train/x_0_hat": wandb.Image(
                    min_max_scale(
                            self.compute_pred_x0(t, x_t, output)
                        )
                    ),
                f"images/train/x_t": wandb.Image(min_max_scale(x_t)),
                f"images/train/y": wandb.Image(min_max_scale(measurement)),
                f"images/train/pinvA_y": wandb.Image(min_max_scale(pinvA_y))})

        return loss


    def sample_x_t_given_x_0_x_T(self, x_0, x_T, y_shape = None):
        
        # sample timesteps from the given interval
        t = torch.randint(0, self.interval, (x_0.shape[0],))

        # sample from q(x_t | x_0, x_T)
        x_t = self.q_sample(t, x_0, x_T)

        return x_t, t
    

    def q_sample(self, t, x0, x1):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[t],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[t],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[t], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1

        if not self.ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)

        return xt.detach()


    def get_target(self, t, x0, xt):
        """ Eq 12 """
        std_fwd = self.get_std_fwd(t, xdim=x0.shape[1:])
        target = (xt - x0) / std_fwd
        return target.detach()


    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)


    def compute_pred_x0(self, t, xt, net_out):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.get_std_fwd(t, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        return pred_x0


    @torch.no_grad()
    def validation_step(self, fabric, batch_idx, x, denoising_network, log_imgs, log_prefix):

        # sample solutions for given x
        log = batch_idx == 0 and log_imgs
        x_0_hat = self.sample(fabric, x, denoising_network, log, log_prefix)

        return x_0_hat
    

    def sample(self, fabric, x_0, denoising_network, log=False, log_prefix=""):

        # if A depends on randomness, i.e., sampled masks, it is fixed here
        self.measurement_likelihood.fix_state(x_0, fabric, eval=True)

        # get final state x_T which is either measurement or pinv
        measurement = self.measurement_likelihood(x_0)
        pinvA_y = self.measurement_likelihood.mean_system_response.pinvA(measurement)

        if EndpointType[self.endpoint_type] == EndpointType.MEASUREMENT:
            x_T = measurement
        
        elif EndpointType[self.endpoint_type] == EndpointType.PSEUDOINVERSE_RECONSTRUCTION:
            x_T = pinvA_y

        # for i2sb and inpainting, mask region is filled with noise at T
        # if isinstance(self.measurement_likelihood.mean_system_response, MaskMeanSystemResponse):
        #     # sample noise for the entire image and apply 1 - mask
        #     noise = self.measurement_likelihood.mean_system_response.ImA(torch.randn_like(x_T))

        #     # add to x_T in place of mask
        #     x_T = self.measurement_likelihood.mean_system_response.A(x_T) + noise

        # pick conditioning based on network info
        if ConditioningType[denoising_network.condition_type] == ConditioningType.PSEUDOINVERSE_RECONSTRUCTION:
            conditioning = pinvA_y
        
        elif ConditioningType[denoising_network.condition_type] == ConditioningType.MEASUREMENT:
            conditioning = measurement

        # define mapping from network's output to x_0 estimate at t
        def pred_x0_fn(xt, t):
            out = denoising_network(xt, conditioning, t)
            return self.compute_pred_x0(t, xt, out)

        nfe = self.n_sampling_timesteps
        if nfe == len(self.betas):
            nfe = nfe - 1
        assert 0 < nfe < self.interval == len(self.betas)
        steps = space_indices(self.interval, nfe+1)

        steps = steps[::-1]
        xt = x_T.clone()
        if log: x_t_traj = [xt[0]]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1)
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=self.ot_ode)
                
            if log: x_t_traj.append(xt[0])

        if log:
            wandb.log({
                f"{log_prefix}images/eval/x_0": wandb.Image(min_max_scale(x_0)),
                f"{log_prefix}images/eval/x_0_hat": wandb.Image(min_max_scale(xt)),
                f"{log_prefix}images/eval/y": wandb.Image(min_max_scale(measurement)),
                f"{log_prefix}images/eval/x_t_traj": wandb.Image(min_max_scale(torch.stack(x_t_traj))),
                f"{log_prefix}images/eval/pinvA_y": wandb.Image(min_max_scale(pinvA_y))})
        
        return xt
    

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
            """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

            assert nprev < n
            std_n     = self.std_fwd[n]
            std_nprev = self.std_fwd[nprev]
            std_delta = (std_n**2 - std_nprev**2).sqrt()

            mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

            xt_prev = mu_x0 * x0 + mu_xn * x_n
            if not ot_ode and nprev > 0:
                xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

            return xt_prev


    def _map_network_output_to_x_0(self):
        pass


    def _map_x_0_to_network_target(self):
        pass


def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps


def get_beta_schedule(interval, linear_start=1e-4, linear_end=2e-2):
    return  torch.linspace(linear_start ** 0.5, linear_end ** 0.5, interval, dtype=torch.float64) ** 2