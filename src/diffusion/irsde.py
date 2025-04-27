import enum
import wandb
import torch
from tqdm import tqdm

from sde.base import BaseIRSDE
from .base import BaseDiffusion
from utils.wandb import min_max_scale
from network.base import PredictionType, ConditioningType
from measurement_model.base import BaseLinearModelWithGaussianNoise


class EndpointType(enum.Enum):
    PSEUDOINVERSE_RECONSTRUCTION = "PSEUDOINVERSE_RECONSTRUCTION"
    MEASUREMENT = "MEASUREMENT"


class ReverseType(enum.Enum):
    SDE = "SDE"
    ODE = "ODE"


class IRSDE(BaseDiffusion):


    def __init__(
            self, 
            measurement_likelihood: BaseLinearModelWithGaussianNoise, 
            endpoint_type: EndpointType,
            reverse_type: ReverseType,
            n_sampling_timesteps: int,
            sde: BaseIRSDE,
        ):
        
        super(IRSDE, self).__init__()

        self.measurement_likelihood = measurement_likelihood
        self.endpoint_type = endpoint_type
        self.reverse_type = reverse_type
        self.n_sampling_timesteps = n_sampling_timesteps
        self.sde = sde
        

    def sample_x_t_given_x_0_x_T(self, x_0, x_T, y_shape = None):
        return self.sde.generate_random_states(x_0, x_T)


    def sample(self, fabric, x_0, denoising_network, log=False, log_prefix="", n_sampling_timesteps=None):

        # if A depends on randomness, i.e., sampled masks, it is fixed here
        self.measurement_likelihood.fix_state(x_0, fabric, eval=True)

        # get final state x_T which is either measurement or pinv
        measurement = self.measurement_likelihood(x_0)
        pinvA_y = self.measurement_likelihood.mean_system_response.pinvA(measurement)

        if EndpointType[self.endpoint_type] == EndpointType.MEASUREMENT:
            x_T = measurement
        
        elif EndpointType[self.endpoint_type] == EndpointType.PSEUDOINVERSE_RECONSTRUCTION:
            x_T = pinvA_y

        # pick final timestep and endpoint
        # n_sampling_timesteps=1
        T = self.sde.T if self.n_sampling_timesteps is None else self.n_sampling_timesteps
        B, C, H, W = x_T.shape
        x = self.sde.noise_state(x_T)

        # set endpoint in sde
        self.sde.set_mu(x_T.clone())

        if log: x_t_traj = [x[0]]

        # run chosen reverse sde starting from x_T
        if ReverseType[self.reverse_type] == ReverseType.SDE:
            # default follows reverse sde 
            reverse_fn = self.sde.reverse_sde_step

        elif ReverseType[self.reverse_type] == ReverseType.ODE:
            # ode follows the pf ode
            reverse_fn = self.sde.reverse_ode_step
        
        # iterate over timesteps
        for t in tqdm(reversed(range(1, T + 1))):
            noise = denoising_network(x, x_T, t)
            score = - noise / self.sde.sigma_bar(t)
            x = reverse_fn(x, score, t)
            if log: x_t_traj.append(x[0])

        if log:
            wandb.log({
                f"{log_prefix}images/eval/x_0": wandb.Image(min_max_scale(x_0)),
                f"{log_prefix}images/eval/x_0_hat": wandb.Image(min_max_scale(x)),
                f"{log_prefix}images/eval/y": wandb.Image(min_max_scale(measurement)),
                f"{log_prefix}images/eval/x_t_traj": wandb.Image(min_max_scale(torch.stack(x_t_traj))),
                f"{log_prefix}images/eval/pinvA_y": wandb.Image(min_max_scale(pinvA_y))})
        
        return x


    def _map_x_0_to_network_target(self, x_0, x_t, t, eps, prediction_type):
        pass


    def _map_network_output_to_x_0(self, x_t, output, prediction_type: PredictionType):

        if PredictionType[prediction_type] == PredictionType.MEAN:
            x_0_hat = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            x_0_hat = output

        elif PredictionType[prediction_type] == PredictionType.EPSILON: 
            x_0_hat = x_t - output

        elif PredictionType[prediction_type] == PredictionType.SCORE:
            x_0_hat = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")
        
        return x_0_hat


    def _map_network_output_to_true_output(self, output, t, prediction_type):

        if PredictionType[prediction_type] == PredictionType.MEAN:
            output = ...

        elif PredictionType[prediction_type] == PredictionType.X_0:
            output = ...

        elif PredictionType[prediction_type] == PredictionType.EPSILON: 
            # in IR-SDE, the network predicts the added noise but is trained
            # on the task of score matching with this noise
            # hence, output is the score from the estimated noise
            output = - output / self.sde.sigma_bar(t)

        elif PredictionType[prediction_type] == PredictionType.SCORE:
            output = ...

        else:
            raise ValueError(f"Invalid value ({prediction_type}) for prediction_type")
        
        return output


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

        # set endpoint for sde
        self.sde.set_mu(x_T)

        # sample x_t given x_0 and x_T
        t, x_t = self.sample_x_t_given_x_0_x_T(x_0, x_T)

        # pick conditioning based on network info
        if ConditioningType[denoising_network.condition_type] == ConditioningType.PSEUDOINVERSE_RECONSTRUCTION:
            conditioning = pinvA_y
        
        elif ConditioningType[denoising_network.condition_type] == ConditioningType.MEASUREMENT:
            conditioning = measurement

        else:
            raise ValueError(f"ConditioningType {denoising_network.condition_type} not recognized.")
        
        # predict with (un)conditional denoising network
        output = denoising_network(x_t, conditioning, t.squeeze())

        # map the output based on what IRSDE requires
        output = self._map_network_output_to_true_output(output, t, denoising_network.prediction_type)

        # compute loss and return
        loss = loss_fn(
            self.sde.reverse_sde_step_mean(x_t, output, t),
            self.sde.reverse_optimum_step(x_t, x_0, t), t)

        # log images when the epoch begins
        if batch_idx == 0 and log_assets:
            wandb.log({
                f"images/train/x_0": wandb.Image(min_max_scale(x_0)),
                f"images/train/x_t": wandb.Image(min_max_scale(x_t)),
                f"images/train/y": wandb.Image(min_max_scale(measurement)),
                f"images/train/pinvA_y": wandb.Image(min_max_scale(pinvA_y))})

        return loss


    @torch.no_grad()
    def validation_step(self, fabric, batch_idx, x, denoising_network, log_imgs, log_prefix):

        # sample solutions for given x
        log = batch_idx == 0 and log_imgs
        x_0_hat = self.sample(fabric, x, denoising_network, log, log_prefix)

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