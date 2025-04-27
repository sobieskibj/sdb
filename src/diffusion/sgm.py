import wandb
import torch

from .base import BaseDiffusion
from network.base import PredictionType, ConditioningType
from timestep_sampler.base import BaseTimestepSampler
from sde.base import BaseLinearSDE
from utils.wandb import min_max_scale


class SGM(BaseDiffusion):
    def __init__(
        self,
        sde: BaseLinearSDE,
        timestep_sampler: BaseTimestepSampler,
        n_sampling_timesteps: int,
    ):
        super(SGM, self).__init__()

        self.sde = sde
        self.timestep_sampler = timestep_sampler
        self.n_sampling_timesteps = n_sampling_timesteps
        self.measurement_likelihood = self.sde.measurement_system

    def sample(self, fabric, x_0, denoising_network, log=False, log_prefix=""):
        # if A depends on randomness, i.e., sampled masks, it is fixed here
        self.fix_state(x_0, fabric, eval=True)

        # get path of timesteps and dts for sampling
        n_timesteps = self.n_sampling_timesteps
        timesteps, dts = self.timestep_sampler.get_path(n_timesteps)

        # extract sizes
        B, C, H, W = x_0.shape

        # get measurement
        y = self.sde.measurement_system(x_0)
        pinvA_y = self.sde.measurement_system.pinvA(y)

        # sample starting point
        T = self.timestep_sampler.get_T(x_0)
        x_T = self.sde.sample_x_T(pinvA_y, T)
        x_t = x_T.clone()

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
                x_t = self.sde.sample_x_t_minus_dt_given_x_t(
                    x_t,
                    t,
                    dt,
                    y,
                    pinvA_y,
                    denoising_network,
                    is_last=(i == n_timesteps - 1),
                )

                if log:
                    x_t_traj.append(x_t[0])

        log_dict = {
            f"{log_prefix}images/eval/x_0": wandb.Image(min_max_scale(x_0)),
            f"{log_prefix}images/eval/x_0_hat": wandb.Image(min_max_scale(x_t)),
            f"{log_prefix}images/eval/y": wandb.Image(min_max_scale(y)),
            f"{log_prefix}images/eval/pinvA_y": wandb.Image(min_max_scale(pinvA_y)),
        }

        if log:
            log_dict[f"{log_prefix}images/eval/x_t_traj"] = wandb.Image(
                min_max_scale(torch.stack(x_t_traj))
            )
            wandb.log(log_dict)

        return x_t

    def training_step(
        self, fabric, batch_idx, x_0, loss_fn, denoising_network, log_assets
    ):
        # if A depends on randomness, e.g., sampled masks, it is fixed here
        self.fix_state(x_0, fabric, eval=False)

        # sample timesteps
        t = self.timestep_sampler.get_sample(x_0.shape[0])

        # sample x_t given x_0 and y together with x_T = pinvA_y
        x_0_target = x_0.clone()
        x_t = self.sde.sample_x_t_given_x_0(x_0, t)

        # pick conditioning based on network info
        y = self.sde.measurement_system(x_0)
        pinvA_y = self.sde.measurement_system.pinvA(y)

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

        # predict with (un)conditional denoising network
        output = denoising_network(x_t, cond, t)

        # derive the network's target based on its prediction type
        target = self._map_x_0_to_network_target(
            x_0_target, x_t, t, denoising_network.prediction_type
        )

        # compute loss and return
        loss = loss_fn(target, output, t)

        # log images when the epoch begins
        if batch_idx == 0 and log_assets:
            wandb.log(
                {
                    "images/train/x_0": wandb.Image(min_max_scale(x_0_target)),
                    "images/train/x_t": wandb.Image(min_max_scale(x_t)),
                    "images/train/x_0_hat": wandb.Image(
                        min_max_scale(
                            self._map_network_output_to_x_0(
                                x_t, output, t, denoising_network.prediction_type
                            )
                        )
                    ),
                }
            )

        return loss

    @torch.no_grad()
    def validation_step(
        self, fabric, batch_idx, x, denoising_network, log_imgs, log_prefix
    ):
        # sample solutions for given x
        log = batch_idx == 0 and log_imgs
        x_0_hat = self.sample(fabric, x, denoising_network, log, log_prefix)

        return x_0_hat

    def fix_state(self, x_0, fabric, eval):
        self.sde.fix_state(x_0, fabric, eval)

    def _map_network_output_to_x_0(
        self, x_t, t, output, prediction_type: PredictionType
    ):
        return self.sde._map_network_output_to_x_0(x_t, t, output, prediction_type)

    def _map_x_0_to_network_target(self, x_0, x_t, t, prediction_type: PredictionType):
        return self.sde._map_x_0_to_network_target(x_0, x_t, t, prediction_type)


def normalize(x):
    x = x - x.flatten(1).min(1)[0][(...,) + (None,) * (x.ndim - 1)]
    x = x / x.flatten(1).max(1)[0][(...,) + (None,) * (x.ndim - 1)]
    return x
