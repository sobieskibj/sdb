import wandb
import torch

from .base import BaseDiffusion
from network.base import PredictionType
from timestep_sampler.base import BaseTimestepSampler
from sde.base import BaseLinearSDE
from solver.base import BaseSolver
from utils.wandb import min_max_scale


class SDB(BaseDiffusion):
    def __init__(
        self,
        sde: BaseLinearSDE,
        timestep_sampler: BaseTimestepSampler,
        solver: BaseSolver,
    ):
        super(SDB, self).__init__()

        self.sde = sde
        self.timestep_sampler = timestep_sampler
        self.solver = solver
        self.measurement_likelihood = self.sde.measurement_system

    def sample(self, fabric, x_0, denoising_network, log=False, log_prefix=""):
        # if A depends on randomness, i.e., sampled masks, it is fixed here
        self.fix_state(x_0, fabric, eval=True)

        # get measurement
        y = self.sde.measurement_system(x_0)

        # sample starting point based on pseudoinverse reconstruction
        pinvA_y = self.sde.measurement_system.pinvA(y)
        T = self.timestep_sampler.get_T(pinvA_y)
        x_T = self.sde.sample_x_T(pinvA_y, T)
        x_t = x_T.clone()

        # sample with solver
        x_t, x_t_traj = self.solver.sample(
            x_t,
            pinvA_y,
            self.sde,
            self.timestep_sampler,
            denoising_network,
            log,
        )

        if log:
            wandb.log(
                {
                    f"{log_prefix}images/eval/x_0": wandb.Image(min_max_scale(x_0)),
                    f"{log_prefix}images/eval/x_0_hat": wandb.Image(min_max_scale(x_t)),
                    f"{log_prefix}images/eval/y": wandb.Image(min_max_scale(y)),
                    f"{log_prefix}images/eval/x_t_traj": wandb.Image(
                        min_max_scale(torch.stack(x_t_traj))
                    ),
                    f"{log_prefix}images/eval/pinvA_y": wandb.Image(
                        min_max_scale(pinvA_y)
                    ),
                }
            )

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
        x_t, x_T, y = self.sde.sample_x_t_given_x_0(x_0, t)

        # predict with (un)conditional denoising network
        output = denoising_network(x_t, x_T, t)

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
                                x_t, t, output, denoising_network.prediction_type
                            )
                        )
                    ),
                    "images/train/y": wandb.Image(min_max_scale(y)),
                    "images/train/pinvA_y": wandb.Image(min_max_scale(x_T)),
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
