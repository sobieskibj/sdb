import torch

from .base import BaseSolver


class EulerSolver(BaseSolver):
    def __init__(self, n_sampling_timesteps: int):
        super(EulerSolver, self).__init__()

        self.n_sampling_timesteps = n_sampling_timesteps

    def sample(self, x_t, pinvA_y, sde, timestep_sampler, denoising_network, log):
        # extract shapes
        B, C, H, W = x_t.shape

        # make list to collect x_t trajectory
        if log:
            x_t_traj = [x_t[0]]
        else:
            x_t_traj = None

        # get path of timesteps and dts for sampling
        n_timesteps = self.n_sampling_timesteps
        timesteps, dts = timestep_sampler.get_path(n_timesteps)

        for i, (t, dt) in enumerate(zip(timesteps, dts)):
            # get time increment and timestep
            t = t.repeat(B)
            dt = dt.repeat(B)

            # predict with the network
            with torch.no_grad():
                # sample at timestep t except the last one
                x_t = sde.sample_x_t_minus_dt_given_x_t(
                    x_t,
                    t,
                    dt,
                    pinvA_y,
                    denoising_network,
                    is_last=(i == n_timesteps - 1),
                )

                if log:
                    x_t_traj.append(x_t[0])

        return x_t, x_t_traj
