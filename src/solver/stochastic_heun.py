import torch

from .base import BaseSolver


class StochasticHeunSolver(BaseSolver):
    def __init__(self, n_sampling_timesteps: int):
        super(StochasticHeunSolver, self).__init__()
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
            t_star = t - dt
            dt = dt.repeat(B)

            with torch.no_grad():
                # Predictor step (Euler–Maruyama)
                x_t_star = sde.sample_x_t_minus_dt_given_x_t(
                    x_t,
                    t,
                    dt,
                    pinvA_y,
                    denoising_network,
                    is_last=(i == n_timesteps - 1),
                )
                if i != n_timesteps - 1:
                    # Corrector step (Refine using Heun method)
                    x_t_correction = sde.sample_x_t_minus_dt_given_x_t(
                        x_t_star,
                        t_star,
                        dt,
                        pinvA_y,
                        denoising_network,
                        is_last=(i == n_timesteps - 1),
                    )

                    # Heun update: combine predictor and corrector steps
                    x_t = 0.5 * (x_t + x_t_correction)

                else:
                    # Skip Heun step at last timestep
                    x_t = x_t_star

                if log:
                    x_t_traj.append(x_t[0])

        return x_t, x_t_traj
