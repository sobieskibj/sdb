import wandb
import torch
import matplotlib.pyplot as plt

from .base import BaseVarianceSchedule


class SDBI2SBSmoothVarianceSchedule(BaseVarianceSchedule):


    def __init__(self, beta_min: float, beta_max: float, log_assets: bool):
        super(SDBI2SBSmoothVarianceSchedule, self).__init__()

        self.beta_min = torch.tensor([beta_min])
        self.beta_max = torch.tensor([beta_max])
        if log_assets: self.log_plots()
    

    def beta(self, t):
        return - 4 * (self.beta_max - self.beta_min) * t.pow(2) + 4 * (self.beta_max - self.beta_min) * t + self.beta_min


    def s_sq(self, t):
        return - (4 / 3) * (self.beta_max - self.beta_min) * t.pow(3) + 2 * (self.beta_max - self.beta_min) * t.pow(2) + self.beta_min * t


    def s_bar_sq(self, t):
        return self.s_sq(1 - t)


    def sigma_sq(self, t):
        s_sq_t = self.s_sq(t)
        s_bar_sq_t = self.s_bar_sq(t)
        return (s_sq_t * s_bar_sq_t) / (s_sq_t + s_bar_sq_t)
    

    def d_sigma_sq(self, t):
        return self.beta(t) * (self.s_bar_sq(t) - self.s_sq(t)) / (self.s_bar_sq(t) + self.s_sq(t))


    def sigma_sq_T(self, T):
        return self.sigma_sq(T)


    def sigma_sq_t_over_sigma_sq_T(self, t):
        return self.s_sq(t) / (self.s_bar_sq(t) + self.s_sq(t))


    def d_sigma_sq_t_over_sigma_sq_T(self, t):
        return self.beta(t) / (self.s_bar_sq(t) + self.s_sq(t))


    def d_sigma_sq_t_over_sigma_sq_t(self, t):
        return self.beta(t) * (self.s_bar_sq(t) - self.s_sq(t)) / (self.s_bar_sq(t) * self.s_sq(t))


    def alpha(self, t):
        return self.s_bar_sq(t) / (self.s_bar_sq(t) + self.s_sq(t))


    def d_alpha(self, t):
        return - self.beta(t) / (self.s_bar_sq(t) + self.s_sq(t))


    def d_alpha_t_over_alpha_t(self, t):
        return - self.beta(t) / self.s_bar_sq(t)


    def sigma(self, t):
        pass


    def dsigma(self, t):
        pass


    def log_plots(self):
        t = torch.linspace(0, 1, 101)
        log_plot([t, self.beta(t)], "beta")
        log_plot([t, self.s_sq(t)], "s_sq")
        log_plot([t, self.s_bar_sq(t)], "s_bar_sq")
        log_plot([t, self.s_sq(t).sqrt()], "s")
        log_plot([t, self.s_bar_sq(t).sqrt()], "s_bar")
        log_plot([t, self.sigma_sq(t)], "sigma_sq")
        log_plot([t, self.sigma_sq(t).sqrt()], "sigma")
        log_plot([t, self.d_sigma_sq(t)], "d_sigma_sq")
        log_plot([t, self.d_sigma_sq(t).sqrt()], "d_sigma")
        log_plot([t, self.sigma_sq_T(t)], "sigma_sq_T")
        log_plot([t, self.sigma_sq_t_over_sigma_sq_T(t)], "sigma_sq_t_over_sigma_sq_T")
        log_plot([t, self.d_sigma_sq_t_over_sigma_sq_T(t)], "d_sigma_sq_t_over_sigma_sq_T")
        log_plot([t, self.d_sigma_sq_t_over_sigma_sq_t(t)], "d_sigma_sq_t_over_sigma_sq_t")
        log_plot([t, self.alpha(t)], "alpha")
        log_plot([t, self.d_alpha(t)], "d_alpha")
        log_plot([t, self.d_alpha_t_over_alpha_t(t)], "d_alpha_t_over_alpha_t")


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]


def log_plot(data, name):
    data = [e.numpy(force=True) for e in data]
    fig = plt.figure()
    plt.plot(*data)
    plt.grid()
    # wandb.log({f'misc/{name}': wandb.Plotly.make_plot_media(fig)})
    wandb.log({f'misc/{name}': wandb.Image(fig)})
    plt.close(fig)