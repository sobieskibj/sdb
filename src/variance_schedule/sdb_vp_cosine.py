import wandb
import torch
import matplotlib.pyplot as plt

from .base import BaseVarianceSchedule


class SDBVPCosineVarianceSchedule(BaseVarianceSchedule):


    def __init__(self, log_images: bool, s: float):
        super(SDBVPCosineVarianceSchedule, self).__init__()

        self.pi = torch.tensor([torch.pi])
        self.s = s
        if log_images: self.log_plots()


    def sigma_sq(self, t):
        return 1 - (((t + self.s) * self.pi / 2).cos().square() / \
                    (self.s * self.pi / 2).cos().square())
    

    def d_sigma_sq(self, t):
        return (self.pi / 2) * (self.pi * (t + self.s)).sin() / (self.s * self.pi / 2).cos().square()


    def sigma_sq_T(self, T, from_t=False):
        if from_t: T = torch.ones_like(T)
        return T - (self.pi * self.s / 2).tan().square()


    def sigma_sq_t_over_sigma_sq_T(self, t):
        return self.sigma_sq(t) / self.sigma_sq_T(t, True)


    def d_sigma_sq_t_over_sigma_sq_T(self, t):
        return self.d_sigma_sq(t) / self.sigma_sq_T(t, True)


    def d_sigma_sq_t_over_sigma_sq_t(self, t):
        return self.d_sigma_sq(t) / self.sigma_sq(t)


    def alpha(self, t):
        return (((t + self.s) * self.pi / 2).cos().square() / \
                    (self.s * self.pi / 2).cos().square()).sqrt()


    def d_alpha(self, t):
        return - (self.pi / 2) * (self.pi * (t + self.s) / 2).sin() / (self.s * self.pi / 2).cos()


    def d_alpha_t_over_alpha_t(self, t):
        return - self.pi * (self.pi * (t + self.s) / 2).tan() / 2


    def sigma(self, t):
        pass


    def dsigma(self, t):
        pass


    def log_plots(self):
        t = torch.linspace(0, 1, 101)
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
    wandb.log({f'misc/{name}': wandb.Image(fig)})
    plt.close(fig)