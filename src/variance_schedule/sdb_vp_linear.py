import wandb
import torch
import matplotlib.pyplot as plt

from .base import BaseVarianceSchedule


class SDBVPLinearVarianceSchedule(BaseVarianceSchedule):


    def __init__(self, log_assets: bool):
        super(SDBVPLinearVarianceSchedule, self).__init__()

        if log_assets: self.log_plots()


    def sigma_sq(self, t):
        """\sigma^2_t"""
        return t
    

    def d_sigma_sq(self, t):
        """d\sigma^2_t"""
        return torch.ones_like(t)


    def sigma_sq_T(self, t):
        """\sigma^2_T"""
        return torch.ones_like(t)


    def sigma_sq_t_over_sigma_sq_T(self, t):
        return self.sigma_sq(t) / self.sigma_sq_T(t)


    def d_sigma_sq_t_over_sigma_sq_T(self, t):
        return self.d_sigma_sq(t) / self.sigma_sq_T(t)


    def d_sigma_sq_t_over_sigma_sq_t(self, t):
        """\frac{d\sigma^2_t}{\sigma^2_t}"""
        return self.d_sigma_sq(t) / self.sigma_sq(t)


    def alpha(self, t):
        """\alpha_t"""
        return (1 - self.sigma_sq(t)).sqrt()


    def d_alpha(self, t):
        """d\alpha_t"""
        return - self.d_sigma_sq(t) / (2 * self.alpha(t))


    def d_alpha_t_over_alpha_t(self, t):
        """\frac{d\alpha_t}{\alpha_t}"""
        return - self.d_sigma_sq(t) / (2 * self.alpha(t).square())


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
    # wandb.log({f'misc/{name}': wandb.Plotly.make_plot_media(fig)})
    wandb.log({f'misc/{name}': wandb.Image(fig)})
    plt.close(fig)