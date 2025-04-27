import torch

def weight(t, k, exp_m1):
    return 1 / ((k / exp_m1) * (- torch.log(1 - t * exp_m1)) ** (1 - (1 / k)) * (1 / (1 - t * exp_m1)))


def inv_exp_weighed_l1(y, y_hat, t, k, exp_m1):
    residual = y - y_hat
    return (expand(weight(t, k, exp_m1), residual) * residual.abs()).mean()


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]