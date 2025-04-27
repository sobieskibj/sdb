
def weighed_mse(y, y_hat, t):
    residual = y - y_hat
    residual = residual / expand(t, residual).sqrt()
    return (residual ** 2).mean()


def expand(input, target):
    """Adds dimension to input to match number of dimensions in target"""
    return input[(...,) + (None,) * (target.ndim - input.ndim)]