
def mse(y, y_hat, t):
    residual = y - y_hat
    return (residual ** 2).mean()