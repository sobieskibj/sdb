import torch.nn.functional as F

def l1(y, y_hat, t):
    return F.l1_loss(y_hat, y)