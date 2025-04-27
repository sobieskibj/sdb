import torch

def random_rotate_90(img, p):
    """
    Transformation inspired by GOUB source code, which randomly rotates an image by 90 degrees.
    """
    if torch.rand((1,)) < p:
        return img.transpose(2, 1)
    else:
        return img