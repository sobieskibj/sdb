import torch.nn.functional as F


def pad_tensor_to_square(img_tensor):
    _, h, w = img_tensor.shape  # (C, H, W)
    diff = abs(h - w)

    if h < w:
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        padding = (0, 0, pad_top, pad_bottom)  # (left, right, top, bottom)
    else:
        pad_left = diff // 2
        pad_right = diff - pad_left
        padding = (pad_left, pad_right, 0, 0)

    return F.pad(img_tensor, padding, mode="constant", value=0)
