import torch
from pathlib import Path
import torch.nn.functional as F
from torchvision.io import read_image

from .base import BaseMeanSystemResponse


class MaskMeanSystemResponse(BaseMeanSystemResponse):
    def __init__(self, path_masks: str, add_mask: bool):
        super(MaskMeanSystemResponse, self).__init__()

        self.masks = self.get_masks(Path(path_masks))
        self.add_mask = add_mask
        self.was_eval = False

    def get_masks(self, path):
        # find all .png files representing masks sorted increasingly by id
        paths = sorted(path.glob("*.png"), key=lambda x: int(x.stem))

        # read all masks into memory
        masks = [read_image(p) for p in paths]
        masks = torch.stack(masks) / 255.0

        return masks

    def A(self, x):
        # get shape of batch and its elements
        B, C, H, W = x.shape

        if self.masks.device != x.device:
            # move to proper device if needed
            self.masks = self.masks.to(x.device)

        # choose masks according to indices saved when the state was reset
        masks = self.masks[self.idx]

        # change if mask shape does not agree
        if (H, W) != masks.shape[-2:]:
            # if not, resize
            masks = F.interpolate(masks, (H, W), mode="nearest").to(x.device)

        return masks * x + (1.0 - masks) if self.add_mask else masks * x

    def AT(self, y):
        return self.A(y)

    def pinvATA(self, z):
        return self.A(z)

    def ImA(self, x):
        # get masks and return the inverse masking of x
        masks = self.masks[self.idx]
        return (1.0 - masks) * x

    def reset_random_state(self, x, fabric, eval):
        if not eval:
            # if not evaluating, sample new indices for random mask selection
            self.idx = fabric.to_device(
                torch.randint(high=len(self.masks), size=(x.shape[0],))
            )

        else:
            # otherwise, next masks should be chosen according to the true ordering

            # provide process rank and world size to preserve correct ordering
            rank = fabric.global_rank
            world_size = fabric.world_size

            # get current total batch size based on number of processes
            total_batch_size = x.shape[0] * world_size

            if self.was_eval:
                # during evaluation, just move indices along the list
                idx = fabric.to_device(
                    torch.arange(
                        start=self.idx[-1] + world_size,
                        end=self.idx[-1] + world_size + total_batch_size,
                        step=world_size,
                    )
                )

            else:
                # when starting evaluation, begin from zero
                idx = fabric.to_device(
                    torch.arange(start=rank, end=total_batch_size, step=world_size)
                )

            # modulo because length of masks is limited
            self.idx = idx % len(self.masks)

        # save eval state to condition on in the next function call
        self.was_eval = eval
