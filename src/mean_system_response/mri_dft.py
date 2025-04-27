import torch
import torch.nn.functional as F

from .base import BaseMeanSystemResponse


class MRIDFTMeanSystemResponse(BaseMeanSystemResponse):
    def __init__(self, path_masks: str):
        super(MRIDFTMeanSystemResponse, self).__init__()

        self.masks = torch.load(path_masks).unsqueeze(1)
        res = self.masks[0].shape[-2]
        self.shape = (res, res)
        self.was_eval = False

    def A(self, x):
        """
        Forward model using real FFT with symmetry-aware reduction.
        Returns real-valued measurement vector.
        """
        if self.masks.device != x.device:
            # move to proper device if needed
            self.masks = self.masks.to(x.device)

        # choose masks according to indices saved when the state was reset
        mask = self.masks[self.idx]

        X = torch.fft.rfft2(x)  # shape (H, W//2+1), complex

        X_masked = X * mask

        # Stack real and imaginary parts from reduced (conjugate-symmetric) domain
        y = torch.cat(
            [X_masked.real, X_masked.imag],
            dim=1,
        )
        return y

    def pinvA(self, y):
        """
        Inverse of the forward model. Reconstruct real image from real-valued vector.
        """
        # choose masks according to indices saved when the state was reset
        mask = self.masks[self.idx]

        # construct k-space from "complex-valued" tensor
        kspace = torch.complex(y[:, 0], y[:, 1]).unsqueeze(1)

        # Apply mask (zero-fill missing entries)
        kspace_filled = kspace * mask
        x_recon = torch.fft.irfft2(kspace_filled, s=self.shape)
        return x_recon

    def AT(self, y):
        pass

    def pinvATA(self, x):
        pass

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
