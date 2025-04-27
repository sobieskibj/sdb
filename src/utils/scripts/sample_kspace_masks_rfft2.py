import torch
import matplotlib.pyplot as plt


def generate_rfft_mask(shape, sampling_fraction=0.3, num_low_freqs=16, seed=None):
    """Generate a variable-density undersampling mask for rfft2."""

    if seed is not None:
        torch.manual_seed(seed)

    H, W = shape
    W_rfft = W // 2 + 1  # RFFT shape width (half-width)
    mask = torch.zeros(
        (H, W_rfft), dtype=torch.float32
    )  # Save as float32 with 0s initially

    # Keep low frequencies (DC + a few low frequencies)
    mask[:, :num_low_freqs] = 1.0

    # Total number of coefficients in the mask
    total_coeffs = H * W_rfft
    num_to_select = (
        int(sampling_fraction * total_coeffs) - H * num_low_freqs
    )  # Exclude low frequencies

    # Candidate columns are those above low-frequency indices
    candidate_cols = torch.arange(num_low_freqs, W_rfft)

    # Randomly sample from these columns, but ensure we don't sample too many
    num_high_freqs = len(candidate_cols)
    num_sampled_high_freqs = int(sampling_fraction * num_high_freqs)
    sampled_cols = torch.randperm(num_high_freqs)[:num_sampled_high_freqs]

    # Set the sampled high frequencies to 1.0 in the mask
    mask[:, candidate_cols[sampled_cols]] = 1.0

    return mask


def generate_and_save_masks_tensor(
    shape=(256, 256),
    sampling_fraction=0.3,
    num_low_freqs=16,
    num_masks=256,
    out_file="masks_rfft_256x256.pt",
    preview_file="masks_preview.png",
    seed=1337,
):
    all_masks = []

    for i in range(num_masks):
        mask = generate_rfft_mask(
            shape=shape,
            sampling_fraction=sampling_fraction,
            num_low_freqs=num_low_freqs,
            seed=seed + i,
        )
        all_masks.append(mask)

    stacked = torch.stack(all_masks)
    torch.save(stacked, out_file)
    print(f"Saved {num_masks} masks to {out_file}, shape: {stacked.shape}")

    # Plot first 8 masks with better contrast
    plt.figure(figsize=(12, 4))
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        # Ensure masks are within the range [0, 1] (already done as float32 with 0s and 1s)
        mask_image = (
            stacked[i].cpu().numpy() * 255
        )  # Scale to [0, 255] for visualization
        plt.imshow(mask_image, cmap="gray", aspect="auto")
        plt.title(f"Mask {i}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(preview_file)
    print(f"Saved preview to {preview_file}")


# Run the mask generation and saving
generate_and_save_masks_tensor()
