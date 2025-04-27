import torch
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", default=128, type=int)
    parser.add_argument("--density", default=0.3, type=float)
    parser.add_argument("--max-distance", default=0.5, type=float)
    parser.add_argument("--keep-low", default=10, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    res = args.resolution
    density = args.density
    max_distance = args.max_distance
    keep_low = args.keep_low

    H, W = res, res
    mask = torch.zeros(H, W)

    # Compute the center column
    center_col = W // 2

    # Ensure low frequencies are kept (center columns)
    low_freq_start = max(center_col - keep_low // 2, 0)
    low_freq_end = min(center_col + keep_low // 2, W)

    mask[:, low_freq_start:low_freq_end] = 1  # Keep low frequencies around the center

    # Create a probability distribution for column masking
    for i in range(low_freq_end, W):
        # Calculate the distance from the center column (using absolute value)
        dist = abs(i - center_col) / center_col
        ratio = torch.tensor(-dist / max_distance)

        # Apply a density factor to determine the probability of keeping this column
        prob = torch.exp(ratio) * density
        if torch.rand(1) < prob:
            mask[:, i] = 1
            mask[:, W - 1 - i] = 1

    mask = mask.unsqueeze(0)

    plt.imshow(mask[0].cpu().numpy(), cmap="gray", origin="upper")
    plt.axis("off")  # Hide axes for better visualization
    plt.savefig("mask.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    # Save the PyTorch tensor mask
    torch.save(mask.unsqueeze(0), "mask.pt")


if __name__ == "__main__":
    main()
