
import numpy as np


def add_gaussian_noise(images, noise_factor=0.10, seed=None):
    """
    Add Gaussian noise to normalized images.

    Args:
        images: Clean images in the range [0, 1].
        noise_factor: Standard deviation of Gaussian noise.
        seed: Optional random seed for reproducibility.

    Returns:
        Noisy images as float32 in the range [0, 1].
    """
    images = np.asarray(images, dtype=np.float32)

    rng = np.random.default_rng(seed)
    noise = rng.normal(
        loc=0.0,
        scale=noise_factor,
        size=images.shape
    ).astype(np.float32)

    noisy_images = np.clip(images + noise, 0.0, 1.0)

    return noisy_images.astype(np.float32)


if __name__ == "__main__":
    from src.data_preprocessing import load_dataset

    x_train, x_test = load_dataset()

    sample = x_train[:5]
    noisy = add_gaussian_noise(sample, noise_factor=0.10, seed=42)

    print("Original shape:", sample.shape)
    print("Noisy shape:", noisy.shape)
    print("Pixel range:", noisy.min(), "to", noisy.max())
    print("Noise generation successful!")