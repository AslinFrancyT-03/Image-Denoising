
import matplotlib.pyplot as plt

from src.data_preprocessing import load_dataset
from src.noise_generator import add_gaussian_noise


def visualize_noisy_images():
    x_train, _ = load_dataset()

    # Select 5 sample images
    clean_images = x_train[:5]

    # Add Gaussian noise
    noisy_images = add_gaussian_noise(
        clean_images,
        noise_factor=0.2
    )

    # Display clean and noisy images
    plt.figure(figsize=(12, 5))

    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(clean_images[i])
        plt.title("Clean")
        plt.axis("off")

        plt.subplot(2, 5, i + 6)
        plt.imshow(noisy_images[i])
        plt.title("Noisy")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_noisy_images()