
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from src.data_preprocessing import load_dataset
from src.noise_generator import add_gaussian_noise

def evaluate_model():
    # Load dataset
    _, x_test = load_dataset()

    # Evaluate on 500 test images
    x_test = x_test[:500]

    # Generate noisy images
    x_noisy = add_gaussian_noise(
        x_test, noise_factor=0.10, seed=123
    )

    # Load best trained model
    model_path = "models/best_denoising_model.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    # Predict
    x_denoised = model.predict(
        x_noisy, batch_size=128, verbose=1
    )
    x_denoised = np.clip(x_denoised, 0, 1)

    # Calculate metrics
    noisy_psnr, denoised_psnr = [], []
    noisy_ssim, denoised_ssim = [], []

    for original, noisy, denoised in zip(
        x_test, x_noisy, x_denoised
    ):
        noisy_psnr.append(peak_signal_noise_ratio(
            original, noisy, data_range=1.0
        ))
        denoised_psnr.append(peak_signal_noise_ratio(
            original, denoised, data_range=1.0
        ))

        noisy_ssim.append(structural_similarity(
            original, noisy, channel_axis=-1, data_range=1.0
        ))
        denoised_ssim.append(structural_similarity(
            original, denoised, channel_axis=-1, data_range=1.0
        ))

    print("\nIMAGE DENOISING EVALUATION")
    print("-" * 55)
    print(f"{'Metric':<12}{'Noisy':>15}{'Denoised':>15}")
    print("-" * 55)
    print(f"{'PSNR (dB)':<12}{np.mean(noisy_psnr):>15.4f}"
          f"{np.mean(denoised_psnr):>15.4f}")
    print(f"{'SSIM':<12}{np.mean(noisy_ssim):>15.4f}"
          f"{np.mean(denoised_ssim):>15.4f}")

    # Save visual comparison
    os.makedirs("outputs", exist_ok=True)

    fig, axes = plt.subplots(3, 5, figsize=(14, 9))
    for i in range(5):
        axes[0, i].imshow(x_test[i])
        axes[0, i].set_title(f"Original {i+1}")

        axes[1, i].imshow(x_noisy[i])
        axes[1, i].set_title(f"Noisy {i+1}")

        axes[2, i].imshow(x_denoised[i])
        axes[2, i].set_title(f"Denoised {i+1}")

        for ax in axes[:, i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        "outputs/evaluation_samples.png",
        dpi=200, bbox_inches="tight"
    )
    plt.close()

    print("\nEvaluation samples saved to outputs/evaluation_samples.png")
    print("Evaluation completed!")

if __name__ == "__main__":
    evaluate_model()