
import os
import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)
from src.data_preprocessing import load_dataset
from src.noise_generator import add_gaussian_noise

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "denoising_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)
OUTPUT_DIR = BASE_DIR / "outputs"

# Evaluation settings
BATCH_SIZE = 128
NUM_VISUAL_SAMPLES = 5


def calculate_metrics(original, restored):
    """Calculate image restoration metrics."""

    original = np.clip(original, 0.0, 1.0)
    restored = np.clip(restored, 0.0, 1.0)

    mse = np.mean((original - restored) ** 2)

    psnr = peak_signal_noise_ratio(
        original,
        restored,
        data_range=1.0
    )

    ssim = structural_similarity(
        original,
        restored,
        channel_axis=-1,
        data_range=1.0
    )

    return mse, psnr, ssim


def save_visual_comparison(original, noisy, denoised):
    """Save sample comparison images."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_count = min(
        NUM_VISUAL_SAMPLES, len(original)
    )

    cell_size = 128
    labels = ["Original", "Noisy", "Denoised"]

    from PIL import Image, ImageDraw

    margin = 10
    label_height = 25

    canvas = Image.new(
        "RGB",
        (
            sample_count * (cell_size + margin) + margin,
            3 * (cell_size + label_height + margin)
        ),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    image_groups = [original, noisy, denoised]

    for row, (label, images) in enumerate(
        zip(labels, image_groups)
    ):
        for col in range(sample_count):
            img = np.clip(
                images[col] * 255, 0, 255
            ).astype(np.uint8)

            pil_img = Image.fromarray(img).resize(
                (cell_size, cell_size)
            )

            x = margin + col * (cell_size + margin)
            y = margin + row * (
                cell_size + label_height + margin
            )

            draw.text(
                (x, y),
                f"{label} {col + 1}",
                fill="black"
            )

            canvas.paste(
                pil_img,
                (x, y + label_height)
            )

    output_path = OUTPUT_DIR / "evaluation_samples.png"
    canvas.save(output_path)

    print(f"Visual comparison saved: {output_path}")


def evaluate_model():

    print("\nLoading test dataset...")

    # Load CIFAR-10 test data
    _, x_test = load_dataset()

    x_test = np.clip(
        x_test.astype(np.float32), 0.0, 1.0
    )

    print("Test data shape:", x_test.shape)

    # Generate noisy images
    print("\nGenerating noisy images...")
    x_test_noisy = add_gaussian_noise(x_test)
    x_test_noisy = np.clip(
        x_test_noisy, 0.0, 1.0
    )

    # Load trained model without compiling
    # This avoids unnecessary metric deserialization.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Train the model first using train.py."
        )

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Validate input and output shape
    if model.input_shape[1:] != x_test.shape[1:]:
        raise ValueError(
            f"Model expects {model.input_shape[1:]}, "
            f"but test images have {x_test.shape[1:]}"
        )

    # Metric storage
    noisy_mse, noisy_psnr, noisy_ssim = [], [], []
    denoised_mse, denoised_psnr, denoised_ssim = [], [], []

    sample_original = []
    sample_noisy = []
    sample_denoised = []

    total = len(x_test)

    print("\nEvaluating model...")

    # Batch-based prediction
    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        original_batch = x_test[start:end]
        noisy_batch = x_test_noisy[start:end]

        predictions = model.predict(
            noisy_batch,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        predictions = np.clip(
            predictions, 0.0, 1.0
        )

        if predictions.shape != original_batch.shape:
            raise ValueError(
                "Model output shape does not match "
                "the original image shape."
            )

        # Calculate metrics per image
        for original, noisy, restored in zip(
            original_batch, noisy_batch, predictions
        ):

            m, p, s = calculate_metrics(
                original, noisy
            )
            noisy_mse.append(m)
            noisy_psnr.append(p)
            noisy_ssim.append(s)

            m, p, s = calculate_metrics(
                original, restored
            )
            denoised_mse.append(m)
            denoised_psnr.append(p)
            denoised_ssim.append(s)

        # Keep a few images for visualization
        if len(sample_original) < NUM_VISUAL_SAMPLES:
            needed = NUM_VISUAL_SAMPLES - len(sample_original)
            sample_original.extend(
                original_batch[:needed]
            )
            sample_noisy.extend(
                noisy_batch[:needed]
            )
            sample_denoised.extend(
                predictions[:needed]
            )

        print(
            f"Processed {end}/{total} images",
            end="\r"
        )

    # Calculate average metrics
    results = {
        "Noisy": {
            "MSE": np.mean(noisy_mse),
            "PSNR": np.mean(noisy_psnr),
            "SSIM": np.mean(noisy_ssim)
        },
        "Denoised": {
            "MSE": np.mean(denoised_mse),
            "PSNR": np.mean(denoised_psnr),
            "SSIM": np.mean(denoised_ssim)
        }
    }

    # Display results
    print("\n\nIMAGE DENOISING EVALUATION")
    print("=" * 60)
    print(f"{'Metric':<15}{'Noisy':>18}{'Denoised':>18}")
    print("-" * 60)

    for metric in ["MSE", "PSNR", "SSIM"]:
        print(
            f"{metric:<15}"
            f"{results['Noisy'][metric]:>18.4f}"
            f"{results['Denoised'][metric]:>18.4f}"
        )

    print("=" * 60)

    # Calculate percentage MSE reduction
    if results["Noisy"]["MSE"] > 0:
        reduction = (
            (results["Noisy"]["MSE"] -
             results["Denoised"]["MSE"])
            / results["Noisy"]["MSE"]
        ) * 100

        print(
            f"MSE reduction: {reduction:.2f}%"
        )

    print(
        f"PSNR improvement: "
        f"{results['Denoised']['PSNR'] - results['Noisy']['PSNR']:.2f} dB"
    )

    print(
        f"SSIM improvement: "
        f"{results['Denoised']['SSIM'] - results['Noisy']['SSIM']:.4f}"
    )

    # Save summary CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "evaluation_summary.csv"

    with open(
        summary_path, "w", newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Image Type", "MSE", "PSNR", "SSIM"
        ])

        for image_type, metrics in results.items():
            writer.writerow([
                image_type,
                metrics["MSE"],
                metrics["PSNR"],
                metrics["SSIM"]
            ])

    # Save individual image metrics
    individual_path = OUTPUT_DIR / "per_image_metrics.csv"

    with open(
        individual_path, "w", newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Image Index",
            "Noisy MSE", "Noisy PSNR", "Noisy SSIM",
            "Denoised MSE", "Denoised PSNR",
            "Denoised SSIM"
        ])

        for i in range(total):
            writer.writerow([
                i + 1,
                noisy_mse[i],
                noisy_psnr[i],
                noisy_ssim[i],
                denoised_mse[i],
                denoised_psnr[i],
                denoised_ssim[i]
            ])

    # Save visual comparison
    save_visual_comparison(
        np.asarray(sample_original),
        np.asarray(sample_noisy),
        np.asarray(sample_denoised)
    )

    print("\nEvaluation completed successfully!")
    print(f"Summary: {summary_path}")
    print(f"Individual metrics: {individual_path}")


if __name__ == "__main__":
    evaluate_model()

