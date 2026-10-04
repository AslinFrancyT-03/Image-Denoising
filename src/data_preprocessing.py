
import os
import pickle
import numpy as np


DATASET_DIR = "data/raw/cifar-10-python/cifar-10-batches-py"


def load_batch(file_path):
    """Load one CIFAR-10 batch."""
    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"]
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images


def load_dataset():
    """Load CIFAR-10 directly from local files."""
    x_train = []

    for i in range(1, 6):
        file_path = os.path.join(
            DATASET_DIR, f"data_batch_{i}"
        )
        x_train.append(load_batch(file_path))

    x_train = np.concatenate(x_train, axis=0)
    x_test = load_batch(
        os.path.join(DATASET_DIR, "test_batch")
    )

    # Normalize pixels to the range [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)

    return x_train, x_test


if __name__ == "__main__":
    x_train, x_test = load_dataset()

    print("Dataset loaded successfully!")
    print("Pixel range:", x_train.min(), "to", x_train.max())