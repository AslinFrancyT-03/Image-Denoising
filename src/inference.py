
import os
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/denoising_model.keras"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found.")
    return tf.keras.models.load_model(MODEL_PATH)

def denoise_image(image_path):
    model = load_model()

    # Read image without losing its original dimensions
    image = Image.open(image_path).convert("RGB")
    img = np.asarray(image, dtype=np.float32) / 255.0

    h, w, _ = img.shape
    patch_size = 32
    stride = 16

    # Pad small images to at least 32x32
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    img = np.pad(
        img,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect"
    )

    H, W, _ = img.shape
    output = np.zeros_like(img)
    count = np.zeros((H, W, 1), dtype=np.float32)

    # Generate overlapping patches
    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))

    if ys[-1] != H - patch_size:
        ys.append(H - patch_size)
    if xs[-1] != W - patch_size:
        xs.append(W - patch_size)

    for y in ys:
        for x in xs:
            patch = img[y:y+patch_size, x:x+patch_size]
            batch = np.expand_dims(patch, axis=0)

            restored = model.predict(batch, verbose=0)[0]
            restored = np.clip(restored, 0, 1)

            output[y:y+patch_size, x:x+patch_size] += restored
            count[y:y+patch_size, x:x+patch_size] += 1

    # Average overlapping regions
    output = output / np.maximum(count, 1)
    output = output[:h, :w]

    # Convert back to an image
    result = (output * 255).astype(np.uint8)
    return Image.fromarray(result)