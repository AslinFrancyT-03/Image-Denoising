
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from io import BytesIO

# Model path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "denoising_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)
@st.cache_resource
def load_denoising_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Page
st.set_page_config(page_title="Image Denoising AI", layout="wide")
st.title("Image Denoising AI")
st.write("Deep Learning-Based Image Restoration")

# Noise setting
noise_level = st.sidebar.slider(
    "Gaussian Noise Level", 0.01, 0.30, 0.10, 0.01
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

# Process only after image is uploaded
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Full-resolution image
        image_array = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        height, width, _ = image_array.shape

        # Add Gaussian noise
        rng = np.random.default_rng(42)
        noise = rng.normal(
            0, noise_level, image_array.shape
        ).astype(np.float32)

        noisy_array = np.clip(
            image_array + noise, 0.0, 1.0
        )

        # Check model
        if not MODEL_PATH.exists():
            st.error(f"Model not found: {MODEL_PATH}")
            st.stop()

        # Overlapping patch-based denoising
        with st.spinner("Denoising image..."):
            model = load_denoising_model()

            patch_size = 32
            stride = 16

            denoised_sum = np.zeros_like(noisy_array)
            weight_sum = np.zeros(
                (height, width, 1), dtype=np.float32
            )

            # Smooth blending weights
            axis = np.hanning(
                patch_size + 2
            )[1:-1].astype(np.float32)
            weights = np.outer(
                axis, axis
            )[..., np.newaxis]

            # Cover image edges
            ys = list(range(
                0, max(height - patch_size, 0) + 1, stride
            ))
            xs = list(range(
                0, max(width - patch_size, 0) + 1, stride
            ))

            if not ys or ys[-1] != max(height - patch_size, 0):
                ys.append(max(height - patch_size, 0))
            if not xs or xs[-1] != max(width - patch_size, 0):
                xs.append(max(width - patch_size, 0))

            patches = []
            positions = []

            for y in ys:
                for x in xs:
                    patch = noisy_array[
                        y:y + patch_size,
                        x:x + patch_size
                    ]

                    ph, pw, _ = patch.shape

                    if ph != patch_size or pw != patch_size:
                        pad_h = patch_size - ph
                        pad_w = patch_size - pw
                        patch = np.pad(
                            patch,
                            ((0, pad_h), (0, pad_w), (0, 0)),
                            mode="reflect"
                        )

                    patches.append(patch)
                    positions.append((y, x, ph, pw))

            # Predict in batches
            predictions = model.predict(
                np.asarray(patches, dtype=np.float32),
                batch_size=64,
                verbose=0
            )

            # Blend overlapping predictions
            for pred, (y, x, ph, pw) in zip(
                predictions, positions
            ):
                w = weights[:ph, :pw]
                denoised_sum[
                    y:y + ph, x:x + pw
                ] += pred[:ph, :pw] * w

                weight_sum[
                    y:y + ph, x:x + pw
                ] += w

            denoised_array = denoised_sum / np.maximum(
                weight_sum, 1e-8
            )
            denoised_array = np.clip(
                denoised_array, 0.0, 1.0
            )

        # Convert to images
        noisy_image = Image.fromarray(
            (noisy_array * 255).astype(np.uint8)
        )
        denoised_image = Image.fromarray(
            (denoised_array * 255).astype(np.uint8)
        )

        # Display results
        st.subheader("Image Restoration Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.write("Noisy Image")
            st.image(noisy_image, use_container_width=True)

        with col3:
            st.write("Denoised Image")
            st.image(denoised_image, use_container_width=True)

        # Image details
        st.subheader("Image Details")
        m1, m2, m3 = st.columns(3)
        m1.metric("Width", f"{width} px")
        m2.metric("Height", f"{height} px")
        m3.metric("Noise Level", f"{noise_level:.2f}")

        # Download
        buffer = BytesIO()
        denoised_image.save(buffer, format="PNG")

        st.download_button(
            "Download Denoised Image",
            data=buffer.getvalue(),
            file_name="denoised_image.png",
            mime="image/png"
        )

        st.success("Image processing completed!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

else:
    st.info("Upload an image to begin denoising.")