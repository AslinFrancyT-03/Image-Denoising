
import os
import tensorflow as tf

from src.data_preprocessing import load_dataset
from src.model_architecture import build_denoising_model
from src.noise_generator import add_gaussian_noise

# Reproducibility
tf.keras.utils.set_random_seed(42)

# Load dataset
x_train, x_test = load_dataset()

# Use consistent noise level
noise_level = 0.10

# Generate noisy inputs
x_train_noisy = add_gaussian_noise(
    x_train, noise_factor=noise_level, seed=42
)
x_test_noisy = add_gaussian_noise(
    x_test, noise_factor=noise_level, seed=123
)

# Build model
model = build_denoising_model()
model.summary()

# Callbacks
os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_denoising_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
]

# Train
history = model.fit(
    x_train_noisy,
    x_train,
    validation_data=(x_test_noisy, x_test),
    epochs=30,
    batch_size=128,
    callbacks=callbacks,
    shuffle=True
)

# Save final best model
model.save("models/denoising_model.keras")

print("Training completed successfully!")