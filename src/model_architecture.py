
import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(x, filters):
    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def build_denoising_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)

    # Bottleneck
    b = conv_block(p2, 128)

    # Decoder
    u2 = layers.UpSampling2D(2)(b)
    u2 = layers.Concatenate()([u2, c2])
    c3 = conv_block(u2, 64)

    u1 = layers.UpSampling2D(2)(c3)
    u1 = layers.Concatenate()([u1, c1])
    c4 = conv_block(u1, 32)

    # Predict noise (residual)
    predicted_noise = layers.Conv2D(
        3, 3, padding="same",
        activation="linear"
    )(c4)

    # Subtract predicted noise from input
    outputs = layers.Subtract()(
        [inputs, predicted_noise]
    )
    outputs = layers.Activation("sigmoid")(outputs)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model