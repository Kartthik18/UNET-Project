"""
model.py
--------
U-Net architecture for semantic image segmentation.

The model follows the classic encoder-decoder design with skip connections:
    - Encoder: cascading Conv-BN-ReLU blocks with MaxPool downsampling.
    - Bottleneck: the deepest convolutional block.
    - Decoder: Conv2DTranspose upsampling + skip-connection concatenation.
    - Output head: 1×1 convolution producing per-pixel class logits.

Reference:
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    U-Net: Convolutional Networks for Biomedical Image Segmentation.
    MICCAI 2015. https://arxiv.org/abs/1505.04597
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# ─── Building blocks ──────────────────────────────────────────────────────────

def conv_block(inputs: tf.Tensor, num_filters: int) -> tf.Tensor:
    """
    Two consecutive Conv2D → BatchNorm → ReLU operations.

    Args:
        inputs:      Input feature map tensor.
        num_filters: Number of output feature maps for each convolution.

    Returns:
        Output feature map tensor (same spatial size as input).
    """
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def encoder_block(inputs: tf.Tensor, num_filters: int):
    """
    One encoder stage: conv_block followed by MaxPooling2D.

    Args:
        inputs:      Input tensor.
        num_filters: Number of filters for the convolutional block.

    Returns:
        Tuple (skip_connection, pooled_output) where skip_connection is the
        pre-pool feature map used in the corresponding decoder stage.
    """
    skip = conv_block(inputs, num_filters)
    pool = layers.MaxPooling2D(pool_size=(2, 2))(skip)
    return skip, pool


def decoder_block(inputs: tf.Tensor, skip: tf.Tensor, num_filters: int) -> tf.Tensor:
    """
    One decoder stage: Conv2DTranspose upsampling → concatenate skip → conv_block.

    Args:
        inputs:      Input tensor from the previous decoder (or bottleneck) stage.
        skip:        Corresponding encoder skip-connection tensor.
        num_filters: Number of filters for the convolutional block.

    Returns:
        Output feature map tensor (spatial size doubled relative to *inputs*).
    """
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x


# ─── Full U-Net ───────────────────────────────────────────────────────────────

def build_unet(
    input_shape: tuple = (128, 128, 3),
    num_classes: int   = 3,
) -> tf.keras.Model:
    """
    Assemble a 4-level U-Net and return a compiled Keras Model.

    Encoder filter sizes: 64 → 128 → 256 → 512
    Bottleneck:            1024
    Decoder filter sizes:  512 → 256 → 128 → 64
    Output:                1×1 Conv with *num_classes* filters + softmax.

    Args:
        input_shape: (H, W, C) tuple. Default matches the preprocessed Pet images.
        num_classes: Number of segmentation classes. Oxford IIIT Pet has 3.

    Returns:
        Compiled tf.keras.Model ready for training.
    """
    inputs = layers.Input(shape=input_shape)

    # ── Encoder ──
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1,     128)
    s3, p3 = encoder_block(p2,     256)
    s4, p4 = encoder_block(p3,     512)

    # ── Bottleneck ──
    b = conv_block(p4, 1024)

    # ── Decoder ──
    d1 = decoder_block(b,  s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # ── Output head ──
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(d4)

    model = models.Model(inputs, outputs, name="U-Net")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    unet = build_unet()
    unet.summary()
