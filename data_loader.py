"""
data_loader.py
--------------
Handles loading, preprocessing, and batching of the Oxford IIIT Pet dataset
using TensorFlow Datasets for U-Net segmentation training.
"""

import tensorflow as tf
import tensorflow_datasets as tfds

# ─── Hyperparameters ──────────────────────────────────────────────────────────
IMAGE_SIZE  = (128, 128)
BATCH_SIZE  = 64
BUFFER_SIZE = 1000


# ─── Normalisation ────────────────────────────────────────────────────────────

def normalize(input_image: tf.Tensor, input_mask: tf.Tensor):
    """
    Normalize the image to [0, 1] and convert the mask to zero-based indexing.

    Args:
        input_image: uint8 image tensor.
        input_mask:  uint8 mask tensor (values 1, 2, 3 in the Pet dataset).

    Returns:
        Tuple of (normalized_image, zero_indexed_mask).
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask  = input_mask - 1          # 1-indexed → 0-indexed
    return input_image, input_mask


# ─── Per-split loading functions ──────────────────────────────────────────────

def load_train_image(sample: dict):
    """
    Load a single training sample: resize, randomly flip, and normalize.

    Args:
        sample: TFDS sample dict with keys 'image' and 'segmentation_mask'.

    Returns:
        Tuple of (preprocessed_image, preprocessed_mask).
    """
    image = tf.image.resize(sample["image"],            IMAGE_SIZE)
    mask  = tf.image.resize(sample["segmentation_mask"], IMAGE_SIZE)

    # Random horizontal flip for data augmentation
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)

    return normalize(image, mask)


def load_test_image(sample: dict):
    """
    Load a single test sample: resize and normalize (no augmentation).

    Args:
        sample: TFDS sample dict with keys 'image' and 'segmentation_mask'.

    Returns:
        Tuple of (preprocessed_image, preprocessed_mask).
    """
    image = tf.image.resize(sample["image"],            IMAGE_SIZE)
    mask  = tf.image.resize(sample["segmentation_mask"], IMAGE_SIZE)
    return normalize(image, mask)


# ─── Dataset builder ──────────────────────────────────────────────────────────

def get_datasets():
    """
    Download (or reuse cached) Oxford IIIT Pet dataset and return ready-to-use
    train and test tf.data pipelines.

    Returns:
        train_dataset (tf.data.Dataset): Shuffled, batched, prefetched pipeline.
        test_dataset  (tf.data.Dataset): Batched pipeline (no shuffle).
        info          (tfds.core.DatasetInfo): Dataset metadata.
    """
    dataset, info = tfds.load("oxford_iiit_pet", with_info=True)

    # ── Training pipeline ──
    train_dataset = (
        dataset["train"]
        .map(load_train_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # ── Test pipeline ──
    test_dataset = (
        dataset["test"]
        .map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
    )

    return train_dataset, test_dataset, info
