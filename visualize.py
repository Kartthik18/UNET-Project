"""
visualize.py
------------
Utilities for displaying and saving segmentation predictions.

Functions
---------
display_sample(image_list)
    Show an [Input Image | True Mask | Predicted Mask] row in a matplotlib figure.

save_predictions(model, dataset, results_dir, num_samples)
    Run the model on the first *num_samples* images from *dataset* and write
    side-by-side PNG comparison files into *results_dir*.
"""

import os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ─── Interactive display ──────────────────────────────────────────────────────

def display_sample(image_list: list, title: list = None):
    """
    Render a row of images in a matplotlib figure.

    Args:
        image_list: List of image/mask tensors or arrays.
        title:      Optional list of subplot titles.  Defaults to
                    ['Input Image', 'True Mask', 'Predicted Mask'] (first 3).
    """
    default_titles = ["Input Image", "True Mask", "Predicted Mask"]
    title = title or default_titles

    n = len(image_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    for i, (ax, img) in enumerate(zip(axes, image_list)):
        ax.set_title(title[i] if i < len(title) else f"Image {i}")
        ax.imshow(tf.keras.utils.array_to_img(img))
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ─── Batch prediction → disk ──────────────────────────────────────────────────

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """
    Convert a softmax probability tensor to a single-channel integer mask.

    Args:
        pred_mask: Tensor of shape (1, H, W, num_classes) — model output.

    Returns:
        Tensor of shape (H, W, 1) with integer class indices.
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)   # (1, H, W)
    pred_mask = pred_mask[..., tf.newaxis]       # (1, H, W, 1)
    return pred_mask[0]                          # (H, W, 1)


def save_predictions(
    model:       tf.keras.Model,
    dataset:     tf.data.Dataset,
    results_dir: str,
    num_samples: int = 5,
):
    """
    Save side-by-side visualisations (Input | True Mask | Predicted Mask) to disk.

    Files are written as:
        results/prediction_01.png
        results/prediction_02.png
        …

    Args:
        model:       Trained Keras model.
        dataset:     Batched tf.data.Dataset yielding (image, mask) pairs.
        results_dir: Directory where PNG files are saved (created if absent).
        num_samples: How many samples to save.
    """
    os.makedirs(results_dir, exist_ok=True)

    count = 0
    for images, masks in dataset.take(1):          # first batch is enough
        for i in range(min(num_samples, len(images))):
            img  = images[i]                       # (H, W, 3)
            mask = masks[i]                        # (H, W, 1)

            pred = model.predict(img[tf.newaxis, ...], verbose=0)
            pred_mask = create_mask(pred)          # (H, W, 1)

            # ── Plot ──────────────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            titles = ["Input Image", "True Mask", "Predicted Mask"]
            imgs   = [img, mask, pred_mask]

            for ax, title, im in zip(axes, titles, imgs):
                ax.set_title(title, fontsize=14)
                ax.imshow(tf.keras.utils.array_to_img(im))
                ax.axis("off")

            count += 1
            filepath = os.path.join(results_dir, f"prediction_{count:02d}.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[INFO] Saved → {filepath}")

            if count >= num_samples:
                return


# ─── Standalone demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick sanity check: load model + first test batch and save predictions."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from data_loader import get_datasets
    from model       import build_unet

    print("[INFO] Loading dataset …")
    _, test_ds, _ = get_datasets()

    print("[INFO] Loading model from checkpoints/best_model.keras …")
    try:
        model = tf.keras.models.load_model("checkpoints/best_model.keras")
    except Exception:
        print("[WARN] No saved model found — using untrained model for demo.")
        model = build_unet()

    save_predictions(model, test_ds, results_dir="results", num_samples=5)
    print("[INFO] Done.  Check the results/ directory.")
