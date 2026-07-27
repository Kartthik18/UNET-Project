"""
train.py
--------
Entry point for training the U-Net segmentation model.

Usage:
    python src/train.py [--epochs N] [--steps-per-epoch S] [--val-steps V]

The script will:
    1. Build the U-Net model.
    2. Load the Oxford IIIT Pet dataset via data_loader.
    3. Train with ModelCheckpoint + EarlyStopping callbacks.
    4. Save training-history plots to results/.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for headless servers)
import matplotlib.pyplot as plt

import tensorflow as tf

# ── make src/ importable when running from the project root ───────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data_loader import get_datasets
from model       import build_unet
from visualize   import display_sample, save_predictions

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train U-Net on Oxford IIIT Pet")
    parser.add_argument("--epochs",           type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--steps-per-epoch",  type=int, default=58,
                        help="Training steps per epoch (default: 58 ≈ 3680/64)")
    parser.add_argument("--val-steps",        type=int, default=58,
                        help="Validation steps (default: 58 ≈ 3669/64)")
    parser.add_argument("--checkpoint-dir",   type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--results-dir",      type=str, default="results",
                        help="Directory to save prediction visualisations")
    return parser.parse_args()


# ─── Callbacks ────────────────────────────────────────────────────────────────

def build_callbacks(checkpoint_dir: str):
    """Return a list of Keras callbacks for training."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    return [checkpoint, early_stop, reduce_lr]


# ─── History plotting ─────────────────────────────────────────────────────────

def plot_history(history, results_dir: str):
    """Save accuracy and loss curves as PNG files."""
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    path = os.path.join(results_dir, "training_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Training history plot saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # GPU memory growth (prevents OOM on shared GPUs)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print("[INFO] Loading dataset …")
    train_ds, test_ds, info = get_datasets()

    print("[INFO] Building model …")
    model = build_unet(input_shape=(128, 128, 3), num_classes=3)
    model.summary()

    callbacks = build_callbacks(args.checkpoint_dir)

    print("[INFO] Starting training …")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=test_ds,
        validation_steps=args.val_steps,
        callbacks=callbacks,
    )

    plot_history(history, args.results_dir)

    print("[INFO] Saving sample predictions …")
    save_predictions(model, test_ds, args.results_dir, num_samples=5)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
