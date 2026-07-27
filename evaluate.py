"""
evaluate.py
-----------
Load the best saved checkpoint and evaluate it on the test split.

Usage:
    python src/evaluate.py [--checkpoint checkpoints/best_model.keras]
                           [--results-dir results]
                           [--num-samples 5]
"""

import argparse
import os
import sys

# ── make src/ importable when running from the project root ───────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import tensorflow as tf

from data_loader import get_datasets
from model       import build_unet
from visualize   import save_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained U-Net model")
    parser.add_argument("--checkpoint",  type=str,
                        default="checkpoints/best_model.keras",
                        help="Path to the saved Keras model")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory where prediction images are saved")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of prediction images to save")
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading dataset …")
    _, test_ds, _ = get_datasets()

    print(f"[INFO] Loading model from {args.checkpoint} …")
    if os.path.exists(args.checkpoint):
        model = tf.keras.models.load_model(args.checkpoint)
    else:
        print("[WARN] Checkpoint not found — falling back to untrained model.")
        model = build_unet()

    print("[INFO] Evaluating on test set …")
    results = model.evaluate(test_ds, verbose=1)
    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")

    print(f"[INFO] Saving {args.num_samples} sample predictions …")
    save_predictions(model, test_ds, args.results_dir, args.num_samples)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
