# U-Net Image Segmentation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Semantic image segmentation of pets (cats & dogs) using a **U-Net** convolutional neural network trained on the [Oxford IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Requirements](#-requirements)
- [References](#-references)
- [License](#-license)

---

## 🔍 Overview

This project implements **U-Net** — the seminal encoder-decoder architecture for semantic segmentation — to separate pets (foreground), background, and border regions in natural images.

Key features:
- Clean, modular codebase split across `data_loader`, `model`, `train`, `visualize`, and `evaluate` modules.
- Training callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.
- Results directory with sample predictions and training-history plots.
- GPU-friendly: automatic memory-growth configuration.

---

## 🏗 Architecture

```
Input (128×128×3)
    │
    ├─ Encoder Block 1 (64 filters)  ──────────────────────────┐  skip s1
    │       └─ MaxPool → 64×64                                  │
    ├─ Encoder Block 2 (128 filters) ──────────────────────┐   │  skip s2
    │       └─ MaxPool → 32×32                              │   │
    ├─ Encoder Block 3 (256 filters) ──────────────────┐   │   │  skip s3
    │       └─ MaxPool → 16×16                          │   │   │
    ├─ Encoder Block 4 (512 filters) ──────────────┐   │   │   │  skip s4
    │       └─ MaxPool → 8×8                        │   │   │   │
    │                                               │   │   │   │
    ├─ Bottleneck (1024 filters, 8×8)               │   │   │   │
    │                                               │   │   │   │
    ├─ Decoder Block 1 (512)  ← Upsample + concat ──┘   │   │   │
    ├─ Decoder Block 2 (256)  ← Upsample + concat ──────┘   │   │
    ├─ Decoder Block 3 (128)  ← Upsample + concat ──────────┘   │
    ├─ Decoder Block 4 (64)   ← Upsample + concat ──────────────┘
    │
    └─ Output Conv 1×1 (3 classes) → Softmax (128×128×3)
```

Each encoder/decoder block consists of two **Conv2D → BatchNorm → ReLU** layers.

---

## 📦 Dataset

| Property | Value |
|---|---|
| Dataset | Oxford IIIT Pet |
| Classes | 37 breeds → 3 segmentation labels |
| Train images | 3,680 |
| Test images | 3,669 |
| Input size | 128 × 128 |
| Augmentation | Random horizontal flip |

**Mask labels:**

| Index | Class |
|---|---|
| 0 | Pet (foreground) |
| 1 | Background |
| 2 | Border / ambiguous |

The dataset is downloaded automatically via `tensorflow-datasets` on the first run.

---

## 📊 Results

### Training History

![Training History](results/training_history.png)

### Sample Predictions

![Sample Prediction](results/sample_prediction_01.png)

*Left: input photo · Centre: ground-truth mask · Right: U-Net predicted mask*

---

## 📁 Project Structure

```
unet-image-segmentation/
├── src/
│   ├── data_loader.py     # Dataset loading & preprocessing pipeline
│   ├── model.py           # U-Net architecture (encoder, bottleneck, decoder)
│   ├── train.py           # Training entry point with CLI arguments
│   ├── evaluate.py        # Load checkpoint & evaluate on test split
│   └── visualize.py       # Prediction visualisation utilities
├── checkpoints/           # Saved model weights (generated at runtime)
├── results/
│   ├── README.md          # Explains result files
│   ├── training_history.png
│   └── sample_prediction_01.png
├── notebooks/
│   └── Image_Segmentation_using_UNET_Project_Clean.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/unet-image-segmentation.git
cd unet-image-segmentation
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋 Training

```bash
python src/train.py
```

**Optional arguments:**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `20` | Number of training epochs |
| `--steps-per-epoch` | `58` | Steps per epoch (≈ 3680 / 64) |
| `--val-steps` | `58` | Validation steps |
| `--checkpoint-dir` | `checkpoints` | Where to save model weights |
| `--results-dir` | `results` | Where to save output images |

**Example — 30 epochs with a custom results folder:**

```bash
python src/train.py --epochs 30 --results-dir my_results
```

The best checkpoint is saved to `checkpoints/best_model.keras` (based on minimum validation loss).

---

## 🔬 Evaluation

```bash
python src/evaluate.py
```

**Optional arguments:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | `checkpoints/best_model.keras` | Path to saved model |
| `--results-dir` | `results` | Directory for saved prediction images |
| `--num-samples` | `5` | Number of prediction images to save |

---

## 📓 Notebook

The original Google Colab notebook is preserved in `notebooks/`:

```
notebooks/Image_Segmentation_using_UNET_Project_Clean.ipynb
```

Open it directly in [Google Colab](https://colab.research.google.com/) for a GPU-accelerated interactive walkthrough.

---

## 📦 Requirements

```
tensorflow>=2.10.0
tensorflow-datasets==4.9.3
matplotlib>=3.5.0
numpy>=1.22.0
```

> **GPU note:** A CUDA-compatible GPU is strongly recommended. Training on CPU is functional but slow (~30 min/epoch on modern hardware).

---

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

2. Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). *Cats and Dogs*. IEEE CVPR 2012. [Dataset page](http://www.robots.ox.ac.uk/~vgg/data/pets/)

---

## 📄 License

This project is released under the **MIT License** — see [LICENSE](LICENSE) for details.
