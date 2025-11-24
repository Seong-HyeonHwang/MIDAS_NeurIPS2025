# MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning

This repository contains the official implementation of our paper accepted at NeurIPS 2025.
[![Paper](https://img.shields.io/badge/Paper-blue)](https://arxiv.org/pdf/2509.25831)

## Abstract
MIDAS addresses the critical challenge of modality imbalance in multimodal learning, where models often over-rely on a dominant modality. Our approach leverages a novel misalignment-based data augmentation strategy to mitigate this dominance and encourage robust joint representation learning. MIDAS outperforms state-of-the-art baselines by effectively balancing the contribution of different modalities.


## Quick Start

> **Note**: Please prepare your dataset and specify the path using the `--dataset_root` argument. The framework automatically handles data splitting (70/10/20) and initiates training.

### Command Line Usage

You can train the model on various multimodal datasets using the following commands:

```bash
# Train on CREMA-D (Audio-Video Emotion Recognition)
python train.py --dataset cremad --dataset_root /path/to/cremad --devices 0

# Train on Food101 (Image-Text Classification)
python train.py --dataset food101 --dataset_root /path/to/Food101 --devices 0

# Train on UCF101 (RGB-Flow Action Recognition)
python train.py --dataset ucf101 --dataset_root /path/to/ucf101 --devices 0

# Train on Kinetics (Audio-Video Action Recognition)
python train.py --dataset kinetics --dataset_root /path/to/kinetics --devices 0
```
---

## Note

### Command Line Arguments

#### Required Arguments
| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--dataset` | str | Dataset name (`cremad`, `ucf101`, `food101`, `kinetics`) | `--dataset cremad` |
| `--dataset_root` | str | Root directory path of the dataset | `--dataset_root /data/cremad` |

#### Optional Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--method` | str | `Midas` | Training method (currently only Midas supported) |
| `--seed` | int | `2` | Random seed for reproducibility |
| `--devices` | str | `'0'` | GPU device IDs (comma-separated, e.g., `'0,1'`) |
| `--batch_size` | int | `64` | Training batch size |
| `--max_epochs` | int | varies* | Maximum training epochs |
| `--lr` | float | varies* | Learning rate |
| `--output_dir` | str | auto | Output directory for checkpoints |

\*Default values vary by dataset
