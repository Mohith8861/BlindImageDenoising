# Real-World Smartphone Image Denoising with PyTorch

A high-performance PyTorch implementation of a modified U-Net architecture for real-world smartphone image denoising. This project uses the **Smartphone Image Denoising Dataset (SIDD)** to train a custom deep learning pipeline optimized for fast I/O and GPU throughput, reaching **~40.7 dB PSNR** on 256x256 image patches.

---

## 📌 Highlights & Performance

- **Model Architecture**: Modernized residual U-Net (~0.94M parameters).
- **Data Pipeline**: Custom Memory-Mapped (`np.memmap`) dataset loading for zero-copy binary disk reads and fast multi-worker batch generation.
- **GPU Optimization**: Mixed-precision training (`torch.autocast`), `torch.compile`, batch-level GPU augmentations, and channels-last memory layout.
- **Results**:
  - **Noisy Baseline PSNR**: ~27.80 dB
  - **Final Validation PSNR**: **~40.71 dB** (+12.9 dB improvement)

---

## 🏗️ Architecture

The model (`ModernImageDenoiser`) is built around a lightweight U-Net style architecture with global residual learning. Rather than predicting clean pixels directly, the network learns to isolate and subtract the noise residual from the input image ($Clean = Noisy - Residual$).

Key components:

- **ResBlock**: Convolutional blocks using `GELU` activations and residual skip connections (without BatchNorm inside the residual block).
- **Learnable Down/Upsampling**: Strided convolutions for encoding and transposed convolutions for decoding.
- **Global Residual Skip Connection**: A zero-initialized tail layer allowing the network to start training as an identity mapping and gradually learn noise subtraction.

```

Input (3x256x256) ──► Head Conv ──► Enc1 ──► Enc2 ──► Enc3 ──► Bottleneck
│ │ │                                                             │
▼ ▼ ▼                                                             │
Output (3x256x256) ◄── Global + ◄── Dec1 ◄── Dec2 ◄── Dec3 ◄──────┘

```

---

## ⚙️ Dataset & Efficient Preprocessing

This pipeline uses the **SIDD Small sRGB** dataset. To eliminate disk I/O bottlenecks during training, full-resolution scene images are pre-tiled into $256 \times 256$ non-overlapping patches and saved as a unified uncompressed binary file (`.bin`).

During training, `SIDD_MemmapDataset` reads directly from disk via **memory mapping**:

- Minimal RAM consumption (memory-mapped regions are paged on demand by the OS).
- Multi-worker compatible using lazy initialization inside standard PyTorch `DataLoader`s.
- Over **34,000 patch pairs** (13.5 GB) accessed on the fly.

---

## 🛠️ Data Augmentation & Training Strategy

- **GPU Data Augmentation**: Implements full 8-fold Dihedral ($D_4$) group symmetries (random horizontal flips, vertical flips, and matrix transposes) computed directly on GPU tensors.
- **Loss Function**: $\mathcal{L}_1$ Loss (MAE) evaluated against target clean patches.
- **Optimizer & Schedule**: `AdamW` with `CosineAnnealingLR` ($10^{-4} \to 10^{-6}$) and weight decay ($10^{-4}$).
- **Gradient Clipping**: Norm clipped at `1.0` to guard against loss spikes.

---

## 📈 Training Progress

Below is a snapshot of validation performance across a 20-epoch run:

| Epoch  |  Train Loss  | Train PSNR (dB) |   Val Loss   | Val PSNR (dB) |
| :----: | :----------: | :-------------: | :----------: | :-----------: |
|   1    |   0.014934   |    35.35 dB     |   0.008922   |   38.53 dB    |
|   5    |   0.009859   |    38.34 dB     |   0.007268   |   40.08 dB    |
|   10   |   0.009120   |    38.94 dB     |   0.006979   |   40.42 dB    |
|   15   |   0.008876   |    39.15 dB     |   0.006773   |   40.65 dB    |
| **20** | **0.008788** |  **39.23 dB**   | **0.006720** | **40.71 dB**  |

---

## 🚀 How to Run

### Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA support)
- `numpy`, `Pillow`, `tqdm`, `matplotlib`, `torchinfo`

### Pipeline Execution

1. **Prepare Data Path**: Set the `data_dir` variable in the notebook to your SIDD directory.
2. **Preprocess & Cache**: Run the tiling cells. The code automatically detects existing cached binary files (`sidd_patches.bin`) to skip re-tiling on subsequent runs.
3. **Train**: Run the training loop cell. Weights will automatically save to `BlindImgDenoise2.pth`.

---
