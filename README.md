# AI-Based Restoration of Degraded Images for Semiconductor Inspection

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SEMICON India](https://img.shields.io/badge/SEMICON%20India-KLA%20Hackathon%202026-brightgreen.svg)](https://www.kla.com/)

**KLA Challenge — Hackathon 2026 (Organized as part of SEMICON India)**  
**Solution:** Nonlinear Activation Free Super-Resolution (**NAFNet-SR**) with Sub-Pixel Convolution & Multi-Domain Composite Loss.

---

## 1. Executive Summary & Problem Formulation

In semiconductor wafer inspection (optical and electron beam / SEM), captured images suffer from photon noise, multiplicative speckle noise, sensor thermal noise, and optical diffraction resolution limits.

This solution delivers a deep learning restoration pipeline that reconstructs **$128 \times 128$ noisy, low-resolution grayscale observations (`NoisyLR`)** back to **$256 \times 256$ ground-truth structures (`GT`)**.

```
[ Degraded Input (128x128 Float32) ] 
       │
       ▼ (Unclipped Range Handling)
┌────────────────────────────────────────────────────────┐
│               NAFNet-SR Architecture                   │
│  - SimpleGate (SG) Activation-Free Blocks              │
│  - Simplified Channel Attention (SCA)                  │
│  - Multi-Scale Skip Connections                        │
│  - Sub-Pixel Upsampling (PixelShuffle 2x)              │
│  - Output Range Clamping [0.0, 1.0]                    │
└────────────────────────────────────────────────────────┘
       │
       ▼
[ Restored Wafer Image (256x256 Float32, [0, 1]) ]
```

### Domain Invariants & Technical Properties
- **Domain Invariants Enforced:**
  1. **Raw Float32 Representation:** Single-channel grayscale inputs and outputs.
  2. **Unclipped Input Handling:** Multiplicative speckle and Gaussian noise push raw input values outside `[0.0, 1.0]`. Inputs are **strictly unclipped** at the dataloader level to preserve physical sensor noise dynamics.
  3. **Output Range Constraint:** Model output predictions are strictly clamped via `torch.clamp(x, 0.0, 1.0)`.
- **Activation-Free Architecture:** NAFNet-SR replaces standard non-linear activation functions (GELU/ReLU) with parameter-free `SimpleGate` and `Simplified Channel Attention (SCA)`.
- **Multi-Domain Composite Loss:** Multi-domain objective combining Charbonnier spatial loss, SSIM structural loss, 2D Fast Fourier Transform (FFT) frequency loss, and LPIPS perceptual loss.

---

## 2. Quantitative Benchmark Results

Evaluated across all 640 held-out validation pairs ($N=640$ unseen semiconductor wafer patterns) with metrics reported as **mean ± standard deviation**:

| Method / Model | Trainable Parameters | PSNR (dB) ↑ *(mean ± std)* | SSIM ↑ *(mean ± std)* | LPIPS ↓ *(mean ± std)* | Single-Image Latency *(RTX 3050 Laptop)* | Throughput *(FPS)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | — | 23.01 ± 3.65 dB | 0.5286 ± 0.1950 | 0.4428 ± 0.1618 | 0.18 ± 0.21 ms | >5,000 FPS |
| **Classical U-Net Baseline** | 1.93M | 27.17 ± 4.30 dB | 0.7121 ± 0.1472 | 0.2600 ± 0.1100 | ~3.3 ms *(measured on RTX 4050)* | ~300 FPS |
| **NAFNet-SR (Ours)** | **29.33M** | **28.16 ± 5.02 dB** | **0.7661 ± 0.1571** | **0.2277 ± 0.1222** | **39.10 ± 8.41 ms** *(Batch=1)*<br>*(10.97 ms eff. @ Batch=8)* | **25.6 FPS** *(Batch=1)*<br>**91.2 FPS** *(Batch=8)* |
| **Net Delta vs. Bicubic** | +29.33M | **+5.15 dB** | **+0.2375** | **-48.6%** | High-Quality Restoration | Real-Time GPU |

> **Hardware & Measurement Note:**  
> - All NAFNet-SR and Bicubic latency numbers were measured programmatically on an **NVIDIA GeForce RTX 3050 Laptop GPU** (Ampere, 4GB VRAM, CUDA 12.6, 2048 CUDA cores).  
> - The Classical U-Net baseline benchmark (1.93M parameters) was evaluated on an **NVIDIA GeForce RTX 4050 Laptop GPU** (Ada Lovelace, 6GB VRAM, 2560 CUDA cores). The RTX 3050 is the lower-tier hardware between the two.  
> - Peak VRAM allocation for NAFNet-SR during inference is **284.0 MB**, enabling execution even on memory-constrained inspection hardware.

---

## 3. Visual Restoration Results

Restoration comparisons showing (left to right): **NoisyLR Observation (128x128)**, **Bicubic Upsampling (256x256)**, **NAFNet-SR Restored Output (256x256)**, and **Ground Truth (256x256)**.

| Sample ID | Visual Comparison Quadruplet |
| :---: | :--- |
| **Sample #01** | ![Sample 01](results/triplet_01_000000.png) |
| **Sample #02** | ![Sample 02](results/triplet_02_000007.png) |
| **Sample #03** | ![Sample 03](results/triplet_03_000011.png) |
| **Sample #04** | ![Sample 04](results/triplet_04_000016.png) |
| **Sample #05** | ![Sample 05](results/triplet_05_000018.png) |
| **Sample #06** | ![Sample 06](results/triplet_06_000020.png) |

---

## 4. Repository Structure & Checkpoint Acquisition

```
KLA-Semicon-AI-Restoration/
├── configs/
│   └── config.yaml               # Model hyperparameters & training configurations
├── src/
│   ├── __init__.py               # Package initialization
│   ├── model.py                  # NAFNet-SR architecture with PixelShuffle & clamping
│   ├── dataset.py                # Float32 dataset & dataloaders (unclipped inputs)
│   ├── degradations.py           # Multi-order synthetic degradation generator
│   ├── loss.py                   # Composite loss (Charbonnier + SSIM + FFT + LPIPS)
│   └── utils.py                  # PSNR, SSIM, LPIPS metrics & seeding utilities
├── tools/                        # Benchmarking, evaluation & utility tools
│   ├── __init__.py
│   ├── benchmark_params.py       # Parameter count verification script
│   ├── benchmark_latency.py      # GPU-labeled latency & throughput benchmark
│   ├── evaluate_val_set.py       # Validation set evaluation (mean ± std) (N=640)
│   ├── generate_results.py       # Visual benchmark quadruplets generator
│   ├── audit_and_split.py        # 80/20 train/val dataset splitter
│   ├── export_weights.py         # Model weight compact exporter
│   ├── benchmark_widths.py       # Channel width scaling analysis
│   └── test_speed_stability.py   # Training speed & numerical stability test
├── weights/
│   ├── nafnet_sr_best.pt         # Trained model weights (56.15 MB, tracked directly in Git)
│   └── README.md                 # Checkpoint metadata and usage instructions
├── results/
│   ├── triplet_01_000000.png     # Visual benchmark quadruplets
│   ├── triplet_02_000007.png
│   ├── ...
│   ├── visual_triplets_summary.json
│   └── validation_evaluation_metrics.json
├── dummy_in/                     # Sample input files for quick testing
├── train.py                      # Core training & validation script (Baseline, Overfit, Full)
├── inference.py                  # Standalone inference script (--input_dir, --output_dir)
├── monitor.py                    # Real-time console training progress monitor
├── requirements.txt              # Environment dependencies
├── LICENSE                       # MIT License
└── README.md                     # Complete solution documentation
```

### Checkpoint Availability
The trained checkpoint used for all benchmark figures is **`weights/nafnet_sr_best.pt`** (file size: **56.15 MB**).  
Because it is under GitHub's 100 MB file limit, it is **tracked directly in this Git repository**. No separate download script or external cloud storage is required — cloning the repository provides the ready-to-run weights immediately.

---

## 5. Architecture & Loss Formulation

### NAFNet-SR Architecture
- **Nonlinear Activation Free Block:** Replaces standard non-linear activations (GELU/ReLU) with a parameter-free `SimpleGate`:
  $$\text{SimpleGate}(X) = X_1 \odot X_2, \quad \text{where } [X_1, X_2] = \text{chunk}(X, \text{dim}=1)$$
- **Simplified Channel Attention (SCA):**
  $$\text{SCA}(X) = X \odot \text{Conv}_{1\times1}(\text{GlobalAvgPool}(X))$$
- **Sub-Pixel Super-Resolution:** Features are upscaled by $2\times$ via sub-pixel `PixelShuffle(2)` followed by output clamping to $[0.0, 1.0]$.
- **Channel LayerNorm:** Exact channel-wise normalization per feature map preventing numerical divergence on flat wafer regions.

### Multi-Domain Composite Loss Objective
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{Charbonnier}} + 0.5 \cdot \mathcal{L}_{\text{SSIM}} + 0.1 \cdot \mathcal{L}_{\text{FFT}} + 0.02 \cdot \mathcal{L}_{\text{LPIPS}}$$

1. **Charbonnier Loss:** Robust spatial pixel loss:
   $$\mathcal{L}_{\text{Charbonnier}}(Y, \hat{Y}) = \frac{1}{N}\sum \sqrt{(Y_i - \hat{Y}_i)^2 + \epsilon^2} \quad (\epsilon=10^{-6})$$
2. **SSIM Loss:** Structural similarity preserving luminance and contrast across wafer edges.
3. **2D-FFT Loss:** Frequency-domain spectral distance preserving fine periodic pitch lines:
   $$\mathcal{L}_{\text{FFT}} = \frac{1}{N}\sum \sqrt{|\mathcal{F}(Y) - \mathcal{F}(\hat{Y})|^2 + \epsilon}$$
4. **LPIPS Loss:** Perceptual feature distance via pre-trained AlexNet backbone.

---

## 6. Installation & Environment Setup

### Prerequisites
- OS: Windows / Linux
- Python: 3.10+
- CUDA: 12.0+ (NVIDIA GPU recommended)

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/Ad1thh/KLA-Semicon-AI-Restoration.git
cd KLA-Semicon-AI-Restoration

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Execution & Verification Guide

### A. Parameter Count Verification
Prints total and trainable parameter counts directly from the model definition:
```bash
python tools/benchmark_params.py
```

### B. GPU Latency & Throughput Benchmark
Measures single-image and batched inference latency programmatically querying the local GPU:
```bash
python tools/benchmark_latency.py
```

### C. Validation Set Evaluation (Mean ± Std)
Evaluates Bicubic baseline and NAFNet-SR across all 640 validation images:
```bash
python tools/evaluate_val_set.py
```

### D. Standalone Inference Evaluation (Strict Evaluator Contract)
Runs inference accepting **only** `--input_dir` and `--output_dir` (automatically loads `weights/nafnet_sr_best.pt`):
```bash
python inference.py --input_dir ./dummy_in --output_dir ./dummy_out
```

### E. Full Model Training from Scratch
```bash
python train.py --full_train
```

---

## 8. Compliance & Open-Source Attribution

- **Frameworks & Libraries:** PyTorch (BSD-3), Torchvision (BSD-3), LPIPS (BSD-2), PyTorch-MSSSIM (MIT), NumPy (BSD), Matplotlib (PSF), python-pptx (MIT).
- **Compliance:** All datasets, models, and code strictly adhere to competition guidelines and open-source licenses.
