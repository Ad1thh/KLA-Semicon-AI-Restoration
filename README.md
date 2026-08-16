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

In high-throughput semiconductor wafer inspection (optical and electron beam / SEM), captured images suffer from photon noise, multiplicative speckle noise, sensor thermal noise, and optical diffraction resolution limits.

This solution delivers a robust, real-time deep learning restoration pipeline that reconstructs **$128 \times 128$ noisy, low-resolution grayscale observations (`NoisyLR`)** back to **$256 \times 256$ pristine ground-truth structures (`GT`)**.

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

### Key Technical Innovations & Domain Invariants
- **Domain Invariants Enforced:**
  1. **Raw Float32 Representation:** Single-channel grayscale inputs and outputs.
  2. **Unclipped Input Handling:** Multiplicative speckle and Gaussian noise push raw input values outside `[0.0, 1.0]`. Inputs are **strictly unclipped** at the dataloader level to preserve true noise physics.
  3. **Output Range Constraint:** Model output predictions are strictly clamped via `torch.clamp(x, 0.0, 1.0)`.
- **High-Throughput Architecture:** NAFNet-SR eliminates non-linear activation bottlenecks (GELU/ReLU) in favor of parameter-free `SimpleGate` and `Simplified Channel Attention (SCA)`, maximizing GPU arithmetic intensity.
- **Physics-Informed Composite Loss:** Multi-domain objective combining Charbonnier spatial loss, Multi-Scale SSIM (MS-SSIM), 2D Fast Fourier Transform (FFT) frequency loss, and LPIPS perceptual loss.

---

## 2. Quantitative Benchmark Results

Evaluated on the official 80/20 validation split ($N=80$ unseen semiconductor wafer patterns):

| Method / Model | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Inference Latency (GPU) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | **23.01 dB** | **0.5286** | **0.4428** | ~1.2 ms | >800 FPS |
| **Classical U-Net (Friend's Baseline)** | **27.17 dB** | **0.7121** | **0.2600** | ~3.3 ms | >300 FPS |
| **NAFNet-SR (Ours)** | **28.16 dB** | **0.7661** | **0.2298** | **~2.8 ms** | **>350 FPS** |
| **Net Improvement vs. Bicubic** | **+5.15 dB** | **+0.2375** | **-48.1%** | **Real-Time** | **Production Ready** |

---

## 3. Visual Restoration Results

Restoration comparisons showing (left to right): **NoisyLR Observation (128x128)**, **Bicubic Upsampling (256x256)**, **NAFNet-SR Restored Output (256x256)**, and **Ground Truth (256x256)**.

| Sample ID | Visual Comparison Quadruplet |
| :---: | :--- |
| **Sample #01** | ![Sample 01](results/triplet_01_000000.png) |
| **Sample #02** | ![Sample 02](results/triplet_02_000007.png) |
| **Sample #03** | ![Sample 03](results/triplet_03_000011.png) |
| **Sample #04** | ![Sample 04](results/triplet_04_000016.png) |

---

## 4. Repository Structure

```
Semicon_Hackathon/
├── configs/
│   └── config.yaml               # Model hyperparameters & training configurations
├── src/
│   ├── __init__.py               # Package initialization
│   ├── model.py                  # NAFNet-SR architecture with PixelShuffle & clamping
│   ├── dataset.py                # Float32 dataset & dataloaders (unclipped inputs)
│   ├── degradations.py           # Multi-order synthetic degradation generator
│   ├── loss.py                   # Composite loss (Charbonnier + MS-SSIM + FFT + LPIPS)
│   └── utils.py                  # PSNR, SSIM, LPIPS metrics & seeding utilities
├── weights/
│   ├── .gitkeep                  # Checkpoints directory
│   └── README.md                 # Checkpoints instructions & usage
├── results/
│   ├── triplet_01_000000.png     # Visual benchmark quadruplets
│   ├── triplet_02_000007.png
│   ├── ...
│   └── visual_triplets_summary.json
├── dummy_in/                     # Sample input files for quick testing
├── train.py                      # Training & validation script (Baseline, Overfit, Full)
├── inference.py                  # Standalone inference script (--input_dir, --output_dir)
├── audit_and_split.py            # Deterministic 80/20 train/val dataset splitter (seed=42)
├── generate_results.py           # Visual comparison & benchmark generator
├── build_presentation.py         # Solution presentation PDF builder (12 slides)
├── benchmark_widths.py           # Model capacity & latency benchmarking tool
├── test_speed_stability.py       # Speed & numerical stability verification test
├── monitor.py                    # Live training progress & GPU monitor
├── solution_presentation.pdf     # 12-slide comprehensive presentation deck
├── requirements.txt              # Environment dependencies
├── LICENSE                       # MIT License
└── README.md                     # Complete solution documentation
```

---

## 5. Architecture & Loss Formulation

### NAFNet-SR Architecture
- **Nonlinear Activation Free Block:** Replaces standard non-linear activations (GELU/ReLU) with a parameter-free `SimpleGate`:
  $$\text{SimpleGate}(X) = X_1 \odot X_2, \quad \text{where } [X_1, X_2] = \text{chunk}(X, \text{dim}=1)$$
- **Simplified Channel Attention (SCA):**
  $$\text{SCA}(X) = X \odot \text{Conv}_{1\times1}(\text{GlobalAvgPool}(X))$$
- **Sub-Pixel Super-Resolution:** Features are upscaled by $2\times$ via sub-pixel `PixelShuffle(2)` followed by output clamping to $[0.0, 1.0]$.

### Multi-Domain Composite Loss Objective
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{Charbonnier}} + 0.5 \cdot \mathcal{L}_{\text{MS-SSIM}} + 0.1 \cdot \mathcal{L}_{\text{FFT}} + 0.05 \cdot \mathcal{L}_{\text{LPIPS}}$$

1. **Charbonnier Loss:** Robust spatial pixel loss:
   $$\mathcal{L}_{\text{Charbonnier}}(Y, \hat{Y}) = \frac{1}{N}\sum \sqrt{(Y_i - \hat{Y}_i)^2 + \epsilon^2} \quad (\epsilon=10^{-6})$$
2. **MS-SSIM Loss:** Multi-scale structural similarity preserving structural edges across 5 spatial scales.
3. **FFT Loss:** Frequency-domain spectral distance preserving fine periodic pitch lines:
   $$\mathcal{L}_{\text{FFT}} = \|\text{Re}(\mathcal{F}(Y)) - \text{Re}(\mathcal{F}(\hat{Y}))\|_1 + \|\text{Im}(\mathcal{F}(Y)) - \text{Im}(\mathcal{F}(\hat{Y}))\|_1$$
4. **LPIPS Loss:** Perceptual feature distance via pre-trained AlexNet backbone.

---

## 6. Installation & Environment Setup

### Prerequisites
- OS: Windows / Linux
- Python: 3.10+
- CUDA: 12.0+ (NVIDIA RTX / H100 GPU recommended)

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/Ad1thh/Semicon_Hackathon.git
cd Semicon_Hackathon

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

## 7. Execution Guide

### A. Dataset Audit & Deterministic 80/20 Split (Seed=42)
Splits the raw dataset deterministically into 80% train and 20% validation sets:
```bash
python audit_and_split.py
```

### B. Standard Bicubic Baseline Evaluation
```bash
python train.py --baseline_only
```

### C. 2-Pair Karpathy Overfit Sanity Check (PSNR > 40 dB)
Validates optimization pipeline before launching full training:
```bash
python train.py --overfit_test
```

### D. Full Model Training
```bash
python train.py --full_train
```

### E. Standalone Inference Evaluation (Strict Evaluator Contract)
The inference script strictly adheres to the competition CLI contract accepting **only** `--input_dir` and `--output_dir`:
```bash
python inference.py --input_dir ./dummy_in --output_dir ./results/predictions
```

### F. Generate Results & Presentation Deck
```bash
python generate_results.py
python build_presentation.py
```

---

## 8. Compliance & Open-Source Attribution

- **Frameworks & Libraries:** PyTorch (BSD-3), Torchvision (BSD-3), LPIPS (BSD-2), PyTorch-MSSSIM (MIT), NumPy (BSD), Matplotlib (PSF), OpenCV (Apache 2.0).
- **Compliance:** All datasets, models, and code strictly adhere to competition guidelines and open-source licenses.
