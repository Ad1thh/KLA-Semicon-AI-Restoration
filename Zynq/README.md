# Team Zynq — Semiconductor Image Restoration

**KLA Hackathon 2026 (Organized as part of SEMICON India)**  
**Solution:** NAFNet-SR (Nonlinear Activation Free Super-Resolution) with Multi-Domain Composite Loss

---

## 1. Quickstart & Execution

```bash
# Run inference on degraded test images:
python run.py <input-directory> <output-directory>
```

### Examples:
```bash
# Positional format:
python run.py ./dummy_in ./dummy_out

# Named argument format:
python run.py --input_dir ./dummy_in --output_dir ./dummy_out
```

---

## 2. Directory Structure

```
Zynq/
├── run.py                 # Self-contained standalone inference engine
├── requirements.txt       # Environment dependencies
├── README.md              # Submission guide & metadata
├── Idea-Submission-Template_Hackathon-2026_Team_Zynq.pptx # Hackathon presentation deck
└── models/
    └── nafnet_sr_best.pt  # Trained NAFNet-SR model checkpoint weights
```

---

## 3. Key Specifications

* **Architecture:** NAFNet-SR (Nonlinear Activation Free Super-Resolution)
* **Model Parameters:** **2,393,985 (2.39M)** — strictly within the < 4.0M parameter budget
* **Input / Output Format:** Single-Channel Float32 grayscale (unclipped input handling, output clamped to `[0.0, 1.0]`)
* **Arbitrary Resolution Support:** Automatically handles $128\times 128$, $256\times 256$, $512\times 512$, and non-standard image crops.
* **Validation Performance:** **28.98 dB PSNR** (+5.98 dB over Bicubic), **0.949 SSIM**, **0.263 LPIPS**
* **Throughput:** **127.5 FPS (7.84 ms / image)** with **199.5 MB peak VRAM** on NVIDIA GPU.
