# Model Checkpoints & Reproducible Weights

This directory contains the trained NAFNet-SR model checkpoint used to produce all reported evaluation metrics and visual results.

## Checkpoint Details
- **File**: `weights/nafnet_sr_best.pt`
- **File Size**: ~56.15 MB (committed directly into Git, under GitHub's 100 MB single-file limit)
- **Model Architecture**: NAFNet-SR (Nonlinear Activation Free Super-Resolution)
- **Architecture Config**: `width=32`, `enc_blk_nums=[2, 2, 4, 8]`, `middle_blk_num=12`, `dec_blk_nums=[2, 2, 2, 2]`, `scale_factor=2`
- **Total Parameters**: 29,333,988 (29.33M)
- **Validation Metrics**: PSNR: 28.16 ± 5.02 dB | SSIM: 0.7661 ± 0.1571 | LPIPS: 0.2277 ± 0.1222

## Obtaining the Checkpoint
Because the checkpoint is under 100 MB, it is **tracked directly in this Git repository**. Simply cloning the repository includes the checkpoint:
```bash
git clone https://github.com/Ad1thh/KLA-Semicon-AI-Restoration.git
cd KLA-Semicon-AI-Restoration
```

## Running Inference
The inference script automatically discovers and loads `weights/nafnet_sr_best.pt`:
```bash
python inference.py --input_dir ./dummy_in --output_dir ./dummy_out
```

## Training from Scratch
To reproduce the training process from scratch:
```bash
python train.py --full_train
```
The checkpoint with the highest validation PSNR will be saved to `weights/nafnet_sr_best.pt`.
