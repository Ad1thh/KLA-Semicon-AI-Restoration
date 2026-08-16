import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.model import NAFNetSR
from src.utils import calculate_psnr, calculate_ssim, LPIPSCalculator, bicubic_upsample

def generate_visual_triplets(
    weights_path: str = "./weights/nafnet_sr_best.pt",
    val_deg_dir: str = "./data/val/degraded",
    val_gt_dir: str = "./data/val/gt",
    output_dir: str = "./results",
    num_samples: int = 6
):
    if not os.path.exists(val_deg_dir):
        val_deg_dir = os.path.join(os.path.dirname(__file__), "..", "data", "val", "degraded")
    if not os.path.exists(val_gt_dir):
        val_gt_dir = os.path.join(os.path.dirname(__file__), "..", "data", "val", "gt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "nafnet_sr_best.pt")
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", output_dir))

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = 32
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "config" in checkpoint and "model" in checkpoint["config"]:
            width = checkpoint["config"]["model"].get("width", 32)
        
        model = NAFNetSR(
            in_channels=1,
            out_channels=1,
            width=width,
            enc_blk_nums=[2, 2, 4, 8],
            middle_blk_num=12,
            dec_blk_nums=[2, 2, 2, 2],
            scale_factor=2
        ).to(device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        print(f"[Results] Loaded weights from {weights_path} (width={width})")
    else:
        model = NAFNetSR(
            in_channels=1,
            out_channels=1,
            width=width,
            enc_blk_nums=[2, 2, 4, 8],
            middle_blk_num=12,
            dec_blk_nums=[2, 2, 2, 2],
            scale_factor=2
        ).to(device)
        print(f"[Results Warning] Weights {weights_path} not found. Running with default initialization.")

    model.eval()
    lpips_calc = LPIPSCalculator(device)

    deg_files = sorted(glob.glob(os.path.join(val_deg_dir, "*.npy")))
    gt_files = sorted(glob.glob(os.path.join(val_gt_dir, "*.npy")))

    common_fns = sorted(list(set(os.path.basename(f) for f in deg_files) & set(os.path.basename(f) for f in gt_files)))
    selected_fns = common_fns[:num_samples]

    print(f"[Results] Generating {len(selected_fns)} visual comparison triplets...")

    triplet_records = []

    for idx, fn in enumerate(selected_fns, 1):
        deg_np = np.load(os.path.join(val_deg_dir, fn)).astype(np.float32)
        gt_np = np.load(os.path.join(val_gt_dir, fn)).astype(np.float32)

        if deg_np.ndim == 2:
            deg_tensor = torch.from_numpy(deg_np).unsqueeze(0).unsqueeze(0).to(device)
        else:
            deg_tensor = torch.from_numpy(deg_np).unsqueeze(0).to(device)

        if gt_np.ndim == 2:
            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).to(device)
        else:
            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).to(device)

        with torch.no_grad():
            bicubic_tensor = torch.clamp(bicubic_upsample(deg_tensor, scale_factor=2), 0.0, 1.0)
            restored_tensor = model(deg_tensor)
            restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)

        bic_psnr = calculate_psnr(bicubic_tensor, gt_tensor)
        bic_ssim = calculate_ssim(bicubic_tensor, gt_tensor)
        bic_lpips = lpips_calc(bicubic_tensor, gt_tensor)

        rest_psnr = calculate_psnr(restored_tensor, gt_tensor)
        rest_ssim = calculate_ssim(restored_tensor, gt_tensor)
        rest_lpips = lpips_calc(restored_tensor, gt_tensor)

        deg_disp = np.clip(deg_np, 0.0, 1.0)
        bic_disp = bicubic_tensor.squeeze().cpu().numpy()
        rest_disp = restored_tensor.squeeze().cpu().numpy()
        gt_disp = gt_np

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        
        axes[0].imshow(deg_disp, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"Input: NoisyLR (128x128)\n{fn}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(bic_disp, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Bicubic Baseline (256x256)\nPSNR: {bic_psnr:.2f} dB | SSIM: {bic_ssim:.4f}\nLPIPS: {bic_lpips:.4f}", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(rest_disp, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"NAFNet-SR Restored (256x256)\nPSNR: {rest_psnr:.2f} dB | SSIM: {rest_ssim:.4f}\nLPIPS: {rest_lpips:.4f}", fontsize=10, color="darkgreen", fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(gt_disp, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title(f"Target: Clean GT (256x256)\nGround Truth Reference", fontsize=11, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"triplet_{idx:02d}_{fn.replace('.npy', '')}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        triplet_records.append({
            "index": idx,
            "filename": fn,
            "bicubic": {"psnr": bic_psnr, "ssim": bic_ssim, "lpips": bic_lpips},
            "nafnet_sr": {"psnr": rest_psnr, "ssim": rest_ssim, "lpips": rest_lpips},
            "psnr_gain": rest_psnr - bic_psnr,
            "image_path": save_path
        })
        print(f"  [+] Triplet {idx:02d} ({fn}): NAFNet-SR PSNR={rest_psnr:.2f} dB (Gain: +{rest_psnr - bic_psnr:.2f} dB)")

    with open(os.path.join(output_dir, "visual_triplets_summary.json"), "w") as f:
        json.dump(triplet_records, f, indent=2)

    print(f"[Results Complete] All 6 visual triplets saved to {output_dir}")

if __name__ == "__main__":
    generate_visual_triplets()
