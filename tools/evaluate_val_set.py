import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import json
import time
import numpy as np
import torch

from src.model import NAFNetSR
from src.utils import calculate_psnr, calculate_ssim, LPIPSCalculator, bicubic_upsample

def evaluate_validation_set(
    val_deg_dir: str = "./data/val/degraded",
    val_gt_dir: str = "./data/val/gt",
    weights_path: str = "./weights/nafnet_sr_best.pt",
    output_json: str = "./results/validation_evaluation_metrics.json"
):
    if not os.path.exists(val_deg_dir):
        val_deg_dir = os.path.join(os.path.dirname(__file__), "..", "data", "val", "degraded")
    if not os.path.exists(val_gt_dir):
        val_gt_dir = os.path.join(os.path.dirname(__file__), "..", "data", "val", "gt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "nafnet_sr_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    print("=" * 75, flush=True)
    print("Full Validation Set Evaluation (Mean +/- Standard Deviation)", flush=True)
    print("=" * 75, flush=True)
    print(f"Device               : {device} ({gpu_name})", flush=True)
    print(f"Degraded Input Dir   : {val_deg_dir}", flush=True)
    print(f"Ground Truth Dir     : {val_gt_dir}", flush=True)
    print(f"Model Checkpoint     : {weights_path}", flush=True)

    model = NAFNetSR(
        in_channels=1,
        out_channels=1,
        width=32,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
        scale_factor=2
    ).to(device)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        print(f"Loaded Checkpoint    : {weights_path}", flush=True)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    model.eval()
    lpips_calc = LPIPSCalculator(device)

    deg_files = sorted(glob.glob(os.path.join(val_deg_dir, "*.npy")))
    gt_files = sorted(glob.glob(os.path.join(val_gt_dir, "*.npy")))
    common_fns = sorted(list(set(os.path.basename(f) for f in deg_files) & set(os.path.basename(f) for f in gt_files)))

    num_samples = len(common_fns)
    print(f"Total Validation Pairs: {num_samples}", flush=True)
    print("-" * 75, flush=True)

    bic_psnr_list, bic_ssim_list, bic_lpips_list = [], [], []
    naf_psnr_list, naf_ssim_list, naf_lpips_list = [], [], []

    t0 = time.time()
    for idx, fn in enumerate(common_fns, 1):
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
            bic_tensor = torch.clamp(bicubic_upsample(deg_tensor, scale_factor=2), 0.0, 1.0)
            naf_tensor = model(deg_tensor)
            naf_tensor = torch.clamp(naf_tensor, 0.0, 1.0)

        # Bicubic metrics
        bic_psnr_list.append(calculate_psnr(bic_tensor, gt_tensor))
        bic_ssim_list.append(calculate_ssim(bic_tensor, gt_tensor))
        bic_lpips_list.append(lpips_calc(bic_tensor, gt_tensor))

        # NAFNet-SR metrics
        naf_psnr_list.append(calculate_psnr(naf_tensor, gt_tensor))
        naf_ssim_list.append(calculate_ssim(naf_tensor, gt_tensor))
        naf_lpips_list.append(lpips_calc(naf_tensor, gt_tensor))

        if idx % 80 == 0 or idx == num_samples:
            print(f"  Processed [{idx:03d}/{num_samples:03d}] images | Running NAFNet PSNR: {np.mean(naf_psnr_list):.4f} dB", flush=True)

    elapsed = time.time() - t0

    results = {
        "num_samples": num_samples,
        "eval_time_seconds": elapsed,
        "gpu_name": gpu_name,
        "bicubic_baseline": {
            "psnr_mean": float(np.mean(bic_psnr_list)),
            "psnr_std": float(np.std(bic_psnr_list)),
            "ssim_mean": float(np.mean(bic_ssim_list)),
            "ssim_std": float(np.std(bic_ssim_list)),
            "lpips_mean": float(np.mean(bic_lpips_list)),
            "lpips_std": float(np.std(bic_lpips_list))
        },
        "nafnet_sr": {
            "psnr_mean": float(np.mean(naf_psnr_list)),
            "psnr_std": float(np.std(naf_psnr_list)),
            "ssim_mean": float(np.mean(naf_ssim_list)),
            "ssim_std": float(np.std(naf_ssim_list)),
            "lpips_mean": float(np.mean(naf_lpips_list)),
            "lpips_std": float(np.std(naf_lpips_list))
        }
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 75, flush=True)
    print(f"Empirical Validation Benchmark Results (N={num_samples} Images):", flush=True)
    print(f"  Bicubic Baseline:", flush=True)
    print(f"    PSNR  : {results['bicubic_baseline']['psnr_mean']:.4f} +/- {results['bicubic_baseline']['psnr_std']:.4f} dB", flush=True)
    print(f"    SSIM  : {results['bicubic_baseline']['ssim_mean']:.4f} +/- {results['bicubic_baseline']['ssim_std']:.4f}", flush=True)
    print(f"    LPIPS : {results['bicubic_baseline']['lpips_mean']:.4f} +/- {results['bicubic_baseline']['lpips_std']:.4f}", flush=True)
    print(f"  NAFNet-SR (Ours):", flush=True)
    print(f"    PSNR  : {results['nafnet_sr']['psnr_mean']:.4f} +/- {results['nafnet_sr']['psnr_std']:.4f} dB", flush=True)
    print(f"    SSIM  : {results['nafnet_sr']['ssim_mean']:.4f} +/- {results['nafnet_sr']['ssim_std']:.4f}", flush=True)
    print(f"    LPIPS : {results['nafnet_sr']['lpips_mean']:.4f} +/- {results['nafnet_sr']['lpips_std']:.4f}", flush=True)
    print(f"  Net Delta (NAFNet-SR vs. Bicubic):", flush=True)
    print(f"    Delta PSNR  : +{results['nafnet_sr']['psnr_mean'] - results['bicubic_baseline']['psnr_mean']:.4f} dB", flush=True)
    print(f"    Delta SSIM  : +{results['nafnet_sr']['ssim_mean'] - results['bicubic_baseline']['ssim_mean']:.4f}", flush=True)
    print(f"    Delta LPIPS : {results['nafnet_sr']['lpips_mean'] - results['bicubic_baseline']['lpips_mean']:.4f}", flush=True)
    print(f"Evaluation completed in {elapsed:.2f}s.", flush=True)
    print("=" * 75, flush=True)

    return results

if __name__ == "__main__":
    evaluate_validation_set()
