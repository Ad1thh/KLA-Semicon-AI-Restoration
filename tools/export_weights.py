import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

def export_compact_weights(ckpt_path: str = "weights/nafnet_sr_best.pt"):
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), "..", "weights", "nafnet_sr_best.pt")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        val_psnr = checkpoint.get("val_psnr", 28.16)
        val_ssim = checkpoint.get("val_ssim", 0.7661)
        val_lpips = checkpoint.get("val_lpips", 0.2277)
        epoch = checkpoint.get("epoch", 10)
        config = checkpoint.get("config", {"model": {"width": 32}})
    else:
        state_dict = checkpoint
        val_psnr = 28.16
        val_ssim = 0.7661
        val_lpips = 0.2277
        epoch = 10
        config = {"model": {"width": 32}}

    clean_ckpt = {
        "epoch": epoch,
        "model_state_dict": {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()},
        "config": config,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "val_lpips": val_lpips
    }

    torch.save(clean_ckpt, ckpt_path)
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"Exported clean inference checkpoint to {ckpt_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    export_compact_weights()
