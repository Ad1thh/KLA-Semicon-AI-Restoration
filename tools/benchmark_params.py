import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
from src.model import NAFNetSR

def print_parameter_count(config_path: str = "configs/config.yaml"):
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    width = model_cfg.get("width", 32)
    enc_blk_nums = model_cfg.get("enc_blk_nums", [2, 2, 4, 8])
    middle_blk_num = model_cfg.get("middle_blk_num", 12)
    dec_blk_nums = model_cfg.get("dec_blk_nums", [2, 2, 2, 2])
    scale_factor = config.get("dataset", {}).get("scale_factor", 2)

    model = NAFNetSR(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        width=width,
        enc_blk_nums=enc_blk_nums,
        middle_blk_num=middle_blk_num,
        dec_blk_nums=dec_blk_nums,
        scale_factor=scale_factor
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("NAFNet-SR Parameter Count Benchmark")
    print("=" * 60)
    print(f"Architecture         : NAFNet-SR")
    print(f"Configuration        : width={width}, enc={enc_blk_nums}, mid={middle_blk_num}, dec={dec_blk_nums}")
    print(f"Scale Factor         : {scale_factor}x (128x128 -> 256x256)")
    print(f"Total Parameters     : {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters : {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print("=" * 60)

    return trainable_params

if __name__ == "__main__":
    print_parameter_count()
