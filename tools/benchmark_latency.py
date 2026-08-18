import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch
from src.model import NAFNetSR

def benchmark_inference_latency(weights_path: str = "weights/nafnet_sr_best.pt"):
    if not os.path.exists(weights_path):
        weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "nafnet_sr_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    
    print("=" * 70, flush=True)
    print("NAFNet-SR GPU-Labeled Latency & Throughput Benchmark", flush=True)
    print("=" * 70, flush=True)
    print(f"Device               : {device}", flush=True)
    print(f"GPU Hardware Name    : {gpu_name}", flush=True)
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Memory (Total)   : {total_vram_gb:.2f} GB", flush=True)
        print(f"CUDA Version (Torch) : {torch.version.cuda}", flush=True)
    print("-" * 70, flush=True)

    model = NAFNetSR(
        in_channels=1,
        out_channels=1,
        width=32,
        enc_blk_nums=[1, 2, 4, 8],
        middle_blk_num=4,
        dec_blk_nums=[1, 1, 2, 2],
        scale_factor=2
    ).to(device)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint    : {weights_path}", flush=True)

    model.eval()

    # 1. Single Image Latency (Batch Size = 1)
    x_single = torch.randn(1, 1, 128, 128, device=device)
    
    with torch.no_grad():
        for _ in range(25):
            _ = model(x_single)
            if device.type == "cuda":
                torch.cuda.synchronize()

    single_latencies = []
    num_single_runs = 100
    with torch.no_grad():
        for _ in range(num_single_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x_single)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            single_latencies.append((t1 - t0) * 1000.0)

    single_lat_mean = float(np.mean(single_latencies))
    single_lat_std = float(np.std(single_latencies))
    single_fps = 1000.0 / single_lat_mean

    print(f"\n[Single-Image Inference (1x1x128x128 -> 1x1x256x256)] (N={num_single_runs})", flush=True)
    print(f"  Latency (mean ± std): {single_lat_mean:.2f} ± {single_lat_std:.2f} ms", flush=True)
    print(f"  Throughput (FPS)    : {single_fps:.1f} FPS", flush=True)

    # 2. Batched Inference (Batch Size = 8)
    x_batch = torch.randn(8, 1, 128, 128, device=device)
    
    with torch.no_grad():
        for _ in range(15):
            _ = model(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()

    batch_latencies = []
    num_batch_runs = 50
    with torch.no_grad():
        for _ in range(num_batch_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_latencies.append((t1 - t0) * 1000.0)

    batch_lat_mean = float(np.mean(batch_latencies))
    batch_lat_std = float(np.std(batch_latencies))
    per_img_batch_lat = batch_lat_mean / 8.0
    batch_fps = (8.0 * 1000.0) / batch_lat_mean

    print(f"\n[Batched Inference (Batch Size = 8)] (N={num_batch_runs})", flush=True)
    print(f"  Batch Latency (mean ± std) : {batch_lat_mean:.2f} ± {batch_lat_std:.2f} ms", flush=True)
    print(f"  Per-Image Effective Latency: {per_img_batch_lat:.2f} ms", flush=True)
    print(f"  Batched Throughput (FPS)   : {batch_fps:.1f} FPS", flush=True)

    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"\n[Memory Footprint]", flush=True)
        print(f"  Peak VRAM Allocated        : {peak_vram_mb:.1f} MB", flush=True)

    print("=" * 70, flush=True)

if __name__ == "__main__":
    benchmark_inference_latency()
