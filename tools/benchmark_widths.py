import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
from src.model import NAFNetSR
from src.loss import CompositeRestorationLoss

def benchmark_all_widths():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    print("=" * 75)
    print("NAFNet-SR Channel Width & Parameter Scaling Benchmark")
    print("=" * 75)
    print(f"Device: {device} ({gpu_name})")

    torch.set_float32_matmul_precision('high')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    for w in [16, 24, 32, 48, 64]:
        model = NAFNetSR(width=w, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2], scale_factor=2).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        criterion = CompositeRestorationLoss(w_charbonnier=1.0, w_ms_ssim=0.5, w_fft=0.1, w_lpips=0.0, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

        x = torch.randn(8, 1, 128, 128, device=device)
        gt = torch.rand(8, 1, 256, 256, device=device)

        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss, _ = criterion(out, gt)
            loss.backward()
            optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            optimizer.zero_grad()
            out = model(x)
            loss, _ = criterion(out, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 10.0
        epoch_min = (dt * 320) / 60.0
        print(f"Width: {w:2d} | Trainable Params: {params/1e6:6.2f}M ({params:,}) | Step Time: {dt:.3f}s | Epoch: {epoch_min:.1f}m", flush=True)

    print("=" * 75)

if __name__ == "__main__":
    benchmark_all_widths()
