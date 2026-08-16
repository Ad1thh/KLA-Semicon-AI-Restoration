import time
import torch
from src.model import NAFNetSR
from src.loss import CompositeRestorationLoss

def test_speed_and_stability():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    print("=" * 70)
    print("NAFNet-SR Training Speed & Numerical Stability Test")
    print("=" * 70)
    print(f"Device               : {device} ({gpu_name})")

    torch.set_float32_matmul_precision('high')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = NAFNetSR(width=32, scale_factor=2).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters     : {params:,} ({params/1e6:.2f}M)")

    criterion = CompositeRestorationLoss(w_charbonnier=1.0, w_ms_ssim=0.5, w_fft=0.1, w_lpips=0.0, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    x = torch.randn(8, 1, 128, 128, device=device)
    gt = torch.rand(8, 1, 256, 256, device=device)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss, _ = criterion(out, gt)
        loss.backward()
        optimizer.step()

    # Benchmark 30 training steps
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        optimizer.zero_grad()
        out = model(x)
        loss, _ = criterion(out, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    sec_per_batch = elapsed / 30.0
    epoch_sec = sec_per_batch * 320

    print(f"Batch Time (B=8)     : {sec_per_batch:.3f} s/batch")
    print(f"Epoch Est. (320 bat) : {epoch_sec:.1f} s ({epoch_sec/60:.2f} mins)")
    print(f"Step Loss            : {loss.item():.5f}")
    print(f"Numerical Stability  : {'PASSED (Zero NaNs)' if not torch.isnan(loss) else 'FAILED (NaN detected)'}")
    print("=" * 70)

if __name__ == "__main__":
    test_speed_and_stability()
