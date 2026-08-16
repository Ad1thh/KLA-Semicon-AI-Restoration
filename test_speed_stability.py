import time
import torch
from src.model import NAFNetSR
from src.loss import CompositeRestorationLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, torch.cuda.get_device_name(0))

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

model = NAFNetSR(width=64, scale_factor=2).to(device)
criterion = CompositeRestorationLoss(w_charbonnier=1.0, w_ms_ssim=0.5, w_fft=0.1, w_lpips=0.0, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

x = torch.randn(8, 1, 128, 128, device=device)
gt = torch.rand(8, 1, 256, 256, device=device)

# Warmup
for _ in range(3):
    optimizer.zero_grad()
    out = model(x)
    loss, d = criterion(out, gt)
    loss.backward()
    optimizer.step()

# Benchmark 20 iterations
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    optimizer.zero_grad()
    out = model(x)
    loss, d = criterion(out, gt)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
torch.cuda.synchronize()
elapsed = time.time() - t0
sec_per_batch = elapsed / 20.0
epoch_sec = sec_per_batch * 320

print(f"Batch time: {sec_per_batch:.3f}s | Full Epoch (320 batches): {epoch_sec:.1f}s ({epoch_sec/60:.1f} mins) | Loss: {loss.item():.5f} | Has NaN: {torch.isnan(loss).item()}")
