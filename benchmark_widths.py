import time
import torch
from src.model import NAFNetSR
from src.loss import CompositeRestorationLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

for w in [32, 48, 64]:
    model = NAFNetSR(width=w, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2], scale_factor=2).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = CompositeRestorationLoss(w_charbonnier=1.0, w_ms_ssim=0.5, w_fft=0.1, w_lpips=0.0, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    x = torch.randn(8, 1, 128, 128, device=device)
    gt = torch.rand(8, 1, 256, 256, device=device)

    # 3 warmup
    for _ in range(3):
        optimizer.zero_grad()
        out = model(x)
        loss, _ = criterion(out, gt)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss, _ = criterion(out, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 10.0
    epoch_min = (dt * 320) / 60.0
    print(f"Width: {w:2d} | Params: {params/1e6:.2f}M | Batch: {dt:.3f}s | Epoch: {epoch_min:.1f} mins | Loss: {loss.item():.5f}", flush=True)
