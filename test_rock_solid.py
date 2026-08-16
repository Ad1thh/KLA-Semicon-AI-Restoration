import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return (x - u) / torch.sqrt(s + self.eps) * self.weight + self.bias

class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        return y + x * self.gamma

class NAFNetSR(nn.Module):
    def __init__(self, width: int = 32, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.intro = nn.Conv2d(1, width, 3, padding=1)
        
        # 4-stage encoder
        self.enc1 = nn.Sequential(*[NAFBlock(width) for _ in range(2)])
        self.down1 = nn.Conv2d(width, width * 2, 2, stride=2)
        
        self.enc2 = nn.Sequential(*[NAFBlock(width * 2) for _ in range(2)])
        self.down2 = nn.Conv2d(width * 2, width * 4, 2, stride=2)

        self.enc3 = nn.Sequential(*[NAFBlock(width * 4) for _ in range(4)])
        self.down3 = nn.Conv2d(width * 4, width * 8, 2, stride=2)

        self.middle = nn.Sequential(*[NAFBlock(width * 8) for _ in range(6)])

        self.up3 = nn.Sequential(nn.Conv2d(width * 8, width * 16, 1), nn.PixelShuffle(2))
        self.dec3 = nn.Sequential(nn.Conv2d(width * 8, width * 4, 1), *[NAFBlock(width * 4) for _ in range(4)])

        self.up2 = nn.Sequential(nn.Conv2d(width * 4, width * 8, 1), nn.PixelShuffle(2))
        self.dec2 = nn.Sequential(nn.Conv2d(width * 4, width * 2, 1), *[NAFBlock(width * 2) for _ in range(2)])

        self.up1 = nn.Sequential(nn.Conv2d(width * 2, width * 4, 1), nn.PixelShuffle(2))
        self.dec1 = nn.Sequential(nn.Conv2d(width * 2, width, 1), *[NAFBlock(width) for _ in range(2)])

        self.up_head = nn.Sequential(
            nn.Conv2d(width, 1 * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
        feat = self.intro(x)
        
        e1 = self.enc1(feat)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)
        d3 = self.down3(e3)

        mid = self.middle(d3)

        u3 = self.up3(mid)
        c3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(c3)
        c2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, e1], dim=1))

        res = self.up_head(c1)
        return torch.clamp(base + res, 0.0, 1.0)

# Smooth, 100% differentiable losses
class SmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from pytorch_msssim import SSIM
        self.ssim_mod = SSIM(data_range=1.0, size_average=True, channel=1)

    def forward(self, pred, target):
        p = pred.float()
        t = target.float()
        
        # Charbonnier
        diff = p - t
        l_charb = torch.mean(torch.sqrt(diff * diff + 1e-6))

        # FFT Charbonnier
        pf = torch.fft.rfft2(p, norm='ortho')
        tf = torch.fft.rfft2(t, norm='ortho')
        dr = pf.real - tf.real
        di = pf.imag - tf.imag
        l_fft = torch.mean(torch.sqrt(dr * dr + di * di + 1e-6))

        # SSIM Loss
        p_clamp = torch.clamp(p, 0.0, 1.0)
        t_clamp = torch.clamp(t, 0.0, 1.0)
        l_ssim = 1.0 - self.ssim_mod(p_clamp, t_clamp)

        total = 1.0 * l_charb + 0.5 * l_ssim + 0.1 * l_fft
        return total, {"charb": l_charb.item(), "fft": l_fft.item(), "ssim": l_ssim.item()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, flush=True)

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

model = NAFNetSR(width=32, scale_factor=2).to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Params: {params:,} ({params/1e6:.2f}M)", flush=True)

loss_fn = SmoothLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# Test 100 consecutive random synthetic batches (with extreme values)
print("\n--- Running 100 Iterations with extreme values & zero patches ---", flush=True)
t0 = time.time()
for i in range(1, 101):
    # Degraded: float32, speckle noise > 1.0, some patches zero
    if i % 10 == 0:
        x = torch.zeros(8, 1, 128, 128, device=device) # Zero patch stress test
        gt = torch.zeros(8, 1, 256, 256, device=device)
    else:
        x = torch.randn(8, 1, 128, 128, device=device) * 0.3 + 0.5
        gt = torch.rand(8, 1, 256, 256, device=device)

    optimizer.zero_grad()
    out = model(x)
    loss, d = loss_fn(out, gt)
    assert not torch.isnan(loss), f"NaN at iter {i}!"
    assert not torch.isinf(loss), f"Inf at iter {i}!"
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if i % 20 == 0 or i == 1:
        print(f"Iter {i:03d}/100 | Loss: {loss.item():.5f} | Charb: {d['charb']:.5f} | FFT: {d['fft']:.5f} | SSIM: {d['ssim']:.5f}", flush=True)

dt = time.time() - t0
print(f"\n100 iterations PASSED in {dt:.2f}s ({dt/100:.3f}s/iter) with ZERO NaNs!")
print(f"Full 320-batch epoch will take ONLY {dt/100 * 320:.1f}s ({dt/100 * 320 / 60:.2f} mins)!")
