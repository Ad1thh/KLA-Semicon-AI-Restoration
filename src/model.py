import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """
    Channel-wise Layer Normalization for 2D feature maps (B, C, H, W).
    Mathematically exact and 100% numerically stable in FP32/TF32.
    """
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
    """Nonlinear Activation Free SimpleGate: splits channels in half and computes element-wise product."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """
    Core NAFNet Block with:
    - Depthwise Convolution (3x3)
    - SimpleGate (nonlinear activation free)
    - Simplified Channel Attention (SCA)
    - Feed-forward Network with SimpleGate
    - Learnable scale factors (beta, gamma)
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, bias=True)
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)

        # Feed-forward network (FFN)
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, stride=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class NAFNetSR(nn.Module):
    """
    Nonlinear Activation Free Network for Super-Resolution (NAFNet-SR).
    Accepts single-channel degraded grayscale images, upsamples by scale_factor (2x),
    and strictly clamps output to [0.0, 1.0].
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 width: int = 32,
                 enc_blk_nums: list[int] = [2, 2, 4, 8],
                 middle_blk_num: int = 12,
                 dec_blk_nums: list[int] = [2, 2, 2, 2],
                 scale_factor: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2)
            )
            chan = chan * 2

        self.middle = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.skips.append(nn.Conv2d(chan * 2, chan, kernel_size=1, bias=True))
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )

        self.up_head = nn.Sequential(
            nn.Conv2d(width, out_channels * (scale_factor ** 2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global residual base: bicubic upsampling of input
        base = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        feat = self.intro(x)
        enc_skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            enc_skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for up, skip_proj, decoder in zip(self.ups, self.skips, self.decoders):
            feat = up(feat)
            enc_feat = enc_skips.pop()
            feat = torch.cat([feat, enc_feat], dim=1)
            feat = skip_proj(feat)
            feat = decoder(feat)

        res = self.up_head(feat)
        out = base + res

        # Mandatory domain invariant constraint: output strictly clamped to [0.0, 1.0]
        return torch.clamp(out, 0.0, 1.0)
