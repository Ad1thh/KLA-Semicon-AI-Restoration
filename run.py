import os
import glob
import time
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# NAFNet-SR Architecture Definition (Self-Contained for Zero-Dependency Deployment)
# =========================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        u = x_f.mean(1, keepdim=True)
        s = (x_f - u).pow(2).mean(1, keepdim=True)
        norm = (x_f - u) / torch.sqrt(s + self.eps) * self.weight.float() + self.bias.float()
        return norm.to(orig_dtype)

class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
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
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 width: int = 32,
                 enc_blk_nums: list[int] = [1, 2, 4, 8],
                 middle_blk_num: int = 4,
                 dec_blk_nums: list[int] = [1, 1, 2, 2],
                 scale_factor: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        channels = [width * (i + 1) for i in range(len(enc_blk_nums))]
        self.intro = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, stride=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()

        for i, num in enumerate(enc_blk_nums):
            c_curr = channels[i]
            c_next = channels[i+1] if i < len(channels) - 1 else channels[-1]
            self.encoders.append(
                nn.Sequential(*[NAFBlock(c_curr) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(c_curr, c_next, kernel_size=2, stride=2)
            )

        self.middle = nn.Sequential(
            *[NAFBlock(channels[-1]) for _ in range(middle_blk_num)]
        )

        dec_channels = channels[::-1]
        for i, num in enumerate(dec_blk_nums):
            c_out = dec_channels[i]
            skip_c = dec_channels[i]
            c_prev = channels[-1] if i == 0 else dec_channels[i-1]
            
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(c_prev, c_out * 4, kernel_size=1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.skips.append(
                nn.Conv2d(c_out + skip_c, c_out, kernel_size=1, bias=True)
            )
            self.decoders.append(
                nn.Sequential(*[NAFBlock(c_out) for _ in range(num)])
            )

        self.up_head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] * (scale_factor ** 2), kernel_size=1, bias=True),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        factor = 16
        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            x_pad = x

        base = F.interpolate(x_pad, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        feat = self.intro(x_pad)
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

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H * self.scale_factor, :W * self.scale_factor]

        return torch.clamp(out, 0.0, 1.0)

# =========================================================================
# Model Loading & Inference Engine
# =========================================================================

def find_weights_path() -> str:
    """Dynamically search common weight locations."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "models", "nafnet_sr_best.pt"),
        os.path.join(base_dir, "weights", "nafnet_sr_best.pt"),
        os.path.join(base_dir, "..", "models", "nafnet_sr_best.pt"),
        os.path.join(base_dir, "..", "weights", "nafnet_sr_best.pt"),
        "./models/nafnet_sr_best.pt",
        "./weights/nafnet_sr_best.pt"
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return candidates[0]

def load_inference_model(weights_path: str, device: torch.device) -> NAFNetSR:
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
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        print(f"[Zynq Inference] Loaded weights successfully from: {weights_path}")
    else:
        print(f"[Zynq Warning] Weight checkpoint {weights_path} not found. Running default initialization.")

    model.eval()
    return model

def run_restoration(input_dir: str, output_dir: str):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[Zynq Inference] Device: {device} ({gpu_name})")

    weights_path = find_weights_path()
    model = load_inference_model(weights_path, device)

    # Gather test files (.npy, .png, .jpg, .tif, etc.)
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    img_files = (
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.jpeg")) +
        glob.glob(os.path.join(input_dir, "*.tif")) +
        glob.glob(os.path.join(input_dir, "*.tiff"))
    )
    all_files = sorted(npy_files + img_files)

    if not all_files:
        print(f"[Zynq Warning] No input files found in {input_dir}")
        return

    print(f"[Zynq Inference] Restoring {len(all_files)} images from '{input_dir}' -> '{output_dir}'...")

    processed = 0
    with torch.no_grad():
        for file_path in all_files:
            fn = os.path.basename(file_path)
            out_file = os.path.join(output_dir, fn)

            if file_path.endswith(".npy"):
                arr = np.load(file_path).astype(np.float32)
                if arr.ndim == 2:
                    inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                elif arr.ndim == 3 and arr.shape[0] == 1:
                    inp = torch.from_numpy(arr).unsqueeze(0)
                elif arr.ndim == 3 and arr.shape[-1] == 1:
                    inp = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
                else:
                    inp = torch.from_numpy(arr).unsqueeze(0)

                inp = inp.to(device)
                out = model(inp)
                out = torch.clamp(out, 0.0, 1.0)

                out_arr = out.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                np.save(out_file, out_arr)

            else:
                from PIL import Image
                img = Image.open(file_path).convert("L")
                arr = np.array(img, dtype=np.float32) / 255.0
                inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

                out = model(inp)
                out = torch.clamp(out, 0.0, 1.0)

                out_arr = (out.squeeze(0).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                Image.fromarray(out_arr).save(out_file)

            processed += 1

    elapsed = time.time() - start_time
    avg_speed = (elapsed / processed) * 1000 if processed > 0 else 0.0
    print(f"[Zynq Complete] Processed {processed} images in {elapsed:.3f}s ({avg_speed:.2f} ms/image).")

# =========================================================================
# CLI Argument Parser (Supporting both positional & flag invocations)
# =========================================================================

def parse_cli_args():
    # If standard positional arguments are provided: python run.py <input-dir> <output-dir>
    raw_args = sys.argv[1:]
    
    # Filter out flags if present
    positional = [a for a in raw_args if not a.startswith("-")]
    
    if len(positional) >= 2:
        return positional[0], positional[1]

    # Fallback to standard argparse if flags used (--input_dir, --output_dir)
    parser = argparse.ArgumentParser(description="Zynq Team Semiconductor Restoration Pipeline")
    parser.add_argument("input_dir_pos", nargs="?", default=None, help="Input directory (positional)")
    parser.add_argument("output_dir_pos", nargs="?", default=None, help="Output directory (positional)")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory (flag)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (flag)")
    args = parser.parse_args()

    inp = args.input_dir_pos or args.input_dir
    out = args.output_dir_pos or args.output_dir

    if not inp or not out:
        print("Usage: python run.py <input-dir> <output-dir>")
        print("   or: python run.py --input_dir <input-dir> --output_dir <output-dir>")
        sys.exit(1)

    return inp, out

if __name__ == "__main__":
    inp_dir, out_dir = parse_cli_args()
    run_restoration(inp_dir, out_dir)
