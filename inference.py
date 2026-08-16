import os
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from src.model import NAFNetSR

def parse_args():
    parser = argparse.ArgumentParser(description="NAFNet-SR Semiconductor Inspection Inference Pipeline")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to directory containing degraded test images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory to save restored output images")
    return parser.parse_args()

def load_model(weights_path: str = "./weights/nafnet_sr_best.pt", device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """Initialize NAFNet-SR model and load trained weights."""
    model = NAFNetSR(
        in_channels=1,
        out_channels=1,
        width=64,
        enc_blk_nums=[2, 2, 4, 8],
        middle_blk_num=12,
        dec_blk_nums=[2, 2, 2, 2],
        scale_factor=2
    ).to(device)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        print(f"[Inference] Loaded model weights from {weights_path}")
    else:
        print(f"[Inference Warning] Weights file {weights_path} not found. Running with default initialization.")

    model.eval()
    return model

def run_inference(input_dir: str, output_dir: str):
    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Running on Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = load_model(weights_path="./weights/nafnet_sr_best.pt", device=device)

    # Search for input files (supporting .npy, .png, .jpg, .tif, etc.)
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    img_files = glob.glob(os.path.join(input_dir, "*.png")) + glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.jpeg"))
    all_files = sorted(npy_files + img_files)

    if not all_files:
        print(f"[Inference Warning] No image or .npy files found in {input_dir}")
        return

    print(f"[Inference] Processing {len(all_files)} files from '{input_dir}' -> '{output_dir}'...")

    processed_count = 0
    with torch.no_grad():
        for file_path in all_files:
            filename = os.path.basename(file_path)
            out_path = os.path.join(output_dir, filename)

            if file_path.endswith(".npy"):
                # Load degraded float32 numpy array
                arr = np.load(file_path).astype(np.float32)
                
                # Handle possible dimensions
                if arr.ndim == 2:
                    inp_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                elif arr.ndim == 3 and arr.shape[0] == 1:
                    inp_tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 1, H, W)
                elif arr.ndim == 3 and arr.shape[-1] == 1:
                    inp_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
                else:
                    inp_tensor = torch.from_numpy(arr).unsqueeze(0)

                inp_tensor = inp_tensor.to(device)
                
                # Run restoration model
                restored_tensor = model(inp_tensor)
                
                # Clamp strictly to [0.0, 1.0] as mandated by domain rules
                restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)
                
                # Convert back to numpy float32 preserving dimensions
                out_arr = restored_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                np.save(out_path, out_arr)

            else:
                # Standard image format (e.g. PNG/JPG)
                from PIL import Image
                img = Image.open(file_path).convert("L")
                arr = np.array(img, dtype=np.float32) / 255.0
                inp_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
                
                restored_tensor = model(inp_tensor)
                restored_tensor = torch.clamp(restored_tensor, 0.0, 1.0)
                
                out_arr = (restored_tensor.squeeze(0).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                Image.fromarray(out_arr).save(out_path)

            processed_count += 1

    total_time = time.time() - start_total_time
    avg_time_per_img = (total_time / processed_count) * 1000 if processed_count > 0 else 0.0
    print(f"[Inference Complete] Restored {processed_count} images in {total_time:.3f}s ({avg_time_per_img:.2f} ms/image).")

def main():
    args = parse_args()
    run_inference(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
