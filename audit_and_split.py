import os
import glob
import shutil
import random

def audit_and_split_dataset(data_dir: str = "./data", val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)

    train_gt_dir = os.path.join(data_dir, "train", "gt")
    train_deg_dir = os.path.join(data_dir, "train", "degraded")
    val_gt_dir = os.path.join(data_dir, "val", "gt")
    val_deg_dir = os.path.join(data_dir, "val", "degraded")

    # Locate source GT and NoisyLR directories
    root_gt_dir = "./train/GT"
    root_deg_dir = "./train/NoisyLR"
    if not os.path.exists(root_deg_dir):
        root_deg_dir = "./NoisyLR"

    root_gts = {os.path.basename(f): f for f in glob.glob(os.path.join(root_gt_dir, "*.npy"))}
    root_degs = {os.path.basename(f): f for f in glob.glob(os.path.join(root_deg_dir, "*.npy"))}

    common_keys = sorted(list(set(root_gts.keys()) & set(root_degs.keys())))
    total_samples = len(common_keys)
    print(f"[Dataset Audit] Total paired images in source dataset: {total_samples}")

    # Deterministic 80/20 train/val split (seed=42)
    n_val = int(round(total_samples * val_ratio))
    n_train = total_samples - n_val

    rng = random.Random(seed)
    shuffled_keys = list(common_keys)
    rng.shuffle(shuffled_keys)

    val_keys = set(shuffled_keys[:n_val])
    train_keys = set(shuffled_keys[n_val:])

    print(f"[Dataset Split] Partitioning: {len(train_keys)} Train ({(1-val_ratio)*100:.0f}%), {len(val_keys)} Val ({val_ratio*100:.0f}%) with seed={seed}")

    # Remove existing files in data/val and data/train that do not belong
    for fn in glob.glob(os.path.join(val_gt_dir, "*.npy")):
        if os.path.basename(fn) not in val_keys:
            os.remove(fn)
    for fn in glob.glob(os.path.join(val_deg_dir, "*.npy")):
        if os.path.basename(fn) not in val_keys:
            os.remove(fn)
    for fn in glob.glob(os.path.join(train_gt_dir, "*.npy")):
        if os.path.basename(fn) not in train_keys:
            os.remove(fn)
    for fn in glob.glob(os.path.join(train_deg_dir, "*.npy")):
        if os.path.basename(fn) not in train_keys:
            os.remove(fn)

    # Populate data/train
    for fn in train_keys:
        dst_gt = os.path.join(train_gt_dir, fn)
        dst_deg = os.path.join(train_deg_dir, fn)
        if not os.path.exists(dst_gt):
            shutil.copy2(root_gts[fn], dst_gt)
        if not os.path.exists(dst_deg):
            shutil.copy2(root_degs[fn], dst_deg)

    # Populate data/val
    for fn in val_keys:
        dst_gt = os.path.join(val_gt_dir, fn)
        dst_deg = os.path.join(val_deg_dir, fn)
        if not os.path.exists(dst_gt):
            shutil.copy2(root_gts[fn], dst_gt)
        if not os.path.exists(dst_deg):
            shutil.copy2(root_degs[fn], dst_deg)

    # Verify counts
    final_train_gt = sorted(glob.glob(os.path.join(train_gt_dir, "*.npy")))
    final_train_deg = sorted(glob.glob(os.path.join(train_deg_dir, "*.npy")))
    final_val_gt = sorted(glob.glob(os.path.join(val_gt_dir, "*.npy")))
    final_val_deg = sorted(glob.glob(os.path.join(val_deg_dir, "*.npy")))

    print("\n" + "=" * 60)
    print("--- [Audit & Split Summary (Seed=42)] ---")
    print(f"  Train Set: {len(final_train_gt)} GT, {len(final_train_deg)} Degraded (Ratio: {len(final_train_gt)/total_samples*100:.1f}%)")
    print(f"  Val Set:   {len(final_val_gt)} GT, {len(final_val_deg)} Degraded (Ratio: {len(final_val_gt)/total_samples*100:.1f}%)")
    print(f"  Total:     {len(final_train_gt) + len(final_val_gt)} images")
    print("=" * 60 + "\n")

    # Record to claude-mem
    import yaml
    os.makedirs("./logs", exist_ok=True)
    state_file = "./logs/claude_mem_state.yaml"
    state = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = yaml.safe_load(f) or {}
        except Exception:
            state = {}
            
    state["dataset_audit_split"] = {
        "train_samples": len(final_train_gt),
        "val_samples": len(final_val_gt),
        "total_samples": len(final_train_gt) + len(final_val_gt),
        "val_ratio": val_ratio,
        "seed": seed,
        "train_gt_dir": train_gt_dir,
        "train_degraded_dir": train_deg_dir,
        "val_gt_dir": val_gt_dir,
        "val_degraded_dir": val_deg_dir
    }
    with open(state_file, "w") as f:
        yaml.dump(state, f, default_flow_style=False)
    print(f"[claude-mem] Dataset split audit successfully recorded to {state_file}")

if __name__ == "__main__":
    audit_and_split_dataset()
