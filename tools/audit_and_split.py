import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import shutil
import random
import yaml

def audit_and_split(
    degraded_dir: str = "./data/train/degraded",
    gt_dir: str = "./data/train/gt",
    val_degraded_dir: str = "./data/val/degraded",
    val_gt_dir: str = "./data/val/gt",
    val_ratio: float = 0.2,
    seed: int = 42
):
    if not os.path.exists(degraded_dir):
        degraded_dir = os.path.join(os.path.dirname(__file__), "..", "data", "train", "degraded")
    if not os.path.exists(gt_dir):
        gt_dir = os.path.join(os.path.dirname(__file__), "..", "data", "train", "gt")
    if not os.path.isabs(val_degraded_dir):
        val_degraded_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", val_degraded_dir))
    if not os.path.isabs(val_gt_dir):
        val_gt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", val_gt_dir))

    random.seed(seed)
    deg_files = sorted(glob.glob(os.path.join(degraded_dir, "*.npy")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))

    deg_dict = {os.path.basename(f): f for f in deg_files}
    gt_dict = {os.path.basename(f): f for f in gt_files}

    common_keys = sorted(list(set(deg_dict.keys()) & set(gt_dict.keys())))
    print(f"Total matching pairs found: {len(common_keys)}")

    os.makedirs(val_degraded_dir, exist_ok=True)
    os.makedirs(val_gt_dir, exist_ok=True)

    val_count = int(len(common_keys) * val_ratio)
    shuffled = common_keys.copy()
    random.shuffle(shuffled)
    val_keys = set(shuffled[:val_count])
    train_keys = set(shuffled[val_count:])

    for k in val_keys:
        shutil.copy2(deg_dict[k], os.path.join(val_degraded_dir, k))
        shutil.copy2(gt_dict[k], os.path.join(val_gt_dir, k))

    print(f"Audit & Split Complete: {len(train_keys)} Train, {len(val_keys)} Val (seed={seed})")

if __name__ == "__main__":
    audit_and_split()
