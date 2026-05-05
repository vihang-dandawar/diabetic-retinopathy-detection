"""
data_loader.py — dataset inspection utilities.

Run directly to print class distribution of your processed dataset:
    python data_loader.py
"""
import os
import numpy as np

try:
    import config as cfg
    TRAIN_DIR = os.path.join(cfg.BASE_DIR, "data_processed", "train")
    VAL_DIR   = os.path.join(cfg.BASE_DIR, "data_processed", "val")
    TEST_DIR  = os.path.join(cfg.BASE_DIR, "data_processed", "test")
except Exception:
    TRAIN_DIR = os.path.join("data_processed", "train")
    VAL_DIR   = os.path.join("data_processed", "val")
    TEST_DIR  = os.path.join("data_processed", "test")


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]



# splits  the dataset  into classes and count the images
def count_split(split_dir):
    if not os.path.exists(split_dir):
        print(f"  [not found] {split_dir}")
        return {}
    counts = {}
    for cls in sorted(os.listdir(split_dir)):
        p = os.path.join(split_dir, cls)
        if os.path.isdir(p):
            counts[cls] = len([
                f for f in os.listdir(p)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
    return counts

# calculates % of each class 
def print_distribution(split_name, counts):
    total = sum(counts.values())
    if total == 0:
        print(f"\n{split_name}: empty")
        return
    print(f"\n{split_name} ({total} images):")
    for cls, n in sorted(counts.items()):
        label = CLASS_NAMES[int(cls)] if cls.isdigit() and int(cls) < len(CLASS_NAMES) else cls
        bar   = "#" * int(30 * n / total)
        pct   = 100 * n / total
        print(f"  Class {cls} ({label:>15}): {n:5d}  {bar:<30}  {pct:.1f}%")

# calculates total steps required for 1 epoch 
def recommend_steps(n_train, batch_size):
    steps = n_train // batch_size
    print(f"\n>>> Recommended steps_per_epoch = {steps} "
          f"(= {n_train} images / batch {batch_size})")
    print(f"    (train.py now sets this automatically)")


if __name__ == "__main__":
    train_counts = count_split(TRAIN_DIR)
    val_counts   = count_split(VAL_DIR)
    test_counts  = count_split(TEST_DIR)

    print_distribution("Train", train_counts)
    print_distribution("Val",   val_counts)
    print_distribution("Test",  test_counts)

    n_train = sum(train_counts.values())
    try:
        recommend_steps(n_train, cfg.BATCH_SIZE)
    except Exception:
        recommend_steps(n_train, 16)
