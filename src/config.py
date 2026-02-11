import os

# ============================================================
# BASE PATH (absolute path, production safe)
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# DATA PATHS
# ============================================================

BASE_DATA_PATH = os.path.join(BASE_DIR, "data")

TRAIN_DIR = os.path.join(BASE_DATA_PATH, "train")
VAL_DIR   = os.path.join(BASE_DATA_PATH, "val")
TEST_DIR  = os.path.join(BASE_DATA_PATH, "test")

# ============================================================
# CHECKPOINTS AND LOGGING PATHS (production safe)
# ============================================================

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# BackupAndRestore will save crash-recovery checkpoints here
BACKUP_DIR = os.path.join(CHECKPOINT_DIR, "backup")

# Final best model will be saved here
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "dr_model_final.keras")

# Training logs
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ============================================================
# HYPERPARAMETERS (optimized for NVIDIA T1000)
# ============================================================

# MobileNetV2 works best at 224
IMG_SIZE = (224, 224)

# Safe batch size for 4GB VRAM GPU
BATCH_SIZE = 16

NUM_CLASSES = 5

# Recommended epochs
EPOCHS = 50

# Learning rate for transfer learning
INITIAL_LR = 1e-4

# Minimum LR for ReduceLROnPlateau
MIN_LR = 1e-6

# ============================================================
# HARDWARE SETTINGS
# ============================================================

# Enable mixed precision for NVIDIA GPU speedup
MIXED_PRECISION = True
