import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_PATH = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(BASE_DATA_PATH, "train")
VAL_DIR   = os.path.join(BASE_DATA_PATH, "val")
TEST_DIR  = os.path.join(BASE_DATA_PATH, "test")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BACKUP_DIR = os.path.join(CHECKPOINT_DIR, "backup")
FINAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "dr_model_final_85.keras")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Standardized to 224 for MobileNetV2 stability
IMG_SIZE = (224, 224)
BATCH_SIZE = 16 # 224 resolution allows 16 batches on 2GB T1000
NUM_CLASSES = 5
EPOCHS = 200
INITIAL_LR = 1e-4
MIXED_PRECISION = True