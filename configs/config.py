import os
import torch

"""
Configuration constants for the colorization project.
Defines paths, model hyperparameters, and training settings.
"""

# Main path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw ImageNet folder
RAW_DATA_DIR = os.path.join(
    BASE_DIR, "imagenet-object-localization-challenge", "ILSVRC", "Data", "CLS-LOC", "train"
)

# Main data folders
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGENET_FULL_DIR = os.path.join(DATA_DIR, "imagenet_full")
PROCESSED_DIR = os.path.join(IMAGENET_FULL_DIR, "processed")

# Validation
VAL_DIR = os.path.join(BASE_DIR, "imagenet-object-localization-challenge", "ILSVRC", "Data", "CLS-LOC", "val")
VAL_MAPPING_CSV = os.path.join(BASE_DIR, "imagenet-object-localization-challenge", "LOC_val_solution.csv")

# Results
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Auxiliary files (created by prepare_data)
CLUSTER_CENTERS_PATH = os.path.join(IMAGENET_FULL_DIR, "cluster_centers.npy")
CLASS_WEIGHTS_PATH = os.path.join(IMAGENET_FULL_DIR, "class_weights.npy")
TEST_SUBSET_PATH = os.path.join(DATA_DIR, "test_subset_1000.json")

# Model config
NUM_COLOR_CLASSES = 313
NUM_CLASSES = 313
IMAGE_SIZE = 224
CHANNELS_IN = 1
CHANNELS_OUT = 313

# Training hyperparameters
DATA_SUBSET_SIZE = None   # None = Full ImageNet (~1.3M images)
RESUME_FROM = None     # e.g. os.path.join(CHECKPOINTS_DIR, "checkpoint_best_11_12_2.pth.tar") or None
TRAIN_NUM_SHARDS = None  # None = Train on FULL dataset or e.g. 100
BATCH_SIZE = 128       
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4          # Number of data loading processes
USE_AMP = True            # Automatic Mixed Precision (faster on GPU)   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Colorization logic
T_ANNEAL = 0.4    # Temperature for color decoding
CLASS_REBAL_LAMBDA = 0.5  # Weight for rare colors
