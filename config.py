# config.py
# Central configuration file for Smart Waste Management System
# All settings, paths, and constants are defined here

import os

# ─────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")        # Raw dataset folder
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")    # Preprocessed data
MODEL_DIR = os.path.join(BASE_DIR, "models")           # Saved models
REPORT_DIR = os.path.join(BASE_DIR, "reports")         # Generated reports
LOG_DIR = os.path.join(BASE_DIR, "logs")               # Audit logs

# Model save path
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "waste_classifier.h5")

# ─────────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────────

# TrashNet categories
WASTE_CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(WASTE_CATEGORIES)

# Image settings
IMAGE_SIZE = (224, 224)       # Width x Height
IMAGE_CHANNELS = 3            # RGB
INPUT_SHAPE = (224, 224, 3)   # For model input

# Train/Validation split ratio
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# ─────────────────────────────────────────────
# MODEL / TRAINING SETTINGS
# ─────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5

# ─────────────────────────────────────────────
# CLASSIFICATION SETTINGS
# ─────────────────────────────────────────────

# Minimum confidence to accept a prediction (below this = "uncertain")
CONFIDENCE_THRESHOLD = 0.6

# Recyclable vs Non-recyclable mapping
RECYCLABLE_MAP = {
    "cardboard": True,
    "glass": True,
    "metal": True,
    "paper": True,
    "plastic": True,
    "trash": False
}

# ─────────────────────────────────────────────
# REPORT SETTINGS
# ─────────────────────────────────────────────
REPORT_FILE = os.path.join(REPORT_DIR, "waste_report.txt")

# ─────────────────────────────────────────────
# LOG SETTINGS
# ─────────────────────────────────────────────
LOG_FILE = os.path.join(LOG_DIR, "audit_log.txt")
LOG_FORMAT = "{timestamp} | {image} | {category} | {confidence:.2f}% | {recyclable}"

# ─────────────────────────────────────────────
# AUTO-CREATE DIRECTORIES IF MISSING
# ─────────────────────────────────────────────
for directory in [DATASET_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)
