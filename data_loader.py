# data_loader.py
# Handles loading, preprocessing, and splitting the dataset
# Works with TrashNet dataset structure

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from config import (
    DATASET_DIR, IMAGE_SIZE, WASTE_CATEGORIES,
    TRAIN_SPLIT, VAL_SPLIT
)


def load_images_from_folder(folder_path, label_index):
    """
    Load all images from a single category folder.
    
    Args:
        folder_path (str): Path to the category folder
        label_index (int): Numeric label for this category

    Returns:
        images (list): List of numpy arrays (preprocessed images)
        labels (list): Corresponding label for each image
    """
    images = []
    labels = []
    valid_extensions = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label_index)
            except Exception as e:
                print(f"  [WARNING] Skipping {filename}: {e}")

    return images, labels


def load_dataset():
    """
    Load the full TrashNet dataset from DATASET_DIR.

    Expected folder structure:
        dataset/
            cardboard/
            glass/
            metal/
            paper/
            plastic/
            trash/

    Returns:
        X (np.array): All images, shape (N, 224, 224, 3)
        y (np.array): All labels, shape (N,)
    """
    all_images = []
    all_labels = []

    print("\n[DATA LOADER] Loading dataset...")
    print(f"  Dataset path: {DATASET_DIR}")
    print(f"  Categories  : {WASTE_CATEGORIES}\n")

    for idx, category in enumerate(WASTE_CATEGORIES):
        folder_path = os.path.join(DATASET_DIR, category)

        if not os.path.exists(folder_path):
            print(f"  [ERROR] Folder not found: {folder_path}")
            print(f"  Make sure your dataset is organized into category subfolders.")
            continue

        images, labels = load_images_from_folder(folder_path, idx)
        all_images.extend(images)
        all_labels.extend(labels)
        print(f"  Loaded {len(images):>4} images from '{category}'")

    if len(all_images) == 0:
        raise ValueError("[DATA LOADER] No images loaded. Check your dataset path and structure.")

    X = np.array(all_images, dtype="float32")
    y = np.array(all_labels, dtype="int32")

    print(f"\n  Total images loaded : {len(X)}")
    print(f"  Image shape         : {X[0].shape}")
    return X, y


def normalize(X):
    """
    Normalize pixel values from [0, 255] to [0.0, 1.0].

    Args:
        X (np.array): Raw image array

    Returns:
        np.array: Normalized image array
    """
    return X / 255.0


def split_dataset(X, y):
    """
    Split dataset into training and validation sets.

    Args:
        X (np.array): Images
        y (np.array): Labels

    Returns:
        X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SPLIT,
        random_state=42,
        stratify=y       # Ensures each class is proportionally represented
    )

    print(f"\n[DATA LOADER] Dataset split:")
    print(f"  Training samples   : {len(X_train)}")
    print(f"  Validation samples : {len(X_val)}")

    return X_train, X_val, y_train, y_val


def get_data():
    """
    Full pipeline: Load → Normalize → Split.
    Call this from model_trainer.py.

    Returns:
        X_train, X_val, y_train, y_val (all normalized)
    """
    X, y = load_dataset()
    X = normalize(X)
    X_train, X_val, y_train, y_val = split_dataset(X, y)
    return X_train, X_val, y_train, y_val


# ─────────────────────────────────────────────
# Quick test — run this file directly to verify
# ─────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_val, y_train, y_val = get_data()
    print("\n[TEST] Data loading successful!")
    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_val shape   : {X_val.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  y_val shape   : {y_val.shape}")
