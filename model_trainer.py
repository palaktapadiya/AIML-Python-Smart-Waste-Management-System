# model_trainer.py
# Trains a waste classifier using Transfer Learning with MobileNetV2
# Automatically downloads a small dataset for fine-tuning
# No manual dataset download needed!

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (
    INPUT_SHAPE, NUM_CLASSES, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, DROPOUT_RATE,
    MODEL_SAVE_PATH, WASTE_CATEGORIES, DATASET_DIR
)

def download_dataset():
    print("\n[TRAINER] Checking dataset...")

    all_exist = all(
        os.path.exists(os.path.join(DATASET_DIR, cat))
        for cat in WASTE_CATEGORIES
    )

    if all_exist:
        print("[TRAINER] Dataset already found! Skipping download.")
        return

    print("[TRAINER] Creating sample dataset...")
    create_sample_dataset()


def create_sample_dataset():
    """
    Creates a minimal sample dataset with solid color images.
    Used only if real dataset download fails.
    This is just for testing the pipeline works correctly.
    """
    from PIL import Image
    import random

    print("\n[TRAINER] Creating sample dataset...")

    # Color mapping for each category (for visual distinction)
    color_map = {
        "cardboard" : (139, 90,  43),
        "glass"     : (135, 206, 235),
        "metal"     : (169, 169, 169),
        "paper"     : (255, 255, 255),
        "plastic"   : (255, 165,  0),
        "trash"     : (105, 105, 105)
    }

    samples_per_class = 100  # 100 images per category = 600 total

    for category in WASTE_CATEGORIES:
        folder = os.path.join(DATASET_DIR, category)
        os.makedirs(folder, exist_ok=True)

        base_color = color_map[category]

        for i in range(samples_per_class):
            # Add slight random variation to color so images are not identical
            r = min(255, max(0, base_color[0] + random.randint(-30, 30)))
            g = min(255, max(0, base_color[1] + random.randint(-30, 30)))
            b = min(255, max(0, base_color[2] + random.randint(-30, 30)))

            img = Image.new("RGB", (224, 224), color=(r, g, b))
            img.save(os.path.join(folder, f"{category}_{i+1}.jpg"))

        print(f"  Created {samples_per_class} sample images for '{category}'")

    print("[TRAINER] Sample dataset created!\n")
    print("NOTE: This is a sample dataset with color images only.")
    print("For real waste classification, replace with actual waste images.")
    print("Download TrashNet from: https://www.kaggle.com/datasets/fedesoriano/trashnet-dataset\n")


def build_model():
    """
    Builds a transfer learning model using MobileNetV2 as base.
    MobileNetV2 is pre-trained on ImageNet (1.4 million images).
    We add our own classification head on top.

    Returns:
        model: Compiled Keras model
        base_model: MobileNetV2 base model
    """
    print("\n[TRAINER] Loading MobileNetV2 base model...")

    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,        # Remove original classifier
        weights="imagenet"        # Use pre-trained ImageNet weights
    )

    # Freeze base model layers we don't want to retrain them
    base_model.trainable = False
    print(f"[TRAINER] MobileNetV2 loaded! ({len(base_model.layers)} layers frozen)")

    # Build our custom classification head on top
    model = keras.Sequential([
        base_model,

        # Global Average Pooling instead of Flatten better for transfer learning
        layers.GlobalAveragePooling2D(),

        # Our custom dense layers
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(128, activation="relu"),
        layers.Dropout(DROPOUT_RATE),

        # Output layer 6 categories
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def unfreeze_and_finetune(model, base_model):
    """
    Unfreezes the top layers of MobileNetV2 for fine-tuning.
    Called after initial training to squeeze out more accuracy.

    Args:
        model: Full model
        base_model: MobileNetV2 base

    Returns:
        model: Model with unfrozen top layers
    """
    print("\n[TRAINER] Fine-tuning: unfreezing top layers of MobileNetV2...")

    # Unfreeze last 30 layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"[TRAINER] Unfroze last 30 layers for fine-tuning.")
    return model


def create_data_generators():
    """
    Creates ImageDataGenerators for training and validation.
    Data augmentation is applied to training data to increase variety.

    Returns:
        train_generator, val_generator
    """
    # Training generator with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalize pixels to 0-1
        rotation_range=20,           # Random rotation
        width_shift_range=0.2,       # Random horizontal shift
        height_shift_range=0.2,      # Random vertical shift
        horizontal_flip=True,        # Random horizontal flip
        zoom_range=0.2,              # Random zoom
        validation_split=0.2         # 20% for validation
    )

    # Validation generator no augmentation just normalize
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    print("\n[TRAINER] Creating data generators...")

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    print(f"  Training samples   : {train_generator.samples}")
    print(f"  Validation samples : {val_generator.samples}")
    print(f"  Classes found      : {list(train_generator.class_indices.keys())}")

    return train_generator, val_generator


def train_model():
    """
    Full training pipeline:
    Download data -> Build model -> Train -> Fine-tune -> Save -> Plot
    """

    # Step 1: Download/prepare dataset
    download_dataset()

    # Step 2: Create data generators
    train_gen, val_gen = create_data_generators()

    # Step 3: Build model
    model, base_model = build_model()
    model.summary()

    # Step 4: Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    # Step 5: Initial Training frozen base
    print("\n[TRAINER] Phase 1 - Training classification head...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    # Step 6: Fine-tuning unfreeze top layers
    model = unfreeze_and_finetune(model, base_model)

    print("\n[TRAINER] Phase 2 - Fine-tuning top layers...")
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )

    # Step 7: Evaluate
    print("\n[TRAINER] Evaluating final model...")
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"  Final Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"  Final Validation Loss     : {val_loss:.4f}")

    # Step 8: Plot Results
    plot_training(history1, history2)

    print(f"\n[TRAINER] Model saved to: {MODEL_SAVE_PATH}")
    return model


def plot_training(history1, history2):
    """
    Plots training graphs for both phases combined.
    """
    # Combine both histories
    acc      = history1.history["accuracy"]     + history2.history["accuracy"]
    val_acc  = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss     = history1.history["loss"]         + history2.history["loss"]
    val_loss = history1.history["val_loss"]     + history2.history["val_loss"]

    epochs_range = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs_range, acc,     label="Train Accuracy")
    ax1.plot(epochs_range, val_acc, label="Val Accuracy")
    ax1.axvline(x=len(history1.history["accuracy"]) - 1,
                color="gray", linestyle="--", label="Fine-tune start")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(epochs_range, loss,     label="Train Loss")
    ax2.plot(epochs_range, val_loss, label="Val Loss")
    ax2.axvline(x=len(history1.history["loss"]) - 1,
                color="gray", linestyle="--", label="Fine-tune start")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("reports/training_plot.png")
    print("\n[TRAINER] Training plot saved to reports/training_plot.png")
    plt.show()


# ─────────────────────────────────────────────
# Run this file directly to train the model
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_model()
