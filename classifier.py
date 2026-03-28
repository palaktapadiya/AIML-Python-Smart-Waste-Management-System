# classifier.py
# Takes an image as input and predicts the waste category
# Uses the trained model saved by model_trainer.py

import numpy as np
from PIL import Image
from tensorflow import keras
from config import (
    MODEL_SAVE_PATH, WASTE_CATEGORIES,
    IMAGE_SIZE, CONFIDENCE_THRESHOLD,
    RECYCLABLE_MAP
)


def load_model():
    """
    Loads the trained model from disk.

    Returns:
        model: Loaded Keras model
    """
    try:
        model = keras.models.load_model(MODEL_SAVE_PATH)
        print(f"[CLASSIFIER] Model loaded from: {MODEL_SAVE_PATH}")
        return model
    except Exception as e:
        raise FileNotFoundError(
            f"[CLASSIFIER] Model not found at {MODEL_SAVE_PATH}.\n"
            f"Please run model_trainer.py first to train and save the model.\n"
            f"Error: {e}"
        )


def preprocess_image(image_path):
    """
    Loads and preprocesses a single image for prediction.

    Args:
        image_path (str): Path to the image file

    Returns:
        img_array (np.array): Preprocessed image array shape (1, 224, 224, 3)
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img, dtype="float32")
        img_array = img_array / 255.0           # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        raise ValueError(f"[CLASSIFIER] Could not process image: {image_path}\nError: {e}")


def classify(image_path, model):
    """
    Classifies a single waste image.

    Args:
        image_path (str): Path to the image
        model     : Loaded Keras model

    Returns:
        result (dict): Contains category, confidence, recyclable, and all scores
    """
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Get predictions (array of 6 probabilities)
    predictions = model.predict(img_array, verbose=0)
    scores = predictions[0]  # Shape: (6,)

    # Get the highest confidence index
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    best_category = WASTE_CATEGORIES[best_index]

    # Check confidence threshold
    if best_score < CONFIDENCE_THRESHOLD:
        category = "uncertain"
        recyclable = "unknown"
    else:
        category = best_category
        recyclable = "Recyclable" if RECYCLABLE_MAP[category] else "Non-Recyclable"

    # Build result dictionary
    result = {
        "image"      : image_path,
        "category"   : category,
        "confidence" : round(best_score * 100, 2),
        "recyclable" : recyclable,
        "all_scores" : {
            WASTE_CATEGORIES[i]: round(float(scores[i]) * 100, 2)
            for i in range(len(WASTE_CATEGORIES))
        }
    }

    return result


def classify_multiple(image_paths, model):
    """
    Classifies a list of images one by one.

    Args:
        image_paths (list): List of image file paths
        model: Loaded Keras model

    Returns:
        results (list): List of result dictionaries
    """
    results = []
    print(f"\n[CLASSIFIER] Classifying {len(image_paths)} image(s)...\n")

    for idx, image_path in enumerate(image_paths, 1):
        print(f"  [{idx}/{len(image_paths)}] Processing: {image_path}")
        result = classify(image_path, model)
        results.append(result)
        print(f"  → Category   : {result['category']}")
        print(f"  → Confidence : {result['confidence']}%")
        print(f"  → Recyclable : {result['recyclable']}\n")

    return results


def print_all_scores(result):
    """
    Prints the confidence scores for all categories for a given result.

    Args:
        result (dict): Result from classify()
    """
    print(f"\n[CLASSIFIER] All scores for: {result['image']}")
    print(f"  {'Category':<12} {'Score':>8}")
    print(f"  {'-'*22}")
    for category, score in result["all_scores"].items():
        marker = " ← predicted" if category == result["category"] else ""
        print(f"  {category:<12} {score:>7.2f}%{marker}")


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python classifier.py <image_path>")
        print("Example: python classifier.py dataset/glass/glass1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_model()
    result = classify(image_path, model)

    print("\n===== CLASSIFICATION RESULT =====")
    print(f"  Image      : {result['image']}")
    print(f"  Category   : {result['category']}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  Recyclable : {result['recyclable']}")
    print("=================================")
    print_all_scores(result)
