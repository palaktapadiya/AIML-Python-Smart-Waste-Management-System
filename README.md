# AIML-Python-Smart-Waste-Management-System
# 🗑️ Smart Waste Management System
### Powered by Deep Learning (CNN) + Transfer Learning (MobileNetV2)

A multi-module AI/ML project that classifies waste images into categories using
Transfer Learning with MobileNetV2, fine-tuned for waste classification.

---

## Project Overview

This system takes an image of waste as input and predicts:
- **What type of waste it is** (cardboard, glass, metal, paper, plastic, trash)
- **How confident the model is** (confidence score in %)
- **Whether it is recyclable or not**

All classifications are logged automatically and a detailed report can be generated.

---

## Project Structure

```
WasteManagement/
│
├── config.py              # All settings, paths and constants
├── data_loader.py         # Loads, preprocesses and splits the dataset
├── model_trainer.py       # Builds and trains using Transfer Learning (MobileNetV2)
├── classifier.py          # Classifies new waste images using trained model
├── audit_log.py           # Logs every classification with timestamp
├── report_generator.py    # Generates waste analysis reports
├── main.py                # Entry point — ties all modules together
│
├── dataset/               # Dataset folders (auto-created on first run)
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── models/                # Trained model gets saved here (auto-created)
├── logs/                  # Audit logs get saved here (auto-created)
├── reports/               # Reports and training graphs saved here (auto-created)
├── processed/             # Processed data folder (auto-created)
│
├── requirements.txt       # Required Python libraries
└── README.md              # This file
```

---

## Waste Categories

| Category  | Recyclable|
|-----------|-----------|
| Cardboard |    Yes    |
| Glass     |    Yes    |
| Metal     |    Yes    |
| Paper     |    Yes    |
| Plastic   |    Yes    |
| Trash     |     No    |

---

## Installation

### 1. Clone or download the project
Download all project files and place them in a folder.

### 2. Setup conda environment (recommended)
This project works best with Python 3.9 and conda:

```bash
conda create -n waste python=3.9 -y
conda activate waste
```

### 3. Install required libraries
```bash
pip install tensorflow pillow numpy scikit-learn matplotlib
```

For ARM64 systems (e.g. ARM-based Linux):
```bash
pip install tensorflow-aarch64 pillow numpy scikit-learn matplotlib
```

### 4. Dataset
The system automatically handles the dataset in one of two ways:
- **Option 1** — Tries to download TrashNet dataset from GitHub automatically
- **Option 2** — If download fails, creates a sample dataset with colored images for demonstration

To use real waste images, place your images in the correct category folders inside `dataset/`:
```
dataset/cardboard/  → cardboard images
dataset/glass/      → glass images
dataset/metal/      → metal images
dataset/paper/      → paper images
dataset/plastic/    → plastic images
dataset/trash/      → trash images
```

---

## How to Run

### Step 1 — Activate environment
```bash
conda activate waste
```

### Step 2 — Train the model
Run this first to train and save the model:
```bash
python3 model_trainer.py
```
This will:
- Automatically download or create dataset
- Load MobileNetV2 pre-trained on ImageNet
- Train classification head (Phase 1)
- Fine-tune top layers (Phase 2)
- Save best model to `models/waste_classifier.h5`
- Save training graph to `reports/training_plot.png`

Training takes around **5-15 minutes** depending on your system.

---

### Step 3 — Run the main system
After training is done, run:
```bash
python3 main.py
```

You will see a menu like this:
```
╔══════════════════════════════════════════════════╗
║       SMART WASTE MANAGEMENT SYSTEM              ║
║       Powered by Deep Learning (CNN)             ║
╚══════════════════════════════════════════════════╝

──────────────────────────────────────────
  MAIN MENU
──────────────────────────────────────────
  [1] Classify a single image
  [2] Classify multiple images from folder
  [3] View audit log
  [4] View log statistics
  [5] Generate waste report
  [6] Exit
──────────────────────────────────────────
```

---

## Testing Individual Modules

You can also test each module independently:

```bash
# Test data loading
python3 data_loader.py

# Test classification on a single image
python3 classifier.py dataset/plastic/plastic_1.jpg

# Test audit log
python3 audit_log.py

# Test report generation
python3 report_generator.py
```

---

## Sample Report Output

```
==================================================
       SMART WASTE MANAGEMENT SYSTEM
          WASTE ANALYSIS REPORT
==================================================
  Generated On  : 2026-03-24 10:30:15
  Total Scanned : 6 image(s)
  Avg Confidence: 84.59%

── CLASSIFICATION SUMMARY ──────────────────────

  cardboard    :    1 item(s)  █  [♻  Recyclable]
  glass        :    0 item(s)    [♻  Recyclable]
  metal        :    1 item(s)  █  [♻  Recyclable]
  paper        :    1 item(s)  █  [♻  Recyclable]
  plastic      :    2 item(s)  ██  [♻  Recyclable]
  trash        :    1 item(s)  █  [🗑  Non-Recyclable]

── RECYCLABILITY BREAKDOWN ─────────────────────

  ♻  Recyclable     :    5 item(s)  (83.3%)
  🗑  Non-Recyclable :    1 item(s)  (16.7%)
  ⚠  Uncertain      :    0 item(s)  (0.0%)

==================================================
```

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.9 | Core programming language |
| TensorFlow / Keras | Building and training the model |
| MobileNetV2 | Pre-trained base model (Transfer Learning) |
| Pillow (PIL) | Image loading and preprocessing |
| NumPy | Array and numerical operations |
| Scikit-learn | Train/validation split |
| Matplotlib | Training graphs and plots |
| Conda | Environment management |

---

## Module Summary

| Module | Responsibility |
|--------|---------------|
| `config.py` | Central settings file — all paths, constants, parameters |
| `data_loader.py` | Load images, normalize, split into train/val |
| `model_trainer.py` | Transfer learning with MobileNetV2, two phase training |
| `classifier.py` | Load model, preprocess image, predict category |
| `audit_log.py` | Log classifications, read log, get statistics |
| `report_generator.py` | Generate and save analysis reports |
| `main.py` | Menu-driven interface connecting all modules |

---

## How Modules Connect

```
Training Phase (run once):
    model_trainer.py → data_loader.py → config.py
                     → MobileNetV2 (downloaded automatically)
                     → saves waste_classifier.h5

Running Phase (every time):
    main.py → classifier.py    → classifies images
            → audit_log.py     → logs results
            → report_generator.py → generates reports
            (all import from config.py)
```

---

## Sample Dataset Note

If the real TrashNet dataset cannot be downloaded automatically, the system
creates a **sample dataset with colored images** (100 images per category,
600 total) for demonstration purposes.

Each category gets a distinct color:


|  Category | Color Used |
|-----------|------------|
| Cardboard |    Brown   |
| Glass     | Light Blue |
| Metal     |    Gray    |
| Paper     |    White   |
| Plastic   |   Orange   |
| Trash     |  Dark Gray |


This allows the **full pipeline to be tested and demonstrated** without
needing the real dataset. The model trains on these colored images and
learns to distinguish between categories based on color.

To use real waste images:
1. Take photos of actual waste items
2. Place them in the correct category folder inside `dataset/`
3. Retrain the model with `python3 model_trainer.py`

---

## Sample Dataset Note

If the real TrashNet dataset cannot be downloaded automatically, the system
creates a **sample dataset with colored images** for demonstration purposes:

|  Category | Color Used |
|-----------|------------|
| Cardboard |    Brown   |
| Glass     | Light Blue |
| Metal     |    Gray    |
| Paper     |    White   |
| Plastic   |   Orange   |
| Trash     |  Dark Gray |

- 100 images per category = 600 total images
- Each image is a 224×224 solid colored square with slight random variation
- This allows the **full pipeline to be tested** without needing real images
- The model trained on this sample dataset achieved **~75% accuracy**

> For real waste classification, replace the sample images with actual
> waste photos in the `dataset/` folders and retrain the model using
> `python3 model_trainer.py`

---

## Notes

- Always activate conda environment before running: `conda activate waste`
- Run `model_trainer.py` before `main.py`
- Model is automatically saved to `models/waste_classifier.h5`
- All folders are auto-created on first run
- Confidence threshold is 60% — predictions below this are marked as "uncertain"
- For best results use real waste images in the dataset folders

---

*Built as a Fundamentals of AI & ML project*
