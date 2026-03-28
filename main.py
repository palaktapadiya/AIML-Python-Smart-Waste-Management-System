# main.py
# Entry point for the Smart Waste Management System
# Ties all modules together into one clean pipeline

import os
import sys
from classifier import load_model, classify, classify_multiple
from audit_log import log_multiple, print_log, get_log_stats
from report_generator import generate_and_save


def print_banner():
    """Prints the welcome banner."""
    print("""
╔══════════════════════════════════════════════════╗
║       SMART WASTE MANAGEMENT SYSTEM              ║
║       Powered by Deep Learning (CNN)             ║
╚══════════════════════════════════════════════════╝
    """)


def print_menu():
    """Prints the main menu."""
    print("""
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
    """)


def get_image_files(folder_path):
    """
    Returns a list of all image files in a folder.

    Args:
        folder_path (str): Path to the folder

    Returns:
        image_files (list): List of full image paths
    """
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            image_files.append(os.path.join(folder_path, filename))

    return image_files


def option_single(model):
    """
    Option 1 — Classify a single image.

    Args:
        model: Loaded Keras model
    """
    print("\n── CLASSIFY SINGLE IMAGE ──────────────────────")
    image_path = input("  Enter image path: ").strip()

    if not os.path.exists(image_path):
        print(f"  [ERROR] File not found: {image_path}")
        return

    result = classify(image_path, model)

    print("\n  ── RESULT ──────────────────────────────────")
    print(f"  Image      : {os.path.basename(result['image'])}")
    print(f"  Category   : {result['category'].upper()}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  Recyclable : {result['recyclable']}")
    print("  ────────────────────────────────────────────")

    # Log the result
    log_multiple([result])


def option_multiple(model):
    """
    Option 2 — Classify all images in a folder.

    Args:
        model: Loaded Keras model
    """
    print("\n── CLASSIFY FOLDER ────────────────────────────")
    folder_path = input("  Enter folder path: ").strip()

    if not os.path.exists(folder_path):
        print(f"  [ERROR] Folder not found: {folder_path}")
        return

    image_files = get_image_files(folder_path)

    if not image_files:
        print("  [ERROR] No image files found in this folder.")
        return

    print(f"  Found {len(image_files)} image(s). Starting classification...\n")

    results = classify_multiple(image_files, model)

    # Log all results
    log_multiple(results)

    # Ask if user wants to generate report
    choice = input("  Generate report for these results? (yes/no): ").strip().lower()
    if choice == "yes":
        generate_and_save(results)


def option_view_log():
    """Option 3 — View the full audit log."""
    print_log()


def option_log_stats():
    """Option 4 — View log statistics."""
    stats = get_log_stats()

    if not stats:
        print("\n  [INFO] No classifications logged yet.")
        return

    print("\n── LOG STATISTICS ─────────────────────────────")
    print(f"  Total Classifications : {stats['total']}")
    print(f"  Recyclable            : {stats['recyclable']}")
    print(f"  Non-Recyclable        : {stats['non_recyclable']}")
    print(f"  Uncertain             : {stats['uncertain']}")
    print("\n  Per Category:")
    for category, count in stats["categories"].items():
        print(f"    {category:<12} : {count} item(s)")
    print("───────────────────────────────────────────────\n")


def option_generate_report(model):
    """
    Option 5 — Classify a folder and generate a full report.

    Args:
        model: Loaded Keras model
    """
    print("\n── GENERATE REPORT ────────────────────────────")
    folder_path = input("  Enter folder path to classify and report: ").strip()

    if not os.path.exists(folder_path):
        print(f"  [ERROR] Folder not found: {folder_path}")
        return

    image_files = get_image_files(folder_path)

    if not image_files:
        print("  [ERROR] No image files found in this folder.")
        return

    results = classify_multiple(image_files, model)
    log_multiple(results)
    generate_and_save(results)


def main():
    """
    Main function — entry point of the program.
    Loads the model once and runs the menu loop.
    """
    print_banner()

    # Load model once at startup
    print("[SYSTEM] Loading trained model...")
    try:
        model = load_model()
        print("[SYSTEM] Model ready!\n")
    except FileNotFoundError as e:
        print(e)
        print("\n[SYSTEM] Please run model_trainer.py first.")
        print("  Command: python model_trainer.py")
        sys.exit(1)

    # Main menu loop
    while True:
        print_menu()
        choice = input("  Enter your choice (1-6): ").strip()

        if choice == "1":
            option_single(model)

        elif choice == "2":
            option_multiple(model)

        elif choice == "3":
            option_view_log()

        elif choice == "4":
            option_log_stats()

        elif choice == "5":
            option_generate_report(model)

        elif choice == "6":
            print("\n[SYSTEM] Goodbye! Thank you for using Smart Waste Management System 👋\n")
            sys.exit(0)

        else:
            print("\n  [ERROR] Invalid choice. Please enter a number between 1 and 6.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
