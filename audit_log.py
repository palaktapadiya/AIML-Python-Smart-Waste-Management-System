# audit_log.py
# Logs every classification with timestamp, category, confidence and recyclable status
# Keeps a complete history of all waste classifications done by the system

import os
from datetime import datetime
from config import LOG_FILE, LOG_FORMAT


def log_classification(image, category, confidence, recyclable):
    """
    Logs a single classification result to the audit log file.

    Args:
        image      (str)  : Image filename or path
        category   (str)  : Predicted waste category
        confidence (float): Confidence score (0-100)
        recyclable (str)  : "Recyclable" or "Non-Recyclable" or "unknown"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = LOG_FORMAT.format(
        timestamp  = timestamp,
        image      = os.path.basename(image),
        category   = category,
        confidence = confidence,
        recyclable = recyclable
    )

    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_entry + "\n")
    except Exception as e:
        print(f"[AUDIT LOG] Warning: Could not write to log file.\nError: {e}")


def log_multiple(results):
    """
    Logs a list of classification results at once.

    Args:
        results (list): List of result dictionaries from classifier.py
    """
    print(f"\n[AUDIT LOG] Logging {len(results)} result(s)...")

    for result in results:
        log_classification(
            image      = result["image"],
            category   = result["category"],
            confidence = result["confidence"],
            recyclable = result["recyclable"]
        )

    print(f"[AUDIT LOG] All results logged to: {LOG_FILE}")


def read_log():
    """
    Reads and returns all entries from the audit log file.

    Returns:
        entries (list): List of log entry strings
    """
    if not os.path.exists(LOG_FILE):
        print("[AUDIT LOG] No log file found. Nothing has been classified yet.")
        return []

    with open(LOG_FILE, "r") as f:
        entries = f.readlines()

    return [entry.strip() for entry in entries if entry.strip()]


def print_log():
    """
    Prints all log entries to the terminal in a readable format.
    """
    entries = read_log()

    if not entries:
        return

    print("\n========== AUDIT LOG ==========")
    print(f"  {'#':<5} {'Timestamp':<22} {'Image':<20} {'Category':<12} {'Confidence':>10} {'Recyclable'}")
    print(f"  {'-'*90}")

    for idx, entry in enumerate(entries, 1):
        parts = entry.split(" | ")
        if len(parts) == 5:
            timestamp, image, category, confidence, recyclable = parts
            print(f"  {idx:<5} {timestamp:<22} {image:<20} {category:<12} {confidence:>10} {recyclable}")
        else:
            print(f"  {idx:<5} {entry}")

    print(f"  {'-'*90}")
    print(f"  Total entries: {len(entries)}")
    print("================================\n")


def clear_log():
    """
    Clears all entries from the audit log file.
    Use carefully — this cannot be undone!
    """
    confirm = input("[AUDIT LOG] Are you sure you want to clear the log? (yes/no): ")

    if confirm.lower() == "yes":
        with open(LOG_FILE, "w") as f:
            f.write("")
        print("[AUDIT LOG] Log cleared successfully.")
    else:
        print("[AUDIT LOG] Clear cancelled.")


def get_log_stats():
    """
    Returns basic statistics from the audit log.

    Returns:
        stats (dict): Total, per category counts, recyclable vs non-recyclable
    """
    entries = read_log()

    if not entries:
        return {}

    stats = {
        "total"          : 0,
        "categories"     : {},
        "recyclable"     : 0,
        "non_recyclable" : 0,
        "uncertain"      : 0
    }

    for entry in entries:
        parts = entry.split(" | ")
        if len(parts) == 5:
            _, _, category, _, recyclable = parts
            stats["total"] += 1

            # Count per category
            category = category.strip()
            stats["categories"][category] = stats["categories"].get(category, 0) + 1

            # Count recyclable status
            recyclable = recyclable.strip().lower()
            if recyclable == "recyclable":
                stats["recyclable"] += 1
            elif recyclable == "non-recyclable":
                stats["non_recyclable"] += 1
            else:
                stats["uncertain"] += 1

    return stats


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Log some dummy entries for testing
    print("[TEST] Logging dummy entries...")

    log_classification("bottle.jpg",   "plastic",   82.35, "Recyclable")
    log_classification("paper1.jpg",   "paper",     91.20, "Recyclable")
    log_classification("garbage.jpg",  "trash",     76.50, "Non-Recyclable")
    log_classification("can.jpg",      "metal",     88.10, "Recyclable")
    log_classification("unknown.jpg",  "uncertain", 45.00, "unknown")

    # Print the log
    print_log()

    # Print stats
    stats = get_log_stats()
    print("[TEST] Log Statistics:")
    print(f"  Total          : {stats['total']}")
    print(f"  Recyclable     : {stats['recyclable']}")
    print(f"  Non-Recyclable : {stats['non_recyclable']}")
    print(f"  Uncertain      : {stats['uncertain']}")
    print(f"  Per Category   : {stats['categories']}")
