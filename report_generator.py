# report_generator.py
# Generates a detailed waste analysis report based on classification results
# Saves the report as a text file in the reports folder

import os
from datetime import datetime
from audit_log import get_log_stats
from config import REPORT_FILE, WASTE_CATEGORIES


def generate_report(results):
    """
    Generates a waste analysis report from a list of classification results.

    Args:
        results (list): List of result dictionaries from classifier.py

    Returns:
        report_text (str): The full report as a string
    """

    # ── Basic Counts ──────────────────────────────
    total = len(results)
    recyclable_count    = sum(1 for r in results if r["recyclable"] == "Recyclable")
    non_recyclable_count = sum(1 for r in results if r["recyclable"] == "Non-Recyclable")
    uncertain_count     = sum(1 for r in results if r["category"] == "uncertain")

    # ── Per Category Counts ───────────────────────
    category_counts = {}
    for r in results:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # ── Average Confidence ────────────────────────
    confident_results = [r for r in results if r["category"] != "uncertain"]
    if confident_results:
        avg_confidence = sum(r["confidence"] for r in confident_results) / len(confident_results)
    else:
        avg_confidence = 0.0

    # ── Recyclable Percentage ─────────────────────
    if total > 0:
        recyclable_pct     = (recyclable_count / total) * 100
        non_recyclable_pct = (non_recyclable_count / total) * 100
        uncertain_pct      = (uncertain_count / total) * 100
    else:
        recyclable_pct = non_recyclable_pct = uncertain_pct = 0.0

    # ── Build Report Text ─────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 50

    lines = [
        separator,
        "       SMART WASTE MANAGEMENT SYSTEM",
        "          WASTE ANALYSIS REPORT",
        separator,
        f"  Generated On  : {now}",
        f"  Total Scanned : {total} image(s)",
        f"  Avg Confidence: {avg_confidence:.2f}%",
        "",
        "── CLASSIFICATION SUMMARY ──────────────────────",
        ""
    ]

    # Per category breakdown
    for category in WASTE_CATEGORIES:
        count = category_counts.get(category, 0)
        recyclable_label = "♻  Recyclable" if category != "trash" else "🗑  Non-Recyclable"
        bar = "█" * count
        lines.append(f"  {category:<12} : {count:>4} item(s)  {bar}  [{recyclable_label}]")

    if uncertain_count > 0:
        lines.append(f"  {'uncertain':<12} : {uncertain_count:>4} item(s)  {'█' * uncertain_count}  [⚠  Unknown]")

    lines += [
        "",
        "── RECYCLABILITY BREAKDOWN ─────────────────────",
        "",
        f"  ♻  Recyclable     : {recyclable_count:>4} item(s)  ({recyclable_pct:.1f}%)",
        f"  🗑  Non-Recyclable : {non_recyclable_count:>4} item(s)  ({non_recyclable_pct:.1f}%)",
        f"  ⚠  Uncertain      : {uncertain_count:>4} item(s)  ({uncertain_pct:.1f}%)",
        "",
        "── INDIVIDUAL RESULTS ──────────────────────────",
        ""
    ]

    # Individual image results
    for idx, r in enumerate(results, 1):
        lines.append(f"  {idx}. {os.path.basename(r['image'])}")
        lines.append(f"     Category   : {r['category']}")
        lines.append(f"     Confidence : {r['confidence']}%")
        lines.append(f"     Recyclable : {r['recyclable']}")
        lines.append("")

    lines += [
        separator,
        "  Thank you for using Smart Waste Management System!",
        separator,
        ""
    ]

    report_text = "\n".join(lines)
    return report_text


def save_report(report_text):
    """
    Saves the report text to the report file.

    Args:
        report_text (str): Report content to save
    """
    try:
        with open(REPORT_FILE, "w") as f:
            f.write(report_text)
        print(f"\n[REPORT] Report saved to: {REPORT_FILE}")
    except Exception as e:
        print(f"[REPORT] Warning: Could not save report.\nError: {e}")


def print_report(report_text):
    """
    Prints the report to the terminal.

    Args:
        report_text (str): Report content to print
    """
    print(report_text)


def generate_and_save(results):
    """
    Full pipeline — generate report, print it, and save it.
    Call this from main.py.

    Args:
        results (list): List of result dictionaries from classifier.py
    """
    if not results:
        print("[REPORT] No results to generate report from.")
        return

    report_text = generate_report(results)
    print_report(report_text)
    save_report(report_text)


# ─────────────────────────────────────────────
# Quick test — run this file directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Dummy results for testing
    dummy_results = [
        {"image": "bottle.jpg",  "category": "plastic", "confidence": 82.35, "recyclable": "Recyclable"},
        {"image": "paper1.jpg",  "category": "paper",   "confidence": 91.20, "recyclable": "Recyclable"},
        {"image": "garbage.jpg", "category": "trash",   "confidence": 76.50, "recyclable": "Non-Recyclable"},
        {"image": "can.jpg",     "category": "metal",   "confidence": 88.10, "recyclable": "Recyclable"},
        {"image": "box.jpg",     "category": "cardboard","confidence": 79.40, "recyclable": "Recyclable"},
        {"image": "unknown.jpg", "category": "uncertain","confidence": 45.00, "recyclable": "unknown"},
    ]

    generate_and_save(dummy_results)
