#!/usr/bin/env python3
"""
Compare two spermatid outlier audits side by side.

Inputs
------
Two audit output folders produced by audit_sperm_outliers.py, each containing:
- outlier_summary.txt
- outliers_long.csv
- outliers_tortuous.csv
- outliers_thick.csv
- outliers_taper.csv
- outliers_single_slice.csv
- outliers_all_flagged.csv

Outputs
-------
Creates a comparison folder with:
- audit_comparison_summary.txt
- audit_comparison_metrics.csv
- audit_delta_table.csv

Usage
-----
python compare_sperm_outlier_audits.py --old "C:\\path\\to\\old\\outlier_audit" --new "C:\\path\\to\\new\\outlier_audit"

Optional:
python compare_sperm_outlier_audits.py --old "..." --new "..." --label-old "old_2d_tuned" --label-new "new_3d_tuned"
"""

import os
import argparse
import pandas as pd
import tkinter as tk
from tkinter import filedialog


def count_rows(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception:
        return None


def read_flagged_count(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception:
        return None


def safe_pct(num, den):
    if den in (None, 0):
        return None
    return 100.0 * num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", help="Path to old outlier_audit folder")
    parser.add_argument("--new", help="Path to new outlier_audit folder")
    parser.add_argument("--label-old", help="Label for old run")
    parser.add_argument("--label-new", help="Label for new run")
    args = parser.parse_args()

    old_dir = args.old
    new_dir = args.new

    if not old_dir or not new_dir:
        root = tk.Tk()
        root.withdraw()
        
        if not old_dir:
            print("Please select the OLD/BASELINE outlier_audit folder...")
            old_dir = filedialog.askdirectory(title="Select OLD/BASELINE Folder (containing 'outlier_audit')")
        
        if not new_dir:
            print("Please select the NEW/OPTIMIZED outlier_audit folder...")
            new_dir = filedialog.askdirectory(title="Select NEW/OPTIMIZED Folder (containing 'outlier_audit')")
            
        root.destroy()
        if not old_dir or not new_dir:
            print("Selection cancelled. Exiting.")
            return

    # Auto-adjust if users picked the parent folder instead of outlier_audit
    for i, d in enumerate([old_dir, new_dir]):
        audit_sub = os.path.join(d, "outlier_audit")
        if os.path.exists(audit_sub) and os.path.isdir(audit_sub):
            if i == 0: old_dir = audit_sub
            else: new_dir = audit_sub

    label_old = args.label_old or os.path.basename(os.path.dirname(old_dir) if "outlier_audit" in old_dir else old_dir).rstrip("/\\")
    label_new = args.label_new or os.path.basename(os.path.dirname(new_dir) if "outlier_audit" in new_dir else new_dir).rstrip("/\\")

    required_files = [
        "outliers_long.csv",
        "outliers_tortuous.csv",
        "outliers_thick.csv",
        "outliers_taper.csv",
        "outliers_single_slice.csv",
        "outliers_all_flagged.csv",
    ]

    for d in [old_dir, new_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Not a directory: {d}")
        for f in required_files:
            p = os.path.join(d, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required file: {p}")

    metrics = []

    old_flagged = read_flagged_count(os.path.join(old_dir, "outliers_all_flagged.csv"))
    new_flagged = read_flagged_count(os.path.join(new_dir, "outliers_all_flagged.csv"))

    items = [
        ("long_outliers", "outliers_long.csv"),
        ("tortuous_outliers", "outliers_tortuous.csv"),
        ("thick_outliers", "outliers_thick.csv"),
        ("taper_outliers", "outliers_taper.csv"),
        ("single_slice_outliers", "outliers_single_slice.csv"),
        ("all_flagged", "outliers_all_flagged.csv"),
    ]

    for metric_name, fname in items:
        old_count = count_rows(os.path.join(old_dir, fname))
        new_count = count_rows(os.path.join(new_dir, fname))

        delta = None if old_count is None or new_count is None else new_count - old_count
        pct_change = None
        if old_count not in (None, 0) and new_count is not None:
            pct_change = 100.0 * (new_count - old_count) / old_count

        winner = "tie"
        if old_count is not None and new_count is not None:
            if new_count < old_count:
                winner = label_new
            elif old_count < new_count:
                winner = label_old

        metrics.append({
            "metric": metric_name,
            f"{label_old}_count": old_count,
            f"{label_new}_count": new_count,
            "delta_new_minus_old": delta,
            "pct_change_new_vs_old": round(pct_change, 3) if pct_change is not None else None,
            "cleaner_run": winner,
        })

    df = pd.DataFrame(metrics)

    # Choose output folder next to new_dir
    out_dir = os.path.join(os.path.dirname(new_dir), "audit_comparison")
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, "audit_comparison_metrics.csv"), index=False)

    # A compact delta table
    delta_df = df[["metric", f"{label_old}_count", f"{label_new}_count", "delta_new_minus_old", "pct_change_new_vs_old", "cleaner_run"]].copy()
    delta_df.to_csv(os.path.join(out_dir, "audit_delta_table.csv"), index=False)

    # Human-readable summary
    lines = []
    lines.append("SPERM OUTLIER AUDIT COMPARISON")
    lines.append("=" * 70)
    lines.append(f"Old audit folder: {old_dir}")
    lines.append(f"New audit folder: {new_dir}")
    lines.append(f"Labels: {label_old} vs {label_new}")
    lines.append("")

    wins_old = 0
    wins_new = 0
    ties = 0

    for _, row in df.iterrows():
        metric = row["metric"]
        old_count = row[f"{label_old}_count"]
        new_count = row[f"{label_new}_count"]
        delta = row["delta_new_minus_old"]
        pct = row["pct_change_new_vs_old"]
        winner = row["cleaner_run"]

        if winner == label_old:
            wins_old += 1
        elif winner == label_new:
            wins_new += 1
        else:
            ties += 1

        lines.append(
            f"{metric:24s} "
            f"{label_old}={old_count:>6}   "
            f"{label_new}={new_count:>6}   "
            f"delta={delta:>6}   "
            f"pct={pct if pct is not None else 'NA'}   "
            f"cleaner={winner}"
        )

    lines.append("")
    lines.append("OVERALL")
    lines.append("-" * 70)
    lines.append(f"{label_old} wins: {wins_old}")
    lines.append(f"{label_new} wins: {wins_new}")
    lines.append(f"Ties: {ties}")

    # crude verdict
    if wins_new > wins_old:
        verdict = f"{label_new} appears cleaner overall based on the selected outlier classes."
    elif wins_old > wins_new:
        verdict = f"{label_old} appears cleaner overall based on the selected outlier classes."
    else:
        verdict = "The runs are mixed or tied across the selected outlier classes."

    lines.append("")
    lines.append(f"Verdict: {verdict}")

    summary_path = os.path.join(out_dir, "audit_comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Done. Comparison written to:\n{out_dir}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
