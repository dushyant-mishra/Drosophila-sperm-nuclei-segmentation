#!/usr/bin/env python3
"""
Outlier audit script for spermatid 3D tracking results.

What it does
------------
Reads:
- track_summary.csv
- measurements_with_tracks.csv

Produces:
- outliers_long.csv
- outliers_tortuous.csv
- outliers_thick.csv
- outliers_taper.csv
- outliers_single_slice.csv
- outliers_top50_long.csv
- outliers_top50_tortuous.csv
- outliers_top50_thick.csv
- outliers_top50_taper.csv
- audit_sheet.csv
- outlier_summary.txt

It also adds:
- overlap categories between outlier classes
- per-track slice span info from measurements_with_tracks.csv

How to run
----------
python audit_sperm_outliers.py --dir "C:\\path\\to\\batch_output_folder"

If omitted, --dir defaults to the current folder.

Optional threshold overrides
----------------------------
python audit_sperm_outliers.py --dir "..." --length-thresh 20 --tort-thresh 2.0 --thick-thresh 2.2 --taper-thresh 2.5
"""

import os
import argparse
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of these columns found: {candidates}")
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Folder containing track_summary.csv and measurements_with_tracks.csv")
    parser.add_argument("--length-thresh", type=float, default=15.0, help="Flag tracks longer than this (µm)")
    parser.add_argument("--tort-thresh", type=float, default=1.5, help="Flag tracks more tortuous than this")
    parser.add_argument("--thick-thresh", type=float, default=2.0, help="Flag tracks thicker than this (µm)")
    parser.add_argument("--taper-thresh", type=float, default=1.5, help="Flag tracks with taper ratio above this")
    parser.add_argument("--single-slice-max", type=float, default=1.0, help="Flag tracks with n_slices <= this")
    parser.add_argument("--topn", type=int, default=50, help="Top N examples per class")
    args = parser.parse_args()

    base_dir = args.dir
    if base_dir == ".":
        # Interactive folder selection
        root = tk.Tk()
        root.withdraw()
        print("Please select the folder containing 'track_summary.csv'...")
        base_dir = filedialog.askdirectory(title="Select Sperm Analysis Results Folder")
        root.destroy()
        if not base_dir:
            print("No folder selected. Exiting.")
            return

    import glob
    
    # Use glob to find versioned outputs
    track_files = glob.glob(os.path.join(base_dir, "track_summary*.csv"))
    meas_files = glob.glob(os.path.join(base_dir, "*measurements_with_tracks*.csv"))
    
    # Filter out single measurement files just in case
    meas_files = [f for f in meas_files if "single" not in os.path.basename(f).lower()]
    
    if not track_files:
        raise FileNotFoundError(f"Missing track_summary CSV in {base_dir}")
    if not meas_files:
        raise FileNotFoundError(f"Missing measurements_with_tracks CSV in {base_dir}. Ensure DO_TRACKING is Enabled.")
        
    track_path = track_files[-1]  # Safest to pick the last if multiple versions exist
    meas_path = meas_files[-1]
    
    print(f"Auditing tracking file: {os.path.basename(track_path)}")
    print(f"Auditing measurements file: {os.path.basename(meas_path)}")

    tracks = pd.read_csv(track_path)
    meas = pd.read_csv(meas_path)

    # Robust column matching
    track_id_col = pick_col(tracks, ["track_id", "Track_ID"])
    length_col = pick_col(tracks, ["total_3d_length_um", "length_3d_um_est", "max_length_um"])
    tort_col = pick_col(tracks, ["tortuosity_3d", "tortuosity"])
    thick_col = pick_col(tracks, ["thickness_um", "effective_thickness_um", "median_width_um"])
    taper_col = pick_col(tracks, ["taper_ratio", "morphological_taper_ratio"])
    nslices_col = pick_col(tracks, ["n_slices", "n_detections"])

    meas_track_id_col = pick_col(meas, ["track_id", "Track_ID"])
    meas_z_col = pick_col(meas, ["z_slice", "z"])
    meas_x_col = pick_col(meas, ["centroid_x", "x"], required=False)
    meas_y_col = pick_col(meas, ["centroid_y", "y"], required=False)

    # Track-level slice span info from measurements_with_tracks.csv
    span = (
        meas.groupby(meas_track_id_col)[meas_z_col]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={
            meas_track_id_col: track_id_col,
            "min": "z_min_from_meas",
            "max": "z_max_from_meas",
            "count": "n_rows_in_measurements"
        })
    )

    tracks = tracks.merge(span, on=track_id_col, how="left")

    # Outlier flags
    tracks["flag_long"] = tracks[length_col] > args.length_thresh
    tracks["flag_tortuous"] = tracks[tort_col] > args.tort_thresh
    tracks["flag_thick"] = tracks[thick_col] > args.thick_thresh
    tracks["flag_taper"] = tracks[taper_col] > args.taper_thresh
    tracks["flag_single_slice"] = tracks[nslices_col] <= args.single_slice_max

    # Count number of flags per track
    flag_cols = ["flag_long", "flag_tortuous", "flag_thick", "flag_taper", "flag_single_slice"]
    tracks["n_flags"] = tracks[flag_cols].sum(axis=1)

    # Output folder
    out_dir = os.path.join(base_dir, "outlier_audit")
    ensure_dir(out_dir)

    # Filtered tables
    long_outliers = tracks[tracks["flag_long"]].copy().sort_values(length_col, ascending=False)
    tort_outliers = tracks[tracks["flag_tortuous"]].copy().sort_values(tort_col, ascending=False)
    thick_outliers = tracks[tracks["flag_thick"]].copy().sort_values(thick_col, ascending=False)
    taper_outliers = tracks[tracks["flag_taper"]].copy().sort_values(taper_col, ascending=False)
    single_outliers = tracks[tracks["flag_single_slice"]].copy().sort_values(nslices_col, ascending=True)

    long_outliers.to_csv(os.path.join(out_dir, "outliers_long.csv"), index=False)
    tort_outliers.to_csv(os.path.join(out_dir, "outliers_tortuous.csv"), index=False)
    thick_outliers.to_csv(os.path.join(out_dir, "outliers_thick.csv"), index=False)
    taper_outliers.to_csv(os.path.join(out_dir, "outliers_taper.csv"), index=False)
    single_outliers.to_csv(os.path.join(out_dir, "outliers_single_slice.csv"), index=False)

    # Top-N examples
    tracks.sort_values(length_col, ascending=False).head(args.topn).to_csv(
        os.path.join(out_dir, "outliers_top50_long.csv"), index=False
    )
    tracks.sort_values(tort_col, ascending=False).head(args.topn).to_csv(
        os.path.join(out_dir, "outliers_top50_tortuous.csv"), index=False
    )
    tracks.sort_values(thick_col, ascending=False).head(args.topn).to_csv(
        os.path.join(out_dir, "outliers_top50_thick.csv"), index=False
    )
    tracks.sort_values(taper_col, ascending=False).head(args.topn).to_csv(
        os.path.join(out_dir, "outliers_top50_taper.csv"), index=False
    )

    # Combined flagged tracks
    flagged_any = tracks[tracks["n_flags"] > 0].copy().sort_values(
        ["n_flags", length_col, tort_col, thick_col, taper_col], ascending=[False, False, False, False, False]
    )
    flagged_any.to_csv(os.path.join(out_dir, "outliers_all_flagged.csv"), index=False)

    # Simple audit template
    audit_cols = [
        track_id_col,
        "reason_flagged",
        "keep_or_reject",
        "error_type",
        "notes"
    ]
    audit_sheet = flagged_any[[track_id_col]].copy()
    audit_sheet["reason_flagged"] = ""
    audit_sheet["keep_or_reject"] = ""
    audit_sheet["error_type"] = ""
    audit_sheet["notes"] = ""
    audit_sheet.to_csv(os.path.join(out_dir, "audit_sheet.csv"), index=False)

    # Build reason strings
    def reason_string(row):
        reasons = []
        if row["flag_long"]:
            reasons.append("long")
        if row["flag_tortuous"]:
            reasons.append("tortuous")
        if row["flag_thick"]:
            reasons.append("thick")
        if row["flag_taper"]:
            reasons.append("taper")
        if row["flag_single_slice"]:
            reasons.append("single_slice")
        return ",".join(reasons)

    flagged_any["reason_flagged"] = flagged_any.apply(reason_string, axis=1)
    flagged_any[[track_id_col, "reason_flagged", length_col, tort_col, thick_col, taper_col, nslices_col,
                 "z_min_from_meas", "z_max_from_meas", "n_rows_in_measurements", "n_flags"]].to_csv(
        os.path.join(out_dir, "outliers_flagged_compact.csv"), index=False
    )

    # Summary stats
    summary_lines = []
    summary_lines.append("OUTLIER AUDIT SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Input folder: {base_dir}")
    summary_lines.append(f"Tracks total: {len(tracks)}")
    summary_lines.append(f"Measurements total: {len(meas)}")
    summary_lines.append("")
    summary_lines.append("Thresholds:")
    summary_lines.append(f"  length > {args.length_thresh}")
    summary_lines.append(f"  tortuosity > {args.tort_thresh}")
    summary_lines.append(f"  thickness > {args.thick_thresh}")
    summary_lines.append(f"  taper > {args.taper_thresh}")
    summary_lines.append(f"  n_slices <= {args.single_slice_max}")
    summary_lines.append("")
    summary_lines.append("Counts:")
    summary_lines.append(f"  Long outliers:         {len(long_outliers)}")
    summary_lines.append(f"  Tortuous outliers:     {len(tort_outliers)}")
    summary_lines.append(f"  Thick outliers:        {len(thick_outliers)}")
    summary_lines.append(f"  Taper outliers:        {len(taper_outliers)}")
    summary_lines.append(f"  Single-slice outliers: {len(single_outliers)}")
    summary_lines.append(f"  Any flagged:           {len(flagged_any)}")
    summary_lines.append("")

    # Overlap table
    overlap = tracks.groupby(flag_cols).size().reset_index(name="count").sort_values("count", ascending=False)
    overlap.to_csv(os.path.join(out_dir, "outlier_flag_overlap.csv"), index=False)
    summary_lines.append("Top overlap patterns:")
    for _, row in overlap.head(10).iterrows():
        labels = []
        for c in flag_cols:
            if row[c]:
                labels.append(c.replace("flag_", ""))
        label = "+".join(labels) if labels else "none"
        summary_lines.append(f"  {label:35s} {int(row['count'])}")
    summary_lines.append("")

    # Basic distribution reference points
    for name, col in [
        ("length", length_col),
        ("tortuosity", tort_col),
        ("thickness", thick_col),
        ("taper", taper_col),
    ]:
        vals = pd.to_numeric(tracks[col], errors="coerce").dropna()
        if len(vals):
            summary_lines.append(
                f"{name:12s} mean={vals.mean():.3f} median={vals.median():.3f} p95={vals.quantile(0.95):.3f} max={vals.max():.3f}"
            )

    summary_path = os.path.join(out_dir, "outlier_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"Done. Output written to:\n{out_dir}")
    print("\n".join(summary_lines[:20]))


if __name__ == "__main__":
    main()
