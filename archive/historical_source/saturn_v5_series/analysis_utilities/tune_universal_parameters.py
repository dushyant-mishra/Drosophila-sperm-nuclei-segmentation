#!/usr/bin/env python3
"""
Evolutionary Parameter Tuner for Saturn V3 Overlap-First Tracking

Uses SciPy's Differential Evolution algorithm (an evolutionary strategy)
to find optimal tracking parameters. Seeds the initial population with
your current CONFIG values so it starts from a known-good baseline
instead of random exploration.

How it works
------------
1. Loads your images and draws/reuses a global ROI
2. Defines biologically-meaningful bounds for each tracking parameter
3. Seeds the population with the current CONFIG as the starting point
4. Runs Differential Evolution: each "individual" in the population is a
   parameter set that gets evaluated by running the full segmentation +
   tracking pipeline in-process (no subprocess overhead)
5. Scores each run using a composite biology-aware objective function
6. Evolves the population over N generations to find the optimal params
7. Saves the best parameters as an incremental JSON file

Usage
-----
GUI mode (recommended):
    python tune_universal_parameters.py

CLI mode:
    python tune_universal_parameters.py --dir "path/to/images" --slices 0-12

With more generations (slower but more thorough):
    python tune_universal_parameters.py --dir "path/to/images" --maxiter 15 --popsize 10

Output
------
- best_params_NNN.json (incremental, also copied to project root)
- tuning_results_saturnv3.json (full search history)
"""

import os
import sys
import re
import json
import time
import glob
import math
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tifffile

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from scipy.optimize import differential_evolution

import tkinter as tk
from tkinter import filedialog, simpledialog

# Import the V3 pipeline functions directly for fast in-process execution
# Add paths to root and archive
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "archive"))

try:
    from sperm_segmentation_saturnv3 import (
        CONFIG,
        segment_slice,
        measure_spermatids,
        track_across_slices,
        rows_from_results,
        normalize_display,
        robust_imread,
    )
except Exception as e:
    print(f"Error: Could not import from sperm_segmentation_saturnv3.py (in /archive): {e}")
    raise

ROI_SAVE_PATH = os.path.join(parent_dir, "last_drawn_roi_saturnv3_tune.tif")

# ── Global State ──────────────────────────────────────────────────────────────
eval_count = 0
best_global_score = -1e18
results_list = []
images_to_eval = []
z_values_eval = []
roi_mask_global = None

# ── Parameter Space ───────────────────────────────────────────────────────────
# Each entry: (CONFIG_KEY, lower_bound, upper_bound, is_integer)
# Bounds are biologically motivated ranges centered around the current CONFIG.
PARAM_SPACE = [
    ("OVERLAP_STABILITY_THRESHOLD",        0.03,  0.15,  False),
    ("OVERLAP_ORIENTATION_DEG",            5.0,   30.0,  False),
    ("OVERLAP_MULTIPLIER",                 1.10,  1.60,  False),
    ("TRACK_MAX_DIST_UM",                  4.0,   10.0,  False),
    ("TRACK_BBOX_PADDING_PX",             1,     6,     True),
    ("CONSERVATIVE_MAX_WIDTH_JUMP_RATIO",  0.20,  0.70,  False),
    ("CONSERVATIVE_MAX_LENGTH_JUMP_RATIO", 0.30,  0.80,  False),
    ("CONSERVATIVE_MAX_AREA_JUMP_RATIO",   0.40,  0.90,  False),
    ("CONSERVATIVE_MAX_CENTROID_JUMP_UM",  5.0,   15.0,  False),
]

# ── Audit Thresholds (matching audit_sperm_outliers.py defaults) ──────────────
AUDIT = {
    "length_thresh": 15.0,
    "tort_thresh": 1.5,
    "thick_thresh": 2.0,
    "taper_thresh": 1.5,
}


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def parse_slices_arg(text):
    text = text.strip()
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            if b < a:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def warn_if_nonconsecutive(zs):
    if len(zs) < 2:
        return
    diffs = np.diff(sorted(zs))
    if np.any(diffs > 1):
        print("\nWARNING:")
        print("  The selected slices are not consecutive.")
        print("  3D tuning works best on a consecutive block from one tubule.")
        print(f"  Current slice gaps: {diffs.tolist()}\n")


def build_roi(images, force_redraw=False):
    """Interactive polygon ROI drawing with right-click undo."""
    roi_img = images[0]
    roi_mask = None

    if force_redraw and os.path.exists(ROI_SAVE_PATH):
        try:
            os.remove(ROI_SAVE_PATH)
        except Exception:
            pass

    if os.path.exists(ROI_SAVE_PATH):
        try:
            print(f"\nLoaded previously drawn ROI from {ROI_SAVE_PATH}.")
            roi_mask = robust_imread(ROI_SAVE_PATH).astype(bool)
            if roi_mask.shape != roi_img.shape:
                print("Saved ROI shape mismatch. Redrawing.")
                roi_mask = None
        except Exception:
            roi_mask = None

    if roi_mask is None:
        print("\nDraw a GLOBAL ROI (Left-Click to place, Right-Click to UNDO, ENTER to Finish).")
        pts = []

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(normalize_display(roi_img), cmap="gray")
        ax.set_title("ROI: Click to place, Right-click to UNDO, ENTER to Finish")
        
        line, = ax.plot([], [], 'r-o', lw=2, markersize=8)

        def redraw():
            if not pts:
                line.set_data([], [])
                ax.set_title("ROI: Click to place points")
            else:
                x, y = zip(*pts)
                line.set_data(x, y)
                ax.set_title(f"ROI Building: {len(pts)} points (Right-click to undo, ENTER to finish)")
            fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                pts.append((event.xdata, event.ydata))
                redraw()
            elif event.button == 3:
                if pts:
                    pts.pop()
                    redraw()

        def on_key(event):
            if event.key == "enter":
                if len(pts) > 2:
                    plt.close(fig)
                else:
                    print("Need at least 3 points before finalizing with ENTER.")

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        try:
            fig.canvas.manager.window.attributes("-topmost", 1)
            fig.canvas.manager.window.attributes("-topmost", 0)
            fig.canvas.manager.window.focus_force()
        except Exception:
            pass

        plt.show(block=True)

        if len(pts) < 3:
            print("Invalid ROI. Exiting.")
            sys.exit(1)

        H, W = roi_img.shape
        y, x = np.mgrid[:H, :W]
        points = np.column_stack((x.ravel(), y.ravel()))
        full_pts = pts + [pts[0]]
        path = MplPath(full_pts)
        roi_mask = path.contains_points(points).reshape(H, W)

        tifffile.imwrite(ROI_SAVE_PATH, roi_mask.astype(np.uint8) * 255)
        print(f"Saved ROI to {ROI_SAVE_PATH}")

    return roi_mask


def safe_median(series, default=np.nan):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size > 0 else default


def safe_fraction(mask_like, default=1.0):
    arr = np.asarray(mask_like)
    return float(np.mean(arr.astype(bool))) if arr.size > 0 else default


def pick_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


# ═════════════════════════════════════════════════════════════════════════════
#  SCORING FUNCTION (Biology-Aware Composite)
# ═════════════════════════════════════════════════════════════════════════════

def score_run(df_2d, df_tracks):
    """
    Score a parameter set by analysing the tracking output.
    Returns (score, metrics_dict). Higher score = better.
    
    Scoring priorities (weights):
    - Maximize multi-slice track count & depth           (reward)
    - Minimize single-slice orphans                      (heavy penalty)
    - Minimize long outliers (monster merges)             (heavy penalty)
    - Penalize tortuous, thick, taper outliers            (moderate penalty)
    - Reward biologically plausible median 3D length      (alignment bonus)
    """
    if df_2d.empty:
        return -1e12, {"reason": "no_2d"}
    if df_tracks is None or df_tracks.empty:
        return -1e12, {"reason": "no_3d"}

    n_2d = len(df_2d)
    n_tracks = len(df_tracks)

    # Column matching
    length_col = pick_col(df_tracks, ["total_3d_length_um", "length_3d_um_est", "max_length_um"])
    tort_col = pick_col(df_tracks, ["tortuosity_3d", "tortuosity"])
    thick_col = pick_col(df_tracks, ["thickness_um", "effective_thickness_um", "median_width_um"])
    taper_col = pick_col(df_tracks, ["taper_ratio", "morphological_taper_ratio"])
    nslices_col = pick_col(df_tracks, ["n_slices", "n_detections"])

    if length_col is None or nslices_col is None:
        return -1e12, {"reason": "missing_columns"}

    # Extract series
    lengths = df_tracks[length_col].astype(float)
    nslices = df_tracks[nslices_col].astype(float)
    l3d_med = safe_median(lengths)

    # Count outliers using the same thresholds as audit_sperm_outliers.py
    n_long = int((lengths > AUDIT["length_thresh"]).sum())
    n_single = int((nslices <= 1).sum())
    
    n_tort = 0
    if tort_col:
        n_tort = int((df_tracks[tort_col].astype(float) > AUDIT["tort_thresh"]).sum())
    
    n_thick = 0
    if thick_col:
        n_thick = int((df_tracks[thick_col].astype(float) > AUDIT["thick_thresh"]).sum())
    
    n_taper = 0
    if taper_col:
        n_taper = int((df_tracks[taper_col].astype(float) > AUDIT["taper_thresh"]).sum())

    # Track depth stats
    multi_slice = int((nslices > 1).sum())
    avg_depth = float(nslices[nslices > 1].mean()) if multi_slice > 0 else 0
    total_linked = int(nslices[nslices > 1].sum())

    single_frac = n_single / max(n_tracks, 1)
    multi_frac = multi_slice / max(n_tracks, 1)

    # ── COMPOSITE SCORE ──────────────────────────────────────────────────
    score = 0.0

    # 1. Reward: more multi-slice tracks and deeper tracking
    score += 1.5 * multi_slice          # Each multi-slice track is good
    score += 0.3 * total_linked         # More detections linked is good
    score += 50.0 * avg_depth           # Deeper tracks are better

    # 2. Heavy penalty: single-slice fragmentation (the #1 problem)
    score -= 2.0 * n_single

    # 3. Heavy penalty: long outliers (monster merges, the #2 problem)
    score -= 3.0 * n_long

    # 4. Moderate penalties: shape outliers
    score -= 1.0 * n_tort
    score -= 1.0 * n_thick
    score -= 1.0 * n_taper

    # 5. Structural penalties
    if single_frac > 0.45:
        score -= 300.0 * (single_frac - 0.45)
    if multi_frac < 0.50:
        score -= 200.0 * (0.50 - multi_frac)

    # 6. Median 3D length alignment (target ~10 um for Drosophila)
    if np.isfinite(l3d_med):
        score -= 30.0 * abs(l3d_med - 10.0)

    metrics = {
        "n_2d": n_2d,
        "n_tracks": n_tracks,
        "multi_slice": multi_slice,
        "total_linked": total_linked,
        "avg_depth": round(avg_depth, 2),
        "single_slice": n_single,
        "single_frac": round(single_frac, 4),
        "n_long": n_long,
        "n_tort": n_tort,
        "n_thick": n_thick,
        "n_taper": n_taper,
        "l3d_median_um": round(l3d_med, 3) if np.isfinite(l3d_med) else None,
        "score": round(score, 2),
    }
    return score, metrics


# ═════════════════════════════════════════════════════════════════════════════
#  OBJECTIVE FUNCTION (called by differential_evolution)
# ═════════════════════════════════════════════════════════════════════════════

def objective_fn(x, um_per_px):
    """
    Called once per 'individual' in the DE population.
    Maps the parameter vector x -> CONFIG overrides -> pipeline run -> score.
    Returns NEGATIVE score because DE minimizes.
    """
    global eval_count, best_global_score, results_list
    global images_to_eval, z_values_eval, roi_mask_global

    # Map vector to CONFIG overrides
    param_dict = {}
    for i, (key, lo, hi, is_int) in enumerate(PARAM_SPACE):
        val = x[i]
        if is_int:
            val = int(round(val))
        param_dict[key] = val

    # Build a fresh CONFIG with overrides
    cfg = CONFIG.copy()
    cfg.update(param_dict)

    # Suppress stdout from the pipeline (it's very chatty)
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    score = -1e12
    metrics = {}
    try:
        # Run segmentation on each slice
        rows = []
        for img, z_idx in zip(images_to_eval, z_values_eval):
            seg = segment_slice(img, cfg, z_idx=z_idx, debug_dir=None, roi_mask=roi_mask_global)
            meas = measure_spermatids(seg, cfg)
            rows.extend(rows_from_results(meas["results"], z_idx, um_per_px))

        df_2d = pd.DataFrame(rows)

        # Run tracking
        if not df_2d.empty and cfg.get("DO_TRACKING", True):
            _, df_tracks = track_across_slices(df_2d, cfg)
        else:
            df_tracks = pd.DataFrame()

        score, metrics = score_run(df_2d, df_tracks)

    except Exception as e:
        score = -1e12
        metrics = {"error": str(e)}

    sys.stdout.close()
    sys.stdout = old_stdout

    # Track progress
    eval_count += 1
    record = {"params": param_dict, **metrics}
    results_list.append(record)

    if score > best_global_score:
        best_global_score = score
        msg = (
            f"\r  Eval {eval_count:4d} | NEW BEST {score:8.1f}"
            f" | tracks={metrics.get('n_tracks', 0)}"
            f" | multi={metrics.get('multi_slice', 0)}"
            f" | single={metrics.get('single_slice', 0)}"
            f" | long={metrics.get('n_long', 0)}"
            f" | depth={metrics.get('avg_depth', 0)}"
        )
        sys.stdout.write(msg + "  \n")
        sys.stdout.flush()

    return -score  # DE minimizes, so negate


def cb_generation(xk, convergence):
    """Called at the end of each DE generation."""
    print(f"  Generation complete. Population convergence: {convergence:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global images_to_eval, z_values_eval, roi_mask_global, results_list

    parser = argparse.ArgumentParser(
        description="Evolutionary parameter tuner for Saturn V3 tracking"
    )
    parser.add_argument("--dir", default=None,
                       help="Directory containing .tif slices")
    parser.add_argument("--slices", default="0-12",
                       help="Z slices to use for tuning, e.g. 0-12 or 3,5,7-10")
    parser.add_argument("--um-per-px", type=float, default=None,
                       help="Override calibration (um/px)")
    parser.add_argument("--new-roi", action="store_true",
                       help="Force drawing a new ROI")
    parser.add_argument("--maxiter", type=int, default=10,
                       help="Number of DE generations (default: 10)")
    parser.add_argument("--popsize", type=int, default=8,
                       help="DE population multiplier (default: 8)")
    args = parser.parse_args()

    base_dir = args.dir
    slice_str = args.slices

    # GUI mode if no --dir given
    if base_dir is None:
        root = tk.Tk()
        root.withdraw()
        print("Please select the folder containing .tif slices...")
        base_dir = filedialog.askdirectory(title="Select Sperm Images Folder")
        if not base_dir:
            print("No folder selected. Exiting.")
            return

        slice_str = simpledialog.askstring(
            "Z-Slice Range",
            "Enter consecutive slices (e.g. 0-12):",
            initialvalue="0-12"
        )
        root.destroy()
        if not slice_str:
            print("No slices specified. Exiting.")
            return

    um_per_px = args.um_per_px if args.um_per_px is not None else CONFIG["UM_PER_PX_XY"]
    print(f"Calibration: {um_per_px:.6f} um/px")

    z_list = parse_slices_arg(slice_str)
    warn_if_nonconsecutive(z_list)

    files = glob.glob(os.path.join(base_dir, "*.tif")) + glob.glob(os.path.join(base_dir, "*.tiff"))
    if not files:
        print(f"No .tif/.tiff files found in {base_dir}")
        sys.exit(1)

    selected = []
    for f in files:
        m = re.search(r"z(\d+)", os.path.basename(f), re.IGNORECASE)
        if not m:
            continue
        z_val = int(m.group(1))
        if z_val in z_list:
            selected.append((z_val, f))
    selected.sort()

    if not selected:
        print("Could not find requested slices in the file names.")
        sys.exit(1)

    for z_val, f in selected:
        print(f"Loading z{z_val}: {os.path.basename(f)}")
        img = robust_imread(f)
        if img.ndim > 2:
            img = img[0]
            if img.ndim > 2:
                img = img[:, :, 0]
        images_to_eval.append(img)
        z_values_eval.append(z_val)

    print(f"\nLoaded {len(images_to_eval)} images for evolutionary optimization: z={z_values_eval}")
    roi_mask_global = build_roi(images_to_eval, force_redraw=args.new_roi)

    # ── Build bounds and seed from current CONFIG ────────────────────────
    bounds = [(lo, hi) for (_, lo, hi, _) in PARAM_SPACE]

    # Seed: extract current CONFIG values as starting point
    x0 = []
    print("\nSeed parameters (from current CONFIG):")
    for key, lo, hi, is_int in PARAM_SPACE:
        val = CONFIG.get(key, (lo + hi) / 2)
        # Clamp to bounds
        val = max(lo, min(hi, val))
        x0.append(val)
        print(f"  {key:45s} = {val}  (bounds: [{lo}, {hi}])")
    
    n_params = len(PARAM_SPACE)
    total_evals_est = args.maxiter * args.popsize * n_params

    print(f"\n{'='*70}")
    print(f"  EVOLUTIONARY PARAMETER TUNING")
    print(f"  Parameters:   {n_params}")
    print(f"  Generations:  {args.maxiter}")
    print(f"  Pop size:     {args.popsize} x {n_params} = {args.popsize * n_params}")
    print(f"  Est. evals:   ~{total_evals_est}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = differential_evolution(
        func=objective_fn,
        args=(um_per_px,),
        bounds=bounds,
        x0=x0,
        maxiter=args.maxiter,
        popsize=args.popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        callback=cb_generation,
        disp=False,
        polish=True,
        seed=42,
    )
    dt = time.time() - t0

    print(f"\nOptimization finished in {dt:.0f}s across {eval_count} evaluations.")

    # ── Extract best parameters ──────────────────────────────────────────
    best_params = {}
    for i, (key, lo, hi, is_int) in enumerate(PARAM_SPACE):
        val = result.x[i]
        if is_int:
            val = int(round(val))
        else:
            val = round(val, 4)
        best_params[key] = val

    # Sort results by score
    results_list.sort(key=lambda d: d.get("score", -1e18), reverse=True)
    best = results_list[0]

    print("\n" + "=" * 70)
    print("  BEST PARAMETERS FOUND")
    print("=" * 70)
    for key, val in best_params.items():
        seed_val = CONFIG.get(key)
        delta = ""
        if seed_val is not None:
            try:
                d = val - seed_val
                delta = f"  (delta: {d:+.4f})"
            except:
                pass
        print(f"  {key:45s} = {val}{delta}")

    print("\nBest metrics:")
    for k in ["n_tracks", "multi_slice", "total_linked", "avg_depth",
              "single_slice", "n_long", "n_tort", "n_thick", "n_taper",
              "l3d_median_um", "score"]:
        print(f"  {k:24s}: {best.get(k)}")

    # ── Save results ─────────────────────────────────────────────────────
    # Incremental best params filename
    from pathlib import Path
    script_dir = Path(__file__).parent
    existing = sorted(script_dir.glob("best_params_*.json"))
    next_num = 1
    if existing:
        for ep in existing:
            try:
                num = int(ep.stem.split("_")[-1])
                next_num = max(next_num, num + 1)
            except ValueError:
                pass

    best_param_filename = f"best_params_{next_num:03d}.json"
    best_param_path = script_dir / best_param_filename
    with open(best_param_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n[OK] Best parameters saved: {best_param_path}")

    # Full search history
    history_path = script_dir / "tuning_results_saturnv3.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)
    print(f"[OK] Full search history: {history_path}")

    print(f"\nTo use these parameters:")
    print(f"  GUI:  Click 'Load Tuned Params' -> select {best_param_filename}")
    print(f"  CLI:  python sperm_segmentation_saturnv3.py --batch --params {best_param_filename}")


if __name__ == "__main__":
    main()
