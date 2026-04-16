#!/usr/bin/env python3
"""
Refined 3D biology-aware tuner for the Saturn spermatid pipeline (v1 Optimized).

What is improved vs the previous 3D tuner
-----------------------------------------
1. Optimizes against the *tracked 3D nucleus population*, not raw 2D count.
2. Adds explicit penalties for:
   - excessively long tracks
   - high tortuosity outliers
   - extreme taper outliers
   - too many single-slice tracks
   - over-fragmentation (raw 2D / 3D ratio too high)
3. Uses tighter, biologically constrained parameter bounds.
4. Saves a ranked JSON history with all score components for debugging.

Biological targets
------------------
These defaults are based on your current v12 report:
- 3D median length ~ 9.3–10.2 µm
- 3D tortuosity median ~ 1.10–1.14
- thickness median ~ 1.20–1.28 µm
- taper median ~ 1.00–1.18

Recommended usage
-----------------
Use a consecutive block of slices from one tubule:
    python tune_parameters_saturnv1.py --dir "C:\\Users\\dmishra\\Desktop\\sperm images" --slices 0-12 --new-roi

Needs
-----
This script expects sperm_segmentation_saturnv1.py in the same folder, with:
- CONFIG
- segment_slice
- measure_spermatids
- track_across_slices
- normalize_display
- make_overlay
- robust_imread
"""

import os
import sys
import re
import json
import time
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import filedialog, simpledialog

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from scipy.optimize import differential_evolution
import tkinter as tk
from tkinter import filedialog, simpledialog

try:
    from sperm_segmentation_saturnv1 import (
        CONFIG,
        segment_slice,
        measure_spermatids,
        track_across_slices,
        normalize_display,
        make_overlay,
        robust_imread,
    )
except Exception as e:
    print("Error: Could not import sperm_segmentation_saturnv1.py from the current directory.")
    raise

ROI_SAVE_PATH = "last_drawn_roi_saturnv1.tif"

eval_count = 0
best_global_score = -1e18
results_list = []
images_to_eval = []
z_values_eval = []
roi_mask_global = None

# -------------------------------------------------------------------------
# Biological targets inferred from your current report
# -------------------------------------------------------------------------
TARGETS = {
    "L3D_MEDIAN_UM": 10.0,
    "L3D_TARGET_RANGE_UM": (8.5, 11.5),
    "L3D_HARD_RANGE_UM": (5.0, 20.0),

    "TORT_MEDIAN": 1.10,
    "TORT_GOOD_MAX": 1.50,
    "TORT_HARD_MAX": 2.00,

    "THICK_MEDIAN_UM": 1.20,
    "THICK_GOOD_RANGE_UM": (0.9, 1.8),
    "THICK_HARD_RANGE_UM": (0.8, 2.2),

    "TAPER_MEDIAN": 1.00,
    "TAPER_GOOD_MAX": 1.80,
    "TAPER_HARD_MAX": 2.50,

    # tracking structure
    "MAX_RAW_TO_3D_RATIO": 4.0,
    "TARGET_MULTI_SLICE_FRACTION": 0.50,   # around report value
    "MAX_SINGLE_SLICE_FRACTION": 0.60,

    # long-tail outlier expectations
    "MAX_LONG_TRACK_FRAC": 0.05,           # >20 µm should be rare
    "MAX_EXTREME_TORT_FRAC": 0.02,         # >2.0 should be very rare
    "MAX_EXTREME_TAPER_FRAC": 0.03,        # >2.5 should be very rare
}


def parse_slices_arg(text):
    text = text.strip()
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip())
            b = int(b.strip())
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
        print("  3D tuning is best on a consecutive block from one tubule.")
        print(f"  Current slice gaps: {diffs.tolist()}\n")


def build_roi(images, force_redraw=False):
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
            if event.inaxes != ax: return
            if event.button == 1: # Left
                pts.append((event.xdata, event.ydata))
                redraw()
            elif event.button == 3: # Right
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
            # Force the Matplotlib window to the front
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
        # Automatically close the loop for the Path logic
        full_pts = pts + [pts[0]]
        path = Path(full_pts)
        roi_mask = path.contains_points(points).reshape(H, W)

        tifffile.imwrite(ROI_SAVE_PATH, roi_mask.astype(np.uint8) * 255)
        print(f"Saved ROI to {ROI_SAVE_PATH}")

    return roi_mask


def rows_from_results(results, z_idx, um):
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append({
            "z_slice": z_idx,
            "sperm_id": i,
            "length_px_geodesic": float(r.get("length_px_geodesic", np.nan)),
            "length_um_geodesic": float(r.get("length_px_geodesic", np.nan)) * um,
            "width_px": float(r.get("width_px", np.nan)),
            "width_um": float(r.get("width_px", np.nan)) * um,
            "length_width_ratio": float(r.get("length_width_ratio", np.nan)),
            "tortuosity": float(r.get("tortuosity", np.nan)),
            "n_endpoints": int(r.get("n_endpoints", 0)),
            "n_branch_nodes": int(r.get("n_branch_nodes", 0)),
            "centroid_x": float(r.get("centroid_x", np.nan)),
            "centroid_y": float(r.get("centroid_y", np.nan)),
            "area_px": float(r.get("area_px", 0.0)),
        })
    return rows


def safe_median(series, default=np.nan):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.median(arr))


def safe_mean(series, default=np.nan):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def safe_fraction(mask_like, default=1.0):
    arr = np.asarray(mask_like)
    if arr.size == 0:
        return default
    return float(np.mean(arr.astype(bool)))


def get_track_columns(df_tracks):
    """Handle slight naming variations safely."""
    cols = set(df_tracks.columns)
    length_col = None
    for c in ["length_3d_um_est", "total_3d_length_um", "max_length_um"]:
        if c in cols:
            length_col = c
            break

    tort_col = None
    for c in ["tortuosity_3d", "tortuosity"]:
        if c in cols:
            tort_col = c
            break

    thick_col = None
    for c in ["thickness_um", "effective_thickness_um", "median_width_um"]:
        if c in cols:
            thick_col = c
            break

    taper_col = None
    for c in ["taper_ratio", "morphological_taper_ratio"]:
        if c in cols:
            taper_col = c
            break

    nslices_col = None
    for c in ["n_slices", "n_detections"]:
        if c in cols:
            nslices_col = c
            break

    return length_col, tort_col, thick_col, taper_col, nslices_col


def score_tracks(df_2d, df_tracks):
    if df_2d.empty:
        return -1e12, {"reason": "no_2d"}
    if df_tracks is None or df_tracks.empty:
        return -1e12, {"reason": "no_3d"}

    length_col, tort_col, thick_col, taper_col, nslices_col = get_track_columns(df_tracks)
    if length_col is None:
        return -1e12, {"reason": "missing_length_col"}

    raw_count = int(len(df_2d))
    n_tracks = int(len(df_tracks))
    raw_to_3d = raw_count / max(n_tracks, 1)

    l3d = df_tracks[length_col].astype(float)
    l3d_med = safe_median(l3d)
    l3d_mean = safe_mean(l3d)

    if tort_col is not None:
        tort = df_tracks[tort_col].astype(float)
        tort_med = safe_median(tort)
    else:
        tort = pd.Series([], dtype=float)
        tort_med = np.nan

    if thick_col is not None:
        thick = df_tracks[thick_col].astype(float)
        thick_med = safe_median(thick)
    else:
        thick = pd.Series([], dtype=float)
        thick_med = np.nan

    if taper_col is not None:
        taper = df_tracks[taper_col].astype(float)
        taper_med = safe_median(taper)
    else:
        taper = pd.Series([], dtype=float)
        taper_med = 1.0

    if nslices_col is not None:
        nslices = df_tracks[nslices_col].astype(float)
        single_slice_frac = safe_fraction(nslices <= 1, default=1.0)
        multi_slice_frac = safe_fraction(nslices > 1, default=0.0)
    else:
        single_slice_frac = np.nan
        multi_slice_frac = np.nan

    # good fractions
    frac_len_good = safe_fraction((l3d >= TARGETS["L3D_TARGET_RANGE_UM"][0]) & (l3d <= TARGETS["L3D_TARGET_RANGE_UM"][1]))
    frac_len_hard = safe_fraction((l3d >= TARGETS["L3D_HARD_RANGE_UM"][0]) & (l3d <= TARGETS["L3D_HARD_RANGE_UM"][1]))
    frac_long = safe_fraction(l3d > TARGETS["L3D_HARD_RANGE_UM"][1], default=0.0)

    if tort_col is not None:
        frac_tort_good = safe_fraction(tort <= TARGETS["TORT_GOOD_MAX"])
        frac_tort_hard = safe_fraction(tort <= TARGETS["TORT_HARD_MAX"])
        frac_extreme_tort = safe_fraction(tort > TARGETS["TORT_HARD_MAX"], default=0.0)
    else:
        frac_tort_good = frac_tort_hard = 1.0
        frac_extreme_tort = 0.0

    if thick_col is not None:
        frac_thick_good = safe_fraction((thick >= TARGETS["THICK_GOOD_RANGE_UM"][0]) & (thick <= TARGETS["THICK_GOOD_RANGE_UM"][1]))
        frac_thick_hard = safe_fraction((thick >= TARGETS["THICK_HARD_RANGE_UM"][0]) & (thick <= TARGETS["THICK_HARD_RANGE_UM"][1]))
    else:
        frac_thick_good = frac_thick_hard = 1.0

    if taper_col is not None:
        frac_taper_good = safe_fraction(taper <= TARGETS["TAPER_GOOD_MAX"])
        frac_taper_hard = safe_fraction(taper <= TARGETS["TAPER_HARD_MAX"])
        frac_extreme_taper = safe_fraction(taper > TARGETS["TAPER_HARD_MAX"], default=0.0)
    else:
        frac_taper_good = frac_taper_hard = 1.0
        frac_extreme_taper = 0.0

    # distribution width
    l3d_iqr = float(np.subtract(*np.percentile(l3d.dropna(), [75, 25]))) if len(l3d.dropna()) > 3 else np.nan

    score = 0.0

    # Base reward: true tracks, not raw fragments
    score += 1.2 * n_tracks

    # Reward good fractions strongly
    score += 500.0 * frac_len_good
    score += 300.0 * frac_len_hard
    score += 250.0 * frac_tort_good
    score += 180.0 * frac_tort_hard
    score += 220.0 * frac_thick_good
    score += 120.0 * frac_thick_hard
    score += 180.0 * frac_taper_good
    score += 80.0 * frac_taper_hard

    # Median alignment
    score -= 100.0 * abs(l3d_med - TARGETS["L3D_MEDIAN_UM"])
    if np.isfinite(tort_med):
        score -= 140.0 * abs(tort_med - TARGETS["TORT_MEDIAN"])
    if np.isfinite(thick_med):
        score -= 100.0 * abs(thick_med - TARGETS["THICK_MEDIAN_UM"])
    if np.isfinite(taper_med):
        score -= 50.0 * abs(taper_med - TARGETS["TAPER_MEDIAN"])

    # Over-fragmentation penalty
    if raw_to_3d > TARGETS["MAX_RAW_TO_3D_RATIO"]:
        score -= 180.0 * (raw_to_3d - TARGETS["MAX_RAW_TO_3D_RATIO"])

    # Single-slice track penalty
    if np.isfinite(single_slice_frac) and single_slice_frac > TARGETS["MAX_SINGLE_SLICE_FRACTION"]:
        score -= 250.0 * (single_slice_frac - TARGETS["MAX_SINGLE_SLICE_FRACTION"])

    # Encourage a healthy multi-slice fraction near the current report
    if np.isfinite(multi_slice_frac):
        score -= 120.0 * abs(multi_slice_frac - TARGETS["TARGET_MULTI_SLICE_FRACTION"])

    # Long-tail outlier penalties
    if frac_long > TARGETS["MAX_LONG_TRACK_FRAC"]:
        score -= 500.0 * (frac_long - TARGETS["MAX_LONG_TRACK_FRAC"])

    if frac_extreme_tort > TARGETS["MAX_EXTREME_TORT_FRAC"]:
        score -= 600.0 * (frac_extreme_tort - TARGETS["MAX_EXTREME_TORT_FRAC"])

    if frac_extreme_taper > TARGETS["MAX_EXTREME_TAPER_FRAC"]:
        score -= 500.0 * (frac_extreme_taper - TARGETS["MAX_EXTREME_TAPER_FRAC"])

    # Too-wide length spread penalty
    if np.isfinite(l3d_iqr) and l3d_iqr > 4.5:
        score -= 80.0 * (l3d_iqr - 4.5)

    metrics = {
        "raw_2d_count": raw_count,
        "n_tracks": n_tracks,
        "raw_to_3d_ratio": round(raw_to_3d, 4),
        "l3d_median_um": round(l3d_med, 4) if np.isfinite(l3d_med) else None,
        "l3d_mean_um": round(l3d_mean, 4) if np.isfinite(l3d_mean) else None,
        "l3d_iqr_um": round(l3d_iqr, 4) if np.isfinite(l3d_iqr) else None,
        "tortuosity_median": round(tort_med, 4) if np.isfinite(tort_med) else None,
        "thickness_median_um": round(thick_med, 4) if np.isfinite(thick_med) else None,
        "taper_median": round(taper_med, 4) if np.isfinite(taper_med) else None,
        "single_slice_fraction": round(single_slice_frac, 4) if np.isfinite(single_slice_frac) else None,
        "multi_slice_fraction": round(multi_slice_frac, 4) if np.isfinite(multi_slice_frac) else None,
        "frac_len_good": round(frac_len_good, 4),
        "frac_len_hard": round(frac_len_hard, 4),
        "frac_long": round(frac_long, 4),
        "frac_tort_good": round(frac_tort_good, 4),
        "frac_tort_hard": round(frac_tort_hard, 4),
        "frac_extreme_tort": round(frac_extreme_tort, 4),
        "frac_thick_good": round(frac_thick_good, 4),
        "frac_thick_hard": round(frac_thick_hard, 4),
        "frac_taper_good": round(frac_taper_good, 4),
        "frac_taper_hard": round(frac_taper_hard, 4),
        "frac_extreme_taper": round(frac_extreme_taper, 4),
        "score": round(score, 4),
    }
    return score, metrics


def objective_fn(x, um_per_px):
    global eval_count, best_global_score, results_list, images_to_eval, z_values_eval, roi_mask_global

    th_hi, th_lo, bg_sigma, clahe_clip, min_len, max_wid, min_rat, max_tort, max_branch, max_endpt = x
    if th_lo >= th_hi:
        th_lo = th_hi - 1.0

    p_comb = {
        "THRESHOLD_HI": float(th_hi),
        "THRESHOLD_LO": float(th_lo),
        "BG_SIGMA": float(bg_sigma),
        "CLAHE_CLIP": float(clahe_clip),
        "MIN_SKEL_LEN_PX": float(min_len),
        "MAX_WIDTH_PX": float(max_wid),
        "MIN_LENGTH_WIDTH_RATIO": float(min_rat),
        "MAX_TORTUOSITY": float(max_tort),
        "MAX_BRANCH_NODES": int(round(max_branch)),
        "MAX_ENDPOINT_COUNT": int(round(max_endpt)),
    }

    cfg = CONFIG.copy()
    cfg.update(p_comb)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    score = -1e12
    metrics = {}
    try:
        rows = []
        for img, z_idx in zip(images_to_eval, z_values_eval):
            seg = segment_slice(img, cfg, z_idx=z_idx, debug_dir=None, roi_mask=roi_mask_global)
            meas = measure_spermatids(seg, cfg)
            rows.extend(rows_from_results(meas["results"], z_idx, um_per_px))

        df_2d = pd.DataFrame(rows)
        if not df_2d.empty and cfg.get("DO_TRACKING", True):
            _, df_tracks = track_across_slices(df_2d, cfg)
        else:
            df_tracks = pd.DataFrame()

        score, metrics = score_tracks(df_2d, df_tracks)

    except Exception as e:
        score = -1e12
        metrics = {"error": str(e)}

    sys.stdout.close()
    sys.stdout = old_stdout

    eval_count += 1
    record = {"params": p_comb, **metrics}
    results_list.append(record)

    if score > best_global_score:
        best_global_score = score
        msg = (
            f"\rEval {eval_count:5d} | New Best Score: {score:9.2f}"
            f" | 3D N={metrics.get('n_tracks', 0)}"
            f" | L3D med={metrics.get('l3d_median_um', 'NA')}"
            f" | tort med={metrics.get('tortuosity_median', 'NA')}"
            f" | thick med={metrics.get('thickness_median_um', 'NA')}"
            f" | single={metrics.get('single_slice_fraction', 'NA')}"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    return -score


def apply_best_config(img, z_idx, roi_mask, best_cfg):
    seg = segment_slice(img, best_cfg, z_idx=z_idx, roi_mask=roi_mask)
    meas = measure_spermatids(seg, best_cfg)
    overlay = make_overlay(img, meas["skel_label"])

    try:
        from skimage.segmentation import find_boundaries
        if roi_mask is not None:
            b = find_boundaries(roi_mask)
            overlay[b] = [255, 0, 0]
    except Exception:
        pass

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Optimized Overlay Z={z_idx} (N={len(meas['results'])})")
    plt.axis("off")
    plt.show(block=False)


def cb_generation(xk, convergence):
    print(f"\nGeneration complete. Population convergence: {convergence:.3f}")


def main():
    global images_to_eval, z_values_eval, roi_mask_global, results_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=r"C:\Users\dmishra\Desktop\sperm images", help="Directory containing .tif slices")
    parser.add_argument("--slices", default="0-12", help="Consecutive Z slices to use for tuning, e.g. 0-12")
    parser.add_argument("--um-per-px", type=float, default=None, help="Override calibration (um/px)")
    parser.add_argument("--new-roi", action="store_true", help="Force drawing a new ROI")
    parser.add_argument("--maxiter", type=int, default=10, help="Differential evolution generations")
    parser.add_argument("--popsize", type=int, default=8, help="Differential evolution population size")
    args = parser.parse_args()

    # Interactive Selection if not provided on CLI
    if not args.dir or args.dir == r"C:\Users\dmishra\Desktop\sperm images":
        root = tk.Tk()
        root.withdraw()
        selected_dir = filedialog.askdirectory(title="Select Folder containing .tif slices")
        if not selected_dir:
            print("Selection cancelled. Exiting.")
            sys.exit(0)
        args.dir = selected_dir
        
        # Ask for slices if default
        if args.slices == "0-12":
            entered_slices = simpledialog.askstring("Z-Slice Range", "Enter Z-slice range (e.g. 0-6):", initialvalue="0-6")
            if entered_slices:
                args.slices = entered_slices
        root.destroy()

    um_per_px = args.um_per_px if args.um_per_px is not None else CONFIG["UM_PER_PX_XY"]
    print(f"Calibration: {um_per_px:.6f} um/px")

    z_list = parse_slices_arg(args.slices)
    warn_if_nonconsecutive(z_list)

    files = glob.glob(os.path.join(args.dir, "*.tif")) + glob.glob(os.path.join(args.dir, "*.tiff"))
    if not files:
        print(f"No .tif/.tiff files found in {args.dir}")
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
        print("Could not find requested slices.")
        sys.exit(1)

    for z_val, f in selected:
        print(f"Loading {os.path.basename(f)}...")
        img = robust_imread(f)
        if img.ndim > 2:
            img = img[0]
            if img.ndim > 2:
                img = img[:, :, 0]
        images_to_eval.append(img)
        z_values_eval.append(z_val)

    print(f"\nLoaded {len(images_to_eval)} images for refined 3D optimization: {z_values_eval}")
    roi_mask_global = build_roi(images_to_eval, force_redraw=args.new_roi)

    # Tighter, biologically constrained bounds
    bounds = [
        (75.0, 96.0),   # THRESHOLD_HI
        (60.0, 88.0),   # THRESHOLD_LO
        (5.0, 26.0),    # BG_SIGMA
        (0.02, 0.12),   # CLAHE_CLIP
        (5.0, 18.0),    # MIN_SKEL_LEN_PX
        (4.0, 12.0),    # MAX_WIDTH_PX
        (1.5, 6.0),     # MIN_LENGTH_WIDTH_RATIO
        (1.2, 8.0),     # MAX_TORTUOSITY
        (0.0, 12.0),    # MAX_BRANCH_NODES
        (2.0, 12.0),    # MAX_ENDPOINT_COUNT
    ]

    print("\n--- Starting refined 3D biology-aware optimization ---")
    print("Targets:")
    print(f"  3D median length ~ {TARGETS['L3D_MEDIAN_UM']} µm")
    print(f"  3D tortuosity median ~ {TARGETS['TORT_MEDIAN']}")
    print(f"  3D thickness median ~ {TARGETS['THICK_MEDIAN_UM']} µm")
    print(f"  3D taper median ~ {TARGETS['TAPER_MEDIAN']}")
    print("Extra penalties:")
    print("  - too many single-slice tracks")
    print("  - too many very long tracks")
    print("  - too many extreme tortuosity/taper outliers")

    t0 = time.time()
    differential_evolution(
        func=objective_fn,
        args=(um_per_px,),
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        callback=cb_generation,
        disp=False,
        polish=True,
    )
    dt = time.time() - t0

    print(f"\nOptimization finished in {dt:.1f} s across {eval_count} evaluations.")

    results_list.sort(key=lambda d: d.get("score", -1e18), reverse=True)
    best = results_list[0]

    print("\n--- BEST CONFIGURATION ---")
    for k in [
        "score", "raw_2d_count", "n_tracks", "raw_to_3d_ratio", "l3d_median_um",
        "l3d_mean_um", "l3d_iqr_um", "tortuosity_median", "thickness_median_um",
        "taper_median", "single_slice_fraction", "multi_slice_fraction",
        "frac_len_good", "frac_long", "frac_extreme_tort", "frac_extreme_taper"
    ]:
        print(f"{k:24s}: {best.get(k)}")
    print(f"Parameters: {best['params']}")

    out_json = "tuning_results_saturnv1.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)
    print(f"Full search history saved to 'tuning_results_saturnv1.json'.")

    best_cfg = CONFIG.copy()
    best_cfg.update(best["params"])

    apply_best_config(images_to_eval[0], z_values_eval[0], roi_mask_global, best_cfg)
    if len(images_to_eval) > 1:
        apply_best_config(images_to_eval[-1], z_values_eval[-1], roi_mask_global, best_cfg)

    plt.show(block=True)


if __name__ == "__main__":
    main()
