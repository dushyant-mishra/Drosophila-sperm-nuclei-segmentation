#!/usr/bin/env python3
"""
Evolutionary Parameter Tuner for Saturn V5 Tracking

Biology- and hardware-aware tuner for the *tracking* stage of the
Drosophila sperm nucleus pipeline.

This version is aligned to the V5 biological assumptions:
- mature Drosophila sperm nuclei are very long in XY but extremely thin in Z
- with this Leica SP8 stack (z-step ~1.04 µm), single-slice nuclei can be biologically valid
- width/area-derived metrics are PSF-sensitive and should be penalized more softly
- long, tortuous, implausibly merged tracks remain strong negatives

Compared with the older tuner, this version:
- imports from sperm_segmentation_v5_biobased_psfaware.py
- saves all outputs to:
    C:/Users/dmishra/Desktop/sperm_project/parameter_tuning_results
- removes the old heavy bias against single-slice tracks
- reduces the weight of PSF-sensitive penalties (thickness, taper)
- keeps a strong bias toward biologically plausible 3D lengths (~9–10 µm)
- keeps a strong penalty for monster merges and excessive tortuosity

Usage
-----
GUI mode:
    python tune_universal_parameters_v5_biobased.py

CLI mode:
    python tune_universal_parameters_v5_biobased.py --dir "path/to/images" --slices 0-12

Notes
-----
This tuner optimizes only the tracking / overlap parameters.
It does not tune the audit thresholds themselves.
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
from pathlib import Path
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
from tkinter import filedialog, simpledialog, messagebox

# -----------------------------------------------------------------------------
# Import V5 pipeline functions directly for fast in-process execution
# -----------------------------------------------------------------------------
try:
    from sperm_segmentation_saturnv5 import (
        CONFIG,
        segment_slice,
        measure_spermatids,
        track_across_slices,
        rows_from_results,
        normalize_display,
        robust_imread,
    )
except Exception as e:
    print(f"Error: Could not import from sperm_segmentation_saturnv5.py: {e}")
    raise

# -----------------------------------------------------------------------------
# Output configuration
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\dmishra\Desktop\sperm_project\parameter_tuning_results")
ROI_SAVE_PATH = DEFAULT_OUTPUT_DIR / "last_drawn_roi_saturnv5_tune.tif"

# -----------------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------------
eval_count = 0
best_global_score = -1e18
results_list = []
images_to_eval = []
z_values_eval = []
roi_mask_global = None

# -----------------------------------------------------------------------------
# Parameter space
# -----------------------------------------------------------------------------
# Centered around the v5 tuned defaults, but still wide enough to explore.
# Keep ranges biologically sane for a crowded, diffraction-limited confocal stack.
PARAM_SPACE = [
    ("OVERLAP_STABILITY_THRESHOLD",        0.05,  0.22,  False),
    ("OVERLAP_ORIENTATION_DEG",            8.0,   35.0,  False),
    ("OVERLAP_MULTIPLIER",                 1.10,  1.80,  False),
    ("TRACK_MAX_DIST_UM",                  4.0,   10.5,  False),
    ("TRACK_BBOX_PADDING_PX",              1,     8,     True),
    ("CONSERVATIVE_MAX_WIDTH_JUMP_RATIO",  0.25,  0.80,  False),
    ("CONSERVATIVE_MAX_LENGTH_JUMP_RATIO", 0.30,  0.85,  False),
    ("CONSERVATIVE_MAX_AREA_JUMP_RATIO",   0.35,  0.80,  False),
    ("CONSERVATIVE_MAX_CENTROID_JUMP_UM",  5.0,   14.0,  False),
]

# -----------------------------------------------------------------------------
# Audit-like review thresholds used only for scoring
# -----------------------------------------------------------------------------
# These are *not* literal physical truths. They are tuning heuristics aligned to
# the V5 biology and optical limitations.
REVIEW = {
    "length_thresh": 15.0,      # strong review threshold for likely over-merges
    "tort_thresh": 1.5,         # strong review threshold for fused/erratic tracks
    "thick_thresh": 2.0,        # PSF-sensitive -> softer penalty only
    "taper_thresh": 1.5,        # PSF-sensitive -> softer penalty only
    "target_len_um": 9.5,       # biology-guided mature nucleus target length
}


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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




def preview_loaded_roi(roi_img, roi_mask, roi_path=None):
    """
    Display a preview of a loaded ROI overlaid on the first image and ask whether to use it.
    Returns True if accepted, False otherwise.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    base = normalize_display(roi_img)
    ax.imshow(base, cmap="gray")
    overlay = np.zeros((*roi_mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = roi_mask.astype(np.float32) * 0.28
    ax.imshow(overlay)
    ys, xs = np.where(roi_mask)
    if ys.size and xs.size:
        ax.contour(roi_mask.astype(np.uint8), levels=[0.5], colors='yellow', linewidths=1.5)
    title = "Loaded ROI Preview"
    if roi_path:
        title += f"\n{roi_path}"
    title += "\nClose window to continue"
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    try:
        fig.canvas.manager.window.attributes("-topmost", 1)
        fig.canvas.manager.window.attributes("-topmost", 0)
        fig.canvas.manager.window.focus_force()
    except Exception:
        pass
    plt.show(block=True)
    try:
        return messagebox.askyesno(
            "Use this ROI?",
            "Does this ROI look correct for the current image stack?"
        )
    except Exception:
        return True
def build_roi(images, force_redraw=False, interactive_prompt=False):
    """Interactive polygon ROI drawing with optional prompt to reuse a saved ROI."""
    ensure_dir(DEFAULT_OUTPUT_DIR)
    roi_img = images[0]
    roi_mask = None

    if force_redraw and ROI_SAVE_PATH.exists():
        try:
            ROI_SAVE_PATH.unlink()
        except Exception:
            pass

    roi_candidate_path = ROI_SAVE_PATH if ROI_SAVE_PATH.exists() else None

    if interactive_prompt and not force_redraw:
        try:
            if ROI_SAVE_PATH.exists():
                choice = messagebox.askyesnocancel(
                    "ROI Reuse",
                    f"A saved ROI mask was found:\n{ROI_SAVE_PATH}\n\n"
                    "Yes = reuse this ROI\n"
                    "No = choose a different saved ROI file\n"
                    "Cancel = draw a new ROI"
                )
            else:
                choice = messagebox.askyesno(
                    "ROI Reuse",
                    "Would you like to load a previously saved ROI mask instead of drawing a new one?"
                )

            if choice is True:
                if ROI_SAVE_PATH.exists():
                    roi_candidate_path = ROI_SAVE_PATH
                else:
                    chosen = filedialog.askopenfilename(
                        title="Select Saved ROI Mask",
                        filetypes=[("ROI mask files", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
                    )
                    roi_candidate_path = Path(chosen) if chosen else None
            elif choice is False and ROI_SAVE_PATH.exists():
                chosen = filedialog.askopenfilename(
                    title="Select Saved ROI Mask",
                    filetypes=[("ROI mask files", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
                )
                roi_candidate_path = Path(chosen) if chosen else None
            else:
                roi_candidate_path = None
        except Exception:
            roi_candidate_path = ROI_SAVE_PATH if ROI_SAVE_PATH.exists() else None

    if roi_candidate_path is not None and Path(roi_candidate_path).exists():
        try:
            print(f"\nLoaded ROI from {roi_candidate_path}.")
            roi_mask = robust_imread(str(roi_candidate_path)).astype(bool)
            if roi_mask.shape != roi_img.shape:
                print("Saved ROI shape mismatch. Redrawing.")
                roi_mask = None
            elif interactive_prompt:
                accepted = preview_loaded_roi(roi_img, roi_mask, roi_candidate_path)
                if not accepted:
                    print("Loaded ROI rejected by user. Redrawing.")
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

        tifffile.imwrite(str(ROI_SAVE_PATH), roi_mask.astype(np.uint8) * 255)
        print(f"Saved ROI to {ROI_SAVE_PATH}")

    return roi_mask


def safe_median(series, default=np.nan):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size > 0 else default


def safe_mean(series, default=np.nan):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else default


def pick_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


# ═════════════════════════════════════════════════════════════════════════════
#  BIOLOGY- AND HARDWARE-AWARE SCORE
# ═════════════════════════════════════════════════════════════════════════════

def score_run(df_2d, df_tracks, z_step_um):
    """
    Score a parameter set by analysing tracking output.

    Design principles for this Leica SP8 dataset:
    - Single-slice nuclei can be biologically valid because z-step (~1.04 µm)
      is much larger than the true mature nucleus diameter (~0.3 µm).
    - Therefore, fragmentation should be discouraged, but single-slice tracks
      should NOT be heavily penalized by default.
    - Monster merges remain a strong negative: excessive 3D length, high
      tortuosity, abrupt area/width inconsistencies.
    - PSF-sensitive metrics (thickness, taper) are still biologically useful,
      especially for matched WT-versus-mutant comparisons, so they remain in
      the score with softer but nonzero penalties.
    - Reward biologically plausible median 3D lengths near ~9–10 µm.
    """
    if df_2d.empty:
        return -1e12, {"reason": "no_2d"}
    if df_tracks is None or df_tracks.empty:
        return -1e12, {"reason": "no_3d"}

    n_2d = len(df_2d)
    n_tracks = len(df_tracks)

    length_col = pick_col(df_tracks, ["total_3d_length_um", "length_3d_um_est", "max_length_um"])
    tort_col = pick_col(df_tracks, ["tortuosity_3d", "tortuosity"])
    thick_col = pick_col(df_tracks, ["thickness_um", "effective_thickness_um", "median_width_um"])
    taper_col = pick_col(df_tracks, ["taper_ratio", "morphological_taper_ratio"])
    nslices_col = pick_col(df_tracks, ["n_slices", "n_detections"])
    zextent_col = pick_col(df_tracks, ["z_extent_um", "z_height_um", "vertical_span_um"])

    if length_col is None or nslices_col is None:
        return -1e12, {"reason": "missing_columns"}

    lengths = df_tracks[length_col].astype(float)
    nslices = df_tracks[nslices_col].astype(float)
    tort = df_tracks[tort_col].astype(float) if tort_col else pd.Series(dtype=float)
    thick = df_tracks[thick_col].astype(float) if thick_col else pd.Series(dtype=float)
    taper = df_tracks[taper_col].astype(float) if taper_col else pd.Series(dtype=float)
    zextent = df_tracks[zextent_col].astype(float) if zextent_col else nslices.clip(lower=1) * float(z_step_um)

    l3d_med = safe_median(lengths)
    l3d_mean = safe_mean(lengths)
    z_med = safe_median(zextent)

    n_long = int((lengths > REVIEW["length_thresh"]).sum())
    n_tort = int((tort > REVIEW["tort_thresh"]).sum()) if tort_col else 0
    n_thick = int((thick > REVIEW["thick_thresh"]).sum()) if thick_col else 0
    n_taper = int((taper > REVIEW["taper_thresh"]).sum()) if taper_col else 0

    n_single = int((nslices <= 1).sum())
    multi_slice = int((nslices > 1).sum())
    total_linked = int(nslices[nslices > 1].sum())
    avg_depth = float(nslices[nslices > 1].mean()) if multi_slice > 0 else 0.0

    single_frac = n_single / max(n_tracks, 1)
    multi_frac = multi_slice / max(n_tracks, 1)
    long_frac = n_long / max(n_tracks, 1)

    score = 0.0

    # 1. Reward biologically plausible track consolidation, but modestly.
    score += 0.90 * multi_slice
    score += 0.18 * total_linked
    score += 18.0 * avg_depth

    # 2. Soft single-slice handling.
    # Single-slice tracks are valid in this hardware regime, so only penalize if
    # they dominate the population excessively.
    if single_frac > 0.70:
        score -= 120.0 * (single_frac - 0.70)
    if multi_frac < 0.20:
        score -= 150.0 * (0.20 - multi_frac)

    # 3. Strong penalties for likely monster merges.
    score -= 2.8 * n_long
    score -= 2.2 * n_tort

    # 4. Softer penalties for PSF-sensitive or derived quantities.
    score -= 0.35 * n_thick
    score -= 0.50 * n_taper

    # 5. Strong biological alignment to mature nucleus length.
    if np.isfinite(l3d_med):
        score -= 40.0 * abs(l3d_med - REVIEW["target_len_um"])
    if np.isfinite(l3d_mean) and l3d_mean > 13.0:
        score -= 60.0 * (l3d_mean - 13.0)

    # 6. Encourage moderate Z continuity, but not excessively.
    if np.isfinite(z_med) and z_med > 4.0:
        score -= 35.0 * (z_med - 4.0)

    # 7. Strong structural penalty if long-track fraction becomes too large.
    if long_frac > 0.20:
        score -= 250.0 * (long_frac - 0.20)

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
        "l3d_mean_um": round(l3d_mean, 3) if np.isfinite(l3d_mean) else None,
        "zextent_median_um": round(z_med, 3) if np.isfinite(z_med) else None,
        "score": round(score, 2),
    }
    return score, metrics


# ═════════════════════════════════════════════════════════════════════════════
#  OBJECTIVE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def objective_fn(x, um_per_px, z_step_um):
    global eval_count, best_global_score, results_list
    global images_to_eval, z_values_eval, roi_mask_global

    param_dict = {}
    for i, (key, lo, hi, is_int) in enumerate(PARAM_SPACE):
        val = x[i]
        if is_int:
            val = int(round(val))
        param_dict[key] = val

    cfg = CONFIG.copy()
    cfg.update(param_dict)

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

        score, metrics = score_run(df_2d, df_tracks, z_step_um)

    except Exception as e:
        score = -1e12
        metrics = {"error": str(e)}

    sys.stdout.close()
    sys.stdout = old_stdout

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
            f" | zmed={metrics.get('zextent_median_um', 0)}"
            f" | Lmed={metrics.get('l3d_median_um', 0)}"
        )
        sys.stdout.write(msg + "  \n")
        sys.stdout.flush()

    return -score


def cb_generation(xk, convergence):
    print(f"  Generation complete. Population convergence: {convergence:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global images_to_eval, z_values_eval, roi_mask_global, results_list

    parser = argparse.ArgumentParser(
        description="Evolutionary parameter tuner for Saturn V5 tracking"
    )
    parser.add_argument("--dir", default=None,
                       help="Directory containing .tif/.tiff slices")
    parser.add_argument("--slices", default="0-12",
                       help="Z slices to use for tuning, e.g. 0-12 or 3,5,7-10")
    parser.add_argument("--um-per-px", type=float, default=None,
                       help="Override calibration (um/px). Defaults to V5 CONFIG.")
    parser.add_argument("--z-step-um", type=float, default=None,
                       help="Override z-step in µm. Defaults to V5 CONFIG.")
    parser.add_argument("--new-roi", action="store_true",
                       help="Force drawing a new ROI")
    parser.add_argument("--maxiter", type=int, default=10,
                       help="Number of DE generations (default: 10)")
    parser.add_argument("--popsize", type=int, default=8,
                       help="DE population multiplier (default: 8)")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR),
                       help="Folder to save tuning results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Update ROI path to follow chosen outdir
    global ROI_SAVE_PATH
    ROI_SAVE_PATH = outdir / "last_drawn_roi_saturnv5_tune.tif"

    base_dir = args.dir
    slice_str = args.slices

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
    z_step_um = args.z_step_um if args.z_step_um is not None else CONFIG["UM_PER_SLICE_Z"]
    print(f"Calibration: {um_per_px:.6f} um/px")
    print(f"Z-step:      {z_step_um:.6f} um")
    print(f"Output dir:  {outdir}")

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

    print(f"\nLoaded {len(images_to_eval)} images for optimization: z={z_values_eval}")
    roi_mask_global = build_roi(images_to_eval, force_redraw=args.new_roi, interactive_prompt=(args.dir is None))

    bounds = [(lo, hi) for (_, lo, hi, _) in PARAM_SPACE]

    x0 = []
    print("\nSeed parameters (from V5 CONFIG):")
    for key, lo, hi, is_int in PARAM_SPACE:
        val = CONFIG.get(key, (lo + hi) / 2)
        val = max(lo, min(hi, val))
        x0.append(val)
        print(f"  {key:45s} = {val}  (bounds: [{lo}, {hi}])")

    n_params = len(PARAM_SPACE)
    total_evals_est = args.maxiter * args.popsize * n_params

    print(f"\n{'='*78}")
    print("  EVOLUTIONARY PARAMETER TUNING (V5 BIOLOGY- / HARDWARE-AWARE)")
    print(f"  Parameters:   {n_params}")
    print(f"  Generations:  {args.maxiter}")
    print(f"  Pop size:     {args.popsize} x {n_params} = {args.popsize * n_params}")
    print(f"  Est. evals:   ~{total_evals_est}")
    print("  Scoring bias: preserve plausible single-slice nuclei; penalize over-merges")
    print(f"{'='*78}\n")

    t0 = time.time()
    result = differential_evolution(
        func=objective_fn,
        args=(um_per_px, z_step_um),
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

    best_params = {}
    for i, (key, lo, hi, is_int) in enumerate(PARAM_SPACE):
        val = result.x[i]
        if is_int:
            val = int(round(val))
        else:
            val = round(val, 4)
        best_params[key] = val

    results_list.sort(key=lambda d: d.get("score", -1e18), reverse=True)
    best = results_list[0]

    print("\n" + "=" * 78)
    print("  BEST PARAMETERS FOUND")
    print("=" * 78)
    for key, val in best_params.items():
        seed_val = CONFIG.get(key)
        delta = ""
        if seed_val is not None:
            try:
                d = val - seed_val
                delta = f"  (delta: {d:+.4f})"
            except Exception:
                pass
        print(f"  {key:45s} = {val}{delta}")

    print("\nBest metrics:")
    for k in [
        "n_tracks", "multi_slice", "total_linked", "avg_depth",
        "single_slice", "n_long", "n_tort", "n_thick", "n_taper",
        "l3d_median_um", "l3d_mean_um", "zextent_median_um", "score"
    ]:
        print(f"  {k:24s}: {best.get(k)}")

    existing = sorted(outdir.glob("best_params_v5_*.json"))
    next_num = 1
    if existing:
        for ep in existing:
            try:
                num = int(ep.stem.split("_")[-1])
                next_num = max(next_num, num + 1)
            except ValueError:
                pass

    best_param_filename = f"best_params_v5_{next_num:03d}.json"
    best_param_path = outdir / best_param_filename
    with open(best_param_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n[OK] Best parameters saved: {best_param_path}")

    history_path = outdir / "tuning_results_saturnv5.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)
    print(f"[OK] Full search history: {history_path}")

    summary_path = outdir / "tuning_summary_saturnv5.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SATURN V5 TRACKING TUNING SUMMARY\n")
        f.write("=" * 72 + "\n")
        f.write(f"Output directory: {outdir}\n")
        f.write(f"Images tuned: {len(images_to_eval)} | z={z_values_eval}\n")
        f.write(f"Calibration: {um_per_px:.6f} um/px | z-step: {z_step_um:.6f} um\n")
        f.write("\nBest parameters:\n")
        for key, val in best_params.items():
            f.write(f"  {key}: {val}\n")
        f.write("\nBest metrics:\n")
        for k in [
            "n_tracks", "multi_slice", "total_linked", "avg_depth",
            "single_slice", "n_long", "n_tort", "n_thick", "n_taper",
            "l3d_median_um", "l3d_mean_um", "zextent_median_um", "score"
        ]:
            f.write(f"  {k}: {best.get(k)}\n")
        f.write("\nBiology / hardware note:\n")
        f.write("  This V5 tuner preserves biologically plausible single-slice nuclei for this Leica\n")
        f.write("  SP8 stack and penalizes likely over-merges more strongly than shallow tracks.\n")
    print(f"[OK] Summary text file: {summary_path}")

    print("\nTo use these parameters:")
    print(f"  GUI: Click 'Load Tuned Params' -> select {best_param_filename}")
    print(f"  CLI: python sperm_segmentation_v5_biobased_psfaware.py --batch --params \"{best_param_path}\"")


if __name__ == "__main__":
    main()
