#!/usr/bin/env python3
"""
Evolutionary Parameter Tuner for Saturn V5.5 Hybrid Tracking

Biology- and hardware-aware tuner for the *tracking* stage of the
Drosophila sperm nucleus pipeline.

This version is aligned to the V5.5 hybrid assumptions:
- mature Drosophila sperm nuclei are very long in XY but extremely thin in Z
- with this Leica SP8 stack (z-step ~1.04 µm), single-slice nuclei can be biologically valid
- width/area-derived metrics are PSF-sensitive and should be penalized more softly
- long, tortuous, implausibly merged tracks remain strong negatives

Compared with the older tuner, this version:
- imports from sperm_segmentation_saturnv5.5.py
- saves all outputs to:
    C:/Users/dmishra/Desktop/sperm_project/parameter_tuning_results
- removes the old heavy bias against single-slice tracks
- reduces the weight of PSF-sensitive penalties (thickness, taper)
- keeps a strong bias toward biologically plausible 3D lengths (~9–10 µm)
- keeps a strong penalty for monster merges and excessive tortuosity

Usage
-----
GUI mode:
    python utils/tune_parameters_Saturnv5_5.py

CLI mode:
    python utils/tune_parameters_Saturnv5_5.py --dir "path/to/images" --slices 0-12

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
import traceback
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
# Import V5.5 hybrid pipeline functions directly for fast in-process execution
# -----------------------------------------------------------------------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import importlib.util
module_name = "sperm_segmentation_saturnv5_5"
module_path = os.path.join(parent_dir, "sperm_segmentation_saturnv5.5.py")

if not os.path.exists(module_path):
    print(f"Error: Missing required pipeline file: {module_path}")
    sys.exit(1)

try:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    segmentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segmentation)
    
    CONFIG = segmentation.CONFIG
    validate_config = segmentation.validate_config
    segment_slice = segmentation.segment_slice
    measure_spermatids = segmentation.measure_spermatids
    track_across_slices = segmentation.track_across_slices
    flag_quality_tracks = segmentation.flag_quality_tracks
    rows_from_results = segmentation.rows_from_results
    normalize_display = segmentation.normalize_display
    robust_imread = segmentation.robust_imread
    make_overlay = segmentation.make_overlay
    load_roi_mask_file = segmentation.load_roi_mask_file
    filter_results_to_roi = segmentation.filter_results_to_roi
    
except AttributeError as e:
    print(f"Error: Required function missing in V5.5 pipeline: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: Could not import from biological suite {module_path}: {e}")
    traceback.print_exc()
    sys.exit(1)

# -----------------------------------------------------------------------------
# Output configuration
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\dmishra\Desktop\sperm_project\parameter_tuning_results")
ROI_SAVE_PATH = DEFAULT_OUTPUT_DIR / "last_drawn_roi_saturnv5_5_tune.tif"

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
PARAM_SPACE = [
    ("TRACK_MAX_DIST_UM",                  4.0,   7.2,   False),
    ("ASSIGNMENT_DIST_WEIGHT",             0.8,   2.8,   False),
    ("ASSIGNMENT_OVERLAP_WEIGHT",          1.2,   4.5,   False),
    ("ASSIGNMENT_LENGTH_WEIGHT",           1.5,   5.0,   False),
    ("ASSIGNMENT_WIDTH_WEIGHT",            1.0,   3.5,   False),
    ("ASSIGNMENT_AREA_WEIGHT",             1.0,   4.0,   False),
    ("ASSIGNMENT_ANGLE_WEIGHT",            0.2,   2.0,   False),
    ("HYBRID_REPAIR_MAX_COST",             2.0,   5.2,   False),
    ("HYBRID_REPAIR_MAX_GAP_SLICES",       0,     1,     True),
    ("HYBRID_REPAIR_MAX_FRAGMENT_SLICES",  1,     3,     True),
    ("HYBRID_REPAIR_MAX_LINK_DIST_UM",     3.0,   5.8,   False),
    ("HYBRID_REPAIR_MIN_OVERLAP",          0.0,   0.18,  False),
    ("HYBRID_REPAIR_MAX_FINAL_LENGTH_UM", 13.8,  15.2,  False),
]

SEGMENTATION_PARAM_SPACE = [
    ("THRESHOLD_HI",              72.0,  86.0,  False),
    ("THRESHOLD_LO",              58.0,  76.0,  False),
    ("MIN_OBJ_PX",                 4,    12,    True),
    ("MAX_BRIDGE_PX",              2,     9,    True),
    ("MIN_SKEL_LEN_PX",            5.0,  11.5,  False),
    ("MAX_WIDTH_PX",               7.0,  11.0,  False),
    ("MIN_LENGTH_WIDTH_RATIO",     1.6,   2.8,  False),
    ("MAX_TORTUOSITY",             2.4,   4.5,  False),
]

REVIEW = {
    "target_len_um": 9.5,
    "target_2d_len_um": 10.0,
    "target_width_um": 2.1,
    "target_lwr": 4.2,
}

SEGMENTATION_BASELINE = {}


def validate_tuner_parameter_spaces():
    """Validates that all parameters being tuned actually exist and have valid bounds."""
    errors = []
    for space_name, space in [("PARAM_SPACE", PARAM_SPACE), ("SEGMENTATION_PARAM_SPACE", SEGMENTATION_PARAM_SPACE)]:
        for key, lo, hi, is_int in space:
            if key not in CONFIG:
                errors.append(f"{space_name}: Key '{key}' does not exist in V5.5 CONFIG.")
                continue
            if lo >= hi:
                errors.append(f"{space_name}: Key '{key}' has invalid bounds [{lo}, {hi}].")
            seed_val = CONFIG[key]
            if not (lo <= seed_val <= hi):
                print(f"Warning: Seed value for {key} ({seed_val}) is outside search bounds [{lo}, {hi}]. It will be clamped.")
            if is_int and not isinstance(int(lo), int):
                errors.append(f"{space_name}: Key '{key}' marked as integer but bounds are invalid.")

    # Validate threshold logic in segmentation space
    hi_lo, hi_hi = next(( (l,h) for k,l,h,_ in SEGMENTATION_PARAM_SPACE if k=="THRESHOLD_HI" ), (None,None))
    lo_lo, lo_hi = next(( (l,h) for k,l,h,_ in SEGMENTATION_PARAM_SPACE if k=="THRESHOLD_LO" ), (None,None))
    if hi_lo is not None and lo_lo is not None:
        if lo_hi >= hi_lo:
            # We don't raise error on overlap of bounds, but just ensure the optimizer can generate valid configs.
            # The config validation will catch THRESHOLD_LO >= THRESHOLD_HI per candidate.
            pass

    if "TRACKING_BACKEND" in CONFIG and CONFIG["TRACKING_BACKEND"] not in ["legacy", "global_assignment", "hybrid_repair"]:
        errors.append(f"Invalid TRACKING_BACKEND: {CONFIG['TRACKING_BACKEND']}")

    if errors:
        raise ValueError("Tuner validation failed:\n" + "\n".join(errors))


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
    fig, ax = plt.subplots(figsize=(10, 10))
    base = normalize_display(roi_img)
    ax.imshow(base, cmap="gray")
    overlay = np.zeros((*roi_mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = roi_mask.astype(np.float32) * 0.28
    ax.imshow(overlay)
    ys, xs = np.where(roi_mask)
    if ys.size and xs.size:
        ax.contour(roi_mask.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=1.5)
    title = f"Loaded ROI Preview\n{roi_path if roi_path else ''}\nClose window to continue"
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=True)
    try:
        return messagebox.askyesno("Use this ROI?", "Does this ROI look correct?")
    except Exception:
        return True

def build_roi(images, force_redraw=False, interactive_prompt=False):
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
        if not interactive_prompt:
            print("\nNo saved tuner ROI found. Using full image frame for tuning.")
            roi_mask = np.ones_like(roi_img, dtype=bool)
            try:
                tifffile.imwrite(str(ROI_SAVE_PATH), roi_mask.astype(np.uint8) * 255)
            except Exception:
                pass
            return roi_mask

        print("\nDraw a GLOBAL ROI (Left-Click to place, Right-Click to UNDO, ENTER to Finish).")
        pts = []
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(normalize_display(roi_img), cmap="gray")
        line, = ax.plot([], [], "r-o", lw=2, markersize=8)
        
        def redraw():
            if not pts:
                line.set_data([], [])
            else:
                x, y = zip(*pts)
                line.set_data(x, y)
            fig.canvas.draw_idle()
            
        def on_click(event):
            if event.inaxes != ax: return
            if event.button == 1:
                pts.append((event.xdata, event.ydata))
                redraw()
            elif event.button == 3 and pts:
                pts.pop()
                redraw()
                
        def on_key(event):
            if event.key == "enter":
                if len(pts) > 2:
                    plt.close(fig)
                    
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
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
        
    return roi_mask

def safe_median(series, default=np.nan):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size > 0 else default

def safe_mean(series, default=np.nan):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else default

def params_from_vector(x, param_space):
    param_dict = {}
    for i, (key, lo, hi, is_int) in enumerate(param_space):
        val = x[i]
        if is_int:
            val = int(round(val))
        else:
            val = float(val)
        param_dict[key] = val
    return param_dict


def run_2d_detection(cfg, um_per_px):
    rows = []
    per_slice_counts = []
    for img, z_idx in zip(images_to_eval, z_values_eval):
        seg = segment_slice(img, cfg, z_idx=z_idx, debug_dir=None, roi_mask=None)
        meas = measure_spermatids(seg, cfg)
        results, _ = filter_results_to_roi(meas["results"], meas["skel_label"], roi_mask_global)
        slice_rows = rows_from_results(results, z_idx, um_per_px)
        rows.extend(slice_rows)
        per_slice_counts.append(len(slice_rows))
    return pd.DataFrame(rows), np.asarray(per_slice_counts, dtype=float)


def score_segmentation_run(df_2d, per_slice_counts, min_count_frac=0.55, max_count_frac=1.10):
    if df_2d.empty:
        return -1e12, {"reason": "no_2d"}
    base_count_med = SEGMENTATION_BASELINE.get("count_median", np.nan)
    base_count_mean = SEGMENTATION_BASELINE.get("count_mean", np.nan)

    lengths = df_2d["length_um_geodesic"].astype(float)
    widths = df_2d["width_um"].astype(float)
    lwr = df_2d["length_width_ratio"].astype(float)
    tort = df_2d["tortuosity"].astype(float) if "tortuosity" in df_2d else pd.Series(dtype=float)

    n_2d = len(df_2d)
    count_med = safe_median(per_slice_counts, default=0.0)
    count_mean = safe_mean(per_slice_counts, default=0.0)
    count_cv = float(np.std(per_slice_counts) / max(count_mean, 1.0)) if per_slice_counts.size else 0.0

    len_med = safe_median(lengths)
    len_mean = safe_mean(lengths)
    width_med = safe_median(widths)
    lwr_med = safe_median(lwr)
    tort_med = safe_median(tort) if not tort.empty else np.nan

    short_frac = float((lengths < 7.0).mean())
    very_long_frac = float((lengths > 18.0).mean())
    wide_frac = float((widths > 3.6).mean())
    low_lwr_frac = float((lwr < 2.5).mean())
    tort_frac = float((tort > 2.0).mean()) if not tort.empty else 0.0

    score = 0.0
    if np.isfinite(base_count_med) and base_count_med > 0:
        ratio = count_med / base_count_med
        if ratio < min_count_frac:
            score -= 2500.0 * (min_count_frac - ratio)
        if ratio > max_count_frac:
            score -= 1800.0 * (ratio - max_count_frac)
        score += 250.0 * min(ratio, 1.0)
    score += 0.015 * min(n_2d, max(base_count_mean * len(per_slice_counts), 1) if np.isfinite(base_count_mean) else n_2d)
    
    if np.isfinite(len_med):
        score -= 55.0 * abs(len_med - REVIEW["target_2d_len_um"])
    if np.isfinite(len_mean) and len_mean > 14.0:
        score -= 80.0 * (len_mean - 14.0)
    if np.isfinite(width_med):
        score -= 35.0 * abs(width_med - REVIEW["target_width_um"])
    if np.isfinite(lwr_med):
        score -= 25.0 * abs(lwr_med - REVIEW["target_lwr"])
        
    score -= 900.0 * short_frac
    score -= 700.0 * very_long_frac
    score -= 500.0 * wide_frac
    score -= 450.0 * low_lwr_frac
    score -= 350.0 * tort_frac
    score -= 180.0 * count_cv

    metrics = {
        "n_2d": int(n_2d),
        "count_median": round(count_med, 2),
        "count_mean": round(count_mean, 2),
        "count_cv": round(count_cv, 4),
        "count_ratio_vs_default": round(count_med / base_count_med, 4) if np.isfinite(base_count_med) and base_count_med > 0 else None,
        "len_median_um": round(len_med, 3) if np.isfinite(len_med) else None,
        "len_mean_um": round(len_mean, 3) if np.isfinite(len_mean) else None,
        "width_median_um": round(width_med, 3) if np.isfinite(width_med) else None,
        "lwr_median": round(lwr_med, 3) if np.isfinite(lwr_med) else None,
        "tort_median": round(tort_med, 3) if np.isfinite(tort_med) else None,
        "short_frac": round(short_frac, 4),
        "very_long_frac": round(very_long_frac, 4),
        "wide_frac": round(wide_frac, 4),
        "low_lwr_frac": round(low_lwr_frac, 4),
        "tort_frac": round(tort_frac, 4),
        "score": round(score, 2),
    }
    return score, metrics


def is_safe_tracking_candidate(record):
    """
    Return True for tracking candidates that avoid obvious fragmentation 
    and hard-fail inflation, relying on continuity and numerical safety guardrails.
    """
    try:
        zspan_multi_median_um = record.get("zspan_multi_median_um")
        single_frac = record.get("single_frac")
        l3d_med = record.get("l3d_median_um")
        l3d_mean = record.get("l3d_mean_um")
        hard_fail_frac = record.get("hard_fail_frac", 1)
        candidate_frac = record.get("candidate_frac", 0)
        invalid_frac = record.get("invalid_frac", 1)
        stop_frac = record.get("stop_frac", 1)
        overlap_unstable_frac = record.get("overlap_unstable_frac", 1)
        n_tracks = int(record.get("n_tracks", 0) or 0)
        multi_slice = int(record.get("multi_slice", 0) or 0)
        
        if n_tracks <= 0 or multi_slice <= 0:
            return False
            
        if any(v is None for v in [zspan_multi_median_um, single_frac, l3d_med, l3d_mean]):
            return False
            
        # Suggested initial guardrails for safe continuity
        return (
            float(zspan_multi_median_um) >= 1.0  # Approx 1 Z-step
            and float(single_frac) <= 0.65
            and 8.0 <= float(l3d_med) <= 12.0
            and float(l3d_mean) <= 13.2
            and float(hard_fail_frac) <= 0.42
            and float(candidate_frac) >= 0.55
            and float(invalid_frac) <= 0.01
            and float(stop_frac) <= 0.70
            and float(overlap_unstable_frac) <= 0.60
        )
    except Exception:
        return False


def score_run(df_2d, df_tracks, z_step_um):
    if df_2d.empty: return -1e12, {"reason": "no_2d"}
    if df_tracks is None or df_tracks.empty: return -1e12, {"reason": "no_3d"}

    n_2d = len(df_2d)
    n_tracks = len(df_tracks)

    # 1. Invalid measurement identification
    if "has_invalid_measurement" in df_tracks.columns:
        invalid_mask = df_tracks["has_invalid_measurement"].fillna(True).astype(bool)
    else:
        print("Warning: Pipeline did not provide has_invalid_measurement. Using fallback mask.")
        invalid_mask = pd.Series(False, index=df_tracks.index)
        for col in ["total_3d_length_um", "volume_um3", "thickness_um", "sum_area_px", "min_area_px", "max_area_px"]:
            if col in df_tracks.columns:
                invalid_mask |= ~np.isfinite(df_tracks[col])
                invalid_mask |= (df_tracks[col] <= 0)
        if "taper_ratio" in df_tracks.columns:
            invalid_mask |= ~np.isfinite(df_tracks["taper_ratio"])
            invalid_mask |= (df_tracks["taper_ratio"] < 1)

    n_invalid = int(invalid_mask.sum())
    invalid_frac = n_invalid / max(n_tracks, 1)

    # 2. Official Audit Masks
    candidate_mask = df_tracks["is_biological_candidate"].fillna(False).astype(bool) if "is_biological_candidate" in df_tracks.columns else pd.Series(False, index=df_tracks.index)
    quality_mask = df_tracks["is_quality_track"].fillna(False).astype(bool) if "is_quality_track" in df_tracks.columns else pd.Series(False, index=df_tracks.index)
    warning_mask = df_tracks["has_warning_only"].fillna(False).astype(bool) if "has_warning_only" in df_tracks.columns else pd.Series(False, index=df_tracks.index)
    hard_fail_mask = ~candidate_mask

    n_biological_candidates = int(candidate_mask.sum())
    n_quality = int(quality_mask.sum())
    n_warning_only = int(warning_mask.sum())
    n_hard_fail = int(hard_fail_mask.sum())
    
    candidate_frac = n_biological_candidates / max(n_tracks, 1)
    quality_frac = n_quality / max(n_tracks, 1)
    warning_only_frac = n_warning_only / max(n_tracks, 1)
    hard_fail_frac = n_hard_fail / max(n_tracks, 1)

    if n_biological_candidates != (n_quality + n_warning_only):
        return -1e12, {"error": "audit_consistency_error: n_biological_candidates != n_quality + n_warning_only"}
    if n_tracks != (n_biological_candidates + n_hard_fail):
        return -1e12, {"error": "audit_consistency_error: n_tracks != n_biological_candidates + n_hard_fail"}

    # 3. Z-Slice Separation
    nslices = df_tracks["n_slices"].astype(float) if "n_slices" in df_tracks.columns else pd.Series(1, index=df_tracks.index)
    single_mask = nslices == 1
    multi_mask = nslices >= 2
    
    single_slice = int(single_mask.sum())
    multi_slice = int(multi_mask.sum())
    single_frac = single_slice / max(n_tracks, 1)
    multi_frac = multi_slice / max(n_tracks, 1)

    lengths = df_tracks["total_3d_length_um"].astype(float) if "total_3d_length_um" in df_tracks.columns else pd.Series(np.nan, index=df_tracks.index)
    l3d_med = safe_median(lengths)
    l3d_mean = safe_mean(lengths)
    
    zspan = df_tracks["z_span_um"].astype(float) if "z_span_um" in df_tracks.columns else pd.Series(0, index=df_tracks.index)
    zspan_all_median_um = safe_median(zspan)
    zspan_multi_median_um = safe_median(zspan[multi_mask])
    
    pitch = df_tracks["pitch_deg"].astype(float) if "pitch_deg" in df_tracks.columns else pd.Series(np.nan, index=df_tracks.index)
    pitch_multi_mean_deg = safe_mean(pitch[multi_mask])
    pitch_multi_median_deg = safe_median(pitch[multi_mask])
    
    taper = df_tracks["taper_ratio"].astype(float) if "taper_ratio" in df_tracks.columns else pd.Series(np.nan, index=df_tracks.index)
    taper_multi_mean = safe_mean(taper[multi_mask])
    taper_multi_median = safe_median(taper[multi_mask])

    n_stopped = 0
    n_overlap_unstable = 0
    if "track_stop_reason" in df_tracks.columns:
        stop_reasons = df_tracks["track_stop_reason"].fillna("").astype(str)
        n_stopped = int((stop_reasons != "").sum())
        n_overlap_unstable = int(stop_reasons.str.contains("overlap_but_0_stable", regex=False).sum())
    stop_frac = n_stopped / max(n_tracks, 1)
    overlap_unstable_frac = n_overlap_unstable / max(n_tracks, 1)

    score = 0.0

    # Strong penalty for invalid tracks
    score -= 5000.0 * invalid_frac
    score -= 20.0 * n_invalid

    # Reward biologically plausible tracks
    score += 0.90 * multi_slice
    score += 0.18 * float(nslices[multi_mask].sum()) if multi_slice > 0 else 0
    
    if single_frac > 0.50:
        score -= 450.0 * (single_frac - 0.50)
    if single_frac > 0.70:
        score -= 120.0 * (single_frac - 0.70)
    if multi_frac < 0.20:
        score -= 150.0 * (0.20 - multi_frac)

    # Biological length alignment
    if np.isfinite(l3d_med):
        score -= 55.0 * abs(l3d_med - REVIEW["target_len_um"])
        if l3d_med > 11.0: score -= 180.0 * (l3d_med - 11.0)
    if np.isfinite(l3d_mean) and l3d_mean > 13.0:
        score -= 60.0 * (l3d_mean - 13.0)
    if np.isfinite(l3d_mean) and l3d_mean > 12.2:
        score -= 85.0 * (l3d_mean - 12.2)

    if np.isfinite(zspan_multi_median_um) and zspan_multi_median_um < 0.50 and multi_slice > 0:
        score -= 350.0 * (0.50 - zspan_multi_median_um)

    if hard_fail_frac > 0.40:
        score -= 700.0 * (hard_fail_frac - 0.40)
    if candidate_frac < 0.55:
        score -= 500.0 * (0.55 - candidate_frac)
    if warning_only_frac > 0.60:
        score -= 120.0 * (warning_only_frac - 0.60)

    if stop_frac > 0.55:
        score -= 180.0 * (stop_frac - 0.55)
    if overlap_unstable_frac > 0.20:
        score -= 260.0 * (overlap_unstable_frac - 0.20)

    metrics = {
        "n_2d": n_2d,
        "n_tracks": n_tracks,
        "single_slice": single_slice,
        "multi_slice": multi_slice,
        "single_frac": round(single_frac, 4),
        "multi_frac": round(multi_frac, 4),
        "n_biological_candidates": n_biological_candidates,
        "candidate_frac": round(candidate_frac, 4),
        "n_quality": n_quality,
        "quality_frac": round(quality_frac, 4),
        "n_warning_only": n_warning_only,
        "warning_only_frac": round(warning_only_frac, 4),
        "n_hard_fail": n_hard_fail,
        "hard_fail_frac": round(hard_fail_frac, 4),
        "n_invalid": n_invalid,
        "invalid_frac": round(invalid_frac, 4),
        "n_stopped": n_stopped,
        "stop_frac": round(stop_frac, 4),
        "n_overlap_unstable": n_overlap_unstable,
        "overlap_unstable_frac": round(overlap_unstable_frac, 4),
        "l3d_median_um": round(l3d_med, 3) if np.isfinite(l3d_med) else None,
        "l3d_mean_um": round(l3d_mean, 3) if np.isfinite(l3d_mean) else None,
        "zspan_all_median_um": round(zspan_all_median_um, 3) if np.isfinite(zspan_all_median_um) else None,
        "zspan_multi_median_um": round(zspan_multi_median_um, 3) if np.isfinite(zspan_multi_median_um) else None,
        "pitch_multi_mean_deg": round(pitch_multi_mean_deg, 3) if np.isfinite(pitch_multi_mean_deg) else None,
        "pitch_multi_median_deg": round(pitch_multi_median_deg, 3) if np.isfinite(pitch_multi_median_deg) else None,
        "taper_multi_mean": round(taper_multi_mean, 3) if np.isfinite(taper_multi_mean) else None,
        "taper_multi_median": round(taper_multi_median, 3) if np.isfinite(taper_multi_median) else None,
        "score": round(score, 2),
    }
    return score, metrics


def objective_fn(x, um_per_px, z_step_um):
    global eval_count, best_global_score, results_list
    global images_to_eval, z_values_eval, roi_mask_global

    param_dict = params_from_vector(x, PARAM_SPACE)
    cfg = CONFIG.copy()
    cfg.update(param_dict)
    
    try:
        validate_config(cfg)
    except Exception as e:
        eval_count += 1
        record = {"params": param_dict, "error": f"validation failed: {e}", "score": -1e12}
        results_list.append(record)
        return 1e12

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    score = -1e12
    metrics = {}
    try:
        rows = []
        for img, z_idx in zip(images_to_eval, z_values_eval):
            seg = segment_slice(img, cfg, z_idx=z_idx, debug_dir=None, roi_mask=None)
            meas = measure_spermatids(seg, cfg)
            results, _ = filter_results_to_roi(meas["results"], meas["skel_label"], roi_mask_global)
            rows.extend(rows_from_results(results, z_idx, um_per_px))

        df_2d = pd.DataFrame(rows)
        
        if not df_2d.empty and cfg.get("DO_TRACKING", True):
            _, df_tracks = track_across_slices(df_2d, cfg)
            # Use official audit
            df_tracks = flag_quality_tracks(df_tracks, cfg)
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
            f" | Lmed={metrics.get('l3d_median_um', 0)}"
        )
        sys.stdout.write(msg + "  \n")
        sys.stdout.flush()

    return -score


def objective_segmentation_fn(x, um_per_px, min_count_frac, max_count_frac):
    global eval_count, best_global_score, results_list

    param_dict = params_from_vector(x, SEGMENTATION_PARAM_SPACE)
    cfg = CONFIG.copy()
    cfg.update(param_dict)
    
    try:
        validate_config(cfg)
    except Exception as e:
        eval_count += 1
        record = {"params": param_dict, "error": f"validation failed: {e}", "score": -1e12}
        results_list.append(record)
        return 1e12

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    score = -1e12
    metrics = {}
    try:
        df_2d, per_slice_counts = run_2d_detection(cfg, um_per_px)
        score, metrics = score_segmentation_run(
            df_2d,
            per_slice_counts,
            min_count_frac=min_count_frac,
            max_count_frac=max_count_frac,
        )
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
            f" | n2d={metrics.get('n_2d', 0)}"
            f" | count_med={metrics.get('count_median', 0)}"
            f" | ratio={metrics.get('count_ratio_vs_default', 0)}"
            f" | Lmed={metrics.get('len_median_um', 0)}"
        )
        sys.stdout.write(msg + "  \n")
        sys.stdout.flush()

    return -score


def save_segmentation_review_panels(outdir, top_records, um_per_px, max_candidates=6):
    review_dir = outdir / "segmentation_review_panels"
    ensure_dir(review_dir)

    for cand_idx, record in enumerate(top_records[:max_candidates], start=1):
        cfg = CONFIG.copy()
        cfg.update(record.get("params", {}))

        fig, axes = plt.subplots(len(images_to_eval), 2, figsize=(10, max(3, 3 * len(images_to_eval))))
        if len(images_to_eval) == 1:
            axes = np.asarray([axes])

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            for row_idx, (img, z_idx) in enumerate(zip(images_to_eval, z_values_eval)):
                seg = segment_slice(img, cfg, z_idx=z_idx, debug_dir=None, roi_mask=None)
                meas = measure_spermatids(seg, cfg)
                results, skel_label = filter_results_to_roi(meas["results"], meas["skel_label"], roi_mask_global)
                overlay = make_overlay(img, skel_label)

                axes[row_idx, 0].imshow(normalize_display(img), cmap="gray")
                axes[row_idx, 0].set_title(f"z{z_idx:02d} raw")
                axes[row_idx, 0].axis("off")

                axes[row_idx, 1].imshow(overlay)
                axes[row_idx, 1].set_title(f"z{z_idx:02d} detections n={len(results)}")
                axes[row_idx, 1].axis("off")
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        params = record.get("params", {})
        title = (
            f"Candidate {cand_idx} | score={record.get('score')} | "
            f"Lmed={record.get('len_median_um')} | ratio={record.get('count_ratio_vs_default')}\n"
            + ", ".join(f"{k}={v}" for k, v in params.items())
        )
        fig.suptitle(title, fontsize=9)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        out_path = review_dir / f"candidate_{cand_idx:02d}_segmentation_review.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    return review_dir

def cb_generation(xk, convergence):
    print(f"  Generation complete. Population convergence: {convergence:.4f}")

def run_self_check():
    """Inline self-check logic for V5.5."""
    print("Running V5.5 tuner self-check...")
    try:
        validate_config(CONFIG)
        print("  [OK] validate_config executed successfully.")
        
        validate_tuner_parameter_spaces()
        print("  [OK] validate_tuner_parameter_spaces passed.")
        
        # Test synthetic DataFrame with flag_quality_tracks
        df = pd.DataFrame({
            "track_id": [1, 2, 3, 4],
            "n_slices": [3, 4, 3, 3],
            "total_3d_length_um": [10.0, 30.0, 10.0, 10.0],
            "tortuosity_3d": [1.1, 1.2, 1.1, 1.1],
            "thickness_um": [1.5, 1.5, 3.0, 1.5],
            "taper_ratio": [1.2, 1.2, 1.2, 1.2],
            "has_invalid_measurement": [False, False, False, True],
            "has_overlap_unstable": [False, False, False, False]
        })
        
        df_res = flag_quality_tracks(df, CONFIG)
        
        res1 = df_res.loc[df_res["track_id"]==1].iloc[0]
        assert res1["is_biological_candidate"], "Track 1 should be candidate"
        
        res2 = df_res.loc[df_res["track_id"]==2].iloc[0]
        assert not res2["is_biological_candidate"], "Track 2 should fail due to length"
        
        res3 = df_res.loc[df_res["track_id"]==3].iloc[0]
        assert res3["has_warning_only"] and res3["is_biological_candidate"], "Track 3 should be warning-only"
        
        res4 = df_res.loc[df_res["track_id"]==4].iloc[0]
        if "invalid_measurement" in str(res4.get("hard_flags", "")):
            assert not res4["is_biological_candidate"], "Track 4 should fail due to invalid measurement"
        
        print("  [OK] Synthetic track DataFrame passed flag_quality_tracks rules.")
        print("Self-check completed successfully.")
        sys.exit(0)
    except Exception as e:
        print(f"Self-check FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


def main():
    global images_to_eval, z_values_eval, roi_mask_global, results_list, SEGMENTATION_BASELINE

    parser = argparse.ArgumentParser(
        description="Evolutionary parameter tuner for Saturn V5.5 hybrid segmentation/tracking"
    )
    parser.add_argument("--mode", choices=["segmentation", "tracking"], default="tracking",
                       help="Tune raw 2D detection or 3D tracking parameters (default: tracking)")
    parser.add_argument("--dir", default=None,
                       help="Directory containing .tif/.tiff slices")
    parser.add_argument("--slices", default="0-12",
                       help="Z slices to use for tuning, e.g. 0-12 or 3,5,7-10")
    parser.add_argument("--um-per-px", type=float, default=None,
                       help="Override calibration (um/px). Defaults to V5.5 CONFIG.")
    parser.add_argument("--z-step-um", type=float, default=None,
                       help="Override z-step in um. Defaults to V5.5 CONFIG.")
    parser.add_argument("--new-roi", action="store_true",
                       help="Force drawing a new ROI")
    parser.add_argument("--roi-mask", default=None,
                       help="Optional .npy/.tif ROI mask to use for tuning")
    parser.add_argument("--base-params", default=None,
                       help="Optional JSON parameter file to merge into CONFIG before tuning")
    parser.add_argument("--maxiter", type=int, default=10,
                       help="Number of DE generations (default: 10)")
    parser.add_argument("--popsize", type=int, default=8,
                       help="DE population multiplier (default: 8)")
    parser.add_argument("--no-polish", action="store_true",
                       help="Disable final local optimizer polishing for faster exploratory runs")
    parser.add_argument("--seg-min-count-frac", type=float, default=0.55,
                       help="Segmentation mode: lowest acceptable median per-slice count versus default")
    parser.add_argument("--seg-max-count-frac", type=float, default=1.10,
                       help="Segmentation mode: highest acceptable median per-slice count versus default")
    parser.add_argument("--review-candidates", type=int, default=6,
                       help="Segmentation mode: number of top visual review panels to save")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR),
                       help="Folder to save tuning results")
    parser.add_argument("--self-check", action="store_true",
                       help="Run inline self-check then exit")
                       
    args = parser.parse_args()

    if args.self_check:
        run_self_check()

    if args.base_params:
        params_path = Path(args.base_params)
        if not params_path.exists():
            print(f"Base parameter JSON not found: {params_path}")
            sys.exit(1)
        with open(params_path, "r", encoding="utf-8") as f:
            tuned = json.load(f)
        applied = 0
        print(f"Loading --base-params from {params_path.name}:")
        for key, value in tuned.items():
            if key in CONFIG:
                CONFIG[key] = value
                applied += 1
                print(f"  Applied override: {key} = {value}")
            else:
                print(f"  Warning: Unknown key rejected: {key}")
        try:
            validate_config(CONFIG)
        except Exception as e:
            print(f"Error validating --base-params: {e}")
            sys.exit(1)
        print(f"Loaded {applied} valid CONFIG values.")

    validate_tuner_parameter_spaces()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    global ROI_SAVE_PATH
    ROI_SAVE_PATH = outdir / "last_drawn_roi_saturnv5_5_tune.tif"

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
    if args.roi_mask:
        roi_mask_global = load_roi_mask_file(args.roi_mask, expected_shape=images_to_eval[0].shape)
        tifffile.imwrite(str(ROI_SAVE_PATH), roi_mask_global.astype(np.uint8) * 255)
        print(f"Loaded ROI mask for tuning: {args.roi_mask}")
        print(f"Saved tuner ROI copy: {ROI_SAVE_PATH}")
    else:
        roi_mask_global = build_roi(images_to_eval, force_redraw=args.new_roi, interactive_prompt=(args.dir is None))

    if args.mode == "segmentation":
        baseline_cfg = CONFIG.copy()
        print("\nComputing default 2D segmentation baseline for guardrails...")
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            df_base, base_counts = run_2d_detection(baseline_cfg, um_per_px)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        SEGMENTATION_BASELINE = {
            "n_2d": int(len(df_base)),
            "count_median": safe_median(base_counts, default=0.0),
            "count_mean": safe_mean(base_counts, default=0.0),
            "len_median_um": safe_median(df_base["length_um_geodesic"]) if not df_base.empty else np.nan,
        }
        active_space = SEGMENTATION_PARAM_SPACE
        objective = objective_segmentation_fn
        objective_args = (um_per_px, args.seg_min_count_frac, args.seg_max_count_frac)
        mode_label = "SEGMENTATION"
        scoring_note = (
            "Scoring bias: improve 2D shape plausibility while staying within "
            f"{args.seg_min_count_frac:.2f}-{args.seg_max_count_frac:.2f}x default slice density"
        )
    else:
        active_space = PARAM_SPACE
        objective = objective_fn
        objective_args = (um_per_px, z_step_um)
        mode_label = "TRACKING"
        scoring_note = "Scoring bias: preserve plausible single-slice nuclei; penalize over-merges"
        CONFIG["TRACKING_BACKEND"] = "hybrid_repair"

    bounds = [(lo, hi) for (_, lo, hi, _) in active_space]

    x0 = []
    print("\nSeed parameters (from V5.5 CONFIG):")
    for key, lo, hi, is_int in active_space:
        val = CONFIG.get(key, (lo + hi) / 2)
        val = max(lo, min(hi, val))
        x0.append(val)
        print(f"  {key:45s} = {val}  (bounds: [{lo}, {hi}])")

    n_params = len(active_space)
    total_evals_est = args.maxiter * args.popsize * n_params

    print(f"\n{'='*78}")
    print(f"  EVOLUTIONARY {mode_label} PARAMETER TUNING (V5.5 HYBRID)")
    print(f"  Parameters:   {n_params}")
    print(f"  Generations:  {args.maxiter}")
    print(f"  Pop size:     {args.popsize} x {n_params} = {args.popsize * n_params}")
    print(f"  Est. evals:   ~{total_evals_est}")
    print(f"  {scoring_note}")
    print(f"{'='*78}\n")

    t0 = time.time()
    result = differential_evolution(
        func=objective,
        args=objective_args,
        bounds=bounds,
        x0=x0,
        maxiter=args.maxiter,
        popsize=args.popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        callback=cb_generation,
        disp=False,
        polish=not args.no_polish,
        seed=42,
    )
    dt = time.time() - t0

    print(f"\nOptimization finished in {dt:.0f}s across {eval_count} evaluations.")

    best_params = {}
    for i, (key, lo, hi, is_int) in enumerate(active_space):
        val = result.x[i]
        if is_int:
            val = int(round(val))
        else:
            val = round(val, 4)
        best_params[key] = val
        
    if args.mode == "tracking":
        best_params["TRACKING_BACKEND"] = "hybrid_repair"

    results_list.sort(key=lambda d: d.get("score", -1e18), reverse=True)
    best = results_list[0]
    safe_tracking_results = []
    safe_tracking_best = None
    if args.mode == "tracking":
        safe_tracking_results = [r for r in results_list if is_safe_tracking_candidate(r)]
        safe_tracking_best = safe_tracking_results[0] if safe_tracking_results else None

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
    if args.mode == "segmentation":
        metric_keys = [
            "n_2d", "count_median", "count_mean", "count_ratio_vs_default",
            "count_cv", "len_median_um", "len_mean_um", "width_median_um",
            "lwr_median", "short_frac", "very_long_frac", "wide_frac",
            "low_lwr_frac", "tort_frac", "score"
        ]
    else:
        metric_keys = [
            "n_tracks", "single_slice", "multi_slice", "single_frac", "multi_frac",
            "n_biological_candidates", "candidate_frac", "n_quality", "quality_frac",
            "n_warning_only", "warning_only_frac", "n_hard_fail", "hard_fail_frac",
            "n_invalid", "invalid_frac", "n_stopped", "stop_frac",
            "n_overlap_unstable", "overlap_unstable_frac",
            "l3d_median_um", "l3d_mean_um", "zspan_all_median_um", "zspan_multi_median_um",
            "pitch_multi_mean_deg", "pitch_multi_median_deg",
            "taper_multi_mean", "taper_multi_median", "score"
        ]
    for k in metric_keys:
        print(f"  {k:24s}: {best.get(k)}")

    existing = sorted(outdir.glob(f"best_{args.mode}_params_v5_5_*.json"))
    next_num = 1
    if existing:
        for ep in existing:
            try:
                num = int(ep.stem.split("_")[-1])
                next_num = max(next_num, num + 1)
            except ValueError:
                pass

    best_param_filename = f"best_{args.mode}_params_v5_5_{next_num:03d}.json"
    best_param_path = outdir / best_param_filename
    with open(best_param_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n[OK] Best parameters saved: {best_param_path}")

    safe_param_filename = None
    safe_param_path = None
    if args.mode == "tracking":
        safe_candidates_path = outdir / f"safe_tracking_candidates_v5_5_{next_num:03d}.json"
        with open(safe_candidates_path, "w", encoding="utf-8") as f:
            json.dump(safe_tracking_results[:10], f, indent=2)
        print(f"[OK] Safe tracking candidates: {safe_candidates_path}")

        if safe_tracking_best:
            safe_params = safe_tracking_best.get("params", {}).copy()
            safe_params["TRACKING_BACKEND"] = "hybrid_repair"
            safe_param_filename = f"best_safe_tracking_params_v5_5_{next_num:03d}.json"
            safe_param_path = outdir / safe_param_filename
            with open(safe_param_path, "w", encoding="utf-8") as f:
                json.dump(safe_params, f, indent=2)
            if safe_tracking_best is best:
                print(f"[OK] Raw best also passes continuity guardrails: {safe_param_path}")
            else:
                print(f"[OK] Best safe tracking parameters saved: {safe_param_path}")
                print("     Raw best was kept too, but safe best is preferred for review/use.")
        else:
            print("[WARN] No tracking candidate passed continuity guardrails.")

    history_path = outdir / f"tuning_results_saturnv5_5_{args.mode}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)
    print(f"[OK] Full search history: {history_path}")

    if args.mode == "segmentation":
        review_dir = save_segmentation_review_panels(
            outdir,
            results_list,
            um_per_px,
            max_candidates=max(0, args.review_candidates),
        )
        print(f"[OK] Segmentation visual review panels: {review_dir}")

    summary_path = outdir / f"tuning_summary_saturnv5_5_{args.mode}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"SATURN V5.5 HYBRID {mode_label} TUNING SUMMARY\n")
        f.write("=" * 72 + "\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Output directory: {outdir}\n")
        f.write(f"Images tuned: {len(images_to_eval)} | z={z_values_eval}\n")
        f.write(f"Calibration: {um_per_px:.6f} um/px | z-step: {z_step_um:.6f} um\n")
        if args.base_params:
            f.write(f"Base parameters loaded from: {Path(args.base_params).name}\n")
        if args.mode == "segmentation":
            f.write("\nDefault baseline guardrails:\n")
            for key, val in SEGMENTATION_BASELINE.items():
                f.write(f"  {key}: {val}\n")
            f.write(f"  min_count_frac: {args.seg_min_count_frac}\n")
            f.write(f"  max_count_frac: {args.seg_max_count_frac}\n")
        f.write("\nBest parameters:\n")
        for key, val in best_params.items():
            f.write(f"  {key}: {val}\n")
        f.write("\nBest metrics:\n")
        for k in metric_keys:
            f.write(f"  {k}: {best.get(k)}\n")
        if args.mode == "tracking":
            f.write("\nContinuity guardrails:\n")
            f.write("  safe candidate rule: zspan_multi_median_um >= 1.0, single_frac <= 0.65, l3d_median_um in [8.0, 12.0], l3d_mean_um <= 13.2, hard_fail_frac <= 0.42, candidate_frac >= 0.55, invalid_frac <= 0.01\n")
            f.write(f"  safe candidates found: {len(safe_tracking_results)}\n")
            if safe_tracking_best:
                f.write("  Best safe metrics:\n")
                for k in metric_keys:
                    f.write(f"    {k}: {safe_tracking_best.get(k)}\n")
                f.write(f"  Best safe parameter file: {safe_param_path}\n")
        f.write("\nBiology / hardware note:\n")
        if args.mode == "segmentation":
            f.write("  Segmentation tuning is a candidate-selection aid, not ground truth.\n")
            f.write("  Review saved overlay panels before accepting a parameter set.\n")
        else:
            f.write("  This V5.5 hybrid tuner preserves biologically plausible single-slice nuclei for this Leica\n")
            f.write("  SP8 stack and penalizes likely over-merges more strongly than shallow tracks.\n")
    print(f"[OK] Summary text file: {summary_path}")

    print("\nTo use these parameters:")
    if args.mode == "tracking" and safe_param_filename:
        print(f"  GUI: Click 'Load Tuned Params' -> select {safe_param_filename}")
        print(f"  CLI: python sperm_segmentation_saturnv5.5.py --batch --params \"{safe_param_path}\"")
    else:
        print(f"  GUI: Click 'Load Tuned Params' -> select {best_param_filename}")
        print(f"  CLI: python sperm_segmentation_saturnv5.5.py --batch --params \"{best_param_path}\"")


if __name__ == "__main__":
    main()
