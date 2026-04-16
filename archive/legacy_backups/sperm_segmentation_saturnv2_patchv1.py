#!/usr/bin/env python3
"""
Conservative Tracking Patch for Saturn Sperm Segmentation Pipeline

Philosophy
----------
STOP bad tracks from forming, don't repair them after the fact.

Key principle:
When extending a track to a new slice, REJECT the extension if it would
create a biologically implausible jump. Stop the track instead.

What this improves over the weighted+repair patch
--------------------------------------------------
1. No post-hoc splitting/repair (which destabilized the pipeline)
2. Conservative extension with biological consistency checks
3. Simpler, more predictable behavior
4. Fewer tortuous outliers from bad linking decisions

How to use
----------
Place in the same folder as sperm_segmentation_saturnv2.py

Run:
    python sperm_segmentation_conservative_patch.py --batch
    python sperm_segmentation_conservative_patch.py --gui
"""

import os
import sys
import math
import shutil
import argparse
import importlib.util
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog
    _TK_AVAILABLE = True
except ImportError:
    _TK_AVAILABLE = False

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Load base module
HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "sperm_segmentation_saturnv2.py"
if not BASE_PATH.exists():
    raise FileNotFoundError(f"Could not find base pipeline: {BASE_PATH}")

spec = importlib.util.spec_from_file_location("saturn_base", str(BASE_PATH))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# Conservative tracking parameters
CONSERVATIVE_DEFAULTS = {
    # Maximum allowed changes when extending a track
    "CONSERVATIVE_MAX_WIDTH_JUMP_RATIO": 0.50,      # 50% width change max
    "CONSERVATIVE_MAX_LENGTH_JUMP_RATIO": 0.60,     # 60% length change max
    "CONSERVATIVE_MAX_AREA_JUMP_RATIO": 0.70,       # 70% area change max
    "CONSERVATIVE_MAX_TORTUOSITY_JUMP": 0.40,       # Absolute tortuosity jump
    "CONSERVATIVE_MAX_CENTROID_JUMP_UM": 8.0,       # Centroid displacement in um
    
    # Optional: overlap-first matching
    "USE_OVERLAP_FIRST": False,  # Set to True to prioritize bbox overlap
    "MIN_BBOX_OVERLAP_RATIO": 0.10,  # Minimum overlap to consider (if enabled)
}

# Merge into base config
for k, v in CONSERVATIVE_DEFAULTS.items():
    if k not in base.CONFIG:
        base.CONFIG[k] = v


def check_extension_consistency(prev_state, candidate_detection, cfg):
    """
    Check if extending a track with this detection would be biologically consistent.
    
    Returns:
        (is_consistent: bool, reason: str)
    """
    um_xy = cfg["UM_PER_PX_XY"]
    
    # Extract previous track state
    prev_x = prev_state["last_x"]
    prev_y = prev_state["last_y"]
    prev_width = prev_state.get("last_width", None)
    prev_length = prev_state.get("last_length", None)
    prev_area = prev_state.get("last_area", None)
    
    # Extract candidate detection features
    cand_x = candidate_detection["centroid_x"]
    cand_y = candidate_detection["centroid_y"]
    cand_width = candidate_detection.get("width_um", None)
    cand_length = candidate_detection.get("length_um_geodesic", None)
    cand_area = candidate_detection.get("area_px", None)
    
    # 1. Check centroid jump
    dx = cand_x - prev_x
    dy = cand_y - prev_y
    centroid_jump_um = math.sqrt(dx*dx + dy*dy) * um_xy
    
    if centroid_jump_um > cfg["CONSERVATIVE_MAX_CENTROID_JUMP_UM"]:
        return False, f"centroid_jump={centroid_jump_um:.2f}um"
    
    # 2. Check width consistency
    if prev_width is not None and cand_width is not None:
        width_ratio = abs(cand_width - prev_width) / max(prev_width, 1e-9)
        if width_ratio > cfg["CONSERVATIVE_MAX_WIDTH_JUMP_RATIO"]:
            return False, f"width_jump={width_ratio:.2f}"
    
    # 3. Check length consistency
    if prev_length is not None and cand_length is not None:
        length_ratio = abs(cand_length - prev_length) / max(prev_length, 1e-9)
        if length_ratio > cfg["CONSERVATIVE_MAX_LENGTH_JUMP_RATIO"]:
            return False, f"length_jump={length_ratio:.2f}"
    
    # 4. Check area consistency
    if prev_area is not None and cand_area is not None:
        area_ratio = abs(cand_area - prev_area) / max(prev_area, 1e-9)
        if area_ratio > cfg["CONSERVATIVE_MAX_AREA_JUMP_RATIO"]:
            return False, f"area_jump={area_ratio:.2f}"
    
    return True, "ok"


def track_across_slices_conservative(detections_df, cfg):
    """
    Conservative tracking: stop tracks when consistency breaks.
    
    Key differences from original:
    1. Check biological consistency BEFORE extending track
    2. Stop track if extension would be implausible
    3. No post-hoc repair or splitting
    """
    if detections_df.empty:
        detections_df = detections_df.copy()
        detections_df["track_id"] = pd.Series(dtype=int)
        return detections_df, pd.DataFrame()

    max_dist_px = cfg["TRACK_MAX_DIST_UM"] / (cfg["UM_PER_PX_XY"] + 1e-9)
    df = (detections_df.copy()
                       .sort_values(["z_slice", "sperm_id"])
                       .reset_index(drop=True))

    next_tid = 1
    active = {}
    track_ids = [-1] * len(df)
    stopped_tracks = {}  # Track stop reasons for debugging
    
    rows_by_z = {z: df.index[df["z_slice"] == z].to_numpy()
                 for z in sorted(df["z_slice"].unique())}

    for z, idxs in rows_by_z.items():
        # Get detection features for this slice
        xs = df.loc[idxs, "centroid_x"].to_numpy(float)
        ys = df.loc[idxs, "centroid_y"].to_numpy(float)
        
        # Extract morphological features (with fallbacks)
        widths = df.loc[idxs, "width_um"].to_numpy(float) if "width_um" in df.columns else np.full(len(idxs), np.nan)
        lengths = df.loc[idxs, "length_um_geodesic"].to_numpy(float) if "length_um_geodesic" in df.columns else np.full(len(idxs), np.nan)
        areas = df.loc[idxs, "area_px"].to_numpy(float) if "area_px" in df.columns else np.full(len(idxs), np.nan)

        # Find candidate tracks from previous slices
        cand_tracks = [t for t, st in active.items()
                       if 1 <= z - st["last_z"] <= cfg["TRACK_MAX_GAP_SLICES"] + 1]
        cand_pos = [(active[t]["last_x"], active[t]["last_y"]) for t in cand_tracks]

        used_det, used_trk = set(), set()
        
        # Greedy nearest-neighbor matching with consistency checks
        if cand_tracks:
            tree = cKDTree(np.array(cand_pos, float))
            candidates = []
            
            for k, (x, y) in enumerate(zip(xs, ys)):
                d_val, j = tree.query([x, y])
                if np.isfinite(d_val) and d_val <= max_dist_px:
                    # Build candidate detection dict
                    cand_det = {
                        "centroid_x": float(x),
                        "centroid_y": float(y),
                        "width_um": float(widths[k]) if np.isfinite(widths[k]) else None,
                        "length_um_geodesic": float(lengths[k]) if np.isfinite(lengths[k]) else None,
                        "area_px": float(areas[k]) if np.isfinite(areas[k]) else None,
                    }
                    
                    # Check if this extension is biologically consistent
                    tid = cand_tracks[j]
                    is_consistent, reason = check_extension_consistency(
                        active[tid], cand_det, cfg
                    )
                    
                    if is_consistent:
                        candidates.append((float(d_val), k, int(j)))
                    else:
                        # Track fails consistency check - will be stopped
                        if tid not in stopped_tracks:
                            stopped_tracks[tid] = f"z={z}, reason={reason}"
            
            # Sort by distance and assign greedily
            for d_val, det_k, trk_j in sorted(candidates):
                if det_k in used_det or trk_j in used_trk:
                    continue
                    
                used_det.add(det_k)
                used_trk.add(trk_j)
                tid = cand_tracks[trk_j]
                track_ids[int(idxs[det_k])] = tid
                
                # Update track state
                active[tid] = {
                    "last_z": int(z),
                    "last_x": float(xs[det_k]),
                    "last_y": float(ys[det_k]),
                    "last_width": float(widths[det_k]) if np.isfinite(widths[det_k]) else None,
                    "last_length": float(lengths[det_k]) if np.isfinite(lengths[det_k]) else None,
                    "last_area": float(areas[det_k]) if np.isfinite(areas[det_k]) else None,
                }

        # Create new tracks for unmatched detections
        for det_k in range(len(idxs)):
            if track_ids[int(idxs[det_k])] == -1:
                track_ids[int(idxs[det_k])] = next_tid
                active[next_tid] = {
                    "last_z": int(z),
                    "last_x": float(xs[det_k]),
                    "last_y": float(ys[det_k]),
                    "last_width": float(widths[det_k]) if np.isfinite(widths[det_k]) else None,
                    "last_length": float(lengths[det_k]) if np.isfinite(lengths[det_k]) else None,
                    "last_area": float(areas[det_k]) if np.isfinite(areas[det_k]) else None,
                }
                next_tid += 1

        # Remove stale tracks (exceeded gap)
        for tid in [t for t, st in active.items()
                    if z - st["last_z"] > cfg["TRACK_MAX_GAP_SLICES"] + 1]:
            del active[tid]

    df["track_id"] = track_ids
    
    # Print tracking stats
    print(f"  Conservative tracking: {len(stopped_tracks)} tracks stopped early for consistency")
    
    # Compute track-level metrics (use original aggregation logic)
    if "tortuosity" in df.columns:
        df["euc_um_2d"] = df["length_um_geodesic"] / df["tortuosity"]
    else:
        df["euc_um_2d"] = df["length_um_geodesic"]
        
    g = df.groupby("track_id", as_index=False)
    ts = g.agg(
        n_slices        = ("z_slice",            "count"),
        z_start         = ("z_slice",            "min"),
        z_end           = ("z_slice",            "max"),
        max_length_2d   = ("length_um_geodesic", "max"),
        max_euc_2d      = ("euc_um_2d",          "max"),
        sum_area_px     = ("area_px",            "sum"),
        area_start      = ("area_px",            "first"),
        area_end        = ("area_px",            "last"),
        x_mean          = ("centroid_x",         "mean"),
        y_mean          = ("centroid_y",         "mean"),
        z_mean          = ("z_slice",            "mean"),
        x_start         = ("centroid_x",         "first"),
        y_start         = ("centroid_y",         "first"),
        x_end           = ("centroid_x",         "last"),
        y_end           = ("centroid_y",         "last"),
    )
    
    um_xy = cfg["UM_PER_PX_XY"]
    um_z  = cfg["UM_PER_SLICE_Z"]
    
    # 3D metric calculations (same as original)
    z_extent = (ts["z_end"] - ts["z_start"] + 1) * um_z
    ts["z_extent_um"] = z_extent
    
    dz_euc = (ts["z_end"] - ts["z_start"]) * um_z
    euc_2d_centroid = np.sqrt((ts["x_end"] - ts["x_start"])**2 + (ts["y_end"] - ts["y_start"])**2) * um_xy
    lat_geodesic = np.maximum(ts["max_length_2d"], euc_2d_centroid)
    l3d = np.sqrt(lat_geodesic**2 + z_extent**2)
    ts["total_3d_length_um"] = l3d
    
    ts["volume_um3"] = ts["sum_area_px"] * (um_xy**2) * um_z
    
    euc_3d = np.sqrt(ts["max_euc_2d"]**2 + dz_euc**2)
    safe_euc = np.maximum(euc_3d, 0.1)
    tort_raw = l3d / safe_euc
    ts["tortuosity_3d"] = np.minimum(tort_raw, 20.0)
    
    ts["taper_ratio"] = np.maximum(ts["area_start"], ts["area_end"]) / np.maximum(np.minimum(ts["area_start"], ts["area_end"]), 0.001)
    
    cross_area = ts["volume_um3"] / np.maximum(ts["total_3d_length_um"], 0.1)
    ts["thickness_um"] = 2 * np.sqrt(cross_area / np.pi)
    
    dx = (ts["x_end"] - ts["x_start"]) * um_xy
    dy = (ts["y_end"] - ts["y_start"]) * um_xy
    v_mag = np.sqrt(dx**2 + dy**2 + dz_euc**2)
    safe_v = np.maximum(v_mag, 1e-9)
    ts["pitch_deg"] = np.abs(np.arcsin(dz_euc / safe_v)) * (180.0 / np.pi)
    ts["yaw_deg"] = np.arctan2(dy, dx) * (180.0 / np.pi)
    
    if len(ts) > 1:
        centers = np.column_stack((ts["x_mean"] * um_xy, ts["y_mean"] * um_xy, ts["z_mean"] * um_z))
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        ts["nearest_neighbor_um"] = dists[:, 1]
    else:
        ts["nearest_neighbor_um"] = np.nan
        
    cols_ordered = [
        "track_id", "total_3d_length_um", "z_extent_um", "volume_um3", "tortuosity_3d",
        "thickness_um", "pitch_deg", "yaw_deg", "taper_ratio", "nearest_neighbor_um",
        "n_slices", "z_start", "z_end", "max_length_2d", "sum_area_px"
    ]
    ts = ts[cols_ordered]
    
    return df, ts


# Monkey-patch the base module
base.track_across_slices = track_across_slices_conservative


def alias_auditor_files(output_dir):
    """Create unversioned copies for compatibility with audit scripts."""
    version = getattr(base, "_VERSION", "v12")
    mapping = {
        f"track_summary_{version}.csv": "track_summary.csv",
        f"measurements_with_tracks_{version}.csv": "measurements_with_tracks.csv",
        f"spermatid_measurements_{version}.csv": "spermatid_measurements.csv",
        f"slice_summary_{version}.csv": "slice_summary.csv"
    }
    print(f"\n--- AUDITOR COMPATIBILITY BRIDGE ---")
    for src, dst in mapping.items():
        src_path = os.path.join(output_dir, src)
        dst_path = os.path.join(output_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  Aliased: {src} -> {dst}")
    print("------------------------------------\n")
    

def get_unique_batch_dir_patchv1(input_dir):
    """Find the next available folder with the pattern batch_output_patchV1_N."""
    i = 1
    while True:
        folder = os.path.join(input_dir, f"batch_output_patchV1_{i}")
        if not os.path.exists(folder):
            return folder
        i += 1


def main():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--gui", action="store_true", help="Launch GUI mode")
    p.add_argument("--batch", action="store_true", help="Run batch mode")
    p.add_argument("--single", action="store_true", help="Run single-image mode")
    p.add_argument("--dir", type=str, default=None, help="Choose input folder")
    p.add_argument("--z", type=int, default=None, help="Override SINGLE_Z_INDEX")
    args, unknown = p.parse_known_args()

    mode = "gui"
    if args.batch: mode = "batch"
    elif args.single or args.z is not None: mode = "single"
    base.CONFIG["RUN_MODE"] = mode if mode != "gui" else "single"

    target_dir = args.dir
    if mode == "batch" and target_dir is None:
        if _TK_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            print("Please select the folder containing your microscopy images...")
            target_dir = filedialog.askdirectory(title="Select Input Image Folder")
            root.destroy()
            if not target_dir:
                print("No folder selected. Operation cancelled.")
                return
        else:
            print("Tkinter not available. Please specify folder using --dir 'path/to/data'")
            return
    
    if target_dir:
        base.CONFIG["INPUT_DIR"] = os.path.abspath(target_dir)
        if mode == "batch":
            base.CONFIG["OUTPUT_DIR"] = get_unique_batch_dir_patchv1(base.CONFIG["INPUT_DIR"])
        else:
            base.CONFIG["OUTPUT_DIR"] = os.path.join(base.CONFIG["INPUT_DIR"], "sperm_results_patchV1")
        print(f"INPUT:  {base.CONFIG['INPUT_DIR']}")
        print(f"OUTPUT: {base.CONFIG['OUTPUT_DIR']}")

    if args.z is not None:
        base.CONFIG["SINGLE_Z_INDEX"] = int(args.z)
        base.CONFIG["SINGLE_IMAGE_SELECTION_MODE"] = "z_index"

    print("\n" + "="*70)
    print(" CONSERVATIVE TRACKING PATCH ACTIVE")
    print(" Strategy: STOP bad tracks, don't repair them")
    print(" Checks: width, length, area, centroid consistency")
    print(" Base module:", BASE_PATH.name)
    print("="*70 + "\n")

    if mode == "gui":
        if hasattr(base, "launch_gui"):
            return base.launch_gui()
        else:
            raise RuntimeError("Base script launch_gui() not found.")

    if mode == "batch" and hasattr(base, "process_batch"):
        base.process_batch(base.CONFIG)
        alias_auditor_files(base.CONFIG["OUTPUT_DIR"])
    elif mode == "single" and hasattr(base, "process_one_image"):
        img = base.choose_single_image(base.CONFIG)
        base.process_one_image(img, base.CONFIG, base.CONFIG["OUTPUT_DIR"])
        alias_auditor_files(base.CONFIG["OUTPUT_DIR"])
    else:
        raise RuntimeError(f"Could not find a runnable entry point for {mode} in base script.")


if __name__ == "__main__":
    main()
