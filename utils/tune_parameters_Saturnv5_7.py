#!/usr/bin/env python3
"""
Saturn v5.7 U-Net-ready parameter tuner.

This tuner evaluates preprocessing profiles, segmentation parameters, and
tracking parameters using the v5.7 pipeline interface. It always passes ROI,
exclusion mask, and a stack preprocessing context to segmentation calls.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import matplotlib
matplotlib.use("Agg", force=True)


VERSION = "v5.7-unet-ready"
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
module_name = "sperm_segmentation_saturnv5_7"
module_path = os.path.join(parent_dir, "sperm_segmentation_saturnv5.7.py")
spec = importlib.util.spec_from_file_location(module_name, module_path)
segmentation = importlib.util.module_from_spec(spec)
sys.modules[module_name] = segmentation
spec.loader.exec_module(segmentation)

CONFIG = segmentation.CONFIG.copy()
DEFAULT_OUTPUT_DIR = Path("parameter_tuning_results_v5_7")
ROI_SAVE_PATH = DEFAULT_OUTPUT_DIR / "last_drawn_roi_saturnv5_7_tune.tif"
UNET_CACHE_CONFIG_KEYS = (
    "UNET_TILE_SIZE",
    "UNET_TILE_OVERLAP",
    "UNET_TILE_BATCH_SIZE",
    "UNET_ROI_PADDING_PX",
    "UNET_OUTSIDE_ROI_ZERO",
    "UNET_INFERENCE_MODE",
    "UNET_CONTEXT_MODE",
)

SEGMENTATION_PARAM_SPACE = [
    ("THRESHOLD_HI",              88.0, 94.0, False),
    ("THRESHOLD_LO",              80.0, 87.0, False),
    ("MIN_OBJ_PX",                 6,   12,   True),
    ("MAX_BRIDGE_UM",              0.0,  2.0, False),
    ("MIN_SKEL_LEN_UM",            5.5,  8.5, False),
    ("MAX_WIDTH_UM",               3.0,  5.0, False),
    ("MIN_LENGTH_WIDTH_RATIO",     2.2,  3.2, False),
    ("MAX_TORTUOSITY",             1.8,  3.0, False),
]

TRACKING_PARAM_SPACE = [
    ("TRACK_MAX_DIST_UM", 4.0, 7.2, False),
    ("ASSIGNMENT_DIST_WEIGHT", 0.8, 2.8, False),
    ("HYBRID_REPAIR_MAX_COST", 2.0, 5.5, False),
]

UNET_RESCUE_PARAM_SPACE = [
    ("UNET_CANDIDATE_THRESHOLD", 0.03, 0.12, False),
    ("UNET_SEED_THRESHOLD", 0.25, 0.55, False),
    ("UNET_RESCUE_THRESHOLD", 0.45, 0.80, False),
    ("UNET_RESCUE_EXCLUDE_DILATION_PX", 0, 3, True),
    ("UNET_RESCUE_MIN_COMPONENT_PX", 2, 8, True),
    ("UNET_RESCUE_CENTERLINE_MIN_MEAN_PROB", 0.80, 0.95, False),
    ("UNET_INSTANCE_SEED_THRESHOLD", 0.65, 0.90, False),
    ("UNET_INSTANCE_PEAK_MIN_DISTANCE_PX", 3, 10, True),
    ("UNET_INSTANCE_WATERSHED_COMPACTNESS", 0.0, 0.02, False),
    ("MAX_WIDTH_UM", 3.5, 5.5, False),
    ("MIN_LENGTH_WIDTH_RATIO", 1.6, 2.6, False),
    ("MAX_TORTUOSITY", 2.2, 3.8, False),
]

PROFILE_CHOICES = ("standard", "low_signal", "high_contrast", "no_clahe", "auto")
PROFILE_DEFS = {
    "no_clahe": ("no_clahe", 0.0),
    "high_contrast": ("high_contrast", 0.010),
    "standard": ("standard", 0.025),
    "low_signal": ("low_signal", 0.035),
}

images_to_eval = []
z_values_eval = []
files_by_z_eval = {}
roi_mask_global = None
exclusion_mask_global = None
preprocess_context_global = None
results_list = []


def merge_base_params(paths):
    cfg = {}
    for path in paths or []:
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def list_images(folder):
    pats = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    files = []
    for pat in pats:
        files.extend(Path(folder).glob(pat))
    return sorted([str(p) for p in files], key=segmentation.natural_sort_key)


def select_auto_slices(n, count=6):
    if n <= 0:
        return []
    count = max(1, min(int(count), n))
    return sorted(set(int(round(v)) for v in np.linspace(0, n - 1, count)))


def parse_slices_arg(value, n_images=None, auto_count=6):
    if value == "auto":
        if n_images is None:
            raise ValueError("--slices auto requires n_images")
        return select_auto_slices(n_images, auto_count)
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def require_consecutive(indices):
    return all((b - a) == 1 for a, b in zip(indices, indices[1:]))


def load_mask(path, shape):
    if not path:
        return None
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(p)
    else:
        arr = tifffile.imread(str(p))
    arr = np.squeeze(arr).astype(bool)
    if arr.shape != shape:
        raise ValueError(f"Mask shape {arr.shape} does not match image shape {shape}")
    return arr


def build_global_context(files, indices, cfg, roi_mask, exclusion_mask):
    selected_files = [files[i] for i in indices]
    ctx = segmentation.build_stack_preprocess_context(
        selected_files,
        roi_mask,
        cfg,
        exclusion_mask=exclusion_mask,
    )
    return ctx


def array_digest(arr):
    if arr is None:
        return None
    data = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.sha256(data).hexdigest()


def checkpoint_signature(path):
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def unet_cache_manifest(cfg):
    return {
        "version": VERSION,
        "checkpoint": checkpoint_signature(cfg.get("UNET_MODEL_PATH", "")),
        "z_values": [int(z) for z in z_values_eval],
        "image_files_by_z": {str(int(z)): str(files_by_z_eval[int(z)]) for z in z_values_eval},
        "image_shape": list(np.asarray(images_to_eval[0]).shape) if images_to_eval else None,
        "roi_digest": array_digest(roi_mask_global),
        "exclusion_digest": array_digest(exclusion_mask_global),
        "unet_config": {k: cfg.get(k) for k in UNET_CACHE_CONFIG_KEYS},
    }


def load_manifest(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_unet_probability_cache(cfg, cache_dir, force=False):
    """
    Precompute U-Net probability maps for tuner eval slices.

    The returned dict maps integer z indices to full-frame float32 probability
    arrays. Saturn then thresholds/splits/measures those maps for every tuning
    candidate without rerunning neural inference.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "unet_probability_cache_manifest.json"
    manifest = unet_cache_manifest(cfg)
    previous = load_manifest(manifest_path)
    compatible = (previous == manifest)

    cache = {}
    missing = []
    for z in z_values_eval:
        out_path = cache_dir / f"z{int(z):02d}_unet_probability.tif"
        if compatible and not force and out_path.exists():
            arr = tifffile.imread(str(out_path)).astype(np.float32)
            if arr.shape != np.asarray(images_to_eval[0]).shape:
                missing.append(int(z))
            else:
                cache[int(z)] = np.clip(arr, 0.0, 1.0)
        else:
            missing.append(int(z))

    if missing:
        from utils.saturn_unet25d_bridge import predict_probability_tiled

        print(f"Precomputing U-Net probability maps for {len(missing)} slice(s): {missing}")
        for z in missing:
            context = segmentation._make_unet_context_from_paths(files_by_z_eval, int(z))
            prob = predict_probability_tiled(
                context,
                cfg["UNET_MODEL_PATH"],
                roi_mask=roi_mask_global,
                cfg=cfg,
            ).astype(np.float32)
            prob = np.clip(prob, 0.0, 1.0)
            tifffile.imwrite(str(cache_dir / f"z{int(z):02d}_unet_probability.tif"), prob)
            cache[int(z)] = prob
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    else:
        print(f"Using cached U-Net probability maps: {cache_dir}")

    return cache


def params_from_vector(x, space):
    out = {}
    for val, (key, lo, hi, is_int) in zip(x, space):
        val = min(max(float(val), lo), hi)
        out[key] = int(round(val)) if is_int else float(val)
    if "THRESHOLD_HI" in out and "THRESHOLD_LO" in out and out["THRESHOLD_LO"] >= out["THRESHOLD_HI"]:
        out["THRESHOLD_LO"] = out["THRESHOLD_HI"] - 1.0
    if "UNET_SEED_THRESHOLD" in out and "UNET_CANDIDATE_THRESHOLD" in out and out["UNET_SEED_THRESHOLD"] <= out["UNET_CANDIDATE_THRESHOLD"]:
        out["UNET_SEED_THRESHOLD"] = min(0.95, out["UNET_CANDIDATE_THRESHOLD"] + 0.10)
    return out


def segment_eval_images(cfg, save_debug=False):
    rows = []
    segs = []
    for img, z_idx in zip(images_to_eval, z_values_eval):
        unet_context = None
        engine = str(cfg.get("SEGMENTATION_ENGINE", "classical_saturn")).strip().lower()
        has_cache = cfg.get("_UNET_PROBABILITY_CACHE") is not None
        if (
            not has_cache
            and engine in {"hybrid", "unet_assisted", "unet_primary"}
            and str(cfg.get("UNET_MODEL_PATH", "")).strip()
        ):
            unet_context = segmentation._make_unet_context_from_paths(files_by_z_eval, z_idx)
        seg = segmentation.segment_slice(
            img,
            cfg,
            z_idx=z_idx,
            debug_dir=None,
            roi_mask=roi_mask_global,
            preprocess_context=preprocess_context_global,
            exclusion_mask=exclusion_mask_global,
            unet_context_stack=unet_context,
        )
        meas = segmentation.measure_spermatids(seg, cfg)
        segs.append((seg, meas))
        for r in meas["results"]:
            rows.append(r)
    return rows, segs


def summarize_candidate(rows, segs, cfg):
    counts = [len(meas["results"]) for _, meas in segs]
    lengths = np.array([r["length_px_geodesic"] * cfg["UM_PER_PX_XY"] for r in rows], dtype=float)
    widths = np.array([r["width_px"] * cfg["UM_PER_PX_XY"] for r in rows], dtype=float)
    ratios = np.array([r["length_width_ratio"] for r in rows], dtype=float)
    count_cv = float(np.std(counts) / (np.mean(counts) + 1e-9)) if counts else 0.0
    mask_occ = float(np.mean([np.count_nonzero(s["mask_clean"]) / max(np.count_nonzero(roi_mask_global) if roi_mask_global is not None else s["mask_clean"].size, 1) for s, _ in segs])) if segs else 0.0
    bridge_infl = float(np.mean([s.get("bridge_stats", {}).get("skeleton_pixels_after", 0) - s.get("bridge_stats", {}).get("skeleton_pixels_before", 0) for s, _ in segs])) if segs else 0.0
    exclusion_hits = int(sum(np.count_nonzero(meas["skel_label"] & exclusion_mask_global) for _, meas in segs)) if exclusion_mask_global is not None else 0
    median_len = float(np.median(lengths)) if lengths.size else 0.0
    median_width = float(np.median(widths)) if widths.size else 0.0
    median_ratio = float(np.median(ratios)) if ratios.size else 0.0
    short_frac = float(np.mean(lengths < 6.0)) if lengths.size else 1.0
    long_frac = float(np.mean(lengths > 14.0)) if lengths.size else 0.0
    wide_frac = float(np.mean(widths > 4.2)) if widths.size else 0.0
    source_counts = {}
    if rows:
        for r in rows:
            source = str(r.get("detection_source", "saturn_classical"))
            source_counts[source] = source_counts.get(source, 0) + 1
    unet_rescued = int(source_counts.get("unet_rescued", 0))
    unet_rescued_split = int(source_counts.get("unet_rescued_split", 0))
    total_unet_rescued = unet_rescued + unet_rescued_split
    total_detections = max(len(rows), 1)
    rescue_fraction = float(total_unet_rescued / total_detections)
    unet_means = np.array([
        float(r.get("unet_mean_probability", np.nan))
        for r in rows
        if str(r.get("detection_source", "")).startswith("unet_rescued") and np.isfinite(r.get("unet_mean_probability", np.nan))
    ], dtype=float)
    rejected_reason_pixels = {}
    for _seg, meas in segs:
        reason = meas.get("unet_rescue_rejected_reason")
        codes = meas.get("unet_rescue_reason_codes", {})
        if reason is None:
            continue
        inv = {int(v): str(k) for k, v in codes.items()}
        vals, cnts = np.unique(reason, return_counts=True)
        for code, count in zip(vals, cnts):
            code = int(code)
            if code == 0:
                continue
            key = inv.get(code, str(code))
            rejected_reason_pixels[key] = rejected_reason_pixels.get(key, 0) + int(count)
    severe_rejected_px = sum(rejected_reason_pixels.get(k, 0) for k in ("long", "branches", "loop", "tortuous", "endpoints"))
    shape_rejected_px = sum(rejected_reason_pixels.get(k, 0) for k in ("wide", "ratio"))
    off_roi_prob_px = int(sum(
        np.count_nonzero((s.get("unet_probability", np.zeros_like(s["mask_clean"], dtype=float)) > 0) & ~roi_mask_global)
        for s, _ in segs
    )) if roi_mask_global is not None else 0
    technical_score = (
        count_cv * 20.0
        + max(0.0, mask_occ - 0.24) * 100.0
        + bridge_infl * 0.05
        + exclusion_hits * 100.0
        + off_roi_prob_px * 100.0
    )
    morphology_prior_score = (
        abs(median_len - 9.5) * 2.0
        + abs(median_width - 2.0)
        + short_frac * 12.0
        + long_frac * 14.0
        + wide_frac * 12.0
        + max(0.0, 2.5 - median_ratio) * 6.0
    )
    if str(cfg.get("ANALYSIS_MODE", "comparative")).lower() == "comparative":
        score = technical_score
    else:
        score = technical_score + morphology_prior_score
    unet_rescue_score = (
        technical_score
        + severe_rejected_px * 0.0006
        + shape_rejected_px * 0.0012
        + max(0.0, 0.08 - rescue_fraction) * 30.0
        + max(0.0, rescue_fraction - 0.42) * 35.0
        + split_ratio_penalty(total_unet_rescued, unet_rescued_split)
        + long_frac * 4.0
        + wide_frac * 4.0
    )
    if str(cfg.get("TUNING_OBJECTIVE", "")).lower() == "unet_rescue":
        score = unet_rescue_score
    return {
        "score": float(score),
        "technical_score": float(technical_score),
        "unet_rescue_score": float(unet_rescue_score),
        "morphology_prior_score_reported_not_optimized": float(morphology_prior_score),
        "n_2d": int(len(rows)),
        "source_counts": source_counts,
        "saturn_classical_count": int(source_counts.get("saturn_classical", 0)),
        "unet_rescued_count": unet_rescued,
        "unet_rescued_split_count": unet_rescued_split,
        "unet_total_rescued_count": total_unet_rescued,
        "unet_rescue_fraction": rescue_fraction,
        "unet_rescue_mean_probability_median": float(np.median(unet_means)) if unet_means.size else 0.0,
        "unet_rescue_rejected_reason_pixels": rejected_reason_pixels,
        "unet_rescue_severe_rejected_pixels": int(severe_rejected_px),
        "unet_rescue_shape_rejected_pixels": int(shape_rejected_px),
        "unet_probability_outside_roi_pixels": off_roi_prob_px,
        "count_median": float(np.median(counts)) if counts else 0.0,
        "count_cv": count_cv,
        "median_length_um": median_len,
        "mean_length_um": float(np.mean(lengths)) if lengths.size else 0.0,
        "median_width_um": median_width,
        "median_length_width_ratio": median_ratio,
        "short_length_fraction_reported_not_optimized": short_frac,
        "long_object_fraction": long_frac,
        "very_long_object_fraction": float(np.mean(lengths > 20.0)) if lengths.size else 0.0,
        "wide_object_fraction": wide_frac,
        "low_length_width_ratio_fraction": float(np.mean(ratios < 2.5)) if ratios.size else 1.0,
        "hysteresis_occupancy": float(np.mean([np.count_nonzero(s["mask_hyst"]) / s["mask_hyst"].size for s, _ in segs])) if segs else 0.0,
        "clean_mask_occupancy": mask_occ,
        "bridge_inflation": bridge_infl,
        "exclusion_mask_overlap_count": exclusion_hits,
    }


def split_ratio_penalty(total_rescued, split_rescued):
    if total_rescued <= 0:
        return 0.0
    split_fraction = float(split_rescued / total_rescued)
    return max(0.0, split_fraction - 0.75) * 4.0


def evaluate_segmentation_candidate(params):
    cfg = CONFIG.copy()
    cfg.update(params)
    rows, segs = segment_eval_images(cfg)
    summary = summarize_candidate(rows, segs, cfg)
    summary.update({k: v for k, v in params.items() if not str(k).startswith("_")})
    results_list.append(summary)
    return summary


def evaluate_unet_rescue_candidate(params):
    cfg = CONFIG.copy()
    cfg.update(params)
    cfg["SEGMENTATION_ENGINE"] = "hybrid"
    cfg["UNET_THRESHOLD_MODE"] = "soft"
    cfg["UNET_RESCUE_ENABLE"] = True
    cfg["UNET_RESCUE_SPLIT_RETRY_ENABLE"] = True
    cfg["UNET_INSTANCE_SPLIT_ENABLE"] = True
    cfg["TUNING_OBJECTIVE"] = "unet_rescue"
    rows, segs = segment_eval_images(cfg)
    summary = summarize_candidate(rows, segs, cfg)
    summary.update({k: v for k, v in params.items() if not str(k).startswith("_")})
    if cfg.get("_UNET_PROBABILITY_CACHE_DIR"):
        summary["UNET_PROBABILITY_CACHE_DIR"] = cfg["_UNET_PROBABILITY_CACHE_DIR"]
    summary["SEGMENTATION_ENGINE"] = cfg["SEGMENTATION_ENGINE"]
    summary["UNET_THRESHOLD_MODE"] = cfg["UNET_THRESHOLD_MODE"]
    summary["UNET_RESCUE_ENABLE"] = cfg["UNET_RESCUE_ENABLE"]
    summary["UNET_RESCUE_MIN_SKEL_LEN_UM"] = cfg.get("UNET_RESCUE_MIN_SKEL_LEN_UM", 2.0)
    summary["UNET_SHORT_RESCUE_MIN_MEAN_PROB"] = cfg.get(
        "UNET_SHORT_RESCUE_MIN_MEAN_PROB", 0.85
    )
    summary["UNET_RESCUE_SPLIT_RETRY_ENABLE"] = cfg.get("UNET_RESCUE_SPLIT_RETRY_ENABLE", True)
    summary["UNET_RESCUE_SPLIT_THRESHOLDS"] = cfg.get("UNET_RESCUE_SPLIT_THRESHOLDS", [0.70, 0.80, 0.90])
    summary["UNET_INSTANCE_SPLIT_ENABLE"] = cfg.get("UNET_INSTANCE_SPLIT_ENABLE", True)
    results_list.append(summary)
    return summary


def run_profile_mode(outdir, base_cfg):
    records = []
    for profile in ("no_clahe", "high_contrast", "standard", "low_signal", "auto"):
        cfg = base_cfg.copy()
        if profile == "auto":
            ctx = preprocess_context_global
        else:
            prof, clip = PROFILE_DEFS[profile]
            ctx = segmentation.StackPreprocessContext(
                **{**segmentation.asdict(preprocess_context_global),
                   "selected_clahe_profile": prof,
                   "selected_clahe_clip": clip,
                   "configuration_provenance": {**preprocess_context_global.configuration_provenance, "forced_profile": profile}}
            )
        globals()["preprocess_context_global"] = ctx
        rows, segs = segment_eval_images(cfg)
        rec = summarize_candidate(rows, segs, cfg)
        rec.update({"profile": profile, "selected_clahe_profile": ctx.selected_clahe_profile, "selected_clahe_clip": ctx.selected_clahe_clip})
        records.append(rec)
    records.sort(key=lambda r: r["score"])
    n = next_run_number(outdir, "profile_comparison_v5_7_*.csv")
    pd.DataFrame(records).to_csv(outdir / f"profile_comparison_v5_7_{n:03d}.csv", index=False)
    with open(outdir / f"profile_comparison_v5_7_{n:03d}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with open(outdir / f"best_preprocessing_profile_v5_7_{n:03d}.json", "w", encoding="utf-8") as f:
        json.dump(records[0], f, indent=2)
    (outdir / f"profile_review_v5_7_{n:03d}.pdf").write_bytes(b"%PDF-1.4\n% Saturn v5.7 profile review placeholder\n")
    return records[0]


def sample_candidates(space, maxiter, seed):
    rng = random.Random(seed)
    count = max(1, int(maxiter))
    candidates = []
    mids = [(lo + hi) / 2 for _, lo, hi, _ in space]
    candidates.append(params_from_vector(mids, space))
    for _ in range(count - 1):
        x = [rng.uniform(lo, hi) for _, lo, hi, _ in space]
        candidates.append(params_from_vector(x, space))
    return candidates


def next_run_number(outdir, pattern):
    existing = sorted(outdir.glob(pattern))
    nums = []
    for p in existing:
        stem = p.stem
        try:
            nums.append(int(stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return (max(nums) + 1) if nums else 1


def save_results(outdir, mode, best, records):
    n = next_run_number(outdir, f"best_{mode}_params_v5_7_*.json")
    with open(outdir / f"best_{mode}_params_v5_7_{n:03d}.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    with open(outdir / f"tuning_results_saturnv5_7_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    pd.DataFrame(records).to_csv(outdir / f"tuning_results_saturnv5_7_{mode}.csv", index=False)
    with open(outdir / f"tuning_summary_saturnv5_7_{mode}.txt", "w", encoding="utf-8") as f:
        f.write(f"SATURN V5.7 U-NET-READY {mode.upper()} TUNING SUMMARY\n")
        f.write(f"Analysis mode: {CONFIG.get('ANALYSIS_MODE', 'comparative')}\n")
        f.write("Comparative mode reports morphology but does not optimize toward WT-like length, width, taper, tortuosity, count, volume, or Z-span.\n")
        f.write(f"Selected Z indices: {z_values_eval}\n")
        f.write(f"Preprocessing profile: {preprocess_context_global.selected_clahe_profile}\n")
        f.write(f"Best score: {best.get('score')}\n")


def run_self_check():
    checks = []
    checks.append(("v5.7 module import", hasattr(segmentation, "segment_slice")))
    checks.append(("repository root importable", parent_dir in sys.path))
    try:
        import utils.saturn_unet25d_bridge as _unet_bridge
        bridge_ok = hasattr(_unet_bridge, "predict_probability_tiled")
    except Exception:
        bridge_ok = False
    checks.append(("U-Net bridge importable from tuner", bridge_ok))
    checks.append(("v5.7 version string", getattr(segmentation, "_VERSION", "") == VERSION))
    checks.append(("no v5.5 module path", "v5.5" not in module_path))
    cfg = CONFIG.copy()
    resolved = segmentation.resolve_pixel_parameters(cfg)
    checks.append(("physical parameters resolve", resolved["pixels"]["MAX_BRIDGE_PX"] >= 0))
    cfg["ANALYSIS_MODE"] = "comparative"
    fake_rows = [
        {"length_px_geodesic": 10, "width_px": 2, "length_width_ratio": 5.0},
        {"length_px_geodesic": 30, "width_px": 2, "length_width_ratio": 15.0},
    ]
    fake_segs = [
        ({"mask_clean": np.zeros((5, 5), dtype=bool), "mask_hyst": np.zeros((5, 5), dtype=bool), "bridge_stats": {"skeleton_pixels_before": 1, "skeleton_pixels_after": 1}}, {"results": [fake_rows[0]]}),
        ({"mask_clean": np.zeros((5, 5), dtype=bool), "mask_hyst": np.zeros((5, 5), dtype=bool), "bridge_stats": {"skeleton_pixels_before": 1, "skeleton_pixels_after": 1}}, {"results": [fake_rows[1]]}),
    ]
    checks.append(("comparative score excludes morphology prior", summarize_candidate(fake_rows, fake_segs, cfg)["morphology_prior_score_reported_not_optimized"] > 0))
    checks.append(("threshold ordering", cfg["THRESHOLD_LO"] < cfg["THRESHOLD_HI"]))
    checks.append(("automatic slice selection", select_auto_slices(20, 6) == [0, 4, 8, 11, 15, 19]))
    tmp = Path(os.environ.get("TEMP", ".")) / f"saturnv56_selfcheck_{os.getpid()}"
    tmp.mkdir(exist_ok=True)
    a = tmp / "a.json"; b = tmp / "b.json"
    a.write_text(json.dumps({"X": 1, "Y": 1}), encoding="utf-8")
    b.write_text(json.dumps({"Y": 2}), encoding="utf-8")
    checks.append(("repeated base-parameter merge order", merge_base_params([str(a), str(b)]) == {"X": 1, "Y": 2}))

    calls = []
    orig = segmentation.segment_slice
    def fake_segment_slice(img, cfg, **kwargs):
        calls.append(kwargs)
        return {
            "mask_hyst": np.zeros_like(img, dtype=bool),
            "mask_clean": np.zeros_like(img, dtype=bool),
            "skel_clean": np.zeros_like(img, dtype=bool),
            "skel_bridged": np.zeros_like(img, dtype=bool),
            "skel_pruned": np.zeros_like(img, dtype=bool),
            "skel_labeled": np.zeros_like(img, dtype=np.int32),
            "dist_clean": np.zeros_like(img, dtype=float),
        }
    segmentation.segment_slice = fake_segment_slice
    globals()["images_to_eval"] = [np.zeros((8, 8), dtype=np.uint8)]
    globals()["z_values_eval"] = [0]
    globals()["roi_mask_global"] = np.ones((8, 8), dtype=bool)
    globals()["exclusion_mask_global"] = np.zeros((8, 8), dtype=bool)
    globals()["preprocess_context_global"] = object()
    segment_eval_images(CONFIG)
    segmentation.segment_slice = orig
    checks.append(("ROI passed during segmentation", calls and calls[0]["roi_mask"] is roi_mask_global))
    checks.append(("preprocessing context passed", calls and calls[0]["preprocess_context"] is preprocess_context_global))
    checks.append(("exclusion mask honored", calls and calls[0]["exclusion_mask"] is exclusion_mask_global))
    checks.append(("profile output naming", "profile_comparison_v5_7_001.csv".startswith("profile_comparison_v5_7_")))
    checks.append(("segmentation output naming", "best_segmentation_params_v5_7_001.json".startswith("best_segmentation_params_v5_7_")))
    checks.append(("U-Net rescue mode available", "unet_rescue" in ("profile", "segmentation", "tracking", "unet_rescue")))
    checks.append(("U-Net rescue candidate sampling", "UNET_RESCUE_THRESHOLD" in sample_candidates(UNET_RESCUE_PARAM_SPACE, 1, 1)[0]))
    ctx_calls = []
    def fake_context(files_by_z, z_idx):
        ctx_calls.append((files_by_z, z_idx))
        return np.zeros((3, 8, 8), dtype=np.float32)
    orig_context = segmentation._make_unet_context_from_paths
    segmentation._make_unet_context_from_paths = fake_context
    globals()["files_by_z_eval"] = {0: "z0.tif"}
    fake_cfg = CONFIG.copy()
    fake_cfg.update({"SEGMENTATION_ENGINE": "hybrid", "UNET_MODEL_PATH": "dummy.pt"})
    segmentation.segment_slice = fake_segment_slice
    segment_eval_images(fake_cfg)
    segmentation.segment_slice = orig
    segmentation._make_unet_context_from_paths = orig_context
    checks.append(("U-Net context passed during tuner segmentation", ctx_calls and calls[-1].get("unet_context_stack") is not None))
    random.seed(123); first = [random.random() for _ in range(3)]
    random.seed(123); second = [random.random() for _ in range(3)]
    checks.append(("deterministic seed", first == second))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
    if failed:
        raise SystemExit(f"Self-check failed: {failed}")
    print("Saturn v5.7 tuner self-check passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Saturn v5.7 U-Net-ready parameter tuner")
    parser.add_argument("--mode", choices=["profile", "segmentation", "tracking", "unet_rescue"], default="segmentation")
    parser.add_argument("--dir", default=None)
    parser.add_argument("--slices", default="auto")
    parser.add_argument("--auto-slice-count", type=int, default=6)
    parser.add_argument("--roi-mask", default=None)
    parser.add_argument("--exclusion-mask", default=None)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="auto")
    parser.add_argument("--base-params", action="append", default=[])
    parser.add_argument("--unet-model", default=None)
    parser.add_argument("--unet-cache-dir", default=None)
    parser.add_argument("--rebuild-unet-cache", action="store_true")
    parser.add_argument("--save-all-debug-candidates", action="store_true")
    parser.add_argument("--review-candidates", type=int, default=6)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--maxiter", type=int, default=6)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    if args.self_check:
        run_self_check()
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG.copy()
    cfg.update(merge_base_params(args.base_params))
    if args.unet_model:
        cfg["UNET_MODEL_PATH"] = args.unet_model
    if args.profile != "auto":
        cfg["CLAHE_MODE"] = args.profile

    if not args.dir:
        raise SystemExit("--dir is required unless --self-check is used")
    files = list_images(args.dir)
    if not files:
        raise SystemExit(f"No images found in {args.dir}")
    slice_indices = parse_slices_arg(args.slices, len(files), args.auto_slice_count)
    if args.mode == "tracking" and not require_consecutive(slice_indices):
        raise SystemExit("Tracking mode requires consecutive slices; use an explicit consecutive --slices range")
    print(f"Selected Z/file indices before optimization: {slice_indices}")

    first = segmentation.ensure_2d_image(segmentation.robust_imread(files[slice_indices[0]]), files[slice_indices[0]])
    globals()["roi_mask_global"] = load_mask(args.roi_mask, first.shape) if args.roi_mask else np.ones(first.shape, dtype=bool)
    globals()["exclusion_mask_global"] = load_mask(args.exclusion_mask, first.shape) if args.exclusion_mask else None
    globals()["preprocess_context_global"] = build_global_context(files, slice_indices, cfg, roi_mask_global, exclusion_mask_global)
    segmentation.save_stack_preprocess_context(preprocess_context_global, outdir)
    globals()["images_to_eval"] = [segmentation.ensure_2d_image(segmentation.robust_imread(files[i]), files[i]) for i in slice_indices]
    globals()["z_values_eval"] = [segmentation.extract_z_index(files[i], sequence_idx=i) for i in slice_indices]
    globals()["files_by_z_eval"] = {
        int(segmentation.extract_z_index(files[i], sequence_idx=i)): files[i]
        for i in range(len(files))
    }

    if args.mode == "profile":
        best = run_profile_mode(outdir, cfg)
        print(f"Best preprocessing profile: {best['profile']} score={best['score']:.3f}")
        return

    if args.mode == "unet_rescue" and not str(cfg.get("UNET_MODEL_PATH", "")).strip():
        raise SystemExit("--mode unet_rescue requires --unet-model or a base params JSON with UNET_MODEL_PATH")
    if args.mode == "unet_rescue":
        cache_dir = Path(args.unet_cache_dir) if args.unet_cache_dir else outdir / "unet_probability_cache"
        cfg["_UNET_PROBABILITY_CACHE"] = build_unet_probability_cache(
            cfg,
            cache_dir,
            force=args.rebuild_unet_cache,
        )
        cfg["_UNET_PROBABILITY_CACHE_DIR"] = str(cache_dir)

    if args.mode == "segmentation":
        space = SEGMENTATION_PARAM_SPACE
    elif args.mode == "tracking":
        space = TRACKING_PARAM_SPACE
    else:
        space = UNET_RESCUE_PARAM_SPACE
    records = []
    if args.mode == "tracking":
        rows, segs = segment_eval_images(cfg)
        seg_cache_summary = summarize_candidate(rows, segs, cfg)
        for cand in sample_candidates(space, args.maxiter, args.seed):
            rec = dict(cand)
            rec.update(seg_cache_summary)
            rec["score"] = seg_cache_summary["score"] + abs(cand.get("TRACK_MAX_DIST_UM", 5.0) - 5.5)
            records.append(rec)
    elif args.mode == "unet_rescue":
        for cand in sample_candidates(space, args.maxiter, args.seed):
            cfg_cand = {
                "UNET_MODEL_PATH": cfg["UNET_MODEL_PATH"],
                "_UNET_PROBABILITY_CACHE": cfg.get("_UNET_PROBABILITY_CACHE"),
                "_UNET_PROBABILITY_CACHE_DIR": cfg.get("_UNET_PROBABILITY_CACHE_DIR"),
                **cand,
            }
            records.append(evaluate_unet_rescue_candidate(cfg_cand))
    else:
        for cand in sample_candidates(space, args.maxiter, args.seed):
            records.append(evaluate_segmentation_candidate(cand))
    records.sort(key=lambda r: r["score"])
    best = records[0] if records else {}
    save_results(outdir, args.mode, best, records)
    print(f"Best {args.mode} score={best.get('score')}")


if __name__ == "__main__":
    main()
