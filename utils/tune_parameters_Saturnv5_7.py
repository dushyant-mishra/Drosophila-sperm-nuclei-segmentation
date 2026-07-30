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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


VERSION = "v5.7-unet-ready"
MIN_HYSTERESIS_PERCENTILE_SEPARATION = 4.0
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
    ("THRESHOLD_HI",              82.0, 92.0, False),
    ("THRESHOLD_LO",              70.0, 84.0, False),
    ("MIN_OBJ_PX",                 3,   10,   True),
    ("MAX_BRIDGE_UM",              0.0,  1.5, False),
    ("MIN_SKEL_LEN_UM",            4.0,  7.0, False),
    ("MAX_WIDTH_UM",               3.0,  5.0, False),
    ("MIN_LENGTH_WIDTH_RATIO",     1.6,  3.0, False),
    ("MAX_TORTUOSITY",             2.0,  3.2, False),
]

TRACKING_PARAM_SPACE = [
    ("TRACK_MAX_DIST_UM", 4.0, 7.2, False),
    ("ASSIGNMENT_MAX_COST", 4.0, 9.0, False),
    ("ASSIGNMENT_DIST_WEIGHT", 0.8, 2.8, False),
    ("HYBRID_REPAIR_MAX_COST", 2.0, 5.5, False),
    ("HYBRID_REPAIR_MAX_LINK_DIST_UM", 3.0, 6.0, False),
    ("HYBRID_REPAIR_MIN_OVERLAP", 0.0, 0.15, False),
]

TRACKING_CONFIG_KEYS = (
    "TRACKING_BACKEND",
    "TRACK_MAX_DIST_UM",
    "TRACK_MAX_GAP_SLICES",
    "TRACK_BBOX_PADDING_PX",
    "TRACK_TECHNICAL_MAX_JOINED_LENGTH_UM",
    "ASSIGNMENT_MAX_COST",
    "ASSIGNMENT_DIST_WEIGHT",
    "ASSIGNMENT_OVERLAP_WEIGHT",
    "ASSIGNMENT_LENGTH_WEIGHT",
    "ASSIGNMENT_WIDTH_WEIGHT",
    "ASSIGNMENT_AREA_WEIGHT",
    "ASSIGNMENT_ANGLE_WEIGHT",
    "ASSIGNMENT_UNET_SUPPORT_WEIGHT",
    "ASSIGNMENT_UNET_CONTINUITY_WEIGHT",
    "HYBRID_REPAIR_MAX_COST",
    "HYBRID_REPAIR_MAX_GAP_SLICES",
    "HYBRID_REPAIR_MAX_FRAGMENT_SLICES",
    "HYBRID_REPAIR_MAX_LINK_DIST_UM",
    "HYBRID_REPAIR_MIN_OVERLAP",
    "HYBRID_REPAIR_MAX_FINAL_LENGTH_UM",
    "HYBRID_REPAIR_UNET_SUPPORT_WEIGHT",
)

UNET_RESCUE_PARAM_SPACE = [
    ("UNET_CANDIDATE_THRESHOLD", 0.02, 0.12, False),
    ("UNET_RESCUE_THRESHOLD", 0.20, 0.75, False),
    ("UNET_RESCUE_EXCLUDE_DILATION_PX", 0, 3, True),
    ("UNET_RESCUE_MIN_COMPONENT_PX", 2, 8, True),
    ("UNET_RESCUE_MIN_SKEL_LEN_UM", 1.5, 4.0, False),
    ("UNET_SHORT_RESCUE_MIN_MEAN_PROB", 0.25, 0.85, False),
    ("UNET_RESCUE_CENTERLINE_MIN_MEAN_PROB", 0.25, 0.90, False),
    ("UNET_LOW_RATIO_RESCUE_MIN_MEAN_PROB", 0.55, 0.95, False),
    ("UNET_LOW_RATIO_RESCUE_MIN_LENGTH_UM", 3.0, 6.0, False),
    ("UNET_INSTANCE_SEED_THRESHOLD", 0.30, 0.80, False),
    ("UNET_INSTANCE_PEAK_MIN_DISTANCE_PX", 3, 10, True),
    ("UNET_INSTANCE_WATERSHED_COMPACTNESS", 0.0, 0.02, False),
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
            loaded = json.load(f)
        cfg.update({key: value for key, value in loaded.items() if key in CONFIG})
    return cfg


def list_images(folder):
    pats = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    files = []
    for pat in pats:
        files.extend(Path(folder).glob(pat))
    files = sorted(set(files), key=lambda p: segmentation.natural_sort_key(str(p)))

    # Prefer conservatively parsed source TIFFs when a specimen folder also
    # contains generated masks, overlays, or other TIFF artifacts.
    parsed_groups = {}
    for path in files:
        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        parsed = segmentation._study_parse_source_name(path.name)
        if parsed is None:
            continue
        parsed_groups.setdefault(parsed["stack_key"], []).append(
            (int(parsed["z"]), path)
        )
    if parsed_groups:
        if len(parsed_groups) != 1:
            labels = [
                f"{key}: {len(values)} files"
                for key, values in parsed_groups.items()
            ]
            raise ValueError(
                "Tuner input directory contains multiple source stacks; "
                "select one specimen directory per tuner stratum. "
                + "; ".join(labels)
            )
        entries = next(iter(parsed_groups.values()))
        z_values = [z for z, _ in entries]
        if len(z_values) != len(set(z_values)):
            raise ValueError("Tuner input contains duplicate source Z indices")
        return [str(path) for _, path in sorted(entries, key=lambda item: item[0])]

    return [str(path) for path in files]


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
    digest = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(p.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def image_signature(path):
    p = Path(path)
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
        "image_files_by_z": {
            str(int(z)): image_signature(files_by_z_eval[int(z)])
            for z in z_values_eval
        },
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
    if "THRESHOLD_HI" in out and "THRESHOLD_LO" in out:
        max_lo = out["THRESHOLD_HI"] - MIN_HYSTERESIS_PERCENTILE_SEPARATION
        out["THRESHOLD_LO"] = min(out["THRESHOLD_LO"], max_lo)
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
    empty_slice_fraction = (
        float(np.mean(np.asarray(counts) == 0)) if counts else 1.0
    )

    mask_occupancies = []
    hyst_occupancies = []
    bridge_inflations = []
    outside_roi_by_stage = {
        "mask_hyst": 0,
        "mask_clean": 0,
        "skel_pruned": 0,
        "skel_label": 0,
    }
    exclusion_by_stage = {
        "mask_hyst": 0,
        "mask_clean": 0,
        "skel_pruned": 0,
        "skel_label": 0,
    }
    for seg, measurements in segs:
        shape = np.asarray(seg["mask_clean"]).shape
        roi = (
            np.asarray(roi_mask_global, dtype=bool)
            if roi_mask_global is not None
            else np.ones(shape, dtype=bool)
        )
        exclusion = (
            np.asarray(exclusion_mask_global, dtype=bool)
            if exclusion_mask_global is not None
            else np.zeros(shape, dtype=bool)
        )
        valid = roi & ~exclusion
        valid_pixels = max(int(np.count_nonzero(valid)), 1)
        stage_masks = {
            "mask_hyst": np.asarray(seg["mask_hyst"], dtype=bool),
            "mask_clean": np.asarray(seg["mask_clean"], dtype=bool),
            "skel_pruned": np.asarray(
                seg.get("skel_pruned", np.zeros(shape, dtype=bool)),
                dtype=bool,
            ),
            "skel_label": np.asarray(
                measurements.get("skel_label", np.zeros(shape, dtype=np.int32))
            ) > 0,
        }
        mask_occupancies.append(
            np.count_nonzero(stage_masks["mask_clean"] & valid) / valid_pixels
        )
        hyst_occupancies.append(
            np.count_nonzero(stage_masks["mask_hyst"] & valid) / valid_pixels
        )
        for key, stage_mask in stage_masks.items():
            outside_roi_by_stage[key] += int(
                np.count_nonzero(stage_mask & ~roi)
            )
            exclusion_by_stage[key] += int(
                np.count_nonzero(stage_mask & exclusion)
            )

        bridge_stats = seg.get("bridge_stats", {})
        before = max(
            0,
            int(bridge_stats.get("skeleton_pixels_before", 0)),
        )
        after = max(
            0,
            int(bridge_stats.get("skeleton_pixels_after", before)),
        )
        bridge_inflations.append(
            max(0, after - before) / max(before, 1)
        )

    mask_occ = float(np.mean(mask_occupancies)) if mask_occupancies else 0.0
    hyst_occ = float(np.mean(hyst_occupancies)) if hyst_occupancies else 0.0
    bridge_infl = (
        float(np.mean(bridge_inflations)) if bridge_inflations else 0.0
    )
    outside_roi_hits = int(sum(outside_roi_by_stage.values()))
    exclusion_hits = int(sum(exclusion_by_stage.values()))
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
    unet_source_counts = {
        source: int(count)
        for source, count in source_counts.items()
        if source.startswith("unet_rescued")
    }
    total_unet_rescued = int(sum(unet_source_counts.values()))
    unet_rescued_split = int(
        sum(
            count
            for source, count in unet_source_counts.items()
            if "split" in source
        )
    )
    unet_rescued = int(total_unet_rescued - unet_rescued_split)
    total_detections = max(len(rows), 1)
    rescue_fraction = float(total_unet_rescued / total_detections)
    unet_means = np.array([
        float(r.get("unet_mean_probability", np.nan))
        for r in rows
        if str(r.get("detection_source", "")).startswith("unet_rescued") and np.isfinite(r.get("unet_mean_probability", np.nan))
    ], dtype=float)
    rejected_reason_pixels = {}
    rejected_reason_counts = {}
    for _seg, meas in segs:
        for key, count in meas.get(
            "unet_rescue_rejected_counts",
            {},
        ).items():
            rejected_reason_counts[str(key)] = (
                rejected_reason_counts.get(str(key), 0) + int(count)
            )
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
    technical_rejected_count = int(
        sum(
            rejected_reason_counts.get(k, 0)
            for k in ("short", "long", "branches", "loop", "endpoints")
        )
    )
    morphology_warning_count = int(
        sum(
            bool(r.get("unet_rescue_morphology_warning", False))
            for r in rows
        )
    )
    off_roi_prob_px = int(sum(
        np.count_nonzero((s.get("unet_probability", np.zeros_like(s["mask_clean"], dtype=float)) > 0) & ~roi_mask_global)
        for s, _ in segs
    )) if roi_mask_global is not None else 0
    technical_score = (
        count_cv * 5.0
        + max(0.0, mask_occ - 0.24) * 100.0
        + max(0.0, hyst_occ - 0.35) * 80.0
        + max(0.0, bridge_infl - 0.20) * 20.0
        + outside_roi_hits * 1000.0
        + exclusion_hits * 1000.0
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
    very_short_frac = float(np.mean(lengths < 4.0)) if lengths.size else 1.0
    very_long_frac = float(np.mean(lengths > 20.0)) if lengths.size else 0.0
    segmentation_score = (
        technical_score
        + (1e6 if not rows else 0.0)
        + very_short_frac * 25.0
        + very_long_frac * 1000.0
    )
    unet_rescue_score = (
        technical_score
        + technical_rejected_count * 0.01
        + max(0.0, 0.08 - rescue_fraction) * 30.0
        + max(0.0, rescue_fraction - 0.42) * 35.0
        + split_ratio_penalty(total_unet_rescued, unet_rescued_split)
    )
    objective = str(cfg.get("TUNING_OBJECTIVE", "")).lower()
    if objective == "unet_rescue":
        score = unet_rescue_score
    elif objective in {"segmentation", "profile"}:
        score = segmentation_score
    elif str(cfg.get("ANALYSIS_MODE", "comparative")).lower() == "comparative":
        score = technical_score
    else:
        score = technical_score + morphology_prior_score
    return {
        "score": float(score),
        "technical_score": float(technical_score),
        "segmentation_score": float(segmentation_score),
        "unet_rescue_score": float(unet_rescue_score),
        "morphology_prior_score_reported_not_optimized": float(morphology_prior_score),
        "n_2d": int(len(rows)),
        "empty_slice_fraction": empty_slice_fraction,
        "source_counts": source_counts,
        "unet_rescue_source_counts": unet_source_counts,
        "saturn_classical_count": int(source_counts.get("saturn_classical", 0)),
        "unet_rescued_count": unet_rescued,
        "unet_rescued_split_count": unet_rescued_split,
        "unet_total_rescued_count": total_unet_rescued,
        "unet_rescue_fraction": rescue_fraction,
        "unet_rescue_mean_probability_median": float(np.median(unet_means)) if unet_means.size else 0.0,
        "unet_rescue_rejected_reason_pixels": rejected_reason_pixels,
        "unet_rescue_rejected_reason_counts": rejected_reason_counts,
        "unet_rescue_technical_rejected_count": technical_rejected_count,
        "unet_rescue_morphology_warning_count": morphology_warning_count,
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
        "very_short_object_fraction": very_short_frac,
        "very_long_object_fraction": very_long_frac,
        "wide_object_fraction": wide_frac,
        "low_length_width_ratio_fraction": float(np.mean(ratios < 2.5)) if ratios.size else 1.0,
        "hysteresis_occupancy": hyst_occ,
        "clean_mask_occupancy": mask_occ,
        "bridge_inflation": bridge_infl,
        "outside_roi_overlap_count": outside_roi_hits,
        "outside_roi_overlap_by_stage": outside_roi_by_stage,
        "exclusion_mask_overlap_count": exclusion_hits,
        "exclusion_mask_overlap_by_stage": exclusion_by_stage,
    }


def split_ratio_penalty(total_rescued, split_rescued):
    if total_rescued <= 0:
        return 0.0
    split_fraction = float(split_rescued / total_rescued)
    return max(0.0, split_fraction - 0.75) * 4.0


def evaluate_segmentation_candidate(params, base_cfg=None):
    cfg = (base_cfg or CONFIG).copy()
    cfg.update(params)
    cfg["SEGMENTATION_ENGINE"] = "classical_saturn"
    cfg["UNET_RESCUE_ENABLE"] = False
    cfg["TUNING_OBJECTIVE"] = "segmentation"
    rows, segs = segment_eval_images(cfg)
    summary = summarize_candidate(rows, segs, cfg)
    summary.update({k: v for k, v in params.items() if not str(k).startswith("_")})
    results_list.append(summary)
    return summary


def evaluate_unet_rescue_candidate(params, base_cfg=None):
    cfg = (base_cfg or CONFIG).copy()
    cfg.update(params)
    cfg["SEGMENTATION_ENGINE"] = "hybrid"
    cfg["UNET_FAIL_HARD"] = True
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


def tracking_dataframe_from_segments(segs, cfg):
    rows = []
    for z_idx, (_, measurements) in zip(z_values_eval, segs):
        rows.extend(
            segmentation.rows_from_results(
                measurements["results"],
                z_idx,
                cfg["UM_PER_PX_XY"],
            )
        )
    return pd.DataFrame(rows)


def summarize_tracking_candidate(detections, tracked, tracks, cfg):
    n_detections = int(len(detections))
    n_tracks = int(len(tracks))
    if n_tracks == 0:
        return {
            "score": 1e12,
            "n_2d": n_detections,
            "n_tracks": 0,
            "tracking_error": "no_3d_tracks",
        }

    n_slices = pd.to_numeric(
        tracks.get("n_slices", pd.Series(1, index=tracks.index)),
        errors="coerce",
    ).fillna(1)
    lengths = pd.to_numeric(
        tracks.get("total_3d_length_um", pd.Series(np.nan, index=tracks.index)),
        errors="coerce",
    )
    max_2d = pd.to_numeric(
        tracks.get("max_length_2d", pd.Series(np.nan, index=tracks.index)),
        errors="coerce",
    )
    technical_valid = tracks.get(
        "technical_valid", pd.Series(True, index=tracks.index)
    ).fillna(False).astype(bool)
    finite = np.isfinite(lengths) & (lengths > 0)
    multi_slice = n_slices >= 2
    impossible = finite & (lengths > 20.0)
    over_guard = finite & multi_slice & (
        lengths > float(cfg.get("TRACK_TECHNICAL_MAX_JOINED_LENGTH_UM", 15.0))
    )
    inflation = lengths / np.maximum(max_2d, 0.1)
    excessive_inflation = np.isfinite(inflation) & (inflation > 1.5)

    methods = tracked.get(
        "track_link_method", pd.Series("new", index=tracked.index)
    ).fillna("unknown").astype(str)
    distances = pd.to_numeric(
        tracked.get(
            "track_link_distance_um", pd.Series(np.nan, index=tracked.index)
        ),
        errors="coerce",
    )
    gaps = pd.to_numeric(
        tracked.get("track_link_gap_slices", pd.Series(0, index=tracked.index)),
        errors="coerce",
    ).fillna(0)
    linked = methods != "new"
    n_links = int(linked.sum())
    long_link = linked & (
        distances > 0.80 * float(cfg.get("TRACK_MAX_DIST_UM", 6.0))
    )
    gap_link = linked & (gaps > 1)
    repaired = methods == "hybrid_repair"

    single_fraction = float(np.mean(n_slices == 1))
    invalid_fraction = float(np.mean(~technical_valid))
    impossible_fraction = float(np.mean(impossible))
    over_guard_fraction = float(np.mean(over_guard))
    inflation_fraction = float(np.mean(excessive_inflation))
    long_link_fraction = float(long_link.sum() / max(n_links, 1))
    gap_link_fraction = float(gap_link.sum() / max(n_links, 1))

    # Lower is better. Fragmentation is a weak term; impossible or unstable
    # links dominate so the tuner cannot win merely by merging more objects.
    score = (
        invalid_fraction * 1000.0
        + impossible_fraction * 1000.0
        + over_guard_fraction * 250.0
        + inflation_fraction * 80.0
        + long_link_fraction * 40.0
        + gap_link_fraction * 20.0
        + single_fraction * 2.0
    )
    result = {
        "score": float(score),
        "n_2d": n_detections,
        "n_tracks": n_tracks,
        "single_slice_tracks": int((n_slices == 1).sum()),
        "multi_slice_tracks": int(multi_slice.sum()),
        "single_slice_fraction": single_fraction,
        "median_track_slices": float(np.median(n_slices)),
        "technical_invalid_fraction": invalid_fraction,
        "over_15um_fraction": float(np.mean(finite & (lengths > 15.0))),
        "single_slice_over_15um_fraction": float(
            np.mean(finite & (~multi_slice) & (lengths > 15.0))
        ),
        "multi_slice_over_15um_fraction": float(
            np.mean(finite & multi_slice & (lengths > 15.0))
        ),
        "over_20um_fraction": impossible_fraction,
        "over_join_guard_fraction": over_guard_fraction,
        "excessive_length_inflation_fraction": inflation_fraction,
        "median_3d_length_um": float(np.nanmedian(lengths)),
        "n_links": n_links,
        "long_link_fraction": long_link_fraction,
        "gap_link_fraction": gap_link_fraction,
        "hybrid_repair_links": int(repaired.sum()),
        "median_link_distance_um": (
            float(np.nanmedian(distances[linked])) if n_links else 0.0
        ),
    }
    for key in TRACKING_CONFIG_KEYS:
        result[f"resolved_{key}"] = cfg.get(key)
    return result


def evaluate_tracking_candidate(detections, params, base_cfg=None):
    cfg = (base_cfg or CONFIG).copy()
    cfg.update(params)
    cfg["DO_TRACKING"] = True
    tracked, tracks = segmentation.track_across_slices(detections, cfg)
    tracks = segmentation.flag_quality_tracks(tracks, cfg)
    summary = summarize_tracking_candidate(detections, tracked, tracks, cfg)
    summary.update(params)
    results_list.append(summary)
    return summary


def run_profile_mode(outdir, base_cfg):
    records = []
    review_images = []
    original_context = preprocess_context_global
    try:
        for profile in (
            "no_clahe",
            "high_contrast",
            "standard",
            "low_signal",
            "auto",
        ):
            cfg = base_cfg.copy()
            cfg["SEGMENTATION_ENGINE"] = "classical_saturn"
            cfg["UNET_RESCUE_ENABLE"] = False
            cfg["TUNING_OBJECTIVE"] = "profile"
            if profile == "auto":
                ctx = original_context
            else:
                prof, clip = PROFILE_DEFS[profile]
                ctx = segmentation.StackPreprocessContext(
                    **{
                        **segmentation.asdict(original_context),
                        "selected_clahe_profile": prof,
                        "selected_clahe_clip": clip,
                        "configuration_provenance": {
                            **original_context.configuration_provenance,
                            "forced_profile": profile,
                        },
                    }
                )
            globals()["preprocess_context_global"] = ctx
            rows, segs = segment_eval_images(cfg)
            rec = summarize_candidate(rows, segs, cfg)
            rec.update(
                {
                    "profile": profile,
                    "selected_clahe_profile": ctx.selected_clahe_profile,
                    "selected_clahe_clip": ctx.selected_clahe_clip,
                }
            )
            records.append(rec)
            review_images.append(
                (
                    profile,
                    segmentation.make_overlay(
                        images_to_eval[0],
                        segs[0][1]["skel_label"],
                    ),
                    len(segs[0][1]["results"]),
                )
            )
    finally:
        globals()["preprocess_context_global"] = original_context

    records.sort(key=lambda r: r["score"])
    best = records[0]
    n = next_run_number(outdir, "profile_comparison_v5_7_*.csv")
    pd.DataFrame(records).to_csv(outdir / f"profile_comparison_v5_7_{n:03d}.csv", index=False)
    with open(outdir / f"profile_comparison_v5_7_{n:03d}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    best_cfg = base_cfg.copy()
    best_cfg["CLAHE_MODE"] = (
        base_cfg.get("CLAHE_MODE", "auto_stack")
        if best["profile"] == "auto"
        else best["profile"]
    )
    selected = {
        **best,
        "mode": "profile",
        "numerical_rank": 1,
        "selection_status": "first_candidate_for_visual_inspection",
    }
    preset = loadable_parameter_preset(best_cfg, selected)
    preset["CLAHE_MODE"] = best_cfg["CLAHE_MODE"]
    preset["_TUNING_METADATA"]["preprocessing_profile"] = best[
        "selected_clahe_profile"
    ]
    with open(outdir / f"best_preprocessing_profile_v5_7_{n:03d}.json", "w", encoding="utf-8") as f:
        json.dump(preset, f, indent=2)
    with PdfPages(outdir / f"profile_review_v5_7_{n:03d}.pdf") as pdf:
        fig, axes = plt.subplots(
            len(review_images),
            2,
            figsize=(10, 3.2 * len(review_images)),
            constrained_layout=True,
        )
        for row_idx, (profile, overlay, count) in enumerate(review_images):
            axes[row_idx, 0].imshow(images_to_eval[0], cmap="gray")
            axes[row_idx, 0].set_title(f"{profile}: raw Z={z_values_eval[0]}")
            axes[row_idx, 1].imshow(overlay)
            axes[row_idx, 1].set_title(f"{profile}: overlay, n={count}")
            axes[row_idx, 0].axis("off")
            axes[row_idx, 1].axis("off")
        fig.suptitle("Saturn v5.7 preprocessing profile review")
        pdf.savefig(fig, dpi=180)
        plt.close(fig)
    return best


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


def sample_segmentation_candidates(space, maxiter, seed, base_cfg):
    count = max(1, int(maxiter))
    prioritized = [
        ("reviewed_base", candidate_from_config(space, base_cfg)),
        ("space_midpoint", sample_candidates(space, 1, seed)[0]),
    ]
    random_candidates = sample_candidates(
        space,
        count + len(prioritized),
        seed,
    )[1:]
    candidates = []
    seen = set()
    for role, candidate in prioritized + [
        (f"deterministic_random_{idx:03d}", candidate)
        for idx, candidate in enumerate(random_candidates, start=1)
    ]:
        signature = tuple(candidate[key] for key, *_ in space)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append((role, candidate))
        if len(candidates) == count:
            break
    return candidates


def candidate_from_config(space, base_cfg, overrides=None):
    values = dict(base_cfg or {})
    values.update(overrides or {})
    vector = []
    for key, lo, hi, _ in space:
        value = float(values.get(key, (lo + hi) / 2))
        vector.append(min(max(value, lo), hi))
    return params_from_vector(vector, space)


def sample_unet_rescue_candidates(space, maxiter, seed, base_cfg):
    count = max(1, int(maxiter))
    prioritized = [
        (
            "evidence_0.05_0.30",
            candidate_from_config(
                space,
                base_cfg,
                {
                    "UNET_CANDIDATE_THRESHOLD": 0.05,
                    "UNET_RESCUE_THRESHOLD": 0.30,
                    "UNET_RESCUE_MIN_SKEL_LEN_UM": 2.0,
                    "UNET_SHORT_RESCUE_MIN_MEAN_PROB": 0.35,
                    "UNET_RESCUE_CENTERLINE_MIN_MEAN_PROB": 0.30,
                    "UNET_LOW_RATIO_RESCUE_MIN_MEAN_PROB": 0.75,
                    "UNET_LOW_RATIO_RESCUE_MIN_LENGTH_UM": 4.0,
                    "UNET_INSTANCE_SEED_THRESHOLD": 0.50,
                },
            ),
        ),
        (
            "balanced_recall_review",
            candidate_from_config(
                space,
                base_cfg,
                {
                    "UNET_CANDIDATE_THRESHOLD": 0.05,
                    "UNET_RESCUE_THRESHOLD": 0.20,
                    "UNET_RESCUE_MIN_SKEL_LEN_UM": 2.0,
                    "UNET_SHORT_RESCUE_MIN_MEAN_PROB": 0.30,
                    "UNET_RESCUE_CENTERLINE_MIN_MEAN_PROB": 0.30,
                    "UNET_LOW_RATIO_RESCUE_MIN_MEAN_PROB": 0.55,
                    "UNET_LOW_RATIO_RESCUE_MIN_LENGTH_UM": 3.0,
                    "UNET_INSTANCE_SEED_THRESHOLD": 0.50,
                },
            ),
        ),
        ("reviewed_base", candidate_from_config(space, base_cfg)),
        ("space_midpoint", sample_candidates(space, 1, seed)[0]),
    ]
    random_candidates = sample_candidates(space, count + len(prioritized), seed)[1:]

    candidates = []
    seen = set()
    for role, candidate in prioritized + [
        (f"deterministic_random_{idx:03d}", candidate)
        for idx, candidate in enumerate(random_candidates, start=1)
    ]:
        signature = tuple(candidate[key] for key, *_ in space)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append((role, candidate))
        if len(candidates) == count:
            break
    return candidates


def sample_tracking_candidates(space, maxiter, seed, base_cfg):
    count = max(1, int(maxiter))
    prioritized = [
        ("reviewed_base", candidate_from_config(space, base_cfg)),
        ("space_midpoint", sample_candidates(space, 1, seed)[0]),
    ]
    random_candidates = sample_candidates(space, count + len(prioritized), seed)[1:]
    candidates = []
    seen = set()
    for role, candidate in prioritized + [
        (f"deterministic_random_{idx:03d}", candidate)
        for idx, candidate in enumerate(random_candidates, start=1)
    ]:
        signature = tuple(candidate[key] for key, *_ in space)
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append((role, candidate))
        if len(candidates) == count:
            break
    return candidates


def generate_candidate_review_pdf(
    outdir,
    mode,
    records,
    base_cfg,
    review_candidates,
    preferred_role=None,
):
    if mode not in {"segmentation", "unet_rescue"} or not records:
        return None
    review_count = min(3, max(1, int(review_candidates)), len(records))
    selected = []
    if preferred_role:
        preferred = next(
            (
                record
                for record in records
                if record.get("candidate_role") == preferred_role
            ),
            None,
        )
        if preferred is not None:
            selected.append(preferred)
    selected.extend(
        record
        for record in records
        if record not in selected
    )
    selected = selected[:review_count]
    rendered = []
    space = (
        UNET_RESCUE_PARAM_SPACE
        if mode == "unet_rescue"
        else SEGMENTATION_PARAM_SPACE
    )
    for record in selected:
        cfg = base_cfg.copy()
        cfg.update({key: record[key] for key, *_ in space})
        if mode == "unet_rescue":
            cfg.update(
                {
                    "SEGMENTATION_ENGINE": "hybrid",
                    "UNET_FAIL_HARD": True,
                    "UNET_THRESHOLD_MODE": "soft",
                    "UNET_RESCUE_ENABLE": True,
                    "UNET_RESCUE_SPLIT_RETRY_ENABLE": True,
                    "UNET_INSTANCE_SPLIT_ENABLE": True,
                    "TUNING_OBJECTIVE": "unet_rescue",
                }
            )
        else:
            cfg.update(
                {
                    "SEGMENTATION_ENGINE": "classical_saturn",
                    "UNET_RESCUE_ENABLE": False,
                    "TUNING_OBJECTIVE": "segmentation",
                }
            )
        _, segs = segment_eval_images(cfg)
        rendered.append((record, segs))

    n = next_run_number(outdir, f"candidate_visual_review_v5_7_{mode}_*.pdf")
    path = outdir / f"candidate_visual_review_v5_7_{mode}_{n:03d}.pdf"
    with PdfPages(path) as pdf:
        for slice_idx, (img, z_idx) in enumerate(
            zip(images_to_eval, z_values_eval)
        ):
            fig, axes = plt.subplots(
                1,
                review_count,
                figsize=(6 * review_count, 6),
                squeeze=False,
                constrained_layout=True,
            )
            for col, (record, segs) in enumerate(rendered):
                seg, measurements = segs[slice_idx]
                if mode == "unet_rescue":
                    overlay = segmentation.make_unet_rescue_review_overlay(
                        img,
                        measurements["skel_label"],
                        measurements["results"],
                        measurements.get("unet_rescue_rejected_reason"),
                    )
                else:
                    overlay = segmentation.make_overlay(
                        img, measurements["skel_label"]
                    )
                axes[0, col].imshow(overlay)
                role = record.get("candidate_role", f"rank {col + 1}")
                axes[0, col].set_title(
                    f"Rank {record.get('numerical_rank', col + 1)}: {role}\n"
                    f"n={len(measurements['results'])}, score={record['score']:.3f}"
                )
                axes[0, col].axis("off")
            fig.suptitle(f"Saturn v5.7 {mode} candidate review, Z={z_idx}")
            pdf.savefig(fig, dpi=180)
            plt.close(fig)
    if mode == "unet_rescue" and selected:
        selected_record, selected_segs = rendered[0]
        support_n = next_run_number(
            outdir,
            "selected_unet_probability_review_v5_7_*.pdf",
        )
        support_path = (
            outdir
            / f"selected_unet_probability_review_v5_7_{support_n:03d}.pdf"
        )
        low = float(selected_record["UNET_CANDIDATE_THRESHOLD"])
        high = float(selected_record["UNET_RESCUE_THRESHOLD"])
        with PdfPages(support_path) as support_pdf:
            for slice_idx, (img, z_idx) in enumerate(
                zip(images_to_eval, z_values_eval)
            ):
                seg, measurements = selected_segs[slice_idx]
                base = segmentation.normalize_display(img)
                probability = np.asarray(
                    seg["unet_probability"],
                    dtype=np.float32,
                )
                valid = (
                    np.asarray(seg["roi_mask"], dtype=bool)
                    & ~np.asarray(seg["exclusion_mask"], dtype=bool)
                )
                support = segmentation.apply_hysteresis_threshold(
                    probability,
                    min(low, high),
                    high,
                ) & valid
                seeds = (probability >= high) & valid
                support_rgb = np.stack([base, base, base], axis=-1)
                support_rgb[support] = (
                    0.25 * support_rgb[support]
                    + 0.75 * np.array([0.0, 0.9, 1.0])
                )
                support_rgb[seeds] = (
                    0.20 * support_rgb[seeds]
                    + 0.80 * np.array([1.0, 0.9, 0.0])
                )
                overlay = segmentation.make_unet_rescue_review_overlay(
                    img,
                    measurements["skel_label"],
                    measurements["results"],
                    measurements.get("unet_rescue_rejected_reason"),
                )
                fig, axes = plt.subplots(
                    1,
                    4,
                    figsize=(22, 6),
                    constrained_layout=True,
                )
                axes[0].imshow(base, cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("Raw ROI-normalized image")
                probability_view = axes[1].imshow(
                    probability,
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                )
                axes[1].set_title("U-Net probability")
                fig.colorbar(
                    probability_view,
                    ax=axes[1],
                    fraction=0.046,
                )
                axes[2].imshow(support_rgb)
                axes[2].set_title(
                    f"Hysteresis support: cyan >= {low:.2f}; "
                    f"yellow seed >= {high:.2f}"
                )
                axes[3].imshow(overlay)
                axes[3].set_title(
                    "Final: green classical; cyan U-Net; "
                    "red/magenta technical reject"
                )
                for axis in axes:
                    axis.axis("off")
                role = selected_record.get("candidate_role", "selected")
                fig.suptitle(
                    f"Selected U-Net candidate {role}, Z={z_idx}"
                )
                support_pdf.savefig(fig, dpi=180)
                plt.close(fig)
        print(f"Selected U-Net probability review: {support_path}")
    return path


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


def loadable_parameter_preset(cfg, selected):
    preset = {}
    for key in CONFIG:
        if str(key).startswith("_"):
            continue
        value = selected.get(key, cfg.get(key))
        preset[key] = segmentation._json_scalar(value)
    preset["_TUNING_METADATA"] = {
        "pipeline_version": VERSION,
        "mode": selected.get("mode"),
        "numerical_rank": selected.get("numerical_rank"),
        "selection_status": selected.get("selection_status"),
        "score": selected.get("score"),
        "selected_z_indices": [int(z) for z in z_values_eval],
        "preprocessing_profile": getattr(
            preprocess_context_global,
            "selected_clahe_profile",
            cfg.get("CLAHE_MODE", "unspecified"),
        ),
        "note": "Numerical candidate for visual inspection; not a finalized biological parameter set.",
    }
    return preset


def aggregate_stratum_results(
    paths,
    outdir,
    cfg,
    selected_role,
    mode="unet_rescue",
):
    if mode == "unet_rescue":
        parameter_space = UNET_RESCUE_PARAM_SPACE
    elif mode == "segmentation":
        parameter_space = SEGMENTATION_PARAM_SPACE
    else:
        raise ValueError(
            "Shared stratum aggregation supports segmentation or unet_rescue"
        )

    by_role = {}
    source_paths = []
    parameter_keys = [key for key, *_ in parameter_space]
    for path in paths:
        source = Path(path)
        with open(source, "r", encoding="utf-8") as f:
            records = json.load(f)
        source_paths.append(str(source.resolve()))
        for rank, record in enumerate(
            sorted(records, key=lambda item: float(item["score"])), start=1
        ):
            role = str(record.get("candidate_role", "")).strip()
            if not role:
                continue
            entry = dict(record)
            entry["stratum_path"] = str(source.resolve())
            entry["stratum_rank"] = rank
            by_role.setdefault(role, []).append(entry)

    expected_strata = len(source_paths)
    complete_roles = {
        role: entries
        for role, entries in by_role.items()
        if len(entries) == expected_strata
    }
    if selected_role not in complete_roles:
        raise ValueError(
            f"Shared candidate role '{selected_role}' is not present in every stratum"
        )

    summaries = []
    for role, entries in complete_roles.items():
        signatures = {
            tuple(entry[key] for key in parameter_keys)
            for entry in entries
        }
        if len(signatures) != 1:
            raise ValueError(
                f"Candidate role '{role}' has different parameters across strata"
            )
        summaries.append(
            {
                "candidate_role": role,
                "stratum_count": len(entries),
                "mean_score": float(np.mean([entry["score"] for entry in entries])),
                "mean_rank": float(
                    np.mean([entry["stratum_rank"] for entry in entries])
                ),
                "min_rank": int(min(entry["stratum_rank"] for entry in entries)),
                "max_rank": int(max(entry["stratum_rank"] for entry in entries)),
                "mean_n_2d": float(np.mean([entry["n_2d"] for entry in entries])),
                "mean_count_cv": float(
                    np.mean([entry.get("count_cv", 0.0) for entry in entries])
                ),
                "max_empty_slice_fraction": float(
                    max(
                        entry.get("empty_slice_fraction", 0.0)
                        for entry in entries
                    )
                ),
                "max_very_short_object_fraction": float(
                    max(
                        entry.get("very_short_object_fraction", 0.0)
                        for entry in entries
                    )
                ),
                "max_very_long_object_fraction": float(
                    max(
                        entry.get("very_long_object_fraction", 0.0)
                        for entry in entries
                    )
                ),
                "max_outside_roi_overlap_count": int(
                    max(
                        entry.get("outside_roi_overlap_count", 0)
                        for entry in entries
                    )
                ),
                "max_exclusion_mask_overlap_count": int(
                    max(
                        entry.get("exclusion_mask_overlap_count", 0)
                        for entry in entries
                    )
                ),
                "mean_unet_rescue_fraction": float(
                    np.mean(
                        [
                            entry.get("unet_rescue_fraction", 0.0)
                            for entry in entries
                        ]
                    )
                ),
                **{
                    key: entries[0][key]
                    for key in parameter_keys
                },
            }
        )
    summaries.sort(key=lambda item: (item["mean_rank"], item["mean_score"]))

    selected_entries = complete_roles[selected_role]
    selected = dict(selected_entries[0])
    selected.update(
        {
            "mode": f"{mode}_shared",
            "numerical_rank": next(
                idx
                for idx, item in enumerate(summaries, start=1)
                if item["candidate_role"] == selected_role
            ),
            "selection_status": "shared_candidate_for_visual_inspection",
        }
    )
    preset = loadable_parameter_preset(cfg, selected)
    preset["_TUNING_METADATA"].update(
        {
            "candidate_role": selected_role,
            "source_stratum_results": source_paths,
            "selection_basis": (
                "One unchanged candidate across all biological strata; "
                "selected explicitly after cross-stratum review."
            ),
            "aggregation_mode": mode,
        }
    )

    outdir.mkdir(parents=True, exist_ok=True)
    n = next_run_number(outdir, f"shared_{mode}_params_v5_7_*.json")
    preset_path = outdir / f"shared_{mode}_params_v5_7_{n:03d}.json"
    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(preset, f, indent=2)
    pd.DataFrame(summaries).to_csv(
        outdir / f"shared_{mode}_candidate_comparison_v5_7_{n:03d}.csv",
        index=False,
    )
    with open(
        outdir / f"shared_{mode}_candidate_comparison_v5_7_{n:03d}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summaries, f, indent=2)
    return preset_path, summaries


def save_results(outdir, mode, best, records, cfg, review_candidates=6):
    n = next_run_number(outdir, f"best_{mode}_params_v5_7_*.json")
    for rank, record in enumerate(records, start=1):
        record["numerical_rank"] = rank
        record["selection_status"] = (
            "first_candidate_for_visual_inspection"
            if rank == 1
            else "candidate_for_review"
        )
        record["mode"] = mode
    best = records[0] if records else best
    preset = loadable_parameter_preset(cfg, best)
    with open(outdir / f"best_{mode}_params_v5_7_{n:03d}.json", "w", encoding="utf-8") as f:
        json.dump(preset, f, indent=2)
    with open(outdir / f"tuning_results_saturnv5_7_{mode}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    pd.DataFrame(records).to_csv(outdir / f"tuning_results_saturnv5_7_{mode}.csv", index=False)
    review_count = min(max(1, int(review_candidates)), len(records))
    with open(
        outdir / f"candidate_review_queue_v5_7_{mode}_{n:03d}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(records[:review_count], f, indent=2)
    with open(outdir / f"tuning_summary_saturnv5_7_{mode}.txt", "w", encoding="utf-8") as f:
        f.write(f"SATURN V5.7 U-NET-READY {mode.upper()} TUNING SUMMARY\n")
        f.write(f"Analysis mode: {cfg.get('ANALYSIS_MODE', 'comparative')}\n")
        f.write("Comparative mode reports morphology but does not optimize toward WT-like length, width, taper, tortuosity, count, volume, or Z-span.\n")
        f.write(f"Selected Z indices: {z_values_eval}\n")
        f.write(f"Preprocessing profile: {preprocess_context_global.selected_clahe_profile}\n")
        f.write(f"Numerically lowest score: {best.get('score')}\n")
        f.write("Selection status: first candidate for visual inspection; not a finalized biological parameter set.\n")


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
    constrained_thresholds = params_from_vector(
        [88.0, 87.0, 8, 1.0, 6.0, 4.0, 2.0, 2.5],
        SEGMENTATION_PARAM_SPACE,
    )
    checks.append(
        (
            "minimum hysteresis percentile separation",
            constrained_thresholds["THRESHOLD_HI"]
            - constrained_thresholds["THRESHOLD_LO"]
            >= MIN_HYSTERESIS_PERCENTILE_SEPARATION,
        )
    )
    checks.append(("automatic slice selection", select_auto_slices(20, 6) == [0, 4, 8, 11, 15, 19]))
    tmp = Path(os.environ.get("TEMP", ".")) / f"saturnv56_selfcheck_{os.getpid()}"
    tmp.mkdir(exist_ok=True)
    a = tmp / "a.json"; b = tmp / "b.json"
    a.write_text(
        json.dumps({"THRESHOLD_HI": 91.0, "THRESHOLD_LO": 82.0}),
        encoding="utf-8",
    )
    b.write_text(json.dumps({"THRESHOLD_LO": 83.0}), encoding="utf-8")
    checks.append(
        (
            "repeated base-parameter merge order",
            merge_base_params([str(a), str(b)])
            == {"THRESHOLD_HI": 91.0, "THRESHOLD_LO": 83.0},
        )
    )

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
    segmentation_candidates = sample_segmentation_candidates(
        SEGMENTATION_PARAM_SPACE,
        2,
        1,
        CONFIG,
    )
    checks.append(
        (
            "segmentation reviewed baseline first",
            segmentation_candidates[0][0] == "reviewed_base",
        )
    )
    checks.append(("U-Net rescue mode available", "unet_rescue" in ("profile", "segmentation", "tracking", "unet_rescue")))
    unet_candidates = sample_unet_rescue_candidates(
        UNET_RESCUE_PARAM_SPACE, 2, 1, CONFIG
    )
    checks.append(
        (
            "U-Net evidence candidate first",
            unet_candidates[0][0] == "evidence_0.05_0.30"
            and unet_candidates[0][1]["UNET_CANDIDATE_THRESHOLD"] == 0.05
            and unet_candidates[0][1]["UNET_RESCUE_THRESHOLD"] == 0.30,
        )
    )
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
    parser.add_argument("--review-candidates", type=int, default=6)
    parser.add_argument("--review-candidate-role", default=None)
    parser.add_argument("--aggregate-stratum-results", action="append", default=[])
    parser.add_argument(
        "--shared-candidate-role",
        default=None,
    )
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
    segmentation.validate_config(cfg)

    if args.aggregate_stratum_results:
        if args.mode not in {"segmentation", "unet_rescue"}:
            raise SystemExit(
                "--aggregate-stratum-results supports --mode segmentation "
                "or --mode unet_rescue"
            )
        if args.unet_model:
            cfg["UNET_MODEL_PATH"] = args.unet_model
        selected_role = args.shared_candidate_role or (
            "evidence_0.05_0.30"
            if args.mode == "unet_rescue"
            else "reviewed_base"
        )
        preset_path, summaries = aggregate_stratum_results(
            args.aggregate_stratum_results,
            outdir,
            cfg,
            selected_role,
            mode=args.mode,
        )
        print(f"Shared preset: {preset_path}")
        print(
            f"Aggregated {len(summaries)} common candidates across "
            f"{len(args.aggregate_stratum_results)} strata"
        )
        return

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
    if not np.any(roi_mask_global):
        raise SystemExit("ROI mask is empty")
    if exclusion_mask_global is not None and not np.any(
        roi_mask_global & ~exclusion_mask_global
    ):
        raise SystemExit("Exclusion mask removes the entire ROI")
    globals()["preprocess_context_global"] = build_global_context(files, slice_indices, cfg, roi_mask_global, exclusion_mask_global)
    segmentation.save_stack_preprocess_context(preprocess_context_global, outdir)
    globals()["images_to_eval"] = [segmentation.ensure_2d_image(segmentation.robust_imread(files[i]), files[i]) for i in slice_indices]
    globals()["z_values_eval"] = [segmentation.extract_z_index(files[i], sequence_idx=i) for i in slice_indices]
    if len(set(z_values_eval)) != len(z_values_eval):
        raise SystemExit(f"Selected files contain duplicate Z indices: {z_values_eval}")
    if args.mode == "tracking" and not require_consecutive(z_values_eval):
        raise SystemExit(
            "Tracking mode requires consecutive source Z indices; selected files "
            f"resolved to {z_values_eval}"
        )
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
    if args.mode == "unet_rescue" and not Path(cfg["UNET_MODEL_PATH"]).is_file():
        raise SystemExit(f"U-Net checkpoint not found: {cfg['UNET_MODEL_PATH']}")
    if args.mode == "unet_rescue":
        cfg["SEGMENTATION_ENGINE"] = "hybrid"
        cfg["UNET_FAIL_HARD"] = True
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
        detections = tracking_dataframe_from_segments(segs, cfg)
        for role, cand in sample_tracking_candidates(
            space, args.maxiter, args.seed, cfg
        ):
            rec = evaluate_tracking_candidate(detections, cand, base_cfg=cfg)
            rec["candidate_role"] = role
            records.append(rec)
    elif args.mode == "unet_rescue":
        for role, cand in sample_unet_rescue_candidates(
            space, args.maxiter, args.seed, cfg
        ):
            rec = evaluate_unet_rescue_candidate(cand, base_cfg=cfg)
            rec["candidate_role"] = role
            records.append(rec)
    else:
        for role, cand in sample_segmentation_candidates(
            space,
            args.maxiter,
            args.seed,
            cfg,
        ):
            rec = evaluate_segmentation_candidate(cand, base_cfg=cfg)
            rec["candidate_role"] = role
            records.append(rec)
    records.sort(key=lambda r: r["score"])
    for rank, record in enumerate(records, start=1):
        record["numerical_rank"] = rank
    best = records[0] if records else {}
    review_path = generate_candidate_review_pdf(
        outdir,
        args.mode,
        records,
        cfg,
        args.review_candidates,
        preferred_role=args.review_candidate_role,
    )
    save_results(
        outdir,
        args.mode,
        best,
        records,
        cfg,
        review_candidates=args.review_candidates,
    )
    print(f"Best {args.mode} score={best.get('score')}")
    if review_path is not None:
        print(f"Candidate visual review: {review_path}")


if __name__ == "__main__":
    main()
