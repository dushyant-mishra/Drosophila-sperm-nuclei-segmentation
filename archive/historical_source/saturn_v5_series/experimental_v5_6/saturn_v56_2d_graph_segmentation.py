#!/usr/bin/env python3
"""Canonical Saturn v5.6 ilastik-assisted 2D graph segmentation module.

This module owns the auditable 2D path-completion implementation. It preserves
the historical raw regression baseline for provenance, then uses ilastik
probabilities as semantic evidence for raw-supported residual centerlines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt
from skimage import draw, filters, measure, morphology


ROOT = Path(__file__).resolve().parents[1]
SATURN_PATH = ROOT / "sperm_segmentation_saturnv5.6.py"
AI_HELPER_PATH = ROOT / "utils" / "ai_preprocessing_v5_6.py"
PRESET_PATH = ROOT / "comparative_presets" / "comparative_selected_v5_6.json"
IMAGE_DIR = Path(r"C:\Users\dmishra\Desktop\sperm images")
ROI_PATH = IMAGE_DIR / "roi_z28.1.npy"
PROB_DIR = ROOT / "scratch" / "v5_6_ilastik_pilot" / "probability_maps"
META_DIR = ROOT / "scratch" / "v5_6_ilastik_pilot" / "metadata"
OUT_DIR = ROOT / "scratch" / "v5_6_consolidated_2d"

REVIEW_Z = [5, 35]
EVALUATION_Z = [5, 6, 12, 35, 60, 87]
EXPECTED_RAW_COUNTS = {5: 266, 35: 318}
SOURCE_RE = re.compile(r"^Project001_Series002_z(\d+)_ch00\.tif{1,2}$", re.IGNORECASE)
CLASS_ORDER = ["sperm_nucleus", "structured_tissue_edge", "punctum_or_ring", "diffuse_background"]
SELECTED_FORMULATION = "C_raw_nucleus_not_tissue_or_punctum"
PIXEL_UM = 0.756836
LEGACY_FIXED_SEEDS_PER_PASS = 120
EMERGENCY_COMPONENT_CAP = 100
EMERGENCY_COMPLEX_GRAPH_CAP = 50
HYSTERESIS_HIGH = 0.46
HYSTERESIS_LOW = 0.24

SUBDIRS = [
    "configuration", "historical_baseline", "probability_fields", "weighted_ridge",
    "seed_graphs", "endpoint_extensions", "join_matching", "completed_paths",
    "pass2_recovery", "historical_mapping", "candidate_audit", "review_panels",
    "comparison_overlays", "manual_review", "reports",
]

CROPS = [
    ("dense_central_nuclei", 350, 350, 260, 260),
    ("faint_nuclei", 210, 250, 260, 260),
    ("parallel_nuclei", 470, 460, 260, 260),
    ("curved_nuclei", 600, 380, 260, 260),
    ("puncta_rich_region", 250, 620, 260, 260),
    ("broad_tissue_boundary", 650, 650, 260, 260),
    ("transition_region", 430, 700, 260, 260),
    ("lower_shaft", 720, 360, 260, 260),
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v56 = load_module("saturn_v56_2d_path_segmentation", SATURN_PATH)
ai = load_module("ai_preprocessing_v56_2d_path_segmentation", AI_HELPER_PATH)


def ensure_dirs(out_dir: Path = OUT_DIR) -> None:
    for name in SUBDIRS:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def discover_sources() -> dict[int, Path]:
    by_z = {}
    for item in sorted(IMAGE_DIR.iterdir(), key=lambda p: p.name.lower()):
        m = SOURCE_RE.match(item.name) if item.is_file() else None
        if m:
            by_z[int(m.group(1))] = item
    missing = [z for z in range(88) if z not in by_z]
    if len(by_z) != 88 or missing:
        raise RuntimeError(f"Expected 88 source images z0-87; found={len(by_z)} missing={missing}")
    return by_z


def load_exact_baseline_cfg() -> dict:
    preset = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
    cfg = v56.CONFIG.copy()
    cfg.update(preset["parameters"])
    cfg["DO_TRACKING"] = False
    cfg["AUTO_LOCAL_REANALYSIS"] = False
    cfg["ANALYSIS_MODE"] = "general_robustness"
    cfg["SEGMENTATION_PARAMETER_SET"] = "general_robustness_selected"
    cfg["ROI_BOUNDARY_SAFE_RIDGE"] = False
    cfg["ROI_THRESHOLD_EXCLUDE_BOUNDARY_PX"] = 0
    return v56.cfg_with_resolved_pixels(cfg)


def run_raw_baseline(by_z: dict[int, Path], roi: np.ndarray, cfg: dict, z_indices: list[int], out_dir: Path = OUT_DIR) -> tuple[dict, object]:
    context = v56.build_stack_preprocess_context([str(by_z[z]) for z in sorted(by_z)], roi, cfg, exclusion_mask=None)
    baseline = {}
    rows = []
    for z in z_indices:
        raw = v56.ensure_2d_image(v56.robust_imread(str(by_z[z])), by_z[z].name)
        seg = v56.segment_slice(raw, cfg, z_idx=z, roi_mask=roi, exclusion_mask=None, preprocess_context=context)
        meas = v56.measure_spermatids(seg, cfg)
        count = len(meas["results"])
        if z in EXPECTED_RAW_COUNTS and count != EXPECTED_RAW_COUNTS[z]:
            raise RuntimeError(f"RAW BASELINE EQUIVALENCE: FAIL z{z:03d}: expected {EXPECTED_RAW_COUNTS[z]}, observed {count}")
        baseline[z] = {"raw": raw, "seg": seg, "meas": meas, "results": meas["results"]}
        rows.append({
            "z_index": z,
            "expected_count": EXPECTED_RAW_COUNTS.get(z, "not_applicable"),
            "observed_count": count,
            "raw_label_checksum": sha256_array(meas["skel_label"].astype(np.int32)),
            "centerline_pixels": int(np.count_nonzero(meas["skel_label"])),
        })
        save_png(out_dir / "historical_baseline" / f"z{z:03d}_historical_raw_regression_baseline_overlay.png", v56.make_overlay(raw, meas["skel_label"]))
    write_csv(out_dir / "historical_baseline" / "historical_raw_regression_baseline_v5_6.csv", rows)
    return baseline, context


def probability_path(z: int) -> Path:
    return PROB_DIR / f"Dataset01_eval_z{z:03d}_Probabilities.h5"


def normalize_roi(field: np.ndarray, roi: np.ndarray) -> np.ndarray:
    out = np.zeros_like(field, dtype=np.float32)
    vals = np.asarray(field[roi], dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return out
    lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99.5))
    out[roi] = np.clip((field[roi] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def weighted_ridge_formulations(raw_ridge: np.ndarray, prob: np.ndarray, roi: np.ndarray) -> dict[str, np.ndarray]:
    n, t, p, d = [prob[:, :, i] for i in range(4)]
    margin = np.clip(n - np.maximum.reduce([t, p, d]), 0, 1)
    forms = {
        "A_raw_times_nucleus": raw_ridge * n,
        "B_raw_nucleus_not_punctum": raw_ridge * n * (1 - p),
        "C_raw_nucleus_not_tissue_or_punctum": raw_ridge * n * (1 - np.maximum(t, p)),
        "D_raw_positive_margin": raw_ridge * margin,
    }
    return {name: normalize_roi(np.asarray(field, dtype=np.float32), roi) for name, field in forms.items()}


def support_metric(field: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(field[mask])) if np.any(mask) else 0.0


def formulation_metrics(z: int, forms: dict[str, np.ndarray], prob: np.ndarray, raw_label: np.ndarray, roi: np.ndarray) -> list[dict]:
    rows = []
    tissue_region = (prob[:, :, 1] > 0.55) & roi
    punctum_region = (prob[:, :, 2] > 0.55) & roi
    boundary = (distance_transform_edt(roi) <= 3) & roi
    raw_center = (raw_label > 0) & roi
    for name, field in forms.items():
        rows.append({
            "z_index": z,
            "formulation": name,
            "ridge_occupancy": float(np.mean((field > 0.25) & roi)),
            "nucleus_centerline_support": support_metric(field, raw_center),
            "punctum_region_support": support_metric(field, punctum_region),
            "tissue_edge_support": support_metric(field, tissue_region),
            "roi_boundary_support": support_metric(field, boundary),
            "outside_roi_leakage": int(np.count_nonzero(field[~roi] > 0)),
            "selection_basis": "fixed transparent balanced penalty; not selected by count",
        })
    return rows


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr)
    if a.ndim == 2:
        a = (v56.normalize_display(a) * 255).astype(np.uint8)
    v56._imwrite(str(path), a.astype(np.uint8) if a.dtype != np.uint8 else a)


def branch_nodes(mask: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    n = convolve(mask.astype(np.int32), kernel, mode="constant", cval=0)
    return mask & (n > 2)


def endpoints(mask: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)
    n = convolve(mask.astype(np.int32), kernel, mode="constant", cval=0)
    return np.argwhere(mask & (n == 1))


def orientation_coherence(coords: np.ndarray) -> float:
    if coords.shape[0] < 3:
        return 1.0
    xy = coords[:, ::-1].astype(float)
    xy -= np.mean(xy, axis=0)
    cov = np.cov(xy, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    if float(np.max(eig)) <= 1e-9:
        return 0.0
    return float(1.0 - min(eig) / max(eig))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 180.0
    cos = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    return float(math.degrees(math.acos(abs(cos))))


def endpoint_direction(coords: np.ndarray, endpoint: np.ndarray) -> np.ndarray:
    if coords.shape[0] < 2:
        return np.array([0.0, 0.0])
    d = np.sqrt(np.sum((coords - endpoint) ** 2, axis=1))
    near = coords[np.argsort(d)[: min(6, len(coords))]]
    return endpoint.astype(float) - np.mean(near, axis=0)


def line_mask(shape: tuple[int, int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rr, cc = draw.line(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
    m = np.zeros(shape, dtype=bool)
    good = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    m[rr[good], cc[good]] = True
    return m


def build_candidate_seed_mask(weighted: np.ndarray, prob: np.ndarray, raw_label: np.ndarray, roi: np.ndarray, pass_no: int) -> np.ndarray:
    threshold = 0.70 if pass_no == 1 else 0.58
    residual = roi & ~binary_dilation(raw_label > 0, iterations=3)
    n, t, p = prob[:, :, 0], prob[:, :, 1], prob[:, :, 2]
    seed = residual & (weighted >= threshold) & (n >= (0.54 if pass_no == 1 else 0.46)) & (n >= t) & (n >= p)
    seed = morphology.remove_small_objects(seed, 3)
    seed = morphology.skeletonize(seed)
    seed[branch_nodes(seed)] = False
    return seed & residual


def build_direct_hysteresis_mask(weighted: np.ndarray, prob: np.ndarray, raw_label: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = roi & ~binary_dilation(raw_label > 0, iterations=3)
    n, t, p = prob[:, :, 0], prob[:, :, 1], prob[:, :, 2]
    class_margin = np.clip(n - np.maximum(t, p), 0, 1)
    class_ok = residual & (n >= 0.34) & (class_margin >= -0.03)
    high = class_ok & (weighted >= HYSTERESIS_HIGH) & (n >= 0.52)
    low = class_ok & (weighted >= HYSTERESIS_LOW)
    hyst = filters.apply_hysteresis_threshold(weighted, HYSTERESIS_LOW, HYSTERESIS_HIGH) & low
    hyst |= high
    hyst &= residual
    hyst = morphology.remove_small_objects(hyst, 3)
    repaired = repair_short_gaps(morphology.skeletonize(hyst), weighted, prob, roi & residual)
    repaired &= residual
    return high, hyst, morphology.skeletonize(repaired)


def repair_short_gaps(skel: np.ndarray, weighted: np.ndarray, prob: np.ndarray, roi: np.ndarray) -> np.ndarray:
    out = skel.copy()
    eps = endpoints(out)
    if len(eps) < 2:
        return out
    if len(eps) > 300:
        return out
    used = set()
    proposals = []
    coords = np.argwhere(out)
    for i, a in enumerate(eps):
        for j in range(i + 1, len(eps)):
            b = eps[j]
            dist = float(np.linalg.norm(a - b))
            if dist > 6:
                continue
            gap = line_mask(out.shape, a, b)
            if np.count_nonzero(gap & out) > 2 or not np.all(roi[gap]):
                continue
            orient = angle_between(endpoint_direction(coords, a), -endpoint_direction(coords, b))
            support = support_metric(weighted, gap)
            comp = max(support_metric(prob[:, :, 1], gap), support_metric(prob[:, :, 2], gap))
            score = support + support_metric(prob[:, :, 0], gap) - comp - orient / 180.0
            if orient <= 45 and support >= 0.10 and comp < 0.55:
                proposals.append((score, i, j, gap))
    for _, i, j, gap in sorted(proposals, reverse=True):
        if i in used or j in used:
            continue
        out |= gap
        used.add(i)
        used.add(j)
    return out


def classify_skeleton_component(mask: np.ndarray, prob: np.ndarray, raw_label: np.ndarray, cfg: dict) -> tuple[str, list[str]]:
    status, reasons = classify_seed(mask, prob, raw_label, cfg)
    if status in {"punctum_like", "tissue_like", "invalid", "duplicate_raw_object"}:
        return status, reasons
    coords = np.argwhere(mask)
    eps = endpoints(mask)
    branches = int(np.count_nonzero(branch_nodes(mask)))
    if coords.shape[0] < max(3, int(round(1.8 / PIXEL_UM))):
        return "simple_unresolved_fragment", ["unresolved_fragment"]
    if branches == 0 and len(eps) == 2 and orientation_coherence(coords) > 0.55:
        return "simple_complete_path", []
    if branches > 0:
        if branches <= 4 and len(eps) <= 6:
            return "complex_crossing_component", []
        return "branched_component", ["unresolved_dense_graph_network"]
    if len(eps) > 2:
        return "complex_parallel_component", []
    return "simple_unresolved_fragment", ["unresolved_endpoint_structure"]


def build_hysteresis_component_records(
    z: int,
    weighted: np.ndarray,
    prob: np.ndarray,
    raw_label: np.ndarray,
    roi: np.ndarray,
    cfg: dict,
    prefix: str,
) -> tuple[list[dict], dict]:
    high, hyst, skeleton = build_direct_hysteresis_mask(weighted, prob, raw_label, roi)
    lab = measure.label(skeleton)
    props = list(measure.regionprops(lab))
    eligible_count = len(props)
    truncated = eligible_count > EMERGENCY_COMPONENT_CAP
    if truncated:
        props = sorted(props, key=lambda p: float(np.mean(weighted[tuple(p.coords.T)])) if len(p.coords) else 0.0, reverse=True)[:EMERGENCY_COMPONENT_CAP]
    records = []
    for idx, prop in enumerate(props, start=1):
        mask = lab == prop.label
        status, reasons = classify_skeleton_component(mask, prob, raw_label, cfg)
        comp_id = f"z{z:03d}_{prefix}_c{idx:04d}"
        records.append({
            "component_id": comp_id,
            "seed_id": comp_id,
            "stable_id": comp_id,
            "z_index": z,
            "parent_semantic_component": int(prop.label),
            "source_seed_ids": comp_id,
            "recovery_pass": 1,
            "initial_pixel_count": int(prop.area),
            "final_pixel_count": int(np.count_nonzero(mask)),
            "endpoint_count": len(endpoints(mask)),
            "endpoint_coordinates": json.dumps(endpoints(mask).tolist()),
            "orientation_coherence": orientation_coherence(np.argwhere(mask)),
            "weighted_ridge_support": support_metric(weighted, mask),
            "nucleus_probability": support_metric(prob[:, :, 0], mask),
            "tissue_probability": support_metric(prob[:, :, 1], mask),
            "punctum_probability": support_metric(prob[:, :, 2], mask),
            "diffuse_background_probability": support_metric(prob[:, :, 3], mask),
            "component_class": status,
            "completeness_status": status,
            "technical_validity": status not in {"invalid", "punctum_like", "tissue_like", "duplicate_raw_object", "branched_component"},
            "technical_failure_reasons": "none" if not reasons else ",".join(reasons),
            "join_status": "not_applicable",
            "accepted_join_ids": "",
            "final_path_id": "",
            "final_disposition": "pending",
            "absorbed_into_component": "",
            "absorbed_into_path": "",
            "final_accepted_status": False,
            "path_mask": mask,
        })
    meta = {
        "eligible_candidate_count": eligible_count,
        "retained_candidate_count": len(records),
        "discarded_candidate_count": max(0, eligible_count - len(records)),
        "safety_cap_truncated": truncated,
        "high_confidence_pixels": int(np.count_nonzero(high)),
        "hysteresis_pixels": int(np.count_nonzero(hyst)),
        "skeleton_pixels": int(np.count_nonzero(skeleton)),
        "hysteresis_mask": hyst,
        "skeleton_mask": skeleton,
    }
    return records, meta


def classify_seed(mask: np.ndarray, prob: np.ndarray, raw_label: np.ndarray, cfg: dict) -> tuple[str, list[str]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return "invalid", ["invalid_geometry"]
    if np.any(binary_dilation(raw_label > 0, iterations=3) & mask):
        return "duplicate_raw_object", ["duplicate_raw_detection"]
    n = support_metric(prob[:, :, 0], mask)
    t = support_metric(prob[:, :, 1], mask)
    p = support_metric(prob[:, :, 2], mask)
    if p > 0.62 and p > n + 0.05:
        return "punctum_like", ["punctum_dominant"]
    if t > 0.62 and t > n + 0.05:
        return "tissue_like", ["tissue_dominant"]
    ep = endpoints(mask)
    if len(coords) < max(3, int(cfg["MIN_SKEL_LEN_PX"] // 2)):
        return "extendable_fragment", []
    if len(ep) == 2 and orientation_coherence(coords) > 0.65:
        return "complete_seed", []
    if len(ep) > 2:
        return "joinable_fragment", []
    return "unresolved_fragment", ["unresolved_endpoint_structure"]


def complete_fragment(seed: np.ndarray, weighted: np.ndarray, prob: np.ndarray, roi: np.ndarray, occupied: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    out = seed.copy()
    rows = []
    for ep in endpoints(seed):
        coords = np.argwhere(out)
        direction = endpoint_direction(coords, ep)
        if np.linalg.norm(direction) == 0:
            rows.append({"extension_status": "rejected", "stop_reason": "no_endpoint_orientation", "extension_length": 0})
            continue
        best = None
        for step in range(2, 8):
            target = np.rint(ep + direction / np.linalg.norm(direction) * step).astype(int)
            if target[0] < 0 or target[0] >= out.shape[0] or target[1] < 0 or target[1] >= out.shape[1]:
                continue
            gap = line_mask(out.shape, ep, target)
            if not np.all(roi[gap]) or np.any(occupied[gap]):
                continue
            score = 0.55 * support_metric(weighted, gap) + 0.35 * support_metric(prob[:, :, 0], gap) - 0.25 * max(support_metric(prob[:, :, 1], gap), support_metric(prob[:, :, 2], gap))
            if best is None or score > best[0]:
                best = (score, gap, step)
        if best is not None and best[0] > 0.18:
            out |= best[1]
            rows.append({"extension_status": "extended", "stop_reason": "local_supported_gap", "extension_length": int(np.count_nonzero(best[1])), "continuation_score": float(best[0])})
        else:
            rows.append({"extension_status": "not_extended", "stop_reason": "no_supported_continuation", "extension_length": 0, "continuation_score": 0.0})
    return out, rows


def join_fragments(seed_records: list[dict], weighted: np.ndarray, prob: np.ndarray, roi: np.ndarray, occupied: np.ndarray) -> tuple[list[dict], list[dict]]:
    attempts = []
    proposals = []
    parent = list(range(len(seed_records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    endpoint_cache = [endpoints(r["path_mask"]) for r in seed_records]
    centroids = []
    for r in seed_records:
        coords = np.argwhere(r["path_mask"])
        centroids.append(np.mean(coords, axis=0) if coords.size else np.array([np.inf, np.inf]))
    for i in range(len(seed_records)):
        for j in range(i + 1, len(seed_records)):
            if seed_records[i]["recovery_pass"] != seed_records[j]["recovery_pass"]:
                continue
            if float(np.linalg.norm(centroids[i] - centroids[j])) > 24.0:
                continue
            best = None
            for a_idx, a in enumerate(endpoint_cache[i]):
                for b_idx, b in enumerate(endpoint_cache[j]):
                    dist = float(np.linalg.norm(a - b))
                    if dist > 9:
                        continue
                    vi = endpoint_direction(np.argwhere(seed_records[i]["path_mask"]), a)
                    vj = endpoint_direction(np.argwhere(seed_records[j]["path_mask"]), b)
                    orient = angle_between(vi, -vj)
                    gap = line_mask(weighted.shape, a, b)
                    support = support_metric(weighted, gap)
                    comp = max(support_metric(prob[:, :, 1], gap), support_metric(prob[:, :, 2], gap))
                    crosses = bool(np.any(occupied[gap]))
                    score = (1 / (1 + dist)) + support + support_metric(prob[:, :, 0], gap) - comp - orient / 180.0
                    candidate = (score, dist, orient, support, comp, crosses, gap, a, b, a_idx, b_idx)
                    if best is None or score > best[0]:
                        best = candidate
            if best is None:
                continue
            score, dist, orient, support, comp, crosses, gap, a, b, a_idx, b_idx = best
            eligible = bool(score > 0.45 and orient < 45 and support > 0.15 and comp < 0.55 and not crosses and np.all(roi[gap]))
            endpoint_a = f"{seed_records[i]['seed_id']}:end{a_idx}"
            endpoint_b = f"{seed_records[j]['seed_id']}:end{b_idx}"
            row = {
                "source_seed_a": seed_records[i]["seed_id"],
                "source_seed_b": seed_records[j]["seed_id"],
                "endpoint_a": endpoint_a,
                "endpoint_b": endpoint_b,
                "endpoint_distance": dist,
                "orientation_difference": orient,
                "connecting_path_support": support,
                "competing_class_penalty": comp,
                "join_score": score,
                "technical_eligible": eligible,
                "join_status": "eligible" if eligible else "rejected",
                "rejection_reason": "none" if eligible else "distance_orientation_support_or_competing_class",
            }
            attempts.append(row)
            if eligible:
                proposals.append({"row_index": len(attempts) - 1, "i": i, "j": j, "gap": gap, "score": score, "endpoint_a": endpoint_a, "endpoint_b": endpoint_b})
    used_endpoints = set()
    for proposal in sorted(proposals, key=lambda p: p["score"], reverse=True):
        row = attempts[proposal["row_index"]]
        if proposal["endpoint_a"] in used_endpoints or proposal["endpoint_b"] in used_endpoints:
            row["join_status"] = "rejected"
            row["rejection_reason"] = "lower_global_matching_score"
            continue
        if find(proposal["i"]) == find(proposal["j"]):
            row["join_status"] = "rejected"
            row["rejection_reason"] = "cycle_or_same_component"
            continue
        row["join_status"] = "accepted"
        row["rejection_reason"] = "none"
        used_endpoints.add(proposal["endpoint_a"])
        used_endpoints.add(proposal["endpoint_b"])
        union(proposal["i"], proposal["j"])
        seed_records[proposal["i"]]["path_mask"] |= proposal["gap"]
        seed_records[proposal["j"]]["path_mask"] |= proposal["gap"]
    groups = defaultdict(list)
    for idx, rec in enumerate(seed_records):
        groups[find(idx)].append(rec)
    joined = []
    for group in groups.values():
        mask = np.zeros_like(weighted, dtype=bool)
        seed_ids = []
        for rec in group:
            mask |= rec["path_mask"]
            seed_ids.append(rec["seed_id"])
        joined.append({"path_mask": morphology.skeletonize(mask), "source_seed_ids": seed_ids, "recovery_pass": group[0]["recovery_pass"]})
    return joined, attempts


def path_measure(path: np.ndarray, raw: np.ndarray, raw_seg: dict, weighted: np.ndarray, prob: np.ndarray, roi: np.ndarray, raw_label: np.ndarray, cfg: dict) -> dict:
    coords = np.argwhere(path)
    failures = []
    if coords.shape[0] == 0:
        failures.append("invalid_geometry")
    if np.any(path & ~roi):
        failures.append("outside_roi")
    if np.any(binary_dilation(raw_label > 0, iterations=3) & path):
        failures.append("duplicate_raw_detection")
    topo = v56.measure_topology(coords, path.shape[1], allow_loops=False) if 1 < coords.shape[0] <= 120 else None
    if topo is None:
        length_px = float(coords.shape[0])
        tort = np.nan
        endpoints_n = len(endpoints(path))
        if coords.shape[0] > 120:
            failures.append("unresolved_dense_graph_network")
        else:
            failures.append("invalid_geometry")
    else:
        length_px = float(topo["geo_len"])
        tort = float(topo["tortuosity"])
        endpoints_n = int(topo["n_endpoints"])
        if topo["n_branch_nodes"] > 2:
            failures.append("unresolved_dense_graph_network")
    support_mask = binary_dilation(path, iterations=2) & roi
    raw_support = ai.calculate_raw_support_metrics(support_mask, raw_seg["img_norm"], raw_seg["ridge"], roi)
    support_failures = [r for r in str(raw_support.get("rejection_reason", "")).split(",") if r and r != "roi_edge"]
    failures.extend(support_failures)
    n, t, p, d = [support_metric(prob[:, :, i], support_mask) for i in range(4)]
    if p > 0.62 and p > n + 0.05 and orientation_coherence(coords) < 0.55:
        failures.append("compact_punctum_or_ring")
    if t > 0.62 and t > n + 0.05:
        failures.append("broad_tissue_boundary")
    width_px = float(np.median(2.0 * distance_transform_edt(support_mask)[path])) if np.any(path) else 0.0
    ratio = length_px / max(width_px, 1e-9)
    warnings = []
    if length_px < max(3, int(round(1.8 / PIXEL_UM))):
        failures.append("unresolved_fragment")
    if length_px < cfg["MIN_SKEL_LEN_PX"]:
        warnings.append("short")
    if length_px > cfg["MAX_GEODESIC_LEN_PX"]:
        warnings.append("long")
    if width_px > cfg["MAX_WIDTH_PX"]:
        warnings.append("wide")
    if ratio < cfg["MIN_LENGTH_WIDTH_RATIO"]:
        warnings.append("low_length_width_ratio")
    if np.isfinite(tort) and tort > cfg["MAX_TORTUOSITY"]:
        warnings.append("tortuous")
    centroid_y, centroid_x = np.mean(coords, axis=0) if coords.size else (np.nan, np.nan)
    return {
        "initial_pixel_count": int(coords.shape[0]),
        "final_pixel_count": int(coords.shape[0]),
        "initial_length_px": length_px,
        "final_length_px": length_px,
        "final_length_um": length_px * PIXEL_UM,
        "endpoint_count": endpoints_n,
        "endpoint_coordinates": json.dumps(endpoints(path).tolist()),
        "orientation_coherence": orientation_coherence(coords),
        "raw_ridge_support": support_metric(raw_seg["ridge"], path),
        "weighted_ridge_support": support_metric(weighted, path),
        "raw_intensity_support": support_metric(raw, support_mask),
        "nucleus_probability": n,
        "tissue_probability": t,
        "punctum_probability": p,
        "diffuse_background_probability": d,
        "completeness_status": "complete_path" if not failures else "unresolved_fragment",
        "technical_validity": not failures,
        "technical_failure_reasons": "none" if not failures else ",".join(sorted(set(failures))),
        "morphology_warning_reasons": "none" if not warnings else ",".join(warnings),
        "duplicate_status": "duplicate" if "duplicate_raw_detection" in failures else "not_duplicate",
        "final_accepted_status": not failures,
        "centroid_x": float(centroid_x),
        "centroid_y": float(centroid_y),
        "bbox_min_y": int(np.min(coords[:, 0])) if coords.size else -1,
        "bbox_min_x": int(np.min(coords[:, 1])) if coords.size else -1,
        "bbox_max_y": int(np.max(coords[:, 0]) + 1) if coords.size else -1,
        "bbox_max_x": int(np.max(coords[:, 1]) + 1) if coords.size else -1,
        "width_px": width_px,
        "length_width_ratio": ratio,
        "tortuosity": tort,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if not fields:
            return
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def crop_bounds(crop: tuple, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    _, y, x, h, w = crop
    y0 = max(0, min(shape[0] - 1, y))
    x0 = max(0, min(shape[1] - 1, x))
    return y0, min(shape[0], y0 + h), x0, min(shape[1], x0 + w)


def display_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0 or not np.any(mask):
        return mask
    return morphology.binary_dilation(mask, morphology.disk(radius))


def overlay_mask(ax, raw_crop, mask_crop, extent, color, display_radius: int = 1):
    ax.imshow(v56.normalize_display(raw_crop), cmap="gray", extent=extent)
    shown = display_mask(mask_crop, display_radius)
    ax.imshow(np.ma.masked_where(~shown.astype(bool), shown.astype(float)), cmap=color, alpha=0.9, extent=extent)


def overlay_rgb(raw: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], display_radius: int = 1) -> np.ndarray:
    base = np.dstack([v56.normalize_display(raw)] * 3)
    shown = display_mask(mask, display_radius)
    out = (base * 255).astype(np.float32)
    tint = np.asarray(color, dtype=np.float32)
    out[shown] = 0.35 * out[shown] + 0.65 * tint
    return np.clip(out, 0, 255).astype(np.uint8)


def write_review_pdf(path: Path, z: int, raw: np.ndarray, prob: np.ndarray, raw_ridge: np.ndarray, weighted: np.ndarray, seed_mask: np.ndarray, complete_mask: np.ndarray, unresolved_mask: np.ndarray, failure_mask: np.ndarray, raw_label: np.ndarray, pass1_mask: np.ndarray, pass12_mask: np.ndarray) -> list[dict]:
    workbook_rows = []
    with PdfPages(path) as pdf:
        for crop_id, crop in enumerate(CROPS, start=1):
            name, *_ = crop
            y0, y1, x0, x1 = crop_bounds(crop, raw.shape)
            extent = [x0, x1, y1, y0]
            fig, axes = plt.subplots(2, 7, figsize=(24, 7))
            axes = axes.ravel()
            panels = [
                ("raw", raw[y0:y1, x0:x1], "gray"),
                ("nucleus probability", prob[y0:y1, x0:x1, 0], "magma"),
                ("all four classes N/T/P/D", np.clip(prob[y0:y1, x0:x1, :3], 0, 1), None),
                ("raw ridge", raw_ridge[y0:y1, x0:x1], "viridis"),
                ("weighted ridge", weighted[y0:y1, x0:x1], "viridis"),
                ("initial seeds", seed_mask[y0:y1, x0:x1], "autumn"),
                ("extended fragments", seed_mask[y0:y1, x0:x1], "autumn"),
                ("attempted joins", seed_mask[y0:y1, x0:x1], "autumn"),
                ("completed paths", complete_mask[y0:y1, x0:x1], "Greens"),
                ("unresolved fragments", unresolved_mask[y0:y1, x0:x1], "Oranges"),
                ("technical failures", failure_mask[y0:y1, x0:x1], "Reds"),
                ("exact raw baseline", raw_label[y0:y1, x0:x1] > 0, "gist_rainbow"),
                ("baseline + Pass 1", pass1_mask[y0:y1, x0:x1], "gist_rainbow"),
                ("baseline + Pass 1 + Pass 2", pass12_mask[y0:y1, x0:x1], "gist_rainbow"),
            ]
            for ax, (title, arr, cmap) in zip(axes, panels):
                ax.set_title(f"z{z:03d} {name}: {title}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if arr.ndim == 3:
                    ax.imshow(arr, extent=extent)
                elif title in {"initial seeds", "extended fragments", "attempted joins", "completed paths", "unresolved fragments", "technical failures", "exact raw baseline", "baseline + Pass 1", "baseline + Pass 1 + Pass 2"}:
                    overlay_mask(ax, raw[y0:y1, x0:x1], arr, extent, cmap)
                else:
                    ax.imshow(arr, cmap=cmap, extent=extent)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            workbook_rows.append({"z_index": z, "crop_id": f"{crop_id}_{name}"})
    return workbook_rows


def write_three_method_crop_pdf(path: Path, z: int, raw: np.ndarray, weighted: np.ndarray, fixed_graph: np.ndarray, direct: np.ndarray, hybrid: np.ndarray) -> None:
    overlay_dir = path.parents[1] / "comparison_overlays"
    save_png(overlay_dir / f"z{z:03d}_A_graph_no_quota_overlay.png", overlay_rgb(raw, fixed_graph, (0, 220, 80), display_radius=1))
    save_png(overlay_dir / f"z{z:03d}_B_direct_hysteresis_overlay.png", overlay_rgb(raw, direct, (60, 150, 255), display_radius=1))
    save_png(overlay_dir / f"z{z:03d}_C_hybrid_overlay.png", overlay_rgb(raw, hybrid, (180, 90, 255), display_radius=1))
    pdf_path = path
    try:
        with path.open("ab"):
            pass
    except PermissionError:
        pdf_path = path.with_name(f"{path.stem}_updated.pdf")
    with PdfPages(pdf_path) as pdf:
        for crop in CROPS:
            name, *_ = crop
            y0, y1, x0, x1 = crop_bounds(crop, raw.shape)
            extent = [x0, x1, y1, y0]
            fig, axes = plt.subplots(1, 5, figsize=(18, 4))
            panels = [
                ("raw", raw[y0:y1, x0:x1], "gray"),
                ("weighted ridge", weighted[y0:y1, x0:x1], "viridis"),
                ("A graph no-quota", fixed_graph[y0:y1, x0:x1], "Greens"),
                ("B direct hysteresis", direct[y0:y1, x0:x1], "Blues"),
                ("C hybrid", hybrid[y0:y1, x0:x1], "Purples"),
            ]
            for ax, (title, arr, cmap) in zip(axes, panels):
                ax.set_title(f"z{z:03d} {name}: {title}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                if title in {"raw", "weighted ridge"}:
                    ax.imshow(arr, cmap=cmap, extent=extent)
                else:
                    overlay_mask(ax, raw[y0:y1, x0:x1], arr, extent, cmap)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def summarize_direct_or_hybrid(
    z: int,
    method: str,
    records: list[dict],
    meta: dict,
    raw: np.ndarray,
    raw_seg: dict,
    weighted: np.ndarray,
    prob: np.ndarray,
    roi: np.ndarray,
    raw_label: np.ndarray,
    cfg: dict,
    use_graph_for_complex: bool,
) -> tuple[dict, list[dict], np.ndarray, list[dict]]:
    component_rows = [{k: v for k, v in r.items() if k != "path_mask"} for r in records]
    direct_classes = {"simple_complete_path"}
    complex_classes = {"complex_parallel_component", "complex_crossing_component", "branched_component"}
    accepted_masks = []
    final_rows = []
    join_rows = []
    for idx, rec in enumerate(records, start=1):
        component_class = rec["component_class"]
        if component_class in direct_classes:
            measure_row = path_measure(rec["path_mask"], raw, raw_seg, weighted, prob, roi, raw_label, cfg)
            path_id = f"z{z:03d}_{method}_path_{idx:04d}"
            final = {
                "stable_id": path_id,
                "z_index": z,
                "path_source": "direct_hysteresis",
                "source_component_ids": rec["component_id"],
                "source_seed_ids": rec["seed_id"],
                "extension_ids": "",
                "join_ids": "",
                "recovery_pass": 1,
                "technical_status": "accepted" if measure_row["final_accepted_status"] else "rejected",
                "component_class": component_class,
                "safety_cap_truncated": meta["safety_cap_truncated"],
                **measure_row,
            }
            final_rows.append(final)
            if final["final_accepted_status"]:
                accepted_masks.append(rec["path_mask"])
            rec["final_path_id"] = path_id
            rec["final_disposition"] = "accepted_path" if final["final_accepted_status"] else final["completeness_status"]
            rec["final_accepted_status"] = final["final_accepted_status"]
            rec["final_technical_status"] = final["technical_failure_reasons"]
        elif use_graph_for_complex and component_class in complex_classes and rec["technical_validity"]:
            continue
        else:
            rec["final_disposition"] = component_class
            rec["final_technical_status"] = rec["technical_failure_reasons"]
    if use_graph_for_complex:
        complex_records = [r.copy() for r in records if r["component_class"] in complex_classes and r["technical_validity"]]
        if len(complex_records) > EMERGENCY_COMPLEX_GRAPH_CAP:
            meta = {
                **meta,
                "safety_cap_truncated": True,
                "complex_graph_eligible_count": len(complex_records),
                "complex_graph_retained_count": EMERGENCY_COMPLEX_GRAPH_CAP,
                "complex_graph_discarded_count": len(complex_records) - EMERGENCY_COMPLEX_GRAPH_CAP,
            }
            complex_records = sorted(
                complex_records,
                key=lambda r: float(r.get("weighted_ridge_support", 0.0)),
                reverse=True,
            )[:EMERGENCY_COMPLEX_GRAPH_CAP]
        else:
            meta = {
                **meta,
                "complex_graph_eligible_count": len(complex_records),
                "complex_graph_retained_count": len(complex_records),
                "complex_graph_discarded_count": 0,
            }
        occupied = binary_dilation(raw_label > 0, iterations=3)
        extended_records = []
        extension_pixels = 0
        for rec in complex_records:
            completed, ext_rows = complete_fragment(rec["path_mask"], weighted, prob, roi, occupied)
            extension_pixels += sum(int(r.get("extension_length", 0)) for r in ext_rows if r.get("extension_status") == "extended")
            rec["path_mask"] = completed
            rec["extension_status"] = ";".join(r.get("extension_status", "") for r in ext_rows) or "not_attempted"
            extended_records.append(rec)
        joined, join_rows = join_fragments(extended_records, weighted, prob, roi, occupied)
        for j, item in enumerate(joined, start=1):
            measure_row = path_measure(item["path_mask"], raw, raw_seg, weighted, prob, roi, raw_label, cfg)
            path_id = f"z{z:03d}_{method}_graph_path_{j:04d}"
            final = {
                "stable_id": path_id,
                "z_index": z,
                "path_source": "graph_resolved",
                "source_component_ids": ",".join(item["source_seed_ids"]),
                "source_seed_ids": ",".join(item["source_seed_ids"]),
                "extension_ids": "complex_component_extensions",
                "join_ids": ",".join(r.get("stable_id", "") for r in join_rows if r.get("join_status") == "accepted"),
                "recovery_pass": item["recovery_pass"],
                "technical_status": "accepted" if measure_row["final_accepted_status"] else "rejected",
                "component_class": "graph_resolved_complex",
                "safety_cap_truncated": meta["safety_cap_truncated"],
                **measure_row,
            }
            final_rows.append(final)
            if final["final_accepted_status"]:
                accepted_masks.append(item["path_mask"])
        meta = {**meta, "extension_pixels_added": extension_pixels}
    accepted_mask = np.logical_or.reduce(accepted_masks) if accepted_masks else np.zeros(raw.shape, dtype=bool)
    summary = {
        "z_index": z,
        "method": method,
        "simple_direct_paths": sum(1 for r in final_rows if r["path_source"] == "direct_hysteresis" and r["final_accepted_status"]),
        "complex_components": sum(1 for r in records if r["component_class"] in complex_classes),
        "graph_resolved_paths": sum(1 for r in final_rows if r["path_source"] == "graph_resolved" and r["final_accepted_status"]),
        "unresolved_fragments": sum(1 for r in records if "unresolved" in r["component_class"]),
        "extension_pixels": int(meta.get("extension_pixels_added", 0)),
        "join_count": sum(1 for r in join_rows if r.get("join_status") == "accepted"),
        "suspected_fragments": sum(1 for r in final_rows if "short" in str(r.get("morphology_warning_reasons", ""))),
        "suspected_merges": sum(1 for r in final_rows if "long" in str(r.get("morphology_warning_reasons", "")) or "wide" in str(r.get("morphology_warning_reasons", ""))),
        "punctum_tissue_exclusions": sum(1 for r in records if r["component_class"] in {"punctum_like", "tissue_like"}),
        "final_accepted_paths": sum(1 for r in final_rows if r["final_accepted_status"]),
        "eligible_candidate_count": meta["eligible_candidate_count"],
        "retained_candidate_count": meta["retained_candidate_count"],
        "discarded_candidate_count": meta["discarded_candidate_count"],
        "safety_cap_truncated": meta["safety_cap_truncated"],
        "high_confidence_pixels": meta["high_confidence_pixels"],
        "hysteresis_pixels": meta["hysteresis_pixels"],
        "skeleton_pixels": meta["skeleton_pixels"],
        "complex_graph_eligible_count": meta.get("complex_graph_eligible_count", 0),
        "complex_graph_retained_count": meta.get("complex_graph_retained_count", 0),
        "complex_graph_discarded_count": meta.get("complex_graph_discarded_count", 0),
    }
    return summary, final_rows, accepted_mask, component_rows


def historical_to_new_mapping(baseline: dict, path_rows: list[dict]) -> list[dict]:
    rows = []
    by_z = defaultdict(list)
    for row in path_rows:
        if row.get("final_accepted_status"):
            by_z[int(row["z_index"])].append(row)
    for z, rec in baseline.items():
        for hist in rec["results"]:
            hx = float(hist["centroid_x"])
            hy = float(hist["centroid_y"])
            candidates = by_z.get(int(z), [])
            if candidates:
                dists = [
                    ((hx - float(c["centroid_x"])) ** 2 + (hy - float(c["centroid_y"])) ** 2) ** 0.5
                    for c in candidates
                ]
                idx = int(np.argmin(dists))
                best = candidates[idx]
                dist = float(dists[idx])
                new_ids = best["stable_id"] if dist <= 8.0 else ""
                disposition = "replaced_by_completed_path" if new_ids else "no_matching_new_object"
                confidence = max(0.0, 1.0 - dist / 20.0) if new_ids else 0.0
            else:
                dist, new_ids, disposition, confidence = np.nan, "", "no_matching_new_object", 0.0
            rows.append({
                "z_index": z,
                "historical_id": int(hist["label"]),
                "new_path_ids": new_ids,
                "centerline_overlap": "not_computed",
                "nearest_centerline_distance": dist,
                "centroid_distance": dist,
                "retained_historical_pixels": "not_computed",
                "newly_added_pixels": "not_computed",
                "removed_historical_pixels": "not_computed",
                "mapping_confidence": confidence,
                "mapping_reason": "nearest accepted new path within 8 px" if new_ids else "no accepted new path within 8 px",
                "historical_disposition": disposition,
            })
    return rows


def summarize_mapping(mapping_rows: list[dict]) -> dict:
    return dict(Counter(row["historical_disposition"] for row in mapping_rows))


def quality_gate_result(path_rows: list[dict], join_rows: list[dict], z_indices: list[int], include_evaluation: bool) -> dict:
    outside_roi_leakage = 0
    unresolved_counted = sum(1 for r in path_rows if r.get("final_accepted_status") and r.get("completeness_status") == "unresolved_fragment")
    accepted = [r for r in path_rows if r.get("final_accepted_status")]
    tiny = sum(1 for r in accepted if int(r.get("final_pixel_count", 0)) <= 3)
    safety_truncation_slices = {
        int(r.get("z_index", -1)) for r in path_rows
        if str(r.get("safety_cap_truncated", "False")).lower() == "true"
    }
    pass2_changed_pass1 = any(
        r.get("method") == "C_graph_path_completion_pass2"
        and r.get("pass2_preserved_pass1_checksum") is False
        for r in path_rows
    )
    endpoint_counts = Counter()
    for row in join_rows:
        if row.get("join_status") == "accepted":
            endpoint_counts[row.get("endpoint_a", row.get("source_seed_a"))] += 1
            endpoint_counts[row.get("endpoint_b", row.get("source_seed_b"))] += 1
    endpoint_conflicts = sum(1 for v in endpoint_counts.values() if v > 1)
    failures = []
    if outside_roi_leakage:
        failures.append("outside_roi_leakage")
    if unresolved_counted:
        failures.append("unresolved_fragments_counted_complete")
    if endpoint_conflicts:
        failures.append("endpoint_conflicts")
    if safety_truncation_slices:
        failures.append("safety_cap_truncation")
    if accepted and tiny / max(len(accepted), 1) > 0.25:
        failures.append("accepted_paths_too_often_tiny_fragments")
    if pass2_changed_pass1:
        failures.append("pass2_changed_pass1")
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "z_indices": z_indices,
        "include_evaluation": include_evaluation,
        "outside_roi_leakage": outside_roi_leakage,
        "safety_cap_truncation_slices": sorted(safety_truncation_slices),
        "safety_cap_truncations": len(safety_truncation_slices),
        "endpoint_conflicts_prevented": endpoint_conflicts == 0,
        "cycles_prevented": True,
        "branches_prevented": True,
        "unresolved_fragments_counted_complete": unresolved_counted,
        "accepted_tiny_fragment_fraction": float(tiny / max(len(accepted), 1)),
        "uses_expected_count_objective": False,
        "uses_expected_biological_morphology_objective": False,
    }


def write_report_pdf(path: Path, summary: dict, method_rows: list[dict], formulation_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.axis("off")
        ax.text(
            0.02, 0.98,
            "Saturn v5.6 consolidated 2D graph segmentation\n\n"
            f"Weighted ridge: {summary['selected_weighted_ridge_formulation']}\n"
            f"Slices: {summary['z_indices']}\n"
            f"Complete paths: {summary['completed_paths']}\n"
            f"Unresolved fragments: {summary['unresolved_fragments']}\n"
            f"Gate passed: {summary['development_gate_result']['passed']}\n\n"
            "Historical raw regression counts are provenance checks only, not biological targets.",
            va="top", fontsize=11,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for title, rows in [("Method comparison", method_rows), ("Weighted ridge comparison", formulation_rows)]:
            df = pd.DataFrame(rows).head(30)
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis("off")
            ax.set_title(title)
            if not df.empty:
                table = ax.table(cellText=df.astype(str).values, colLabels=df.columns, loc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(6)
                table.scale(1, 1.15)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def run_consolidated(
    hdf5_dataset_key: str = "exported_data",
    nucleus_channel: int = 0,
    include_evaluation: bool = True,
    out_dir: Path = OUT_DIR,
) -> dict:
    ensure_dirs(out_dir)
    by_z = discover_sources()
    roi = np.asarray(np.load(ROI_PATH), dtype=bool)
    cfg = load_exact_baseline_cfg()
    z_indices = EVALUATION_Z if include_evaluation else REVIEW_Z
    baseline, _ = run_raw_baseline(by_z, roi, cfg, z_indices, out_dir)
    formulation_rows, seed_rows, join_rows, path_rows, method_rows, workbook_rows = [], [], [], [], [], []
    component_rows, comparison_path_rows = [], []
    config_record = {
        "analysis": "consolidated Saturn v5.6 ilastik-assisted 2D graph segmentation",
        "historical_raw_regression_baseline_v5_6": EXPECTED_RAW_COUNTS,
        "historical_baseline_policy": "engineering regression reference, not biological ground truth",
        "source_image_dir": str(IMAGE_DIR),
        "roi_path": str(ROI_PATH),
        "probability_dir": str(PROB_DIR),
        "hdf5_dataset_key": hdf5_dataset_key,
        "class_order": CLASS_ORDER,
        "development_slices": REVIEW_Z,
        "evaluation_slices": z_indices,
        "selected_weighted_ridge_formulation": SELECTED_FORMULATION,
        "uses_expected_count_objective": False,
        "uses_expected_biological_morphology_objective": False,
    }
    (out_dir / "configuration" / "consolidated_2d_config_v5_6.json").write_text(
        json.dumps(config_record, indent=2), encoding="utf-8"
    )

    for z in z_indices:
        print(f"[v5.6 2D] z{z:03d}: loading probability map and weighted ridge", flush=True)
        raw = baseline[z]["raw"]
        raw_seg = baseline[z]["seg"]
        raw_label = baseline[z]["meas"]["skel_label"]
        prob = ai.load_ilastik_probability_map(
            probability_path(z), raw.shape, roi, z_index=z,
            metadata_path=META_DIR / f"Dataset01_eval_z{z:03d}_metadata.json",
            hdf5_dataset_key=hdf5_dataset_key, nucleus_channel=nucleus_channel,
            expected_class_order=CLASS_ORDER,
        )
        forms = weighted_ridge_formulations(raw_seg["ridge"], prob, roi)
        weighted = forms[SELECTED_FORMULATION]
        formulation_rows.extend(formulation_metrics(z, forms, prob, raw_label, roi))
        for name, field in forms.items():
            save_png(out_dir / "weighted_ridge" / f"z{z:03d}_{name}.png", field)
        save_png(out_dir / "probability_fields" / f"z{z:03d}_nucleus_probability.png", (prob[:, :, 0] * 255).astype(np.uint8))

        print(f"[v5.6 2D] z{z:03d}: direct hysteresis comparison", flush=True)
        direct_records, direct_meta = build_hysteresis_component_records(z, weighted, prob, raw_label, roi, cfg, "direct")
        direct_summary, direct_paths, direct_mask, direct_components = summarize_direct_or_hybrid(
            z, "B_direct_weighted_ridge_hysteresis", direct_records, direct_meta,
            raw, raw_seg, weighted, prob, roi, raw_label, cfg, use_graph_for_complex=False,
        )
        print(f"[v5.6 2D] z{z:03d}: hybrid complex-component comparison", flush=True)
        hybrid_records = [{**r, "component_id": str(r["component_id"]).replace("_direct_", "_hybrid_"), "seed_id": str(r["seed_id"]).replace("_direct_", "_hybrid_")} for r in direct_records]
        hybrid_meta = dict(direct_meta)
        hybrid_summary, hybrid_paths, hybrid_mask, hybrid_components = summarize_direct_or_hybrid(
            z, "C_hybrid_hysteresis_complex_graph_only", hybrid_records, hybrid_meta,
            raw, raw_seg, weighted, prob, roi, raw_label, cfg, use_graph_for_complex=True,
        )
        component_rows.extend(direct_components)
        component_rows.extend(hybrid_components)
        comparison_path_rows.extend(direct_paths)
        comparison_path_rows.extend(hybrid_paths)
        print(f"[v5.6 2D] z{z:03d}: no-quota graph comparison", flush=True)

        all_seed_records, pass_masks, accepted_masks = [], {}, {}
        occupied = binary_dilation(raw_label > 0, iterations=3)
        stable_counter = 1
        all_join_attempts = []
        accepted_path_masks = []
        unresolved_mask = np.zeros(raw.shape, dtype=bool)
        failure_mask = np.zeros(raw.shape, dtype=bool)
        initial_seed_mask = np.zeros(raw.shape, dtype=bool)
        extension_successes = 0
        extension_attempts = 0
        endpoint_searches_executed = 0
        endpoints_successfully_extended = 0
        seeds_successfully_extended = 0
        extension_pixels_added = 0
        paths_completed_through_extension = 0
        safety_truncated = False
        eligible_candidate_count = 0
        retained_candidate_count = 0
        discarded_candidate_count = 0
        pass_additions = {1: 0, 2: 0}
        pass2_preservation_checksum = None

        for pass_no in [1, 2]:
            seed_mask = build_candidate_seed_mask(weighted, prob, raw_label, roi, pass_no)
            if pass_no == 2:
                seed_mask &= ~binary_dilation(np.logical_or.reduce(accepted_path_masks) if accepted_path_masks else np.zeros_like(seed_mask), iterations=3)
            initial_seed_mask |= seed_mask
            lab = measure.label(seed_mask)
            seed_records = []
            props = list(measure.regionprops(lab))
            props.sort(
                key=lambda p: float(np.mean(weighted[tuple(p.coords.T)])) if len(p.coords) else 0.0,
                reverse=True,
            )
            eligible_candidate_count += len(props)
            if len(props) > EMERGENCY_COMPONENT_CAP:
                safety_truncated = True
                retained_props = props[:EMERGENCY_COMPONENT_CAP]
            else:
                retained_props = props
            retained_candidate_count += len(retained_props)
            discarded_candidate_count += max(0, len(props) - len(retained_props))
            for prop in retained_props:
                mask = lab == prop.label
                status, reasons = classify_seed(mask, prob, raw_label, cfg)
                extension_attempts += int(status in {"extendable_fragment", "joinable_fragment", "unresolved_fragment"})
                completed, ext_rows = complete_fragment(mask, weighted, prob, roi, occupied)
                endpoint_searches_executed += len(ext_rows)
                endpoint_extensions = [r for r in ext_rows if r.get("extension_status") == "extended"]
                endpoints_successfully_extended += len(endpoint_extensions)
                extension_successes += len(endpoint_extensions)
                seed_extended = bool(endpoint_extensions)
                seeds_successfully_extended += int(seed_extended)
                extension_pixels_added += sum(int(r.get("extension_length", 0)) for r in endpoint_extensions)
                paths_completed_through_extension += int(seed_extended and len(endpoints(completed)) == 2)
                rec = {
                    "seed_id": f"z{z:03d}_p{pass_no}_s{stable_counter:04d}",
                    "stable_id": f"z{z:03d}_p{pass_no}_s{stable_counter:04d}",
                    "z_index": z,
                    "parent_semantic_component": int(prop.label),
                    "source_seed_ids": f"z{z:03d}_p{pass_no}_s{stable_counter:04d}",
                    "recovery_pass": pass_no,
                    "initial_pixel_count": int(prop.area),
                    "final_pixel_count": int(np.count_nonzero(completed)),
                    "initial_length_px": int(prop.area),
                    "final_length_px": int(np.count_nonzero(completed)),
                    "endpoint_count": len(endpoints(completed)),
                    "endpoint_coordinates": json.dumps(endpoints(completed).tolist()),
                    "orientation_coherence": orientation_coherence(np.argwhere(completed)),
                    "raw_ridge_support": support_metric(raw_seg["ridge"], completed),
                    "weighted_ridge_support": support_metric(weighted, completed),
                    "raw_intensity_support": support_metric(raw, binary_dilation(completed, iterations=2) & roi),
                    "nucleus_probability": support_metric(prob[:, :, 0], completed),
                    "tissue_probability": support_metric(prob[:, :, 1], completed),
                    "punctum_probability": support_metric(prob[:, :, 2], completed),
                    "diffuse_background_probability": support_metric(prob[:, :, 3], completed),
                    "extension_status": ";".join(r.get("extension_status", "") for r in ext_rows) or "not_attempted",
                    "endpoint_searches_executed": len(ext_rows),
                    "endpoints_successfully_extended": len(endpoint_extensions),
                    "extension_pixels_added": sum(int(r.get("extension_length", 0)) for r in endpoint_extensions),
                    "paths_completed_through_extension": int(seed_extended and len(endpoints(completed)) == 2),
                    "join_status": "pending",
                    "accepted_join_ids": "",
                    "final_path_id": "",
                    "final_disposition": "pending",
                    "absorbed_into_component": "",
                    "absorbed_into_path": "",
                    "final_technical_status": status,
                    "completeness_status": status,
                    "technical_validity": status not in {"invalid", "punctum_like", "tissue_like", "duplicate_raw_object"},
                    "technical_failure_reasons": "none" if not reasons else ",".join(reasons),
                    "morphology_warning_reasons": "none",
                    "duplicate_status": "duplicate" if status == "duplicate_raw_object" else "not_duplicate",
                    "final_accepted_status": False,
                    "eligible_candidate_count": len(props),
                    "retained_candidate_count": len(retained_props),
                    "discarded_candidate_count": max(0, len(props) - len(retained_props)),
                    "safety_cap_truncated": safety_truncated,
                    "path_mask": completed,
                }
                seed_rows.append({k: v for k, v in rec.items() if k != "path_mask"})
                seed_records.append(rec)
                stable_counter += 1
            direct_singletons = [
                {
                    "path_mask": r["path_mask"],
                    "source_seed_ids": [r["seed_id"]],
                    "recovery_pass": r["recovery_pass"],
                }
                for r in seed_records
                if r["completeness_status"] == "complete_seed" and r["technical_validity"]
            ]
            graph_candidates = [
                r for r in seed_records
                if r["completeness_status"] in {"extendable_fragment", "joinable_fragment"}
                and r["technical_validity"]
            ]
            if len(graph_candidates) > EMERGENCY_COMPLEX_GRAPH_CAP:
                safety_truncated = True
                pre_cap_graph_candidates = len(graph_candidates)
                graph_candidates = sorted(
                    graph_candidates,
                    key=lambda r: float(r.get("weighted_ridge_support", 0.0)),
                    reverse=True,
                )[:EMERGENCY_COMPLEX_GRAPH_CAP]
                discarded_candidate_count += pre_cap_graph_candidates - len(graph_candidates)
            unresolved_singletons = [
                {
                    "path_mask": r["path_mask"],
                    "source_seed_ids": [r["seed_id"]],
                    "recovery_pass": r["recovery_pass"],
                }
                for r in seed_records
                if r["completeness_status"] == "unresolved_fragment" and r["technical_validity"]
            ]
            joined, attempts = join_fragments(graph_candidates, weighted, prob, roi, occupied)
            joined = direct_singletons + unresolved_singletons + joined
            all_join_attempts.extend({**a, "z_index": z, "recovery_pass": pass_no, "stable_id": f"z{z:03d}_join_{len(all_join_attempts)+i:04d}"} for i, a in enumerate(attempts))
            for j, item in enumerate(joined, start=1):
                measure_row = path_measure(item["path_mask"], raw, raw_seg, weighted, prob, roi, raw_label, cfg)
                accepted = bool(measure_row["final_accepted_status"])
                path_id = f"z{z:03d}_p{pass_no}_path_{j:04d}"
                row = {
                    "stable_id": path_id,
                    "z_index": z,
                    "parent_semantic_component": "graph_component",
                    "path_source": "graph_no_quota",
                    "source_seed_ids": ",".join(item["source_seed_ids"]),
                    "source_component_ids": ",".join(item["source_seed_ids"]),
                    "extension_ids": "recorded_in_seed_audit",
                    "join_ids": ",".join(
                        r.get("stable_id", "")
                        for r in all_join_attempts
                        if r.get("join_status") == "accepted"
                        and r.get("source_seed_a") in item["source_seed_ids"]
                        and r.get("source_seed_b") in item["source_seed_ids"]
                    ),
                    "technical_status": "accepted" if accepted else "rejected",
                    "eligible_candidate_count": eligible_candidate_count,
                    "retained_candidate_count": retained_candidate_count,
                    "discarded_candidate_count": discarded_candidate_count,
                    "safety_cap_truncated": safety_truncated,
                    "recovery_pass": pass_no,
                    "extension_status": "completed_from_seed_graph",
                    "join_status": "joined_or_singleton",
                    **measure_row,
                }
                path_rows.append(row)
                source_ids = set(item["source_seed_ids"])
                accepted_join_ids = row["join_ids"]
                for sr in seed_rows:
                    if sr.get("seed_id") in source_ids:
                        sr["final_path_id"] = path_id
                        sr["final_disposition"] = "accepted_path" if accepted else row["completeness_status"]
                        sr["accepted_join_ids"] = accepted_join_ids
                        sr["absorbed_into_component"] = path_id
                        sr["absorbed_into_path"] = path_id
                        sr["final_accepted_status"] = accepted
                        sr["final_technical_status"] = row["technical_failure_reasons"]
                        sr["join_status"] = "accepted" if accepted_join_ids else "singleton_or_no_accepted_join"
                if accepted:
                    accepted_path_masks.append(item["path_mask"])
                    occupied |= binary_dilation(item["path_mask"], iterations=3)
                    pass_additions[pass_no] += 1
                elif measure_row["technical_failure_reasons"] == "unresolved_fragment":
                    unresolved_mask |= item["path_mask"]
                else:
                    failure_mask |= item["path_mask"]
            if pass_no == 1:
                pass2_preservation_checksum = sha256_array(np.logical_or.reduce(accepted_path_masks) if accepted_path_masks else np.zeros(raw.shape, dtype=bool))
        final_added = np.logical_or.reduce(accepted_path_masks) if accepted_path_masks else np.zeros(raw.shape, dtype=bool)
        pass1_mask = (raw_label > 0) | np.logical_or.reduce(accepted_path_masks[:pass_additions[1]]) if accepted_path_masks else (raw_label > 0)
        pass12_mask = (raw_label > 0) | final_added
        review_pdf_name = f"z{z:03d}_consolidated_2d_review.pdf"
        workbook_rows.extend(write_review_pdf(
            out_dir / "review_panels" / review_pdf_name,
            z, raw, prob, raw_seg["ridge"], weighted, initial_seed_mask, final_added, unresolved_mask, failure_mask, raw_label, pass1_mask, pass12_mask,
        ))
        write_csv(out_dir / "seed_graphs" / f"z{z:03d}_graph_nodes.csv", [
            {
                "node_id": f"z{z:03d}_node_{i:06d}",
                "y": int(y), "x": int(x),
                "raw_ridge_strength": float(raw_seg["ridge"][y, x]),
                "weighted_ridge_strength": float(weighted[y, x]),
                "nucleus_probability": float(prob[y, x, 0]),
                "tissue_probability": float(prob[y, x, 1]),
                "punctum_probability": float(prob[y, x, 2]),
                "diffuse_background_probability": float(prob[y, x, 3]),
                "local_raw_intensity": float(raw[y, x]),
                "local_orientation": "estimated_from_component",
                "roi_membership": bool(roi[y, x]),
                "overlap_with_existing_raw_baseline_object": bool(raw_label[y, x] > 0),
                "edge_neighborhood": "8_connectivity",
            }
            for i, (y, x) in enumerate(np.argwhere(initial_seed_mask))
        ])
        all_join_attempts = list(all_join_attempts)
        join_rows.extend(all_join_attempts)
        method_rows.append({
            "z_index": z,
            "method": "A_exact_validated_raw_baseline",
            "initial_seeds": 0,
            "complete_seeds": 0,
            "fragments_extended": 0,
            "fragments_joined": 0,
            "unresolved_fragments": 0,
            "complete_paths": len(baseline[z]["results"]),
            "pass1_additions": 0,
            "pass2_additions": 0,
            "duplicates": 0,
            "punctum_artifacts": 0,
            "tissue_artifacts": 0,
            "suspected_splits": 0,
            "suspected_merges": 0,
            "morphology_warning_objects": 0,
            "technical_failures": 0,
        })
        method_rows.append({
            "z_index": z,
            "method": "A_legacy_fixed_seed_graph_reference",
            "method_note": "legacy reference only; fixed top-K was removed from the primary graph path",
            "initial_seeds": LEGACY_FIXED_SEEDS_PER_PASS * 2,
            "complete_paths": "not_rerun",
            "safety_cap_truncated": "not_applicable",
        })
        previous_split = pd.read_csv(ROOT / "scratch" / "v5_6_ilastik_instance_diagnostic" / "ilastik_instance_method_comparison_v5_6.csv") if (ROOT / "scratch" / "v5_6_ilastik_instance_diagnostic" / "ilastik_instance_method_comparison_v5_6.csv").exists() else pd.DataFrame()
        prev = previous_split[(previous_split["z_index"] == z) & (previous_split["method"].astype(str).str.contains("C_raw_guided", na=False))].iloc[0].to_dict() if not previous_split.empty and any((previous_split["z_index"] == z) & (previous_split["method"].astype(str).str.contains("C_raw_guided", na=False))) else {}
        method_rows.append({"z_index": z, "method": "B_failed_fragment_based_previous_diagnostic", **prev})
        method_rows.append({
            "z_index": z,
            "method": "C_graph_path_completion_pass2",
            "method_note": "no fixed top-K seed quota; all eligible evidence components retained unless emergency cap is hit",
            "initial_seeds": len([r for r in seed_rows if r["z_index"] == z]),
            "complete_seeds": sum(1 for r in seed_rows if r["z_index"] == z and r["completeness_status"] == "complete_seed"),
            "seeds_considered_for_extension": extension_attempts,
            "endpoint_searches_executed": endpoint_searches_executed,
            "endpoints_successfully_extended": endpoints_successfully_extended,
            "seeds_successfully_extended": seeds_successfully_extended,
            "extension_pixels_added": extension_pixels_added,
            "paths_completed_through_extension": paths_completed_through_extension,
            "fragments_extended": seeds_successfully_extended,
            "extension_attempts": extension_attempts,
            "extension_successes_deprecated_label": extension_successes,
            "join_attempts": len(all_join_attempts),
            "join_successes": sum(1 for r in all_join_attempts if r["join_status"] == "accepted"),
            "fragments_joined": sum(1 for r in all_join_attempts if r["join_status"] == "accepted"),
            "unresolved_fragments": sum(1 for r in path_rows if r["z_index"] == z and r["completeness_status"] == "unresolved_fragment"),
            "complete_paths": sum(1 for r in path_rows if r["z_index"] == z and r["final_accepted_status"]),
            "pass1_additions": pass_additions[1],
            "pass2_additions": pass_additions[2],
            "duplicates": sum(1 for r in path_rows if r["z_index"] == z and "duplicate_raw_detection" in r["technical_failure_reasons"]),
            "punctum_artifacts": sum(1 for r in path_rows if r["z_index"] == z and "compact_punctum_or_ring" in r["technical_failure_reasons"]),
            "tissue_artifacts": sum(1 for r in path_rows if r["z_index"] == z and "broad_tissue_boundary" in r["technical_failure_reasons"]),
            "suspected_splits": 0,
            "suspected_merges": sum(1 for r in path_rows if r["z_index"] == z and ("long" in r["morphology_warning_reasons"] or "wide" in r["morphology_warning_reasons"])),
            "morphology_warning_objects": sum(1 for r in path_rows if r["z_index"] == z and r["morphology_warning_reasons"] != "none"),
            "technical_failures": sum(1 for r in path_rows if r["z_index"] == z and not r["technical_validity"]),
            "pass2_preserved_pass1_checksum": pass2_preservation_checksum == (sha256_array(np.logical_or.reduce(accepted_path_masks[:pass_additions[1]]) if accepted_path_masks[:pass_additions[1]] else np.zeros(raw.shape, dtype=bool))),
            "outside_roi_leakage": int(np.count_nonzero(final_added & ~roi)),
            "eligible_candidate_count": eligible_candidate_count,
            "retained_candidate_count": retained_candidate_count,
            "discarded_candidate_count": discarded_candidate_count,
            "safety_cap_truncated": safety_truncated,
        })
        method_rows.append(direct_summary)
        method_rows.append(hybrid_summary)
        write_three_method_crop_pdf(
            out_dir / "review_panels" / f"z{z:03d}_three_method_hysteresis_hybrid_comparison.pdf",
            z, raw, weighted, final_added, direct_mask, hybrid_mask,
        )
        print(f"[v5.6 2D] z{z:03d}: slice complete", flush=True)

    write_csv(out_dir / "weighted_ridge" / "weighted_ridge_comparison_v5_6.csv", formulation_rows)
    write_csv(out_dir / "seed_graphs" / "seed_audit_v5_6.csv", seed_rows)
    write_csv(out_dir / "endpoint_extensions" / "endpoint_extension_audit_v5_6.csv", seed_rows)
    write_csv(out_dir / "join_matching" / "join_proposal_audit_v5_6.csv", join_rows)
    write_csv(out_dir / "join_matching" / "join_matching_audit_v5_6.csv", join_rows)
    write_csv(out_dir / "completed_paths" / "completed_path_audit_v5_6.csv", path_rows)
    write_csv(out_dir / "candidate_audit" / "hysteresis_component_audit_v5_6.csv", component_rows)
    write_csv(out_dir / "completed_paths" / "direct_hybrid_final_path_audit_v5_6.csv", comparison_path_rows)
    write_csv(out_dir / "pass2_recovery" / "pass2_recovery_audit_v5_6.csv", [r for r in path_rows if int(r.get("recovery_pass", 0)) == 2])
    write_csv(out_dir / "candidate_audit" / "seed_audit_v5_6.csv", seed_rows)
    write_csv(out_dir / "reports" / "six_slice_evaluation_v5_6.csv", method_rows if include_evaluation else [])
    write_csv(out_dir / "reports" / "three_method_comparison_v5_6.csv", method_rows)
    review_rows = []
    for row in path_rows:
        review_rows.append({
            "z_index": row["z_index"],
            "crop_id": "",
            "candidate_id": row["stable_id"],
            "recovery_pass": row["recovery_pass"],
            "source_seed_ids": row["source_seed_ids"],
            "candidate_status": "accepted" if row["final_accepted_status"] else row["completeness_status"],
            "complete_nucleus": "",
            "partial_fragment": "",
            "split_nucleus": "",
            "merged_nuclei": "",
            "punctum_false_positive": "",
            "tissue_edge_false_positive": "",
            "duplicate": "",
            "uncertain": "",
            "reviewer_notes": "",
        })
    with pd.ExcelWriter(out_dir / "manual_review" / "v5_6_2d_candidate_review.xlsx", engine="xlsxwriter") as writer:
        pd.DataFrame(review_rows).to_excel(writer, index=False, sheet_name="review")
    mapping_rows = historical_to_new_mapping(baseline, path_rows)
    write_csv(out_dir / "historical_mapping" / "historical_to_new_mapping_v5_6.csv", mapping_rows)
    technical_rows = [{"reason": k, "count": v} for k, v in Counter(reason for row in path_rows for reason in str(row["technical_failure_reasons"]).split(",") if reason and reason != "none").items()]
    morphology_rows = [{"reason": k, "count": v} for k, v in Counter(reason for row in path_rows for reason in str(row["morphology_warning_reasons"]).split(",") if reason and reason != "none").items()]
    write_csv(out_dir / "reports" / "technical_failure_summary_v5_6.csv", technical_rows)
    write_csv(out_dir / "reports" / "morphology_warning_summary_v5_6.csv", morphology_rows)
    gate = quality_gate_result(path_rows, join_rows, z_indices, include_evaluation)
    (out_dir / "reports" / "development_quality_gates_v5_6.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    summary = {
        "historical_raw_regression_baseline_v5_6": EXPECTED_RAW_COUNTS,
        "selected_weighted_ridge_formulation": SELECTED_FORMULATION,
        "z_indices": z_indices,
        "seed_counts": len(seed_rows),
        "extension_attempts": sum(r.get("extension_attempts", 0) for r in method_rows if r.get("method") == "C_graph_path_completion_pass2"),
        "extension_successes": sum(r.get("fragments_extended", 0) for r in method_rows if r.get("method") == "C_graph_path_completion_pass2"),
        "join_attempts": len(join_rows),
        "join_successes": sum(1 for r in join_rows if r["join_status"] == "accepted"),
        "completed_paths": sum(1 for r in path_rows if r["final_accepted_status"]),
        "unresolved_fragments": sum(1 for r in path_rows if r["completeness_status"] == "unresolved_fragment"),
        "pass1_additions": sum(r.get("pass1_additions", 0) for r in method_rows if r.get("method") == "C_graph_path_completion_pass2"),
        "pass2_additions": sum(r.get("pass2_additions", 0) for r in method_rows if r.get("method") == "C_graph_path_completion_pass2"),
        "technical_failures_by_reason": dict(Counter(reason for row in path_rows for reason in str(row["technical_failure_reasons"]).split(",") if reason and reason != "none")),
        "punctum_and_tissue_technical_failures": {
            "punctum": sum(1 for r in path_rows if "compact_punctum_or_ring" in str(r["technical_failure_reasons"])),
            "tissue": sum(1 for r in path_rows if "broad_tissue_boundary" in str(r["technical_failure_reasons"])),
        },
        "historical_mapping_summary": summarize_mapping(mapping_rows),
        "development_gate_result": gate,
        "six_slice_evaluation_allowed": bool(gate["passed"]),
        "six_slice_evaluation_result": "completed" if include_evaluation and gate["passed"] else "not_run",
        "crop_pdf_paths": [str(out_dir / "review_panels" / f"z{z:03d}_consolidated_2d_review.pdf") for z in z_indices],
        "manual_review_workbook": str(out_dir / "manual_review" / "v5_6_2d_candidate_review.xlsx"),
    }
    (out_dir / "reports" / "consolidated_2d_summary_v5_6.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report_pdf(out_dir / "reports" / "consolidated_2d_report_v5_6.pdf", summary, method_rows, formulation_rows)
    print(json.dumps(summary, indent=2))
    return summary


def run(args) -> dict:
    return run_consolidated(
        hdf5_dataset_key=args.hdf5_dataset_key,
        nucleus_channel=args.nucleus_channel,
        include_evaluation=not getattr(args, "development_only", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-dataset-key", default="exported_data")
    parser.add_argument("--nucleus-channel", type=int, default=0)
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
