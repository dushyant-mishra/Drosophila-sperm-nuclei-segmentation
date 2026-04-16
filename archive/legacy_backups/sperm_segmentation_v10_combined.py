#!/usr/bin/env python3
"""
Spermatid Segmentation Pipeline  v8
=====================================
Skeleton-first detection of thin, dim, partially fragmented spermatids.

Changes vs v7  (all verified against real z04 data before committing)
----------------------------------------------------------------------
PARAMETER CHANGES
  P1. MAX_BRIDGE_PX:  10 → 5
      At 10, nearby but distinct spermatids were chained into long networks
      (max geodesic 303–399 µm).  At 5, max stays below 60 µm and N is
      higher (238 vs 96) because individual fragments are less over-merged.

  P2. MAX_WIDTH_PX:  6.5 → 5.5
      p90 of true spermatid width distribution is 6.4 px.  Tightening to
      5.5 removes the chunky outliers without losing narrow true positives.

  P3. MIN_LENGTH_WIDTH_RATIO:  3.0 → 3.5
      Removes rounder/stubbier detections from the tail of the ratio
      distribution while keeping the elongated majority.

  P4. MIN_SKEL_LEN_PX:  8 → 8 (unchanged)
      With bridge=5, fragments are shorter than with bridge=10, so 8 px
      (≈3 µm) is the right lower bound. 10 would lose too many real fragments.

NEW FILTERS
  N1. MAX_BRANCH_NODES  (replaces the boolean REJECT_BRANCHES)
      v7's boolean REJECT_BRANCHES=True rejected 76 of 95 valid spermatids
      because skeleton-level bridging creates branch nodes at every junction
      point (where the bridge line meets the original skeleton).  A single
      bridged spermatid has exactly 1 such node.  A genuine merged network
      has many.  The new parameter rejects components with more than
      MAX_BRANCH_NODES branch points (pixels with 3+ skeleton neighbours).
      Default: 4  — tolerates a bridged spermatid (1 node) with some
      measurement noise, rejects sprawling networks.
      Set to 0 to replicate the old REJECT_BRANCHES=True behaviour.
      Set to 9999 to disable entirely.

  N2. MAX_TORTUOSITY
      tortuosity = geodesic_length / euclidean(endpoint_A, endpoint_B)
      A straight spermatid has tortuosity ≈ 1.  A curved one can reach
      ≈ 1.5–2.  A merged network snaking back on itself easily exceeds 3.
      Measured on real data: p90 of narrow high-ratio components = 2.78,
      p90 of wide/highly-branched components = 6.4+.
      Default: 2.5  — removes 13/88 endpoint components with benign loss.
      Only applied to components with ≥ 2 endpoints (open filaments).
      Set to 9999 to disable.

  N3. MAX_ENDPOINT_COUNT
      A clean spermatid has 2 endpoints (both tips).  Bridging two fragments
      gives 2 endpoints (the outer tips merge, internal ones disappear).
      Components with many endpoints are complex branched networks.
      Default: 6  — removes 9 components with minimal false-negative risk.

  DEPRECATED
      "REJECT_BRANCHES" boolean removed.  Use MAX_BRANCH_NODES instead:
        REJECT_BRANCHES=True  ↔  MAX_BRANCH_NODES=0
        REJECT_BRANCHES=False ↔  MAX_BRANCH_NODES=9999

Usage
-----
  python sperm_segmentation_v8.py              # uses RUN_MODE in CONFIG
  python sperm_segmentation_v8.py --batch
  python sperm_segmentation_v8.py --single --z 15
"""

import os, sys, glob, re, time, warnings, heapq, argparse
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tifffile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
print(f"[matplotlib backend: {matplotlib.get_backend()}]")

from skimage import measure, morphology, exposure
from skimage.filters import meijering, gaussian, apply_hysteresis_threshold
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, grey_dilation
from scipy.spatial import cKDTree
from matplotlib.backends.backend_pdf import PdfPages

try:
    import cv2 as _cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # ── run mode ─────────────────────────────────────────────────────────────
    "RUN_MODE": "single",          # "single" | "batch"

    # ── single-image selection ────────────────────────────────────────────────
    "SINGLE_IMAGE_SELECTION_MODE": "dialog",  # "path" | "z_index" | "dialog"
    "SINGLE_TEST_IMAGE": r"C:\Users\dmishra\Desktop\sperm images\Project001_Series002_z15_ch00.tif",
    "SINGLE_Z_INDEX": 15,

    # ── input / output ────────────────────────────────────────────────────────
    "INPUT_DIR":    r"C:\Users\dmishra\Desktop\sperm images",
    "OUTPUT_DIR":   r"C:\Users\dmishra\Desktop\sperm images\sperm_results_v8",
    "FILE_PATTERN": "Project001_Series002_z*_ch00.tif",

    # ── calibration ───────────────────────────────────────────────────────────
    "UM_PER_PX_XY":   0.378788,
    "UM_PER_SLICE_Z": 0.346184,

    # ── image enhancement ─────────────────────────────────────────────────────
    "CLAHE_CLIP":   0.120,   # Raised significantly to pull out dim edges inside solid center cysts
    "CLAHE_KERNEL": 64,
    "BG_SIGMA":     10.9,    # Widened background subtraction to flatten out the solid centers
    "RIDGE_SIGMAS": [1, 2, 3, 4],

    # ── hysteresis threshold ──────────────────────────────────────────────────
    "THRESHOLD_HI": 90.1,    # High percentile suppresses all background noise cleanly
    "THRESHOLD_LO": 81.0,    # Firm lower bound stops fragmentation

    # ── morphological cleanup ─────────────────────────────────────────────────
    "CLOSE_RADIUS":  1,
    "MIN_HOLE_AREA": 10,
    "MIN_OBJ_PX":    5,

    # ── skeleton-level gap bridging ───────────────────────────────────────────
    # P1: reduced from 10 → 5 to prevent chaining distinct spermatids
    "MAX_BRIDGE_PX": 5,

    # ── branch pruning & automated splitting ──────────────────────────────────
    "MAX_BRANCH_LEN_PX": 5,   # prune spurs shorter than this before measuring
    "BREAK_JUNCTIONS": True,  # automatically sever all branching intersections into distinct lines

    # ── optional early mask-level shape filter ────────────────────────────────
    "USE_EARLY_SHAPE_FILTER": False,
    "MIN_ECCENTRICITY": 0.60,
    "MAX_MINOR_PX":     12.0,
    "MIN_AXIS_RATIO":   1.4,
    "MIN_MAJOR_PX":     5,

    # ── post-skeleton filters ─────────────────────────────────────────────────
    "MIN_SKEL_LEN_PX":        13.9, # min geodesic length (px)
    "MAX_GEODESIC_LEN_PX":    65.0, # (~13.0um); backend actively cuts chains longer than this
    "MAX_WIDTH_PX":           25.2, # Wide enough to allow multi-stranded clusters to survive
    "MIN_LENGTH_WIDTH_RATIO": 1.6,  # Relaxed to allow clustered blobs to pass

    # ── NEW topology filters ──────────────────────────────────────────────────
    "MAX_BRANCH_NODES": 872,      # Allowed to be massive so user can Split the resulting thick webs
    "MAX_TORTUOSITY": 840.6,      # Relaxed to let extremely zig-zagging clusters live safely
    "MAX_ENDPOINT_COUNT": 734,    # Relaxed so massive webs aren't dropped outright

    # N4: loops
    "ALLOW_LOOPS": True,     # don't reject spermatids that cross themselves  # moderate

    # ── tracking across z ─────────────────────────────────────────────────────
    "DO_TRACKING":          True,
    "TRACK_MAX_DIST_UM":    4.0,
    "TRACK_MAX_GAP_SLICES": 1,

    # ── output / debug ────────────────────────────────────────────────────────
    "SAVE_DEBUG_IMAGES":   True,
    "SAVE_MASK_TIFS":      True,
    "SAVE_LABEL_TIFS":     True,
    "SAVE_OVERLAYS":       True,
    "SAVE_DETAIL_FIGURE":  True,
    "SHOW_PREVIEW_WINDOW": True,
    "SHOW_DEBUG_PREVIEW":  True,
}


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

_REQUIRED = {
    "RUN_MODE": str, "SINGLE_IMAGE_SELECTION_MODE": str,
    "SINGLE_TEST_IMAGE": str, "SINGLE_Z_INDEX": int,
    "INPUT_DIR": str, "OUTPUT_DIR": str, "FILE_PATTERN": str,
    "UM_PER_PX_XY": float, "UM_PER_SLICE_Z": float,
    "CLAHE_CLIP": float, "CLAHE_KERNEL": int, "BG_SIGMA": (int, float),
    "RIDGE_SIGMAS": list,
    "THRESHOLD_HI": (int, float), "THRESHOLD_LO": (int, float),
    "CLOSE_RADIUS": int, "MIN_HOLE_AREA": int, "MIN_OBJ_PX": int,
    "MAX_BRIDGE_PX": (int, float), "MAX_BRANCH_LEN_PX": (int, float),
    "USE_EARLY_SHAPE_FILTER": bool,
    "MIN_SKEL_LEN_PX": (int, float), "MAX_GEODESIC_LEN_PX": (int, float),
    "MAX_WIDTH_PX": (int, float), "MIN_LENGTH_WIDTH_RATIO": (int, float),
    "MAX_BRANCH_NODES": (int, float),
    "MAX_TORTUOSITY": (int, float),
    "MAX_ENDPOINT_COUNT": (int, float),
    "DO_TRACKING": bool, "TRACK_MAX_DIST_UM": (int, float),
    "TRACK_MAX_GAP_SLICES": int,
    "SAVE_DEBUG_IMAGES": bool, "SAVE_MASK_TIFS": bool,
    "SAVE_LABEL_TIFS": bool, "SAVE_OVERLAYS": bool,
    "SAVE_DETAIL_FIGURE": bool, "SHOW_PREVIEW_WINDOW": bool,
    "SHOW_DEBUG_PREVIEW": bool,
}


def validate_config(cfg):
    errors = []
    for key, expected in _REQUIRED.items():
        if key not in cfg:
            errors.append(f"  MISSING: '{key}'")
        elif not isinstance(cfg[key], expected):
            errors.append(f"  WRONG TYPE '{key}': "
                          f"got {type(cfg[key]).__name__}, want {expected}")
    if cfg.get("THRESHOLD_LO", 0) >= cfg.get("THRESHOLD_HI", 100):
        errors.append("  THRESHOLD_LO must be < THRESHOLD_HI")
    if cfg.get("RUN_MODE", "") not in ("single", "batch"):
        errors.append("  RUN_MODE must be 'single' or 'batch'")
    if "REJECT_BRANCHES" in cfg:
        errors.append("  'REJECT_BRANCHES' was removed in v8. "
                      "Use MAX_BRANCH_NODES (int) instead:\n"
                      "    REJECT_BRANCHES=True  → MAX_BRANCH_NODES=0\n"
                      "    REJECT_BRANCHES=False → MAX_BRANCH_NODES=9999")
    if errors:
        raise ValueError("CONFIG errors:\n" + "\n".join(errors))


# =============================================================================
# UTILITIES
# =============================================================================

_N8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def extract_z_index(path):
    m = re.search(r"_z(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def normalize_display(img):
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def _imwrite(path, arr_uint8, cmap="gray"):
    if _HAVE_CV2:
        if arr_uint8.ndim == 2:
            _cv2.imwrite(path, arr_uint8)
        else:
            _cv2.imwrite(path, _cv2.cvtColor(arr_uint8, _cv2.COLOR_RGB2BGR))
    else:
        plt.imsave(path, arr_uint8,
                   cmap=(cmap if arr_uint8.ndim == 2 else None),
                   vmin=0, vmax=255)


def save_gray(path, img_float):
    a = img_float.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    _imwrite(path, (a * 255).astype(np.uint8), cmap="gray")


def save_mask(path, mask_bool):
    _imwrite(path, mask_bool.astype(np.uint8) * 255, cmap="gray")


def load_batch_files(input_dir, pattern):
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        files = glob.glob(os.path.join(input_dir,
                                        pattern.replace(".tif", ".tiff")))
    if not files:
        raise FileNotFoundError(f"No files: '{pattern}' in '{input_dir}'")
    files = sorted(files, key=extract_z_index)
    z_idx = [extract_z_index(f) for f in files]
    print(f"Found {len(files)} slices: Z = {z_idx}")
    return files, z_idx


def choose_single_image(cfg):
    mode = cfg["SINGLE_IMAGE_SELECTION_MODE"].lower()
    if mode == "path":
        p = cfg["SINGLE_TEST_IMAGE"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")
        return p
    if mode == "z_index":
        z = int(cfg["SINGLE_Z_INDEX"])
        files = (glob.glob(os.path.join(cfg["INPUT_DIR"], cfg["FILE_PATTERN"])) or
                 glob.glob(os.path.join(cfg["INPUT_DIR"],
                                        cfg["FILE_PATTERN"].replace(".tif", ".tiff"))))
        hits = [f for f in files if extract_z_index(f) == z]
        if not hits:
            raise FileNotFoundError(f"No file for z={z}")
        print(f"Using z={z}: {hits[0]}")
        return hits[0]
    if mode == "dialog":
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            p = filedialog.askopenfilename(
                title="Choose image", initialdir=cfg["INPUT_DIR"],
                filetypes=[("TIFF", "*.tif *.tiff"), ("All", "*.*")])
            root.destroy()
        except Exception as e:
            raise RuntimeError(f"File dialog failed: {e}")
        if not p:
            raise RuntimeError("No file selected.")
        print(f"Selected: {p}")
        return p
    raise ValueError("SINGLE_IMAGE_SELECTION_MODE: 'path'|'z_index'|'dialog'")


# =============================================================================
# SKELETON UTILITIES
# =============================================================================

def find_endpoints(skel_bool):
    H, W = skel_bool.shape
    ys, xs = np.where(skel_bool)
    sk_set = set(zip(ys.tolist(), xs.tolist()))
    return [(r, c) for r, c in sk_set
            if sum(1 for dr, dc in _N8
                   if 0 <= r+dr < H and 0 <= c+dc < W
                   and skel_bool[r+dr, c+dc]) == 1]


def bridge_skeleton_endpoints(skel_bool, skel_labeled, max_gap_px):
    """
    Join endpoint pairs from DIFFERENT components within max_gap_px by a
    1-px straight line.  Preserves the original mask so width estimates
    are not inflated.
    """
    if max_gap_px <= 0:
        return skel_bool.copy()
    H, W   = skel_bool.shape
    out    = skel_bool.copy()
    eps    = find_endpoints(skel_bool)
    if not eps:
        return out
    ep_arr    = np.array(eps, dtype=np.float32)
    ep_labels = skel_labeled[ep_arr[:, 0].astype(int), ep_arr[:, 1].astype(int)]
    pairs     = cKDTree(ep_arr).query_pairs(r=max_gap_px, output_type="ndarray")
    for i, j in pairs:
        if ep_labels[i] == ep_labels[j]:
            continue
        r0, c0 = int(eps[i][0]), int(eps[i][1])
        r1, c1 = int(eps[j][0]), int(eps[j][1])
        n = max(abs(r1-r0), abs(c1-c0)) + 1
        rs = np.clip(np.round(np.linspace(r0, r1, n)).astype(int), 0, H-1)
        cs = np.clip(np.round(np.linspace(c0, c1, n)).astype(int), 0, W-1)
        out[rs, cs] = True
    return out


def prune_branches(skel_bool, max_branch_len):
    """
    Iteratively remove endpoints to shorten side-branches ≤ max_branch_len px.
    """
    if max_branch_len <= 0:
        return skel_bool.copy()
    H, W  = skel_bool.shape
    skel  = skel_bool.copy()
    for _ in range(int(max_branch_len)):
        eps = find_endpoints(skel)
        if not eps:
            break
        for r, c in eps:
            n = sum(1 for dr, dc in _N8
                    if 0 <= r+dr < H and 0 <= c+dc < W and skel[r+dr, c+dc])
            if n == 1:
                skel[r, c] = False
    return skel


# =============================================================================
# GEODESIC & TOPOLOGY MEASUREMENT
# =============================================================================

def _build_adj(coords, W):
    n       = len(coords)
    lin     = coords[:, 0] * W + coords[:, 1]
    lin2idx = {int(v): i for i, v in enumerate(lin.tolist())}
    lin_set = set(lin.tolist())
    adj     = [[] for _ in range(n)]
    for i, (r, c) in enumerate(coords.tolist()):
        for dr, dc in _N8:
            lk = (r + dr) * W + (c + dc)
            if lk in lin_set:
                w = 1.41421356 if (dr != 0 and dc != 0) else 1.0
                adj[i].append((lin2idx[lk], w))
    return adj


def _dijkstra(adj, src, n):
    d = np.full(n, np.inf); d[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        cost, u = heapq.heappop(pq)
        if cost > d[u]:
            continue
        for v, w in adj[u]:
            nd = cost + w
            if nd < d[v]:
                d[v] = nd
                heapq.heappush(pq, (nd, v))
    far = int(np.argmax(d))
    return far, float(d[far])


def measure_topology(coords, W, allow_loops=False):
    """
    Compute geodesic length, tortuosity, endpoint count, and branch-node count
    for one skeleton component.

    Returns
    -------
    dict with keys: geo_len, tortuosity, n_endpoints, n_branch_nodes, reason
    or None if the component is a loop and allow_loops=False.
    """
    n   = len(coords)
    adj = _build_adj(coords, W)
    deg = [len(a) for a in adj]

    n_endpoints    = sum(1 for d in deg if d == 1)
    n_branch_nodes = sum(1 for d in deg if d >  2)
    eps_idx        = [i for i, d in enumerate(deg) if d == 1]

    # ── Loop handling ─────────────────────────────────────────────────────────
    if not eps_idx:
        if not allow_loops:
            return None  # discard loop
        seen, total = set(), 0.0
        for u, nbrs in enumerate(adj):
            for v, w in nbrs:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key); total += w
        return {"geo_len": total, "tortuosity": 1.0,
                "n_endpoints": 0, "n_branch_nodes": n_branch_nodes,
                "reason": "loop"}

    # ── Double-BFS for geodesic ───────────────────────────────────────────────
    b, _  = _dijkstra(adj, eps_idx[0], n)
    c, gl = _dijkstra(adj, b, n)

    # ── Tortuosity ────────────────────────────────────────────────────────────
    p0  = coords[b]
    p1  = coords[c]
    euc = float(np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2))
    tort = gl / (euc + 1e-9)

    return {"geo_len": float(gl), "tortuosity": tort,
            "n_endpoints": n_endpoints, "n_branch_nodes": n_branch_nodes,
            "reason": "ok"}


# =============================================================================
# SEGMENTATION PIPELINE
# =============================================================================

def apply_optional_early_shape_filter(mask, cfg):
    if not cfg["USE_EARLY_SHAPE_FILTER"]:
        return mask
    labeled = measure.label(mask)
    keep = [p.label for p in measure.regionprops(labeled)
            if (p.eccentricity      >= cfg["MIN_ECCENTRICITY"] and
                p.minor_axis_length <= cfg["MAX_MINOR_PX"]     and
                p.major_axis_length / (p.minor_axis_length + 1e-9) >= cfg["MIN_AXIS_RATIO"] and
                p.major_axis_length >= cfg["MIN_MAJOR_PX"])]
    return np.isin(labeled, keep)


def segment_slice(img_raw, cfg, z_idx=None, debug_dir=None, roi_mask=None):
    img = img_raw.astype(np.float32)
    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
        if roi_mask.shape != img.shape:
            raise ValueError(f"roi_mask shape {roi_mask.shape} does not match image shape {img.shape}")
        # NOTE: do NOT zero out pixels here — that destroys image statistics
        # (CLAHE, percentiles, ridge detection).  Instead, process full image
        # and apply roi_mask only at the mask/skeleton stages below.

    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)

    img_eq = exposure.equalize_adapthist(
        img_norm, clip_limit=cfg["CLAHE_CLIP"], kernel_size=cfg["CLAHE_KERNEL"])

    bg  = gaussian(img_eq, sigma=cfg["BG_SIGMA"])
    fg  = np.clip(img_eq - bg, 0, None)
    fgn = fg / (fg.max() + 1e-9)

    ridge = meijering(fgn, sigmas=cfg["RIDGE_SIGMAS"], black_ridges=False)

    th_hi     = np.percentile(ridge, cfg["THRESHOLD_HI"])
    th_lo     = np.percentile(ridge, cfg["THRESHOLD_LO"])
    mask_hyst = apply_hysteresis_threshold(ridge, th_lo, th_hi)
    if roi_mask is not None:
        mask_hyst &= roi_mask

    mask_clean = morphology.binary_closing(mask_hyst, morphology.disk(cfg["CLOSE_RADIUS"]))
    mask_clean = morphology.remove_small_holes(mask_clean, area_threshold=cfg["MIN_HOLE_AREA"])
    mask_clean = morphology.remove_small_objects(mask_clean, min_size=cfg["MIN_OBJ_PX"])
    mask_clean = apply_optional_early_shape_filter(mask_clean, cfg)
    if roi_mask is not None:
        mask_clean &= roi_mask

    # Width is measured from the CLEAN (un-bridged) distance map
    dist_clean   = distance_transform_edt(mask_clean)
    skel_clean   = skeletonize(mask_clean)
    skel_labeled = measure.label(skel_clean)

    # Skeleton-level bridging (preserves mask / width integrity)
    skel_bridged    = bridge_skeleton_endpoints(
        skel_clean, skel_labeled, cfg["MAX_BRIDGE_PX"])
    if roi_mask is not None:
        skel_bridged &= roi_mask
    # Branch pruning before measurement
    skel_pruned     = prune_branches(skel_bridged, cfg["MAX_BRANCH_LEN_PX"])
    
    # Optional: Automatically sever complex webs into isolated individual strands
    if cfg.get("BREAK_JUNCTIONS", False):
        from scipy.ndimage import convolve
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.int32)
        skel_int = skel_pruned.astype(np.int32)
        neighbors = convolve(skel_int, kernel, mode='constant', cval=0)
        # Any skeleton pixel with more than 2 neighbors is a junction
        junctions = (skel_int > 0) & (neighbors > 2)
        skel_pruned[junctions] = 0
        
    if roi_mask is not None:
        skel_pruned &= roi_mask
    skel_labeled_fn = measure.label(skel_pruned)

    # NEW: The Recursive Adaptive Micro-Crop Reanalyzer for dense webs and chains
    if cfg.get("MAX_GEODESIC_LEN_PX", 0) > 0 and cfg.get("AUTO_LOCAL_REANALYSIS", True):
        max_px = cfg["MAX_GEODESIC_LEN_PX"]
        
        # Create a rigid dict of sub-parameters to forcefully shatter the isolated components
        sub_cfg = cfg.copy()
        sub_cfg["AUTO_LOCAL_REANALYSIS"] = False # Prevent infinite recursion
        
        # Because the bounding box crop contains almost NO dark background, 
        # a median 50% threshold perfectly isolates the brightest centers of the blobs!
        sub_cfg["THRESHOLD_HI"] = 55.0
        sub_cfg["THRESHOLD_LO"] = 45.0
        
        # Deactivate topological limits in the recursive sub-call so fragments aren't 
        # instantly dropped before they can be spliced back into the master image.
        sub_cfg["MIN_SKEL_LEN_PX"] = 1.0
        sub_cfg["MIN_OBJ_PX"] = 3
        sub_cfg["MIN_HOLE_AREA"] = 0

        props = measure.regionprops(skel_labeled_fn)
        for sp in props:
            # Note: A dense 2D web might have len(sp.coords) = 5000 pixels.
            if len(sp.coords) > max_px:
                minr, minc, maxr, maxc = sp.bbox
                pad = 12
                minr = max(0, minr - pad)
                minc = max(0, minc - pad)
                maxr = min(img.shape[0], maxr + pad)
                maxc = min(img.shape[1], maxc + pad)
                
                crop_img = img[minr:maxr, minc:maxc]
                
                # Create a strict ROI mask over the target structure to ignore neighbors
                obj_mask = (skel_labeled_fn[minr:maxr, minc:maxc] == sp.label)
                crop_roi = morphology.dilation(obj_mask, morphology.disk(6))
                
                try:
                    # RECURE: Run the entire engine on the tiny isolated crop!
                    sub_seg = segment_slice(crop_img, sub_cfg, roi_mask=crop_roi)
                    sub_skel = sub_seg["skel_pruned"]
                    sub_lab = measure.label(sub_skel)
                    
                    if sub_lab.max() > 1:
                        # SUCCESS: The adaptive threshold organically shattered the web!
                        skel_pruned[minr:maxr, minc:maxc][obj_mask] = 0
                        new_frags = (sub_skel > 0)
                        skel_pruned[minr:maxr, minc:maxc][new_frags] = 1
                    else:
                        # Failsafe: if the intensity was perfectly uniform, geometric centroid chop
                        cy, cx = sp.centroid
                        dists = (sp.coords[:, 0] - cy)**2 + (sp.coords[:, 1] - cx)**2
                        mid_idx = np.argmin(dists)
                        my, mx = sp.coords[mid_idx]
                        skel_pruned[my-1:my+2, mx-1:mx+2] = 0
                        
                except Exception:
                    # Absolute geometric failsafe
                    cy, cx = sp.centroid
                    dists = (sp.coords[:, 0] - cy)**2 + (sp.coords[:, 1] - cx)**2
                    mid_idx = np.argmin(dists)
                    my, mx = sp.coords[mid_idx]
                    skel_pruned[my-1:my+2, mx-1:mx+2] = 0

        # Run one final relabeling array refresh after all splicing
        skel_labeled_fn = measure.label(skel_pruned)

    out = {
        "mask_hyst":    mask_hyst,
        "mask_clean":   mask_clean,
        "skel_clean":   skel_clean,
        "skel_bridged": skel_bridged,
        "skel_pruned":  skel_pruned,
        "skel_labeled": skel_labeled_fn,
        "dist_clean":   dist_clean,
        "img_eq":       img_eq,
        "ridge":        ridge,
        "roi_mask":     roi_mask,
    }

    if cfg["SAVE_DEBUG_IMAGES"] and debug_dir and z_idx is not None:
        save_gray(os.path.join(debug_dir, f"z{z_idx:02d}_01_norm.png"),          img_norm)
        save_gray(os.path.join(debug_dir, f"z{z_idx:02d}_02_clahe.png"),         img_eq)
        save_gray(os.path.join(debug_dir, f"z{z_idx:02d}_03_fg.png"),            fgn)
        save_gray(os.path.join(debug_dir, f"z{z_idx:02d}_04_ridge.png"),         ridge)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_05_hysteresis.png"),    mask_hyst)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_06_clean.png"),         mask_clean)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_07_skel_clean.png"),    skel_clean)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_08_skel_bridged.png"),  skel_bridged)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_09_skel_pruned.png"),   skel_pruned)

    return out


# =============================================================================
# MEASUREMENT  (single geodesic pass, all topology in one function)
# =============================================================================

def measure_spermatids(seg, cfg):
    skel     = seg["skel_pruned"]
    dist     = seg["dist_clean"]
    skel_lab = seg["skel_labeled"]
    H, W     = skel.shape

    # ── Filter pass ───────────────────────────────────────────────────────────
    accepted_labels = []
    cache           = {}
    reasons = {"short": 0, "loop": 0, "long": 0, "wide": 0, "ratio": 0, "branches": 0, "tortuous": 0, "endpoints": 0}

    for sp in measure.regionprops(skel_lab):
        coords = sp.coords
        if coords.shape[0] < cfg["MIN_SKEL_LEN_PX"]:
            reasons["short"] += 1
            continue

        topo = measure_topology(coords, W, allow_loops=cfg.get("ALLOW_LOOPS", False))
        if topo is None:
            reasons["loop"] += 1
            continue  # loop, not allowed

        gl   = topo["geo_len"]
        tort = topo["tortuosity"]
        n_ep = topo["n_endpoints"]
        n_br = topo["n_branch_nodes"]

        if not (cfg["MIN_SKEL_LEN_PX"] <= gl <= cfg["MAX_GEODESIC_LEN_PX"]):
            if gl < cfg["MIN_SKEL_LEN_PX"]: reasons["short"] += 1
            else: reasons["long"] += 1
            continue

        width = float(np.median(2.0 * dist[coords[:, 0], coords[:, 1]]))
        if width > cfg["MAX_WIDTH_PX"]:
            reasons["wide"] += 1
            continue

        if gl / (width + 1e-9) < cfg["MIN_LENGTH_WIDTH_RATIO"]:
            reasons["ratio"] += 1
            continue

        # N1: branch-node count filter
        if n_br > cfg["MAX_BRANCH_NODES"]:
            reasons["branches"] += 1
            continue

        # N2: tortuosity filter (only for open filaments with 2+ endpoints)
        if n_ep >= 2 and tort > cfg["MAX_TORTUOSITY"]:
            reasons["tortuous"] += 1
            continue

        # N3: endpoint count filter
        if n_ep > cfg["MAX_ENDPOINT_COUNT"]:
            reasons["endpoints"] += 1
            continue

        cy, cx = sp.centroid
        accepted_labels.append(sp.label)
        cache[sp.label] = {
            "geo_len":            gl,
            "tortuosity":         tort,
            "n_endpoints":        n_ep,
            "n_branch_nodes":     n_br,
            "width":              width,
            "length_width_ratio": gl / (width + 1e-9),
            "length_px_count":    float(coords.shape[0]),
            "cx": cx, "cy": cy,
            "area_px": float(sp.area),
        }

    total_rejected = sum(reasons.values())
    if total_rejected > 0:
        print(f"    measure_spermatids rejected {total_rejected} blobs:")
        for k, v in reasons.items():
            if v > 0: print(f"      {k}: {v}")

    if not accepted_labels:
        return {"skel_label": np.zeros_like(skel_lab, dtype=np.int32),
                "results": []}

    clean_skel  = np.isin(skel_lab, accepted_labels)
    final_label = measure.label(clean_skel).astype(np.int32)

    # ── Re-index using cached values (no second Dijkstra pass) ───────────────
    final_results = []
    for new_i, sp in enumerate(measure.regionprops(final_label), start=1):
        old_label = skel_lab[sp.coords[0, 0], sp.coords[0, 1]]
        if old_label not in cache:
            continue
        c = cache[old_label]
        final_results.append({
            "label":               new_i,
            "length_px_geodesic":  c["geo_len"],
            "length_px_count":     c["length_px_count"],
            "width_px":            c["width"],
            "length_width_ratio":  c["length_width_ratio"],
            "tortuosity":          c["tortuosity"],
            "n_endpoints":         c["n_endpoints"],
            "n_branch_nodes":      c["n_branch_nodes"],
            "centroid_x":          c["cx"],
            "centroid_y":          c["cy"],
            "area_px":             c["area_px"],
        })

    return {"skel_label": final_label, "results": final_results}


# =============================================================================
# OVERLAY  (vectorized LUT)
# =============================================================================

def make_overlay(img_raw, skel_label):
    base = normalize_display(img_raw)
    n    = int(skel_label.max())
    if n <= 0:
        return (np.stack([base]*3, -1) * 255).astype(np.uint8)
    cols    = plt.cm.gist_rainbow(np.linspace(0, 1, n))[:, :3]
    dilated = grey_dilation(skel_label.astype(np.int32), size=3)
    lut     = np.vstack([[0., 0., 0.], cols[:n]])
    rgb     = lut[dilated]
    m0      = dilated == 0
    rgb[m0, 0] = base[m0]
    rgb[m0, 1] = base[m0]
    rgb[m0, 2] = base[m0]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# =============================================================================
# DISPLAY / SAVE
# =============================================================================

def _safe_show():
    try:
        plt.show(block=True)
    except Exception as e:
        print(f"[WARNING] plt.show() failed: {e}")


def show_single_preview(img_raw, seg, overlay_rgb, results, z_idx, cfg):
    um         = cfg["UM_PER_PX_XY"]
    lengths_um = [r["length_px_geodesic"] * um for r in results]

    nrows, ncols = (2, 4) if cfg["SHOW_DEBUG_PREVIEW"] else (1, 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows))

    def _ax(r, c):
        return axes[r, c] if nrows > 1 else axes[c]

    _ax(0,0).imshow(normalize_display(img_raw), cmap="gray")
    _ax(0,0).set_title(f"Original Z={z_idx:02d}"); _ax(0,0).axis("off")

    if cfg["SHOW_DEBUG_PREVIEW"]:
        _ax(0,1).imshow(seg["img_eq"], cmap="gray")
        _ax(0,1).set_title("CLAHE"); _ax(0,1).axis("off")
        _ax(0,2).imshow(seg["ridge"], cmap="gray")
        _ax(0,2).set_title("Ridge"); _ax(0,2).axis("off")
        _ax(0,3).imshow(seg["mask_hyst"], cmap="gray")
        _ax(0,3).set_title("Hysteresis"); _ax(0,3).axis("off")
        _ax(1,0).imshow(seg["mask_clean"], cmap="gray")
        _ax(1,0).set_title("Clean mask"); _ax(1,0).axis("off")
        _ax(1,1).imshow(seg["skel_pruned"], cmap="gray")
        _ax(1,1).set_title("Skeleton (pruned)"); _ax(1,1).axis("off")
        ov_ax   = _ax(1,2)
        hist_ax = _ax(1,3)
    else:
        ov_ax   = _ax(0,1)
        hist_ax = _ax(0,2)

    ov_ax.imshow(overlay_rgb)
    ov_ax.set_title(f"Overlay N={len(results)}"); ov_ax.axis("off")
    for r in results:
        ov_ax.text(r["centroid_x"], r["centroid_y"],
                   f"{r['length_px_geodesic']*um:.1f}",
                   color="white", fontsize=5, ha="center", va="center")

    if lengths_um:
        hist_ax.hist(lengths_um, bins=20, edgecolor="white")
        hist_ax.axvline(np.median(lengths_um), lw=2,
                        label=f"Median={np.median(lengths_um):.1f} µm")
        hist_ax.set_xlabel("Geodesic length (µm)"); hist_ax.legend(fontsize=8)
        hist_ax.set_title("Length distribution")
    else:
        hist_ax.text(0.5, 0.5, "No detections", ha="center", va="center")
        hist_ax.axis("off")

    plt.tight_layout()

    # Save preview to file and open with default viewer
    preview_path = os.path.join(cfg["OUTPUT_DIR"], "preview.png")
    ensure_dir(cfg["OUTPUT_DIR"])
    plt.savefig(preview_path, dpi=120, bbox_inches="tight")
    print(f"  Preview saved to: {preview_path}")

    # Try interactive display
    _safe_show()
    plt.close()

    # Open with default viewer as fallback
    try:
        os.startfile(preview_path)
    except Exception:
        pass


def save_detail_figure(img_raw, overlay_rgb, results, out_path, z_idx, um):
    lengths_um = [r["length_px_geodesic"] * um for r in results]
    fig, axes  = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(normalize_display(img_raw), cmap="gray")
    axes[0].set_title(f"Z={z_idx:02d} — Original"); axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Z={z_idx:02d} — Spermatids (N={len(results)})")
    axes[1].axis("off")
    for r in results:
        axes[1].text(r["centroid_x"], r["centroid_y"],
                     f"{r['length_px_geodesic']*um:.1f}",
                     color="white", fontsize=4, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))

    if lengths_um:
        axes[2].hist(lengths_um, bins=20, edgecolor="white")
        axes[2].axvline(np.median(lengths_um), lw=2,
                        label=f"Median={np.median(lengths_um):.1f} µm")
        axes[2].set_xlabel("Geodesic length (µm)"); axes[2].set_ylabel("Count")
        axes[2].set_title(f"Z={z_idx:02d} — Length distribution")
        axes[2].legend(fontsize=9)
    else:
        axes[2].text(0.5, 0.5, "No spermatids detected",
                     transform=axes[2].transAxes, ha="center", va="center")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# =============================================================================
# CSV
# =============================================================================

_VERSION = "v10"


def rows_from_results(results, z_idx, um):
    return [{
        "pipeline_version":    _VERSION,
        "z_slice":             z_idx,
        "sperm_id":            i,
        "length_px_geodesic":  round(r["length_px_geodesic"], 3),
        "length_um_geodesic":  round(r["length_px_geodesic"] * um, 3),
        "length_px_count":     round(r["length_px_count"], 1),
        "length_um_count":     round(r["length_px_count"]  * um, 3),
        "width_px":            round(r["width_px"], 2),
        "width_um":            round(r["width_px"]          * um, 3),
        "length_width_ratio":  round(r["length_width_ratio"], 3),
        "tortuosity":          round(r["tortuosity"], 3),
        "n_endpoints":         r["n_endpoints"],
        "n_branch_nodes":      r["n_branch_nodes"],
        "centroid_x":          round(r["centroid_x"], 1),
        "centroid_y":          round(r["centroid_y"], 1),
        "area_px":             round(r["area_px"], 1),
    } for i, r in enumerate(results, start=1)]


# =============================================================================
# TRACKING
# =============================================================================

def track_across_slices(detections_df, cfg):
    if detections_df.empty:
        detections_df = detections_df.copy()
        detections_df["track_id"] = pd.Series(dtype=int)
        return detections_df, pd.DataFrame()

    max_dist_px = cfg["TRACK_MAX_DIST_UM"] / (cfg["UM_PER_PX_XY"] + 1e-9)
    df = (detections_df.copy()
                       .sort_values(["z_slice","sperm_id"])
                       .reset_index(drop=True))

    next_tid  = 1
    active    = {}
    track_ids = [-1] * len(df)
    rows_by_z = {z: df.index[df["z_slice"] == z].to_numpy()
                 for z in sorted(df["z_slice"].unique())}

    for z, idxs in rows_by_z.items():
        xs = df.loc[idxs, "centroid_x"].to_numpy(float)
        ys = df.loc[idxs, "centroid_y"].to_numpy(float)

        cand_tracks = [t for t, st in active.items()
                       if 1 <= z - st["last_z"] <= cfg["TRACK_MAX_GAP_SLICES"] + 1]
        cand_pos    = [(active[t]["last_x"], active[t]["last_y"]) for t in cand_tracks]

        used_det, used_trk = set(), set()
        if cand_tracks:
            tree = cKDTree(np.array(cand_pos, float))
            candidates = []
            for k, (x, y) in enumerate(zip(xs, ys)):
                d_val, j = tree.query([x, y])
                if np.isfinite(d_val) and d_val <= max_dist_px:
                    candidates.append((float(d_val), k, int(j)))
            for d_val, det_k, trk_j in sorted(candidates):
                if det_k in used_det or trk_j in used_trk:
                    continue
                used_det.add(det_k); used_trk.add(trk_j)
                tid = cand_tracks[trk_j]
                track_ids[int(idxs[det_k])] = tid
                active[tid] = {"last_z": int(z), "last_x": float(xs[det_k]),
                               "last_y": float(ys[det_k])}

        for det_k in range(len(idxs)):
            if track_ids[int(idxs[det_k])] == -1:
                track_ids[int(idxs[det_k])] = next_tid
                active[next_tid] = {"last_z": int(z), "last_x": float(xs[det_k]),
                                    "last_y": float(ys[det_k])}
                next_tid += 1

        for tid in [t for t, st in active.items()
                    if z - st["last_z"] > cfg["TRACK_MAX_GAP_SLICES"] + 1]:
            del active[tid]

    df["track_id"] = track_ids
    g = df.groupby("track_id", as_index=False)
    ts = g.agg(
        n_detections    = ("sperm_id",           "count"),
        z_min           = ("z_slice",            "min"),
        z_max           = ("z_slice",            "max"),
        mean_centroid_x = ("centroid_x",         "mean"),
        mean_centroid_y = ("centroid_y",         "mean"),
        max_length_um   = ("length_um_geodesic", "max"),
        median_width_um = ("width_um",           "median"),
    )
    dz = (ts["z_max"] - ts["z_min"]) * cfg["UM_PER_SLICE_Z"]
    ts["z_extent_um"]      = dz
    ts["length_3d_um_est"] = np.sqrt(ts["max_length_um"]**2 + dz**2)
    return df, ts


# =============================================================================
# PROCESS ONE IMAGE
# =============================================================================

def process_one_image(image_path, cfg, output_dir):
    ensure_dir(output_dir)
    overlay_dir = os.path.join(output_dir, "overlays")
    debug_dir   = os.path.join(output_dir, "debug")
    ensure_dir(overlay_dir)
    if cfg["SAVE_DEBUG_IMAGES"]:
        ensure_dir(debug_dir)

    z_idx   = extract_z_index(image_path)
    img_raw = tifffile.imread(image_path)
    print(f"\nProcessing: {os.path.basename(image_path)}")

    t0      = time.time()
    seg     = segment_slice(img_raw, cfg, z_idx=z_idx,
                            debug_dir=debug_dir if cfg["SAVE_DEBUG_IMAGES"] else None)
    meas    = measure_spermatids(seg, cfg)
    results = meas["results"]
    elapsed = time.time() - t0

    um = cfg["UM_PER_PX_XY"]
    print(f"  Detected: {len(results)} spermatids  ({elapsed:.1f}s)")
    if results:
        ls = [r["length_px_geodesic"]*um for r in results]
        print(f"  Geodesic length µm: median={np.median(ls):.2f}  max={max(ls):.2f}")

    overlay_rgb = make_overlay(img_raw, meas["skel_label"])

    if cfg["SAVE_OVERLAYS"]:
        _imwrite(os.path.join(overlay_dir, f"z{z_idx:02d}_overlay.png"), overlay_rgb)
    if cfg["SAVE_DETAIL_FIGURE"]:
        save_detail_figure(img_raw, overlay_rgb, results,
                           os.path.join(overlay_dir, f"z{z_idx:02d}_detail.png"),
                           z_idx, um)
    if cfg["SAVE_MASK_TIFS"]:
        tifffile.imwrite(os.path.join(output_dir, f"z{z_idx:02d}_mask.tif"),
                         seg["mask_clean"].astype(np.uint8) * 255)
    if cfg["SAVE_LABEL_TIFS"]:
        tifffile.imwrite(os.path.join(output_dir, f"z{z_idx:02d}_skel_labels.tif"),
                         meas["skel_label"].astype(np.uint16))

    pd.DataFrame(rows_from_results(results, z_idx, um)).to_csv(
        os.path.join(output_dir, f"single_measurements_{_VERSION}.csv"), index=False)

    if cfg["SHOW_PREVIEW_WINDOW"]:
        show_single_preview(img_raw, seg, overlay_rgb, results, z_idx, cfg)

    print(f"Saved to: {output_dir}")


# =============================================================================
# PROCESS BATCH
# =============================================================================

def process_batch(cfg):
    ensure_dir(cfg["OUTPUT_DIR"])
    overlay_dir = os.path.join(cfg["OUTPUT_DIR"], "overlays")
    debug_dir   = os.path.join(cfg["OUTPUT_DIR"], "debug")
    ensure_dir(overlay_dir)
    if cfg["SAVE_DEBUG_IMAGES"]:
        ensure_dir(debug_dir)

    files, z_indices = load_batch_files(cfg["INPUT_DIR"], cfg["FILE_PATTERN"])
    um         = cfg["UM_PER_PX_XY"]
    all_rows   = []
    summaries  = []
    t_batch    = time.time()
    t_slices   = []

    for idx_i, (fpath, z_idx) in enumerate(zip(files, z_indices)):
        t0 = time.time()
        print(f"\n[{idx_i+1}/{len(files)}]  Z={z_idx:02d}  {os.path.basename(fpath)}")
        img_raw = tifffile.imread(fpath)
        seg     = segment_slice(img_raw, cfg, z_idx=z_idx,
                                debug_dir=debug_dir if cfg["SAVE_DEBUG_IMAGES"] else None)
        meas    = measure_spermatids(seg, cfg)
        results = meas["results"]
        ls_um   = [r["length_px_geodesic"]*um for r in results]
        ws_um   = [r["width_px"]*um for r in results]

        t_s = time.time() - t0
        t_slices.append(t_s)
        eta = (len(files) - idx_i - 1) * float(np.mean(t_slices))
        print(f"  N={len(results)}", end="")
        if ls_um:
            print(f"  med_len={np.median(ls_um):.2f}µm", end="")
        print(f"  {t_s:.1f}s  ETA {eta:.0f}s")

        all_rows.extend(rows_from_results(results, z_idx, um))
        summaries.append({
            "z_slice":          z_idx,
            "n_spermatids":     len(results),
            "mean_length_um":   round(float(np.mean(ls_um)),   3) if ls_um else 0,
            "median_length_um": round(float(np.median(ls_um)), 3) if ls_um else 0,
            "mean_width_um":    round(float(np.mean(ws_um)),   3) if ws_um  else 0,
        })

        overlay_rgb = make_overlay(img_raw, meas["skel_label"])
        if cfg["SAVE_OVERLAYS"]:
            # Create side-by-side panel: [Original | Overlay]
            orig_rgb = (normalize_display(img_raw) * 255).astype(np.uint8)
            # if grayscale, convert to RGB for hstack
            if orig_rgb.ndim == 2:
                orig_rgb = np.stack([orig_rgb]*3, axis=-1)
            
            panel = np.hstack([orig_rgb, overlay_rgb])
            _imwrite(os.path.join(overlay_dir, f"z{z_idx:02d}_panel.png"), panel)

        if cfg["SAVE_DETAIL_FIGURE"]:
            save_detail_figure(img_raw, overlay_rgb, results,
                               os.path.join(overlay_dir, f"z{z_idx:02d}_detail.png"),
                               z_idx, um)
        if cfg["SAVE_MASK_TIFS"]:
            tifffile.imwrite(os.path.join(cfg["OUTPUT_DIR"], f"z{z_idx:02d}_mask.tif"),
                             seg["mask_clean"].astype(np.uint8) * 255)
        if cfg["SAVE_LABEL_TIFS"]:
            tifffile.imwrite(os.path.join(cfg["OUTPUT_DIR"], f"z{z_idx:02d}_skel_labels.tif"),
                             meas["skel_label"].astype(np.uint16))

    df     = pd.DataFrame(all_rows)
    df_sum = pd.DataFrame(summaries)
    df.to_csv(    os.path.join(cfg["OUTPUT_DIR"], f"spermatid_measurements_{_VERSION}.csv"), index=False)
    df_sum.to_csv(os.path.join(cfg["OUTPUT_DIR"], f"slice_summary_{_VERSION}.csv"), index=False)

    if cfg["DO_TRACKING"] and not df.empty:
        df_trk, ts = track_across_slices(df, cfg)
        df_trk.to_csv(
            os.path.join(cfg["OUTPUT_DIR"], f"measurements_with_tracks_{_VERSION}.csv"),
            index=False)
        ts.to_csv(
            os.path.join(cfg["OUTPUT_DIR"], f"track_summary_{_VERSION}.csv"),
            index=False)

    total = time.time() - t_batch
    print(f"\n{'='*55}")
    print(f"DONE  {len(files)} slices  {total:.1f}s  ({total/len(files):.1f}s/slice)")
    print(f"Saved to: {cfg['OUTPUT_DIR']}")
    print(df_sum.to_string(index=False))

    # Generate High-Res Graphical Report
    generate_batch_report(cfg['OUTPUT_DIR'], df, df_sum, um, None, None)


def generate_batch_report(out_dir, df, df_summary, um, df_tracks=None, gui_callback=None):
    """
    Generates a high-resolution (300 DPI) multi-page PDF summary report.
    Page 1: Global batch statistics and trends.
    Following Pages: Per-slice panels (Original vs Overlay vs Plot).
    """
    pdf_path = os.path.join(out_dir, f"batch_report_{_VERSION}.pdf")
    print(f"Generating high-res PDF report: {pdf_path} ...")
    
    # Create directory for high-res standalone plots (for easy copy-pasting into papers/presentations)
    plot_dir = os.path.join(out_dir, "summary_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    try:
        with PdfPages(pdf_path) as pdf:
            # --- PAGE 1: GLOBAL SUMMARY ---
            fig_sum = plt.figure(figsize=(11, 8.5))
            fig_sum.suptitle(f"Spermatid Analysis Batch Summary - {_VERSION}\nLocation: {out_dir}", fontsize=14, fontweight='bold')
            
            # Global Z-Projection Image (Top Center)
            z_proj_path = os.path.join(out_dir, "global_z_projection.png")
            if os.path.exists(z_proj_path):
                ax_z = fig_sum.add_axes([0.15, 0.62, 0.7, 0.28]) # [left, bottom, width, height]
                ax_z.imshow(plt.imread(z_proj_path))
                ax_z.set_title("Global Z-Projection (Composite [Original | Overlay])", fontsize=10)
                ax_z.axis('off')
            
            # Plot 1: Counts per slice
            ax1 = fig_sum.add_subplot(2, 2, 3)
            ax1.plot(df_summary['z_slice'], df_summary['n_spermatids'], 'bo-', markersize=4)
            ax1.set_title("Detections per Z-Slice")
            ax1.set_xlabel("Z-Index")
            ax1.set_ylabel("Count")
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Length Distribution (Global)
            ax2 = fig_sum.add_subplot(2, 2, 4)
            if not df.empty:
                vals = df['length_um_geodesic']
                ax2.hist(vals, bins=25, color='forestgreen', edgecolor='black', alpha=0.7)
                m_med = vals.median()
                m_avg = vals.mean()
                ax2.axvline(m_med, color='red', linestyle='-', label=f"Med: {m_med:.1f}")
                ax2.axvline(m_avg, color='orange', linestyle='--', label=f"Avg: {m_avg:.1f}")
                ax2.set_title("Global 2D Length Distribution")
                ax2.set_xlabel("Geodesic Length (um)")
                ax2.set_ylabel("Frequency")
                ax2.legend(fontsize=8)
            
            fig_sum.savefig(os.path.join(plot_dir, "global_summary.png"), dpi=300, bbox_inches='tight')
            pdf.savefig(fig_sum, dpi=300, bbox_inches='tight')
            plt.close(fig_sum)

            # --- PAGE 2: 3D MORPHOMETRICS SUMMARY ---
            if df_tracks is not None and not df_tracks.empty:
                fig_3d = plt.figure(figsize=(11, 8.5))
                fig_3d.suptitle("3D Population Statistics (Tracked Spermatids)", fontsize=14, fontweight='bold')
                
                # 3D Length
                ax3d_1 = fig_3d.add_subplot(2, 2, 1)
                vals_3d = df_tracks['total_3d_length_um']
                ax3d_1.hist(vals_3d, bins=20, color='darkorange', edgecolor='black', alpha=0.7)
                m3d_med = vals_3d.median()
                m3d_avg = vals_3d.mean()
                ax3d_1.axvline(m3d_med, color='red', linestyle='-', label=f"Med: {m3d_med:.1f}")
                ax3d_1.axvline(m3d_avg, color='black', linestyle='--', label=f"Avg: {m3d_avg:.1f}")
                ax3d_1.set_title("Total 3D Geodesic Length")
                ax3d_1.set_xlabel("Length (um)")
                ax3d_1.set_ylabel("Frequency")
                ax3d_1.legend(fontsize=8)

                # 3D Tortuosity
                ax3d_2 = fig_3d.add_subplot(2, 2, 2)
                vt = df_tracks['tortuosity_3d']
                ax3d_2.hist(vt, bins=20, color='purple', edgecolor='black', alpha=0.6)
                ax3d_2.axvline(vt.median(), color='red', linestyle='-', label=f"Med: {vt.median():.2f}")
                ax3d_2.axvline(vt.mean(), color='black', linestyle='--', label=f"Avg: {vt.mean():.2f}")
                ax3d_2.set_title("3D Tortuosity (Curvature)")
                ax3d_2.set_xlabel("Ratio (Length / Distance)")
                ax3d_2.set_ylabel("Frequency")
                ax3d_2.legend(fontsize=8)

                # Vertical Extent
                ax3d_3 = fig_3d.add_subplot(2, 2, 3)
                ve = df_tracks['z_extent_um']
                ax3d_3.hist(ve, bins=15, color='teal', edgecolor='black', alpha=0.7)
                ax3d_3.axvline(ve.median(), color='red', linestyle='-', label=f"Med: {ve.median():.1f}")
                ax3d_3.axvline(ve.mean(), color='black', linestyle='--', label=f"Avg: {ve.mean():.1f}")
                ax3d_3.set_title("Z-Extent (Vertical Span)")
                ax3d_3.set_xlabel("Vertical Height (um)")
                ax3d_3.set_ylabel("Frequency")
                ax3d_3.legend(fontsize=8)

                # Volume
                ax3d_4 = fig_3d.add_subplot(2, 2, 4)
                vv = df_tracks['volume_um3']
                ax3d_4.hist(vv, bins=20, color='gray', edgecolor='black', alpha=0.7)
                ax3d_4.axvline(vv.median(), color='red', linestyle='-', label=f"Med: {vv.median():.0f}")
                ax3d_4.axvline(vv.mean(), color='black', linestyle='--', label=f"Avg: {vv.mean():.0f}")
                ax3d_4.set_title("Approximated 3D Volume")
                ax3d_4.set_xlabel("Volume (um³)")
                ax3d_4.set_ylabel("Frequency")
                ax3d_4.legend(fontsize=8)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig_3d.savefig(os.path.join(plot_dir, "3d_population_stats.png"), dpi=300, bbox_inches='tight')
                pdf.savefig(fig_3d, dpi=300, bbox_inches='tight')
                plt.close(fig_3d)

            # --- PAGE 3: METHODS & INTERPRETATION GUIDE ---
            fig_guide = plt.figure(figsize=(11, 8.5))
            ax_g = fig_guide.add_axes([0.05, 0.05, 0.9, 0.9])
            ax_g.axis('off')
            guide_full = (
                "METHODS & CALCULATION DETAILS\n"
                f"{'='*60}\n\n"
                "1. Total 3D Geodesic Length (µm)\n"
                "   FORMULA: L_3d = sqrt( L_max_2d^2 + Z_extent^2 )\n"
                "   DETAILS: Uses the Pythagorean theorem to combine the maximum lateral \n"
                "   extension (XY) with the vertical depth (Z). This prevents overestimation \n"
                "   from overlapping slices and accounts for nuclei tilt.\n\n"
                "2. 3D Tortuosity (Curvature Index)\n"
                "   FORMULA: T_3d = L_3d / D_euclidean_3d\n"
                "   DETAILS: Ratio of the 3D geodesic path to the straight-line distance \n"
                "   between the 3D start and end centroids. 1.0 = Straight.\n\n"
                "3. Z-Extent (Vertical Span)\n"
                "   FORMULA: Z_ext = (Slice_max - Slice_min + 1) * Z_step_um\n"
                "   DETAILS: Measures the absolute vertical depth of the tracked string.\n\n"
                "4. 3D Volumetric Approximation (µm³)\n"
                "   FORMULA: V_3d = sum( Area_i * Z_step_um )\n"
                "   DETAILS: Numerical integration of cross-sectional areas.\n\n"
                "5. Statistical Indicators\n"
                "   - MEDIAN: The middle value (robust to segmentation outliers).\n"
                "   - AVERAGE (MEAN): The arithmetic mean (sensitive to population shifts).\n"
                "   - FREQUENCY: The count of detections per category."
            )
            ax_g.text(0, 1, guide_full, transform=ax_g.transAxes, fontsize=11, family='monospace', verticalalignment='top', linespacing=1.3)
            fig_guide.savefig(os.path.join(plot_dir, "methods_guide.png"), dpi=300, bbox_inches='tight')
            pdf.savefig(fig_guide, dpi=300, bbox_inches='tight')
            plt.close(fig_guide)

            # --- PAGE 4: GLOBAL STATISTICS TABLE ---
            fig_tab = plt.figure(figsize=(11, 8.5))
            ax_t = fig_tab.add_subplot(1, 1, 1)
            ax_t.axis('off')
            ax_t.set_title("Global Population Statistics Summary", fontsize=14, fontweight='bold', pad=20)
            
            stats_rows = []
            if not df.empty:
                l2d = df['length_um_geodesic']
                stats_rows.append(["2D Geodesic Length (um)", f"{l2d.mean():.2f}", f"{l2d.median():.2f}", f"{l2d.std():.2f}"])
            
            if df_tracks is not None and not df_tracks.empty:
                l3d = df_tracks['total_3d_length_um']
                ze = df_tracks['z_extent_um']
                vo = df_tracks['volume_um3']
                to = df_tracks['tortuosity_3d']
                stats_rows.append(["3D Geodesic Length (um)", f"{l3d.mean():.2f}", f"{l3d.median():.2f}", f"{l3d.std():.2f}"])
                stats_rows.append(["3D Z-Extent (um)", f"{ze.mean():.2f}", f"{ze.median():.2f}", f"{ze.std():.2f}"])
                stats_rows.append(["3D Volume (um3)", f"{vo.mean():.1f}", f"{vo.median():.1f}", f"{vo.std():.1f}"])
                stats_rows.append(["3D Tortuosity Index", f"{to.mean():.3f}", f"{to.median():.3f}", f"{to.std():.3f}"])
            
            if stats_rows:
                table = ax_t.table(cellText=stats_rows, 
                                 colLabels=["Metric", "Average (Mean)", "Median", "Std Dev"],
                                 loc='center', cellLoc='center', colWidths=[0.35, 0.2, 0.2, 0.2])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 2.5)
            
            fig_tab.savefig(os.path.join(plot_dir, "global_statistics_table.png"), dpi=300, bbox_inches='tight')
            pdf.savefig(fig_tab, dpi=300, bbox_inches='tight')
            plt.close(fig_tab)
            
            # --- SUBSEQUENT PAGES: PER-SLICE DETAILS (2 panels: [Panel] | [Histogram]) ---
            overlay_dir = os.path.join(out_dir, "overlays")
            for idx_p, (row_idx, row) in enumerate(df_summary.iterrows()):
                z = int(row['z_slice'])
                panel_path = os.path.join(overlay_dir, f"z{z:02d}_panel.png")
                
                if not os.path.exists(panel_path):
                    continue
                
                fig_slice = plt.figure(figsize=(18, 7))
                fig_slice.suptitle(f"Z-Slice {z:02d} Analysis [Original | Overlay | Distribution]", fontsize=12, fontweight='bold')
                
                # Panel: Side-by-Side (Original | Overlay)
                ax_panel = fig_slice.add_subplot(1, 2, 1)
                ax_panel.imshow(plt.imread(panel_path))
                ax_panel.set_title(f"Visual Verification (N={int(row['n_spermatids'])})")
                ax_panel.axis('off')
                
                # Plot: Stats
                ax_hist = fig_slice.add_subplot(1, 2, 2)
                slice_data = df[df['z_slice'] == z]
                if not slice_data.empty:
                    ax_hist.hist(slice_data['length_um_geodesic'], bins=15, color='skyblue', edgecolor='black')
                    ax_hist.set_title(f"Z={z} Length Distribution")
                    ax_hist.set_xlabel("Spermatid Length (um)")
                    ax_hist.set_ylabel("Frequency (Count)")
                    
                    m_med = slice_data['length_um_geodesic'].median()
                    m_avg = slice_data['length_um_geodesic'].mean()
                    ax_hist.axvline(m_med, color='red', linestyle='-', alpha=0.7, label=f"Median: {m_med:.1f}")
                    ax_hist.axvline(m_avg, color='orange', linestyle='--', alpha=0.7, label=f"Average: {m_avg:.1f}")
                    ax_hist.legend(fontsize=9)
                else:
                    ax_hist.text(0.5, 0.5, "No Detections", ha='center', va='center')
                
                pdf.savefig(fig_slice, dpi=300)
                plt.close(fig_slice)
                
                if gui_callback:
                    gui_callback(int(80 + (20 * (idx_p+1) / len(df_summary))))
                
        print(f"Report successfully saved to {pdf_path}")
    except Exception as e:
        print(f"ERROR generating report: {e}")
        traceback.print_exc()




# =============================================================================
# ROI GUI (single-file v9)
# =============================================================================
import traceback
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False

class SpermGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Sperm Segmentation ROI Tool')
        self.root.geometry('1450x920')

        self.input_dir = ''
        self.files = []
        self.current_idx = 0
        self.current_img = None

        self.roi_points = []
        self.drawing = False
        self.roi_active = False
        self._loaded_roi_mask = None

        self.sidebar = tk.Frame(root, width=280, bg='#f0f0f0')
        self.sidebar.pack(side='left', fill='y')

        tk.Button(self.sidebar, text='Load Directory', command=self.load_directory, height=2).pack(fill='x', padx=6, pady=6)
        self.lbl_status = tk.Label(self.sidebar, text='No directory loaded', wraplength=260, justify='left')
        self.lbl_status.pack(pady=6)

        tk.Label(self.sidebar, text='Z-Slice Navigation').pack(pady=(20, 0))
        self.scale_z = tk.Scale(self.sidebar, from_=0, to=0, orient='horizontal', command=self.on_slide_change)
        self.scale_z.pack(fill='x', padx=10)
        self.lbl_z = tk.Label(self.sidebar, text='Z: 0 / 0')
        self.lbl_z.pack()

        tk.Label(self.sidebar, text='Tools').pack(pady=(20, 5))
        self.mode_var = tk.StringVar(value='view')
        tk.Radiobutton(self.sidebar, text='View/Nav', variable=self.mode_var, value='view').pack(anchor='w', padx=10)
        tk.Radiobutton(self.sidebar, text='Draw ROI (Polygon)', variable=self.mode_var, value='roi').pack(anchor='w', padx=10)
        tk.Label(self.sidebar, text='(Left-click points, Right-click undo)', font=('Arial', 8, 'italic'), fg='dimgray').pack()
        tk.Button(self.sidebar, text='Finalize Polygon', command=self.finalize_roi, bg='#ffeeba').pack(fill='x', padx=20, pady=2)

        tk.Button(self.sidebar, text='Run Analysis on Slice', command=self.run_analysis_slice).pack(fill='x', padx=6, pady=18)
        tk.Button(self.sidebar, text='Run Batch (All Slices + 3D Track)', command=self.run_batch_analysis, bg='#d4edda', font=('Arial', 10, 'bold')).pack(fill='x', padx=6, pady=6)
        tk.Button(self.sidebar, text='Reset ROI', command=self.reset_roi).pack(fill='x', padx=6, pady=6)
        tk.Button(self.sidebar, text='Save ROI Mask', command=self.save_roi_mask).pack(fill='x', padx=6, pady=6)
        tk.Button(self.sidebar, text='Load ROI Mask', command=self.load_roi_mask).pack(fill='x', padx=6, pady=6)

        self.lbl_roi = tk.Label(self.sidebar, text='ROI: none', wraplength=260, justify='left')
        self.lbl_roi.pack(pady=10)

        # Batch Progress Bar (2D Segmentation)
        tk.Label(self.sidebar, text='Batch Progress (2D)', font=('Arial', 9, 'bold')).pack(pady=(20, 0))
        self.progress = ttk.Progressbar(self.sidebar, orient='horizontal', length=220, mode='determinate')
        self.progress.pack(padx=10, pady=5)
        self.lbl_progress_val = tk.Label(self.sidebar, text='0%', font=('Arial', 10, 'bold'), fg='blue')
        self.lbl_progress_val.pack()

        # Post-Analysis Progress Bar (3D Track + Report)
        tk.Label(self.sidebar, text='Post-Analysis Progress', font=('Arial', 9, 'bold')).pack(pady=(15, 0))
        self.progress_post = ttk.Progressbar(self.sidebar, orient='horizontal', length=220, mode='determinate')
        self.progress_post.pack(padx=10, pady=5)
        self.lbl_post_progress_val = tk.Label(self.sidebar, text='Waiting...', font=('Arial', 10, 'bold'), fg='dimgray')
        self.lbl_post_progress_val.pack()

        self.canvas_frame = tk.Frame(root, bg='black')
        self.canvas_frame.pack(side='right', expand=True, fill='both')

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('key_press_event', self.on_key)

    def load_directory(self):
        initial = CONFIG.get('INPUT_DIR', os.getcwd())
        # Let the user pick *any* .tif file inside the target folder, and we'll extract the directory from it.
        # This completely avoids the confusing Windows 11 "Select Folder" dialog issue.
        filepath = filedialog.askopenfilename(initialdir=initial, title="Select ANY Image in the Target Folder",
                                              filetypes=[("TIFF Files", "*.tif *.tiff")])
        if not filepath:
            return
            
        d = os.path.dirname(filepath)
        self.input_dir = d
        self.files = sorted(glob.glob(os.path.join(d, '*.tif')))
        if not self.files:
            self.files = sorted(glob.glob(os.path.join(d, '*.tiff')))
        if not self.files:
            messagebox.showerror('Error', f'No TIFF files found in:\n{d}')
            return
            
        self.current_idx = 0
        self.scale_z.config(to=len(self.files) - 1)
        self.scale_z.set(0)
        self.reset_roi(redraw=False)
        self.load_image()
        self.lbl_status.config(text=f'Loaded: {os.path.basename(d)}\n{len(self.files)} slices')

    def load_image(self):
        if not self.files:
            return
        self.current_img = tifffile.imread(self.files[self.current_idx])
        self.lbl_z.config(text=f'Z: {self.current_idx} / {len(self.files)-1}')
        self.render()

    def on_slide_change(self, val):
        self.current_idx = int(val)
        self.load_image()

    def render(self):
        self.ax.clear()
        self.ax.axis('off')
        if self.current_img is not None:
            img = self.current_img.astype(float)
            p1, p99 = np.percentile(img, 1), np.percentile(img, 99.5)
            disp = np.clip((img - p1) / (p99 - p1 + 1e-9), 0, 1)
            self.ax.imshow(disp, cmap='gray')
            if len(self.roi_points) > 0:
                pts = np.array(self.roi_points)
                # If active (closed), draw it closed. Otherwise, draw the open line and endpoints.
                if self.roi_active:
                    self.ax.plot(pts[:,0], pts[:,1], 'r-', linewidth=2)
                else:
                    self.ax.plot(pts[:,0], pts[:,1], 'r-', linewidth=1.5)
                    self.ax.plot(pts[:,0], pts[:,1], 'ro', markersize=4)
            # If a loaded mask exists, always redraw the red contour so it persists across slices
            elif self._loaded_roi_mask is not None:
                self.ax.contour(self._loaded_roi_mask.astype(float), levels=[0.5], colors='red', linewidths=1.5)
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self.mode_var.get() == 'roi':
            # Left click = add point
            if event.button == 1:
                # If they start clicking after already having a finished ROI, wipe it to start fresh
                if self.roi_active:
                    self.reset_roi(redraw=False)
                self.roi_points.append([event.xdata, event.ydata])
                self.lbl_roi.config(text=f'ROI: building ({len(self.roi_points)} points)')
                self.render()
            
            # Right click = undo last point
            elif event.button == 3:
                if not self.roi_active and len(self.roi_points) > 0:
                    self.roi_points.pop()
                    self.lbl_roi.config(text=f'ROI: building ({len(self.roi_points)} points)')
                    self.render()

    def finalize_roi(self):
        if not self.roi_active and len(self.roi_points) > 2:
            self.roi_points.append(self.roi_points[0])
            self.roi_active = True
            self.lbl_roi.config(text=f'ROI: active ({len(self.roi_points)-1} points)')
            self.render()
        elif not self.roi_active and len(self.roi_points) > 0:
            messagebox.showwarning('Draw ROI', 'Please place at least 3 points before finalizing.')

    def on_key(self, event):
        if self.mode_var.get() == 'roi' and event.key == 'enter':
            self.finalize_roi()

    def reset_roi(self, redraw=True):
        self.roi_points = []
        self.roi_active = False
        self._loaded_roi_mask = None
        self.lbl_roi.config(text='ROI: none')
        if redraw:
            self.render()

    def build_roi_mask(self):
        # If a mask was loaded from file, use it directly
        if self._loaded_roi_mask is not None and self.current_img is not None:
            return self._loaded_roi_mask
        if not self.roi_active or len(self.roi_points) < 4 or self.current_img is None:
            return None
        h, w = self.current_img.shape
        yy, xx = np.mgrid[:h, :w]
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        path = Path(self.roi_points)
        return path.contains_points(pts).reshape(h, w)

    def save_roi_mask(self):
        mask = self.build_roi_mask()
        if mask is None:
            messagebox.showinfo('Save ROI', 'No active ROI to save.')
            return
        default = f'roi_z{self.current_idx:02d}.npy'
        path = filedialog.asksaveasfilename(defaultextension='.npy', initialfile=default,
                                            filetypes=[('NumPy array', '*.npy')])
        if not path:
            return
        np.save(path, mask.astype(np.uint8))
        messagebox.showinfo('Save ROI', f'Saved ROI mask to:\n{path}')

    def load_roi_mask(self):
        if self.current_img is None:
            messagebox.showinfo('Load ROI', 'Load an image first.')
            return
        path = filedialog.askopenfilename(filetypes=[('NumPy array', '*.npy')])
        if not path:
            return
        mask = np.load(path).astype(bool)
        if mask.shape != self.current_img.shape:
            messagebox.showerror('Load ROI', f'Mask shape {mask.shape} does not match image shape {self.current_img.shape}')
            return
        # Convert mask boundary to points for display
        ys, xs = np.where(mask)
        if len(xs) == 0:
            messagebox.showerror('Load ROI', 'Loaded ROI mask is empty.')
            return
        self.roi_points = []
        self.roi_active = True
        self._loaded_roi_mask = mask  # Store the loaded mask for build_roi_mask!
        self.lbl_roi.config(text=f'ROI: loaded mask\n{os.path.basename(path)}')
        self.render()
        self.ax.contour(mask.astype(float), levels=[0.5], colors='red', linewidths=1.5)
        self.canvas.draw_idle()

    def run_analysis_slice(self):
        if self.current_img is None:
            messagebox.showinfo('Info', 'No image loaded. Load a directory first.')
            return

        self.lbl_roi.config(text='Running analysis...')
        self.root.update_idletasks()

        try:
            import sys, time as _t
            log_lines = []
            def log(msg):
                log_lines.append(msg)
                print(msg)
                sys.stdout.flush()

            log(f"\n--- GUI Analysis: slice {self.current_idx} ---")
            log(f"  File: {os.path.basename(self.files[self.current_idx]) if self.files else 'N/A'}")
            log(f"  Image shape: {self.current_img.shape}, dtype: {self.current_img.dtype}")

            params = CONFIG.copy()
            params['SAVE_DEBUG_IMAGES'] = False
            roi_mask = self.build_roi_mask()

            # Determine image to process
            full_img = self.current_img
            crop_offset_y, crop_offset_x = 0, 0
            H, W = full_img.shape

            # ═══════════════════════════════════════════════════════════════════
            # TWO-PASS HYBRID SEGMENTATION ENGINE
            # Pass 1: Full-frame analysis (correct global percentiles for sparse outer nuclei)
            # Pass 2: Tight ROI crop analysis (local percentiles naturally separate dense clusters)
            # Merge: Spatially deduplicate overlapping centroids, keeping unique detections from both
            # ═══════════════════════════════════════════════════════════════════

            t0 = _t.time()

            # ── PASS 1: Full-frame analysis ──────────────────────────────────
            log(f"  PASS 1: Full-frame analysis...")
            seg1 = segment_slice(full_img, params, roi_mask=roi_mask, z_idx=self.current_idx)
            meas1 = measure_spermatids(seg1, params)
            results1 = meas1['results']
            skel_label1 = meas1['skel_label']

            # Filter Pass 1 results to ROI
            if roi_mask is not None and results1:
                filtered1 = []
                keep1 = []
                for r in results1:
                    cy = min(max(int(round(r['centroid_y'])), 0), roi_mask.shape[0] - 1)
                    cx = min(max(int(round(r['centroid_x'])), 0), roi_mask.shape[1] - 1)
                    if roi_mask[cy, cx]:
                        filtered1.append(r)
                        keep1.append(r['label'])
                results1 = filtered1
                skel_label1 = np.where(np.isin(skel_label1, keep1), skel_label1, 0).astype(np.int32)
            log(f"  PASS 1: {len(results1)} detections (full-frame statistics)")

            # ── PASS 2: Tight ROI crop analysis ──────────────────────────────
            results2_global = []
            skel_label2_full = np.zeros((H, W), dtype=np.int32)

            if roi_mask is not None:
                log(f"  PASS 2: Tight-crop local analysis...")
                ys, xs = np.where(roi_mask)
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                pad = 10
                y0c = max(0, y0 - pad)
                x0c = max(0, x0 - pad)
                y1c = min(H, y1 + pad)
                x1c = min(W, x1 + pad)

                crop_img = full_img[y0c:y1c, x0c:x1c]
                crop_roi = roi_mask[y0c:y1c, x0c:x1c]

                seg2 = segment_slice(crop_img, params, roi_mask=crop_roi, z_idx=self.current_idx)
                meas2 = measure_spermatids(seg2, params)
                results2 = meas2['results']
                skel_label2_crop = meas2['skel_label']

                # Filter Pass 2 to ROI
                if results2:
                    filtered2 = []
                    keep2 = []
                    for r in results2:
                        cy = min(max(int(round(r['centroid_y'])), 0), crop_roi.shape[0] - 1)
                        cx = min(max(int(round(r['centroid_x'])), 0), crop_roi.shape[1] - 1)
                        if crop_roi[cy, cx]:
                            filtered2.append(r)
                            keep2.append(r['label'])
                    results2 = filtered2
                    skel_label2_crop = np.where(np.isin(skel_label2_crop, keep2), skel_label2_crop, 0).astype(np.int32)

                # Map Pass 2 back to full-image coordinates
                ch2, cw2 = skel_label2_crop.shape
                max_label1 = int(skel_label1.max()) if skel_label1.max() > 0 else 0
                skel_label2_crop_shifted = np.where(skel_label2_crop > 0, skel_label2_crop + max_label1, 0).astype(np.int32)
                skel_label2_full[y0c:y0c+ch2, x0c:x0c+cw2] = skel_label2_crop_shifted

                for r in results2:
                    r['centroid_x'] += x0c
                    r['centroid_y'] += y0c
                    r['label'] += max_label1
                results2_global = results2
                log(f"  PASS 2: {len(results2_global)} detections (local-crop statistics)")
            else:
                log(f"  PASS 2: Skipped (no ROI)")

            # ── MERGE: Spatial deduplication ──────────────────────────────────
            # For each Pass 2 detection, check if Pass 1 already found a spermatid
            # within 5 pixels of the same centroid. If not, it's a new unique detection.
            dedup_radius = 5.0  # pixels
            merged_results = list(results1)
            skel_label_full = skel_label1.copy()
            new_from_pass2 = 0

            for r2 in results2_global:
                cx2, cy2 = r2['centroid_x'], r2['centroid_y']
                is_duplicate = False
                for r1 in results1:
                    dx = r1['centroid_x'] - cx2
                    dy = r1['centroid_y'] - cy2
                    if (dx*dx + dy*dy) < dedup_radius * dedup_radius:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    merged_results.append(r2)
                    # Splice the Pass 2 skeleton pixels into the merged map
                    mask2 = (skel_label2_full == r2['label'])
                    skel_label_full[mask2] = r2['label']
                    new_from_pass2 += 1

            results = merged_results
            log(f"  MERGE: {len(results1)} (Pass1) + {new_from_pass2} unique from Pass2 = {len(results)} total")

            elapsed = _t.time() - t0
            log(f"  RESULT: {len(results)} spermatids detected ({elapsed:.1f}s)")

            overlay = make_overlay(full_img, skel_label_full)

            # Write log to file
            try:
                log_path = os.path.join(CONFIG['OUTPUT_DIR'], 'gui_analysis_log.txt')
                ensure_dir(CONFIG['OUTPUT_DIR'])
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(log_lines) + '\n\n')
            except Exception:
                pass

            # Show results popup
            top = tk.Toplevel(self.root)
            top.title(f'Results Z={self.current_idx} - {len(results)} spermatids')
            top.geometry('1200x650')

            fig = Figure(figsize=(14, 6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            img = full_img.astype(float)
            p1, p99 = np.percentile(img, 1), np.percentile(img, 99.5)
            disp = np.clip((img - p1) / (p99 - p1 + 1e-9), 0, 1)
            ax1.imshow(disp, cmap='gray')
            if roi_mask is not None:
                ax1.contour(roi_mask.astype(float), levels=[0.5], colors='red', linewidths=1.2)
            ax1.set_title('Original + ROI')
            ax1.axis('off')

            ax2.imshow(overlay)
            ax2.set_title(f'Overlay (N={len(results)})')
            ax2.axis('off')

            um = params['UM_PER_PX_XY']
            for r in results:
                ax2.text(r['centroid_x'], r['centroid_y'],
                         f"{r['length_px_geodesic'] * um:.1f}",
                         color='white', fontsize=5, ha='center', va='center')

            fig.tight_layout()
            can = FigureCanvasTkAgg(fig, master=top)
            can.get_tk_widget().pack(fill='both', expand=True)
            can.draw()

            if results:
                lengths = [r['length_px_geodesic'] * um for r in results]
                text = f'Found {len(results)} spermatids | median length {np.median(lengths):.2f} um ({elapsed:.1f}s)'
            else:
                text = f'Found 0 spermatids ({elapsed:.1f}s) - see gui_analysis_log.txt for diagnostics'
            lbl_stats = tk.Label(top, text=text, font=('Arial', 11))
            lbl_stats.pack(pady=4)

            lbl_tool = tk.Label(top, text="Active Tool: None (Press 'E' to Erase, 'S' to Split, 'Esc' to Cancel)", fg='blue', font=('Arial', 10, 'bold'))
            lbl_tool.pack(pady=2)

            self.lbl_roi.config(text=f'Analysis done: {len(results)} spermatids')

            # ── INTERACTIVE MANUAL CORRECTION LOGIC ──
            class ManualCorrector:
                def __init__(self, canvas, ax_overlay, seg_data, prms, crop_oy, crop_ox, fimg):
                    self.canvas = canvas
                    self.ax = ax_overlay
                    self.seg = seg_data
                    self.params = prms
                    self.crop_oy = crop_oy
                    self.crop_ox = crop_ox
                    self.fimg = fimg
                    
                    self.active_tool = None
                    self.cid_press = self.canvas.mpl_connect('button_press_event', self.on_click)
                    self.cid_key = self.canvas.mpl_connect('key_press_event', self.on_key)
                    
                    self.overlay_imshow = None
                    self.text_artists = []
                    
                def on_key(self, event):
                    if event.key == 'e':
                        self.active_tool = 'erase'
                        lbl_tool.config(text="Active Tool: ERASE (Click a colored spermatid to delete it)", fg='red')
                    elif event.key == 's':
                        self.active_tool = 'split'
                        lbl_tool.config(text="Active Tool: SPLIT (Click on a skeleton branch to sever it)", fg='orange')
                    elif event.key == 'escape':
                        self.active_tool = None
                        lbl_tool.config(text="Active Tool: None (Press 'E' to Erase, 'S' to Split, 'Esc' to Cancel)", fg='blue')

                def on_click(self, event):
                    if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
                        return
                    if not self.active_tool:
                        return

                    # Map global click to cropped coordinate space
                    x_crop = int(round(event.xdata)) - self.crop_ox
                    y_crop = int(round(event.ydata)) - self.crop_oy
                    
                    ch, cw = self.seg['skel_pruned'].shape
                    if not (0 <= x_crop < cw and 0 <= y_crop < ch):
                        return # Clicked outside the ROI working area

                    modified = False
                    if self.active_tool == 'erase':
                        # Find the label at this pixel in the current skeleton
                        lab = self.seg['skel_labeled'][y_crop, x_crop]
                        # If the user clicked slightly off, search a 3x3 neighborhood
                        if lab == 0:
                            for dy in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    yy, xx = y_crop+dy, x_crop+dx
                                    if 0 <= yy < ch and 0 <= xx < cw and self.seg['skel_labeled'][yy, xx] > 0:
                                        lab = self.seg['skel_labeled'][yy, xx]
                                        break
                        if lab > 0:
                            self.seg['skel_pruned'][self.seg['skel_labeled'] == lab] = False
                            modified = True
                            
                    elif self.active_tool == 'split':
                        # Draw a small black circle to sever the topological skeleton connection
                        from skimage.draw import disk
                        rr, cc = disk((y_crop, x_crop), radius=2.5, shape=self.seg['skel_pruned'].shape)
                        self.seg['skel_pruned'][rr, cc] = False
                        modified = True
                        
                    if modified:
                        self.recalculate_and_redraw()
                        
                def recalculate_and_redraw(self):
                    # Re-label the modified skeleton
                    self.seg['skel_labeled'] = measure.label(self.seg['skel_pruned'])
                    
                    # Re-run measurement filters
                    new_meas = measure_spermatids(self.seg, self.params)
                    new_results = new_meas['results']
                    
                    # Filter by ROI inside centroid logic
                    if crop_roi is not None:
                        filtered = []
                        keep_labels = []
                        for r in new_results:
                            c_y = min(max(int(round(r['centroid_y'])), 0), crop_roi.shape[0] - 1)
                            c_x = min(max(int(round(r['centroid_x'])), 0), crop_roi.shape[1] - 1)
                            if crop_roi[c_y, c_x]:
                                filtered.append(r)
                                keep_labels.append(r['label'])
                        new_results = filtered
                        skel_l = np.where(np.isin(new_meas['skel_label'], keep_labels), new_meas['skel_label'], 0).astype(np.int32)
                    else:
                        skel_l = new_meas['skel_label']

                    # Map back to full image
                    H, W = self.fimg.shape
                    skel_label_full = np.zeros((H, W), dtype=np.int32)
                    ch, cw = skel_l.shape
                    skel_label_full[self.crop_oy:self.crop_oy+ch, self.crop_ox:self.crop_ox+cw] = skel_l

                    # Redraw Overlay
                    new_overlay = make_overlay(self.fimg, skel_label_full)
                    
                    self.ax.clear()
                    self.ax.imshow(new_overlay)
                    self.ax.set_title(f'Overlay (N={len(new_results)}) - Manual Corrections Applied')
                    self.ax.axis('off')

                    # Redraw Text
                    _um = self.params['UM_PER_PX_XY']
                    for r in new_results:
                        self.ax.text(r['centroid_x'] + self.crop_ox, r['centroid_y'] + self.crop_oy,
                                 f"{r['length_px_geodesic'] * _um:.1f}",
                                 color='white', fontsize=5, ha='center', va='center')
                                 
                    self.canvas.draw()
                    
                    # Update Stats Label
                    if new_results:
                        lengths = [r['length_px_geodesic'] * _um for r in new_results]
                        lbl_stats.config(text=f'Corrected: {len(new_results)} spermatids | median length {np.median(lengths):.2f} um')
                    else:
                        lbl_stats.config(text=f'Corrected: 0 spermatids')

            # Attach to popup so it doesn't get garbage collected
            top.corrector = ManualCorrector(can, ax2, seg1, params, crop_offset_y, crop_offset_x, full_img)

        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            self.lbl_roi.config(text=f'Analysis error: {e}')
            messagebox.showerror('Analysis Error', f'{type(e).__name__}: {e}')

    def run_batch_analysis(self):
        if not self.files:
            messagebox.showinfo('Info', 'No directory loaded.')
            return

        out_dir = os.path.join(self.input_dir, "batch_output")
        ensure_dir(out_dir)
        overlay_dir = os.path.join(out_dir, "overlays")
        ensure_dir(overlay_dir)

        params = CONFIG.copy()
        params['OUTPUT_DIR'] = out_dir
        params['SAVE_DEBUG_IMAGES'] = False
        params['DO_TRACKING'] = True
        
        roi_mask = self.build_roi_mask()

        self.lbl_roi.config(text=f'Processing {len(self.files)} slices... Check console.')
        self.root.update_idletasks()
        
        try:
            import time as _t
            t_batch = _t.time()
            all_rows = []
            summaries = []
            
            # Z-Projection accumulation
            max_proj_raw = None
            max_proj_ov = None

            self.progress['value'] = 0
            self.progress['maximum'] = len(self.files)
            self.progress_post['value'] = 0
            self.progress_post['maximum'] = 100
            self.lbl_post_progress_val.config(text="0%", fg='orange')
            
            for idx, fpath in enumerate(self.files):
                z_idx = extract_z_index(fpath)
                if z_idx == -1: z_idx = idx
                
                pct = int(((idx + 1) / len(self.files)) * 100)
                self.progress['value'] = idx + 1
                self.lbl_progress_val.config(text=f"{pct}%")
                self.lbl_roi.config(text=f"Processing slice {idx+1}/{len(self.files)} (Z={z_idx:02d})...")
                self.root.update()
                
                print(f"[{idx+1}/{len(self.files)}] Processing Z={z_idx:02d}...")
                
                img_raw = tifffile.imread(fpath)
                
                # Handling ROI cropping to speed up processing
                full_img = img_raw
                crop_oy, crop_ox = 0, 0
                
                # ── BUGFIX: NEVER crop the image before analysis ──
                # Cropping to a tight ROI artificially removes the dark background,
                # which completely destroys the np.percentile() thresholding math that
                # relies on the global image statistics!
                process_img = full_img
                crop_roi = roi_mask
                seg = segment_slice(process_img, params, z_idx=z_idx)
                meas = measure_spermatids(seg, params)
                res = meas['results']
                sl_crop = meas['skel_label']
                
                if crop_roi is not None and res:
                    filtered = []
                    keep = []
                    for r in res:
                        cy = min(max(int(round(r['centroid_y'])), 0), crop_roi.shape[0]-1)
                        cx = min(max(int(round(r['centroid_x'])), 0), crop_roi.shape[1]-1)
                        if crop_roi[cy, cx]:
                            filtered.append(r)
                            keep.append(r['label'])
                    res = filtered
                    sl_crop = np.where(np.isin(sl_crop, keep), sl_crop, 0).astype(np.int32)
                
                for r in res:
                    r['centroid_x'] += crop_ox
                    r['centroid_y'] += crop_oy
                    
                H, W = full_img.shape
                sl_full = np.zeros((H, W), dtype=np.int32)
                ch, cw = sl_crop.shape
                sl_full[crop_oy:crop_oy+ch, crop_ox:crop_ox+cw] = sl_crop
                
                um = params['UM_PER_PX_XY']
                ls_um = [r['length_px_geodesic']*um for r in res]
                
                all_rows.extend(rows_from_results(res, z_idx, um))
                summaries.append({
                    "z_slice": z_idx,
                    "n_spermatids": len(res),
                    "median_length_um": round(float(np.median(ls_um)), 3) if ls_um else 0,
                })
                
                if params['SAVE_OVERLAYS']:
                    ov = make_overlay(full_img, sl_full)
                    # Create side-by-side panel
                    orig_rgb = (normalize_display(full_img) * 255).astype(np.uint8)
                    if orig_rgb.ndim == 2:
                        orig_rgb = np.stack([orig_rgb]*3, axis=-1)
                    panel = np.hstack([orig_rgb, ov])
                    _imwrite(os.path.join(overlay_dir, f"z{z_idx:02d}_panel.png"), panel)
                    
                    # Update Z-Projections
                    if max_proj_raw is None:
                        max_proj_raw = img_raw.copy().astype(np.float32)
                        max_proj_ov = ov.copy().astype(np.float32)
                    else:
                        max_proj_raw = np.maximum(max_proj_raw, img_raw)
                        max_proj_ov = np.maximum(max_proj_ov, ov.astype(np.float32))
            
            df = pd.DataFrame(all_rows)
            df_sum = pd.DataFrame(summaries)
            df.to_csv(os.path.join(out_dir, "spermatid_measurements.csv"), index=False)
            df_sum.to_csv(os.path.join(out_dir, "slice_summary.csv"), index=False)
            
            if not df.empty:
                self.lbl_roi.config(text='Running 3D Tracking & Morphometrics...')
                self.progress_post['value'] = 25
                self.lbl_post_progress_val.config(text="25%")
                self.root.update()

                df_trk, ts = track_across_slices(df, params)
                
                # --- Advanced 3D Morphometrics ---
                self.lbl_roi.config(text='Calculating Advanced 3D Metrics...')
                ts['total_3d_length_um'] = 0.0
                ts['z_extent_um'] = 0.0
                ts['volume_um3'] = 0.0
                ts['tortuosity_3d'] = 1.0
                
                for tid in ts['track_id']:
                    track_data = df_trk[df_trk['track_id'] == tid]
                    if track_data.empty: continue
                    
                    # 1. Total 3D Length (Hypotenuse of Max Lateral x Z-Extent)
                    # Use the maximum lateral geodesic length found in any single slice as the XY projection
                    xy_comp = track_data['length_um_geodesic'].max()
                    
                    # 2. Z-Extent
                    z_span = (track_data['z_slice'].max() - track_data['z_slice'].min() + 1) * params['UM_PER_SLICE_Z']
                    ts.loc[ts['track_id'] == tid, 'z_extent_um'] = z_span
                    
                    # Calculate 3D Geodesic Hypotenuse
                    tot_len_3d = np.sqrt(xy_comp**2 + z_span**2)
                    ts.loc[ts['track_id'] == tid, 'total_3d_length_um'] = tot_len_3d
                    
                    # 3. Volume (Area * Z)
                    vol = track_data['area_px'].sum() * (params['UM_PER_PX_XY']**2) * params['UM_PER_SLICE_Z']
                    ts.loc[ts['track_id'] == tid, 'volume_um3'] = vol
                    
                    # 4. 3D Tortuosity
                    # Calculate 3D Euclidean distance between start and end centroids
                    start = track_data.iloc[0]
                    end = track_data.iloc[-1]
                    dz = (end['z_slice'] - start['z_slice']) * params['UM_PER_SLICE_Z']
                    dy = (end['centroid_y'] - start['centroid_y']) * params['UM_PER_PX_XY']
                    dx = (end['centroid_x'] - start['centroid_x']) * params['UM_PER_PX_XY']
                    dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
                    ts.loc[ts['track_id'] == tid, 'tortuosity_3d'] = tot_len_3d / (dist_3d + 1e-6)

                self.progress_post['value'] = 60
                self.lbl_post_progress_val.config(text="60%")
                self.root.update()

                df_trk.to_csv(os.path.join(out_dir, "measurements_with_tracks.csv"), index=False)
                ts.to_csv(os.path.join(out_dir, "track_summary.csv"), index=False)
                
                # Save Global Z-Projection
                if max_proj_raw is not None:
                    self.lbl_roi.config(text='Generating Global Z-Projection...')
                    raw_p = (normalize_display(max_proj_raw.astype(np.uint16)) * 255).astype(np.uint8)
                    if raw_p.ndim == 2: raw_p = np.stack([raw_p]*3, axis=-1)
                    ov_p = max_proj_ov.astype(np.uint8)
                    global_panel = np.hstack([raw_p, ov_p])
                    _imwrite(os.path.join(out_dir, "global_z_projection.png"), global_panel)
                    
                self.progress_post['value'] = 80
                self.lbl_post_progress_val.config(text="80%")
                self.root.update()
            
            elapsed = _t.time() - t_batch
            msg = f"Batch complete in {elapsed:.1f}s!\nSaved to: {out_dir}"
            print(msg)
            self.lbl_roi.config(text='Batch analysis complete.')
            self.lbl_progress_val.config(text="100% - Done", fg='green')
            self.root.update()
            
            # Generate High-Res Graphical Report
            self.lbl_roi.config(text='Generating PDF Report...')
            
            def update_cb(v):
                self.progress_post['value'] = v
                self.lbl_post_progress_val.config(text=f"{v}%")
                self.root.update()

            generate_batch_report(out_dir, df, df_sum, um, ts if not df.empty else None, update_cb)
            
            msg = f"Batch complete in {elapsed:.1f}s!\nSaved to: {out_dir}"
            print(msg)
            self.lbl_roi.config(text='Batch analysis complete. DONE.')
            self.lbl_progress_val.config(text="100%", fg='green')
            self.lbl_post_progress_val.config(text="100% - DONE", fg='green')
            self.progress_post['value'] = 100
            self.root.update()

            messagebox.showinfo('Batch Complete', msg)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_roi.config(text=f'Batch error: {e}')
            messagebox.showerror('Batch Error', str(e))



def launch_gui():
    if not _TK_AVAILABLE:
        raise RuntimeError("Tkinter GUI components are not available in this Python environment.")
    root = tk.Tk()
    app = SpermGUI(root)
    root.mainloop()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"Spermatid segmentation {_VERSION} / ROI GUI v9")
    ap.add_argument("--batch",  action="store_true", help="Run batch processing")
    ap.add_argument("--single", action="store_true", help="Run a single-image analysis")
    ap.add_argument("--gui",    action="store_true", help="Launch ROI GUI")
    ap.add_argument("--z",      type=int, default=None, help="Choose z-index in single mode")
    args = ap.parse_args()

    validate_config(CONFIG)

    # Launch GUI by default if no explicit CLI flags are provided
    if args.gui or not (args.batch or args.single or args.z is not None):
        launch_gui()
        raise SystemExit

    if args.batch:
        CONFIG["RUN_MODE"] = "batch"
    if args.single:
        CONFIG["RUN_MODE"] = "single"
    if args.z is not None:
        CONFIG["SINGLE_IMAGE_SELECTION_MODE"] = "z_index"
        CONFIG["SINGLE_Z_INDEX"] = args.z

    if CONFIG["RUN_MODE"] == "single":
        process_one_image(choose_single_image(CONFIG), CONFIG, CONFIG["OUTPUT_DIR"])
    else:
        process_batch(CONFIG)
