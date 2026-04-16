#!/usr/bin/env python3
"""
Spermatid Segmentation Pipeline v5

Features
--------
1. All tunable parameters in one CONFIG block at the top
2. Single-image mode or batch mode
3. In single-image mode you can choose image by:
   - explicit path
   - z-index
   - file picker dialog
4. Optional on-screen preview using matplotlib
5. Saves debug images + overlays + measurements
6. Geodesic skeleton length
7. Optional tracking across Z in batch mode

Typical workflow
----------------
1. Set RUN_MODE = "single"
2. Choose one of:
      SINGLE_IMAGE_SELECTION_MODE = "path"
      SINGLE_IMAGE_SELECTION_MODE = "z_index"
      SINGLE_IMAGE_SELECTION_MODE = "dialog"
3. Run script and inspect preview/debug output
4. Tune parameters in CONFIG
5. When good, set RUN_MODE = "batch"
"""

import os, glob, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tifffile

import matplotlib.pyplot as plt

from skimage import measure, morphology, exposure
from skimage.filters import meijering, gaussian, apply_hysteresis_threshold
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

# =============================================================================
# CONFIG -- EDIT HERE
# =============================================================================

CONFIG = {
    # -------------------------
    # RUN MODE
    # -------------------------
    "RUN_MODE": "single",   # "single" or "batch"

    # -------------------------
    # SINGLE IMAGE SELECTION
    # -------------------------
    # options: "path", "z_index", "dialog"
    "SINGLE_IMAGE_SELECTION_MODE": "dialog",

    # used if mode == "path"
    "SINGLE_TEST_IMAGE": r"C:\Users\dmishra\Desktop\sperm images\Project001_Series002_z15_ch00.tif",

    # used if mode == "z_index"
    "SINGLE_Z_INDEX": 15,

    # -------------------------
    # INPUT / OUTPUT
    # -------------------------
    "INPUT_DIR":  r"C:\Users\dmishra\Desktop\sperm images",
    "OUTPUT_DIR": r"C:\Users\dmishra\Desktop\sperm images\sperm_results_v5",
    "FILE_PATTERN": "Project001_Series002_z*_ch00.tif",

    # -------------------------
    # CALIBRATION
    # -------------------------
    "UM_PER_PX_XY": 0.378788,
    "UM_PER_SLICE_Z": 0.346184,

    # -------------------------
    # IMAGE ENHANCEMENT
    # -------------------------
    "CLAHE_CLIP": 0.06,
    "CLAHE_KERNEL": 64,
    "BG_SIGMA": 18,
    "RIDGE_SIGMAS": [1, 2, 3],

    # -------------------------
    # THRESHOLDING
    # -------------------------
    "THRESHOLD_HI": 95,
    "THRESHOLD_LO": 80,

    # -------------------------
    # MORPHOLOGY / CLEANUP
    # -------------------------
    "CLOSE_RADIUS": 1,
    "MIN_HOLE_AREA": 20,
    "MIN_OBJ_PX": 12,

    # -------------------------
    # SHAPE FILTERS
    # -------------------------
    "MIN_ECCENTRICITY": 0.82,
    "MAX_MINOR_PX": 7.5,
    "MIN_AXIS_RATIO": 2.2,
    "MIN_MAJOR_PX": 8,

    # -------------------------
    # BRIDGING
    # -------------------------
    "BRIDGE_RADIUS": 2,

    # -------------------------
    # SKELETON / TOPOLOGY
    # -------------------------
    "MIN_SKEL_LEN_PX": 6,
    "REJECT_BRANCHES": False,
    "ALLOW_LOOPS": False,

    # -------------------------
    # TRACKING
    # -------------------------
    "DO_TRACKING": True,
    "TRACK_MAX_DIST_UM": 4.0,
    "TRACK_MAX_GAP_SLICES": 1,

    # -------------------------
    # OUTPUT / DEBUG
    # -------------------------
    "SAVE_DEBUG_IMAGES": True,
    "SAVE_MASK_TIFS": True,
    "SAVE_LABEL_TIFS": True,
    "SAVE_OVERLAYS": True,
    "SAVE_DETAIL_FIGURE": True,

    # show preview window in single mode
    "SHOW_PREVIEW_WINDOW": True,

    # if True, also show debug stages in preview
    "SHOW_DEBUG_PREVIEW": True,
}

# =============================================================================
# END CONFIG
# =============================================================================

_NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_z_index(path):
    m = re.search(r"_z(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else -1


def normalize_display(img):
    img = img.astype(np.float32)
    return np.clip(
        (img - np.percentile(img, 1)) /
        (np.percentile(img, 99.5) - np.percentile(img, 1) + 1e-9),
        0, 1
    )


def load_batch_files(input_dir, pattern):
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in '{input_dir}'")
    files = sorted(files, key=extract_z_index)
    z_indices = [extract_z_index(f) for f in files]
    print(f"Found {len(files)} slices: {z_indices}")
    return files, z_indices


def choose_single_image(cfg):
    mode = cfg["SINGLE_IMAGE_SELECTION_MODE"].lower()

    if mode == "path":
        path = cfg["SINGLE_TEST_IMAGE"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Single test image not found:\n{path}")
        return path

    if mode == "z_index":
        z = int(cfg["SINGLE_Z_INDEX"])
        files = glob.glob(os.path.join(cfg["INPUT_DIR"], cfg["FILE_PATTERN"]))
        if not files:
            raise FileNotFoundError(f"No files matching {cfg['FILE_PATTERN']} in {cfg['INPUT_DIR']}")
        hits = [f for f in files if extract_z_index(f) == z]
        if not hits:
            raise FileNotFoundError(f"No file found for z-index {z}")
        hits = sorted(hits)
        print(f"Using z-index {z}: {hits[0]}")
        return hits[0]

    if mode == "dialog":
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Choose single image to process",
                initialdir=cfg["INPUT_DIR"],
                filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
            )
            root.destroy()
        except Exception as e:
            raise RuntimeError(f"Could not open file dialog: {e}")

        if not path:
            raise RuntimeError("No file selected.")
        print(f"Selected: {path}")
        return path

    raise ValueError("SINGLE_IMAGE_SELECTION_MODE must be 'path', 'z_index', or 'dialog'")


def save_gray(path, img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    plt.imsave(path, img, cmap="gray")


def save_mask(path, mask):
    plt.imsave(path, mask.astype(np.uint8) * 255, cmap="gray")


def segment_slice(img_raw, cfg, z_idx=None, debug_dir=None):
    img = img_raw.astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)

    img_eq = exposure.equalize_adapthist(
        img_norm,
        clip_limit=cfg["CLAHE_CLIP"],
        kernel_size=cfg["CLAHE_KERNEL"]
    )

    bg = gaussian(img_eq, sigma=cfg["BG_SIGMA"])
    fg = np.clip(img_eq - bg, 0, None)
    fgn = fg / (fg.max() + 1e-9)

    ridge = meijering(fgn, sigmas=cfg["RIDGE_SIGMAS"], black_ridges=False)

    th_hi = np.percentile(ridge, cfg["THRESHOLD_HI"])
    th_lo = np.percentile(ridge, cfg["THRESHOLD_LO"])
    mask = apply_hysteresis_threshold(ridge, th_lo, th_hi)

    mask = morphology.binary_closing(mask, morphology.disk(cfg["CLOSE_RADIUS"]))
    mask = morphology.remove_small_holes(mask, area_threshold=cfg["MIN_HOLE_AREA"])
    mask = morphology.remove_small_objects(mask, min_size=cfg["MIN_OBJ_PX"])

    labeled1 = measure.label(mask)
    keep1 = []
    for prop in measure.regionprops(labeled1):
        ratio = prop.major_axis_length / (prop.minor_axis_length + 1e-9)
        if prop.eccentricity < cfg["MIN_ECCENTRICITY"]:
            continue
        if prop.minor_axis_length > cfg["MAX_MINOR_PX"]:
            continue
        if ratio < cfg["MIN_AXIS_RATIO"]:
            continue
        if prop.major_axis_length < cfg["MIN_MAJOR_PX"]:
            continue
        keep1.append(prop.label)
    thin_mask = np.isin(labeled1, keep1)

    br = cfg["BRIDGE_RADIUS"]
    bridged = morphology.binary_dilation(thin_mask, morphology.disk(br))
    bridged = morphology.remove_small_objects(bridged, min_size=cfg["MIN_OBJ_PX"])

    labeled2 = measure.label(bridged)
    keep2 = []
    for prop in measure.regionprops(labeled2):
        ratio = prop.major_axis_length / (prop.minor_axis_length + 1e-9)
        if ratio < max(1.8, cfg["MIN_AXIS_RATIO"] - 0.4):
            continue
        if prop.major_axis_length < max(6, cfg["MIN_MAJOR_PX"] - 1):
            continue
        keep2.append(prop.label)

    bridged_filtered = np.isin(labeled2, keep2)

    if br > 0:
        final_mask = morphology.binary_erosion(bridged_filtered, morphology.disk(br))
    else:
        final_mask = bridged_filtered.copy()

    final_mask = morphology.remove_small_objects(final_mask, min_size=cfg["MIN_OBJ_PX"])

    if bridged_filtered.sum() > 0 and final_mask.sum() < 0.55 * bridged_filtered.sum():
        final_mask = bridged_filtered.copy()

    skel = skeletonize(final_mask)
    dist = distance_transform_edt(final_mask)

    if cfg["SAVE_DEBUG_IMAGES"] and debug_dir is not None and z_idx is not None:
        save_gray(os.path.join(debug_dir, f"z{z_idx:02d}_01_norm.png"), img_norm)
        save_gray(os.path.join(debug_dir, f"z{z_idx:02d}_02_clahe.png"), img_eq)
        save_gray(os.path.join(debug_dir, f"z{z_idx:02d}_03_fg.png"), fgn)
        save_gray(os.path.join(debug_dir, f"z{z_idx:02d}_04_ridge.png"), ridge)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_05_hysteresis.png"), mask)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_06_pass1.png"), thin_mask)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_07_bridged.png"), bridged)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_08_pass2.png"), bridged_filtered)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_09_final_mask.png"), final_mask)
        save_mask(os.path.join(debug_dir, f"z{z_idx:02d}_10_skeleton.png"), skel)

    return {
        "mask": final_mask,
        "skel": skel,
        "dist": dist,
        "img_norm": img_norm,
        "img_eq": img_eq,
        "fg": fgn,
        "ridge": ridge,
        "hysteresis": mask,
        "pass1": thin_mask,
        "bridged": bridged,
        "pass2": bridged_filtered,
    }


def _build_skel_graph(skel_bool):
    coords = np.argwhere(skel_bool)
    if coords.size == 0:
        return None
    node_set = set((int(r), int(c)) for r, c in coords)
    adj = {}
    for r, c in node_set:
        nbrs = []
        for dr, dc in _NEIGHBORS_8:
            rr, cc = r + dr, c + dc
            if (rr, cc) in node_set:
                w = 1.41421356237 if (dr != 0 and dc != 0) else 1.0
                nbrs.append(((rr, cc), w))
        adj[(r, c)] = nbrs
    return adj


def _dijkstra_farthest(adj, start):
    import heapq
    dist = {start: 0.0}
    pq = [(0.0, start)]
    visited = set()
    farthest_node = start
    farthest_dist = 0.0

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if d > farthest_dist:
            farthest_dist = d
            farthest_node = u
        for v, w in adj.get(u, []):
            nd = d + w
            if (v not in dist) or (nd < dist[v]):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return farthest_node, farthest_dist


def geodesic_length_px_from_skeleton(skel_component_bool, reject_branches=False, allow_loops=False):
    adj = _build_skel_graph(skel_component_bool)
    if not adj:
        return None, {"reason": "empty"}

    deg = {k: len(v) for k, v in adj.items()}
    degrees = np.array(list(deg.values()), dtype=int)

    if reject_branches and np.any(degrees > 2):
        return None, {"reason": "branch_detected"}

    endpoints = [n for n, d in deg.items() if d == 1]

    if len(endpoints) == 0:
        if not allow_loops:
            return None, {"reason": "loop_detected"}
        edge_sum = 0.0
        seen = set()
        for u, nbrs in adj.items():
            for v, w in nbrs:
                key = tuple(sorted([u, v]))
                if key in seen:
                    continue
                seen.add(key)
                edge_sum += w
        return float(edge_sum), {"reason": "loop_ok"}

    start = endpoints[0]
    b, _ = _dijkstra_farthest(adj, start)
    _, dist_bc = _dijkstra_farthest(adj, b)
    return float(dist_bc), {"reason": "ok", "endpoints": len(endpoints)}


def measure_spermatids(seg, cfg):
    skel = seg["skel"]
    dist = seg["dist"]
    H, W = skel.shape

    skel_labeled = measure.label(skel)
    results = []

    for sp in measure.regionprops(skel_labeled):
        coords = sp.coords
        if coords.shape[0] < cfg["MIN_SKEL_LEN_PX"]:
            continue

        comp = np.zeros((H, W), dtype=bool)
        comp[coords[:, 0], coords[:, 1]] = True

        geo_len_px, info = geodesic_length_px_from_skeleton(
            comp,
            reject_branches=cfg["REJECT_BRANCHES"],
            allow_loops=cfg["ALLOW_LOOPS"]
        )
        if geo_len_px is None:
            continue

        width_px = float(np.median(2.0 * dist[coords[:, 0], coords[:, 1]]))
        cy, cx = sp.centroid

        results.append({
            "label": sp.label,
            "length_px_geodesic": float(geo_len_px),
            "length_px_count": float(coords.shape[0]),
            "width_px": width_px,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "topo_reason": info.get("reason", "ok"),
            "topo_endpoints": info.get("endpoints", np.nan),
        })

    if not results:
        return {"skel_label": np.zeros_like(skel_labeled, dtype=np.int32), "results": []}

    keep_labels = [r["label"] for r in results]
    clean = np.isin(skel_labeled, keep_labels)
    final_label = measure.label(clean).astype(np.int32)

    final_results = []
    for i, sp in enumerate(measure.regionprops(final_label), start=1):
        coords = sp.coords
        comp = np.zeros((H, W), dtype=bool)
        comp[coords[:, 0], coords[:, 1]] = True
        geo_len_px, info = geodesic_length_px_from_skeleton(
            comp,
            reject_branches=cfg["REJECT_BRANCHES"],
            allow_loops=cfg["ALLOW_LOOPS"]
        )
        if geo_len_px is None:
            continue
        width_px = float(np.median(2.0 * dist[coords[:, 0], coords[:, 1]]))
        cy, cx = sp.centroid
        final_results.append({
            "label": i,
            "length_px_geodesic": float(geo_len_px),
            "length_px_count": float(coords.shape[0]),
            "width_px": width_px,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "topo_reason": info.get("reason", "ok"),
            "topo_endpoints": info.get("endpoints", np.nan),
        })

    return {"skel_label": final_label, "results": final_results}


def make_overlay(img_raw, skel_label):
    base = normalize_display(img_raw)
    rgb = np.stack([base] * 3, axis=-1)

    n = int(skel_label.max())
    if n <= 0:
        return (rgb * 255).astype(np.uint8)

    cols = plt.cm.gist_rainbow(np.linspace(0, 1, n))[:, :3]
    for idx in range(1, n + 1):
        tmp = morphology.binary_dilation(skel_label == idx, morphology.disk(1))
        c = cols[(idx - 1) % n]
        rgb[tmp, 0] = c[0]
        rgb[tmp, 1] = c[1]
        rgb[tmp, 2] = c[2]

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def show_single_preview(img_raw, seg, overlay_rgb, results, z_idx, cfg):
    um = cfg["UM_PER_PX_XY"]
    lengths_um = [r["length_px_geodesic"] * um for r in results]

    if cfg["SHOW_DEBUG_PREVIEW"]:
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))

        axes[0, 0].imshow(normalize_display(img_raw), cmap="gray")
        axes[0, 0].set_title(f"Original (Z={z_idx:02d})")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(seg["img_eq"], cmap="gray")
        axes[0, 1].set_title("CLAHE")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(seg["ridge"], cmap="gray")
        axes[0, 2].set_title("Ridge")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(seg["hysteresis"], cmap="gray")
        axes[0, 3].set_title("Hysteresis")
        axes[0, 3].axis("off")

        axes[1, 0].imshow(seg["pass1"], cmap="gray")
        axes[1, 0].set_title("Pass 1")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(seg["pass2"], cmap="gray")
        axes[1, 1].set_title("Pass 2")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(overlay_rgb)
        axes[1, 2].set_title(f"Overlay (N={len(results)})")
        axes[1, 2].axis("off")

        for r in results:
            axes[1, 2].text(
                r["centroid_x"], r["centroid_y"],
                f"{r['length_px_geodesic'] * um:.1f}",
                color="white", fontsize=5, ha="center", va="center"
            )

        if lengths_um:
            axes[1, 3].hist(lengths_um, bins=20, edgecolor="white")
            axes[1, 3].axvline(np.median(lengths_um), lw=2)
            axes[1, 3].set_title(f"Length hist\nmedian={np.median(lengths_um):.1f} um")
            axes[1, 3].set_xlabel("um")
            axes[1, 3].set_ylabel("Count")
        else:
            axes[1, 3].text(0.5, 0.5, "No detections", ha="center", va="center")
            axes[1, 3].axis("off")

        plt.tight_layout()
        plt.show()

    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(normalize_display(img_raw), cmap="gray")
        axes[0].set_title(f"Original (Z={z_idx:02d})")
        axes[0].axis("off")

        axes[1].imshow(overlay_rgb)
        axes[1].set_title(f"Overlay (N={len(results)})")
        axes[1].axis("off")

        for r in results:
            axes[1].text(
                r["centroid_x"], r["centroid_y"],
                f"{r['length_px_geodesic'] * um:.1f}",
                color="white", fontsize=5, ha="center", va="center"
            )

        if lengths_um:
            axes[2].hist(lengths_um, bins=20, edgecolor="white")
            axes[2].axvline(np.median(lengths_um), lw=2)
            axes[2].set_title(f"Length hist\nmedian={np.median(lengths_um):.1f} um")
            axes[2].set_xlabel("um")
            axes[2].set_ylabel("Count")
        else:
            axes[2].text(0.5, 0.5, "No detections", ha="center", va="center")
            axes[2].axis("off")

        plt.tight_layout()
        plt.show()


def save_detail_figure(img_raw, overlay_rgb, results, out_path, z_idx, um):
    lengths_um = [r["length_px_geodesic"] * um for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(normalize_display(img_raw), cmap="gray")
    axes[0].set_title(f"Z={z_idx:02d} - Original")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Z={z_idx:02d} - Spermatids (N={len(results)})")
    axes[1].axis("off")

    for r in results:
        axes[1].text(
            r["centroid_x"], r["centroid_y"],
            f"{r['length_px_geodesic'] * um:.1f}",
            color="white", fontsize=4, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0)
        )

    if lengths_um:
        axes[2].hist(lengths_um, bins=20, edgecolor="white")
        axes[2].axvline(np.median(lengths_um), lw=2, label=f"Median={np.median(lengths_um):.1f} um")
        axes[2].set_xlabel("Geodesic length (um)")
        axes[2].set_ylabel("Count")
        axes[2].set_title(f"Z={z_idx:02d} - Length distribution")
        axes[2].legend(fontsize=9)
    else:
        axes[2].text(0.5, 0.5, "No spermatids detected",
                     transform=axes[2].transAxes, ha="center", va="center")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def process_one_image(image_path, cfg, output_dir):
    ensure_dir(output_dir)
    overlay_dir = os.path.join(output_dir, "overlays")
    debug_dir = os.path.join(output_dir, "debug")
    ensure_dir(overlay_dir)
    if cfg["SAVE_DEBUG_IMAGES"]:
        ensure_dir(debug_dir)

    z_idx = extract_z_index(image_path)
    print(f"\nProcessing single image: {os.path.basename(image_path)}")

    img_raw = tifffile.imread(image_path)
    seg = segment_slice(
        img_raw, cfg,
        z_idx=z_idx,
        debug_dir=debug_dir if cfg["SAVE_DEBUG_IMAGES"] else None
    )
    meas = measure_spermatids(seg, cfg)

    results = meas["results"]
    skel_label = meas["skel_label"]
    um = cfg["UM_PER_PX_XY"]

    print(f"Detected spermatids: {len(results)}")
    if results:
        lengths_um = [r["length_px_geodesic"] * um for r in results]
        print(f"Median geodesic length: {np.median(lengths_um):.2f} um")

    overlay_rgb = make_overlay(img_raw, skel_label)

    if cfg["SAVE_OVERLAYS"]:
        plt.imsave(os.path.join(overlay_dir, f"z{z_idx:02d}_overlay.png"), overlay_rgb)

    if cfg["SAVE_DETAIL_FIGURE"]:
        save_detail_figure(
            img_raw,
            overlay_rgb,
            results,
            os.path.join(overlay_dir, f"z{z_idx:02d}_detail.png"),
            z_idx,
            um
        )

    if cfg["SAVE_MASK_TIFS"]:
        tifffile.imwrite(
            os.path.join(output_dir, f"z{z_idx:02d}_mask.tif"),
            seg["mask"].astype(np.uint8) * 255
        )

    if cfg["SAVE_LABEL_TIFS"]:
        tifffile.imwrite(
            os.path.join(output_dir, f"z{z_idx:02d}_skel_labels.tif"),
            skel_label.astype(np.uint16)
        )

    rows = []
    for i, r in enumerate(results, start=1):
        rows.append({
            "z_slice": z_idx,
            "sperm_id": i,
            "length_px_geodesic": round(r["length_px_geodesic"], 3),
            "length_um_geodesic": round(r["length_px_geodesic"] * um, 3),
            "length_px_count": round(r["length_px_count"], 1),
            "length_um_count": round(r["length_px_count"] * um, 3),
            "width_px": round(r["width_px"], 2),
            "width_um": round(r["width_px"] * um, 3),
            "centroid_x": round(r["centroid_x"], 1),
            "centroid_y": round(r["centroid_y"], 1),
            "topo_reason": r["topo_reason"],
            "topo_endpoints": r["topo_endpoints"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "single_image_measurements.csv"), index=False)

    if cfg["SHOW_PREVIEW_WINDOW"]:
        show_single_preview(img_raw, seg, overlay_rgb, results, z_idx, cfg)

    print(f"DONE - single-image results saved to:\n{output_dir}")


def track_across_slices(detections_df, cfg):
    if detections_df.empty:
        detections_df["track_id"] = []
        return detections_df, pd.DataFrame()

    max_dist_px = cfg["TRACK_MAX_DIST_UM"] / (cfg["UM_PER_PX_XY"] + 1e-9)
    df = detections_df.copy().sort_values(["z_slice", "sperm_id"]).reset_index(drop=True)

    next_track_id = 1
    active = {}
    track_ids = [-1] * len(df)

    z_values = sorted(df["z_slice"].unique().tolist())
    rows_by_z = {z: df.index[df["z_slice"] == z].to_numpy() for z in z_values}

    for z in z_values:
        idxs = rows_by_z[z]
        xs = df.loc[idxs, "centroid_x"].to_numpy(float)
        ys = df.loc[idxs, "centroid_y"].to_numpy(float)

        cand_tracks, cand_pos = [], []
        for tid, st in active.items():
            dz = z - st["last_z"]
            if 1 <= dz <= (cfg["TRACK_MAX_GAP_SLICES"] + 1):
                cand_tracks.append(tid)
                cand_pos.append((st["last_x"], st["last_y"]))

        if cand_tracks:
            tree = cKDTree(np.array(cand_pos, dtype=float))
            matches = []
            for k, (x, y) in enumerate(zip(xs, ys)):
                d, j = tree.query([x, y], k=1)
                if np.isfinite(d) and d <= max_dist_px:
                    matches.append((float(d), k, int(j)))
            matches.sort(key=lambda t: t[0])

            used_det, used_trk = set(), set()
            for d, det_k, trk_j in matches:
                if det_k in used_det or trk_j in used_trk:
                    continue
                used_det.add(det_k)
                used_trk.add(trk_j)
                tid = cand_tracks[trk_j]
                row_i = int(idxs[det_k])
                track_ids[row_i] = tid
                active[tid] = {"last_z": int(z), "last_x": float(xs[det_k]), "last_y": float(ys[det_k])}

            for det_k in range(len(idxs)):
                row_i = int(idxs[det_k])
                if track_ids[row_i] == -1:
                    tid = next_track_id
                    next_track_id += 1
                    track_ids[row_i] = tid
                    active[tid] = {"last_z": int(z), "last_x": float(xs[det_k]), "last_y": float(ys[det_k])}
        else:
            for det_k in range(len(idxs)):
                row_i = int(idxs[det_k])
                tid = next_track_id
                next_track_id += 1
                track_ids[row_i] = tid
                active[tid] = {"last_z": int(z), "last_x": float(xs[det_k]), "last_y": float(ys[det_k])}

        to_del = []
        for tid, st in active.items():
            if (z - st["last_z"]) > (cfg["TRACK_MAX_GAP_SLICES"] + 1):
                to_del.append(tid)
        for tid in to_del:
            del active[tid]

    df["track_id"] = track_ids

    g = df.groupby("track_id", as_index=False)
    track_sum = g.agg(
        n_detections=("sperm_id", "count"),
        z_min=("z_slice", "min"),
        z_max=("z_slice", "max"),
        mean_centroid_x=("centroid_x", "mean"),
        mean_centroid_y=("centroid_y", "mean"),
        max_length_um=("length_um_geodesic", "max"),
        median_width_um=("width_um", "median"),
    )
    track_sum["z_extent_um"] = (track_sum["z_max"] - track_sum["z_min"]) * cfg["UM_PER_SLICE_Z"]
    track_sum["length_3d_um_est"] = np.sqrt(track_sum["max_length_um"]**2 + track_sum["z_extent_um"]**2)

    return df, track_sum


def process_batch(cfg):
    ensure_dir(cfg["OUTPUT_DIR"])
    overlay_dir = os.path.join(cfg["OUTPUT_DIR"], "overlays")
    debug_dir = os.path.join(cfg["OUTPUT_DIR"], "debug")
    ensure_dir(overlay_dir)
    if cfg["SAVE_DEBUG_IMAGES"]:
        ensure_dir(debug_dir)

    files, z_indices = load_batch_files(cfg["INPUT_DIR"], cfg["FILE_PATTERN"])
    um = cfg["UM_PER_PX_XY"]

    all_rows = []
    summaries = []

    for fpath, z_idx in zip(files, z_indices):
        print(f"\n--- Z={z_idx:02d}  {os.path.basename(fpath)} ---")

        img_raw = tifffile.imread(fpath)
        seg = segment_slice(
            img_raw, cfg,
            z_idx=z_idx,
            debug_dir=debug_dir if cfg["SAVE_DEBUG_IMAGES"] else None
        )
        meas = measure_spermatids(seg, cfg)

        results = meas["results"]
        skel_label = meas["skel_label"]
        lengths_um = [r["length_px_geodesic"] * um for r in results]
        widths_um = [r["width_px"] * um for r in results]

        print(f"  Spermatids: {len(results)}")
        if lengths_um:
            print(f"  Median geodesic length: {np.median(lengths_um):.2f} um")

        for i, r in enumerate(results, start=1):
            all_rows.append({
                "z_slice": z_idx,
                "sperm_id": i,
                "length_px_geodesic": round(r["length_px_geodesic"], 3),
                "length_um_geodesic": round(r["length_px_geodesic"] * um, 3),
                "length_px_count": round(r["length_px_count"], 1),
                "length_um_count": round(r["length_px_count"] * um, 3),
                "width_px": round(r["width_px"], 2),
                "width_um": round(r["width_px"] * um, 3),
                "centroid_x": round(r["centroid_x"], 1),
                "centroid_y": round(r["centroid_y"], 1),
                "topo_reason": r["topo_reason"],
                "topo_endpoints": r["topo_endpoints"],
            })

        summaries.append({
            "z_slice": z_idx,
            "n_spermatids": len(results),
            "mean_length_um": round(np.mean(lengths_um), 3) if lengths_um else 0,
            "median_length_um": round(np.median(lengths_um), 3) if lengths_um else 0,
            "mean_width_um": round(np.mean(widths_um), 3) if widths_um else 0,
        })

        overlay_rgb = make_overlay(img_raw, skel_label)

        if cfg["SAVE_OVERLAYS"]:
            plt.imsave(os.path.join(overlay_dir, f"z{z_idx:02d}_overlay.png"), overlay_rgb)

        if cfg["SAVE_DETAIL_FIGURE"]:
            save_detail_figure(
                img_raw,
                overlay_rgb,
                results,
                os.path.join(overlay_dir, f"z{z_idx:02d}_detail.png"),
                z_idx,
                um
            )

        if cfg["SAVE_MASK_TIFS"]:
            tifffile.imwrite(
                os.path.join(cfg["OUTPUT_DIR"], f"z{z_idx:02d}_mask.tif"),
                seg["mask"].astype(np.uint8) * 255
            )

        if cfg["SAVE_LABEL_TIFS"]:
            tifffile.imwrite(
                os.path.join(cfg["OUTPUT_DIR"], f"z{z_idx:02d}_skel_labels.tif"),
                skel_label.astype(np.uint16)
            )

    df = pd.DataFrame(all_rows)
    df_sum = pd.DataFrame(summaries)

    df.to_csv(os.path.join(cfg["OUTPUT_DIR"], "spermatid_measurements_v5.csv"), index=False)
    df_sum.to_csv(os.path.join(cfg["OUTPUT_DIR"], "slice_summary_v5.csv"), index=False)

    if cfg["DO_TRACKING"] and not df.empty:
        df_trk, track_sum = track_across_slices(df, cfg)
        df_trk.to_csv(os.path.join(cfg["OUTPUT_DIR"], "spermatid_measurements_v5_with_tracks.csv"), index=False)
        track_sum.to_csv(os.path.join(cfg["OUTPUT_DIR"], "track_summary_v5.csv"), index=False)

    print(f"\nDONE - batch results saved to:\n{cfg['OUTPUT_DIR']}")


if __name__ == "__main__":
    if CONFIG["RUN_MODE"].lower() == "single":
        img_path = choose_single_image(CONFIG)
        process_one_image(img_path, CONFIG, CONFIG["OUTPUT_DIR"])
    elif CONFIG["RUN_MODE"].lower() == "batch":
        process_batch(CONFIG)
    else:
        raise ValueError("CONFIG['RUN_MODE'] must be 'single' or 'batch'")