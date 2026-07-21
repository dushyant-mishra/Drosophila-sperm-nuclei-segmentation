# --- ROBUST LOGGING SYSTEM (macOS STABILITY PATCH) ---
import os
import sys

try:
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sperm_error_log.txt")
    f_log = open(log_file_path, 'a', encoding='utf-8')
except (PermissionError, IOError):
    # Fallback to User Home if script directory is read-only (common in macOS .app bundles)
    log_file_path = os.path.join(os.path.expanduser("~"), "sperm_error_log.txt")
    f_log = open(log_file_path, 'a', encoding='utf-8')

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except: pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except: pass

sys.stdout = Tee(sys.stdout, f_log)
sys.stderr = Tee(sys.stderr, f_log)
print(f"\n--- NEW SESSION STARTED (v5.7-unet-ready): {os.path.basename(__file__)} ---")
print(f"Log Path: {log_file_path}\n")
# ---------------------------------

#!/usr/bin/env python3
"""
Sperm Nucleus Segmentation & 3D Morphometrics Pipeline  -  Saturn Project
=========================================================================
A production-ready image-analysis pipeline for automated detection,
measurement, and 3D reconstruction of sperm nuclei tailored to
the *Saturn* experimental dataset acquired on the Leica confocal system.

Biological context
------------------
Spermiogenesis - the post-meiotic differentiation of round spermatids into
mature spermatozoa - involves dramatic nuclear elongation, chromatin condensation,
and apical plunging of the nucleus toward the basal lamina of the seminiferous
tubule.  Quantifying these morphological changes requires imaging the highly
elongated, condensed sperm nuclei.  This pipeline automates the process by:

1. Detecting sperm nuclei as thin, dim, ridge-like objects in each 2D
   Z-slice using a skeleton-first strategy (Frangi ridge filter -> Otsu
   binarisation -> morphological closing -> geodesic skeleton).
2. Measuring per-cell biometrics: geodesic length (um), Euclidean width,
   tortuosity, endpoint count, branch complexity, centroid, and estimated area.
3. Linking detections across Z-slices into 3D tracks using nearest-neighbor and
   overlap-aware frame-to-frame assignment.
4. Computing 3D morphometrics from tracks: estimated 3D length, Z-span,
   sampled Z-coverage, approximated volume, pitch angle, taper ratio, and
   nearest-neighbor packing density.
5. Exporting results as CSV, a multi-sheet Excel workbook, a multi-page PDF
   report, and a native PowerPoint (.pptx) dashboard with editable charts.

Saturn-specific calibration
----------------------------
Physical scale factors are derived from Leica confocal metadata for the
Saturn dataset:

- ``UM_PER_PX_XY  = 0.7568``  um/pixel  (lateral resolution)
- ``UM_PER_SLICE_Z = 1.0404``  um/slice  (z-step size)

These values are set as defaults in ``CONFIG`` and can be overridden via
the Parameter Editor GUI or by loading a JSON settings file.

v5.2 measurement notes
----------------------
``area_px`` is an estimated slender-object area, computed as 2D geodesic
length times median width. The raw skeleton-pixel count is exported separately
as ``skeleton_area_px`` for audit/debug use.

3D length is a projection-length-plus-Z-span estimate:
``sqrt(max(2D geodesic, XY displacement)^2 + z_span^2)``.  ``z_span_um`` is
endpoint-to-endpoint vertical span; ``z_covered_um`` is the sampled slab
coverage including both end slices. Volume, effective thickness, taper, and
other width/area-derived metrics are PSF- and sampling-sensitive, so use them
mainly for relative comparisons under matched imaging settings.

Pipeline architecture
---------------------
::

    Image stack (TIF / PNG / JPG)
        -> load_batch_files()
        -> process_batch() / process_one_image()
        -> segment_slice()
        -> measure_spermatids()
        -> rows_from_results()
        -> track_across_slices()
        -> generate_excel_report() / generate_batch_report() / generate_pptx_report()

Key configuration parameters (``CONFIG`` dict)
-----------------------------------------------
``UM_PER_PX_XY``       Physical pixel size in um (Leica metadata: 0.7568).
``UM_PER_SLICE_Z``     Z-step size in um (Leica metadata: 1.0404).
``FRANGI_SCALE_RANGE`` Tubeness filter scales (px); set to match sperm nucleus width.
``MIN_SKEL_LEN_PX``    Minimum skeleton length accepted (removes debris/noise).
``MAX_WIDTH_PX``       Maximum skeleton width accepted (rejects merged clusters).
``MAX_BRANCH_NODES``   Maximum branch-point count; >0 tolerates bridged tips.
``MAX_TORTUOSITY``     Maximum curvature index; rejects snaking merged networks.
``DO_TRACKING``        Enable/disable cross-slice 3D linking.
``TRACK_MAX_DIST_UM``  Maximum centroid displacement between adjacent Z-slices.

Usage
-----
Launch the interactive GUI (recommended)::

    python sperm_segmentation_saturnv5.2.py

Run a headless batch analysis::

    python sperm_segmentation_saturnv5.2.py --batch

Analyze a single slice::

    python sperm_segmentation_saturnv5.2.py --single --z 4

Dependencies
------------
numpy, scipy, scikit-image, pandas, matplotlib, tifffile, opencv-python,
Pillow, xlsxwriter, python-pptx, tkinter (stdlib)

Author
------
Dushyant Mishra  |  Findlay Lab  |  Saturn Dataset Branch
"""

import os, sys, glob, re, time, warnings, heapq, argparse, math, pathlib as pl
import time as _t
import json, webbrowser, threading, subprocess
from dataclasses import dataclass, asdict
try:
    import requests
    _HAVE_REQUESTS = True
except ImportError:
    _HAVE_REQUESTS = False
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tifffile
import matplotlib
# --- macOS STABILITY PATCH: Force non-interactive backend for background reporting ---
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path
print(f"[matplotlib backend: {matplotlib.get_backend()}]")

from skimage import measure, morphology, exposure
from skimage.filters import meijering, gaussian, apply_hysteresis_threshold
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, grey_dilation
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
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
    # ------ run mode ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    "RUN_MODE": "single",          # "single" | "batch"
    "ANALYSIS_MODE": "comparative",  # "comparative" | "reference_morphology" | "legacy"

    # ------ single-image selection ------------------------------------------------------------------------------------------------------------------------------------------------
    "SINGLE_IMAGE_SELECTION_MODE": "dialog",  # "path" | "z_index" | "dialog"
    "SINGLE_TEST_IMAGE": r"C:\Users\dmishra\Desktop\sperm images\Project001_Series002_z15_ch00.tif",
    "SINGLE_Z_INDEX": 15,

    # ------ input / output ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    "INPUT_DIR":    ".",
    "OUTPUT_DIR":   "./sperm_results_saturnv5_7",
    "FILE_PATTERN": "Project001_Series002_z*_ch00.tif",
    "ROI_MASK_PATH": "",
    "PREPROCESS_MODE": "roi_adaptive",
    "LEGACY_TWO_PASS_ROI": False,
    "STACK_QC_SAMPLE_COUNT": 12,
    "NORM_LOW_PERCENTILE": 1.0,
    "NORM_HIGH_PERCENTILE": 99.5,
    "NORM_STACK_WEIGHT": 0.80,
    "CLAHE_MODE": "auto_stack",
    "CLAHE_CLIP_HIGH_CONTRAST": 0.010,
    "CLAHE_CLIP_STANDARD": 0.025,
    "CLAHE_CLIP_LOW_CONTRAST": 0.035,
    "AUTO_CONTRAST_HIGH_THRESHOLD": 0.45,
    "AUTO_CONTRAST_LOW_THRESHOLD": 0.25,
    "ROI_CROP_PADDING_PX": 16,
    "ROI_THRESHOLD_PERCENTILES_ONLY": True,
    "EXCLUSION_MASK_PATH": "",
    "ROI_BOUNDARY_SAFE_RIDGE": True,
    "ROI_THRESHOLD_EXCLUDE_BOUNDARY_PX": 4,
    "AI_PREPROCESSING_MODE": "off",

    # ------ optional U-Net 2.5D integration scaffold ---------------------------------------------------------------------------------------------------------------------------------------
    # COCO is training-only and is never read by Saturn during inference.
    # Runtime U-Net inference uses a trained checkpoint plus [z-1, z, z+1] raw planes.
    "SEGMENTATION_ENGINE": "classical_saturn",  # "classical_saturn" | "unet_assisted" | "hybrid"
    "UNET_MODEL_PATH": "",
    "UNET_THRESHOLD": 0.10,
    "UNET_THRESHOLD_MODE": "soft",
    "UNET_CANDIDATE_THRESHOLD": 0.05,
    "UNET_SEED_THRESHOLD": 0.30,
    "UNET_CONTEXT_MODE": "z_minus_z_z_plus",
    "UNET_INFERENCE_MODE": "roi_tiled",
    "UNET_TILE_SIZE": 256,
    "UNET_TILE_OVERLAP": 64,
    "UNET_ROI_PADDING_PX": 32,
    "UNET_STITCH_MODE": "weighted_average",
    "UNET_OUTSIDE_ROI_ZERO": True,
    "UNET_SAVE_PROBABILITY_MAPS": True,
    "UNET_CANDIDATE_ACCOUNTING": True,
    "UNET_RESCUE_ENABLE": True,
    "UNET_RESCUE_THRESHOLD": 0.50,
    "UNET_RESCUE_EXCLUDE_DILATION_PX": 3,
    "UNET_RESCUE_MIN_COMPONENT_PX": 3,
    "UNET_RESCUE_MAX_ADDITIONS_PER_SLICE": 0,
    "UNET_TRACKING_SUPPORT": True,
    "ASSIGNMENT_UNET_SUPPORT_WEIGHT": 0.6,
    "ASSIGNMENT_UNET_CONTINUITY_WEIGHT": 0.25,
    "HYBRID_REPAIR_UNET_SUPPORT_WEIGHT": 0.4,
    "UNET_DETECTED_LABEL": "Detected by U-Net",
    "UNET_RESCUED_LABEL": "U-Net rescued",
    "UNET_COMPLETED_BY_BRIDGE_LABEL": "Completed by bridge",
    "UNET_COMPLETED_BY_EXTENSION_LABEL": "Completed by extension",
    "UNET_MERGED_CANDIDATE_LABEL": "Merged candidate",
    "UNET_QC_BORDERLINE_LABEL": "QC borderline",
    "SATURN_RECOVERED_LABEL": "Recovered by Saturn",
    "FINAL_ACCEPTED_LABEL": "Final accepted",
    "EXCLUDED_FROM_MEASUREMENT_LABEL": "Excluded from measurement",

    # ------ calibration ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    "UM_PER_PX_XY":   0.756836,
    "UM_PER_SLICE_Z": 1.040460,

    # ------ image enhancement ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    "CLAHE_CLIP":   0.025,
    "CLAHE_KERNEL": 128,
    "BG_SIGMA":     5.065,
    "BG_SIGMA_UM": 6.0,
    "DENOISE_SIGMA_UM": 0.45,
    "RIDGE_SIGMAS": [1, 2, 3, 4],
    "RIDGE_SIGMAS_UM": [0.60, 0.90, 1.20, 1.50],

    # ------ hysteresis threshold ------------------------------------------------------------------------------------------------------------------------------------------------------
    "THRESHOLD_HI": 91.0,
    "THRESHOLD_LO": 83.0,

    # ------ morphological cleanup ---------------------------------------------------------------------------------------------------------------------------------------------------
    "CLOSE_RADIUS":  0,
    "MIN_HOLE_AREA": 0,
    "MIN_OBJ_PX":    8,

    # ------ skeleton-level gap bridging ---------------------------------------------------------------------------------------------------------------------------------
    # P1: reduced from 10 -> 5 to prevent chaining distinct spermatids
    "MAX_BRIDGE_PX": 5,
    "MAX_BRIDGE_UM": 1.5,
    "MAX_BRIDGE_ANGLE_DEG": 35.0,

    # ------ branch pruning & automated splitting ------------------------------------------------------------------------------------------------------
    "MAX_BRANCH_LEN_PX": 5,   # prune spurs shorter than this before measuring
    "MAX_BRANCH_LEN_UM": 2.3,
    "BREAK_JUNCTIONS": True,  # automatically sever all branching intersections into distinct lines

    # ------ optional early mask-level shape filter ------------------------------------------------------------------------------------------------
    "USE_EARLY_SHAPE_FILTER": False,
    "MIN_ECCENTRICITY": 0.60,
    "MAX_MINOR_PX":     12.0,
    "MIN_AXIS_RATIO":   1.4,
    "MIN_MAJOR_PX":     5,

    # ------ post-skeleton filters ---------------------------------------------------------------------------------------------------------------------------------------------------
    "MIN_SKEL_LEN_PX":        6.02, # legacy JSON compatibility
    "MIN_SKEL_LEN_UM":        6.0,
    "MAX_GEODESIC_LEN_PX":    65.0, # backend actively cuts chains longer than this
    "MAX_GEODESIC_LEN_UM":    20.0,
    "MAX_WIDTH_PX":           8.94, # legacy JSON compatibility
    "MAX_WIDTH_UM":           4.2,
    "MIN_LENGTH_WIDTH_RATIO": 2.5,

    # ------ NEW topology filters ------------------------------------------------------------------------------------------------------------------------------------------------------
    "MAX_BRANCH_NODES": 0,        # 30-Gen V2 Optimized
    "MAX_TORTUOSITY": 2.5,
    "MAX_ENDPOINT_COUNT": 4,

    # N4: loops
    "ALLOW_LOOPS": False,
    "AUTO_LOCAL_REANALYSIS": False,
    "ROI_EDGE_QC_DISTANCE_UM": 1.0,

    # ------ tracking across z ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    "DO_TRACKING":          True,
    "TRACKING_BACKEND":     "hybrid_repair",  # legacy, global_assignment, or hybrid_repair
    "TRACK_MAX_DIST_UM":    6.8711,
    "TRACK_MAX_GAP_SLICES": 1,
    "TRACK_BBOX_PADDING_PX": 2,

    # ------ conservative tracking stop-rules ---------------------------------------------------------------------------------------------------------------------------
    "CONSERVATIVE_MAX_WIDTH_JUMP_RATIO": 0.7668,      # Saturn V5 tuned default
    "CONSERVATIVE_MAX_LENGTH_JUMP_RATIO": 0.3134,     # Saturn V5 tuned default
    "CONSERVATIVE_MAX_AREA_JUMP_RATIO": 0.5522,       # Saturn V5 tuned default
    "CONSERVATIVE_MAX_TORTUOSITY_JUMP": 0.40,       # Absolute tortuosity jump
    "CONSERVATIVE_MAX_CENTROID_JUMP_UM": 5.2397,      # Saturn V5 tuned default

    # ------ overlap-first Stage 2 parameters ------------------------------------------------------------------------------------------------------------------------------
    "OVERLAP_STABILITY_THRESHOLD": 0.2026,            # Saturn V5 tuned default
    "OVERLAP_ORIENTATION_DEG":     27.2181,             # Saturn V5 tuned default
    "OVERLAP_MULTIPLIER":          1.6255,             # Saturn V5 tuned default
    "OVERLAP_MIN_STABLE_COUNT":    1,                # Min stable metrics required for overlap continuation
    "ASSIGNMENT_MAX_COST":          8.0,              # global-assignment tracker gate
    "ASSIGNMENT_DIST_WEIGHT":       1.0,
    "ASSIGNMENT_OVERLAP_WEIGHT":    2.0,
    "ASSIGNMENT_LENGTH_WEIGHT":     2.0,
    "ASSIGNMENT_WIDTH_WEIGHT":      1.2,
    "ASSIGNMENT_AREA_WEIGHT":       0.9,
    "ASSIGNMENT_ANGLE_WEIGHT":      0.4,
    "HYBRID_REPAIR_MAX_COST":       3.6,       # v5.5 fragment-repair gate; lower is stricter
    "HYBRID_REPAIR_MAX_GAP_SLICES": 1,
    "HYBRID_REPAIR_MAX_FRAGMENT_SLICES": 2,
    "HYBRID_REPAIR_MAX_LINK_DIST_UM": 4.8,
    "HYBRID_REPAIR_MIN_OVERLAP":    0.05,
    "HYBRID_REPAIR_MAX_FINAL_LENGTH_UM": 15.0,

    # ------ quality audit thresholds (for automated outlier filtering) ---------------------------------
    "AUDIT_MAX_LENGTH_UM":     15.0,    # Flag tracks longer than this (um)
    "AUDIT_MAX_TORTUOSITY":    1.5,     # Flag tracks more tortuous than this
    "AUDIT_MAX_THICKNESS_UM":  2.0,     # Flag tracks thicker than this (um)
    "AUDIT_MAX_TAPER_RATIO":   1.5,     # Flag tracks with taper ratio above this
    "AUDIT_EXTREME_THICKNESS_UM": 3.5,  # Hard-fail only very extreme PSF/merge thickness
    "AUDIT_EXTREME_TAPER_RATIO":  3.0,  # Hard-fail only very extreme taper instability
    "AUDIT_MIN_SLICES":        1,       # single-slice nuclei may be biologically valid at this z-step

    # ------ output / debug ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    "RUN_MODE": str, "ANALYSIS_MODE": str, "SINGLE_IMAGE_SELECTION_MODE": str,
    "SINGLE_TEST_IMAGE": str, "SINGLE_Z_INDEX": int,
    "INPUT_DIR": str, "OUTPUT_DIR": str, "FILE_PATTERN": str, "ROI_MASK_PATH": str,
    "PREPROCESS_MODE": str, "LEGACY_TWO_PASS_ROI": bool,
    "STACK_QC_SAMPLE_COUNT": int, "NORM_LOW_PERCENTILE": (int, float),
    "NORM_HIGH_PERCENTILE": (int, float), "NORM_STACK_WEIGHT": (int, float),
    "CLAHE_MODE": str, "CLAHE_CLIP_HIGH_CONTRAST": (int, float),
    "CLAHE_CLIP_STANDARD": (int, float), "CLAHE_CLIP_LOW_CONTRAST": (int, float),
    "AUTO_CONTRAST_HIGH_THRESHOLD": (int, float), "AUTO_CONTRAST_LOW_THRESHOLD": (int, float),
    "ROI_CROP_PADDING_PX": int, "ROI_THRESHOLD_PERCENTILES_ONLY": bool,
    "EXCLUSION_MASK_PATH": str,
    "ROI_BOUNDARY_SAFE_RIDGE": bool, "ROI_THRESHOLD_EXCLUDE_BOUNDARY_PX": int,
    "AI_PREPROCESSING_MODE": str,
    "SEGMENTATION_ENGINE": str, "UNET_MODEL_PATH": str,
    "UNET_THRESHOLD": (int, float), "UNET_THRESHOLD_MODE": str,
    "UNET_CANDIDATE_THRESHOLD": (int, float), "UNET_SEED_THRESHOLD": (int, float),
    "UNET_CONTEXT_MODE": str, "UNET_INFERENCE_MODE": str,
    "UNET_TILE_SIZE": int, "UNET_TILE_OVERLAP": int, "UNET_ROI_PADDING_PX": int,
    "UNET_STITCH_MODE": str, "UNET_OUTSIDE_ROI_ZERO": bool,
    "UNET_SAVE_PROBABILITY_MAPS": bool, "UNET_CANDIDATE_ACCOUNTING": bool,
    "UNET_RESCUE_ENABLE": bool, "UNET_RESCUE_THRESHOLD": (int, float),
    "UNET_RESCUE_EXCLUDE_DILATION_PX": int, "UNET_RESCUE_MIN_COMPONENT_PX": int,
    "UNET_RESCUE_MAX_ADDITIONS_PER_SLICE": int,
    "UNET_TRACKING_SUPPORT": bool,
    "ASSIGNMENT_UNET_SUPPORT_WEIGHT": (int, float),
    "ASSIGNMENT_UNET_CONTINUITY_WEIGHT": (int, float),
    "HYBRID_REPAIR_UNET_SUPPORT_WEIGHT": (int, float),
    "UNET_DETECTED_LABEL": str, "UNET_RESCUED_LABEL": str,
    "UNET_COMPLETED_BY_BRIDGE_LABEL": str,
    "UNET_COMPLETED_BY_EXTENSION_LABEL": str, "UNET_MERGED_CANDIDATE_LABEL": str,
    "UNET_QC_BORDERLINE_LABEL": str,
    "UM_PER_PX_XY": float, "UM_PER_SLICE_Z": float,
    "CLAHE_CLIP": float, "CLAHE_KERNEL": int, "BG_SIGMA": (int, float),
    "BG_SIGMA_UM": (int, float), "DENOISE_SIGMA_UM": (int, float),
    "RIDGE_SIGMAS": list, "RIDGE_SIGMAS_UM": list,
    "THRESHOLD_HI": (int, float), "THRESHOLD_LO": (int, float),
    "CLOSE_RADIUS": int, "MIN_HOLE_AREA": int, "MIN_OBJ_PX": int,
    "MAX_BRIDGE_PX": (int, float), "MAX_BRIDGE_UM": (int, float),
    "MAX_BRIDGE_ANGLE_DEG": (int, float), "MAX_BRANCH_LEN_PX": (int, float),
    "MAX_BRANCH_LEN_UM": (int, float),
    "USE_EARLY_SHAPE_FILTER": bool,
    "MIN_SKEL_LEN_PX": (int, float), "MIN_SKEL_LEN_UM": (int, float),
    "MAX_GEODESIC_LEN_PX": (int, float), "MAX_GEODESIC_LEN_UM": (int, float),
    "MAX_WIDTH_PX": (int, float), "MAX_WIDTH_UM": (int, float),
    "MIN_LENGTH_WIDTH_RATIO": (int, float),
    "MAX_BRANCH_NODES": (int, float),
    "MAX_TORTUOSITY": (int, float),
    "MAX_ENDPOINT_COUNT": (int, float),
    "AUTO_LOCAL_REANALYSIS": bool, "ROI_EDGE_QC_DISTANCE_UM": (int, float),
    "DO_TRACKING": bool, "TRACKING_BACKEND": str, "TRACK_MAX_DIST_UM": (int, float),
    "TRACK_MAX_GAP_SLICES": int,
    "TRACK_BBOX_PADDING_PX": int,
    "CONSERVATIVE_MAX_WIDTH_JUMP_RATIO": float,
    "CONSERVATIVE_MAX_LENGTH_JUMP_RATIO": float,
    "CONSERVATIVE_MAX_AREA_JUMP_RATIO": float,
    "CONSERVATIVE_MAX_TORTUOSITY_JUMP": float,
    "CONSERVATIVE_MAX_CENTROID_JUMP_UM": float,
    "OVERLAP_STABILITY_THRESHOLD": float,
    "OVERLAP_ORIENTATION_DEG": float,
    "OVERLAP_MULTIPLIER": float,
    "OVERLAP_MIN_STABLE_COUNT": int,
    "ASSIGNMENT_MAX_COST": (int, float),
    "ASSIGNMENT_DIST_WEIGHT": (int, float),
    "ASSIGNMENT_OVERLAP_WEIGHT": (int, float),
    "ASSIGNMENT_LENGTH_WEIGHT": (int, float),
    "ASSIGNMENT_WIDTH_WEIGHT": (int, float),
    "ASSIGNMENT_AREA_WEIGHT": (int, float),
    "ASSIGNMENT_ANGLE_WEIGHT": (int, float),
    "HYBRID_REPAIR_MAX_COST": (int, float),
    "HYBRID_REPAIR_MAX_GAP_SLICES": int,
    "HYBRID_REPAIR_MAX_FRAGMENT_SLICES": int,
    "HYBRID_REPAIR_MAX_LINK_DIST_UM": (int, float),
    "HYBRID_REPAIR_MIN_OVERLAP": (int, float),
    "HYBRID_REPAIR_MAX_FINAL_LENGTH_UM": (int, float),
    "AUDIT_MAX_LENGTH_UM": (int, float),
    "AUDIT_MAX_TORTUOSITY": (int, float),
    "AUDIT_MAX_THICKNESS_UM": (int, float),
    "AUDIT_MAX_TAPER_RATIO": (int, float),
    "AUDIT_EXTREME_THICKNESS_UM": (int, float),
    "AUDIT_EXTREME_TAPER_RATIO": (int, float),
    "AUDIT_MIN_SLICES": int,
    "SAVE_DEBUG_IMAGES": bool, "SAVE_MASK_TIFS": bool,
    "SAVE_LABEL_TIFS": bool, "SAVE_OVERLAYS": bool,
    "SAVE_DETAIL_FIGURE": bool, "SHOW_PREVIEW_WINDOW": bool,
    "SHOW_DEBUG_PREVIEW": bool,
}


def validate_config(cfg):
    """
    Validates the pipeline CONFIG dictionary for required keys, correct types, and logical consistency.

    Raises a descriptive ValueError listing ALL problems found so engineers can fix
    everything in one go rather than playing whack-a-mole with sequential errors.

    Checks performed
    ----------------
    - Every key in ``_REQUIRED`` is present in *cfg*.
    - Each value has the expected Python type (e.g., ``float`` for calibration parameters).
    - ``THRESHOLD_LO`` < ``THRESHOLD_HI`` (hysteresis thresholding only works in this order).
    - ``RUN_MODE`` is exactly one of ``'single'`` or ``'batch'``.
    - The deprecated ``'REJECT_BRANCHES'`` Boolean key is absent (replaced by ``MAX_BRANCH_NODES`` int).

    Args:
        cfg (dict): The configuration mapping to validate.

    Raises:
        ValueError: If *any* of the above checks fail. The message lists all errors.
    """
    errors = []
    for key, expected in _REQUIRED.items():
        if key not in cfg:
            errors.append(f"  MISSING: '{key}'")
        elif not isinstance(cfg[key], expected):
            errors.append(f"  WRONG TYPE '{key}': "
                          f"got {type(cfg[key]).__name__}, want {expected}")
    if cfg.get("THRESHOLD_LO", 0) >= cfg.get("THRESHOLD_HI", 100):
        errors.append("  THRESHOLD_LO must be < THRESHOLD_HI")
    if cfg.get("UM_PER_PX_XY", 0) <= 0 or cfg.get("UM_PER_SLICE_Z", 0) <= 0:
        errors.append("  UM_PER_PX_XY and UM_PER_SLICE_Z must be positive")
    if not (0 <= cfg.get("NORM_LOW_PERCENTILE", -1) < cfg.get("NORM_HIGH_PERCENTILE", -1) <= 100):
        errors.append("  NORM_LOW_PERCENTILE must be < NORM_HIGH_PERCENTILE within [0, 100]")
    if not (0 <= cfg.get("NORM_STACK_WEIGHT", -1) <= 1):
        errors.append("  NORM_STACK_WEIGHT must be between 0 and 1")
    if cfg.get("CLAHE_MODE") not in ("auto_stack", "auto", "no_clahe", "high_contrast", "standard", "low_signal"):
        errors.append("  CLAHE_MODE must be auto_stack/auto/no_clahe/high_contrast/standard/low_signal")
    if cfg.get("PREPROCESS_MODE") not in ("roi_adaptive", "full_frame"):
        errors.append("  PREPROCESS_MODE must be roi_adaptive or full_frame")
    for key in ("CLAHE_CLIP_HIGH_CONTRAST", "CLAHE_CLIP_STANDARD", "CLAHE_CLIP_LOW_CONTRAST",
                "DENOISE_SIGMA_UM", "MAX_BRIDGE_UM", "MIN_SKEL_LEN_UM", "MAX_WIDTH_UM",
                "MAX_GEODESIC_LEN_UM", "MAX_BRANCH_LEN_UM"):
        if cfg.get(key, 0) < 0:
            errors.append(f"  {key} must be nonnegative")
    if cfg.get("ROI_CROP_PADDING_PX", 0) < 0:
        errors.append("  ROI_CROP_PADDING_PX must be nonnegative")
    if cfg.get("BG_SIGMA_UM", 0) <= 0:
        errors.append("  BG_SIGMA_UM must be positive")
    if not cfg.get("RIDGE_SIGMAS_UM") or any(float(v) <= 0 for v in cfg.get("RIDGE_SIGMAS_UM", [])):
        errors.append("  RIDGE_SIGMAS_UM must contain positive values")
    if not (0 <= cfg.get("MAX_BRIDGE_ANGLE_DEG", -1) <= 180):
        errors.append("  MAX_BRIDGE_ANGLE_DEG must be between 0 and 180")
    if cfg.get("MAX_GEODESIC_LEN_UM", 0) <= cfg.get("MIN_SKEL_LEN_UM", 0):
        errors.append("  MAX_GEODESIC_LEN_UM must exceed MIN_SKEL_LEN_UM")
    if cfg.get("RUN_MODE", "") not in ("single", "batch"):
        errors.append("  RUN_MODE must be 'single' or 'batch'")
    if cfg.get("ANALYSIS_MODE", "") not in ("comparative", "reference_morphology", "legacy"):
        errors.append("  ANALYSIS_MODE must be comparative, reference_morphology, or legacy")
    if cfg.get("AI_PREPROCESSING_MODE", "off") != "off":
        errors.append("  AI_PREPROCESSING_MODE must be 'off' unless explicitly enabled by experimental code")
    if "REJECT_BRANCHES" in cfg:
        errors.append("  'REJECT_BRANCHES' was removed in v8. "
                      "Use MAX_BRANCH_NODES (int) instead:\n"
                      "    REJECT_BRANCHES=True  -> MAX_BRANCH_NODES=0\n"
                      "    REJECT_BRANCHES=False -> MAX_BRANCH_NODES=9999")
    if errors:
        raise ValueError("CONFIG errors:\n" + "\n".join(errors))


# =============================================================================
# UTILITIES
# =============================================================================

_N8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def ensure_dir(p):
    """
    Creates directory *p* (and all missing parents) if it does not already exist.

    Uses ``exist_ok=True`` to avoid TOCTOU race conditions in parallel runs.

    Args:
        p (str): Absolute or relative path of the directory to create.
    """
    os.makedirs(p, exist_ok=True)

def get_unique_batch_dir(base_dir):
    """
    Checks for 'batch_output', then 'batch_output_1', 'batch_output_2', etc.
    Returns the first available path.
    """
    candidate = os.path.join(base_dir, "batch_output")
    if not os.path.exists(candidate):
        return candidate

    counter = 1
    while True:
        candidate = os.path.join(base_dir, f"batch_output_{counter}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def natural_sort_key(s):
    """
    Key for natural sorting (e.g., img2.tif comes before img10.tif).
    """
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', os.path.basename(s))]

def extract_z_index(path, sequence_idx=0):
    """
    Extracts Z-index from filename, fallback to sequence index.
    """
    import re
    m = re.search(r"[zZ](\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    return sequence_idx


def ensure_2d_image(img, name="image"):
    """
    Forces an image array to be 2D grayscale.
    Handles squeezing, channel-first (Z,H,W), channel-last (H,W,C), and RGB conversion.
    """
    img = np.asarray(img)
    img = np.squeeze(img)

    if img.ndim == 2:
        return img

    if img.ndim == 3:
        # channel-last RGB/RGBA
        if img.shape[-1] in (3, 4):
            return img[..., 0]
        # channel-first
        if img.shape[0] in (1, 3, 4):
            return img[0]
        # singleton channel
        if img.shape[-1] == 1:
            return img[..., 0]
        if img.shape[0] == 1:
            return img[0]

    raise ValueError(f"{name} must be 2D after loading, got shape {img.shape}")


def robust_imread(path):
    """
    Reads image with multi-engine fallback and forced 2D grayscale enforcement.
    """
    p_lower = path.lower()

    # 1. Primary Loader: tifffile
    if p_lower.endswith(".tif") or p_lower.endswith(".tiff"):
        try:
            img = tifffile.imread(path)
            return ensure_2d_image(img, os.path.basename(path))
        except Exception:
            pass

    # 2. Fallback A: Pillow
    try:
        from PIL import Image
        img = np.array(Image.open(path))
        return ensure_2d_image(img, os.path.basename(path))
    except Exception:
        pass

    # 3. Fallback B: OpenCV
    if _HAVE_CV2:
        try:
            img = _cv2.imread(path, _cv2.IMREAD_UNCHANGED)
            if img is not None:
                return ensure_2d_image(img, os.path.basename(path))
        except Exception:
            pass

    # 4. Fallback C: Matplotlib
    try:
        img = plt.imread(path)
        return ensure_2d_image(img, os.path.basename(path))
    except Exception:
        pass

    raise RuntimeError(f"All image engines failed to read: {os.path.basename(path)}")

def normalize_display(img):
    """
    Contrast-stretches a raw image for colourmap display.

    Uses the 1st-99.5th percentile range rather than min-max to suppress hot
    pixels and dark-corner vignetting artefacts common in fluorescence microscopy.
    The small epsilon (1e-9) prevents division-by-zero on flat images.

    Args:
        img (np.ndarray): Input image of any integer or float dtype.

    Returns:
        np.ndarray: Float32 array clipped to [0, 1] ready for matplotlib display.
    """
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def _imwrite(path, arr_uint8, cmap="gray"):
    """
    Writes a uint8 image to disk using the best available engine.

    Engine priority:
    1. **OpenCV** - fastest; handles large images without matplotlib overhead.
       Converts RGB->BGR before writing (OpenCV internal convention).
    2. **Matplotlib** - fallback when OpenCV is absent; uses ``plt.imsave``
       which respects the *cmap* argument for single-channel saves.

    Args:
        path (str): Destination file path (extension determines format, e.g. ``.png``).
        arr_uint8 (np.ndarray): uint8 image array, shape ``(H, W)`` or ``(H, W, 3)``.
        cmap (str): Matplotlib colourmap name used only for grayscale fallback saves.
    """
    if _HAVE_CV2:
        if arr_uint8.ndim == 2:
            _cv2.imwrite(path, arr_uint8)
        else:
            # OpenCV stores BGR; convert from RGB before writing
            _cv2.imwrite(path, _cv2.cvtColor(arr_uint8, _cv2.COLOR_RGB2BGR))
    else:
        plt.imsave(path, arr_uint8,
                   cmap=(cmap if arr_uint8.ndim == 2 else None),
                   vmin=0, vmax=255)


def save_gray(path, img_float):
    """
    Saves a floating-point single-channel image as an 8-bit grayscale PNG.

    Applies a full min-max stretch (appropriate for debug images where absolute
    intensity is less important than structure visibility).

    Args:
        path (str): Output file path.
        img_float (np.ndarray): Float image of arbitrary range.
    """
    a = img_float.astype(np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    _imwrite(path, (a * 255).astype(np.uint8), cmap="gray")


def save_mask(path, mask_bool):
    """
    Saves a binary boolean mask as a black-and-white 8-bit PNG.

    Pixels where *mask_bool* is ``True`` are written as 255 (white);
    background pixels are 0 (black). This is the standard convention
    for binary mask overlays in ImageJ / FIJI.

    Args:
        path (str): Output file path.
        mask_bool (np.ndarray[bool]): Boolean mask array.
    """
    _imwrite(path, mask_bool.astype(np.uint8) * 255, cmap="gray")


def load_batch_files(input_dir, pattern):
    """
    Discovers and sorts all image files for a batch run.

    Uses a three-tier fallback strategy so the pipeline works even when
    users supply unusual file extensions or mixed case (.TIF vs .tif):

    1. Exact pattern match (e.g. ``Project001_Series002_z*_ch00.tif``).
    2. ``.tif`` -> ``.tiff`` substitution.
    3. Broad glob over all supported extensions (tif, tiff, png, jpg, jpeg)
       in both lower- and upper-case variants.

    After discovery, files are sorted with :func:`natural_sort_key` so that
    ``z2`` comes before ``z10`` (lexicographic sort would reverse this).  A
    Z-index is extracted from each filename or assigned by sequence position
    as a fallback.

    Args:
        input_dir (str): Directory to search for image files.
        pattern (str): Glob pattern relative to *input_dir*.

    Returns:
        tuple[list[str], list[int]]: Sorted file paths and corresponding Z-indices.

    Raises:
        FileNotFoundError: If no matching images are found after all fallbacks.
    """
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        # Fallback 1: .tif -> .tiff extension swap
        files = glob.glob(os.path.join(input_dir, pattern.replace(".tif", ".tiff")))
    if not files:
        # Fallback 2: Broad scan of all supported image formats
        for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
            files.extend(glob.glob(os.path.join(input_dir, ext)))
            files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        files = list(set(files))  # remove cross-extension duplicates

    if not files:
        raise FileNotFoundError(f"No supported image files found in '{input_dir}'")

    # Natural sort preserves correct Z-stack ordering (z2 < z10)
    files = sorted(files, key=natural_sort_key)

    # Extract z-index from filename pattern; fall back to sequence index
    z_idx = [extract_z_index(f, sequence_idx=i) for i, f in enumerate(files)]

    print(f"Found {len(files)} slices: Z = {z_idx}")
    return files, z_idx


def load_roi_mask_file(path, expected_shape=None):
    """Load a boolean ROI mask from .npy or image-mask formats."""
    if not path:
        return None
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ROI mask not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        mask = np.load(path).astype(bool)
    else:
        mask = robust_imread(path).astype(bool)
    if mask.ndim > 2:
        mask = mask[..., 0].astype(bool)
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(f"ROI mask shape {mask.shape} does not match image shape {expected_shape}")
    return mask


def filter_results_to_roi(results, skel_label, roi_mask):
    """
    Keep detections whose centroid lies inside ROI and mask label pixels outside it.

    The detector can still use full-frame image statistics; this function only
    limits which detections are accepted for downstream tracking/reporting.
    """
    if roi_mask is None or not results:
        return results, skel_label
    roi_mask = roi_mask.astype(bool)
    filtered = []
    keep = []
    for r in results:
        cy = min(max(int(round(r["centroid_y"])), 0), roi_mask.shape[0] - 1)
        cx = min(max(int(round(r["centroid_x"])), 0), roi_mask.shape[1] - 1)
        if roi_mask[cy, cx]:
            filtered.append(r)
            keep.append(r["label"])
    skel_roi = np.where(np.isin(skel_label, keep) & roi_mask, skel_label, 0).astype(np.int32)
    return filtered, skel_roi


def choose_single_image(cfg):
    """
    Resolves the path to a single test image according to the configured selection mode.

    Three modes are supported (set via ``SINGLE_IMAGE_SELECTION_MODE``):

    - ``'path'``     - Use the hard-coded ``SINGLE_TEST_IMAGE`` path directly.
    - ``'z_index'``  - Find the file whose filename encodes the Z-index in
                      ``SINGLE_Z_INDEX`` (e.g. ``z15`` -> ``...z15_ch00.tif``).
    - ``'dialog'``   - Open a Tkinter file-picker dialog for interactive selection.
                      Falls back gracefully if Tkinter is unavailable.

    Args:
        cfg (dict): Pipeline configuration dictionary.

    Returns:
        str: Absolute path to the selected image file.

    Raises:
        FileNotFoundError: For ``'path'`` mode when the file does not exist, or
                           ``'z_index'`` mode when no file matches the Z-index.
        RuntimeError: For ``'dialog'`` mode when Tkinter fails or the user cancels.
        ValueError: If ``SINGLE_IMAGE_SELECTION_MODE`` is not one of the three valid values.
    """
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
            root = tk.Tk(); root.withdraw()  # hide the empty root window
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
    """
    Identifies all endpoint pixels (tip pixels) in a binary skeleton image.

    In an 8-connected skeleton, a pixel is an *endpoint* if exactly one of its
    eight neighbours is also part of the skeleton.  These pixels correspond to
    the physical tips of spermatid filaments.  A clean spermatid should have
    exactly 2 endpoints (head and tail end); a bridged fragment pair still has
    2 endpoints after merging.

    Args:
        skel_bool (np.ndarray[bool]): 2D binary skeleton image (True = skeleton pixel).

    Returns:
        list[tuple[int, int]]: List of ``(row, col)`` pixel coordinates of all endpoints.
    """
    H, W = skel_bool.shape
    ys, xs = np.where(skel_bool)
    sk_set = set(zip(ys.tolist(), xs.tolist()))
    return [(r, c) for r, c in sk_set
            if sum(1 for dr, dc in _N8
                   if 0 <= r+dr < H and 0 <= c+dc < W
                   and skel_bool[r+dr, c+dc]) == 1]


def _endpoint_tangent(endpoint, skel_bool, radius=5):
    r, c = endpoint
    ys, xs = np.where(skel_bool)
    if ys.size < 2:
        return None
    d2 = (ys - r) ** 2 + (xs - c) ** 2
    keep = (d2 > 0) & (d2 <= radius ** 2)
    if np.count_nonzero(keep) < 1:
        return None
    idx = int(np.argmin(d2[keep]))
    pts = np.column_stack([ys[keep], xs[keep]])
    nbr = pts[idx]
    v = np.array([float(nbr[0] - r), float(nbr[1] - c)])
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _angle_between(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def bridge_skeleton_endpoints(skel_bool, skel_labeled, max_gap_px, valid_mask=None, max_angle_deg=35.0, return_stats=False):
    """
    Join endpoint pairs from DIFFERENT components within max_gap_px by a
    1-px straight line.  Preserves the original mask so width estimates
    are not inflated.
    """
    stats = {
        "skeleton_pixels_before": int(np.count_nonzero(skel_bool)),
        "skeleton_pixels_after": int(np.count_nonzero(skel_bool)),
        "proposed_bridges": 0,
        "rejected_distance": 0,
        "rejected_orientation": 0,
        "rejected_roi": 0,
        "rejected_exclusion": 0,
        "accepted": 0,
    }
    if max_gap_px <= 0:
        return (skel_bool.copy(), stats) if return_stats else skel_bool.copy()
    H, W   = skel_bool.shape
    out    = skel_bool.copy()
    if valid_mask is None:
        valid_mask = np.ones_like(skel_bool, dtype=bool)
    eps    = find_endpoints(skel_bool)
    if not eps:
        return (out, stats) if return_stats else out
    ep_arr    = np.array(eps, dtype=np.float32)
    ep_labels = skel_labeled[ep_arr[:, 0].astype(int), ep_arr[:, 1].astype(int)]
    pairs     = cKDTree(ep_arr).query_pairs(r=max_gap_px, output_type="ndarray")
    for i, j in pairs:
        stats["proposed_bridges"] += 1
        if ep_labels[i] == ep_labels[j]:
            continue
        r0, c0 = int(eps[i][0]), int(eps[i][1])
        r1, c1 = int(eps[j][0]), int(eps[j][1])
        if math.hypot(r1 - r0, c1 - c0) > max_gap_px:
            stats["rejected_distance"] += 1
            continue
        v0 = _endpoint_tangent((r0, c0), skel_bool)
        v1 = _endpoint_tangent((r1, c1), skel_bool)
        bridge_vec = np.array([float(r1 - r0), float(c1 - c0)])
        bridge_vec /= (np.linalg.norm(bridge_vec) + 1e-9)
        angle0 = _angle_between(v0, bridge_vec)
        angle1 = _angle_between(v1, -bridge_vec)
        if max(angle0, angle1) > max_angle_deg:
            stats["rejected_orientation"] += 1
            continue
        n = max(abs(r1-r0), abs(c1-c0)) + 1
        rs = np.clip(np.round(np.linspace(r0, r1, n)).astype(int), 0, H-1)
        cs = np.clip(np.round(np.linspace(c0, c1, n)).astype(int), 0, W-1)
        if not np.all(valid_mask[rs, cs]):
            stats["rejected_roi"] += 1
            continue
        out[rs, cs] = True
        stats["accepted"] += 1
    out &= valid_mask
    stats["skeleton_pixels_after"] = int(np.count_nonzero(out))
    print(f"    bridge stats: {stats}")
    return (out, stats) if return_stats else out


def prune_branches(skel_bool, max_branch_len):
    """
    Iteratively remove endpoints to shorten side-branches <= max_branch_len px.
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
    """
    Builds a lightweight adjacency list for a set of skeleton pixel coordinates.

    Uses a *linearised index* trick: each pixel ``(r, c)`` is mapped to an integer
    ``r * W + c`` so that neighbour lookup is an O(1) dictionary lookup rather than
    a 2D array access.  This is important for large skeleton components.

    Edge weights follow the Chebyshev / Euclidean convention:
    - Axis-aligned move (4-connected step): weight = 1.0
    - Diagonal move (8-connected step):     weight = sqrt2 ~ 1.41421

    Args:
        coords (np.ndarray): Shape ``(N, 2)`` integer array of ``(row, col)`` pixel positions.
        W (int): Image width in pixels (used to compute the linear index).

    Returns:
        list[list[tuple[int, float]]]: Adjacency list where ``adj[i]`` is a list of
        ``(neighbour_index, edge_weight)`` pairs.
    """
    n       = len(coords)
    # Linearise (row, col) -> single int for O(1) membership test
    lin     = coords[:, 0] * W + coords[:, 1]
    lin2idx = {int(v): i for i, v in enumerate(lin.tolist())}
    lin_set = set(lin.tolist())
    adj     = [[] for _ in range(n)]
    for i, (r, c) in enumerate(coords.tolist()):
        for dr, dc in _N8:
            lk = (r + dr) * W + (c + dc)
            if lk in lin_set:
                # Diagonal edges are longer by a factor of sqrt2
                w = 1.41421356 if (dr != 0 and dc != 0) else 1.0
                adj[i].append((lin2idx[lk], w))
    return adj


def _dijkstra(adj, src, n):
    """
    Runs Dijkstra's shortest-path algorithm from source node *src* and returns
    the *farthest* reachable node and its distance.

    This is the first BFS pass in the double-BFS algorithm for computing the
    true *geodesic diameter* (longest shortest path) of a skeleton component.
    The second BFS starts from the farthest node found here.

    Mathematical note:
        Geodesic length = shortest path distance through the skeleton graph,
        which accounts for the actual curvature of the filament rather than
        the straight-line Euclidean distance between endpoints.

    Args:
        adj (list[list[tuple[int, float]]]): Adjacency list from :func:`_build_adj`.
        src (int): Index of the source node in *adj*.
        n (int): Total number of nodes.

    Returns:
        tuple[int, float]: ``(farthest_node_index, distance_to_farthest)``
    """
    d = np.full(n, np.inf); d[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        cost, u = heapq.heappop(pq)
        if cost > d[u]:
            continue  # stale entry in the heap - skip
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

    # ------ Loop handling ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

    # ------ Double-BFS for geodesic ---------------------------------------------------------------------------------------------------------------------------------------------
    b, _  = _dijkstra(adj, eps_idx[0], n)
    c, gl = _dijkstra(adj, b, n)

    # ------ Tortuosity ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    """
    Optionally removes non-elongated connected components from the binary mask
    *before* skeletonisation, based on region shape descriptors.

    This is an early-stage pre-filter controlled by ``USE_EARLY_SHAPE_FILTER``.
    When enabled, it eliminates round debris, fat nuclei, and large globular
    artefacts that would otherwise generate spurious skeleton trees downstream.

    Biological rationale
    --------------------
    Round debris (dead cells, lipid droplets, imaging artefacts) tends to have:
    - Low eccentricity (close to circular)
    - Short major axis (small, compact object)
    - Low axis ratio (major ~ minor, i.e., not elongated)

    Elongated spermatid nuclei, by contrast, have high eccentricity (> 0.6),
    a long major axis, and a large major/minor axis ratio.

    Args:
        mask (np.ndarray[bool]): Binary mask from hysteresis thresholding.
        cfg (dict): Pipeline config; reads ``USE_EARLY_SHAPE_FILTER``,
            ``MIN_ECCENTRICITY``, ``MAX_MINOR_PX``, ``MIN_AXIS_RATIO``,
            and ``MIN_MAJOR_PX``.

    Returns:
        np.ndarray[bool]: Filtered binary mask (unchanged if the filter is disabled).
    """
    if not cfg["USE_EARLY_SHAPE_FILTER"]:
        return mask  # filter disabled - pass through unchanged
    labeled = measure.label(mask)
    keep = [p.label for p in measure.regionprops(labeled)
            if (p.eccentricity      >= cfg["MIN_ECCENTRICITY"] and   # must be elongated
                p.minor_axis_length <= cfg["MAX_MINOR_PX"]     and   # must be narrow
                p.major_axis_length / (p.minor_axis_length + 1e-9) >= cfg["MIN_AXIS_RATIO"] and  # must be rod-like
                p.major_axis_length >= cfg["MIN_MAJOR_PX"])]
    return np.isin(labeled, keep)


def remove_objects_smaller_than(mask, min_size):
    """Compatibility wrapper preserving old remove_small_objects(min_size=...) semantics."""
    min_size = int(min_size)
    if min_size <= 1:
        return mask
    return morphology.remove_small_objects(mask, max_size=min_size - 1)


@dataclass
class StackPreprocessContext:
    normalization_low: float
    normalization_high: float
    selected_clahe_clip: float
    selected_clahe_profile: str
    contrast_score: float
    sampled_z_indices: list
    roi_percentiles: dict
    saturation_fraction: float
    slice_brightness_statistics: list
    source_dtype: str
    inferred_bit_depth: int
    resolved_pixel_parameters: dict
    configuration_provenance: dict
    image_shape: tuple
    roi_pixel_count: int
    excluded_pixel_count: int


def _json_scalar(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_json_scalar(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _json_scalar(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_scalar(x) for x in v]
    return v


def resolve_pixel_parameters(cfg):
    um = float(cfg.get("UM_PER_PX_XY", 1.0))
    if um <= 0:
        raise ValueError("UM_PER_PX_XY must be positive for physical-unit resolution")

    def length_px(um_key, px_key, default=0.0):
        if um_key in cfg and cfg.get(um_key) is not None:
            return float(cfg[um_key]) / um, float(cfg[um_key]), "physical"
        return float(cfg.get(px_key, default)), float(cfg.get(px_key, default)) * um, "legacy_px"

    min_skel_px, min_skel_um, min_skel_src = length_px("MIN_SKEL_LEN_UM", "MIN_SKEL_LEN_PX")
    max_geo_px, max_geo_um, max_geo_src = length_px("MAX_GEODESIC_LEN_UM", "MAX_GEODESIC_LEN_PX")
    max_width_px, max_width_um, max_width_src = length_px("MAX_WIDTH_UM", "MAX_WIDTH_PX")
    max_bridge_px_f, max_bridge_um, max_bridge_src = length_px("MAX_BRIDGE_UM", "MAX_BRIDGE_PX")
    max_branch_px_f, max_branch_um, max_branch_src = length_px("MAX_BRANCH_LEN_UM", "MAX_BRANCH_LEN_PX")
    bg_px, bg_um, bg_src = length_px("BG_SIGMA_UM", "BG_SIGMA")
    denoise_px, denoise_um, denoise_src = length_px("DENOISE_SIGMA_UM", "DENOISE_SIGMA_PX", 0.0)
    if "RIDGE_SIGMAS_UM" in cfg:
        ridge_um = [float(v) for v in cfg["RIDGE_SIGMAS_UM"]]
        ridge_px = [v / um for v in ridge_um]
        ridge_src = "physical"
    else:
        ridge_px = [float(v) for v in cfg.get("RIDGE_SIGMAS", [1, 2])]
        ridge_um = [v * um for v in ridge_px]
        ridge_src = "legacy_px"

    return {
        "physical": {
            "MIN_SKEL_LEN_UM": min_skel_um,
            "MAX_GEODESIC_LEN_UM": max_geo_um,
            "MAX_WIDTH_UM": max_width_um,
            "MAX_BRIDGE_UM": max_bridge_um,
            "MAX_BRANCH_LEN_UM": max_branch_um,
            "BG_SIGMA_UM": bg_um,
            "DENOISE_SIGMA_UM": denoise_um,
            "RIDGE_SIGMAS_UM": ridge_um,
        },
        "pixels": {
            "MIN_SKEL_LEN_PX": min_skel_px,
            "MAX_GEODESIC_LEN_PX": max_geo_px,
            "MAX_WIDTH_PX": max_width_px,
            "MAX_BRIDGE_PX": max(0, int(round(max_bridge_px_f))),
            "MAX_BRANCH_LEN_PX": max(0, int(round(max_branch_px_f))),
            "BG_SIGMA": bg_px,
            "DENOISE_SIGMA": denoise_px,
            "RIDGE_SIGMAS": ridge_px,
        },
        "source": {
            "MIN_SKEL_LEN": min_skel_src,
            "MAX_GEODESIC_LEN": max_geo_src,
            "MAX_WIDTH": max_width_src,
            "MAX_BRIDGE": max_bridge_src,
            "MAX_BRANCH_LEN": max_branch_src,
            "BG_SIGMA": bg_src,
            "DENOISE_SIGMA": denoise_src,
            "RIDGE_SIGMAS": ridge_src,
        },
    }


def cfg_with_resolved_pixels(cfg):
    out = cfg.copy()
    resolved = resolve_pixel_parameters(out)
    out.update(resolved["pixels"])
    out["_RESOLVED_PIXEL_PARAMETERS"] = resolved
    return out


def infer_bit_depth(arr):
    if np.issubdtype(arr.dtype, np.integer):
        return int(np.iinfo(arr.dtype).bits)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 32
    mx = float(np.nanmax(finite))
    if mx <= 1.0:
        return 1
    if mx <= 255:
        return 8
    if mx <= 65535:
        return 16
    return 32


def representative_indices(n, count):
    if n <= 0:
        return []
    count = max(1, min(int(count), n))
    return sorted(set(int(round(v)) for v in np.linspace(0, n - 1, count)))


def _clahe_profile_from_score(score, cfg):
    mode = cfg.get("CLAHE_MODE", "auto_stack")
    if mode in ("no_clahe", "high_contrast", "standard", "low_signal"):
        profile = mode
    elif score >= float(cfg.get("AUTO_CONTRAST_HIGH_THRESHOLD", 0.45)):
        profile = "no_clahe"
    elif score >= float(cfg.get("AUTO_CONTRAST_LOW_THRESHOLD", 0.25)):
        profile = "standard"
    else:
        profile = "low_signal"
    clip = {
        "no_clahe": 0.0,
        "high_contrast": float(cfg.get("CLAHE_CLIP_HIGH_CONTRAST", 0.010)),
        "standard": float(cfg.get("CLAHE_CLIP_STANDARD", 0.025)),
        "low_signal": float(cfg.get("CLAHE_CLIP_LOW_CONTRAST", 0.035)),
    }[profile]
    return profile, clip


def build_stack_preprocess_context(image_files, roi_mask, cfg, exclusion_mask=None):
    if not image_files:
        raise ValueError("build_stack_preprocess_context requires at least one image file")
    sample_positions = representative_indices(len(image_files), cfg.get("STACK_QC_SAMPLE_COUNT", 12))
    pooled, slice_stats = [], []
    image_shape = None
    source_dtype = None
    bit_depth = None
    saturated = 0
    total_valid = 0
    warnings_out = []

    for pos in sample_positions:
        arr = ensure_2d_image(robust_imread(image_files[pos]), os.path.basename(image_files[pos]))
        if image_shape is None:
            image_shape = arr.shape
            source_dtype = str(arr.dtype)
            bit_depth = infer_bit_depth(arr)
        elif arr.shape != image_shape:
            raise ValueError(f"Inconsistent sampled image dimensions: {arr.shape} vs {image_shape}")

        if roi_mask is None:
            valid = np.ones(arr.shape, dtype=bool)
            warnings_out.append("missing ROI: stack QC used full frame")
        else:
            if roi_mask.shape != arr.shape:
                raise ValueError(f"ROI shape {roi_mask.shape} does not match image shape {arr.shape}")
            valid = roi_mask.astype(bool).copy()
        if exclusion_mask is not None:
            if exclusion_mask.shape != arr.shape:
                raise ValueError(f"Exclusion mask shape {exclusion_mask.shape} does not match image shape {arr.shape}")
            valid &= ~exclusion_mask.astype(bool)

        vals = arr.astype(np.float64)[valid]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            warnings_out.append(f"empty valid pixels at sampled z position {pos}")
            continue
        pooled.append(vals)
        p95 = float(np.percentile(vals, 95))
        med = float(np.median(vals))
        slice_stats.append({"z_index": int(extract_z_index(image_files[pos])), "median": med, "p95": p95})
        total_valid += int(vals.size)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            saturated += int(np.count_nonzero(vals >= info.max))
        else:
            saturated += int(np.count_nonzero(vals >= 1.0)) if float(np.nanmax(vals)) <= 1.5 else int(np.count_nonzero(vals >= np.nanpercentile(vals, 99.99)))

    if not pooled:
        raise ValueError("No finite stack QC pixels inside ROI after exclusion")
    pooled_vals = np.concatenate(pooled)
    pct = {f"p{p:g}": float(np.percentile(pooled_vals, p)) for p in (1, 20, 50, 95, 99.5)}
    low = float(np.percentile(pooled_vals, cfg.get("NORM_LOW_PERCENTILE", 1.0)))
    high = float(np.percentile(pooled_vals, cfg.get("NORM_HIGH_PERCENTILE", 99.5)))
    dyn = max(high - low, 1e-9)
    contrast_score = float((pct["p95"] - pct["p20"]) / dyn)
    profile, clip = _clahe_profile_from_score(contrast_score, cfg)
    medians = np.array([s["median"] for s in slice_stats], dtype=float)
    p95s = np.array([s["p95"] for s in slice_stats], dtype=float)
    cv = float(np.std(medians) / (np.mean(medians) + 1e-9)) if medians.size else 0.0
    if dyn < 1e-6:
        warnings_out.append("very low dynamic range in valid ROI pixels")
    sat_frac = float(saturated / max(total_valid, 1))
    if sat_frac > 0.01:
        warnings_out.append(f"excessive saturation fraction {sat_frac:.4f}")

    return StackPreprocessContext(
        normalization_low=low,
        normalization_high=high,
        selected_clahe_clip=clip,
        selected_clahe_profile=profile,
        contrast_score=contrast_score,
        sampled_z_indices=[int(extract_z_index(image_files[pos])) for pos in sample_positions],
        roi_percentiles=pct,
        saturation_fraction=sat_frac,
        slice_brightness_statistics=[dict(s, brightness_cv=cv, p95_mean=float(np.mean(p95s))) for s in slice_stats],
        source_dtype=source_dtype or "unknown",
        inferred_bit_depth=int(bit_depth or 0),
        resolved_pixel_parameters=resolve_pixel_parameters(cfg),
        configuration_provenance={
            "version": "v5.7-unet-ready",
            "preprocess_mode": cfg.get("PREPROCESS_MODE"),
            "legacy_two_pass_roi": bool(cfg.get("LEGACY_TWO_PASS_ROI", False)),
            "warnings": warnings_out,
        },
        image_shape=tuple(int(v) for v in image_shape),
        roi_pixel_count=int(np.count_nonzero(roi_mask)) if roi_mask is not None else int(np.prod(image_shape)),
        excluded_pixel_count=int(np.count_nonzero(exclusion_mask)) if exclusion_mask is not None else 0,
    )


def save_stack_preprocess_context(context, output_dir):
    ensure_dir(output_dir)
    data = _json_scalar(asdict(context))
    with open(os.path.join(output_dir, "stack_preprocessing_qc.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    flat = data.copy()
    flat["sampled_z_indices"] = ",".join(str(v) for v in data.get("sampled_z_indices", []))
    flat["configuration_warnings"] = "; ".join(data.get("configuration_provenance", {}).get("warnings", []))
    pd.DataFrame([flat]).to_csv(os.path.join(output_dir, "stack_preprocessing_qc.csv"), index=False)


def _full_like(shape, crop, bbox, dtype=float):
    out = np.zeros(shape, dtype=dtype)
    y0, y1, x0, x1 = bbox
    out[y0:y1, x0:x1] = crop
    return out


def _valid_bbox(mask, pad, shape):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0, shape[0], 0, shape[1])
    return (
        max(0, int(ys.min()) - int(pad)),
        min(shape[0], int(ys.max()) + int(pad) + 1),
        max(0, int(xs.min()) - int(pad)),
        min(shape[1], int(xs.max()) + int(pad) + 1),
    )


def _normalize_with_context(img, valid_mask, cfg, preprocess_context):
    valid_vals = img[valid_mask]
    valid_vals = valid_vals[np.isfinite(valid_vals)]
    if valid_vals.size == 0:
        raise ValueError("No finite valid pixels available for slice normalization")
    slice_low = float(np.percentile(valid_vals, cfg.get("NORM_LOW_PERCENTILE", 1.0)))
    slice_high = float(np.percentile(valid_vals, cfg.get("NORM_HIGH_PERCENTILE", 99.5)))
    if preprocess_context is None:
        stack_low, stack_high = slice_low, slice_high
        profile, clip = _clahe_profile_from_score((slice_high - slice_low) / (slice_high + 1e-9), cfg)
    else:
        stack_low = float(preprocess_context.normalization_low)
        stack_high = float(preprocess_context.normalization_high)
        profile = preprocess_context.selected_clahe_profile
        clip = float(preprocess_context.selected_clahe_clip)
    w = float(cfg.get("NORM_STACK_WEIGHT", 0.80))
    blended_low = w * stack_low + (1.0 - w) * slice_low
    blended_high = w * stack_high + (1.0 - w) * slice_high
    if blended_high <= blended_low:
        blended_high = blended_low + 1e-6
    norm = np.clip((img - blended_low) / (blended_high - blended_low), 0, 1)
    return norm, {
        "stack_low": stack_low, "stack_high": stack_high,
        "slice_low": slice_low, "slice_high": slice_high,
        "blended_low": float(blended_low), "blended_high": float(blended_high),
        "profile": profile, "clip": clip,
    }


def _apply_clahe(img_norm, profile, clip, cfg):
    if profile == "no_clahe" or clip <= 0:
        return img_norm.copy()
    return exposure.equalize_adapthist(
        np.clip(img_norm, 0, 1),
        clip_limit=float(clip),
        kernel_size=int(cfg.get("CLAHE_KERNEL", 128)),
    )


def _save_v56_debug(debug_dir, z_idx, stages, debug_record):
    ensure_dir(debug_dir)
    names = [
        ("01_raw_robust_normalized", stages.get("img_norm")),
        ("02_denoised", stages.get("img_denoised")),
        ("03_clahe", stages.get("img_eq")),
        ("04_background", stages.get("background")),
        ("05_foreground", stages.get("foreground")),
        ("06_ridge", stages.get("ridge")),
        ("07_hysteresis", stages.get("mask_hyst")),
        ("08_clean", stages.get("mask_clean")),
        ("09_skeleton_clean", stages.get("skel_clean")),
        ("10_skeleton_bridged", stages.get("skel_bridged")),
        ("11_skeleton_pruned", stages.get("skel_pruned")),
        ("12_final_detections", stages.get("skel_labeled")),
        ("13_unet_probability", stages.get("unet_probability")),
        ("14_unet_candidate_mask", stages.get("unet_candidate_mask")),
        ("15_unet_seed_mask", stages.get("unet_seed_mask")),
    ]
    for name, arr in names:
        path = os.path.join(debug_dir, f"z{int(z_idx or 0):02d}_{name}.png")
        if arr is None:
            continue
        if arr.dtype == bool:
            save_mask(path, arr)
        else:
            save_gray(path, arr)
    with open(os.path.join(debug_dir, f"z{int(z_idx or 0):02d}_debug_record.json"), "w", encoding="utf-8") as f:
        json.dump(_json_scalar(debug_record), f, indent=2)


def _make_unet_context_from_paths(files_by_z, z_idx):
    """Build [z-1, z, z+1] context planes, clamping to the nearest available slice."""
    if not files_by_z:
        return None
    z_keys = sorted(int(z) for z in files_by_z)
    if not z_keys:
        return None
    z_min, z_max = z_keys[0], z_keys[-1]
    planes = []
    for zz in (int(z_idx) - 1, int(z_idx), int(z_idx) + 1):
        zz = min(max(zz, z_min), z_max)
        if zz not in files_by_z:
            nearest = min(z_keys, key=lambda k: abs(k - zz))
            zz = nearest
        arr = robust_imread(files_by_z[zz])
        planes.append(ensure_2d_image(arr, os.path.basename(files_by_z[zz])).astype(np.float32))
    return np.stack(planes, axis=0)


def _apply_unet_candidate_support(mask_hyst, ridge, valid_crop, full_shape, bbox, roi_mask_full, cfg, unet_context_stack, z_idx=None):
    engine = str(cfg.get("SEGMENTATION_ENGINE", "classical_saturn")).strip().lower()
    model_path = str(cfg.get("UNET_MODEL_PATH", "")).strip()
    empty = np.zeros(full_shape, dtype=np.float32)
    if engine not in {"unet_assisted", "hybrid"} or not model_path or unet_context_stack is None:
        return mask_hyst, ridge, empty, empty.astype(bool), empty.astype(bool), {
            "unet_enabled": False,
            "unet_reason": "disabled_or_missing_context",
        }

    try:
        from utils.saturn_unet25d_bridge import predict_probability_tiled

        unet_prob = predict_probability_tiled(
            unet_context_stack,
            model_path,
            roi_mask=roi_mask_full,
            cfg=cfg,
        )
        y0, y1, x0, x1 = bbox
        prob_crop = unet_prob[y0:y1, x0:x1].astype(np.float32)
        cand_thr = float(cfg.get("UNET_CANDIDATE_THRESHOLD", cfg.get("UNET_THRESHOLD", 0.05)))
        seed_thr = float(cfg.get("UNET_SEED_THRESHOLD", max(cand_thr, 0.30)))
        threshold_mode = str(cfg.get("UNET_THRESHOLD_MODE", "soft")).strip().lower()
        candidate_crop = (prob_crop >= cand_thr) & valid_crop
        seed_crop = (prob_crop >= seed_thr) & valid_crop
        mask_action = "none"
        if engine == "unet_assisted":
            mask_hyst = candidate_crop.copy()
            ridge = prob_crop.copy()
            ridge[~valid_crop] = 0.0
            mask_action = "replace_with_unet_candidate"
        elif engine == "hybrid" and threshold_mode in {"hard", "candidate_union", "union"}:
            mask_hyst = (mask_hyst | candidate_crop) & valid_crop
            mask_action = "union_unet_candidate"

        full_prob = unet_prob.astype(np.float32)
        full_candidate = np.zeros(full_shape, dtype=bool)
        full_seed = np.zeros(full_shape, dtype=bool)
        full_candidate[y0:y1, x0:x1] = candidate_crop
        full_seed[y0:y1, x0:x1] = seed_crop
        return mask_hyst, ridge, full_prob, full_candidate, full_seed, {
            "unet_enabled": True,
            "unet_engine": engine,
            "unet_threshold_mode": threshold_mode,
            "unet_mask_action": mask_action,
            "unet_candidate_threshold": cand_thr,
            "unet_seed_threshold": seed_thr,
            "unet_candidate_pixels": int(np.count_nonzero(full_candidate)),
            "unet_seed_pixels": int(np.count_nonzero(full_seed)),
            "unet_probability_mean_inside_roi": float(np.mean(full_prob[roi_mask_full])) if np.any(roi_mask_full) else 0.0,
        }
    except Exception as exc:
        if bool(cfg.get("UNET_FAIL_HARD", False)):
            raise
        print(f"  WARNING: U-Net inference failed for z={z_idx}: {exc}")
        return mask_hyst, ridge, empty, empty.astype(bool), empty.astype(bool), {
            "unet_enabled": False,
            "unet_reason": f"error: {exc}",
        }


def segment_slice(img_raw, cfg, z_idx=None, debug_dir=None, roi_mask=None, preprocess_context=None, exclusion_mask=None, unet_context_stack=None):
    """
    Executes advanced 2D multi-stage morphology segmentation to detect and isolate spermatid nuclei.

    Args:
        img_raw (np.ndarray): The raw 2D grayscale image array of the z-slice.
        cfg (dict): Pipeline hyperparameters including thresholding, CLAHE limits, and morphological radius.
        z_idx (int, optional): The current Z-index integer for debugging context.
        debug_dir (str, optional): Directory to save intermediate thresholding outputs if tracking is on.
        roi_mask (np.ndarray, optional): A boolean boolean layer representing the user's manual bounding box.

    Returns:
        np.ndarray: A labeled discrete array of contiguous pixel bodies corresponding to valid single nuclei.
    """
    cfg = cfg_with_resolved_pixels(cfg)
    img = ensure_2d_image(img_raw, f"segment_slice z={z_idx}").astype(np.float32)
    full_shape = img.shape
    if roi_mask is None:
        roi_mask_full = np.ones(full_shape, dtype=bool)
    else:
        roi_mask_full = roi_mask.astype(bool)
        if roi_mask_full.shape != full_shape:
            raise ValueError(f"roi_mask shape {roi_mask_full.shape} does not match image shape {full_shape}")
    if exclusion_mask is None:
        exclusion_full = np.zeros(full_shape, dtype=bool)
    else:
        exclusion_full = exclusion_mask.astype(bool)
        if exclusion_full.shape != full_shape:
            raise ValueError(f"exclusion_mask shape {exclusion_full.shape} does not match image shape {full_shape}")
    valid_full = roi_mask_full & ~exclusion_full
    if not np.any(valid_full):
        raise ValueError("ROI/exclusion combination leaves no valid pixels")

    bbox = _valid_bbox(roi_mask_full, cfg.get("ROI_CROP_PADDING_PX", 16), full_shape)
    y0, y1, x0, x1 = bbox
    img_crop = img[y0:y1, x0:x1].copy()
    roi_crop = roi_mask_full[y0:y1, x0:x1]
    excl_crop = exclusion_full[y0:y1, x0:x1]
    valid_crop = roi_crop & ~excl_crop

    finite_valid = img_crop[valid_crop]
    finite_valid = finite_valid[np.isfinite(finite_valid)]
    fill_value = float(np.median(finite_valid)) if finite_valid.size else float(np.nanmedian(img_crop))
    work = img_crop.copy()
    work[~roi_crop] = fill_value
    work[excl_crop] = fill_value

    img_norm, norm_record = _normalize_with_context(work, valid_crop, cfg, preprocess_context)
    boundary_safe = bool(cfg.get("ROI_BOUNDARY_SAFE_RIDGE", True))
    norm_fill = float(np.median(img_norm[valid_crop])) if np.any(valid_crop) else 0.0
    if boundary_safe:
        # Keep a continuous exterior through denoising/background/ridge filtering.
        # The exact biological ROI is applied after ridge calculation.
        img_norm[~roi_crop | excl_crop] = norm_fill
    else:
        img_norm[~roi_crop | excl_crop] = 0

    denoise_sigma = float(cfg.get("DENOISE_SIGMA", 0.0))
    img_denoised = gaussian(img_norm, sigma=denoise_sigma) if denoise_sigma > 0 else img_norm.copy()
    if not boundary_safe:
        img_denoised[~roi_crop | excl_crop] = 0
    img_eq = _apply_clahe(img_denoised, norm_record["profile"], norm_record["clip"], cfg)
    if boundary_safe:
        eq_fill = float(np.median(img_eq[valid_crop])) if np.any(valid_crop) else 0.0
        img_eq[~roi_crop | excl_crop] = eq_fill
    else:
        img_eq[~roi_crop | excl_crop] = 0

    bg  = gaussian(img_eq, sigma=float(cfg["BG_SIGMA"]))
    fg  = np.clip(img_eq - bg, 0, None)
    valid_fg = fg[valid_crop]
    hi_fg = float(np.percentile(valid_fg, 99.5)) if valid_fg.size else 1.0
    fgn = np.clip(fg / (hi_fg + 1e-9), 0, 1)
    if boundary_safe:
        fgn_fill = float(np.median(fgn[valid_crop])) if np.any(valid_crop) else 0.0
        fgn[~roi_crop | excl_crop] = fgn_fill
    else:
        fg[~roi_crop | excl_crop] = 0
        fgn[~roi_crop | excl_crop] = 0

    ridge = meijering(fgn, sigmas=cfg["RIDGE_SIGMAS"], black_ridges=False)
    ridge[~roi_crop | excl_crop] = 0

    threshold_crop = valid_crop
    if boundary_safe:
        boundary_exclude_px = max(0, int(cfg.get("ROI_THRESHOLD_EXCLUDE_BOUNDARY_PX", 4)))
        if boundary_exclude_px > 0:
            interior = distance_transform_edt(valid_crop) > boundary_exclude_px
            if np.count_nonzero(interior) > 100:
                threshold_crop = interior
    ridge_valid = ridge[threshold_crop]
    ridge_valid = ridge_valid[np.isfinite(ridge_valid)]
    if ridge_valid.size == 0:
        th_hi, th_lo = 1.0, 0.5
    else:
        if cfg["THRESHOLD_LO"] >= cfg["THRESHOLD_HI"]:
            raise ValueError("THRESHOLD_LO must be < THRESHOLD_HI")
        th_hi = float(np.percentile(ridge_valid, cfg["THRESHOLD_HI"]))
        th_lo = float(np.percentile(ridge_valid, cfg["THRESHOLD_LO"]))
    norm_record["threshold_hi"] = th_hi
    norm_record["threshold_lo"] = th_lo
    mask_hyst = apply_hysteresis_threshold(ridge, th_lo, th_hi)
    mask_hyst &= valid_crop
    mask_hyst, ridge, unet_prob, unet_candidate, unet_seed, unet_record = _apply_unet_candidate_support(
        mask_hyst,
        ridge,
        valid_crop,
        full_shape,
        bbox,
        roi_mask_full,
        cfg,
        unet_context_stack,
        z_idx=z_idx,
    )

    if mask_hyst.ndim != 2:
        raise ValueError(f"mask_hyst must be 2D, got shape {mask_hyst.shape}")
    mask_clean = mask_hyst.copy()
    if int(cfg["CLOSE_RADIUS"]) > 0:
        mask_clean = morphology.binary_closing(mask_clean, morphology.disk(int(cfg["CLOSE_RADIUS"])))
        mask_clean &= valid_crop
    if int(cfg["MIN_HOLE_AREA"]) > 0:
        mask_clean = morphology.remove_small_holes(mask_clean, area_threshold=int(cfg["MIN_HOLE_AREA"]))
        mask_clean &= valid_crop
    mask_clean = remove_objects_smaller_than(mask_clean, cfg["MIN_OBJ_PX"])
    mask_clean = apply_optional_early_shape_filter(mask_clean, cfg)
    mask_clean &= valid_crop

    # Width is measured from the CLEAN (un-bridged) distance map
    dist_clean   = distance_transform_edt(mask_clean)
    skel_clean   = skeletonize(mask_clean)
    skel_clean &= valid_crop
    skel_labeled = measure.label(skel_clean)

    # Skeleton-level bridging (preserves mask / width integrity)
    skel_bridged, bridge_stats = bridge_skeleton_endpoints(
        skel_clean, skel_labeled, cfg["MAX_BRIDGE_PX"],
        valid_mask=valid_crop,
        max_angle_deg=float(cfg.get("MAX_BRIDGE_ANGLE_DEG", 35.0)),
        return_stats=True)
    # Branch pruning before measurement
    skel_pruned     = prune_branches(skel_bridged, cfg["MAX_BRANCH_LEN_PX"])
    skel_pruned &= valid_crop

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
        skel_pruned &= valid_crop

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
        skel_pruned &= valid_crop
        skel_labeled_fn = measure.label(skel_pruned)

    full = lambda crop, dtype=None: _full_like(full_shape, crop, bbox, dtype=(dtype or crop.dtype))
    out = {
        "mask_hyst":    full(mask_hyst, bool),
        "mask_clean":   full(mask_clean, bool),
        "skel_clean":   full(skel_clean, bool),
        "skel_bridged": full(skel_bridged, bool),
        "skel_pruned":  full(skel_pruned, bool),
        "skel_labeled": full(skel_labeled_fn.astype(np.int32), np.int32),
        "dist_clean":   full(dist_clean, np.float32),
        "img_norm":     full(img_norm, np.float32),
        "img_denoised": full(img_denoised, np.float32),
        "img_eq":       full(img_eq, np.float32),
        "background":   full(bg, np.float32),
        "foreground":   full(fgn, np.float32),
        "ridge":        full(ridge, np.float32),
        "roi_mask":     roi_mask_full,
        "exclusion_mask": exclusion_full,
        "unet_probability": unet_prob,
        "unet_candidate_mask": unet_candidate,
        "unet_seed_mask": unet_seed,
        "unet_debug": unet_record,
        "preprocess_context": preprocess_context,
        "preprocess_debug": norm_record,
        "bridge_stats": bridge_stats,
        "bbox": bbox,
    }

    if cfg["SAVE_DEBUG_IMAGES"] and debug_dir and z_idx is not None:
        valid_full_after = roi_mask_full & ~exclusion_full
        debug_record = {
            "z_index": z_idx,
            "stack_normalization_low": norm_record["stack_low"],
            "stack_normalization_high": norm_record["stack_high"],
            "slice_normalization_low": norm_record["slice_low"],
            "slice_normalization_high": norm_record["slice_high"],
            "blended_normalization_low": norm_record["blended_low"],
            "blended_normalization_high": norm_record["blended_high"],
            "selected_clahe_profile": norm_record["profile"],
            "selected_clahe_clip": norm_record["clip"],
            "denoise_sigma_um": cfg.get("DENOISE_SIGMA_UM"),
            "denoise_sigma_px": cfg.get("DENOISE_SIGMA"),
            "background_sigma_um": cfg.get("BG_SIGMA_UM"),
            "background_sigma_px": cfg.get("BG_SIGMA"),
            "ridge_sigmas_um": cfg.get("RIDGE_SIGMAS_UM"),
            "ridge_sigmas_px": cfg.get("RIDGE_SIGMAS"),
            "numeric_ridge_high_threshold": th_hi,
            "numeric_ridge_low_threshold": th_lo,
            "valid_roi_pixel_count": int(np.count_nonzero(valid_full_after)),
            "foreground_occupancy_inside_roi": float(np.count_nonzero(out["mask_hyst"] & valid_full_after) / max(np.count_nonzero(valid_full_after), 1)),
            "foreground_occupancy_outside_roi": int(np.count_nonzero(out["mask_hyst"] & ~roi_mask_full)),
            "foreground_occupancy_inside_exclusion_mask": int(np.count_nonzero(out["mask_hyst"] & exclusion_full)),
            "unet_debug": unet_record,
            "skeleton_pixels_before_bridging": bridge_stats["skeleton_pixels_before"],
            "skeleton_pixels_after_bridging": bridge_stats["skeleton_pixels_after"],
            "bridge_inflation_fraction": float((bridge_stats["skeleton_pixels_after"] - bridge_stats["skeleton_pixels_before"]) / max(bridge_stats["skeleton_pixels_before"], 1)),
            "final_detection_count": int(np.max(out["skel_labeled"])),
            "outside_roi_skeleton_occupancy": int(np.count_nonzero(out["skel_pruned"] & ~roi_mask_full)),
            "exclusion_mask_skeleton_occupancy": int(np.count_nonzero(out["skel_pruned"] & exclusion_full)),
        }
        _save_v56_debug(debug_dir, z_idx, out, debug_record)

    return out


# =============================================================================
# MEASUREMENT  (single geodesic pass, all topology in one function)
# =============================================================================

def measure_spermatids(seg, cfg):
    """
    Analyzes mathematically discretised nuclei arrays and derives geometric indices for individual shape profiles.

    Args:
        seg (dict): Dictionary containing labeled binary masks ("primary", "skel_pruned", "dist_clean").
        cfg (dict): Pipeline hyperparameters for absolute scaling calculations.

    Returns:
        pd.DataFrame: Table indexing each isolated object dynamically with mathematical biometrics.
                      Attributes include region area, bounding-boxes, minor/major ellipse axis mapping,
                      orientation, and geodesic structural lengths via binary skeleton processing.
    """
    cfg = cfg_with_resolved_pixels(cfg)
    skel     = seg["skel_pruned"]
    dist     = seg["dist_clean"]
    skel_lab = seg["skel_labeled"]
    unet_prob = seg.get("unet_probability")
    H, W     = skel.shape

    # ------ Filter pass ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        unet_vals = None
        if unet_prob is not None:
            unet_vals = np.asarray(unet_prob[coords[:, 0], coords[:, 1]], dtype=np.float32)
            unet_vals = unet_vals[np.isfinite(unet_vals)]
        accepted_labels.append(sp.label)
        area_est_px = float(gl * width)
        cache[sp.label] = {
            "geo_len":            gl,
            "tortuosity":         tort,
            "n_endpoints":        n_ep,
            "n_branch_nodes":     n_br,
            "width":              width,
            "length_width_ratio": gl / (width + 1e-9),
            "length_px_count":    float(coords.shape[0]),
            "cx": cx, "cy": cy,
            "area_px": area_est_px,
            "skeleton_area_px": float(sp.area),
            "bbox_min_y": float(sp.bbox[0]),
            "bbox_min_x": float(sp.bbox[1]),
            "bbox_max_y": float(sp.bbox[2]),
            "bbox_max_x": float(sp.bbox[3]),
            "orientation": float(sp.orientation),
            "unet_mean_probability": float(np.mean(unet_vals)) if unet_vals is not None and unet_vals.size else np.nan,
            "unet_max_probability": float(np.max(unet_vals)) if unet_vals is not None and unet_vals.size else np.nan,
            "detection_source": "saturn_classical",
        }

    total_rejected = sum(reasons.values())
    if total_rejected > 0:
        print(f"    measure_spermatids rejected {total_rejected} blobs:")
        for k, v in reasons.items():
            if v > 0: print(f"      {k}: {v}")

    if accepted_labels:
        clean_skel = np.isin(skel_lab, accepted_labels)
        final_label = measure.label(clean_skel).astype(np.int32)
    else:
        clean_skel = np.zeros_like(skel, dtype=bool)
        final_label = np.zeros_like(skel_lab, dtype=np.int32)

    # ------ Re-index using cached values (no second Dijkstra pass) ---------------------------------------------
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
            "skeleton_area_px":    c["skeleton_area_px"],
            "bbox_min_y":          c["bbox_min_y"],
            "bbox_min_x":          c["bbox_min_x"],
            "bbox_max_y":          c["bbox_max_y"],
            "bbox_max_x":          c["bbox_max_x"],
            "orientation":         c["orientation"],
            "unet_mean_probability": c.get("unet_mean_probability", np.nan),
            "unet_max_probability": c.get("unet_max_probability", np.nan),
            "detection_source":    c.get("detection_source", "saturn_classical"),
        })

    rescue_reasons = {"short": 0, "loop": 0, "long": 0, "wide": 0, "ratio": 0, "branches": 0, "tortuous": 0, "endpoints": 0}
    if (
        cfg.get("UNET_RESCUE_ENABLE", True)
        and unet_prob is not None
        and np.any(unet_prob > 0)
        and str(cfg.get("SEGMENTATION_ENGINE", "classical_saturn")).strip().lower() in {"hybrid", "unet_assisted"}
    ):
        roi = seg.get("roi_mask", np.ones_like(skel, dtype=bool)).astype(bool)
        exclusion = seg.get("exclusion_mask", np.zeros_like(skel, dtype=bool)).astype(bool)
        valid = roi & ~exclusion
        rescue_thr = float(cfg.get("UNET_RESCUE_THRESHOLD", cfg.get("UNET_SEED_THRESHOLD", 0.50)))
        exclude_px = max(0, int(cfg.get("UNET_RESCUE_EXCLUDE_DILATION_PX", 3)))
        min_component = max(1, int(cfg.get("UNET_RESCUE_MIN_COMPONENT_PX", cfg.get("MIN_OBJ_PX", 3))))
        max_additions = max(0, int(cfg.get("UNET_RESCUE_MAX_ADDITIONS_PER_SLICE", 0)))

        occupied = clean_skel.copy()
        if exclude_px > 0 and np.any(occupied):
            occupied = morphology.binary_dilation(occupied, morphology.disk(exclude_px))
        rescue_mask = (unet_prob >= rescue_thr) & valid & ~occupied
        rescue_mask = remove_objects_smaller_than(rescue_mask, min_component)
        rescue_dist = distance_transform_edt(rescue_mask)
        rescue_skel = skeletonize(rescue_mask)
        rescue_skel &= valid & ~occupied
        rescue_lab = measure.label(rescue_skel)
        rescue_candidates = []

        for sp in measure.regionprops(rescue_lab):
            coords = sp.coords
            if coords.shape[0] < cfg["MIN_SKEL_LEN_PX"]:
                rescue_reasons["short"] += 1
                continue

            topo = measure_topology(coords, W, allow_loops=cfg.get("ALLOW_LOOPS", False))
            if topo is None:
                rescue_reasons["loop"] += 1
                continue

            gl = topo["geo_len"]
            tort = topo["tortuosity"]
            n_ep = topo["n_endpoints"]
            n_br = topo["n_branch_nodes"]
            if not (cfg["MIN_SKEL_LEN_PX"] <= gl <= cfg["MAX_GEODESIC_LEN_PX"]):
                if gl < cfg["MIN_SKEL_LEN_PX"]:
                    rescue_reasons["short"] += 1
                else:
                    rescue_reasons["long"] += 1
                continue

            width = float(np.median(2.0 * rescue_dist[coords[:, 0], coords[:, 1]]))
            if width > cfg["MAX_WIDTH_PX"]:
                rescue_reasons["wide"] += 1
                continue
            if gl / (width + 1e-9) < cfg["MIN_LENGTH_WIDTH_RATIO"]:
                rescue_reasons["ratio"] += 1
                continue
            if n_br > cfg["MAX_BRANCH_NODES"]:
                rescue_reasons["branches"] += 1
                continue
            if n_ep >= 2 and tort > cfg["MAX_TORTUOSITY"]:
                rescue_reasons["tortuous"] += 1
                continue
            if n_ep > cfg["MAX_ENDPOINT_COUNT"]:
                rescue_reasons["endpoints"] += 1
                continue

            unet_vals = np.asarray(unet_prob[coords[:, 0], coords[:, 1]], dtype=np.float32)
            unet_vals = unet_vals[np.isfinite(unet_vals)]
            cy, cx = sp.centroid
            rescue_candidates.append({
                "coords": coords,
                "score": float(np.mean(unet_vals)) if unet_vals.size else 0.0,
                "result": {
                    "length_px_geodesic": gl,
                    "length_px_count": float(coords.shape[0]),
                    "width_px": width,
                    "length_width_ratio": gl / (width + 1e-9),
                    "tortuosity": tort,
                    "n_endpoints": n_ep,
                    "n_branch_nodes": n_br,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "area_px": float(gl * width),
                    "skeleton_area_px": float(sp.area),
                    "bbox_min_y": float(sp.bbox[0]),
                    "bbox_min_x": float(sp.bbox[1]),
                    "bbox_max_y": float(sp.bbox[2]),
                    "bbox_max_x": float(sp.bbox[3]),
                    "orientation": float(sp.orientation),
                    "unet_mean_probability": float(np.mean(unet_vals)) if unet_vals.size else np.nan,
                    "unet_max_probability": float(np.max(unet_vals)) if unet_vals.size else np.nan,
                    "detection_source": "unet_rescued",
                },
            })

        rescue_candidates.sort(key=lambda item: item["score"], reverse=True)
        if max_additions > 0:
            rescue_candidates = rescue_candidates[:max_additions]
        next_label = int(final_label.max()) + 1
        for item in rescue_candidates:
            coords = item["coords"]
            final_label[coords[:, 0], coords[:, 1]] = next_label
            item["result"]["label"] = next_label
            final_results.append(item["result"])
            next_label += 1

        if rescue_candidates or sum(rescue_reasons.values()) > 0:
            print(f"    U-Net rescue accepted {len(rescue_candidates)} blobs at probability >= {rescue_thr:.3f}")
            rejected = sum(rescue_reasons.values())
            if rejected:
                print(f"    U-Net rescue rejected {rejected} blobs:")
                for k, v in rescue_reasons.items():
                    if v > 0:
                        print(f"      {k}: {v}")

    return {"skel_label": final_label, "results": final_results}


# =============================================================================
# OVERLAY  (vectorized LUT)
# =============================================================================

def make_overlay(img_raw, skel_label):
    """
    Generates a colour-coded skeleton overlay on the grayscale raw image.

    Each detected spermatid is assigned a unique hue from the ``gist_rainbow``
    colourmap, dilated by 3 pixels for visibility, and composited onto the
    contrast-stretched raw image.  Background pixels (label == 0) retain the
    original grayscale intensity.

    Implementation notes
    --------------------
    - Vectorised LUT (Look-Up Table) approach avoids per-label loop for speed.
    - ``grey_dilation`` makes thin single-pixel skeletons visible at any zoom.
    - Colour assignment is deterministic: same label ordering -> same colour.

    Args:
        img_raw (np.ndarray): Raw microscopy image (any dtype).
        skel_label (np.ndarray[int]): Integer-labelled skeleton array
            (0 = background, 1..N = individual spermatids).

    Returns:
        np.ndarray: uint8 RGB image, shape ``(H, W, 3)``, ready for ``plt.imshow``
        or saving with :func:`_imwrite`.
    """
    base = normalize_display(img_raw)
    n    = int(skel_label.max())
    if n <= 0:
        # No detections: return grayscale image as RGB
        return (np.stack([base]*3, -1) * 255).astype(np.uint8)
    # Assign one colour per label; prepend black for background (index 0)
    cols    = plt.cm.gist_rainbow(np.linspace(0, 1, n))[:, :3]
    dilated = grey_dilation(skel_label.astype(np.int32), size=3)
    lut     = np.vstack([[0., 0., 0.], cols[:n]])
    rgb     = lut[dilated]
    # Restore original grayscale for background pixels
    m0      = dilated == 0
    rgb[m0, 0] = base[m0]
    rgb[m0, 1] = base[m0]
    rgb[m0, 2] = base[m0]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def make_quality_overlay(img_raw, skel_label, slice_tracks, track_quality_map):
    """
    Draw an audit-coded overlay for a single Z slice.

    Colors:
    - green: biological candidate without warnings
    - yellow/orange: biological candidate with warning-only PSF-sensitive flags
    - red: hard-failed track
    - gray: detection has no track/audit mapping
    """
    base = normalize_display(img_raw)
    rgb = np.stack([base, base, base], axis=-1)
    if skel_label is None or int(np.max(skel_label)) <= 0:
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    label_quality = {}
    if slice_tracks is not None and not slice_tracks.empty:
        for _, row in slice_tracks.iterrows():
            try:
                label = int(row["sperm_id"])
                track_id = int(row["track_id"])
            except Exception:
                continue
            label_quality[label] = track_quality_map.get(track_id)

    dilated = grey_dilation(skel_label.astype(np.int32), size=3)
    for label in np.unique(dilated):
        label = int(label)
        if label == 0:
            continue
        q = label_quality.get(label)
        if q == "candidate":
            color = np.array([0.0, 0.85, 0.25])      # candidate/pass
        elif q == "warning":
            color = np.array([1.0, 0.75, 0.05])      # warning-only
        elif q == "hard_fail":
            color = np.array([1.0, 0.18, 0.05])      # hard fail
        elif q is True:
            color = np.array([0.0, 0.85, 0.25])      # backward-compatible strict pass
        elif q is False:
            color = np.array([1.0, 0.18, 0.05])      # backward-compatible strict fail
        else:
            color = np.array([0.65, 0.65, 0.65])     # untracked/unknown
        mask = dilated == label
        rgb[mask] = 0.25 * rgb[mask] + 0.75 * color

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def export_quality_overlays(out_dir, slice_cache, df_tracked, track_summary):
    """
    Save per-slice and global audit-coded overlays after 3D tracking is audited.
    """
    if not slice_cache or df_tracked is None or df_tracked.empty or track_summary is None or track_summary.empty:
        return None
    if "is_quality_track" not in track_summary.columns or "track_id" not in track_summary.columns:
        return None

    quality_dir = os.path.join(out_dir, "quality_overlays")
    ensure_dir(quality_dir)

    track_quality_map = {}
    for _, row in track_summary.iterrows():
        if pd.isna(row.get("track_id")):
            continue
        tid = int(row["track_id"])
        if "technical_valid" in track_summary.columns:
            if bool(row.get("technical_valid", False)):
                track_quality_map[tid] = "warning" if bool(row.get("morphology_warning", False)) else "candidate"
            else:
                track_quality_map[tid] = "hard_fail"
        elif "is_biological_candidate" in track_summary.columns:
            if bool(row.get("is_biological_candidate")):
                track_quality_map[tid] = "warning" if bool(row.get("has_warning_only", False)) else "candidate"
            else:
                track_quality_map[tid] = "hard_fail"
        else:
            track_quality_map[tid] = bool(row.get("is_quality_track", False))

    max_proj_raw = None
    max_proj_quality = None
    for z_idx in sorted(slice_cache):
        item = slice_cache[z_idx]
        img = item["image"]
        skel_label = item["skel_label"]
        slice_tracks = df_tracked[df_tracked["z_slice"].astype(int) == int(z_idx)]
        quality_rgb = make_quality_overlay(img, skel_label, slice_tracks, track_quality_map)

        raw_rgb = (normalize_display(img) * 255).astype(np.uint8)
        if raw_rgb.ndim == 2:
            raw_rgb = np.stack([raw_rgb] * 3, axis=-1)
        panel = np.hstack([raw_rgb, quality_rgb])
        _imwrite(os.path.join(quality_dir, f"z{int(z_idx):02d}_quality_panel.png"), panel)

        if max_proj_raw is None:
            max_proj_raw = img.copy().astype(np.float32)
            max_proj_quality = quality_rgb.copy().astype(np.float32)
        else:
            max_proj_raw = np.maximum(max_proj_raw, img.astype(np.float32))
            max_proj_quality = np.maximum(max_proj_quality, quality_rgb.astype(np.float32))

    if max_proj_raw is None:
        return None
    raw_p = (normalize_display(max_proj_raw.astype(np.uint16)) * 255).astype(np.uint8)
    if raw_p.ndim == 2:
        raw_p = np.stack([raw_p] * 3, axis=-1)
    quality_p = np.clip(max_proj_quality, 0, 255).astype(np.uint8)
    global_panel = np.hstack([raw_p, quality_p])
    out_path = os.path.join(out_dir, "quality_global_z_projection.png")
    _imwrite(out_path, global_panel)
    return out_path


# =============================================================================
# DISPLAY / SAVE
# =============================================================================

def _safe_show():
    """
    Calls ``plt.show(block=True)`` with a try/except guard.

    On headless systems (Linux CI servers, SSH sessions without X11 forwarding,
    or Windows without a display) matplotlib will raise a ``RuntimeError`` or
    ``_tkinter.TclError`` when trying to open an interactive window.  This
    wrapper silently swallows that error so the pipeline can continue and saves
    the image to disk as a fallback.
    """
    try:
        plt.show(block=True)
    except Exception as e:
        print(f"[WARNING] plt.show() failed: {e}")


def show_single_preview(img_raw, seg, overlay_rgb, results, z_idx, cfg):
    """
    Renders and saves an interactive preview figure for a single analysed Z-slice.

    Layout operates in two modes controlled by ``SHOW_DEBUG_PREVIEW``:

    - **Standard mode** (2 panels): Raw image | Spermatid overlay + length histogram.
    - **Debug mode** (8 panels): Adds intermediate stage images - CLAHE, ridge filter,
      hysteresis mask, cleaned mask, and pruned skeleton - for visual pipeline QC.

    After saving to ``output_dir/preview.png``, the function attempts to open the
    image with the OS default viewer (``os.startfile`` on Windows) as an immediate
    visual check, falling back silently if that fails.

    Args:
        img_raw (np.ndarray): Raw microscopy image.
        seg (dict): Segmentation dictionary from :func:`segment_slice` containing
            intermediate images (``img_eq``, ``ridge``, ``mask_hyst``, etc.).
        overlay_rgb (np.ndarray): Colour overlay from :func:`make_overlay`.
        results (list[dict]): Per-spermatid measurement dictionaries.
        z_idx (int): Z-slice index used for figure titles.
        cfg (dict): Pipeline configuration; reads ``SHOW_DEBUG_PREVIEW``,
            ``OUTPUT_DIR``, and ``UM_PER_PX_XY``.
    """
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
                        label=f"Median={np.median(lengths_um):.1f} um")
        hist_ax.set_xlabel("Geodesic length (um)"); hist_ax.legend(fontsize=8)
        hist_ax.set_title("Length distribution")
    else:
        hist_ax.text(0.5, 0.5, "No detections", ha="center", va="center")
        hist_ax.axis("off")

    plt.tight_layout()

    # Persist preview image so it can be opened by OS or reloaded by GUI
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
    """
    Saves a three-panel publication-quality figure for a single Z-slice.

    Panels:
    1. **Original** - contrast-stretched raw image.
    2. **Spermatid overlay** - colour-coded detections with length labels.
    3. **Length distribution** - histogram of geodesic lengths in um with median line.

    This figure is saved as a PNG per Z-slice and is useful for visual quality
    control and inclusion in lab reports.

    Args:
        img_raw (np.ndarray): Raw microscopy image.
        overlay_rgb (np.ndarray): Colour overlay from :func:`make_overlay`.
        results (list[dict]): Per-spermatid measurement dictionaries.
        out_path (str): Destination PNG file path.
        z_idx (int): Z-slice index for figure titles.
        um (float): Microns-per-pixel scale factor (``UM_PER_PX_XY``).
    """
    lengths_um = [r["length_px_geodesic"] * um for r in results]
    fig, axes  = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(normalize_display(img_raw), cmap="gray")
    axes[0].set_title(f"Z={z_idx:02d} - Original"); axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Z={z_idx:02d} - Spermatids (N={len(results)})")
    axes[1].axis("off")
    for r in results:
        axes[1].text(r["centroid_x"], r["centroid_y"],
                     f"{r['length_px_geodesic']*um:.1f}",
                     color="white", fontsize=4, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))

    if lengths_um:
        axes[2].hist(lengths_um, bins=20, edgecolor="white")
        axes[2].axvline(np.median(lengths_um), lw=2,
                        label=f"Median={np.median(lengths_um):.1f} um")
        axes[2].set_xlabel("Geodesic length (um)"); axes[2].set_ylabel("Count")
        axes[2].set_title(f"Z={z_idx:02d} - Length distribution")
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

_VERSION = "v5.7-unet-ready"


def rows_from_results(results, z_idx, um):
    """
    Converts per-spermatid measurement dictionaries into flat CSV row dictionaries.

    Applies pixel-to-micron scaling (``UM_PER_PX_XY``) for all linear dimensions
    and rounds floating-point values to 3 decimal places for clean spreadsheet output.

    Columns emitted per detection
    ------------------------------
    - ``pipeline_version``     - Code version tag for traceability.
    - ``z_slice``              - Z-plane index of this detection.
    - ``sperm_id``             - Skeleton label ID within this slice.
    - ``length_px_geodesic``   - Geodesic skeleton length in pixels.
    - ``length_um_geodesic``   - Geodesic length in micrometres.
    - ``length_px_count``      - Pixel-count skeleton length (alternative measure).
    - ``length_um_count``      - Pixel-count length in micrometres.
    - ``width_px`` / ``width_um`` - Mean width across the skeleton.
    - ``length_width_ratio``   - Elongation ratio (key morphological filter metric).
    - ``tortuosity``           - Curvature index: geodesic / Euclidean tip-to-tip.
    - ``n_endpoints``          - Number of skeleton endpoints (expect 2 for clean cells).
    - ``n_branch_nodes``       - Number of branch points (3+ neighbours); should be low.
    - ``centroid_x`` / ``centroid_y`` - Pixel centroid for overlay annotation.
    - ``area_px``              - Slender-object area estimate in pixels^2
      (geodesic length * median width), used for area-derived metrics.
    - ``skeleton_area_px``     - Raw skeleton pixel count, kept for audit/debug.

    Args:
        results (list[dict]): Output of :func:`measure_spermatids`.
        z_idx (int): Z-slice index.
        um (float): Microns-per-pixel (``UM_PER_PX_XY``).

    Returns:
        list[dict]: One flat dictionary per detected spermatid.
    """
    return [{
        "pipeline_version":    _VERSION,
        "z_slice":             z_idx,
        "sperm_id":            int(r.get("label", i)),
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
        "skeleton_area_px":    round(r.get("skeleton_area_px", 0.0), 1),
        "bbox_min_y":          r.get("bbox_min_y"),
        "bbox_min_x":          r.get("bbox_min_x"),
        "bbox_max_y":          r.get("bbox_max_y"),
        "bbox_max_x":          r.get("bbox_max_x"),
        "orientation":         round(r.get("orientation", 0.0), 3),
        "detection_source":    r.get("detection_source", "saturn_classical"),
        "unet_mean_probability": round(float(r.get("unet_mean_probability", np.nan)), 4) if np.isfinite(r.get("unet_mean_probability", np.nan)) else np.nan,
        "unet_max_probability":  round(float(r.get("unet_max_probability", np.nan)), 4) if np.isfinite(r.get("unet_max_probability", np.nan)) else np.nan,
    } for i, r in enumerate(results, start=1)]


# =============================================================================
# TRACKING
# =============================================================================

def check_extension_consistency(prev_state, candidate_detection, cfg, overlap_exists=False):
    """
    Check if extending a track with this detection would be biologically consistent.
    Implements 'Continue Unless Implausible' logic for overlapping footprints.

    Stage 2b: All thresholds are now driven by CONFIG for hyperparameter tuning.
    """
    um_xy = cfg["UM_PER_PX_XY"]

    # Read tunable Stage 2 parameters from CONFIG
    stab_thresh = cfg.get("OVERLAP_STABILITY_THRESHOLD", 0.08)
    ori_deg     = cfg.get("OVERLAP_ORIENTATION_DEG", 15.0)
    ovl_mult    = cfg.get("OVERLAP_MULTIPLIER", 1.35)
    min_stable  = cfg.get("OVERLAP_MIN_STABLE_COUNT", 1)

    # Extract previous track state
    prev_x = prev_state["last_x"]
    prev_y = prev_state["last_y"]
    prev_width = prev_state.get("last_width")
    prev_length = prev_state.get("last_length")
    prev_area = prev_state.get("last_area")
    prev_ori = prev_state.get("last_orientation")

    # Extract candidate detection features
    cand_x = candidate_detection["centroid_x"]
    cand_y = candidate_detection["centroid_y"]
    cand_width = candidate_detection.get("width_um")
    cand_length = candidate_detection.get("length_um_geodesic")
    cand_area = candidate_detection.get("area_px")
    cand_ori = candidate_detection.get("orientation")

    # Logic:
    # If overlap_exists, we allow the track to continue IF enough primary metrics are stable.
    # This prevents 'monster merges' where a track jumps onto a totally different cell.
    if overlap_exists:
        stable_count = 0

        # 1. Width stability
        if prev_width and cand_width:
            if (abs(cand_width - prev_width) / max(prev_width, 1e-9)) < stab_thresh:
                stable_count += 1

        # 2. Area stability
        if prev_area and cand_area:
            if (abs(cand_area - prev_area) / max(prev_area, 1e-9)) < stab_thresh:
                stable_count += 1

        # 3. Orientation stability
        if prev_ori is not None and cand_ori is not None:
            diff_rad = abs(cand_ori - prev_ori)
            if diff_rad > math.pi / 2:
                diff_rad = math.pi - diff_rad
            if diff_rad < (ori_deg * math.pi / 180):
                stable_count += 1

        # 4. Length stability
        if prev_length and cand_length:
            if (abs(cand_length - prev_length) / max(prev_length, 1e-9)) < stab_thresh:
                stable_count += 1

        # Require minimum stable metrics to continue an overlapping track
        if stable_count < min_stable:
            return False, f"overlap_but_{stable_count}_stable"

        # Even with stable metrics, still apply capped multiplier for fallback checks
        multiplier = ovl_mult
    else:
        multiplier = 1.0

    # 1. Check centroid jump
    dx = cand_x - prev_x
    dy = cand_y - prev_y
    centroid_jump_um = math.sqrt(dx*dx + dy*dy) * um_xy

    if not overlap_exists and centroid_jump_um > cfg["CONSERVATIVE_MAX_CENTROID_JUMP_UM"]:
        return False, f"centroid_jump={centroid_jump_um:.2f}um"

    # 2. Check width consistency
    if prev_width is not None and cand_width is not None:
        width_ratio = abs(cand_width - prev_width) / max(prev_width, 1e-9)
        if width_ratio > cfg["CONSERVATIVE_MAX_WIDTH_JUMP_RATIO"] * multiplier:
            return False, f"width_jump={width_ratio:.2f}"

    # 3. Check length consistency
    if prev_length is not None and cand_length is not None:
        length_ratio = abs(cand_length - prev_length) / max(prev_length, 1e-9)
        if length_ratio > cfg["CONSERVATIVE_MAX_LENGTH_JUMP_RATIO"] * multiplier:
            return False, f"length_jump={length_ratio:.2f}"

    # 4. Check area consistency
    if prev_area is not None and cand_area is not None:
        area_ratio = abs(cand_area - prev_area) / max(prev_area, 1e-9)
        if area_ratio > cfg["CONSERVATIVE_MAX_AREA_JUMP_RATIO"] * multiplier:
            return False, f"area_jump={area_ratio:.2f}"

    return True, "ok"

def track_across_slices_legacy(detections_df, cfg):
    """
    Conservative tracking natively: stop tracks when consistency breaks.
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
    link_methods = ["new"] * len(df)
    link_distances_um = [np.nan] * len(df)
    link_gap_slices = [0] * len(df)
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
        oris = df.loc[idxs, "orientation"].to_numpy(float) if "orientation" in df.columns else np.full(len(idxs), np.nan)

        # Extract Bounding Box Arrays for Overlap-First Algorithm
        bbox_min_ys = df.loc[idxs, "bbox_min_y"].to_numpy(float) if "bbox_min_y" in df.columns else np.full(len(idxs), np.nan)
        bbox_min_xs = df.loc[idxs, "bbox_min_x"].to_numpy(float) if "bbox_min_x" in df.columns else np.full(len(idxs), np.nan)
        bbox_max_ys = df.loc[idxs, "bbox_max_y"].to_numpy(float) if "bbox_max_y" in df.columns else np.full(len(idxs), np.nan)
        bbox_max_xs = df.loc[idxs, "bbox_max_x"].to_numpy(float) if "bbox_max_x" in df.columns else np.full(len(idxs), np.nan)

        # Find candidate tracks from previous slices
        cand_tracks = [t for t, st in active.items()
                       if 1 <= z - st["last_z"] <= cfg["TRACK_MAX_GAP_SLICES"] + 1]

        used_det, used_trk = set(), set()

        if cand_tracks:
            candidates = []
            pad = cfg.get("TRACK_BBOX_PADDING_PX", 5.0)

            for k, (x, y) in enumerate(zip(xs, ys)):
                # Prepare Candidate State
                det_min_y, det_min_x = bbox_min_ys[k], bbox_min_xs[k]
                det_max_y, det_max_x = bbox_max_ys[k], bbox_max_xs[k]
                has_bbox = np.isfinite(det_min_y)

                cand_det = {
                    "centroid_x": float(x),
                    "centroid_y": float(y),
                    "width_um": float(widths[k]) if np.isfinite(widths[k]) else None,
                    "length_um_geodesic": float(lengths[k]) if np.isfinite(lengths[k]) else None,
                    "area_px": float(areas[k]) if np.isfinite(areas[k]) else None,
                    "orientation": float(oris[k]) if np.isfinite(oris[k]) else None,
                }

                for j, tid in enumerate(cand_tracks):
                    trk_st = active[tid]

                    # Compute standard spatial distance
                    dx = float(x) - trk_st["last_x"]
                    dy = float(y) - trk_st["last_y"]
                    d_val = np.sqrt(dx*dx + dy*dy)

                    # Perform Overlap-First physical footprint collision check
                    overlap_exists = False
                    if has_bbox and "last_bbox" in trk_st and trk_st["last_bbox"] is not None:
                        t_min_y, t_min_x, t_max_y, t_max_x = trk_st["last_bbox"]
                        # Bounding box intersection formula with padding forgiveness
                        if not (det_max_y + pad < t_min_y or det_min_y - pad > t_max_y or
                                det_max_x + pad < t_min_x or det_min_x - pad > t_max_x):
                            overlap_exists = True

                    # Accept candidate if it physically overlaps, OR if it's within pure centroid limits (implausible fallback)
                    if overlap_exists or d_val <= max_dist_px:
                        # Continue unless implausible bounds
                        is_consistent, reason = check_extension_consistency(
                            trk_st, cand_det, cfg, overlap_exists=overlap_exists
                        )

                        if is_consistent:
                            # Massive scoring favor (+10,000) for overlaps so they greedily override standard loose distance matches
                            score = float(d_val) if not overlap_exists else (float(d_val) - 10000.0)
                            method = "overlap" if overlap_exists else "centroid"
                            gap_slices = int(z - trk_st["last_z"])
                            candidates.append((score, k, j, method, float(d_val), gap_slices))
                        else:
                            if tid not in stopped_tracks:
                                stopped_tracks[tid] = f"z={z}, reason={reason}"

            # Sort by score and assign greedily
            candidates.sort(key=lambda x: x[0])
            for score, det_k, trk_j, method, d_val_px, gap_slices in candidates:
                if det_k in used_det or trk_j in used_trk:
                    continue

                used_det.add(det_k)
                used_trk.add(trk_j)
                tid = cand_tracks[trk_j]
                row_idx = int(idxs[det_k])
                track_ids[row_idx] = tid
                link_methods[row_idx] = method
                link_distances_um[row_idx] = d_val_px * cfg["UM_PER_PX_XY"]
                link_gap_slices[row_idx] = gap_slices

                # Update track state
                active[tid] = {
                    "last_z": int(z),
                    "last_x": float(xs[det_k]),
                    "last_y": float(ys[det_k]),
                    "last_width": float(widths[det_k]) if np.isfinite(widths[det_k]) else None,
                    "last_length": float(lengths[det_k]) if np.isfinite(lengths[det_k]) else None,
                    "last_area": float(areas[det_k]) if np.isfinite(areas[det_k]) else None,
                    "last_orientation": float(oris[det_k]) if np.isfinite(oris[det_k]) else None,
                    "last_bbox": (bbox_min_ys[det_k], bbox_min_xs[det_k], bbox_max_ys[det_k], bbox_max_xs[det_k]) if np.isfinite(bbox_min_ys[det_k]) else None,
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
                    "last_orientation": float(oris[det_k]) if np.isfinite(oris[det_k]) else None,
                    "last_bbox": (bbox_min_ys[det_k], bbox_min_xs[det_k], bbox_max_ys[det_k], bbox_max_xs[det_k]) if np.isfinite(bbox_min_ys[det_k]) else None,
                }
                next_tid += 1

        # Remove stale tracks (exceeded gap)
        for tid in [t for t, st in active.items()
                    if z - st["last_z"] > cfg["TRACK_MAX_GAP_SLICES"] + 1]:
            del active[tid]

    df["track_id"] = track_ids
    df["track_link_method"] = link_methods
    df["track_link_distance_um"] = np.round(link_distances_um, 3)
    df["track_link_gap_slices"] = link_gap_slices

    # Print tracking stats
    print(f"  Conservative tracking: {len(stopped_tracks)} tracks stopped early for consistency")

    # Inject maximum 2D Euclidean distance of the physical shape prior to grouping
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
        min_area_px     = ("area_px",            "min"),
        max_area_px     = ("area_px",            "max"),
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

    # 1. Z span and sampled Z coverage are distinct:
    # span is endpoint-to-endpoint displacement; covered includes slice thickness.
    z_span = (ts["z_end"] - ts["z_start"]) * um_z
    z_covered = (ts["z_end"] - ts["z_start"] + 1) * um_z
    ts["z_extent_um"] = z_span
    ts["z_span_um"] = z_span
    ts["z_covered_um"] = z_covered

    # Geodesic vertical displacement for Euclidean distance is purely centroid-to-centroid
    dz_euc = z_span

    # 2. Total 3D Length (Lateral-Corrected Hypotenuse)
    # The physical 3D shape arc relies on the maximum length of its 2D projection
    euc_2d_centroid = np.sqrt((ts["x_end"] - ts["x_start"])**2 + (ts["y_end"] - ts["y_start"])**2) * um_xy
    lat_geodesic = np.maximum(ts["max_length_2d"], euc_2d_centroid)
    l3d = np.sqrt(lat_geodesic**2 + z_span**2)
    ts["total_3d_length_um"] = l3d

    # 3. 3D Volume (sum of per-slice projected area estimates * Z_step)
    ts["volume_um3"] = ts["sum_area_px"] * (um_xy**2) * um_z

    # 4. 3D Tortuosity (Total 3D Geodesic Length / 3D End-To-End Euclidean Distance)
    euc_3d = np.sqrt(ts["max_euc_2d"]**2 + dz_euc**2)
    safe_euc = np.maximum(euc_3d, 0.1)
    tort_raw = l3d / safe_euc
    ts["tortuosity_3d"] = np.minimum(tort_raw, 20.0)

    # 5. Taper Ratio (max/min area across the full track)
    ts["taper_ratio"] = ts["max_area_px"] / np.maximum(ts["min_area_px"], 0.001)

    # 6. Effective Thickness / Diameter
    cross_area = ts["volume_um3"] / np.maximum(ts["total_3d_length_um"], 0.1)
    ts["thickness_um"] = 2 * np.sqrt(cross_area / np.pi)

    # 7. Orientation Angles (Pitch and Yaw)
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
        "track_id", "total_3d_length_um", "z_extent_um", "z_span_um", "z_covered_um", "volume_um3", "tortuosity_3d",
        "thickness_um", "pitch_deg", "yaw_deg", "taper_ratio", "nearest_neighbor_um",
        "n_slices", "z_start", "z_end", "max_length_2d", "sum_area_px",
        "min_area_px", "max_area_px", "area_start", "area_end"
    ]
    ts = ts[cols_ordered]
    ts["track_stop_reason"] = ts["track_id"].map(stopped_tracks).fillna("")

    return df, ts


def _angle_diff_deg(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _relative_change(a, b):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-9)


def _bbox_overlap_fraction(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    vals = [ay0, ax0, ay1, ax1, by0, bx0, by1, bx1]
    if not all(np.isfinite(v) for v in vals):
        return 0.0
    inter_y = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter = inter_y * inter_x
    area_a = max(0.0, (ay1 - ay0) * (ax1 - ax0))
    area_b = max(0.0, (by1 - by0) * (bx1 - bx0))
    denom = max(min(area_a, area_b), 1e-9)
    return float(min(inter / denom, 1.0))


def _finite_float_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _read_unet_probability(row):
    """
    Read optional U-Net support from a detection row.

    v5.7 treats U-Net fields as optional evidence. Classical Saturn detections
    do not need these columns, and missing fields produce no tracking penalty.
    """
    for col in (
        "unet_mean_probability",
        "unet_max_probability",
        "unet_seed_probability",
        "unet_probability",
    ):
        if col in row.index:
            value = _finite_float_or_none(row[col])
            if value is not None:
                return float(np.clip(value, 0.0, 1.0))
    return None


def _unet_tracking_enabled(cfg):
    engine = str(cfg.get("SEGMENTATION_ENGINE", "classical_saturn")).strip().lower()
    return bool(cfg.get("UNET_TRACKING_SUPPORT", True)) and engine in ("unet_assisted", "hybrid")


def _unet_link_cost_terms(det_prob, prev_prob, cfg, repair=False):
    if (not _unet_tracking_enabled(cfg)) or (det_prob is None and prev_prob is None):
        return 0.0, 0.0, 0.0

    support_weight = float(
        cfg.get(
            "HYBRID_REPAIR_UNET_SUPPORT_WEIGHT" if repair else "ASSIGNMENT_UNET_SUPPORT_WEIGHT",
            0.0,
        )
    )
    continuity_weight = float(cfg.get("ASSIGNMENT_UNET_CONTINUITY_WEIGHT", 0.0))

    probs = [p for p in (det_prob, prev_prob) if p is not None]
    mean_prob = float(np.mean(probs)) if probs else 0.0
    support_term = support_weight * (1.0 - mean_prob)
    continuity_term = 0.0
    if det_prob is not None and prev_prob is not None:
        continuity_term = continuity_weight * abs(float(det_prob) - float(prev_prob))
    return support_term + continuity_term, support_term, continuity_term


def _summarize_tracked_detections(df, stopped_tracks, cfg):
    if df.empty:
        return df, pd.DataFrame()

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
        min_area_px     = ("area_px",            "min"),
        max_area_px     = ("area_px",            "max"),
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
    um_z = cfg["UM_PER_SLICE_Z"]
    z_span = (ts["z_end"] - ts["z_start"]) * um_z
    z_covered = (ts["z_end"] - ts["z_start"] + 1) * um_z
    ts["z_extent_um"] = z_span
    ts["z_span_um"] = z_span
    ts["z_covered_um"] = z_covered

    euc_2d_centroid = np.sqrt((ts["x_end"] - ts["x_start"])**2 + (ts["y_end"] - ts["y_start"])**2) * um_xy
    lat_geodesic = np.maximum(ts["max_length_2d"], euc_2d_centroid)
    l3d = np.sqrt(lat_geodesic**2 + z_span**2)
    ts["total_3d_length_um"] = l3d
    ts["volume_um3"] = ts["sum_area_px"] * (um_xy**2) * um_z

    euc_3d = np.sqrt(ts["max_euc_2d"]**2 + z_span**2)
    safe_euc = np.maximum(euc_3d, 0.1)
    ts["tortuosity_3d"] = np.minimum(l3d / safe_euc, 20.0)
    ts["taper_ratio"] = ts["max_area_px"] / np.maximum(ts["min_area_px"], 0.001)

    cross_area = ts["volume_um3"] / np.maximum(ts["total_3d_length_um"], 0.1)
    ts["thickness_um"] = 2 * np.sqrt(cross_area / np.pi)

    dx = (ts["x_end"] - ts["x_start"]) * um_xy
    dy = (ts["y_end"] - ts["y_start"]) * um_xy
    v_mag = np.sqrt(dx**2 + dy**2 + z_span**2)
    safe_v = np.maximum(v_mag, 1e-9)
    ts["pitch_deg"] = np.abs(np.arcsin(z_span / safe_v)) * (180.0 / np.pi)
    ts["yaw_deg"] = np.arctan2(dy, dx) * (180.0 / np.pi)

    if len(ts) > 1:
        centers = np.column_stack((ts["x_mean"] * um_xy, ts["y_mean"] * um_xy, ts["z_mean"] * um_z))
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        ts["nearest_neighbor_um"] = dists[:, 1]
    else:
        ts["nearest_neighbor_um"] = np.nan

    unet_summary_cols = []
    for prob_col in (
        "unet_mean_probability",
        "unet_max_probability",
        "unet_seed_probability",
        "unet_probability",
    ):
        if prob_col in df.columns:
            vals = pd.to_numeric(df[prob_col], errors="coerce")
            if vals.notna().any():
                df[prob_col] = vals
                mean_by_track = df.groupby("track_id")[prob_col].mean()
                max_by_track = df.groupby("track_id")[prob_col].max()
                ts[f"track_mean_{prob_col}"] = ts["track_id"].map(mean_by_track)
                ts[f"track_max_{prob_col}"] = ts["track_id"].map(max_by_track)
                unet_summary_cols.extend([f"track_mean_{prob_col}", f"track_max_{prob_col}"])

    cols_ordered = [
        "track_id", "total_3d_length_um", "z_extent_um", "z_span_um", "z_covered_um", "volume_um3", "tortuosity_3d",
        "thickness_um", "pitch_deg", "yaw_deg", "taper_ratio", "nearest_neighbor_um",
        "n_slices", "z_start", "z_end", "max_length_2d", "sum_area_px",
        "min_area_px", "max_area_px", "area_start", "area_end"
    ] + unet_summary_cols
    ts = ts[cols_ordered]
    ts["track_stop_reason"] = ts["track_id"].map(stopped_tracks).fillna("")
    return df, ts


def track_across_slices_global_assignment(detections_df, cfg):
    """
    V5.6 ROI-ADAPTIVE tracker: link each slice with a global assignment cost matrix.

    This is intentionally self-contained. It gives us a LapTrack-like assignment
    backend without requiring a new dependency while we evaluate whether global
    assignment improves the fragmentation/over-linking tradeoff.
    """
    if detections_df.empty:
        detections_df = detections_df.copy()
        detections_df["track_id"] = pd.Series(dtype=int)
        return detections_df, pd.DataFrame()

    df = (detections_df.copy()
                       .sort_values(["z_slice", "sperm_id"])
                       .reset_index(drop=True))
    track_ids = [-1] * len(df)
    link_methods = ["new"] * len(df)
    link_distances_um = [np.nan] * len(df)
    link_gap_slices = [0] * len(df)
    stopped_tracks = {}

    um_xy = cfg["UM_PER_PX_XY"]
    max_dist_um = float(cfg.get("TRACK_MAX_DIST_UM", 7.0))
    max_cost = float(cfg.get("ASSIGNMENT_MAX_COST", 8.0))
    weights = {
        "dist": float(cfg.get("ASSIGNMENT_DIST_WEIGHT", 1.0)),
        "overlap": float(cfg.get("ASSIGNMENT_OVERLAP_WEIGHT", 2.0)),
        "length": float(cfg.get("ASSIGNMENT_LENGTH_WEIGHT", 2.0)),
        "width": float(cfg.get("ASSIGNMENT_WIDTH_WEIGHT", 1.2)),
        "area": float(cfg.get("ASSIGNMENT_AREA_WEIGHT", 0.9)),
        "angle": float(cfg.get("ASSIGNMENT_ANGLE_WEIGHT", 0.4)),
    }

    next_tid = 1
    active = {}
    rows_by_z = {z: df.index[df["z_slice"] == z].to_numpy()
                 for z in sorted(df["z_slice"].unique())}

    for z, idxs in rows_by_z.items():
        dets = []
        for row_idx in idxs:
            row = df.loc[row_idx]
            bbox = None
            if {"bbox_min_y", "bbox_min_x", "bbox_max_y", "bbox_max_x"}.issubset(df.columns):
                bbox = (row["bbox_min_y"], row["bbox_min_x"], row["bbox_max_y"], row["bbox_max_x"])
            dets.append({
                "row_idx": int(row_idx),
                "x": float(row["centroid_x"]),
                "y": float(row["centroid_y"]),
                "width": float(row["width_um"]) if "width_um" in df.columns and np.isfinite(row["width_um"]) else None,
                "length": float(row["length_um_geodesic"]) if "length_um_geodesic" in df.columns and np.isfinite(row["length_um_geodesic"]) else None,
                "area": float(row["area_px"]) if "area_px" in df.columns and np.isfinite(row["area_px"]) else None,
                "orientation": float(row["orientation"]) if "orientation" in df.columns and np.isfinite(row["orientation"]) else None,
                "unet_probability": _read_unet_probability(row),
                "bbox": bbox,
            })

        cand_tids = [tid for tid, st in active.items()
                     if 1 <= z - st["last_z"] <= cfg["TRACK_MAX_GAP_SLICES"] + 1]
        assigned_dets = set()
        assigned_tracks = set()

        if cand_tids and dets:
            cost = np.full((len(cand_tids), len(dets)), 1e9, dtype=float)
            dist_cache = {}
            method_cache = {}
            for ti, tid in enumerate(cand_tids):
                st = active[tid]
                for di, det in enumerate(dets):
                    dx = det["x"] - st["last_x"]
                    dy = det["y"] - st["last_y"]
                    dist_um = math.sqrt(dx * dx + dy * dy) * um_xy
                    overlap = _bbox_overlap_fraction(det["bbox"], st.get("last_bbox"))
                    if dist_um > max_dist_um and overlap <= 0:
                        continue

                    dist_term = dist_um / max(max_dist_um, 1e-9)
                    overlap_term = 1.0 - overlap
                    length_term = _relative_change(det["length"], st.get("last_length"))
                    width_term = _relative_change(det["width"], st.get("last_width"))
                    area_term = _relative_change(det["area"], st.get("last_area"))
                    angle_term = _angle_diff_deg(det["orientation"], st.get("last_orientation")) / 90.0
                    unet_term, _, _ = _unet_link_cost_terms(
                        det["unet_probability"],
                        st.get("last_unet_probability"),
                        cfg,
                        repair=False,
                    )
                    c = (
                        weights["dist"] * dist_term
                        + weights["overlap"] * overlap_term
                        + weights["length"] * length_term
                        + weights["width"] * width_term
                        + weights["area"] * area_term
                        + weights["angle"] * angle_term
                        + unet_term
                    )
                    cost[ti, di] = c
                    dist_cache[(ti, di)] = dist_um
                    method_cache[(ti, di)] = "assignment_overlap" if overlap > 0 else "assignment_cost"

            row_ind, col_ind = linear_sum_assignment(cost)
            for ti, di in zip(row_ind, col_ind):
                if cost[ti, di] > max_cost:
                    continue
                tid = cand_tids[int(ti)]
                det = dets[int(di)]
                row_idx = det["row_idx"]
                track_ids[row_idx] = tid
                link_methods[row_idx] = method_cache.get((int(ti), int(di)), "assignment_cost")
                link_distances_um[row_idx] = dist_cache.get((int(ti), int(di)), np.nan)
                link_gap_slices[row_idx] = int(z - active[tid]["last_z"])
                assigned_dets.add(int(di))
                assigned_tracks.add(tid)
                active[tid] = {
                    "last_z": int(z),
                    "last_x": det["x"],
                    "last_y": det["y"],
                    "last_width": det["width"],
                    "last_length": det["length"],
                    "last_area": det["area"],
                    "last_orientation": det["orientation"],
                    "last_unet_probability": det["unet_probability"],
                    "last_bbox": det["bbox"],
                }

            for ti, tid in enumerate(cand_tids):
                if tid not in assigned_tracks and np.isfinite(cost[ti]).any():
                    best = float(np.min(cost[ti]))
                    if best <= max_cost * 1.5 and tid not in stopped_tracks:
                        stopped_tracks[tid] = f"z={z}, assignment_unmatched_cost={best:.2f}"

        for di, det in enumerate(dets):
            row_idx = det["row_idx"]
            if di in assigned_dets or track_ids[row_idx] != -1:
                continue
            track_ids[row_idx] = next_tid
            active[next_tid] = {
                "last_z": int(z),
                "last_x": det["x"],
                "last_y": det["y"],
                "last_width": det["width"],
                "last_length": det["length"],
                "last_area": det["area"],
                "last_orientation": det["orientation"],
                "last_unet_probability": det["unet_probability"],
                "last_bbox": det["bbox"],
            }
            next_tid += 1

        for tid in [t for t, st in active.items()
                    if z - st["last_z"] > cfg["TRACK_MAX_GAP_SLICES"] + 1]:
            del active[tid]

    df["track_id"] = track_ids
    df["track_link_method"] = link_methods
    df["track_link_distance_um"] = np.round(link_distances_um, 3)
    df["track_link_gap_slices"] = link_gap_slices
    print(f"  Global-assignment tracking: {len(stopped_tracks)} near-miss tracks recorded")
    return _summarize_tracked_detections(df, stopped_tracks, cfg)


def _row_endpoint(row):
    bbox = None
    if {"bbox_min_y", "bbox_min_x", "bbox_max_y", "bbox_max_x"}.issubset(row.index):
        bbox = (row["bbox_min_y"], row["bbox_min_x"], row["bbox_max_y"], row["bbox_max_x"])
    return {
        "z": int(row["z_slice"]),
        "x": float(row["centroid_x"]),
        "y": float(row["centroid_y"]),
        "width": float(row["width_um"]) if "width_um" in row.index and np.isfinite(row["width_um"]) else None,
        "length": float(row["length_um_geodesic"]) if "length_um_geodesic" in row.index and np.isfinite(row["length_um_geodesic"]) else None,
        "area": float(row["area_px"]) if "area_px" in row.index and np.isfinite(row["area_px"]) else None,
        "orientation": float(row["orientation"]) if "orientation" in row.index and np.isfinite(row["orientation"]) else None,
        "unet_probability": _read_unet_probability(row),
        "bbox": bbox,
    }


def _hybrid_repair_cost(src_end, dst_start, cfg):
    um_xy = cfg["UM_PER_PX_XY"]
    max_dist_um = float(cfg.get("HYBRID_REPAIR_MAX_LINK_DIST_UM", cfg.get("TRACK_MAX_DIST_UM", 6.0)))
    weights = {
        "dist": float(cfg.get("ASSIGNMENT_DIST_WEIGHT", 1.0)),
        "overlap": float(cfg.get("ASSIGNMENT_OVERLAP_WEIGHT", 2.0)),
        "length": float(cfg.get("ASSIGNMENT_LENGTH_WEIGHT", 2.0)),
        "width": float(cfg.get("ASSIGNMENT_WIDTH_WEIGHT", 1.2)),
        "area": float(cfg.get("ASSIGNMENT_AREA_WEIGHT", 0.9)),
        "angle": float(cfg.get("ASSIGNMENT_ANGLE_WEIGHT", 0.4)),
    }
    dx = dst_start["x"] - src_end["x"]
    dy = dst_start["y"] - src_end["y"]
    dist_um = math.sqrt(dx * dx + dy * dy) * um_xy
    overlap = _bbox_overlap_fraction(dst_start["bbox"], src_end["bbox"])
    if dist_um > max_dist_um and overlap <= 0:
        return np.inf, dist_um, overlap

    dist_term = dist_um / max(max_dist_um, 1e-9)
    overlap_term = 1.0 - overlap
    unet_term, _, _ = _unet_link_cost_terms(
        dst_start.get("unet_probability"),
        src_end.get("unet_probability"),
        cfg,
        repair=True,
    )
    cost = (
        weights["dist"] * dist_term
        + weights["overlap"] * overlap_term
        + weights["length"] * _relative_change(dst_start["length"], src_end["length"])
        + weights["width"] * _relative_change(dst_start["width"], src_end["width"])
        + weights["area"] * _relative_change(dst_start["area"], src_end["area"])
        + weights["angle"] * (_angle_diff_deg(dst_start["orientation"], src_end["orientation"]) / 90.0)
        + unet_term
    )
    return cost, dist_um, overlap


def _estimated_merged_length_um(df, track_ids, cfg):
    sub = df[df["track_id"].isin(track_ids)].sort_values(["z_slice", "sperm_id"])
    if sub.empty:
        return np.inf
    um_xy = cfg["UM_PER_PX_XY"]
    um_z = cfg["UM_PER_SLICE_Z"]
    first = sub.iloc[0]
    last = sub.iloc[-1]
    z_span = (float(sub["z_slice"].max()) - float(sub["z_slice"].min())) * um_z
    centroid_span = math.sqrt(
        (float(last["centroid_x"]) - float(first["centroid_x"])) ** 2
        + (float(last["centroid_y"]) - float(first["centroid_y"])) ** 2
    ) * um_xy
    lat_len = max(float(sub["length_um_geodesic"].max()), centroid_span)
    return math.sqrt(lat_len * lat_len + z_span * z_span)


def track_across_slices_hybrid_repair(detections_df, cfg):
    """
    V5.6 ROI-ADAPTIVE tracker: keep the conservative legacy tracker, then run a
    narrow global-assignment-style repair only on short fragments.

    The repair pass is deliberately asymmetric: it can merge obvious fragments,
    but ambiguous candidates remain split so the audit does not inherit long
    false-positive chains.
    """
    df, ts = track_across_slices_legacy(detections_df, cfg)
    if df.empty or ts.empty or len(ts) < 2:
        return df, ts

    max_gap = int(cfg.get("HYBRID_REPAIR_MAX_GAP_SLICES", cfg.get("TRACK_MAX_GAP_SLICES", 1)))
    max_frag_slices = int(cfg.get("HYBRID_REPAIR_MAX_FRAGMENT_SLICES", 2))
    max_cost = float(cfg.get("HYBRID_REPAIR_MAX_COST", 3.6))
    min_overlap = float(cfg.get("HYBRID_REPAIR_MIN_OVERLAP", 0.05))
    max_final_length = float(cfg.get("HYBRID_REPAIR_MAX_FINAL_LENGTH_UM", cfg.get("AUDIT_MAX_LENGTH_UM", 15.0)))

    endpoints = {}
    for tid, grp in df.sort_values(["z_slice", "sperm_id"]).groupby("track_id"):
        endpoints[int(tid)] = {
            "first": _row_endpoint(grp.iloc[0]),
            "last": _row_endpoint(grp.iloc[-1]),
        }

    ts_by_tid = ts.set_index("track_id")
    candidates = []
    tids = list(endpoints.keys())
    starts_by_z = {}
    for tid in tids:
        starts_by_z.setdefault(endpoints[tid]["first"]["z"], []).append(tid)

    for src_tid in tids:
        src_end = endpoints[src_tid]["last"]
        src_n = int(ts_by_tid.loc[src_tid, "n_slices"])
        possible_dsts = []
        for gap in range(max_gap + 1):
            possible_dsts.extend(starts_by_z.get(src_end["z"] + gap + 1, []))
        for dst_tid in possible_dsts:
            if src_tid == dst_tid:
                continue
            dst_start = endpoints[dst_tid]["first"]
            dst_n = int(ts_by_tid.loc[dst_tid, "n_slices"])
            z_gap = dst_start["z"] - src_end["z"] - 1
            if z_gap < 0 or z_gap > max_gap:
                continue
            if src_n > max_frag_slices and dst_n > max_frag_slices:
                continue
            cost, dist_um, overlap = _hybrid_repair_cost(src_end, dst_start, cfg)
            if not np.isfinite(cost) or cost > max_cost:
                continue
            if overlap < min_overlap and dist_um > float(cfg.get("HYBRID_REPAIR_MAX_LINK_DIST_UM", cfg.get("TRACK_MAX_DIST_UM", 6.0))) * 0.6:
                continue
            est_len = _estimated_merged_length_um(df, [src_tid, dst_tid], cfg)
            if est_len > max_final_length:
                continue
            candidates.append((cost, dist_um, overlap, src_tid, dst_tid))

    parent = {tid: tid for tid in tids}

    def find(tid):
        while parent[tid] != tid:
            parent[tid] = parent[parent[tid]]
            tid = parent[tid]
        return tid

    repair_count = 0
    repaired_targets = set()
    for cost, dist_um, overlap, src_tid, dst_tid in sorted(candidates, key=lambda x: x[0]):
        src_root = find(src_tid)
        dst_root = find(dst_tid)
        if src_root == dst_root or dst_root in repaired_targets:
            continue
        merged_members = [tid for tid in tids if find(tid) in (src_root, dst_root)]
        if _estimated_merged_length_um(df, merged_members, cfg) > max_final_length:
            continue
        parent[dst_root] = src_root
        repaired_targets.add(dst_root)
        repair_count += 1

        dst_start_z = endpoints[dst_tid]["first"]["z"]
        first_dst_mask = (df["track_id"] == dst_tid) & (df["z_slice"] == dst_start_z)
        first_dst_idx = df[first_dst_mask].index[:1]
        if len(first_dst_idx):
            idx = first_dst_idx[0]
            df.loc[idx, "track_link_method"] = "hybrid_repair"
            df.loc[idx, "track_link_distance_um"] = round(float(dist_um), 3)
            df.loc[idx, "track_link_gap_slices"] = int(endpoints[dst_tid]["first"]["z"] - endpoints[src_tid]["last"]["z"])

    if repair_count:
        df["track_id"] = df["track_id"].map(lambda tid: find(int(tid)))

    stopped_tracks = {}
    print(f"  Hybrid repair tracking: {repair_count} conservative fragment merges accepted")
    return _summarize_tracked_detections(df, stopped_tracks, cfg)


def track_across_slices(detections_df, cfg):
    backend = str(cfg.get("TRACKING_BACKEND", "legacy")).strip().lower()
    if backend in ("hybrid_repair", "hybrid", "repair"):
        return track_across_slices_hybrid_repair(detections_df, cfg)
    if backend in ("global_assignment", "assignment", "hungarian"):
        return track_across_slices_global_assignment(detections_df, cfg)
    return track_across_slices_legacy(detections_df, cfg)


# =============================================================================
# QUALITY AUDIT & OUTLIER FLAGGING
# =============================================================================

def _join_flag_names(mask_list, n_rows):
    strings = [""] * n_rows
    any_mask = np.zeros(n_rows, dtype=bool)
    for flag_name, mask in mask_list:
        mask_arr = mask.values if hasattr(mask, "values") else np.array(mask)
        mask_arr = mask_arr.astype(bool)
        any_mask |= mask_arr
        for i in range(n_rows):
            if mask_arr[i]:
                strings[i] = f"{strings[i]},{flag_name}" if strings[i] else flag_name
    return strings, any_mask


def _series_false(df_tracks):
    return pd.Series(False, index=df_tracks.index)


def _comparative_audit_masks(df_tracks, cfg):
    """Return technical-failure and morphology-warning masks for comparative analysis."""
    technical_masks = []
    morphology_masks = []
    reference_masks = []
    n = len(df_tracks)
    false = _series_false(df_tracks)

    for col in ("track_id", "centroid_x", "centroid_y"):
        if col in df_tracks.columns:
            bad = ~np.isfinite(pd.to_numeric(df_tracks[col], errors="coerce"))
            technical_masks.append((f"invalid_{col}", bad))

    if "total_3d_length_um" in df_tracks.columns:
        length = pd.to_numeric(df_tracks["total_3d_length_um"], errors="coerce")
        invalid_length = (~np.isfinite(length)) | (length <= 0)
        technical_masks.append(("invalid_length", invalid_length))
        morphology_masks.extend([
            ("long", length > cfg.get("AUDIT_MAX_LENGTH_UM", 15.0)),
            ("short", length < cfg.get("MIN_SKEL_LEN_UM", 0.0)),
            ("extreme_component_length", length > cfg.get("MAX_GEODESIC_LEN_UM", 20.0) * 2.0),
        ])
        reference_masks.extend([
            ("long", length > cfg.get("AUDIT_MAX_LENGTH_UM", 15.0)),
            ("short", length < cfg.get("MIN_SKEL_LEN_UM", 0.0)),
        ])

    if "tortuosity_3d" in df_tracks.columns:
        tort = pd.to_numeric(df_tracks["tortuosity_3d"], errors="coerce")
        morphology_masks.append(("high_tortuosity", tort > cfg.get("AUDIT_MAX_TORTUOSITY", 1.5)))
        reference_masks.append(("high_tortuosity", tort > cfg.get("AUDIT_MAX_TORTUOSITY", 1.5)))

    if "thickness_um" in df_tracks.columns:
        thick = pd.to_numeric(df_tracks["thickness_um"], errors="coerce")
        morphology_masks.extend([
            ("wide", thick > cfg.get("AUDIT_MAX_THICKNESS_UM", 2.0)),
            ("thin", thick < max(0.0, cfg.get("AUDIT_MAX_THICKNESS_UM", 2.0) * 0.20)),
        ])
        reference_masks.append(("wide", thick > cfg.get("AUDIT_MAX_THICKNESS_UM", 2.0)))

    if "taper_ratio" in df_tracks.columns:
        taper = pd.to_numeric(df_tracks["taper_ratio"], errors="coerce")
        morphology_masks.extend([
            ("high_taper", taper > cfg.get("AUDIT_MAX_TAPER_RATIO", 1.5)),
            ("low_taper", taper < 1.0),
        ])
        reference_masks.append(("high_taper", taper > cfg.get("AUDIT_MAX_TAPER_RATIO", 1.5)))

    if "length_width_ratio" in df_tracks.columns:
        lwr = pd.to_numeric(df_tracks["length_width_ratio"], errors="coerce")
        morphology_masks.append(("low_length_width_ratio", lwr < cfg.get("MIN_LENGTH_WIDTH_RATIO", 2.5)))
        reference_masks.append(("low_length_width_ratio", lwr < cfg.get("MIN_LENGTH_WIDTH_RATIO", 2.5)))

    if "pitch_deg" in df_tracks.columns:
        pitch = pd.to_numeric(df_tracks["pitch_deg"], errors="coerce")
        morphology_masks.append(("unusual_pitch", (pitch < 5.0) | (pitch > 175.0)))

    if "volume_um3" in df_tracks.columns:
        volume = pd.to_numeric(df_tracks["volume_um3"], errors="coerce")
        morphology_masks.append(("unusual_volume", volume <= 0))

    z_col = next((c for c in ("z_span_um", "z_covered_um") if c in df_tracks.columns), None)
    if z_col:
        zspan = pd.to_numeric(df_tracks[z_col], errors="coerce")
        morphology_masks.append(("unusual_z_span", zspan < 0))

    if "nearest_neighbor_um" in df_tracks.columns:
        nn = pd.to_numeric(df_tracks["nearest_neighbor_um"], errors="coerce")
        morphology_masks.append(("unusual_nearest_neighbor_distance", (nn >= 0) & (nn < cfg.get("UM_PER_PX_XY", 1.0))))

    branch_col = next((c for c in ("max_branch_nodes", "branch_nodes", "n_branch_nodes") if c in df_tracks.columns), None)
    if branch_col:
        branches = pd.to_numeric(df_tracks[branch_col], errors="coerce").fillna(0)
        technical_masks.append(("gross_branched_tissue_network", branches > max(3, cfg.get("MAX_BRANCH_NODES", 0) + 3)))

    if "segmentation_leakage" in df_tracks.columns:
        technical_masks.append(("segmentation_leakage", df_tracks["segmentation_leakage"].astype(bool)))
    if "outside_roi_pixel_count" in df_tracks.columns:
        technical_masks.append(("outside_roi", pd.to_numeric(df_tracks["outside_roi_pixel_count"], errors="coerce").fillna(0) > 0))
    if "exclusion_mask_overlap_count" in df_tracks.columns:
        technical_masks.append(("exclusion_mask_overlap", pd.to_numeric(df_tracks["exclusion_mask_overlap_count"], errors="coerce").fillna(0) > 0))
    if "suspected_multi_object_merge" in df_tracks.columns:
        technical_masks.append(("clear_multi_object_connected_component", df_tracks["suspected_multi_object_merge"].astype(bool)))
    if "unrecoverable_tracking_inconsistency" in df_tracks.columns:
        technical_masks.append(("unrecoverable_label_or_tracking_inconsistency", df_tracks["unrecoverable_tracking_inconsistency"].astype(bool)))

    if not technical_masks:
        technical_masks.append(("none", false))
    if not morphology_masks:
        morphology_masks.append(("none", false))
    if not reference_masks:
        reference_masks.append(("none", false))
    return technical_masks, morphology_masks, reference_masks


def flag_quality_tracks(df_tracks, cfg):
    """
    Annotate completed tracks without deleting data.

    In comparative mode, morphology outliers are warnings and remain in the
    primary ``technical_valid`` population. Reference-morphology filtering is a
    diagnostic subset only. Legacy columns are still emitted for older reports.
    """
    if df_tracks.empty:
        df_tracks["is_quality_track"] = pd.Series(dtype=bool)
        df_tracks["quality_flags"] = pd.Series(dtype=str)
        df_tracks["is_biological_candidate"] = pd.Series(dtype=bool)
        df_tracks["hard_flags"] = pd.Series(dtype=str)
        df_tracks["warning_flags"] = pd.Series(dtype=str)
        df_tracks["technical_valid"] = pd.Series(dtype=bool)
        df_tracks["technical_failure_reasons"] = pd.Series(dtype=str)
        df_tracks["morphology_warning"] = pd.Series(dtype=bool)
        df_tracks["morphology_warning_reasons"] = pd.Series(dtype=str)
        df_tracks["reference_morphology_pass"] = pd.Series(dtype=bool)
        df_tracks["segmentation_parameter_set"] = pd.Series(dtype=str)
        df_tracks["preprocessing_profile"] = pd.Series(dtype=str)
        df_tracks["analysis_mode"] = pd.Series(dtype=str)
        return df_tracks

    mode = str(cfg.get("ANALYSIS_MODE", "comparative")).strip().lower()
    n = len(df_tracks)
    if mode == "comparative":
        technical_masks, morphology_masks, reference_masks = _comparative_audit_masks(df_tracks, cfg)
        technical_strs, any_technical = _join_flag_names(
            [(name, mask) for name, mask in technical_masks if name != "none"], n)
        morphology_strs, any_morphology = _join_flag_names(
            [(name, mask) for name, mask in morphology_masks if name != "none"], n)
        reference_strs, any_reference_fail = _join_flag_names(
            [(name, mask) for name, mask in reference_masks if name != "none"], n)

        df_tracks["technical_valid"] = ~any_technical
        df_tracks["technical_failure_reasons"] = technical_strs
        df_tracks["morphology_warning"] = any_morphology & (~any_technical)
        df_tracks["morphology_warning_reasons"] = morphology_strs
        df_tracks["reference_morphology_pass"] = (~any_technical) & (~any_reference_fail)
        df_tracks["analysis_mode"] = mode
        df_tracks["segmentation_parameter_set"] = cfg.get("SEGMENTATION_PARAMETER_SET", "unspecified")
        df_tracks["preprocessing_profile"] = cfg.get("CLAHE_MODE", cfg.get("PREPROCESS_MODE", "unspecified"))

        df_tracks["quality_flags"] = [
            ",".join([x for x in [technical_strs[i], morphology_strs[i]] if x])
            for i in range(n)
        ]
        df_tracks["is_quality_track"] = df_tracks["reference_morphology_pass"].astype(bool)
        df_tracks["hard_flags"] = technical_strs
        df_tracks["warning_flags"] = morphology_strs
        df_tracks["is_biological_candidate"] = df_tracks["technical_valid"].astype(bool)
        df_tracks["has_warning_only"] = df_tracks["morphology_warning"].astype(bool)

        print(
            "  Comparative audit: "
            f"{int(df_tracks['technical_valid'].sum())} technical-valid, "
            f"{int(df_tracks['reference_morphology_pass'].sum())} reference-morphology, "
            f"{int(df_tracks['morphology_warning'].sum())} morphology-warning, "
            f"{int((~df_tracks['technical_valid']).sum())} technical failures out of {n} total"
        )
        return df_tracks

    strict_masks = []
    hard_masks = []
    warning_masks = []

    # Length
    if "total_3d_length_um" in df_tracks.columns:
        long_mask = df_tracks["total_3d_length_um"] > cfg.get("AUDIT_MAX_LENGTH_UM", 15.0)
        strict_masks.append(("long", long_mask))
        hard_masks.append(("long", long_mask))

    # Tortuosity
    if "tortuosity_3d" in df_tracks.columns:
        tort_mask = df_tracks["tortuosity_3d"] > cfg.get("AUDIT_MAX_TORTUOSITY", 1.5)
        strict_masks.append(("tortuous", tort_mask))
        hard_masks.append(("tortuous", tort_mask))

    # Thickness
    if "thickness_um" in df_tracks.columns:
        thick_mask = df_tracks["thickness_um"] > cfg.get("AUDIT_MAX_THICKNESS_UM", 2.0)
        extreme_thick_mask = df_tracks["thickness_um"] > cfg.get("AUDIT_EXTREME_THICKNESS_UM", 3.5)
        strict_masks.append(("thick", thick_mask))
        warning_masks.append(("thick", thick_mask & ~extreme_thick_mask))
        hard_masks.append(("extreme_thick", extreme_thick_mask))

    # Taper
    if "taper_ratio" in df_tracks.columns:
        taper_mask = df_tracks["taper_ratio"] > cfg.get("AUDIT_MAX_TAPER_RATIO", 1.5)
        extreme_taper_mask = df_tracks["taper_ratio"] > cfg.get("AUDIT_EXTREME_TAPER_RATIO", 3.0)
        strict_masks.append(("taper", taper_mask))
        warning_masks.append(("taper", taper_mask & ~extreme_taper_mask))
        hard_masks.append(("extreme_taper", extreme_taper_mask))

    # Single-slice / shallow
    if "n_slices" in df_tracks.columns:
        shallow_mask = df_tracks["n_slices"] < cfg.get("AUDIT_MIN_SLICES", 2)
        strict_masks.append(("single_slice", shallow_mask))
        hard_masks.append(("single_slice", shallow_mask))

    flag_strs, any_flagged = _join_flag_names(strict_masks, n)
    hard_strs, any_hard = _join_flag_names(hard_masks, n)
    warning_strs, any_warning = _join_flag_names(warning_masks, n)

    df_tracks["quality_flags"] = flag_strs
    df_tracks["is_quality_track"] = ~any_flagged
    df_tracks["hard_flags"] = hard_strs
    df_tracks["warning_flags"] = warning_strs
    df_tracks["is_biological_candidate"] = ~any_hard
    df_tracks["has_warning_only"] = any_warning & ~any_hard
    df_tracks["technical_valid"] = ~any_hard
    df_tracks["technical_failure_reasons"] = hard_strs
    df_tracks["morphology_warning"] = any_flagged & ~any_hard
    df_tracks["morphology_warning_reasons"] = warning_strs
    df_tracks["reference_morphology_pass"] = ~any_flagged
    df_tracks["segmentation_parameter_set"] = cfg.get("SEGMENTATION_PARAMETER_SET", "unspecified")
    df_tracks["preprocessing_profile"] = cfg.get("CLAHE_MODE", cfg.get("PREPROCESS_MODE", "unspecified"))
    df_tracks["analysis_mode"] = mode

    n_quality = int((~any_flagged).sum())
    n_flagged = int(any_flagged.sum())
    n_candidates = int((~any_hard).sum())
    n_hard = int(any_hard.sum())
    n_warning_only = int((any_warning & ~any_hard).sum())
    print(
        f"  Quality audit: {n_quality} strict quality, {n_candidates} biological candidates, "
        f"{n_hard} hard fails, {n_warning_only} warning-only out of {n} total"
    )

    return df_tracks


def export_comparative_track_tables(out_dir, track_summary, version_label=None):
    """
    Save comparative-mode output populations.

    Primary comparative analysis should use ``track_summary_technical_valid``.
    Reference morphology is diagnostic only and must not be treated as ground
    truth for WT-versus-mutant morphology.
    """
    if track_summary is None or track_summary.empty:
        return {}
    ensure_dir(out_dir)
    suffix = f"_{version_label}" if version_label else ""
    technical_valid = track_summary["technical_valid"].astype(bool) if "technical_valid" in track_summary.columns else pd.Series(True, index=track_summary.index)
    reference_pass = track_summary["reference_morphology_pass"].astype(bool) if "reference_morphology_pass" in track_summary.columns else technical_valid
    morphology_warning = track_summary["morphology_warning"].astype(bool) if "morphology_warning" in track_summary.columns else pd.Series(False, index=track_summary.index)
    tables = {
        f"track_summary_all{suffix}.csv": track_summary,
        f"track_summary_technical_valid{suffix}.csv": track_summary[technical_valid].copy(),
        f"track_summary_reference_morphology{suffix}.csv": track_summary[reference_pass].copy(),
        f"track_summary_morphology_warning{suffix}.csv": track_summary[morphology_warning].copy(),
        f"track_summary_technical_failures{suffix}.csv": track_summary[~technical_valid].copy(),
    }
    paths = {}
    for name, df in tables.items():
        path = os.path.join(out_dir, name)
        df.to_csv(path, index=False)
        paths[name] = path
    note = (
        "Morphology warnings are retained in the comparative population because "
        "they may represent genuine genotype-dependent phenotypes.\n"
        "Primary WT-versus-mutant table: track_summary_technical_valid*.csv\n"
        "Reference-morphology tables are diagnostic only.\n"
    )
    with open(os.path.join(out_dir, f"comparative_population_note{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(note)
    return paths


def summarize_comparative_population(df):
    """Return population-level summary metrics for sensitivity and blinded reports."""
    if df is None or df.empty:
        return {
            "total_technical_valid_count": 0,
            "technical_failure_fraction": 0.0,
            "morphology_warning_fraction": 0.0,
        }
    technical_valid = df["technical_valid"].astype(bool) if "technical_valid" in df.columns else pd.Series(True, index=df.index)
    morphology_warning = df["morphology_warning"].astype(bool) if "morphology_warning" in df.columns else pd.Series(False, index=df.index)
    valid = df[technical_valid].copy()
    denom = max(len(df), 1)
    out = {
        "total_technical_valid_count": int(len(valid)),
        "technical_failure_fraction": float((~technical_valid).sum() / denom),
        "morphology_warning_fraction": float(morphology_warning.sum() / denom),
    }
    metric_cols = {
        "total_3d_length_um": "length",
        "thickness_um": "width",
        "taper_ratio": "taper",
        "tortuosity_3d": "tortuosity",
        "volume_um3": "volume",
        "z_span_um": "z_span",
        "pitch_deg": "pitch",
        "nearest_neighbor_um": "nearest_neighbor_distance",
    }
    for col, label in metric_cols.items():
        if col in valid.columns and not valid.empty:
            vals = pd.to_numeric(valid[col], errors="coerce").dropna()
            out[f"median_{label}"] = float(vals.median()) if not vals.empty else np.nan
            out[f"mean_{label}"] = float(vals.mean()) if not vals.empty else np.nan
        else:
            out[f"median_{label}"] = np.nan
            out[f"mean_{label}"] = np.nan
    for flag_col, label in [
        ("technical_failure_reasons", "technical_failure"),
        ("morphology_warning_reasons", "morphology_warning"),
    ]:
        if flag_col in df.columns:
            reasons = df[flag_col].fillna("").astype(str)
            for token in sorted(set(",".join(reasons).split(",")) - {""}):
                out[f"{label}_{token}_fraction"] = float(reasons.map(lambda s, t=token: t in [p for p in s.split(",") if p]).sum() / denom)
    return out


def compare_preset_track_summaries(preset_tables):
    """
    Compare already-generated track summaries from comparative presets.

    ``preset_tables`` maps preset names to annotated track-summary DataFrames.
    Object identity is approximated by rounded centroid and z-span fields when
    available; this function never asserts which biological distribution is
    correct.
    """
    summaries = []
    identity_sets = {}
    for name, df in preset_tables.items():
        rec = {"preset": name, **summarize_comparative_population(df)}
        summaries.append(rec)
        if df is None or df.empty:
            identity_sets[name] = set()
            continue
        cols = [c for c in ("centroid_x", "centroid_y", "z_min", "z_max") if c in df.columns]
        if len(cols) >= 2:
            technical_valid = df["technical_valid"].astype(bool) if "technical_valid" in df.columns else pd.Series(True, index=df.index)
            valid = df[technical_valid].copy()
            keys = set(tuple(round(float(row[c]), 1) for c in cols) for _, row in valid.iterrows())
        elif "track_id" in df.columns:
            technical_valid = df["technical_valid"].astype(bool) if "technical_valid" in df.columns else pd.Series(True, index=df.index)
            keys = set(df.loc[technical_valid, "track_id"].astype(str))
        else:
            keys = set(df.index.astype(str))
        identity_sets[name] = keys
    all_names = list(preset_tables.keys())
    shared = set.intersection(*(identity_sets[n] for n in all_names)) if all_names else set()
    permissive = identity_sets.get("permissive", set())
    conservative = identity_sets.get("conservative", set())
    selected = identity_sets.get("selected", set())
    sensitivity = {
        "detected_by_all_presets": len(shared),
        "detected_only_by_permissive": len(permissive - set.union(*(identity_sets[n] for n in all_names if n != "permissive"))) if permissive and len(all_names) > 1 else 0,
        "lost_only_by_conservative": len((selected | permissive | identity_sets.get("intermediate", set())) - conservative),
        "preset_count": len(all_names),
    }
    return pd.DataFrame(summaries), sensitivity


def assign_blinded_dataset_ids(manifest_df, seed=560123):
    """
    Return a manifest copy with anonymized dataset IDs.

    Genotype labels are preserved only in a separate reveal table. Segmentation
    code should consume the blinded manifest and not the reveal table.
    """
    rng = np.random.default_rng(seed)
    manifest = manifest_df.copy()
    order = np.arange(len(manifest))
    rng.shuffle(order)
    blinded_ids = [f"anon_{i + 1:03d}" for i in range(len(manifest))]
    manifest["blinded_dataset_id"] = ""
    for blinded_id, idx in zip(blinded_ids, order):
        manifest.loc[manifest.index[idx], "blinded_dataset_id"] = blinded_id
    reveal_cols = [c for c in ("blinded_dataset_id", "genotype", "group", "dataset_label") if c in manifest.columns]
    reveal = manifest[reveal_cols].copy()
    blinded = manifest.drop(columns=[c for c in ("genotype", "group", "dataset_label") if c in manifest.columns])
    return blinded, reveal


def make_blinded_review_sheet(crop_records, candidates=None):
    """Create blank manual-review rows without exposing genotype labels."""
    candidates = candidates or ["comparative_selected"]
    rows = []
    manual_cols = [
        "true_detection", "missed_nucleus", "split_nucleus", "merged_nuclei",
        "tissue_edge_false_positive", "puncta_ring_false_positive", "uncertain",
        "reviewer_notes",
    ]
    for crop in crop_records:
        for cand in candidates:
            row = {
                "blinded_dataset_id": crop.get("blinded_dataset_id", ""),
                "crop_id": crop.get("crop_id", ""),
                "candidate": cand,
                "z_index": crop.get("z_index", ""),
                "x0": crop.get("x0", ""),
                "y0": crop.get("y0", ""),
                "x1": crop.get("x1", ""),
                "y1": crop.get("y1", ""),
            }
            row.update({col: "" for col in manual_cols})
            rows.append(row)
    return pd.DataFrame(rows)


def differential_error_indicators(summary_by_group, warning_threshold=0.15):
    """
    Compare segmentation error indicators between anonymized groups.

    The function reports differences only; it does not correct or normalize
    biological distributions to make groups agree.
    """
    df = pd.DataFrame(summary_by_group).copy()
    if df.empty or "group" not in df.columns:
        return df, []
    indicators = [
        "technical_failure_fraction", "morphology_warning_fraction",
        "short_fragment_fraction", "suspected_merge_fraction",
        "branch_network_fraction", "roi_edge_fraction",
        "permissive_only_detection_fraction", "conservative_loss_fraction",
    ]
    warnings = []
    for col in indicators:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() >= 2 and float(vals.max() - vals.min()) >= warning_threshold:
            warnings.append(f"{col} differs by {float(vals.max() - vals.min()):.3f} across anonymized groups")
    return df, warnings


def export_outlier_audit(out_dir, df_tracks, cfg):
    """
    Generate the outlier_audit/ subfolder with per-class CSVs and a summary,
    mirroring the output of audit_sperm_outliers.py but done automatically.
    """
    if df_tracks.empty:
        return

    audit_dir = os.path.join(out_dir, "outlier_audit")
    ensure_dir(audit_dir)

    # Robust column matching
    length_col = next((c for c in ["total_3d_length_um", "length_3d_um_est", "max_length_um"] if c in df_tracks.columns), None)
    tort_col = next((c for c in ["tortuosity_3d", "tortuosity"] if c in df_tracks.columns), None)
    thick_col = next((c for c in ["thickness_um", "effective_thickness_um", "median_width_um"] if c in df_tracks.columns), None)
    taper_col = next((c for c in ["taper_ratio", "morphological_taper_ratio"] if c in df_tracks.columns), None)
    nslices_col = next((c for c in ["n_slices", "n_detections"] if c in df_tracks.columns), None)

    # Apply thresholds
    thresh_len = cfg.get("AUDIT_MAX_LENGTH_UM", 15.0)
    thresh_tort = cfg.get("AUDIT_MAX_TORTUOSITY", 1.5)
    thresh_thick = cfg.get("AUDIT_MAX_THICKNESS_UM", 2.0)
    thresh_taper = cfg.get("AUDIT_MAX_TAPER_RATIO", 1.5)
    thresh_slices = cfg.get("AUDIT_MIN_SLICES", 2)

    outlier_sets = {}

    if length_col:
        long_df = df_tracks[df_tracks[length_col] > thresh_len].sort_values(length_col, ascending=False)
        long_df.to_csv(os.path.join(audit_dir, "outliers_long.csv"), index=False)
        outlier_sets["long"] = len(long_df)

    if tort_col:
        tort_df = df_tracks[df_tracks[tort_col] > thresh_tort].sort_values(tort_col, ascending=False)
        tort_df.to_csv(os.path.join(audit_dir, "outliers_tortuous.csv"), index=False)
        outlier_sets["tortuous"] = len(tort_df)

    if thick_col:
        thick_df = df_tracks[df_tracks[thick_col] > thresh_thick].sort_values(thick_col, ascending=False)
        thick_df.to_csv(os.path.join(audit_dir, "outliers_thick.csv"), index=False)
        outlier_sets["thick"] = len(thick_df)

    if taper_col:
        taper_df = df_tracks[df_tracks[taper_col] > thresh_taper].sort_values(taper_col, ascending=False)
        taper_df.to_csv(os.path.join(audit_dir, "outliers_taper.csv"), index=False)
        outlier_sets["taper"] = len(taper_df)

    if nslices_col:
        single_df = df_tracks[df_tracks[nslices_col] < thresh_slices].sort_values(nslices_col, ascending=True)
        single_df.to_csv(os.path.join(audit_dir, "outliers_single_slice.csv"), index=False)
        outlier_sets["single_slice"] = len(single_df)

    if "is_biological_candidate" in df_tracks.columns:
        candidate_df = df_tracks[df_tracks["is_biological_candidate"]].copy()
        hard_fail_df = df_tracks[~df_tracks["is_biological_candidate"]].copy()
        warning_df = df_tracks[df_tracks.get("has_warning_only", False)].copy()
        candidate_df.to_csv(os.path.join(audit_dir, "biological_candidates.csv"), index=False)
        hard_fail_df.to_csv(os.path.join(audit_dir, "hard_fails.csv"), index=False)
        warning_df.to_csv(os.path.join(audit_dir, "warning_only.csv"), index=False)
        outlier_sets["biological_candidates"] = len(candidate_df)
        outlier_sets["hard_fails"] = len(hard_fail_df)
        outlier_sets["warning_only"] = len(warning_df)
    if "technical_valid" in df_tracks.columns:
        tech_valid_df = df_tracks[df_tracks["technical_valid"]].copy()
        tech_fail_df = df_tracks[~df_tracks["technical_valid"]].copy()
        morph_warn_df = df_tracks[df_tracks["morphology_warning"]].copy() if "morphology_warning" in df_tracks.columns else pd.DataFrame()
        ref_df = df_tracks[df_tracks["reference_morphology_pass"]].copy() if "reference_morphology_pass" in df_tracks.columns else pd.DataFrame()
        tech_valid_df.to_csv(os.path.join(audit_dir, "technical_valid.csv"), index=False)
        tech_fail_df.to_csv(os.path.join(audit_dir, "technical_failures.csv"), index=False)
        morph_warn_df.to_csv(os.path.join(audit_dir, "morphology_warnings.csv"), index=False)
        ref_df.to_csv(os.path.join(audit_dir, "reference_morphology.csv"), index=False)
        outlier_sets["technical_valid"] = len(tech_valid_df)
        outlier_sets["technical_failures"] = len(tech_fail_df)
        outlier_sets["morphology_warnings"] = len(morph_warn_df)
        outlier_sets["reference_morphology"] = len(ref_df)

    # All flagged
    if "is_quality_track" in df_tracks.columns:
        flagged = df_tracks[~df_tracks["is_quality_track"]].copy()
    else:
        flagged = pd.DataFrame()

    if not flagged.empty:
        flagged.to_csv(os.path.join(audit_dir, "outliers_all_flagged.csv"), index=False)

    # Summary
    lines = []
    lines.append("OUTLIER AUDIT SUMMARY (Auto-generated)")
    lines.append("=" * 60)
    lines.append(f"Tracks total: {len(df_tracks)}")
    if "is_quality_track" in df_tracks.columns:
        lines.append(f"Quality tracks: {int(df_tracks['is_quality_track'].sum())}")
        lines.append(f"Flagged tracks: {int((~df_tracks['is_quality_track']).sum())}")
    if "is_biological_candidate" in df_tracks.columns:
        lines.append(f"Biological candidate tracks: {int(df_tracks['is_biological_candidate'].sum())}")
        lines.append(f"Hard-fail tracks: {int((~df_tracks['is_biological_candidate']).sum())}")
    if "has_warning_only" in df_tracks.columns:
        lines.append(f"Warning-only tracks: {int(df_tracks['has_warning_only'].sum())}")
    if "technical_valid" in df_tracks.columns:
        lines.append(f"Technical-valid tracks: {int(df_tracks['technical_valid'].sum())}")
        lines.append(f"Technical-failure tracks: {int((~df_tracks['technical_valid']).sum())}")
    if "morphology_warning" in df_tracks.columns:
        lines.append(f"Morphology-warning tracks retained in comparative population: {int(df_tracks['morphology_warning'].sum())}")
    if "reference_morphology_pass" in df_tracks.columns:
        lines.append(f"Reference-morphology diagnostic subset: {int(df_tracks['reference_morphology_pass'].sum())}")
    lines.append("Morphology warnings are retained in the comparative population because they may represent genuine genotype-dependent phenotypes.")
    lines.append("")
    lines.append("Thresholds:")
    lines.append(f"  length > {thresh_len}")
    lines.append(f"  tortuosity > {thresh_tort}")
    lines.append(f"  thickness > {thresh_thick}")
    lines.append(f"  taper > {thresh_taper}")
    lines.append(f"  extreme thickness > {cfg.get('AUDIT_EXTREME_THICKNESS_UM', 3.5)}")
    lines.append(f"  extreme taper > {cfg.get('AUDIT_EXTREME_TAPER_RATIO', 3.0)}")
    lines.append(f"  n_slices < {thresh_slices}")
    lines.append("")
    lines.append("Counts:")
    for name, count in outlier_sets.items():
        lines.append(f"  {name:20s}: {count}")
    lines.append(f"  {'all_flagged':20s}: {len(flagged)}")

    with open(os.path.join(audit_dir, "outlier_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Outlier audit exported to: {audit_dir}")


def export_post_detection_qc(out_dir, df_detections, df_tracks):
    """
    Save compact diagnostics for the post-detection 3D processing stage.

    These metrics make GUI and CLI 3D runs easier to audit: how detections were
    linked, how many tracks are single-slice, and which quality flags dominate.
    """
    try:
        ensure_dir(out_dir)
        lines = []
        lines.append("POST-DETECTION 3D QC")
        lines.append("=" * 72)

        n_det = int(len(df_detections)) if df_detections is not None else 0
        n_tracks = int(len(df_tracks)) if df_tracks is not None else 0
        lines.append(f"2D detections: {n_det}")
        lines.append(f"3D tracks: {n_tracks}")

        if df_detections is not None and not df_detections.empty and "track_link_method" in df_detections.columns:
            method_counts = df_detections["track_link_method"].fillna("unknown").value_counts()
            lines.append("\nDetection link methods:")
            for method, count in method_counts.items():
                lines.append(f"  {method}: {int(count)}")
            if "track_link_distance_um" in df_detections.columns:
                linked_dist = pd.to_numeric(df_detections["track_link_distance_um"], errors="coerce").dropna()
                if not linked_dist.empty:
                    lines.append(f"Median link distance um: {float(linked_dist.median()):.3f}")
                    lines.append(f"95th percentile link distance um: {float(linked_dist.quantile(0.95)):.3f}")

        if df_tracks is not None and not df_tracks.empty:
            if "n_slices" in df_tracks.columns:
                single_frac = float((df_tracks["n_slices"] <= 1).mean() * 100)
                lines.append(f"\nSingle-slice tracks: {single_frac:.1f}%")
                lines.append(f"Median n_slices: {float(df_tracks['n_slices'].median()):.2f}")
            if "total_3d_length_um" in df_tracks.columns:
                lines.append(f"Median 3D length um: {float(df_tracks['total_3d_length_um'].median()):.3f}")
            if "z_span_um" in df_tracks.columns:
                lines.append(f"Median Z-span um: {float(df_tracks['z_span_um'].median()):.3f}")
            if "quality_flags" in df_tracks.columns:
                flags = df_tracks["quality_flags"].fillna("").astype(str)
                flag_counts = {
                    "quality": int((flags == "").sum()),
                    "all_flagged": int((flags != "").sum()),
                    "long": int(flags.str.contains("long", regex=False).sum()),
                    "tortuous": int(flags.str.contains("tortuous", regex=False).sum()),
                    "thick": int(flags.str.contains("thick", regex=False).sum()),
                    "taper": int(flags.str.contains("taper", regex=False).sum()),
                    "single_slice": int(flags.str.contains("single_slice", regex=False).sum()),
                }
                lines.append("\nQuality / outlier counts:")
                for key in ["quality", "all_flagged", "long", "tortuous", "thick", "taper", "single_slice"]:
                    lines.append(f"  {key}: {flag_counts.get(key, 0)}")
            if "is_biological_candidate" in df_tracks.columns:
                lines.append("\nBiological candidate tier:")
                lines.append(f"  biological_candidates: {int(df_tracks['is_biological_candidate'].sum())}")
                lines.append(f"  hard_fails: {int((~df_tracks['is_biological_candidate']).sum())}")
            if "has_warning_only" in df_tracks.columns:
                lines.append(f"  warning_only: {int(df_tracks['has_warning_only'].sum())}")
            if "track_stop_reason" in df_tracks.columns:
                stopped = df_tracks["track_stop_reason"].fillna("").astype(str)
                stopped = stopped[stopped != ""]
                lines.append(f"\nTracks with recorded stop reason: {len(stopped)}")
                if len(stopped):
                    for reason, count in stopped.value_counts().head(10).items():
                        lines.append(f"  {reason}: {int(count)}")

        qc_path = os.path.join(out_dir, "post_detection_qc.txt")
        with open(qc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Post-detection QC exported to: {qc_path}")
    except Exception as e:
        print(f"  WARNING: Could not export post-detection QC: {e}")


def get_audit_flag_counts(df_tracks, cfg=None):
    """Return consistent audit category counts for reports, regardless of GUI/report settings."""
    counts = {
        "long": 0, "tortuous": 0, "thick": 0, "taper": 0, "single_slice": 0,
        "all_flagged": 0, "quality": 0, "shape_outliers": 0,
        "biological_candidates": 0, "hard_fails": 0, "warning_only": 0,
    }
    if df_tracks is None or df_tracks.empty:
        return counts

    if "quality_flags" in df_tracks.columns:
        flags = df_tracks["quality_flags"].fillna("").astype(str)
        split_flags = flags.str.split(',')
        for key in ["long", "tortuous", "thick", "taper", "single_slice"]:
            counts[key] = int(split_flags.apply(lambda items: key in items if isinstance(items, list) else False).sum())
    elif cfg is not None:
        if "total_3d_length_um" in df_tracks.columns:
            counts["long"] = int((df_tracks["total_3d_length_um"] > cfg.get("AUDIT_MAX_LENGTH_UM", 15.0)).sum())
        if "tortuosity_3d" in df_tracks.columns:
            counts["tortuous"] = int((df_tracks["tortuosity_3d"] > cfg.get("AUDIT_MAX_TORTUOSITY", 1.5)).sum())
        if "thickness_um" in df_tracks.columns:
            counts["thick"] = int((df_tracks["thickness_um"] > cfg.get("AUDIT_MAX_THICKNESS_UM", 2.0)).sum())
        if "taper_ratio" in df_tracks.columns:
            counts["taper"] = int((df_tracks["taper_ratio"] > cfg.get("AUDIT_MAX_TAPER_RATIO", 1.5)).sum())
        if "n_slices" in df_tracks.columns:
            counts["single_slice"] = int((df_tracks["n_slices"] < cfg.get("AUDIT_MIN_SLICES", 2)).sum())

    if "is_quality_track" in df_tracks.columns:
        counts["quality"] = int(df_tracks["is_quality_track"].sum())
        counts["all_flagged"] = int((~df_tracks["is_quality_track"]).sum())
    else:
        counts["quality"] = len(df_tracks)
        counts["all_flagged"] = 0

    counts["shape_outliers"] = max(counts["all_flagged"] - counts["single_slice"], 0)
    if "is_biological_candidate" in df_tracks.columns:
        counts["biological_candidates"] = int(df_tracks["is_biological_candidate"].sum())
        counts["hard_fails"] = int((~df_tracks["is_biological_candidate"]).sum())
    else:
        counts["biological_candidates"] = counts["quality"]
        counts["hard_fails"] = counts["all_flagged"]
    if "has_warning_only" in df_tracks.columns:
        counts["warning_only"] = int(df_tracks["has_warning_only"].sum())
    return counts


# =============================================================================
# PROCESS ONE IMAGE
# =============================================================================

def process_one_image(image_path, cfg, output_dir):
    """
    Runs the complete segmentation-and-measurement pipeline on a single Z-slice image.

    This function is the fundamental unit of the batch processing loop.  For each
    input image it:

    1. Loads the raw image (multi-format with TIFF / OpenCV fallback).
    2. Selects the correct Z-slice from a multi-plane volume if needed.
    3. Calls :func:`segment_slice` to produce the binary mask, ridge map, and skeleton.
    4. Calls :func:`measure_spermatids` to extract per-cell biometrics.
    5. Optionally validates or post-filters results with shape-quality checks.
    6. Saves the colour overlay PNG, detail figure, mask TIFs, and per-slice CSV.
    7. Optionally opens the preview window for interactive QC.

    Args:
        image_path (str): Absolute path to the input image file.
        cfg (dict): Full pipeline configuration dictionary.  Key fields:
            - ``UM_PER_PX_XY`` - physical scale factor.
            - ``Z_INDEX`` - which plane to extract from a Z-stack.
            - ``OUTPUT_DIR`` - top-level output folder.
            - ``SHOW_PREVIEW_WINDOW`` - whether to open the preview GUI.
            - ``SAVE_MASK_TIFS``, ``SAVE_LABEL_TIFS`` - optional TIF outputs.
            - ``SAVE_DETAIL_FIGURE`` - whether to save 3-panel figure.
        output_dir (str): Destination directory for this image's outputs.

    Returns:
        tuple[list[dict], dict]:
            - ``results`` - list of per-spermatid measurement dicts.
            - ``seg``     - full segmentation dict from :func:`segment_slice`
              including ``mask_clean``, ``skel_pruned``, ``skel_label``, etc.
    """
    ensure_dir(output_dir)
    overlay_dir = os.path.join(output_dir, "overlays")
    debug_dir   = os.path.join(output_dir, "debug")
    ensure_dir(overlay_dir)
    if cfg["SAVE_DEBUG_IMAGES"]:
        ensure_dir(debug_dir)

    z_idx   = extract_z_index(image_path)
    img_raw = robust_imread(image_path)
    img_2d  = ensure_2d_image(img_raw, os.path.basename(image_path))
    roi_mask = None
    exclusion_mask = None
    if cfg.get("ROI_MASK_PATH"):
        roi_mask = load_roi_mask_file(cfg["ROI_MASK_PATH"], expected_shape=img_2d.shape)
    if cfg.get("EXCLUSION_MASK_PATH"):
        exclusion_mask = load_roi_mask_file(cfg["EXCLUSION_MASK_PATH"], expected_shape=img_2d.shape)
    preprocess_context = build_stack_preprocess_context([image_path], roi_mask, cfg, exclusion_mask=exclusion_mask)
    save_stack_preprocess_context(preprocess_context, output_dir)
    print(f"\nProcessing: {os.path.basename(image_path)}")

    t0      = time.time()
    unet_context = np.stack([img_2d.astype(np.float32)] * 3, axis=0)
    seg     = segment_slice(img_2d, cfg, z_idx=z_idx,
                            debug_dir=debug_dir if cfg["SAVE_DEBUG_IMAGES"] else None,
                            roi_mask=roi_mask,
                            preprocess_context=preprocess_context,
                            exclusion_mask=exclusion_mask,
                            unet_context_stack=unet_context)
    meas    = measure_spermatids(seg, cfg)
    results = meas["results"]
    elapsed = time.time() - t0

    um = cfg["UM_PER_PX_XY"]
    print(f"  Detected: {len(results)} spermatids  ({elapsed:.1f}s)")
    if results:
        ls = [r["length_px_geodesic"]*um for r in results]
        print(f"  Geodesic length um: median={np.median(ls):.2f}  max={max(ls):.2f}")

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
    if cfg.get("UNET_SAVE_PROBABILITY_MAPS", True):
        if seg.get("unet_probability") is not None and np.any(seg.get("unet_probability")):
            tifffile.imwrite(os.path.join(output_dir, f"z{z_idx:02d}_unet_probability.tif"),
                             seg["unet_probability"].astype(np.float32))
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
    """
    Orchestrates the entire end-to-end biological data processing engine for batch image iterations.

    Iterates over all `.tif`/`.tiff` files across defined spatial bounds, extracting single
    segmentation matrices and triggering the 3D concatenation models upon directory exhaustion.

    Args:
        cfg (dict): Active session configuration parameter dictionary.
    """
    ensure_dir(cfg["OUTPUT_DIR"])
    overlay_dir = os.path.join(cfg["OUTPUT_DIR"], "overlays")
    debug_dir   = os.path.join(cfg["OUTPUT_DIR"], "debug")
    ensure_dir(overlay_dir)
    if cfg["SAVE_DEBUG_IMAGES"]:
        ensure_dir(debug_dir)

    files, z_indices = load_batch_files(cfg["INPUT_DIR"], cfg["FILE_PATTERN"])
    files_by_z = {int(z): f for f, z in zip(files, z_indices)}
    um         = cfg["UM_PER_PX_XY"]
    roi_mask = None
    exclusion_mask = None
    first_img = ensure_2d_image(robust_imread(files[0]), os.path.basename(files[0]))
    if cfg.get("ROI_MASK_PATH"):
        roi_mask = load_roi_mask_file(cfg["ROI_MASK_PATH"], expected_shape=first_img.shape)
        roi_out = os.path.join(cfg["OUTPUT_DIR"], "roi_mask_used.tif")
        tifffile.imwrite(roi_out, roi_mask.astype(np.uint8) * 255)
        print(f"Using ROI mask: {cfg['ROI_MASK_PATH']}")
        print(f"Saved ROI copy: {roi_out}")
    if cfg.get("EXCLUSION_MASK_PATH"):
        exclusion_mask = load_roi_mask_file(cfg["EXCLUSION_MASK_PATH"], expected_shape=first_img.shape)
        excl_out = os.path.join(cfg["OUTPUT_DIR"], "exclusion_mask_used.tif")
        tifffile.imwrite(excl_out, exclusion_mask.astype(np.uint8) * 255)
        print(f"Using exclusion mask: {cfg['EXCLUSION_MASK_PATH']}")
        print(f"Saved exclusion copy: {excl_out}")
    preprocess_context = build_stack_preprocess_context(files, roi_mask, cfg, exclusion_mask=exclusion_mask)
    save_stack_preprocess_context(preprocess_context, cfg["OUTPUT_DIR"])
    print(f"Stack preprocessing: profile={preprocess_context.selected_clahe_profile}, clip={preprocess_context.selected_clahe_clip}, norm=({preprocess_context.normalization_low:.3f}, {preprocess_context.normalization_high:.3f}), sampled_z={preprocess_context.sampled_z_indices}")
    all_rows   = []
    summaries  = []
    t_batch    = time.time()
    t_slices   = []
    max_proj_raw = None
    max_proj_ov = None
    slice_cache = {}

    for idx_i, (fpath, z_idx) in enumerate(zip(files, z_indices)):
        t0 = time.time()
        print(f"\n[{idx_i+1}/{len(files)}]  Z={z_idx:02d}  {os.path.basename(fpath)}")
        img_raw = robust_imread(fpath)
        img_2d  = ensure_2d_image(img_raw, os.path.basename(fpath))
        unet_context = _make_unet_context_from_paths(files_by_z, z_idx)
        seg     = segment_slice(img_2d, cfg, z_idx=z_idx,
                                debug_dir=debug_dir if cfg["SAVE_DEBUG_IMAGES"] else None,
                                roi_mask=roi_mask,
                                preprocess_context=preprocess_context,
                                exclusion_mask=exclusion_mask,
                                unet_context_stack=unet_context)
        meas    = measure_spermatids(seg, cfg)
        results = meas["results"]
        skel_label = meas["skel_label"]
        ls_um   = [r["length_px_geodesic"]*um for r in results]
        ws_um   = [r["width_px"]*um for r in results]

        t_s = time.time() - t0
        t_slices.append(t_s)
        eta = (len(files) - idx_i - 1) * float(np.mean(t_slices))
        print(f"  N={len(results)}", end="")
        if ls_um:
            print(f"  med_len={np.median(ls_um):.2f}um", end="")
        print(f"  {t_s:.1f}s  ETA {eta:.0f}s")

        all_rows.extend(rows_from_results(results, z_idx, um))
        summaries.append({
            "z_slice":          z_idx,
            "n_spermatids":     len(results),
            "mean_length_um":   round(float(np.mean(ls_um)),   3) if ls_um else 0,
            "median_length_um": round(float(np.median(ls_um)), 3) if ls_um else 0,
            "mean_width_um":    round(float(np.mean(ws_um)),   3) if ws_um  else 0,
        })

        overlay_rgb = make_overlay(img_2d, skel_label)
        if cfg["SAVE_OVERLAYS"]:
            # Create side-by-side panel: [Original | Overlay]
            orig_rgb = (normalize_display(img_2d) * 255).astype(np.uint8)
            # if grayscale, convert to RGB for hstack
            if orig_rgb.ndim == 2:
                orig_rgb = np.stack([orig_rgb]*3, axis=-1)

            panel = np.hstack([orig_rgb, overlay_rgb])
            _imwrite(os.path.join(overlay_dir, f"z{z_idx:02d}_panel.png"), panel)

            if max_proj_raw is None:
                max_proj_raw = img_2d.copy().astype(np.float32)
                max_proj_ov = overlay_rgb.copy().astype(np.float32)
            else:
                max_proj_raw = np.maximum(max_proj_raw, img_2d.astype(np.float32))
                max_proj_ov = np.maximum(max_proj_ov, overlay_rgb.astype(np.float32))
            if cfg.get("DO_TRACKING", True):
                slice_cache[int(z_idx)] = {
                    "image": img_2d.copy(),
                    "skel_label": skel_label.copy().astype(np.int32),
                }

        if cfg["SAVE_DETAIL_FIGURE"]:
            save_detail_figure(img_2d, overlay_rgb, results,
                               os.path.join(overlay_dir, f"z{z_idx:02d}_detail.png"),
                               z_idx, um)
        if cfg["SAVE_MASK_TIFS"]:
            tifffile.imwrite(os.path.join(cfg["OUTPUT_DIR"], f"z{z_idx:02d}_mask.tif"),
                             (seg["mask_clean"] & roi_mask if roi_mask is not None else seg["mask_clean"]).astype(np.uint8) * 255)
        if cfg.get("UNET_SAVE_PROBABILITY_MAPS", True):
            if seg.get("unet_probability") is not None and np.any(seg.get("unet_probability")):
                tifffile.imwrite(os.path.join(cfg["OUTPUT_DIR"], f"z{z_idx:02d}_unet_probability.tif"),
                                 seg["unet_probability"].astype(np.float32))
        if cfg["SAVE_LABEL_TIFS"]:
            tifffile.imwrite(os.path.join(cfg["OUTPUT_DIR"], f"z{z_idx:02d}_skel_labels.tif"),
                             skel_label.astype(np.uint16))

    df     = pd.DataFrame(all_rows)
    df_sum = pd.DataFrame(summaries)
    if cfg["SAVE_OVERLAYS"] and max_proj_raw is not None and max_proj_ov is not None:
        raw_p = (normalize_display(max_proj_raw.astype(np.uint16)) * 255).astype(np.uint8)
        if raw_p.ndim == 2:
            raw_p = np.stack([raw_p] * 3, axis=-1)
        ov_p = np.clip(max_proj_ov, 0, 255).astype(np.uint8)
        global_panel = np.hstack([raw_p, ov_p])
        _imwrite(os.path.join(cfg["OUTPUT_DIR"], "global_z_projection.png"), global_panel)

    df.to_csv(    os.path.join(cfg["OUTPUT_DIR"], f"spermatid_measurements_{_VERSION}.csv"), index=False)
    df_sum.to_csv(os.path.join(cfg["OUTPUT_DIR"], f"slice_summary_{_VERSION}.csv"), index=False)

    # Robust initialization for reporting
    df_trk = None
    ts = None

    if cfg["DO_TRACKING"] and not df.empty:
        df_trk, ts = track_across_slices(df, cfg)
        df_trk.to_csv(
            os.path.join(cfg["OUTPUT_DIR"], f"measurements_with_tracks_{_VERSION}.csv"),
            index=False)

        # Auto quality audit
        ts = flag_quality_tracks(ts, cfg)
        ts.to_csv(
            os.path.join(cfg["OUTPUT_DIR"], f"track_summary_{_VERSION}.csv"),
            index=False)

        # Quality-only CSV
        ts_quality = ts[ts["is_quality_track"]].copy()
        ts_quality.to_csv(
            os.path.join(cfg["OUTPUT_DIR"], f"track_summary_quality_{_VERSION}.csv"),
            index=False)
        ts_candidates = ts[ts["is_biological_candidate"]].copy() if "is_biological_candidate" in ts.columns else ts_quality
        ts_candidates.to_csv(
            os.path.join(cfg["OUTPUT_DIR"], f"track_summary_biological_candidates_{_VERSION}.csv"),
            index=False)
        export_comparative_track_tables(cfg["OUTPUT_DIR"], ts, _VERSION)

        # Quality-coded overlays: green = audit-passed, red = audit-flagged.
        if cfg["SAVE_OVERLAYS"]:
            export_quality_overlays(cfg["OUTPUT_DIR"], slice_cache, df_trk, ts)

        # Export outlier_audit/ folder
        export_outlier_audit(cfg["OUTPUT_DIR"], ts, cfg)
        export_post_detection_qc(cfg["OUTPUT_DIR"], df_trk, ts)

    # --- Reporting Phase (CLI/Batch) ---
    print(f"\nGenerating final reports in {cfg['OUTPUT_DIR']}...")
    generate_batch_report(cfg["OUTPUT_DIR"], df, df_sum, um, ts)
    generate_excel_report(cfg["OUTPUT_DIR"], df, df_sum, ts)

    total = time.time() - t_batch
    print(f"\n{'='*55}")
    print(f"{_VERSION} DONE | {len(files)} slices | {total:.1f}s")
    print(f"Saved to: {cfg['OUTPUT_DIR']}")
    print(df_sum.to_string(index=False))


def write_error_log(out_dir, component, message):
    """
    Writes a persistent error log to report_generation_errors.txt in the output directory.
    """
    try:
        import os as _os
        import time as _time
        from datetime import datetime as _dt
        log_path = _os.path.join(out_dir, "report_generation_errors.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{_dt.now().strftime('%Y-%m-%d %H:%M:%S')}] COMPONENT: {component}\n")
            f.write(f"MESSAGE:\n{message}\n")
            f.write("-" * 80 + "\n")
    except Exception:
        pass


def generate_excel_report(out_dir, df, df_summary, df_tracks=None):
    """
    Generates a multi-tab Excel workbook with formatted data, summary statistics,
    embedded chart images, and source-data hyperlinks.

    Workbook structure
    ------------------
    - **Batch_Summary** - one row per Z-slice with detection counts, mean/median
      length, and total area.  Includes an embedded histogram image and conditional
      formatting for high/low detection slices.
    - **3D_Morphometrics** - track-level 3D metrics exported from
      :func:`track_across_slices`, with an embedded 3D length distribution plot.
    - **Raw_2D_Detections** - full per-spermatid measurement table matching the CSV
      export, with number formatting and frozen header pane.
    - **Statistics_Summary** - descriptive statistics (mean, median, std, IQR,
      percentiles) for the primary biometric columns.

    All sheets include a hyperlink from cell A1 back to the source Excel file so
    that clicking within PowerPoint linked charts opens the correct workbook row.

    Biological interpretation
    -------------------------
    The statistical summary sheet is designed to be directly copy-pasteable into
    lab reports.  The IQR (Interquartile Range = Q75 - Q25) is reported because
    spermatid length distributions are often right-skewed and IQR is more robust
    than standard deviation in these cases.

    Args:
        out_dir (str): Top-level analysis output directory.  The workbook is saved
            as ``<out_dir>/batch_analysis_results_<ver>.xlsx``.
        df (pd.DataFrame): Per-spermatid measurement table (all Z-slices combined).
        df_summary (pd.DataFrame): Per-slice summary statistics.
        df_tracks (pd.DataFrame, optional): 3D track table from
            :func:`track_across_slices`.  ``None`` if tracking was not run.
    """
    excel_path = os.path.join(out_dir, f"batch_analysis_results_{_VERSION}.xlsx")
    plot_dir = os.path.join(out_dir, "summary_plots")
    print(f"Generating Interactive Excel Audit: {excel_path} ...")

    try:
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            workbook  = writer.book

            def excel_col_letter(frame, column_name):
                """Return the Excel column letter for a DataFrame column."""
                idx = frame.columns.get_loc(column_name)
                letters = ""
                idx += 1
                while idx:
                    idx, rem = divmod(idx - 1, 26)
                    letters = chr(65 + rem) + letters
                return letters

            # Formats
            bold = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            num_fmt = workbook.add_format({'num_format': '0.00'})

            # --- Sheet 1: Population Summary (DYNAMIC FORMULAS) --- [FIRST TAB]
            ws_sum = workbook.add_worksheet('Population_Summary')
            headers = ["Metric", "Average (Formula)", "Median (Formula)", "Std Dev (Formula)"]
            for col, h in enumerate(headers):
                ws_sum.write(0, col, h, bold)

            row = 1
            # 2D Length from Raw_2D_Detections length_um_geodesic column.
            if not df.empty:
                n_2d = len(df) + 1
                length_col = excel_col_letter(df, "length_um_geodesic")
                ws_sum.write(row, 0, "2D Geodesic Length (um)")
                ws_sum.write_formula(row, 1, f"=AVERAGE('Raw_2D_Detections'!{length_col}2:{length_col}{n_2d})", num_fmt)
                ws_sum.write_formula(row, 2, f"=MEDIAN('Raw_2D_Detections'!{length_col}2:{length_col}{n_2d})", num_fmt)
                ws_sum.write_formula(row, 3, f"=STDEV.P('Raw_2D_Detections'!{length_col}2:{length_col}{n_2d})", num_fmt)
                row += 1

            # 3D Metrics (all tracks as the primary report population)
            if df_tracks is not None and not df_tracks.empty:
                n_3d = len(df_tracks) + 1 if not df_tracks.empty else 2  # Default to 2 to avoid #DIV/0! bounds
                metrics_3d = [
                    ("3D Geodesic Length (um)", "total_3d_length_um"),
                    ("3D Z-Span (um)", "z_span_um"),
                    ("3D Z-Covered (um)", "z_covered_um"),
                    ("3D Volume (um3)", "volume_um3"),
                    ("3D Tortuosity", "tortuosity_3d")
                ]
                for m_name, col_name in metrics_3d:
                    if col_name not in df_tracks.columns:
                        continue
                    col_letter = excel_col_letter(df_tracks, col_name)
                    ws_sum.write(row, 0, m_name)
                    ws_sum.write_formula(row, 1, f"=AVERAGE('3D_Morphometrics'!{col_letter}2:{col_letter}{n_3d})", num_fmt)
                    ws_sum.write_formula(row, 2, f"=MEDIAN('3D_Morphometrics'!{col_letter}2:{col_letter}{n_3d})", num_fmt)
                    ws_sum.write_formula(row, 3, f"=STDEV.P('3D_Morphometrics'!{col_letter}2:{col_letter}{n_3d})", num_fmt)
                    row += 1

            ws_sum.set_column('A:A', 30)

            # --- Sheet 2: 3D Morphometrics (All Tracks) ---
            if df_tracks is not None and not df_tracks.empty:
                has_qf = "is_quality_track" in df_tracks.columns
                df_tracks.to_excel(writer, sheet_name='3D_Morphometrics', index=False)
                ws_3d = writer.sheets['3D_Morphometrics']
                ws_3d.set_column('A:Z', 15)
                # Insert 3D Distribution Graph (all tracks)
                p_3d = os.path.join(plot_dir, "3d_length_distribution.png")
                if os.path.exists(p_3d):
                    ws_3d.insert_image('K2', p_3d, {'x_scale': 0.4, 'y_scale': 0.4})

                # Add quality-only tracks as a separate sheet instead of silently replacing the main population
                if has_qf:
                    df_q = df_tracks[df_tracks["is_quality_track"]].copy()
                    df_q.to_excel(writer, sheet_name='3D_Morphometrics_Quality', index=False)
                    ws_q3d = writer.sheets['3D_Morphometrics_Quality']
                    ws_q3d.set_column('A:Z', 15)
                if "is_biological_candidate" in df_tracks.columns:
                    df_c = df_tracks[df_tracks["is_biological_candidate"]].copy()
                    df_c.to_excel(writer, sheet_name='3D_Biological_Candidates', index=False)
                    ws_c3d = writer.sheets['3D_Biological_Candidates']
                    ws_c3d.set_column('A:Z', 15)

            # --- Sheet 3: Raw 2D Detections ---
            if not df.empty:
                df.to_excel(writer, sheet_name='Raw_2D_Detections', index=False)
                ws_2d = writer.sheets['Raw_2D_Detections']
                ws_2d.set_column('A:Z', 15)

            # --- Sheet 4: Slice Summary ---
            if not df_summary.empty:
                df_summary.to_excel(writer, sheet_name='Slice_Summary', index=False)

            ws_sum.set_column('B:D', 18)

            # Insert Histograms into Summary
            p_hist = os.path.join(plot_dir, "global_histograms.png")
            p_slice = os.path.join(plot_dir, "length_by_slice.png")
            if os.path.exists(p_hist):
                ws_sum.insert_image('F2', p_hist, {'x_scale': 0.5, 'y_scale': 0.5})
            if os.path.exists(p_slice):
                ws_sum.insert_image('F25', p_slice, {'x_scale': 0.5, 'y_scale': 0.5})

            # --- Sheet 5: Methods Dictionary ---
            dictionary_data = [
                ["Metric", "Formula / Definition", "Biological Interpretation"],
                ["2D Geodesic Length", "Shortest-path length along a 2D skeleton", "Curved fragment length within a single optical slice."],
                ["Total 3D Geodesic Length", "sqrt(max(2D geodesic, XY displacement)^2 + z_span^2)", "Projection-length plus Z-span estimate of whole-nucleus 3D length."],
                ["3D Euclidean Distance", "sqrt(XY displacement^2 + z_span^2)", "Straight-line span used as the tortuosity denominator."],
                ["3D Tortuosity", "Total 3D length / 3D Euclidean distance", "Curvature or over-merge index. Values near 1 indicate straighter nuclei."],
                ["Z-Span (Vertical Span)", "(max_z - min_z) * UM_PER_SLICE_Z", "Endpoint-to-endpoint vertical span; single-slice tracks have span 0."],
                ["Z-Covered", "(max_z - min_z + 1) * UM_PER_SLICE_Z", "Sampled slab thickness covered by the detections."],
                ["3D Volume (um3) *", "sum(area_est_slice * XY_pixel_area * Z_step)", "PSF- and voxel-sensitive Riemann-sum approximation of nuclear volume. Use mainly for relative comparisons acquired under matched imaging settings."],
                ["Effective Diameter Proxy (um) *", "2 * sqrt((V_3D / L_3D) / pi)", "PSF-sensitive cylinder-equivalent diameter. Comparative metric only; do not interpret as literal physical diameter."],
                ["Pitch Angle (degrees)", "abs(arcsin(z_span / Euclidean_3D)) * 180/pi", "Absolute plunge angle relative to the imaging plane."],
                ["Taper Ratio *", "max(area_est across track) / min(area_est across track)", "PSF-sensitive area-derived metric. Useful for relative comparison and instability screening, not as a literal anatomical ratio."],
                ["Nearest Neighbor (um)", "Nearest 3D centroid-to-centroid distance", "Simple local packing-density readout."],
                ["Quality Audit", "Strict post-tracking flag based on length, tortuosity, thickness, taper, and minimum slice count", "Strict audit labels completed tracks after tracking. It does not change segmentation; it defines the no-warning subset used for conservative summaries."],
                ["Biological Candidate", "Softer post-tracking tier that hard-fails long, tortuous, extreme-thick, extreme-taper, or shallow tracks while keeping thick/taper as warning-only", "Candidate tier is intended for visually plausible biological positives while retaining PSF-sensitive warnings for review."],
                ["Parameter Tuning Guidance", "Candidate audit first -> Tracking second -> Segmentation last", "Change audit interpretation when summaries are too strict; change tracking when tracks are fragmented or over-merged; change segmentation only when the raw 2D detections themselves are wrong."],
                ["Standard Deviation", "Std Dev", "Population spread around the mean."]
            ]
            pd.DataFrame(dictionary_data[1:], columns=dictionary_data[0]).to_excel(writer, sheet_name='Methods_Dictionary', index=False)
            ws_dict = writer.sheets['Methods_Dictionary']

            # Manually write the dictionary to avoid any encoding or header issues
            for r_idx, r_data in enumerate(dictionary_data):
                for c_idx, val in enumerate(r_data):
                    ws_dict.write(r_idx, c_idx, val, bold if r_idx==0 else None)

            ws_dict.set_column('A:A', 25)
            ws_dict.set_column('B:B', 40)
            ws_dict.set_column('C:C', 60)
            p_guide = os.path.join(plot_dir, "methods_guide.png")
            if os.path.exists(p_guide):
                ws_dict.insert_image('A12', p_guide, {'x_scale': 0.6, 'y_scale': 0.6})

            print(f"Interactive Excel report successfully saved to {excel_path}")

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"ERROR generating Excel report: {e}")
        write_error_log(out_dir, "Excel Reporter", err_msg)
        try:
            from tkinter import messagebox
            messagebox.showwarning("Reporting Warning", f"Excel Report failed to generate completely.\n{e}")
        except Exception:
            pass



def generate_batch_report(out_dir, df, df_summary, um, df_tracks=None, gui_callback=None, generate_pptx=True):
    """
    Compiles standard summary global output architectures including histograms, mathematical summaries,
    biological methodology pages, and graphical slice overlays natively to a `.pdf` file.

    Args:
        out_dir (str): Root export directory.
        df (pd.DataFrame): Flat 2D analysis parameters.
        df_summary (pd.DataFrame): Top-level slice aggregation tracking.
        um (dict): User-defined pixel to micron mapping ratios.
        df_tracks (pd.DataFrame, optional): Unified 3D tracking geometries array.
        gui_callback (function, optional): Live variable passing to front-end dashboards elements.

    Returns:
        None (Saves directly to absolute path PDF)
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
            has_candidate_tracks = df_tracks is not None and not df_tracks.empty and "is_biological_candidate" in df_tracks.columns
            has_quality_tracks = df_tracks is not None and not df_tracks.empty and "is_quality_track" in df_tracks.columns
            df_candidate_tracks = df_tracks[df_tracks["is_biological_candidate"]].copy() if has_candidate_tracks else (
                df_tracks[df_tracks["is_quality_track"]].copy() if has_quality_tracks else pd.DataFrame()
            )
            has_candidate_data = not df_candidate_tracks.empty

            # Global Z-Projection Image (Top Center). Prefer audit-coded overlay when available.
            quality_z_proj_path = os.path.join(out_dir, "quality_global_z_projection.png")
            raw_z_proj_path = os.path.join(out_dir, "global_z_projection.png")
            z_proj_path = quality_z_proj_path if os.path.exists(quality_z_proj_path) else raw_z_proj_path
            if os.path.exists(z_proj_path):
                ax_z = fig_sum.add_axes([0.15, 0.62, 0.7, 0.28]) # [left, bottom, width, height]
                ax_z.imshow(plt.imread(z_proj_path))
                if z_proj_path == quality_z_proj_path:
                    ax_z.set_title("Candidate Audit Z-Projection (green=candidate, yellow=warning, red=hard fail)", fontsize=10)
                else:
                    ax_z.set_title("Global Z-Projection (Composite [Original | Overlay])", fontsize=10)
                ax_z.axis('off')

            # Plot 1: Counts per slice
            ax1 = fig_sum.add_subplot(2, 2, 3)
            ax1.plot(
                df_summary['z_slice'],
                df_summary['n_spermatids'],
                color='lightgray',
                marker='o',
                markersize=3,
                linewidth=1,
                label='Raw 2D detections'
            )
            if has_candidate_data and "z_start" in df_candidate_tracks.columns:
                q_counts = (
                    df_candidate_tracks["z_start"]
                    .astype(int)
                    .value_counts()
                    .reindex(df_summary["z_slice"].astype(int), fill_value=0)
                    .sort_index()
                )
                ax1.plot(q_counts.index, q_counts.values, 'go-', markersize=4, linewidth=1.5, label='Biological candidates by start Z')
                ax1.set_title("Biological Candidates by Z-Start")
            else:
                ax1.set_title("Raw 2D Detections per Z-Slice")
            ax1.set_xlabel("Z-Index")
            ax1.set_ylabel("Count")
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=7)

            # Plot 2: Length Distribution (biological-candidate population when available)
            ax2 = fig_sum.add_subplot(2, 2, 4)
            if has_candidate_data and "total_3d_length_um" in df_candidate_tracks.columns:
                vals = df_candidate_tracks['total_3d_length_um'].dropna()
                if df_tracks is not None and "total_3d_length_um" in df_tracks.columns:
                    vals_all = df_tracks['total_3d_length_um'].dropna()
                    ax2.hist(vals_all, bins=25, color='lightgray', edgecolor='gray', alpha=0.45, label='All 3D tracks')
                ax2.hist(vals, bins=25, color='darkorange', edgecolor='black', alpha=0.75, label='Biological candidates')
                m_med = vals.median()
                m_avg = vals.mean()
                ax2.axvline(m_med, color='red', linestyle='-', label=f"Median: {m_med:.1f}")
                ax2.axvline(m_avg, color='black', linestyle='--', label=f"Mean: {m_avg:.1f}")
                ax2.set_title("Biological Candidate 3D Length Distribution")
                ax2.set_xlabel("Total 3D Length (um)")
                ax2.set_ylabel("Frequency")
                ax2.legend(fontsize=8)
            elif not df.empty:
                vals = df['length_um_geodesic']
                ax2.hist(vals, bins=25, color='forestgreen', edgecolor='black', alpha=0.7)
                m_med = vals.median()
                m_avg = vals.mean()
                ax2.axvline(m_med, color='red', linestyle='-', label=f"Median: {m_med:.1f}")
                ax2.axvline(m_avg, color='orange', linestyle='--', label=f"Mean: {m_avg:.1f}")
                ax2.set_title("Raw 2D Length Distribution (No Candidate Tracks)")
                ax2.set_xlabel("2D Geodesic Length (um)")
                ax2.set_ylabel("Frequency")
                ax2.legend(fontsize=8)

            fig_sum.savefig(os.path.join(plot_dir, "global_summary.png"), dpi=300, bbox_inches='tight')
            # NEW: Explicitly save global_histograms.png for Excel embedding
            fig_sum.savefig(os.path.join(plot_dir, "global_histograms.png"), dpi=300, bbox_inches='tight')

            # Save length_by_slice.png
            fig_l_slice = plt.figure(figsize=(10, 5))
            ax_ls = fig_l_slice.add_subplot(1, 1, 1)
            ax_ls.plot(df_summary['z_slice'], df_summary['median_length_um'], 'go-', label='Median Length')
            ax_ls.set_title("Median Length by Slice")
            ax_ls.set_xlabel("Z-Slice")
            ax_ls.set_ylabel("Length (um)")
            ax_ls.grid(True, alpha=0.3)
            fig_l_slice.savefig(os.path.join(plot_dir, "length_by_slice.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_l_slice)

            # --- PAGE 1.5: POPULATION CONSOLIDATION ---
            if df_tracks is not None and not df_tracks.empty:
                fig_dyn = plt.figure(figsize=(11, 8.5))
                fig_dyn.suptitle("3D Population Tracking & Candidate Audit", fontsize=15, fontweight='bold')

                total_2d = len(df)
                total_3d = len(df_tracks)
                has_quality = "is_quality_track" in df_tracks.columns
                n_quality = int(df_tracks["is_quality_track"].sum()) if has_quality else total_3d
                n_flagged = total_3d - n_quality
                has_candidate = "is_biological_candidate" in df_tracks.columns
                n_candidate = int(df_tracks["is_biological_candidate"].sum()) if has_candidate else n_quality
                n_hard_fail = total_3d - n_candidate
                n_warning_only = int(df_tracks["has_warning_only"].sum()) if "has_warning_only" in df_tracks.columns else 0

                # A) Reduction Funnel
                ax_bar = fig_dyn.add_subplot(1, 2, 1)
                y_pos = [3, 2, 1, 0]
                counts = [total_2d, total_3d, n_candidate, n_quality]
                colors = ['coral', 'steelblue', '#2ca02c', '#145a32']
                labels = [
                    'Raw 2D Detections\n(All Fragments)',
                    'All 3D Tracks\n(Consolidated)',
                    'Biological Candidates\n(Hard-Fail Removed)',
                    'Strict Quality\n(No Warnings)'
                ]

                bars = ax_bar.barh(y_pos, counts, color=colors, edgecolor='black', height=0.55)
                ax_bar.set_xlim(0, max(counts) * 1.35)

                for i, v in enumerate(counts):
                    ax_bar.text(v + (max(counts)*0.02), y_pos[i], f"{v:,}", va='center', fontweight='bold', fontsize=12)

                ax_bar.set_yticks(y_pos)
                ax_bar.set_yticklabels(labels, fontsize=10, fontweight='bold')
                ax_bar.set_xlabel("Total Count", fontsize=12)
                ax_bar.set_title("Tracking & Candidate Reduction", fontsize=13, fontweight='bold')
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)

                # B) Donut Chart: Candidate vs Warning vs Hard-Fail Breakdown
                ax_pie = fig_dyn.add_subplot(1, 2, 2)

                audit_counts = get_audit_flag_counts(df_tracks)
                n_candidate_clean = max(n_candidate - n_warning_only, 0)
                pie_sizes = [n_candidate_clean, n_warning_only, n_hard_fail]
                pie_labels = [
                    f"Candidate Clean\n({n_candidate_clean:,})",
                    f"Warning Only\n({n_warning_only:,})",
                    f"Hard Fail\n({n_hard_fail:,})"
                ]
                pie_colors = ['#2ca02c', '#ffbf00', '#d62728']

                # Filter out zero segments
                valid = [(s, l, c) for s, l, c in zip(pie_sizes, pie_labels, pie_colors) if s > 0]
                if valid:
                    pie_sizes, pie_labels, pie_colors = zip(*valid)

                wedge_props = dict(width=0.45, edgecolor='white', linewidth=2)
                wedges, texts, autotexts = ax_pie.pie(
                    pie_sizes, labels=None, colors=pie_colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    pctdistance=0.75,
                    wedgeprops=wedge_props,
                    textprops={'fontsize': 12, 'fontweight': 'bold'}
                )

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(13)
                    autotext.set_fontweight('bold')

                ax_pie.legend(
                    wedges, pie_labels,
                    title="Track Candidate Status",
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=len(pie_sizes),
                    fontsize=10,
                    frameon=False
                )

                ax_pie.set_title(f"Candidate Breakdown of {total_3d:,} 3D Tracks", fontsize=13, fontweight='bold')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig_dyn.savefig(os.path.join(plot_dir, "population_consolidation.png"), dpi=300, bbox_inches='tight')
                pdf.savefig(fig_dyn, dpi=300, bbox_inches='tight')
                plt.close(fig_dyn)

            # Write the global summary after the tracking/audit page so the PDF opens
            # on the population-quality overview when tracking results are available.
            pdf.savefig(fig_sum, dpi=300, bbox_inches='tight')
            plt.close(fig_sum)

            # --- PAGE 2: 3D MORPHOMETRICS SUMMARY (ALL TRACKS + BIOLOGICAL CANDIDATE OVERLAY) ---
            if df_tracks is not None and not df_tracks.empty:
                fig_3d = plt.figure(figsize=(11, 8.5))

                has_candidate = "is_biological_candidate" in df_tracks.columns
                df_q = df_tracks[df_tracks["is_biological_candidate"]] if has_candidate else (
                    df_tracks[df_tracks["is_quality_track"]] if "is_quality_track" in df_tracks.columns else df_tracks
                )
                has_q_data = not df_q.empty
                q_label = f" (Biological Candidates: {len(df_q):,} / {len(df_tracks):,})" if has_candidate else ""
                fig_3d.suptitle(f"3D Population Statistics{q_label}", fontsize=14, fontweight='bold')

                # 3D Length
                ax3d_1 = fig_3d.add_subplot(2, 2, 1)
                vals_all = df_tracks['total_3d_length_um']
                vals_q = df_q['total_3d_length_um']
                ax3d_1.hist(vals_all, bins=20, color='lightgray', edgecolor='gray', alpha=0.5, label='All Tracks')
                if has_q_data:
                    ax3d_1.hist(vals_q, bins=20, color='darkorange', edgecolor='black', alpha=0.7, label='Biological Candidates')
                stats_len = vals_q if has_q_data else vals_all
                m_med = stats_len.median()
                m_avg = stats_len.mean()
                ax3d_1.axvline(m_med, color='red', linestyle='-', label=f"Median: {m_med:.1f}")
                ax3d_1.axvline(m_avg, color='black', linestyle='--', label=f"Mean: {m_avg:.1f}")
                ax3d_1.set_title("Total 3D Geodesic Length")
                ax3d_1.set_xlabel("Length (um)")
                ax3d_1.set_ylabel("Frequency")
                ax3d_1.legend(fontsize=7)

                # Save 3d_length_distribution.png for Excel embedding (all tracks plus candidate overlay)
                fig_3d_len = plt.figure(figsize=(6, 4))
                ax_3dl = fig_3d_len.add_subplot(1, 1, 1)
                ax_3dl.hist(vals_all, bins=20, color='steelblue', edgecolor='black', alpha=0.75)
                if has_candidate and not vals_q.empty:
                    ax_3dl.hist(vals_q, bins=20, color='darkorange', edgecolor='black', alpha=0.45)
                ax_3dl.set_title("3D Length Distribution (All Tracks)")
                fig_3d_len.savefig(os.path.join(plot_dir, "3d_length_distribution.png"), dpi=300, bbox_inches='tight')
                plt.close(fig_3d_len)

                # 3D Tortuosity
                ax3d_2 = fig_3d.add_subplot(2, 2, 2)
                vt_all = df_tracks['tortuosity_3d']
                vt_q = df_q['tortuosity_3d']
                vt_all_viz = vt_all[(vt_all >= 0.95) & (vt_all <= 3.0)]
                vt_q_viz = vt_q[(vt_q >= 0.95) & (vt_q <= 3.0)]
                ax3d_2.hist(vt_all_viz, bins=25, color='lightgray', edgecolor='gray', alpha=0.5, label='All Tracks')
                if has_q_data:
                    ax3d_2.hist(vt_q_viz, bins=25, color='purple', edgecolor='black', alpha=0.6, label='Biological Candidates')
                stats_tort = vt_q if has_q_data else vt_all
                ax3d_2.axvline(stats_tort.median(), color='red', linestyle='-', label=f"Median: {stats_tort.median():.2f}")
                ax3d_2.axvline(stats_tort.mean(), color='black', linestyle='--', label=f"Mean: {stats_tort.mean():.2f}")
                ax3d_2.set_xlim(0.95, 3.0)
                ax3d_2.set_title("3D Tortuosity (Curvature)")
                ax3d_2.set_xlabel("Ratio (Length / Distance)")
                ax3d_2.set_ylabel("Frequency")
                ax3d_2.legend(fontsize=7)

                # Vertical Extent
                ax3d_3 = fig_3d.add_subplot(2, 2, 3)
                z_col = "z_span_um" if "z_span_um" in df_tracks.columns else "z_extent_um"
                ve_all = df_tracks[z_col]
                ve_q = df_q[z_col]
                ax3d_3.hist(ve_all, bins=15, color='lightgray', edgecolor='gray', alpha=0.5, label='All Tracks')
                if has_q_data:
                    ax3d_3.hist(ve_q, bins=15, color='teal', edgecolor='black', alpha=0.7, label='Biological Candidates')
                stats_z = ve_q if has_q_data else ve_all
                ax3d_3.axvline(stats_z.median(), color='red', linestyle='-', label=f"Median: {stats_z.median():.1f}")
                ax3d_3.axvline(stats_z.mean(), color='black', linestyle='--', label=f"Mean: {stats_z.mean():.1f}")
                ax3d_3.set_title("Z-Span (Vertical Span)")
                ax3d_3.set_xlabel("Vertical Span (um)")
                ax3d_3.set_ylabel("Frequency")
                ax3d_3.legend(fontsize=7)

                # Volume
                ax3d_4 = fig_3d.add_subplot(2, 2, 4)
                vv_all = df_tracks['volume_um3']
                vv_q = df_q['volume_um3']
                ax3d_4.hist(vv_all, bins=20, color='lightgray', edgecolor='gray', alpha=0.5, label='All Tracks')
                if has_q_data:
                    ax3d_4.hist(vv_q, bins=20, color='gray', edgecolor='black', alpha=0.7, label='Biological Candidates')
                stats_vol = vv_q if has_q_data else vv_all
                ax3d_4.axvline(stats_vol.median(), color='red', linestyle='-', label=f"Median: {stats_vol.median():.0f}")
                ax3d_4.axvline(stats_vol.mean(), color='black', linestyle='--', label=f"Mean: {stats_vol.mean():.0f}")
                ax3d_4.set_title("Approximated 3D Volume")
                ax3d_4.set_xlabel("Volume (um\u00b3)")
                ax3d_4.set_ylabel("Frequency")
                ax3d_4.legend(fontsize=7)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                fig_3d.savefig(os.path.join(plot_dir, "3d_population_stats.png"), dpi=300, bbox_inches='tight')
                pdf.savefig(fig_3d, dpi=300, bbox_inches='tight')
                plt.close(fig_3d)

            # Methods Guide securely moved down past the Advanced Biometrics block.

            # --- PAGE 4: ADVANCED 3D BIOMETRICS (BIOLOGICAL CANDIDATES OVERLAID ON ALL TRACKS) ---
            if df_tracks is not None and not df_tracks.empty:
                fig_adv = plt.figure(figsize=(11, 8.5))
                has_qf = "is_biological_candidate" in df_tracks.columns
                df_q = df_tracks[df_tracks["is_biological_candidate"]] if has_qf else df_tracks
                q_label = f" (Biological Candidates: {len(df_q):,} / {len(df_tracks):,})" if has_qf else ""
                fig_adv.suptitle(f"Advanced 3D Biometrics Dashboard{q_label}", fontsize=16, fontweight='bold', y=0.96)

                # Helper for Mean/Median
                def add_stats_lines(ax, data_series):
                    if data_series.empty or data_series.isna().all(): return
                    m = data_series.mean()
                    med = data_series.median()
                    ax.axvline(med, color='red', linestyle='--', linewidth=1.5, label=f'Median: {med:.2f}')
                    ax.axvline(m, color='green', linestyle=':', linewidth=2, label=f'Mean: {m:.2f}')
                    ax.legend(fontsize=8)

                def dual_hist(ax, col, title, xlabel, color_q, bins=30):
                    vals_all = df_tracks[col].dropna() if col in df_tracks.columns else pd.Series(dtype=float)
                    vals_q = df_q[col].dropna() if col in df_q.columns else pd.Series(dtype=float)
                    if not vals_all.empty:
                        sns.histplot(vals_all, bins=bins, ax=ax, color='lightgray', edgecolor='gray', alpha=0.5, label='All Tracks')
                    if not vals_q.empty:
                        sns.histplot(vals_q, bins=bins, ax=ax, color=color_q, edgecolor='black', alpha=0.7, label='Biological Candidates')
                        add_stats_lines(ax, vals_q)
                    ax.set_title(title)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Frequency")
                    if len(ax.get_legend_handles_labels()[0]) > 0:
                        ax.legend(fontsize=7)

                # 4 panels: Pitch, Thickness, Taper, Nearest Neighbor
                ax_p = fig_adv.add_subplot(2, 2, 1)
                dual_hist(ax_p, 'pitch_deg', "Pitch Angle (Vertical Plunge)", "Degrees (0=Flat, 90=Vertical)", 'orange')

                ax_th = fig_adv.add_subplot(2, 2, 2)
                dual_hist(ax_th, 'thickness_um', "Effective Nucleus Thickness", "Average Diameter (\u00b5m)", '#17becf')

                ax_ta = fig_adv.add_subplot(2, 2, 3)
                dual_hist(ax_ta, 'taper_ratio', "Morphological Taper Ratio", "Max Area / Min Area", 'purple')

                ax_nn = fig_adv.add_subplot(2, 2, 4)
                dual_hist(ax_nn, 'nearest_neighbor_um', "Spatial Packing Density", "Distance to Nearest Neighbor (\u00b5m)", 'brown')

                plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                fig_adv.savefig(os.path.join(plot_dir, "advanced_biometrics.png"), dpi=300, bbox_inches='tight')
                pdf.savefig(fig_adv, dpi=300, bbox_inches='tight')
                plt.close(fig_adv)

            # --- PAGE 5: METHODS & INTERPRETATION GUIDE ---
            fig_guide = plt.figure(figsize=(11, 8.5))
            ax_g = fig_guide.add_axes([0.05, 0.05, 0.9, 0.9])
            ax_g.axis('off')
            guide_full = (
                "METHODS, FORMULAE, AND AUDIT GUIDE\n"
                f"{'='*80}\n\n"
                "1. Total 3D Geodesic Length (um)\n"
                "   Formula: L_3D = sqrt(max(2D geodesic, XY displacement)^2 + z_span^2)\n"
                "   Meaning: Projection-length plus Z-span estimate of whole-nucleus 3D length.\n\n"
                "2. 3D Euclidean Distance (um)\n"
                "   Formula: D_3D = sqrt(XY displacement^2 + z_span^2)\n"
                "   Meaning: Straight-line span used as the tortuosity denominator.\n\n"
                "3. 3D Tortuosity\n"
                "   Formula: T = L_3D / D_3D\n"
                "   Meaning: Curvature index. Values near 1 are straighter; high values suggest bent or fused tracks.\n\n"
                "4. Z-Span and Z-Covered\n"
                "   Formula: z_span = (max_z - min_z) * UM_PER_SLICE_Z; z_covered = (max_z - min_z + 1) * UM_PER_SLICE_Z\n"
                "   Meaning: Z-span is endpoint-to-endpoint displacement; Z-covered is sampled slab thickness.\n\n"
                "5. Approximated 3D Volume (um3)\n"
                "   Formula: V_3D = sum(area_est_slice * XY_pixel_area * Z_step)\n"
                "   Meaning: PSF- and voxel-sensitive volume estimate. Best used for relative comparisons between datasets acquired with matched imaging settings, not as a literal absolute volume.\n\n"
                "6. Effective Diameter Proxy (Average Diameter, um)\n"
                "   Formula: D_avg = 2 * sqrt((V_3D / L_3D) / pi)\n"
                "   Meaning: PSF-sensitive cylinder-equivalent diameter. Useful for relative comparisons under matched imaging settings, not as a literal physical diameter.\n\n"
                "7. Pitch Angle (Plunge Vector, Degrees)\n"
                "   Formula: theta = abs(arcsin(z_span / D_3D)) * 180/pi\n"
                "   Meaning: Absolute plunge angle relative to the imaging plane.\n\n"
                "8. Taper Ratio\n"
                "   Formula: R = max(area_est across track) / min(area_est across track)\n"
                "   Meaning: PSF-sensitive area-derived instability metric. Large values can reflect abrupt narrowing, fusion, or segmentation instability; use mainly for relative comparison.\n\n"
                "9. Spatial Packing Density (Nearest Neighbor Dist, um)\n"
                "   Formula: nearest 3D centroid-to-centroid distance\n"
                "   Meaning: Simple local packing-density readout.\n\n"
                "10. Biological Candidate Audit (post-tracking)\n"
                "   Hard-fail rules: length > AUDIT_MAX_LENGTH_UM, tortuosity > AUDIT_MAX_TORTUOSITY, extreme thickness > AUDIT_EXTREME_THICKNESS_UM, extreme taper > AUDIT_EXTREME_TAPER_RATIO, and n_slices < AUDIT_MIN_SLICES.\n"
                "   Warning-only rules: thickness > AUDIT_MAX_THICKNESS_UM and taper > AUDIT_MAX_TAPER_RATIO. These PSF-sensitive flags are retained for review but no longer remove tracks from the main candidate population.\n"
                "   Interpretation: Audit annotates completed tracks after tracking. Main reports use biological candidates while strict no-warning quality remains a diagnostic subset.\n"
                "\n11. PSF-sensitive metrics note\n"
                "   Volume, effective thickness, taper, and other width/area-derived values are broadened by microscope PSF and voxel sampling. Use them mainly for relative comparisons between matched WT and mutant datasets, not as literal physical dimensions.\n"
            )
            ax_g.text(0, 1, guide_full, transform=ax_g.transAxes, fontsize=10, family='monospace', verticalalignment='top', linespacing=1.3)
            fig_guide.savefig(os.path.join(plot_dir, "methods_guide.png"), dpi=300, bbox_inches='tight')
            pdf.savefig(fig_guide, dpi=300, bbox_inches='tight')
            plt.close(fig_guide)

            # --- PAGE 6: GLOBAL STATISTICS TABLE (after Methods) ---
            fig_tab = plt.figure(figsize=(11, 8.5))
            ax_t = fig_tab.add_subplot(1, 1, 1)
            ax_t.axis('off')
            ax_t.set_title("Global Population Statistics Summary", fontsize=14, fontweight='bold', pad=20)

            stats_rows = []
            if not df.empty:
                l2d = df['length_um_geodesic']
                stats_rows.append(["2D Fragment Geodesic Length (um)", f"{l2d.mean():.2f}", f"{l2d.median():.2f}", f"{l2d.std():.2f}"])

            if df_tracks is not None and not df_tracks.empty:
                # Biological-candidate subset
                has_qf = "is_biological_candidate" in df_tracks.columns
                df_q = df_tracks[df_tracks["is_biological_candidate"]] if has_qf else df_tracks

                # -- All Tracks Section --
                stats_rows.append(["--- ALL TRACKS ---", f"N={len(df_tracks)}", "", ""])
                l3d = df_tracks['total_3d_length_um']
                z_col = "z_span_um" if "z_span_um" in df_tracks.columns else "z_extent_um"
                ze  = df_tracks[z_col]
                vo  = df_tracks['volume_um3']
                to  = df_tracks['tortuosity_3d']
                th  = df_tracks['thickness_um']
                stats_rows.append(["3D Length (um)", f"{l3d.mean():.2f}", f"{l3d.median():.2f}", f"{l3d.std():.2f}"])
                stats_rows.append(["3D Z-Span (um)", f"{ze.mean():.2f}", f"{ze.median():.2f}", f"{ze.std():.2f}"])
                stats_rows.append(["3D Volume (um3)*", f"{vo.mean():.1f}", f"{vo.median():.1f}", f"{vo.std():.1f}"])
                stats_rows.append(["3D Tortuosity", f"{to.mean():.3f}", f"{to.median():.3f}", f"{to.std():.3f}"])
                stats_rows.append(["3D Thickness (um)*", f"{th.mean():.2f}", f"{th.median():.2f}", f"{th.std():.2f}"])

                # -- Biological Candidate Population Section --
                if has_qf and len(df_q) > 0:
                    stats_rows.append(["--- BIOLOGICAL CANDIDATES ---", f"N={len(df_q)}", "", ""])
                    l3q = df_q['total_3d_length_um']
                    zeq = df_q[z_col]
                    voq = df_q['volume_um3']
                    toq = df_q['tortuosity_3d']
                    thq = df_q['thickness_um']
                    piq = df_q['pitch_deg']
                    taq = df_q['taper_ratio']
                    nnq = df_q['nearest_neighbor_um'].dropna()
                    stats_rows.append(["3D Length (um)", f"{l3q.mean():.2f}", f"{l3q.median():.2f}", f"{l3q.std():.2f}"])
                    stats_rows.append(["3D Z-Span (um)", f"{zeq.mean():.2f}", f"{zeq.median():.2f}", f"{zeq.std():.2f}"])
                    stats_rows.append(["3D Volume (um3)*", f"{voq.mean():.1f}", f"{voq.median():.1f}", f"{voq.std():.1f}"])
                    stats_rows.append(["3D Tortuosity", f"{toq.mean():.3f}", f"{toq.median():.3f}", f"{toq.std():.3f}"])
                    stats_rows.append(["3D Thickness (um)*", f"{thq.mean():.2f}", f"{thq.median():.2f}", f"{thq.std():.2f}"])
                    stats_rows.append(["3D Pitch (degrees)", f"{piq.mean():.1f}", f"{piq.median():.1f}", f"{piq.std():.1f}"])
                    stats_rows.append(["3D Taper Ratio*", f"{taq.mean():.2f}", f"{taq.median():.2f}", f"{taq.std():.2f}"])
                    if not nnq.empty:
                        stats_rows.append(["Nearest Neighbor (um)", f"{nnq.mean():.1f}", f"{nnq.median():.1f}", f"{nnq.std():.1f}"])

            if stats_rows:
                table = ax_t.table(
                    cellText=stats_rows,
                    colLabels=["Metric", "Mean", "Median", "Std Dev"],
                    loc='center', cellLoc='center',
                    colWidths=[0.38, 0.2, 0.2, 0.2]
                )
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
                    ax_hist.set_xlabel("Spermatid Nucleus Length (um)")
                    ax_hist.set_ylabel("Frequency (Count)")

                    m_med = slice_data['length_um_geodesic'].median()
                    m_avg = slice_data['length_um_geodesic'].mean()
                    ax_hist.axvline(m_med, color='red', linestyle='-', alpha=0.7, label=f"Median: {m_med:.1f}")
                    ax_hist.axvline(m_avg, color='orange', linestyle='--', alpha=0.7, label=f"Mean: {m_avg:.1f}")
                    ax_hist.legend(fontsize=9)
                else:
                    ax_hist.text(0.5, 0.5, "No Detections", ha='center', va='center')

                pdf.savefig(fig_slice, dpi=300)
                plt.close(fig_slice)

                if gui_callback:
                    gui_callback(int(80 + (20 * (idx_p+1) / len(df_summary))))

        print(f"Report successfully saved to {pdf_path}")

        # --- GENERATE POWERPOINT REPORT ---
        try:
            if generate_pptx:
                generate_pptx_report(out_dir, df, df_summary, um, df_tracks)
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"PPTX Report failed: {e}")
            write_error_log(out_dir, "PowerPoint Generator (via Batch)", err_msg)

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"ERROR generating PDF report: {e}")
        write_error_log(out_dir, "PDF Reporter", err_msg)
        try:
            from tkinter import messagebox
            messagebox.showwarning("Reporting Warning", f"PDF Report failed to generate completely.\n{e}")
        except Exception:
            pass



def generate_pptx_report(out_dir, df, df_summary, um, df_tracks=None):
    """
    Generates a native Microsoft Office PowerPoint presentation (.pptx) with
    fully editable, data-embedded charts for each key biometric.

    Each chart is a *true* Office Open XML chart object (not a rasterised image),
    so the researcher can re-style, re-colour, and re-export from PowerPoint without
    any additional software.  The chart data tables are embedded inside the PPTX
    file itself, enabling standalone sharing.

    Slide structure
    ---------------
    1. **Global Population Analytics** - side-by-side column histograms of:
       - 2D geodesic length distribution (from the current batch).
       - 3D estimated track length distribution (if tracking was run).
    2. **Population Tracking Consolidation** (if tracking was run) - pie chart
       showing the fraction of single-slice vs. multi-slice reassigned tracks.
    3. **3D Biometrics Dashboard** - scatter plot of tortuosity vs. 3D length,
       plus a bar chart of pitch angle distribution.
    4. **Methods & Calculation Details** - text slide with biological justification
       and mathematical formulae for all metrics.

    Hyperlink behaviour
    -------------------
    The title text on each slide contains a hyperlink to the source Excel workbook.
    Clicking the title in Slide Show mode opens the associated Excel file so the
    viewer can inspect the raw numbers behind any chart.

    Biological context of each chart
    ---------------------------------
    - **Length histogram**: A Gaussian distribution centred near the species-expected
      nuclear length (about 10 um for D. melanogaster in this workflow) supports
      mature elongation. Bimodal distributions may indicate simultaneous maturation
      cohorts, heteromorphic sperm classes, or segmentation/tracking mixtures.
    - **Pie chart**: Quantifies how many tracked objects span multiple Z-slices
      (multi-slice continuity) vs. transient single-slice events (possibly debris).
    - **Tortuosity vs. 3D length scatter**: Healthy spermatids cluster in the
      low-tortuosity, high-length quadrant.  High-tortuosity outliers indicate
      coiled or bent morphologies.
    - **Pitch angle**: A right-skewed distribution toward 90 degrees is expected during
      the apical plunging phase of spermatid elongation.

    Args:
        out_dir (str): Top-level analysis output directory.  The presentation is
            saved as ``<out_dir>/spermatid_analysis_report.pptx``.
        df (pd.DataFrame): Per-spermatid 2D measurement table.
        df_summary (pd.DataFrame): Per-slice summary statistics table.
        um (float): Microns-per-pixel scale factor (``UM_PER_PX_XY``).
        df_tracks (pd.DataFrame, optional): 3D track table.  ``None`` skips
            tracking-specific slides (slides 2 and 3).
    """
    try:
        import os as _os
        import numpy as _np
        from pptx import Presentation
        from pptx.chart.data import CategoryChartData
        from pptx.dml.color import RGBColor
        from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
        from pptx.util import Inches, Pt
        from pptx.enum.text import PP_ALIGN
        from tkinter import messagebox

        print("PPTX: Starting report generation...")
        # We need a fallback blank pptx
        prs = Presentation()

        # Safe blank layout (often index 6 or 5)
        try:
            blank_slide_layout = prs.slide_layouts[6]
        except Exception:
            blank_slide_layout = prs.slide_layouts[0]

        def add_hyperlink(slide, sheet_name="Population_Summary"):
            excel_name = f"batch_analysis_results_{_VERSION}.xlsx"
            # PowerPoint/Excel deep-link fragment: filename.xlsx#Sheet!A1
            target = f"{excel_name}#'{sheet_name}'!A1"

            txBox = slide.shapes.add_textbox(Inches(0.2), Inches(7.1), Inches(9.5), Inches(0.4))
            tf = txBox.text_frame
            p = tf.add_paragraph()
            run = p.add_run()
            run.text = f"Click to View Detailed Data: {excel_name} [{sheet_name}]"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0, 0, 255) # Blue
            run.font.underline = True
            try:
                # Shape-level link
                txBox.click_action.hyperlink.address = target
                # Text-run level link
                run.hyperlink.address = target
            except Exception:
                pass

        def add_line_chart(slide, x_data, y_data, left, top, width, height, title):
            chart_data = CategoryChartData()
            chart_data.categories = list(x_data)
            chart_data.add_series('Count', list(y_data))

            chart = slide.shapes.add_chart(
                XL_CHART_TYPE.LINE, left, top, width, height, chart_data
            ).chart
            chart.has_legend = False
            chart.chart_title.text_frame.text = title
            chart.chart_title.text_frame.paragraphs[0].font.size = Pt(12)
            chart.category_axis.tick_labels.font.size = Pt(8)
            chart.value_axis.tick_labels.font.size = Pt(8)
            chart.value_axis.has_major_gridlines = True

        def add_horizontal_bar_chart(slide, categories, values, colors, left, top, width, height, title):
            chart_data = CategoryChartData()
            chart_data.categories = categories
            chart_data.add_series('Count', values)

            chart = slide.shapes.add_chart(
                XL_CHART_TYPE.BAR_CLUSTERED, left, top, width, height, chart_data
            ).chart
            chart.has_legend = False
            chart.chart_title.text_frame.text = title
            chart.chart_title.text_frame.paragraphs[0].font.size = Pt(12)
            chart.category_axis.tick_labels.font.size = Pt(9)
            chart.value_axis.tick_labels.font.size = Pt(8)

            # Show data labels for bar values
            plot = chart.plots[0]
            plot.has_data_labels = True
            for i, point in enumerate(plot.series[0].points):
                point.data_label.font.size = Pt(10)
                point.data_label.font.bold = True

        def add_histogram(slide, data_series, left, top, width, height, title, bins=20):
            if data_series is None or data_series.empty or data_series.isna().all():
                return

            clean_data = data_series.dropna()
            counts, bin_edges = _np.histogram(clean_data, bins=bins)
            avg = clean_data.mean()
            med = clean_data.median()

            # Create category names from bin boundaries
            categories = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))]

            chart_data = CategoryChartData()
            chart_data.categories = categories
            chart_data.add_series('Count', list(counts))

            chart = slide.shapes.add_chart(
                XL_CHART_TYPE.COLUMN_CLUSTERED, left, top, width, height, chart_data
            ).chart
            chart.has_legend = False
            chart.chart_title.text_frame.text = f"{title} | Mean {avg:.2f}, Median {med:.2f}"
            chart.chart_title.text_frame.paragraphs[0].font.size = Pt(12)

            # Reduce axis label crowding
            chart.category_axis.tick_labels.font.size = Pt(8)
            chart.value_axis.tick_labels.font.size = Pt(8)

        # --- Slide 1: Global Analytics Overview ---
        slide1 = prs.slides.add_slide(blank_slide_layout)
        txBox = slide1.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(9), Inches(0.5))
        tf = txBox.text_frame
        tf.text = "Spermatid Population Overview"
        tf.paragraphs[0].font.size = Pt(22)
        tf.paragraphs[0].font.bold = True

        # Bottom Left: Detections per Slice (Line Chart) - Identical to PDF Page 1
        if not df_summary.empty:
            add_line_chart(slide1, df_summary['z_slice'], df_summary['n_spermatids'], Inches(0.2), Inches(4.0), Inches(4.5), Inches(3.0), "Detections per Z-Slice (Raw)")

        # Top Center Left: 2D Length Dist
        if not df.empty:
            add_histogram(slide1, df['length_um_geodesic'], Inches(0.2), Inches(0.8), Inches(4.5), Inches(3.0), "Global 2D Geodesic Length Distribution")

        # Top Center Right: 3D Length Dist
        if df_tracks is not None and not df_tracks.empty:
            add_histogram(slide1, df_tracks['total_3d_length_um'], Inches(5.0), Inches(0.8), Inches(4.5), Inches(3.0), "Estimated 3D Track Length")

        add_hyperlink(slide1)

        # --- Slide 2: Population Consolidation ---
        if df_tracks is not None and not df_tracks.empty:
            slide2 = prs.slides.add_slide(blank_slide_layout)
            txBox = slide2.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(9), Inches(0.5))
            tf = txBox.text_frame
            tf.text = "3D Population Tracking & Candidate Audit"
            tf.paragraphs[0].font.size = Pt(22)
            tf.paragraphs[0].font.bold = True

            # Left: Reduction Bar Chart (PDF Parity)
            total_2d = len(df)
            total_3d = len(df_tracks)
            has_candidates = "is_biological_candidate" in df_tracks.columns
            has_qf = "is_quality_track" in df_tracks.columns
            n_candidate = int(df_tracks["is_biological_candidate"].sum()) if has_candidates else (
                int(df_tracks["is_quality_track"].sum()) if has_qf else total_3d
            )
            n_quality = int(df_tracks["is_quality_track"].sum()) if has_qf else n_candidate
            n_warning_only = int(df_tracks["has_warning_only"].sum()) if "has_warning_only" in df_tracks.columns else 0
            n_hard_fail = total_3d - n_candidate

            add_horizontal_bar_chart(slide2,
                                     ['Strict No-Warning', 'Biological Candidates', 'All 3D Tracks', 'Raw 2D Detections'],
                                     [n_quality, n_candidate, total_3d, total_2d],
                                     None, Inches(0.2), Inches(1.5), Inches(4.5), Inches(4.5), "Tracking & Candidate Reduction")

            # Right: Composition Pie Chart
            n_candidate_clean = max(n_candidate - n_warning_only, 0)
            pie_sizes = [n_candidate_clean, n_warning_only, n_hard_fail]
            pie_labels = ['Candidate Clean', 'Warning Only', 'Hard Fail']

            # Filter zero values for pptx chart
            valid_idx = [i for i, v in enumerate(pie_sizes) if v > 0]
            pie_sizes = [pie_sizes[i] for i in valid_idx]
            pie_labels = [pie_labels[i] for i in valid_idx]

            chart_data = CategoryChartData()
            chart_data.categories = pie_labels
            chart_data.add_series('Population', pie_sizes)

            chart2 = slide2.shapes.add_chart(
                XL_CHART_TYPE.PIE, Inches(4.6), Inches(1.2), Inches(5.2), Inches(5.2), chart_data
            ).chart
            chart2.has_legend = True
            chart2.legend.position = XL_LEGEND_POSITION.CORNER
            chart2.legend.font.size = Pt(8)
            chart2.chart_title.text_frame.text = f"Candidate Breakdown of {total_3d:,} 3D Tracks"
            chart2.chart_title.text_frame.paragraphs[0].font.size = Pt(12)

            plot = chart2.plots[0]
            plot.has_data_labels = True
            total = sum(pie_sizes) if sum(pie_sizes) > 0 else 1
            for i, point in enumerate(plot.series[0].points):
                val = pie_sizes[i]
                pct = (val / total) * 100
                label_text = f"{val:,}\n({pct:.1f}%)"
                point.data_label.text_frame.text = label_text
                point.data_label.font.size = Pt(9)
                point.data_label.font.bold = True

            add_hyperlink(slide2)

        # ---------------------------------------------------------------------
        # SLIDE 3: Advanced 3D Biometrics (Biological Candidate Population)
        # ---------------------------------------------------------------------
        if df_tracks is not None and not df_tracks.empty:
            slide3 = prs.slides.add_slide(blank_slide_layout)
            txBox = slide3.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(9), Inches(0.5))
            tf = txBox.text_frame
            has_qf = "is_biological_candidate" in df_tracks.columns
            df_q = df_tracks[df_tracks["is_biological_candidate"]] if has_qf else df_tracks
            plot_df = df_q if not df_q.empty else df_tracks
            q_label = f" (Biological Candidates: {len(df_q):,} / {len(df_tracks):,})" if has_qf else ""
            if has_qf and df_q.empty:
                q_label += " - all tracks shown"
            tf.text = f"Advanced 3D Biometrics Dashboard{q_label}"
            tf.paragraphs[0].font.size = Pt(22)
            tf.paragraphs[0].font.bold = True

            # Slide 3 shows biological candidates when available, falling back to all tracks
            # if the audit-passed subset is empty.
            add_histogram(slide3, plot_df['pitch_deg'], Inches(0.2), Inches(0.8), Inches(4.5), Inches(2.9), "Pitch Angle (Degrees)", bins=20)
            add_histogram(slide3, plot_df['thickness_um'], Inches(5.0), Inches(0.8), Inches(4.5), Inches(2.9), "Effective Diameter Proxy (\u00b5m)", bins=20)
            add_histogram(slide3, plot_df['taper_ratio'], Inches(0.2), Inches(3.8), Inches(4.5), Inches(2.9), "Morphological Taper Ratio", bins=20)
            add_histogram(slide3, plot_df['nearest_neighbor_um'].dropna(), Inches(5.0), Inches(3.8), Inches(4.5), Inches(2.9), "Nearest-Neighbor Distance (\u00b5m)", bins=20)

            add_hyperlink(slide3, "3D_Morphometrics")

        # ---------------------------------------------------------------------
        # SLIDE 4: Global Population Statistics Summary Table
        # ---------------------------------------------------------------------
        slide4 = prs.slides.add_slide(blank_slide_layout)
        txBox = slide4.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.8))
        tf = txBox.text_frame
        tf.text = "Global Population Statistics Summary"
        tf.paragraphs[0].font.size = Pt(22)
        tf.paragraphs[0].font.bold = True

        # Prepare stats matching PDF (All vs Biological Candidates)
        stats_rows = [["Metric", "Mean", "Median", "Std Dev"]]
        if not df.empty:
            l2d = df['length_um_geodesic']
            stats_rows.append(["2D Fragment Length (\u00b5m)", f"{l2d.mean():.2f}", f"{l2d.median():.2f}", f"{l2d.std():.2f}"])

        if df_tracks is not None and not df_tracks.empty:
            has_qf = "is_biological_candidate" in df_tracks.columns
            df_q = df_tracks[df_tracks["is_biological_candidate"]] if has_qf else df_tracks

            # Section Header for All Tracks
            stats_rows.append(["--- ALL TRACKS ---", f"N={len(df_tracks)}", "", ""])

            def add_pop_rows(pop_df, prefix=""):
                l3 = pop_df['total_3d_length_um']
                z_col = "z_span_um" if "z_span_um" in pop_df.columns else "z_extent_um"
                ze = pop_df[z_col]
                vo = pop_df['volume_um3']
                to = pop_df['tortuosity_3d']
                th = pop_df['thickness_um']
                stats_rows.append([f"{prefix}3D Length (\u00b5m)", f"{l3.mean():.2f}", f"{l3.median():.2f}", f"{l3.std():.2f}"])
                stats_rows.append([f"{prefix}3D Z-Span (\u00b5m)", f"{ze.mean():.2f}", f"{ze.median():.2f}", f"{ze.std():.2f}"])
                stats_rows.append([f"{prefix}3D Volume (\u00b5m\u00b3)", f"{vo.mean():.1f}", f"{vo.median():.1f}", f"{vo.std():.1f}"])
                stats_rows.append([f"{prefix}3D Tortuosity", f"{to.mean():.3f}", f"{to.median():.3f}", f"{to.std():.3f}"])
                stats_rows.append([f"{prefix}3D Thickness (\u00b5m)", f"{th.mean():.2f}", f"{th.median():.2f}", f"{th.std():.2f}"])

            add_pop_rows(df_tracks)

            if has_qf and not df_q.empty:
                # Section Header for Biological Candidate Population
                stats_rows.append(["--- BIOLOGICAL CANDIDATES ---", f"N={len(df_q)}", "", ""])
                add_pop_rows(df_q)

        if len(stats_rows) > 1:
            rows = len(stats_rows)
            cols = 4
            table_shape = slide4.shapes.add_table(rows, cols, Inches(0.5), Inches(1.0), Inches(9), Inches(6.0))
            table = table_shape.table

            # Header styling
            for c in range(cols):
                cell = table.cell(0, c)
                cell.text_frame.text = stats_rows[0][c]
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(68, 114, 196) # Standard Blue
                cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
                cell.text_frame.paragraphs[0].font.size = Pt(11)
                cell.text_frame.paragraphs[0].font.bold = True

            # Body styling
            for r in range(1, rows):
                is_separator = "---" in stats_rows[r][0]
                for c in range(cols):
                    cell = table.cell(r, c)
                    cell.text_frame.text = stats_rows[r][c]
                    cell.text_frame.paragraphs[0].font.size = Pt(8)
                    if is_separator:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = RGBColor(240, 240, 240)
                        cell.text_frame.paragraphs[0].font.bold = True
                    elif c == 0:
                        cell.text_frame.paragraphs[0].font.bold = True

        add_hyperlink(slide4, "Population_Summary")

        # ---------------------------------------------------------------------
        # SLIDE 5: Methods & Interpretation Guide (Exact PDF Synchronization)
        # ---------------------------------------------------------------------
        slide5 = prs.slides.add_slide(blank_slide_layout)
        txBox_title = slide5.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
        tf_title = txBox_title.text_frame
        tf_title.text = "Methods, Formulae, Parameter Guide & Audit Details"
        tf_title.paragraphs[0].font.size = Pt(20)
        tf_title.paragraphs[0].font.bold = True

        # Synchronized text from the PDF report version
        methods_items = [
            ("1. Total 3D Geodesic Length (um)", [
                ("Formula: ", "L_3D = sqrt(max(2D geodesic, XY displacement)^2 + z_span^2)"),
                ("Meaning: ", "Projection-length plus Z-span estimate of whole-nucleus 3D length.")
            ]),
            ("2. 3D Euclidean Distance (um)", [
                ("Formula: ", "D_3D = sqrt(XY displacement^2 + z_span^2)"),
                ("Meaning: ", "Straight-line span used as the tortuosity denominator.")
            ]),
            ("3. 3D Tortuosity", [
                ("Formula: ", "T = L_3D / D_3D"),
                ("Meaning: ", "Curvature or over-merge index. Values near 1 are straighter; high values suggest bent or fused tracks.")
            ]),
            ("4. Z-Span and Z-Covered", [
                ("Formula: ", "z_span = (max_z - min_z) * UM_PER_SLICE_Z; z_covered = (max_z - min_z + 1) * UM_PER_SLICE_Z"),
                ("Meaning: ", "Z-span is endpoint-to-endpoint displacement; Z-covered is sampled slab thickness.")
            ]),
            ("5. Approximated 3D Volume (um3)", [
                ("Formula: ", "V_3D = sum(area_est_slice * XY_pixel_area * Z_step)"),
                ("Meaning: ", "PSF- and voxel-sensitive volume estimate. Use mainly for relative comparisons between matched datasets, not as a literal absolute volume.")
            ]),
            ("6. Effective Diameter Proxy (Average Diameter, um)", [
                ("Formula: ", "D_avg = 2 * sqrt((V_3D / L_3D) / pi)"),
                ("Meaning: ", "PSF-sensitive cylinder-equivalent diameter. Useful for relative comparisons under matched imaging settings, not as a literal physical diameter.")
            ]),
            ("7. Pitch Angle (Plunge Vector, Degrees)", [
                ("Formula: ", "theta = abs(arcsin(z_span / D_3D)) * 180/pi"),
                ("Meaning: ", "Absolute plunge angle relative to the imaging plane.")
            ]),
            ("8. Taper Ratio", [
                ("Formula: ", "R = max(area_est across track) / min(area_est across track)"),
                ("Meaning: ", "PSF-sensitive area-derived instability metric. Large values can reflect abrupt narrowing, fusion, or segmentation instability; use mainly for relative comparison.")
            ]),
            ("9. Spatial Packing Density (Nearest Neighbor Dist, um)", [
                ("Formula: ", "nearest 3D centroid-to-centroid distance"),
                ("Meaning: ", "Simple local packing-density readout.")
            ]),
            ("10. Tracking Parameter Provenance", [
                ("Source: ", "Several overlap and conservative tracking thresholds were originally selected by an evolutionary tuning script."),
                ("Goal: ", "The optimizer rewarded multi-slice continuity while penalizing fragmentation, implausible merges, and biology-aware outlier categories."),
                ("Practical use: ", "Treat tuned tracking values as strong starting points. Adjust them only if a new dataset clearly shows fragmentation or over-merging.")
            ]),
            ("11. Candidate Audit (post-tracking)", [
                ("Audit rules: ", "hard-fail tracks when 3D length > AUDIT_MAX_LENGTH_UM, 3D tortuosity > AUDIT_MAX_TORTUOSITY, effective thickness > AUDIT_EXTREME_THICKNESS_UM, extreme taper > AUDIT_EXTREME_TAPER_RATIO, or track slices < AUDIT_MIN_SLICES. Ordinary thick/taper tracks remain warning-only."),
                ("Meaning: ", "Audit does not change raw detection or tracking. It labels completed tracks as biological candidates, warning-only candidates, hard fails, and strict no-warning quality tracks."),
                ("Practical use: ", "Use biological candidates as the main analysis population. Use strict no-warning quality as a conservative diagnostic subset."),
                ("Biology note: ", "At this acquisition z-step (~1.04 um), single-slice nuclei can be biologically valid because the true mature Drosophila sperm nucleus is much thinner in z than the optical sampling.")
            ]),
            ("12. PSF-sensitive metrics note", [
                ("Important: ", "Volume, effective thickness, taper, and other width/area-derived values are broadened by microscope PSF and voxel sampling."),
                ("Use them for: ", "relative comparison between WT and mutant datasets acquired with matched settings."),
                ("Do not use them as: ", "literal physical dimensions or absolute biophysical ground truth.")
            ])
        ]

        top_y = 1.0
        for title, content_list in methods_items:
            tb = slide5.shapes.add_textbox(Inches(0.5), Inches(top_y), Inches(9), Inches(0.72))
            tf = tb.text_frame
            tf.word_wrap = True

            p = tf.paragraphs[0]
            p.text = title
            p.font.bold = True
            p.font.size = Pt(10)

            for label, text in content_list:
                p2 = tf.add_paragraph()
                p2.font.size = Pt(8.5)
                if label:
                    run1 = p2.add_run()
                    run1.text = label
                    run1.font.bold = True
                run2 = p2.add_run()
                run2.text = text
                run2.font.bold = False

            top_y += 0.74

        add_hyperlink(slide5, "Population_Summary")

        # Save the presentation matching Batch Name
        final_pptx_name = f"batch_analysis_results_{_VERSION}.pptx"
        output_path = os.path.join(out_dir, final_pptx_name)
        try:
            print(f"PPTX: Saving to {output_path}...")
            prs.save(output_path)
            print(f"PPTX Report successfully saved to: {output_path}")
            return True
        except PermissionError:
            print(f"CRITICAL ERROR: Could not save PowerPoint because the file '{final_pptx_name}' is currently open!")
            return False

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Failed to generate PPTX report: {e}")
        write_error_log(out_dir, "PowerPoint Generator", err_msg)
        return False

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Failed to generate PPTX report: {e}")
        write_error_log(out_dir, "PowerPoint Generator", err_msg)
        try:
            from tkinter import messagebox
            messagebox.showerror("Reporting Error", f"Failed to generate PowerPoint Report:\n{e}\n\nSee report_generation_errors.txt for details.")
        except Exception:
            pass
        return False




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

PARAM_SECTIONS = {
    "Calibration & Scale": [
        "UM_PER_PX_XY", "UM_PER_SLICE_Z"
    ],
    "Image Enhancement": [
        "CLAHE_CLIP", "CLAHE_KERNEL", "BG_SIGMA", "THRESHOLD_HI", "THRESHOLD_LO"
    ],
    "Binary Mask Cleanup": [
        "CLOSE_RADIUS", "MIN_HOLE_AREA", "MIN_OBJ_PX"
    ],
    "Skeleton Cleanup & Splitting": [
        "MAX_BRIDGE_PX", "MAX_BRANCH_LEN_PX", "BREAK_JUNCTIONS"
    ],
    "Early Shape Filter": [
        "USE_EARLY_SHAPE_FILTER", "MIN_ECCENTRICITY", "MAX_MINOR_PX", "MIN_AXIS_RATIO", "MIN_MAJOR_PX"
    ],
    "Post-Skeleton Acceptance Filters": [
        "MIN_SKEL_LEN_PX", "MAX_GEODESIC_LEN_PX", "MAX_WIDTH_PX", "MIN_LENGTH_WIDTH_RATIO",
        "MAX_BRANCH_NODES", "MAX_TORTUOSITY", "MAX_ENDPOINT_COUNT", "ALLOW_LOOPS"
    ],
    "3D Tracking": [
        "DO_TRACKING", "TRACKING_BACKEND", "TRACK_MAX_DIST_UM", "TRACK_MAX_GAP_SLICES", "TRACK_BBOX_PADDING_PX",
        "CONSERVATIVE_MAX_WIDTH_JUMP_RATIO", "CONSERVATIVE_MAX_LENGTH_JUMP_RATIO",
        "CONSERVATIVE_MAX_AREA_JUMP_RATIO", "CONSERVATIVE_MAX_TORTUOSITY_JUMP",
        "CONSERVATIVE_MAX_CENTROID_JUMP_UM"
    ],
    "Overlap-First Continuation": [
        "OVERLAP_STABILITY_THRESHOLD", "OVERLAP_ORIENTATION_DEG", "OVERLAP_MULTIPLIER", "OVERLAP_MIN_STABLE_COUNT"
    ],
    "3D Tracking - Global Assignment Prototype": [
        "ASSIGNMENT_MAX_COST", "ASSIGNMENT_DIST_WEIGHT", "ASSIGNMENT_OVERLAP_WEIGHT",
        "ASSIGNMENT_LENGTH_WEIGHT", "ASSIGNMENT_WIDTH_WEIGHT", "ASSIGNMENT_AREA_WEIGHT",
        "ASSIGNMENT_ANGLE_WEIGHT"
    ],
    "3D Tracking - Hybrid Fragment Repair": [
        "HYBRID_REPAIR_MAX_COST", "HYBRID_REPAIR_MAX_GAP_SLICES", "HYBRID_REPAIR_MAX_FRAGMENT_SLICES",
        "HYBRID_REPAIR_MAX_LINK_DIST_UM", "HYBRID_REPAIR_MIN_OVERLAP", "HYBRID_REPAIR_MAX_FINAL_LENGTH_UM"
    ],
    "2.5D U-Net Integration": [
        "SEGMENTATION_ENGINE", "UNET_MODEL_PATH", "UNET_THRESHOLD_MODE",
        "UNET_CANDIDATE_THRESHOLD", "UNET_SEED_THRESHOLD", "UNET_CONTEXT_MODE",
        "UNET_INFERENCE_MODE", "UNET_TILE_SIZE", "UNET_TILE_OVERLAP", "UNET_ROI_PADDING_PX",
        "UNET_RESCUE_ENABLE", "UNET_RESCUE_THRESHOLD", "UNET_RESCUE_EXCLUDE_DILATION_PX",
        "UNET_RESCUE_MIN_COMPONENT_PX", "UNET_RESCUE_MAX_ADDITIONS_PER_SLICE",
        "UNET_TRACKING_SUPPORT", "ASSIGNMENT_UNET_SUPPORT_WEIGHT",
        "ASSIGNMENT_UNET_CONTINUITY_WEIGHT", "HYBRID_REPAIR_UNET_SUPPORT_WEIGHT"
    ],
    "Candidate Audit (post-tracking)": [
        "AUDIT_MAX_LENGTH_UM", "AUDIT_MAX_TORTUOSITY", "AUDIT_MAX_THICKNESS_UM", "AUDIT_MAX_TAPER_RATIO",
        "AUDIT_EXTREME_THICKNESS_UM", "AUDIT_EXTREME_TAPER_RATIO", "AUDIT_MIN_SLICES"
    ],
}

PARAM_TITLES = {
    "UM_PER_PX_XY": "XY pixel size (um / pixel)",
    "UM_PER_SLICE_Z": "Z step size (um / slice)",
    "CLAHE_CLIP": "CLAHE contrast limit",
    "CLAHE_KERNEL": "CLAHE tile size",
    "BG_SIGMA": "Background subtraction sigma",
    "THRESHOLD_HI": "High ridge threshold percentile",
    "THRESHOLD_LO": "Low ridge threshold percentile",
    "CLOSE_RADIUS": "Mask closing radius",
    "MIN_HOLE_AREA": "Fill holes up to this size (px)",
    "MIN_OBJ_PX": "Minimum object size (px)",
    "MAX_BRIDGE_PX": "Maximum skeleton bridge gap (px)",
    "MAX_BRANCH_LEN_PX": "Maximum spur length to prune (px)",
    "BREAK_JUNCTIONS": "Break skeleton junctions",
    "USE_EARLY_SHAPE_FILTER": "Use early blob / round-shape filter",
    "MIN_ECCENTRICITY": "Minimum eccentricity",
    "MAX_MINOR_PX": "Maximum minor axis (px)",
    "MIN_AXIS_RATIO": "Minimum major/minor ratio",
    "MIN_MAJOR_PX": "Minimum major axis (px)",
    "MIN_SKEL_LEN_PX": "Minimum 2D skeleton length (px)",
    "MAX_GEODESIC_LEN_PX": "Maximum 2D skeleton length (px)",
    "MAX_WIDTH_PX": "Maximum 2D width (px)",
    "MIN_LENGTH_WIDTH_RATIO": "Minimum length/width ratio",
    "MAX_BRANCH_NODES": "Maximum branch points",
    "MAX_TORTUOSITY": "Maximum 2D tortuosity",
    "MAX_ENDPOINT_COUNT": "Maximum endpoints",
    "ALLOW_LOOPS": "Allow looped skeletons",
    "DO_TRACKING": "Enable 3D tracking",
    "TRACKING_BACKEND": "Tracking backend",
    "TRACK_MAX_DIST_UM": "Maximum slice-to-slice centroid jump (um)",
    "TRACK_MAX_GAP_SLICES": "Maximum missing slices in one track",
    "TRACK_BBOX_PADDING_PX": "Bounding-box overlap padding (px)",
    "CONSERVATIVE_MAX_WIDTH_JUMP_RATIO": "Maximum width change ratio",
    "CONSERVATIVE_MAX_LENGTH_JUMP_RATIO": "Maximum length change ratio",
    "CONSERVATIVE_MAX_AREA_JUMP_RATIO": "Maximum area change ratio",
    "CONSERVATIVE_MAX_TORTUOSITY_JUMP": "Maximum tortuosity change",
    "CONSERVATIVE_MAX_CENTROID_JUMP_UM": "Hard centroid jump stop-rule (um)",
    "OVERLAP_STABILITY_THRESHOLD": "Overlap stability tolerance",
    "OVERLAP_ORIENTATION_DEG": "Maximum stable orientation difference (deg)",
    "OVERLAP_MULTIPLIER": "Overlap relaxation multiplier",
    "OVERLAP_MIN_STABLE_COUNT": "Minimum stable features for overlap continuation",
    "ASSIGNMENT_MAX_COST": "Assignment maximum accepted cost",
    "ASSIGNMENT_DIST_WEIGHT": "Assignment centroid-distance weight",
    "ASSIGNMENT_OVERLAP_WEIGHT": "Assignment non-overlap penalty weight",
    "ASSIGNMENT_LENGTH_WEIGHT": "Assignment length-change weight",
    "ASSIGNMENT_WIDTH_WEIGHT": "Assignment width-change weight",
    "ASSIGNMENT_AREA_WEIGHT": "Assignment area-change weight",
    "ASSIGNMENT_ANGLE_WEIGHT": "Assignment orientation-change weight",
    "HYBRID_REPAIR_MAX_COST": "Hybrid repair maximum accepted cost",
    "HYBRID_REPAIR_MAX_GAP_SLICES": "Hybrid repair maximum missing slices",
    "HYBRID_REPAIR_MAX_FRAGMENT_SLICES": "Hybrid repair maximum fragment size",
    "HYBRID_REPAIR_MAX_LINK_DIST_UM": "Hybrid repair maximum link distance (um)",
    "HYBRID_REPAIR_MIN_OVERLAP": "Hybrid repair minimum bounding-box overlap",
    "HYBRID_REPAIR_MAX_FINAL_LENGTH_UM": "Hybrid repair maximum merged 3D length (um)",
    "SEGMENTATION_ENGINE": "Segmentation engine",
    "UNET_MODEL_PATH": "U-Net checkpoint path",
    "UNET_THRESHOLD_MODE": "U-Net threshold mode",
    "UNET_CANDIDATE_THRESHOLD": "U-Net candidate threshold",
    "UNET_SEED_THRESHOLD": "U-Net seed threshold",
    "UNET_CONTEXT_MODE": "U-Net 2.5D context",
    "UNET_INFERENCE_MODE": "U-Net inference mode",
    "UNET_TILE_SIZE": "U-Net tile size (px)",
    "UNET_TILE_OVERLAP": "U-Net tile overlap (px)",
    "UNET_ROI_PADDING_PX": "U-Net ROI padding (px)",
    "UNET_RESCUE_ENABLE": "Enable U-Net rescue lane",
    "UNET_RESCUE_THRESHOLD": "U-Net rescue probability threshold",
    "UNET_RESCUE_EXCLUDE_DILATION_PX": "Rescue exclusion around Saturn hits (px)",
    "UNET_RESCUE_MIN_COMPONENT_PX": "Minimum rescue component size (px)",
    "UNET_RESCUE_MAX_ADDITIONS_PER_SLICE": "Maximum rescued detections per slice",
    "UNET_TRACKING_SUPPORT": "Use U-Net evidence for 3D linking",
    "ASSIGNMENT_UNET_SUPPORT_WEIGHT": "Assignment U-Net support penalty",
    "ASSIGNMENT_UNET_CONTINUITY_WEIGHT": "Assignment U-Net continuity penalty",
    "HYBRID_REPAIR_UNET_SUPPORT_WEIGHT": "Hybrid repair U-Net support penalty",
    "AUDIT_MAX_LENGTH_UM": "Audit: maximum 3D length (um)",
    "AUDIT_MAX_TORTUOSITY": "Audit: maximum 3D tortuosity",
    "AUDIT_MAX_THICKNESS_UM": "Audit: maximum effective thickness (um, PSF-sensitive)",
    "AUDIT_MAX_TAPER_RATIO": "Audit: maximum taper ratio (PSF-sensitive)",
    "AUDIT_EXTREME_THICKNESS_UM": "Candidate hard-fail: extreme thickness (um)",
    "AUDIT_EXTREME_TAPER_RATIO": "Candidate hard-fail: extreme taper ratio",
    "AUDIT_MIN_SLICES": "Audit: minimum slices required (1 recommended here)"
}

PARAM_DESCRIPTIONS = {
    "UM_PER_PX_XY": "What it affects: all lateral measurements. Increase only if the microscope pixel size really is larger. Wrong values scale every reported x/y length, width, and distance.",
    "UM_PER_SLICE_Z": "What it affects: z-span, 3D length, pitch angle, and volume. Increase only if the physical slice spacing is larger. This does not change segmentation, only measurement scaling.",
    "CLAHE_CLIP": "What it affects: local contrast before thresholding. Increase to pull out dim nuclei, but too high amplifies noise and halos. Decrease if background texture starts getting segmented.",
    "CLAHE_KERNEL": "What it affects: how local the contrast correction is. Smaller values react to finer local contrast; larger values give smoother correction across the field.",
    "BG_SIGMA": "What it affects: background haze removal. Increase to subtract broader blur/haze; decrease if real diffuse signal is being removed together with background.",
    "THRESHOLD_HI": "What it affects: the strongest ridge pixels used as confident seeds. Raise it for stricter detection; lower it to keep weaker nuclei. High impact on raw detection counts.",
    "THRESHOLD_LO": "What it affects: how far masks expand outward from the confident seeds. Raise it for tighter masks; lower it for more inclusive masks. Must stay below THRESHOLD_HI.",
    "CLOSE_RADIUS": "What it affects: tiny breaks in the thresholded mask. Increase slightly to reconnect small gaps; too high can fuse neighboring nuclei.",
    "MIN_HOLE_AREA": "What it affects: small dark holes inside detected objects. Increase to fill more internal holes; decrease if thin structures are being overfilled.",
    "MIN_OBJ_PX": "What it affects: debris removal. Increase to discard more tiny specks; decrease only if real nuclei are being lost because they are very small.",
    "MAX_BRIDGE_PX": "What it affects: skeleton fragmentation. Increase to reconnect nearby broken skeleton ends; too high can join neighboring nuclei and create long false chains.",
    "MAX_BRANCH_LEN_PX": "What it affects: tiny branch spurs. Increase to prune more side spikes; too high can shorten real tips.",
    "BREAK_JUNCTIONS": "What it affects: dense skeleton webs. True = cut branch intersections so complex tangles split into simpler strands. This is high impact and can over-fragment if used too aggressively.",
    "USE_EARLY_SHAPE_FILTER": "What it affects: pre-skeleton removal of round blobs. Turn on only if round debris is a major problem. It does not fix tracking; it changes what enters tracking.",
    "MIN_ECCENTRICITY": "Early shape filter only. Higher values keep only more elongated objects; lower values allow rounder objects to survive.",
    "MAX_MINOR_PX": "Early shape filter only. Lower values enforce thinner objects; higher values allow thicker blobs through.",
    "MIN_AXIS_RATIO": "Early shape filter only. Higher values require stronger elongation; lower values allow shorter/fatter objects.",
    "MIN_MAJOR_PX": "Early shape filter only. Rejects very short components before skeletonization.",
    "MIN_SKEL_LEN_PX": "What it affects: short fragment rejection in 2D. Increase to remove tiny fragments and debris; decrease if real short nuclei fragments are being lost.",
    "MAX_GEODESIC_LEN_PX": "What it affects: very long 2D fragments. Lower values reject long merged chains; higher values keep more long objects but risk allowing fused webs.",
    "MAX_WIDTH_PX": "What it affects: thick 2D fragments. Lower values remove broad fused objects; higher values allow thicker candidates to pass.",
    "MIN_LENGTH_WIDTH_RATIO": "What it affects: how rod-like an object must be. Increase to reject blobs/fusions; decrease if real nuclei are slightly thicker or shorter than expected.",
    "MAX_BRANCH_NODES": "What it affects: branching complexity. Lower values reject tangled skeletons; higher values tolerate more branching and may admit fused objects.",
    "MAX_TORTUOSITY": "What it affects: very curved 2D fragments. Lower values are stricter; higher values tolerate more bending. Useful for removing tangled chains.",
    "MAX_ENDPOINT_COUNT": "What it affects: cluttered skeleton topology. Lower values reject objects with many tips; higher values tolerate more fragmented-looking shapes.",
    "ALLOW_LOOPS": "If True, looped skeletons can still be measured. Usually safe to leave on unless loops are a known artifact source.",
    "DO_TRACKING": "What it affects: whether 2D detections are linked into 3D tracks. Turn off only for debugging pure per-slice detection.",
    "TRACKING_BACKEND": "Which tracking engine to use. v5.5 default is hybrid_repair: legacy tracking first, followed by a conservative fragment-repair pass. Other options are legacy and global_assignment.",
    "TRACK_MAX_DIST_UM": "What it affects: fragmentation versus false merging across z. Tuned by the evolutionary optimizer. Increase if the same nucleus is failing to link between slices; decrease if neighboring nuclei are being fused into one track. This is one of the most important tracking parameters.",
    "TRACK_MAX_GAP_SLICES": "What it affects: tolerance for missed slices. Increase to allow tracks to bridge one or more blank slices; too high can stitch unrelated objects.",
    "TRACK_BBOX_PADDING_PX": "What it affects: overlap matching sensitivity. Tuned by the evolutionary optimizer. Increase to count near-touching detections as overlapping; too high can make crowded nuclei look like one track.",
    "CONSERVATIVE_MAX_WIDTH_JUMP_RATIO": "What it affects: whether sudden width changes are allowed inside a track. Tuned by the evolutionary optimizer. Increase if real tracks are breaking on modest width changes; decrease if fused tracks are slipping through.",
    "CONSERVATIVE_MAX_LENGTH_JUMP_RATIO": "What it affects: whether sudden length changes are allowed inside a track. Tuned by the evolutionary optimizer. Lower values reduce false merges; higher values reduce fragmentation.",
    "CONSERVATIVE_MAX_AREA_JUMP_RATIO": "What it affects: how much 2D area can change from one slice to the next. Tuned by the evolutionary optimizer. Useful for preventing track jumps in crowded regions.",
    "CONSERVATIVE_MAX_TORTUOSITY_JUMP": "What it affects: abrupt curvature change between linked detections. Increase only if real nuclei bend more than expected across slices.",
    "CONSERVATIVE_MAX_CENTROID_JUMP_UM": "Hard stop-rule for large centroid jumps when overlap evidence is weak. Tuned by the evolutionary optimizer. Increase carefully if tracking is fragmented; too high can cause track hopping.",
    "OVERLAP_STABILITY_THRESHOLD": "What it affects: how similar width/area/length must remain when overlap suggests the same object. Tuned by the evolutionary optimizer. Higher values reduce fragmentation; lower values are stricter.",
    "OVERLAP_ORIENTATION_DEG": "What it affects: how much the local orientation can rotate between slices while still counting as stable. Tuned by the evolutionary optimizer. Increase if slight angular jitter is breaking tracks.",
    "OVERLAP_MULTIPLIER": "What it affects: how much continuation rules are relaxed when overlap is strong and the object looks stable. Tuned by the evolutionary optimizer. Higher values are more forgiving, but can increase false continuation.",
    "OVERLAP_MIN_STABLE_COUNT": "What it affects: how many stable features must agree before an overlapping candidate continues a track. Lower values reduce fragmentation; higher values are stricter.",
    "ASSIGNMENT_MAX_COST": "Global-assignment tracker only. Candidate links above this total cost are rejected. Lower values reduce false links; higher values reduce fragmentation.",
    "ASSIGNMENT_DIST_WEIGHT": "Global-assignment and hybrid repair weight for centroid displacement between slices.",
    "ASSIGNMENT_OVERLAP_WEIGHT": "Global-assignment and hybrid repair penalty for weak/no bounding-box overlap.",
    "ASSIGNMENT_LENGTH_WEIGHT": "Global-assignment and hybrid repair penalty for sudden 2D length changes.",
    "ASSIGNMENT_WIDTH_WEIGHT": "Global-assignment and hybrid repair penalty for sudden width changes.",
    "ASSIGNMENT_AREA_WEIGHT": "Global-assignment and hybrid repair penalty for sudden area changes.",
    "ASSIGNMENT_ANGLE_WEIGHT": "Global-assignment and hybrid repair penalty for orientation changes between slices.",
    "HYBRID_REPAIR_MAX_COST": "V5.6 ROI-ADAPTIVE repair only. Candidate fragment merges above this cost are rejected. Lower values are safer and leave more fragments split.",
    "HYBRID_REPAIR_MAX_GAP_SLICES": "V5.6 ROI-ADAPTIVE repair only. Maximum blank slices allowed between two fragments being considered for repair.",
    "HYBRID_REPAIR_MAX_FRAGMENT_SLICES": "V5.6 ROI-ADAPTIVE repair only. At least one side of a repaired link must have this many slices or fewer, so the pass targets fragments rather than rewriting stable tracks.",
    "HYBRID_REPAIR_MAX_LINK_DIST_UM": "V5.6 ROI-ADAPTIVE repair only. Hard distance gate for fragment merges unless there is direct bounding-box overlap evidence.",
    "HYBRID_REPAIR_MIN_OVERLAP": "V5.6 ROI-ADAPTIVE repair only. Preferred minimum bounding-box overlap for a repair link. Very close non-overlap links can still pass if the total cost is low.",
    "HYBRID_REPAIR_MAX_FINAL_LENGTH_UM": "V5.6 ROI-ADAPTIVE repair only. Rejects a proposed merge if the estimated merged 3D length would exceed this value.",
    "SEGMENTATION_ENGINE": "v5.7 scaffold. classical_saturn keeps the existing pipeline. hybrid or unet_assisted allows optional U-Net probability evidence to support candidate detection and 3D linking.",
    "UNET_MODEL_PATH": "Path to a trained 2.5D U-Net checkpoint. Leave blank to keep Saturn fully classical.",
    "UNET_THRESHOLD_MODE": "soft keeps U-Net output as probability evidence; hard turns the probability map into a binary candidate mask. Soft is safer while the model is still being validated.",
    "UNET_CANDIDATE_THRESHOLD": "Low probability cutoff for inclusive U-Net candidate support. Lower values recover faint nuclei but produce more candidates for downstream biological QC.",
    "UNET_SEED_THRESHOLD": "Higher probability cutoff for confident U-Net seed support. This should usually stay above the candidate threshold.",
    "UNET_CONTEXT_MODE": "How neighboring z-slices are presented to the model. z_minus_z_z_plus uses previous/current/next slices as a 2.5D input.",
    "UNET_INFERENCE_MODE": "roi_tiled runs U-Net only on ROI-aware tiles and stitches probabilities back into full-frame coordinates.",
    "UNET_TILE_SIZE": "Tile width/height sent to the U-Net. Smaller tiles zoom into local detail less by themselves, but reduce memory and keep nuclei prominent in each crop.",
    "UNET_TILE_OVERLAP": "Overlap between tiles. More overlap reduces edge artifacts during stitched inference but costs more GPU time.",
    "UNET_ROI_PADDING_PX": "Extra context around the selected ROI when preparing U-Net tiles. Helps avoid boundary artifacts without letting off-ROI tissue influence output.",
    "UNET_RESCUE_ENABLE": "If enabled, U-Net high-probability regions not already covered by accepted Saturn detections are skeletonized and measured as a separate rescue lane.",
    "UNET_RESCUE_THRESHOLD": "Probability cutoff for the rescue lane. Higher values rescue fewer, more confident U-Net detections; lower values increase sensitivity but can add fragments.",
    "UNET_RESCUE_EXCLUDE_DILATION_PX": "How far to dilate accepted Saturn skeletons before searching for U-Net-only missed detections. Increase to avoid duplicate detections around existing nuclei.",
    "UNET_RESCUE_MIN_COMPONENT_PX": "Minimum binary component size before U-Net rescue skeletonization. Increase to suppress tiny U-Net specks; decrease to recover very faint fragments.",
    "UNET_RESCUE_MAX_ADDITIONS_PER_SLICE": "Optional cap on rescued objects per slice. Set to 0 for no cap. Use only if visual review shows the rescue lane is too permissive.",
    "UNET_TRACKING_SUPPORT": "If enabled in hybrid/U-Net mode, per-detection U-Net probabilities can reduce confidence in weak 3D links and favor links with consistent model support.",
    "ASSIGNMENT_UNET_SUPPORT_WEIGHT": "Global-assignment penalty for linking detections with weak U-Net support. Set to 0 to ignore U-Net evidence during assignment.",
    "ASSIGNMENT_UNET_CONTINUITY_WEIGHT": "Global-assignment penalty for abrupt U-Net probability changes across adjacent slices.",
    "HYBRID_REPAIR_UNET_SUPPORT_WEIGHT": "Hybrid fragment-repair penalty for repairing tracks through weak U-Net support. Set to 0 if U-Net evidence is too conservative.",
    "AUDIT_MAX_LENGTH_UM": "Audit only. Tracks longer than this are flagged after tracking. This does not change segmentation or tracking itself; it changes the audit-passed subset.",
    "AUDIT_MAX_TORTUOSITY": "Audit only. Flags unusually curved 3D tracks. Lower values make the quality set stricter; higher values keep more bent nuclei.",
    "AUDIT_MAX_THICKNESS_UM": "Audit only. Flags tracks that look too thick for a nucleus. PSF-sensitive: use mainly for relative WT-versus-mutant comparison under matched imaging settings rather than as a literal physical diameter cutoff.",
    "AUDIT_MAX_TAPER_RATIO": "Audit only. Flags tracks with extreme change from thickest to thinnest slice. PSF-sensitive and area-derived: useful for instability screening and relative comparison, not as a literal anatomical ratio.",
    "AUDIT_EXTREME_THICKNESS_UM": "Biological-candidate audit only. Tracks above this very high effective-thickness threshold hard-fail the candidate tier; ordinary thick tracks remain warning-only.",
    "AUDIT_EXTREME_TAPER_RATIO": "Biological-candidate audit only. Tracks above this very high taper threshold hard-fail the candidate tier; ordinary taper tracks remain warning-only.",
    "AUDIT_MIN_SLICES": "Audit only. Minimum number of slices required to avoid the single-slice flag. For this Leica SP8 stack (z-step ~1.04 um), set to 1 because mature Drosophila sperm nuclei can be biologically valid even when they appear in a single optical slice."
}

PARAM_TITLES.update({
    "UM_PER_PX_XY": "XY pixel size (um / pixel)",
    "UM_PER_SLICE_Z": "Z step size (um / slice)",
    "TRACK_MAX_DIST_UM": "Maximum slice-to-slice centroid jump (um)",
    "CONSERVATIVE_MAX_CENTROID_JUMP_UM": "Hard centroid jump stop-rule (um)",
    "AUDIT_MAX_LENGTH_UM": "Audit: maximum 3D length (um)",
    "AUDIT_MAX_THICKNESS_UM": "Audit: maximum effective thickness (um, PSF-sensitive)",
    "AUDIT_EXTREME_THICKNESS_UM": "Candidate hard-fail: extreme thickness (um)",
    "AUDIT_EXTREME_TAPER_RATIO": "Candidate hard-fail: extreme taper ratio",
})

PARAM_DESCRIPTIONS.update({
    "UM_PER_SLICE_Z": (
        "What it affects: z-span, z-covered, 3D length, pitch angle, and volume. "
        "Increase only if the physical slice spacing is larger. This does not change "
        "segmentation, only measurement scaling."
    ),
    "CONSERVATIVE_MAX_AREA_JUMP_RATIO": (
        "What it affects: how much estimated 2D area can change from one slice to the next. "
        "Area is estimated as geodesic length times median width, not raw skeleton pixel count. "
        "Tuned by the evolutionary optimizer and useful for preventing track jumps in crowded regions."
    ),
    "OVERLAP_STABILITY_THRESHOLD": (
        "What it affects: how similar width, estimated area, and length must remain when overlap "
        "suggests the same object. Higher values reduce fragmentation; lower values are stricter."
    ),
    "AUDIT_MAX_LENGTH_UM": (
        "Audit only. Tracks longer than this projection-length-plus-Z 3D estimate are flagged after "
        "tracking. This does not change segmentation or tracking itself; it changes the audit-passed subset."
    ),
    "AUDIT_MAX_THICKNESS_UM": (
        "Audit only. Flags tracks with a large cylinder-equivalent thickness derived from estimated volume "
        "and 3D length. This is PSF-sensitive, so use mainly for relative comparisons under matched imaging "
        "settings rather than as a literal physical diameter cutoff."
    ),
    "AUDIT_MAX_TAPER_RATIO": (
        "Audit only. Flags tracks with extreme max/min estimated area across slices. Area is derived from "
        "length times median width, so this is PSF-sensitive and best used for instability screening."
    ),
    "AUDIT_MIN_SLICES": (
        "Audit only. Minimum number of detected slices required to avoid the single-slice flag. For this "
        "Leica SP8 stack (z-step ~1.04 um), set to 1 because mature Drosophila sperm nuclei can be "
        "biologically valid even when they appear in a single optical slice."
    ),
    "AUDIT_EXTREME_THICKNESS_UM": (
        "Candidate tier only. Effective thickness above this high threshold becomes a hard fail. "
        "Values above AUDIT_MAX_THICKNESS_UM but below this remain warning-only because thickness is PSF-sensitive."
    ),
    "AUDIT_EXTREME_TAPER_RATIO": (
        "Candidate tier only. Taper above this high threshold becomes a hard fail. "
        "Values above AUDIT_MAX_TAPER_RATIO but below this remain warning-only because taper is PSF-sensitive."
    ),
})

TRACKING_TUNER_EXPLANATION = (
    "Several tracking parameters in this section were originally selected by an evolutionary tuning script. "
    "That script rewarded multi-slice continuity while penalizing fragmentation, implausible merges, and several biology-aware outlier categories. "
    "These values are strong starting points, but they can still be adjusted for a new dataset."
)

AUDIT_EXPLANATION = (
    "Candidate audit is applied after 3D tracking. It does not delete tracks or change raw segmentation. "
    "Instead, it labels completed tracks as biological candidates, warning-only candidates, hard fails, and strict no-warning quality tracks. "
    "Use biological candidates as the main analysis population; use strict quality as a conservative diagnostic subset."
)



class ParameterEditor(tk.Toplevel):
    """Interactive configuration editor with grouped sections and mouse-wheel scrolling."""
    def __init__(self, parent, current_config, default_config, apply_callback):
        super().__init__(parent)
        self.title("Parameter Configuration")
        self.geometry("1180x760")
        self.minsize(980, 620)
        self.current_config = current_config
        self.default_config = default_config
        self.apply_callback = apply_callback
        self.entries = {}

        tools = tk.Frame(self, bg='#e0e0e0', padx=10, pady=10)
        tools.pack(side='top', fill='x')

        tk.Button(tools, text="Apply to Session", command=self.apply, bg="#d4edda", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(tools, text="Reset to Defaults", command=self.reset_defaults, bg="#f8d7da").pack(side="left", padx=5)
        tk.Button(tools, text="Load JSON", command=self.load_json).pack(side="left", padx=5)
        tk.Button(tools, text="Save JSON", command=self.save_json).pack(side="left", padx=5)
        tk.Label(tools, text="Grouped sections with mouse-wheel / trackpad scrolling. Audit rules are listed separately.", bg='#e0e0e0', fg='dimgray').pack(side="right", padx=8)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")

        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        self.populate_form(self.current_config)

    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=max(event.width - 4, 900))

    def _bind_mousewheel(self, event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_linux_scroll_up)
        self.canvas.bind_all("<Button-5>", self._on_linux_scroll_down)

    def _unbind_mousewheel(self, event=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Shift-MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        delta = event.delta
        if delta == 0:
            return
        step = -1 * int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self.canvas.yview_scroll(step, "units")

    def _on_shift_mousewheel(self, event):
        delta = event.delta
        if delta == 0:
            return
        step = -1 * int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self.canvas.xview_scroll(step, "units")

    def _on_linux_scroll_up(self, event):
        self.canvas.yview_scroll(-1, "units")

    def _on_linux_scroll_down(self, event):
        self.canvas.yview_scroll(1, "units")

    def populate_form(self, cfg):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.entries = {}
        row = 0

        intro = tk.Label(
            self.scrollable_frame,
            text=(
                "Edit numeric and boolean parameters below. Descriptions explain what each threshold does biologically or computationally.\n"
                + AUDIT_EXPLANATION
            ),
            justify="left", anchor="w", fg="#333333", wraplength=1040, pady=8
        )
        intro.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=(8, 12))
        row += 1

        displayed = set()
        for section, keys in PARAM_SECTIONS.items():
            section_frame = tk.Frame(self.scrollable_frame, bg="#f6f8fb", bd=1, relief="groove")
            section_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=(4, 10))
            section_frame.grid_columnconfigure(2, weight=1)

            tk.Label(section_frame, text=section, font=("Arial", 11, "bold"), bg="#dce6f7", anchor="w", padx=8, pady=6).grid(row=0, column=0, columnspan=3, sticky="ew")

            if section == "3D Tracking":
                track_note = tk.Label(
                    section_frame,
                    text="These settings control how 2D detections are linked into 3D nuclei. Several of them were originally selected by an evolutionary tuning script, so treat them as tuned starting points rather than arbitrary defaults.",
                    fg="dimgray", bg="#f6f8fb", justify="left", anchor="w", wraplength=980
                )
                track_note.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(6, 8))
                local_row = 2
            elif section == "Candidate Audit (post-tracking)":
                audit_note = tk.Label(
                    section_frame,
                    text="Audit labels suspicious tracks after tracking. It should annotate the population, and reports should distinguish all tracks from audit-passed tracks. Audit thresholds do not change segmentation or 3D linkage.",
                    fg="dimgray", bg="#f6f8fb", justify="left", anchor="w", wraplength=980
                )
                audit_note.grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(6, 8))
                local_row = 2
            else:
                local_row = 1

            for k in keys:
                if k not in cfg or k not in PARAM_DESCRIPTIONS or not isinstance(cfg[k], (int, float, bool)):
                    continue
                displayed.add(k)
                v = cfg[k]
                label_txt = f"{PARAM_TITLES.get(k, k)}\n[{k}]"
                tk.Label(section_frame, text=label_txt, font=("Arial", 10, "bold"), width=34, anchor="e", justify="right", bg="#f6f8fb").grid(row=local_row, column=0, padx=(8, 10), pady=4, sticky="e")
                var = tk.StringVar(value=str(v))
                ent = tk.Entry(section_frame, textvariable=var, width=16)
                ent.grid(row=local_row, column=1, padx=(0, 10), pady=4, sticky="w")
                self.entries[k] = (var, type(v))
                desc = tk.Label(section_frame, text=PARAM_DESCRIPTIONS[k], fg="dimgray", bg="#f6f8fb", anchor="w", justify="left", wraplength=760)
                desc.grid(row=local_row, column=2, padx=(0, 10), pady=4, sticky="w")
                local_row += 1

            row += 1

        remaining = [k for k, v in cfg.items() if k not in displayed and isinstance(v, (int, float, bool))]
        if remaining:
            extra = tk.Frame(self.scrollable_frame, bg="#fdfdfd", bd=1, relief="groove")
            extra.grid(row=row, column=0, columnspan=3, sticky="ew", padx=10, pady=(4, 10))
            extra.grid_columnconfigure(2, weight=1)
            tk.Label(extra, text="Other Numeric / Boolean Parameters", font=("Arial", 11, "bold"), bg="#ececec", anchor="w", padx=8, pady=6).grid(row=0, column=0, columnspan=3, sticky="ew")
            local_row = 1
            for k in remaining:
                v = cfg[k]
                label_txt = f"{PARAM_TITLES.get(k, k)}\n[{k}]"
                tk.Label(extra, text=label_txt, font=("Arial", 10, "bold"), width=34, anchor="e", justify="right", bg="#fdfdfd").grid(row=local_row, column=0, padx=(8, 10), pady=4, sticky="e")
                var = tk.StringVar(value=str(v))
                ent = tk.Entry(extra, textvariable=var, width=16)
                ent.grid(row=local_row, column=1, padx=(0, 10), pady=4, sticky="w")
                self.entries[k] = (var, type(v))
                tk.Label(extra, text="No extended description registered yet. Use caution and compare with nearby parameters in the same stage.", fg="dimgray", bg="#fdfdfd", anchor="w", justify="left", wraplength=760).grid(row=local_row, column=2, padx=(0, 10), pady=4, sticky="w")
                local_row += 1

    def apply(self):
        new_cfg = self.current_config.copy()
        try:
            for k, (var, t) in self.entries.items():
                val = var.get().strip()
                if t == bool:
                    new_cfg[k] = val.lower() in ['true', '1', 't', 'y', 'yes']
                else:
                    new_cfg[k] = t(val)
        except ValueError as e:
            messagebox.showerror("Validation Error", f"Invalid input format: {e}")
            return

        self.apply_callback(new_cfg)
        self.destroy()

    def reset_defaults(self):
        self.populate_form(self.default_config)

    def load_json(self):
        fpath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if fpath:
            try:
                import json
                with open(fpath, 'r') as f:
                    loaded = json.load(f)
                filtered = {k: v for k, v in loaded.items() if k in self.current_config}
                temp = self.current_config.copy()
                temp.update(filtered)
                self.populate_form(temp)
            except Exception as e:
                messagebox.showerror("Load Error", str(e))

    def save_json(self):
        fpath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if fpath:
            try:
                import json
                new_cfg = self.current_config.copy()
                for k, (var, t) in self.entries.items():
                    val = var.get().strip()
                    if t == bool:
                        new_cfg[k] = val.lower() in ['true', '1', 't', 'y', 'yes']
                    else:
                        new_cfg[k] = t(val)
                with open(fpath, 'w') as f:
                    json.dump(new_cfg, f, indent=4)
                messagebox.showinfo("Saved", f"Parameters saved to {os.path.basename(fpath)}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))

# =============================================================================
# V5.2 ADVANCED AI BIOLOGICAL INTERPRETATION ENGINE
# =============================================================================

SPECIES_PROFILES = {
    "D. melanogaster": {
        "length_target": "10.0 um",
        "thickness": "0.3-0.4 um",
        "context": "Standard model species. Needle-like, highly condensed nucleus.",
        "heteromorphism": False
    },
    "D. simulans": {
        "length_target": "9.5-10.5 um",
        "thickness": "0.3 um",
        "context": "Close relative of D. mel, very similar nuclear profile.",
        "heteromorphism": False
    },
    "D. yakuba": {
        "length_target": "10.0-11.0 um",
        "thickness": "0.35 um",
        "context": "Close relative of D. mel, slightly different condensation timing.",
        "heteromorphism": False
    },
    "D. ananassae": {
        "length_target": "8.0-10.0 um",
        "thickness": "0.3-0.4 um",
        "context": "Melanogaster subgroup. Moderate nuclear elongation with distinct chromatin packaging.",
        "heteromorphism": False
    },
    "D. pseudoobscura (Dpse)": {
        "length_target": "Variable (Heteromorphic)",
        "thickness": "Variable",
        "context": "Produces both fertile 'eusperm' and shorter, non-fertilizing 'parasperm'.",
        "heteromorphism": True
    },
    "D. virilis (Dvir)": {
        "length_target": "15.0-18.0 um",
        "thickness": "0.4 um",
        "context": "Large species with robust, stable nuclear morphology.",
        "heteromorphism": False
    },
    "General / Evolutionary": {
        "length_target": "Unknown",
        "thickness": "Unknown",
        "context": "Perform comparative discovery mode to infer species strategy.",
        "heteromorphism": "Possible"
    }
}

SPECIES_PROFILES.update({
    "D. melanogaster": {
        "length_target": "10.0 um",
        "thickness": "0.3-0.4 um anatomical; reported thickness is PSF-sensitive",
        "context": "Standard model species. Needle-like, highly condensed nucleus.",
        "heteromorphism": False
    },
    "D. simulans": {
        "length_target": "9.5-10.5 um",
        "thickness": "0.3 um anatomical; reported thickness is PSF-sensitive",
        "context": "Close relative of D. mel, very similar nuclear profile.",
        "heteromorphism": False
    },
    "D. yakuba": {
        "length_target": "10.0-11.0 um",
        "thickness": "0.35 um anatomical; reported thickness is PSF-sensitive",
        "context": "Close relative of D. mel, slightly different condensation timing.",
        "heteromorphism": False
    },
    "D. ananassae": {
        "length_target": "8.0-10.0 um",
        "thickness": "0.3-0.4 um anatomical; reported thickness is PSF-sensitive",
        "context": "Melanogaster subgroup. Moderate nuclear elongation with distinct chromatin packaging.",
        "heteromorphism": False
    },
    "D. virilis (Dvir)": {
        "length_target": "15.0-18.0 um",
        "thickness": "0.4 um anatomical; reported thickness is PSF-sensitive",
        "context": "Large species with robust, stable nuclear morphology.",
        "heteromorphism": False
    },
})

def get_ai_biological_interpretation(csv_summary_str, species, folder_name, model_id="gemini-2.5-pro"):
    """Calls Gemini API for biological narrative. Priority: Local File > Env Var."""
    profile = SPECIES_PROFILES.get(species, SPECIES_PROFILES["General / Evolutionary"])

    if not _HAVE_REQUESTS:
        return "AI ANALYSIS SKIPPED: 'requests' library not found. Please run 'pip install requests'."

    # 1. Try local file path first
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_api_key.txt")
    api_key = ""
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            api_key = f.read().strip()

    # 2. Fallback to Env Var
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        return "AI ANALYSIS SKIPPED: No Gemini API Key found. Use 'Set API Key' in the GUI (Free at aistudio.google.com)."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    system_prompt = f"""You are a world-class Drosophila reproductive biologist and evolutionary morphologist.
    You are interpreting 3D-tracked sperm nuclei data from a confocal Z-stack batch analysis.

    BIOLOGICAL CONTEXT:
    - Target Species: {species}
    - Baseline Profile: {profile['context']}
    - Expected Length Class: {profile['length_target']}
    - Thickness Reference: {profile['thickness']}
    - Heteromorphism (Multiple Morphs): {'Yes' if profile.get('heteromorphism') else 'No'}
    - Source Data Folder: {folder_name}
    - Measurement Caveat: reported 3D length is a projection-length-plus-Z-span estimate. Volume,
      effective thickness, taper, and area-derived metrics are PSF- and sampling-sensitive; use them
      mainly for relative comparisons under matched imaging settings, not literal anatomical dimensions.

    INPUT DATA (CSV Summary):
    {csv_summary_str}

    YOUR TASK:
    1. Determine the 'Morphological Class' of this population. Does it align with the expected {species} profile?
    2. Analyze maturation: Are these likely mature motile sperm or transitionary elongating spermatids based primarily on length and track morphology, with thickness treated as PSF-sensitive supporting evidence?
    3. Identify Anomalies: Highlight outliers in tortuosity, estimated thickness, taper, or abrupt area change that may indicate fixation artifacts, segmentation instability, over-merging, or developmental defects.
    4. Evolutionary Insight: Provide 2-3 sentences on the evolutionary context of this sperm morphology.
    5. Formatting: Use professional, high-density scientific language. Use Markdown formatting.
    """
    payload = {"contents": [{"parts": [{"text": system_prompt}]}]}
    try:
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else: return f"AI API Error ({response.status_code}): {response.text}"
    except Exception as e: return f"Failed to connect to AI Service: {str(e)}"

def generate_ai_html_report(out_dir, ai_text, stats_summary, species):
    """Generates premium HTML dashboard."""
    html_path = os.path.join(out_dir, f"AI_Biological_Analysis_{_VERSION}.html")
    try:
        import markdown
        ai_html = markdown.markdown(ai_text)
    except ImportError:
        ai_html = ai_text.replace("\n", "<br>")

    stats_html = "".join([f'<div class="stat-box"><div class="stat-val">{v}</div><div class="stat-label">{k.replace("_", " ")}</div></div>' for k,v in stats_summary.items()])

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Sperm AI Analysis - {_VERSION}</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {{ --primary: #2563eb; --bg: #f8fafc; --card: rgba(255, 255, 255, 0.8); }}
            body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: #1e293b; line-height: 1.6; padding: 40px; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            .badge {{ background: #dbeafe; color: #1e40af; padding: 4px 12px; border-radius: 99px; font-size: 0.8rem; font-weight: bold; }}
            .glass-card {{ background: var(--card); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.3);
                          border-radius: 16px; padding: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 30px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .stat-box {{ background: white; padding: 15px; border-radius: 12px; text-align: center; border: 1px solid #e2e8f0; }}
            .stat-val {{ font-size: 1.5rem; font-weight: 800; color: var(--primary); }}
            .stat-label {{ font-size: 0.75rem; text-transform: uppercase; color: #64748b; }}
            .ai-content h2 {{ color: var(--primary); border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <span class="badge">Drosophila Morphometrics {_VERSION}</span>
            <h1>AI Biological Interpretation</h1>
            <p>Advanced Analysis for <strong>{species}</strong></p>
            <div class="stats-grid">{stats_html}</div>
            <div class="glass-card">
                <div class="ai-content">{ai_html}</div>
            </div>
            <footer>Generated | {time.strftime('%Y-%m-%d %H:%M:%S')} | {_VERSION} AI Layer</footer>
        </div>
    </body>
    </html>
    """
    with open(html_path, "w", encoding="utf-8") as f: f.write(html_content)
    return html_path

# =============================================================================

class SpermGUI:
    """
    The primary Tkinter-based graphical user interface for the Sperm Segmentation ROI Tool.

    The GUI provides a two-panel layout:
    - **Left sidebar** - controls for directory loading, Z-slice navigation,
      tool selection, ROI drawing, single-slice analysis, batch analysis,
      and two progress bars (2D segmentation and post-analysis reporting).
    - **Right canvas** - a matplotlib ``FigureCanvasTkAgg`` that renders the
      currently selected Z-slice image (raw, overlay, or debug) and accepts
      polygon ROI drawing interactions.

    Key interaction modes (controlled by ``mode_var``)
    ---------------------------------------------------
    - ``'view'``   - pans and inspects the raw image.
    - ``'review'`` - displays the saved overlay PNG for the current slice (requires
      a prior batch run).
    - ``'roi'``    - left-click to add polygon vertices; right-click to undo the
      last vertex; call *Finalize Polygon* to close and rasterise the mask.

    Thread architecture
    -------------------
    Batch processing (:meth:`run_batch_analysis`) runs in a background ``threading.Thread``
    so the Tkinter event loop remains responsive.  Progress bar updates are marshalled
    back to the main thread via ``root.after()`` calls.
    """
    def open_parameter_editor(self):
        """
        Opens the :class:`ParameterEditor` ``Toplevel`` window.

        Defines an ``on_apply`` callback that merges the edited values into the
        global ``CONFIG`` dict and updates the status label so the researcher
        knows the new parameters are active.
        """
        def on_apply(new_cfg):
            CONFIG.update(new_cfg)
            self.lbl_roi.config(text="Parameters updated in memory.")

        editor = ParameterEditor(self.root, CONFIG, self.default_config, on_apply)

    def _load_tuned_params(self):
        """Load a tuned parameters JSON file and merge into CONFIG."""
        from tkinter import filedialog, messagebox
        filepath = filedialog.askopenfilename(
            title="Select Tuned Parameters JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        if not filepath:
            return
        try:
            with open(filepath, 'r') as f:
                tuned = json.load(f)

            # Only update keys that exist in CONFIG (safety check)
            applied = []
            for key, value in tuned.items():
                if key in CONFIG:
                    old_val = CONFIG[key]
                    CONFIG[key] = value
                    applied.append(f"  {key}: {old_val} -> {value}")

            if applied:
                n = len(applied)
                short_name = os.path.basename(filepath)
                self.lbl_params_status.config(
                    text=f'OK Loaded {n} params from {short_name}',
                    fg='green'
                )
                detail = "\n".join(applied)
                messagebox.showinfo(
                    "Parameters Loaded",
                    f"Loaded {n} parameters from:\n{short_name}\n\n"
                    f"Changes applied:\n{detail}\n\n"
                    f"Use 'Revert Defaults' to undo."
                )
            else:
                messagebox.showwarning("No Matching Keys",
                    f"No recognized CONFIG keys found in {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load parameters:\n{e}")

    def _launch_parameter_tuner(self, mode):
        """Launch the external V5.6 ROI-ADAPTIVE tuner without blocking the GUI."""
        from tkinter import messagebox, simpledialog

        if not self.input_dir:
            messagebox.showwarning(
                "Load Directory First",
                "Load the image directory before launching the tuner."
            )
            return

        default_slices = "0-12"
        if self.files:
            z_values = []
            for f in self.files:
                m = re.search(r"z(\d+)", os.path.basename(f), re.IGNORECASE)
                if m:
                    z_values.append(int(m.group(1)))
            if z_values:
                z_values = sorted(set(z_values))
                start = z_values[0]
                end = z_values[min(len(z_values) - 1, 12)]
                default_slices = f"{start}-{end}"

        slice_str = simpledialog.askstring(
            "Tuner Slices",
            "Enter representative z-slices for tuning (for example 0-12 or 10,15,20,25):",
            initialvalue=default_slices,
            parent=self.root
        )
        if not slice_str:
            return

        tuner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "tune_parameters_Saturnv5_7.py")
        if not os.path.exists(tuner_path):
            messagebox.showerror("Tuner Missing", f"Could not find tuner:\n{tuner_path}")
            return

        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameter_tuning_results")
        ensure_dir(outdir)
        try:
            roi_mask = self.build_roi_mask()
            if roi_mask is not None:
                roi_path = os.path.join(outdir, "last_drawn_roi_saturnv5_7_tune.tif")
                tifffile.imwrite(roi_path, roi_mask.astype(np.uint8) * 255)
        except Exception as e:
            print(f"Could not export current ROI for tuner: {e}")

        cmd = [
            sys.executable,
            tuner_path,
            "--mode", mode,
            "--dir", self.input_dir,
            "--slices", slice_str,
            "--outdir", outdir,
        ]

        if mode == "segmentation":
            cmd.extend(["--maxiter", "6", "--popsize", "6", "--review-candidates", "8"])

        try:
            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), creationflags=creationflags)
            self.lbl_params_status.config(
                text=f"Started {mode} tuner for z={slice_str}",
                fg="#0f766e"
            )
            messagebox.showinfo(
                "Tuner Started",
                f"Started {mode} tuning in a separate process.\n\n"
                f"Results will be saved to:\n{outdir}\n\n"
                "When it finishes, use 'Load Tuned Params' to apply a JSON candidate."
            )
        except Exception as e:
            messagebox.showerror("Tuner Launch Error", f"Could not launch tuner:\n{e}")

    def _on_sidebar_frame_configure(self, event=None):
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox('all'))

    def _on_sidebar_canvas_configure(self, event):
        target_width = max(event.width - 4, 320)
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=target_width)

    def _bind_sidebar_widget_tree(self, widget):
        """Make mouse wheel scrolling work over buttons, labels, entries, and frames."""
        widget.bind('<MouseWheel>', self._on_sidebar_mousewheel, add='+')
        widget.bind('<Shift-MouseWheel>', self._on_sidebar_shift_mousewheel, add='+')
        widget.bind('<Button-4>', self._on_sidebar_linux_scroll_up, add='+')
        widget.bind('<Button-5>', self._on_sidebar_linux_scroll_down, add='+')
        widget.bind('<Home>', self._sidebar_scroll_top, add='+')
        widget.bind('<End>', self._sidebar_scroll_bottom, add='+')
        widget.bind('<Prior>', self._sidebar_page_up, add='+')
        widget.bind('<Next>', self._sidebar_page_down, add='+')
        for child in widget.winfo_children():
            self._bind_sidebar_widget_tree(child)

    def _bind_sidebar_mousewheel(self, event=None):
        self.sidebar_canvas.bind_all('<MouseWheel>', self._on_sidebar_mousewheel)
        self.sidebar_canvas.bind_all('<Shift-MouseWheel>', self._on_sidebar_shift_mousewheel)
        self.sidebar_canvas.bind_all('<Button-4>', self._on_sidebar_linux_scroll_up)
        self.sidebar_canvas.bind_all('<Button-5>', self._on_sidebar_linux_scroll_down)

    def _unbind_sidebar_mousewheel(self, event=None):
        self.sidebar_canvas.unbind_all('<MouseWheel>')
        self.sidebar_canvas.unbind_all('<Shift-MouseWheel>')
        self.sidebar_canvas.unbind_all('<Button-4>')
        self.sidebar_canvas.unbind_all('<Button-5>')

    def _on_sidebar_mousewheel(self, event):
        delta = getattr(event, 'delta', 0)
        if delta == 0:
            return
        step = -1 * int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self.sidebar_canvas.yview_scroll(step * 4, 'units')
        return "break"

    def _on_sidebar_shift_mousewheel(self, event):
        delta = getattr(event, 'delta', 0)
        if delta == 0:
            return
        step = -1 * int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self.sidebar_canvas.xview_scroll(step * 4, 'units')
        return "break"

    def _on_sidebar_linux_scroll_up(self, event):
        self.sidebar_canvas.yview_scroll(-4, 'units')
        return "break"

    def _on_sidebar_linux_scroll_down(self, event):
        self.sidebar_canvas.yview_scroll(4, 'units')
        return "break"

    def _sidebar_scroll_top(self, event=None):
        self.sidebar_canvas.yview_moveto(0.0)
        return "break"

    def _sidebar_scroll_bottom(self, event=None):
        self.sidebar_canvas.yview_moveto(1.0)
        return "break"

    def _sidebar_page_up(self, event=None):
        self.sidebar_canvas.yview_scroll(-1, 'pages')
        return "break"

    def _sidebar_page_down(self, event=None):
        self.sidebar_canvas.yview_scroll(1, 'pages')
        return "break"

    def _make_sidebar_section(self, parent, title, default_open=True, accent="#dbeafe"):
        """Create a compact collapsible section for the main sidebar."""
        outer = tk.Frame(parent, bg="#f0f0f0", bd=1, relief="groove")
        outer.pack(fill="x", padx=6, pady=(6, 2))

        body = tk.Frame(outer, bg="#f7f7f7")
        state = {"open": bool(default_open)}

        def sync_header():
            marker = "-" if state["open"] else "+"
            header.config(text=f"{marker} {title}")

        def toggle():
            state["open"] = not state["open"]
            if state["open"]:
                body.pack(fill="x", padx=0, pady=(0, 4))
            else:
                body.pack_forget()
            sync_header()
            self._on_sidebar_frame_configure()

        header = tk.Button(
            outer,
            command=toggle,
            anchor="w",
            bg=accent,
            relief="flat",
            font=("Arial", 9, "bold"),
            padx=8,
            pady=4
        )
        header.pack(fill="x")
        if state["open"]:
            body.pack(fill="x", padx=0, pady=(0, 4))
        sync_header()
        return body

    def _revert_to_defaults(self):
        """Revert CONFIG back to the original defaults captured at startup."""
        from tkinter import messagebox
        CONFIG.update(self.default_config)
        self.lbl_params_status.config(
            text='Using default parameters',
            fg='#555'
        )
        messagebox.showinfo("Reverted", "All parameters have been reverted to their original defaults.")

    def __init__(self, root):
        """
        Initialises the main application window, all sidebar controls, the matplotlib
        canvas, and mouse/key event bindings.

        Args:
            root (tk.Tk): The root Tkinter window created by :func:`launch_gui`.
        """
        self.root = root
        self.root.title(f'Sperm Segmentation ROI Tool - Saturn Project')
        self.root.geometry('1450x920')

        self.input_dir = ''
        self.files = []
        self.current_idx = 0
        self.current_img = None
        self.last_out_dir = ""

        self.roi_points = []
        self.drawing = False
        self.roi_active = False
        self._loaded_roi_mask = None

        # --- AI ANALYSIS CONTROLS ---
        self.species_var = tk.StringVar(value='D. melanogaster')
        self.model_var = tk.StringVar(value='Gemini 2.5 Pro (Thinking)')
        self._last_batch_ts = None
        self._last_batch_out_dir = None

        # --- Main GUI sidebar: flexibly scrollable with mouse wheel, touchpad, and horizontal support ---
        self.sidebar_container = tk.Frame(root, width=340, bg='#f0f0f0')
        self.sidebar_container.pack(side='left', fill='y')
        self.sidebar_container.pack_propagate(False)

        self.sidebar_canvas = tk.Canvas(self.sidebar_container, bg='#f0f0f0', highlightthickness=0)
        self.sidebar_scrollbar = ttk.Scrollbar(self.sidebar_container, orient='vertical', command=self.sidebar_canvas.yview)
        self.sidebar_hscrollbar = ttk.Scrollbar(self.sidebar_container, orient='horizontal', command=self.sidebar_canvas.xview)
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scrollbar.set, xscrollcommand=self.sidebar_hscrollbar.set)

        self.sidebar_quick_actions = tk.Frame(self.sidebar_container, bg='#e5e7eb', bd=1, relief='ridge')
        self.sidebar_quick_actions.pack(side='bottom', fill='x')
        tk.Label(
            self.sidebar_quick_actions,
            text='Quick Actions',
            bg='#e5e7eb',
            fg='#374151',
            font=('Arial', 8, 'bold')
        ).pack(anchor='w', padx=6, pady=(4, 0))
        quick_row1 = tk.Frame(self.sidebar_quick_actions, bg='#e5e7eb')
        quick_row1.pack(fill='x', padx=6, pady=(3, 2))
        tk.Button(
            quick_row1,
            text='Run Slice',
            command=self.run_analysis_slice,
            width=10
        ).pack(side='left', expand=True, fill='x', padx=(0, 2))
        tk.Button(
            quick_row1,
            text='Run Batch',
            command=self.run_batch_analysis,
            bg='#d4edda',
            font=('Arial', 9, 'bold'),
            width=10
        ).pack(side='right', expand=True, fill='x', padx=(2, 0))
        quick_row2 = tk.Frame(self.sidebar_quick_actions, bg='#e5e7eb')
        quick_row2.pack(fill='x', padx=6, pady=(2, 6))
        tk.Button(quick_row2, text='Top', command=self._sidebar_scroll_top, width=6).pack(side='left', fill='x', padx=(0, 2))
        tk.Button(quick_row2, text='Bottom', command=self._sidebar_scroll_bottom, width=7).pack(side='left', fill='x', padx=(2, 2))
        tk.Button(quick_row2, text='Save ROI', command=self.save_roi_mask, width=8).pack(side='left', expand=True, fill='x', padx=(2, 2))
        tk.Button(quick_row2, text='Load ROI', command=self.load_roi_mask, width=8).pack(side='right', expand=True, fill='x', padx=(2, 0))

        self.sidebar_canvas.pack(side='left', fill='both', expand=True)
        self.sidebar_scrollbar.pack(side='right', fill='y')
        self.sidebar_hscrollbar.pack(side='bottom', fill='x')

        self.sidebar = tk.Frame(self.sidebar_canvas, bg='#f0f0f0')
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=self.sidebar, anchor='nw')

        self.sidebar.bind("<Configure>", self._on_sidebar_frame_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        self.sidebar_canvas.bind('<Enter>', self._bind_sidebar_mousewheel)
        self.sidebar_canvas.bind('<Leave>', self._unbind_sidebar_mousewheel)
        self.sidebar_container.bind('<Enter>', self._bind_sidebar_mousewheel)
        self.sidebar_container.bind('<Leave>', self._unbind_sidebar_mousewheel)
        # -----------------------------------------------------------------------------------------------

        self.default_config = CONFIG.copy()
        self.mode_var = tk.StringVar(value='view')

        data_section = self._make_sidebar_section(self.sidebar, "Data", default_open=True, accent="#dbeafe")
        tk.Button(data_section, text='Load Directory', command=self.load_directory, height=2).pack(fill='x', padx=8, pady=(8, 4))
        self.lbl_status = tk.Label(data_section, text='No directory loaded', wraplength=260, justify='left', bg="#f7f7f7", fg="#374151")
        self.lbl_status.pack(fill='x', padx=8, pady=(0, 8))

        params_section = self._make_sidebar_section(self.sidebar, "Parameters", default_open=True, accent="#e2e3e5")
        tk.Button(params_section, text='Configure Parameters', command=self.open_parameter_editor, bg='#e2e3e5').pack(fill='x', padx=8, pady=(8, 4))
        params_frame = tk.Frame(params_section, bg='#f7f7f7')
        params_frame.pack(fill='x', padx=8, pady=(0, 4))
        tk.Button(params_frame, text='Load Tuned Params', command=self._load_tuned_params, bg='#d4edda', width=16).pack(side='left', expand=True, fill='x', padx=(0, 2))
        tk.Button(params_frame, text='Revert Defaults', command=self._revert_to_defaults, bg='#f8d7da', width=14).pack(side='right', expand=True, fill='x', padx=(2, 0))
        self.lbl_params_status = tk.Label(params_section, text='Using default parameters', wraplength=260, justify='left', fg='#555', bg="#f7f7f7", font=('Arial', 8))
        self.lbl_params_status.pack(fill='x', padx=8, pady=(0, 8))

        tuning_section = self._make_sidebar_section(self.sidebar, "Tuning", default_open=True, accent="#cffafe")
        tuner_frame = tk.Frame(tuning_section, bg='#f7f7f7')
        tuner_frame.pack(fill='x', padx=8, pady=8)
        tk.Button(
            tuner_frame,
            text='Tune Segmentation',
            command=lambda: self._launch_parameter_tuner("segmentation"),
            bg='#cffafe',
            width=16
        ).pack(side='left', expand=True, fill='x', padx=(0, 2))
        tk.Button(
            tuner_frame,
            text='Tune Tracking',
            command=lambda: self._launch_parameter_tuner("tracking"),
            bg='#e0e7ff',
            width=14
        ).pack(side='right', expand=True, fill='x', padx=(2, 0))

        nav_section = self._make_sidebar_section(self.sidebar, "Z Navigation", default_open=True, accent="#fef3c7")
        self.scale_z = tk.Scale(nav_section, from_=0, to=0, orient='horizontal', command=self.on_slide_change, bg="#f7f7f7", highlightthickness=0)
        self.scale_z.pack(fill='x', padx=8, pady=(8, 2))
        self.lbl_z = tk.Label(nav_section, text='Z: 0 / 0', bg="#f7f7f7")
        self.lbl_z.pack(pady=(0, 8))

        tools_section = self._make_sidebar_section(self.sidebar, "View / ROI Drawing", default_open=True, accent="#fde68a")
        tk.Radiobutton(tools_section, text='View/Nav (Raw Image)', variable=self.mode_var, value='view', command=self.render, bg="#f7f7f7").pack(anchor='w', padx=10, pady=(6, 0))
        tk.Radiobutton(tools_section, text='Review Overlays (After Batch)', variable=self.mode_var, value='review', command=self.render, bg="#f7f7f7").pack(anchor='w', padx=10)
        tk.Radiobutton(tools_section, text='Draw ROI (Polygon)', variable=self.mode_var, value='roi', command=self.render, bg="#f7f7f7").pack(anchor='w', padx=10)
        tk.Label(tools_section, text='Left-click points, right-click undo', font=('Arial', 8, 'italic'), fg='dimgray', bg="#f7f7f7").pack(fill='x', padx=10, pady=(2, 4))
        tk.Button(tools_section, text='Finalize Polygon', command=self.finalize_roi, bg='#ffeeba').pack(fill='x', padx=20, pady=(0, 8))

        roi_section = self._make_sidebar_section(self.sidebar, "ROI Files", default_open=True, accent="#dcfce7")
        roi_buttons = tk.Frame(roi_section, bg="#f7f7f7")
        roi_buttons.pack(fill='x', padx=8, pady=(8, 4))
        tk.Button(roi_buttons, text='Reset ROI', command=self.reset_roi).pack(side='left', expand=True, fill='x', padx=(0, 2))
        tk.Button(roi_buttons, text='Save ROI', command=self.save_roi_mask).pack(side='left', expand=True, fill='x', padx=2)
        tk.Button(roi_buttons, text='Load ROI', command=self.load_roi_mask).pack(side='right', expand=True, fill='x', padx=(2, 0))
        self.lbl_roi = tk.Label(roi_section, text='ROI: none', wraplength=260, justify='left', bg="#f7f7f7", fg="#374151")
        self.lbl_roi.pack(fill='x', padx=8, pady=(0, 8))

        analysis_section = self._make_sidebar_section(self.sidebar, "Analysis", default_open=True, accent="#d4edda")
        tk.Button(analysis_section, text='Run Analysis on Slice', command=self.run_analysis_slice).pack(fill='x', padx=8, pady=(8, 4))
        tk.Button(analysis_section, text='Run Batch (All Slices + 3D Track)', command=self.run_batch_analysis, bg='#d4edda', font=('Arial', 10, 'bold')).pack(fill='x', padx=8, pady=(0, 8))

        ai_section = self._make_sidebar_section(self.sidebar, f'{_VERSION} AI Biological Analysis', default_open=False, accent="#ede9fe")
        species_list = [
            'D. melanogaster', 'D. simulans', 'D. yakuba',
            'D. ananassae', 'D. pseudoobscura (Dpse)',
            'D. virilis (Dvir)', 'General / Evolutionary'
        ]
        self.species_dropdown = ttk.Combobox(ai_section, textvariable=self.species_var, values=species_list, state='readonly')
        self.species_dropdown.pack(fill='x', padx=8, pady=(8, 5))
        tk.Label(ai_section, text='AI Model:', font=('Arial', 8), bg="#f7f7f7").pack(anchor='w', padx=8)
        models = ['Gemini 2.5 Pro (Thinking)', 'Gemini 2.5 Flash (Fast)']
        self.model_dropdown = ttk.Combobox(ai_section, textvariable=self.model_var, values=models, state='readonly')
        self.model_dropdown.pack(fill='x', padx=8, pady=(0, 5))
        tk.Button(ai_section, text='Set API Key (Free)', command=self.set_ai_key, font=('Arial', 8)).pack(fill='x', padx=8, pady=(0, 5))
        self.btn_ai = tk.Button(ai_section, text='Run AI Analysis', command=self.run_ai_analysis,
                                 bg='#8b5cf6', fg='white', font=('Arial', 9, 'bold'), state='disabled')
        self.btn_ai.pack(fill='x', padx=8, pady=(0, 8))

        self._bind_sidebar_widget_tree(self.sidebar)
        self._bind_sidebar_widget_tree(self.sidebar_quick_actions)
        self.sidebar_canvas.bind('<Button-1>', lambda event: self.sidebar_canvas.focus_set(), add='+')

        self.canvas_frame = tk.Frame(root, bg='black')
        self.canvas_frame.pack(side='right', expand=True, fill='both')

        # --- TOP STATUS BAR (For Progress Visibility) ---
        self.top_status_frame = tk.Frame(self.canvas_frame, bg='#f0f0f0', height=80)
        self.top_status_frame.pack(side='top', fill='x')

        # --- Sub-frame for Status (Left) ---
        self.status_sub_frame = tk.Frame(self.top_status_frame, bg='#f0f0f0')
        self.status_sub_frame.pack(side='left', padx=10, fill='y')

        # --- Dynamic Status Label ---
        self.lbl_batch_op = tk.Label(self.status_sub_frame, text='GUI Ready', font=('Arial', 10, 'bold'), fg='#2c3e50', bg='#f0f0f0')
        self.lbl_batch_op.pack(side='left', pady=5)

        # --- PROGRESS BOX: Sequential Progress Bars (At Top Right) ---
        self.p_container = tk.Frame(self.top_status_frame, bg='#f0f0f0')
        self.p_container.pack(side='right', padx=20, pady=5)

        # Frame 1: 2D Batch Segmentation
        self.batch_p_frame = tk.Frame(self.p_container, bg='#f0f0f0')
        self.batch_p_frame.pack(fill='x')
        tk.Label(self.batch_p_frame, text='Batch Progress (2D)', font=('Arial', 9, 'bold'), bg='#f0f0f0').pack(side='left', padx=10)
        self.progress = ttk.Progressbar(self.batch_p_frame, orient='horizontal', length=200, mode='determinate')
        self.progress.pack(side='left', padx=5, pady=5)
        self.lbl_progress_val = tk.Label(self.batch_p_frame, text='0%', font=('Arial', 10, 'bold'), fg='blue', bg='#f0f0f0')
        self.lbl_progress_val.pack(side='left', padx=10)

        # Frame 2: Post-Analysis (Initially Hidden)
        self.post_p_frame = tk.Frame(self.p_container, bg='#f0f0f0')
        # We don't pack self.post_p_frame here
        tk.Label(self.post_p_frame, text='Post-Analysis Progress', font=('Arial', 9, 'bold'), bg="#f0f0f0").pack(side='left', padx=10)
        self.progress_post = ttk.Progressbar(self.post_p_frame, orient='horizontal', length=200, mode='determinate')
        self.progress_post.pack(side='left', padx=5, pady=5)
        self.lbl_post_progress_val = tk.Label(self.post_p_frame, text='Waiting...', font=('Arial', 10, 'bold'), fg='dimgray', bg="#f0f0f0")
        self.lbl_post_progress_val.pack(side='left', padx=10)

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('key_press_event', self.on_key)


    def set_ai_key(self):
        """Opens a small dialog to save the Gemini API Key to a local file."""
        win = tk.Toplevel(self.root)
        win.title("AI Key Management")
        win.geometry("420x260")
        win.grab_set()

        tk.Label(win, text="Gemini AI Key Setup", font=('Arial', 12, 'bold')).pack(pady=(15,5))

        link = tk.Label(win, text="Get your FREE Key at Google AI Studio", fg='blue', cursor="hand2", font=('Arial', 9, 'underline'))
        link.pack()
        link.bind("<Button-1>", lambda e: webbrowser.open("https://aistudio.google.com/app/apikey"))

        tk.Label(win, text="(No credit card required for Free Tier)", font=('Arial', 8), fg='#64748b').pack(pady=(0, 10))

        tk.Label(win, text="Paste API Key here:", font=('Arial', 9)).pack()
        entry = tk.Entry(win, show='*', width=45)
        entry.pack(pady=5)

        # Load current key if exists
        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_api_key.txt")
        exists = os.path.exists(key_file)
        if exists:
            with open(key_file, 'r') as f:
                entry.insert(0, f.read().strip())

        def save():
            k = entry.get().strip()
            if not k:
                messagebox.showwarning("Warning", "Key cannot be empty.")
                return
            with open(key_file, 'w') as f:
                f.write(k)
            status = "updated" if exists else "saved"
            messagebox.showinfo("Success", f"API Key {status} successfully!\nPortable analysis is ready.")
            win.destroy()

        btn_text = "Update API Key" if exists else "Save Key Permanently"
        tk.Button(win, text=btn_text, command=save, bg='#8b5cf6', fg='white', font=('Arial', 9, 'bold'), padx=20).pack(pady=15)

    def run_ai_analysis(self):
        """Runs AI biological interpretation on the last completed batch data."""
        if self._last_batch_ts is None or self._last_batch_out_dir is None:
            messagebox.showinfo("AI Analysis", "Please run a batch analysis first before requesting AI interpretation.")
            return

        ts = self._last_batch_ts
        out_dir = self._last_batch_out_dir

        if ts.empty or len(ts) == 0:
            messagebox.showinfo("AI Analysis", "No valid 3D tracks found in the last batch. AI requires track data.")
            return

        self.lbl_batch_op.config(text='AI Analyzing Biological Data...', fg='#8b5cf6')
        self.btn_ai.config(state='disabled', text='AI Working...')
        self.root.update()

        # Token mitigation: Send biological candidates so AI sees the main analysis population.
        has_candidates = "is_biological_candidate" in ts.columns
        ts_main = ts[ts["is_biological_candidate"]] if has_candidates else (
            ts[ts["is_quality_track"]] if "is_quality_track" in ts.columns else ts
        )
        csv_summary_str = ts_main.head(100).to_csv(index=False)
        species = self.species_var.get()
        folder_name = os.path.basename(self.input_dir or "Current Project")

        # Model Selection Mapping
        m_map = {
            'Gemini 2.5 Pro (Thinking)': 'gemini-2.5-pro',
            'Gemini 2.5 Flash (Fast)': 'gemini-2.5-flash'
        }
        model_id = m_map.get(self.model_var.get(), 'gemini-2.5-pro')

        def _ai_thread():
            try:
                print(f"AI START: Interpreting {len(ts)} tracks for {species} via {model_id}...")
                ai_text = get_ai_biological_interpretation(csv_summary_str, species, folder_name, model_id=model_id)

                if not ai_text or "AI ANALYSIS SKIPPED" in ai_text or "AI API Error" in ai_text:
                    err_hint = ""
                    if "(429)" in ai_text:
                        err_hint = "\n\nTIP: Gemini 2.5 Pro has lower free-tier quotas. Try switching to 'Gemini 2.5 Flash' in the sidebar or wait a minute."
                    print(f"AI FAIL: {ai_text}")
                    self.root.after(0, lambda: messagebox.showwarning("AI Analysis", f"The AI analysis could not proceed:\n\n{ai_text}{err_hint}"))
                    self.root.after(0, lambda: self.btn_ai.config(state='normal', text='Run AI Analysis'))
                    return

                stats_summary = {
                    "Median_Length_um": f"{ts['total_3d_length_um'].median():.2f}",
                    "Median_Z_Span_um": f"{ts['z_span_um'].median():.2f}" if "z_span_um" in ts.columns else "NA",
                    "Median_Effective_Thickness_um_PSF_sensitive": f"{ts['thickness_um'].median():.2f}" if "thickness_um" in ts.columns else "NA",
                    "Median_Taper_Ratio_PSF_sensitive": f"{ts['taper_ratio'].median():.2f}" if "taper_ratio" in ts.columns else "NA",
                    "Avg_Tortuosity": f"{ts['tortuosity_3d'].mean():.2f}",
                    "Track_Count": f"{len(ts)}",
                    "Species": species
                }
                report_path = generate_ai_html_report(out_dir, ai_text, stats_summary, species)

                if os.path.exists(report_path):
                    print(f"AI SUCCESS: Report generated at {report_path}")
                    abs_report = os.path.abspath(report_path)

                    def _open_report():
                        try:
                            if os.name == 'nt':
                                os.startfile(abs_report)
                            else:
                                webbrowser.open("file:///" + abs_report.replace("\\", "/"))
                        except Exception as oe:
                            print(f"AI OPEN ERROR: {oe}")

                    self.root.after(0, lambda: self.lbl_batch_op.config(text='AI REPORT READY OK', fg='green'))
                    self.root.after(500, _open_report)
                    self.root.after(1200, lambda: messagebox.showinfo("AI Analysis Complete",
                        f"Biological interpretation report generated and opened:\n\n{os.path.basename(report_path)}"))
                else:
                    raise FileNotFoundError(f"Report file was not created at {report_path}")

            except Exception as e:
                import traceback
                print(f"AI EXCEPTION: {traceback.format_exc()}")
                err_msg = f"Failed during biological interpretation:\n\n{str(e)}"
                self.root.after(0, lambda m=err_msg: messagebox.showerror("AI Error", m))
            finally:
                self.root.after(0, lambda: self.btn_ai.config(state='normal', text='Run AI Analysis'))

        threading.Thread(target=_ai_thread, daemon=True).start()

    def load_directory(self):
        """
        Opens a file-picker dialog so the user can select any image in a Z-stack folder.

        Discovers all supported image files (``*.tif``, ``*.tiff``, ``*.png``, ``*.jpg``,
        ``*.jpeg``) in the same directory as the selected file, applies natural sort order
        (so ``z01`` comes before ``z10``), synchronises the Z-slice slider to the selected
        file, and calls :meth:`load_image` to display the first slice.
        """
        initial = CONFIG.get('INPUT_DIR', os.getcwd())

        # Use file picker so user can see images
        fpath = filedialog.askopenfilename(
            initialdir=str(initial),
            title="Select any image in the stack",
            filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg"), ("All files", "*.*")]
        )

        if not fpath:
            return

        selected_file = pl.Path(fpath)
        p = selected_file.parent
        self.input_dir = str(p)

        exts = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        found_files_path = []

        # 1. Discover all images in the same folder
        for ext in exts:
            found_files_path.extend(list(p.glob(f"*{ext}")))
            found_files_path.extend(list(p.glob(f"*{ext.upper()}")))

        # 2. Recursive fallback (if needed, though file picker implies they are in the right spot)
        if not found_files_path:
            for ext in exts:
                found_files_path.extend(list(p.rglob(f"*{ext}")))
                found_files_path.extend(list(p.rglob(f"*{ext.upper()}")))

        if not found_files_path:
            messagebox.showerror('Error', f"No supported images found in:\n{p}")
            return

        # Standardize and Natural Sort
        found_files_str = [os.path.abspath(str(f)) for f in found_files_path]
        unique_files = list(set(found_files_str))
        self.files = sorted(unique_files, key=natural_sort_key)

        # Sync to the selected file
        try:
            self.current_idx = self.files.index(os.path.abspath(fpath))
        except ValueError:
            self.current_idx = 0

        self.scale_z.config(to=len(self.files) - 1)
        self.scale_z.set(self.current_idx)
        self.reset_roi(redraw=False)
        self.load_image()
        self.lbl_status.config(text=f'Opened: {selected_file.name}\n({len(self.files)} slices in folder)', fg='blue')
        self.root.update()

    def load_image(self):
        if not self.files:
            return
        try:
            self.current_img = robust_imread(self.files[self.current_idx])
            self.lbl_z.config(text=f'Z: {self.current_idx} / {len(self.files)-1}')
            self.render()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{self.files[self.current_idx]}\n\nError: {e}")

    def on_slide_change(self, val):
        self.current_idx = int(val)
        self.load_image()

    def render(self):
        self.ax.clear()
        self.ax.axis('off')

        # --- NEW: Review Overlays Logic ---
        if self.mode_var.get() == 'review':
            try:
                if not hasattr(self, 'last_out_dir') or not self.last_out_dir:
                    self.ax.text(0.5, 0.5, "No Batch Analysis Results Found.\nRun Batch First.",
                                 ha='center', va='center', color='red', transform=self.ax.transAxes)
                    self.canvas.draw_idle()
                    return

                z_idx = extract_z_index(self.files[self.current_idx])
                panel_path = os.path.join(self.last_out_dir, "overlays", f"z{z_idx:02d}_panel.png")

                if os.path.exists(panel_path):
                    if _HAVE_CV2:
                        img = _cv2.imread(panel_path)
                        img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
                    else:
                        img = plt.imread(panel_path)
                    self.ax.imshow(img)
                    self.canvas.draw_idle()
                    return
                else:
                    self.ax.text(0.5, 0.5, f"Overlay not found for Z={z_idx:02d}\n{os.path.basename(panel_path)}",
                                 ha='center', va='center', color='orange', transform=self.ax.transAxes)
                    self.canvas.draw_idle()
                    return
            except Exception as e:
                print(f"Overlay Render Error: {e}")

        if self.current_img is not None and isinstance(self.current_img, np.ndarray):
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
            import time as _t
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

            full_img = self.current_img
            crop_offset_y, crop_offset_x = 0, 0

            t0 = _t.time()
            log("  v5.7 U-Net-ready single-pass analysis...")
            preview_context = build_stack_preprocess_context(
                self.files if self.files else [self.files[self.current_idx]],
                roi_mask,
                params,
                exclusion_mask=None,
            )
            log(f"  Temporary preview context: profile={preview_context.selected_clahe_profile}, sampled_z={preview_context.sampled_z_indices}")
            seg1 = segment_slice(
                full_img,
                params,
                roi_mask=roi_mask,
                preprocess_context=preview_context,
                exclusion_mask=None,
                z_idx=self.current_idx,
            )
            meas1 = measure_spermatids(seg1, params)
            results = meas1['results']
            skel_label_full = meas1['skel_label']

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

            # ------ INTERACTIVE MANUAL CORRECTION LOGIC ------
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

        # Auto-incremental output directory inside selected folder
        out_dir = get_unique_batch_dir(self.input_dir)
        self.last_out_dir = out_dir

        # EXPLICIT CONFIRMATION: Show the user where the data will go
        confirm = messagebox.askokcancel("Confirm Output",
            f"Results (Excel, PDF, CSV) will be saved to:\n\n{out_dir}\n\nContinue?")
        if not confirm:
            return

        ensure_dir(out_dir)
        overlay_dir = os.path.join(out_dir, "overlays")
        ensure_dir(overlay_dir)

        params = CONFIG.copy()
        params['OUTPUT_DIR'] = out_dir
        params['SAVE_DEBUG_IMAGES'] = False
        params['DO_TRACKING'] = True

        roi_mask = self.build_roi_mask()
        exclusion_mask = None

        self.lbl_roi.config(text="Processing... See Top Bar")
        self.root.update_idletasks()

        try:
            import time as _t
            t_batch = _t.time()
            self.lbl_batch_op.config(text=f"Batch Segmenting: 0 / {len(self.files)} slices...", fg='blue')
            self.root.update()

            all_rows = []
            summaries = []

            # Z-Projection accumulation
            max_proj_raw = None
            max_proj_ov = None
            slice_cache = {}

            # Robust initialization
            ts = None
            df_trk = None
            first_img = ensure_2d_image(robust_imread(self.files[0]), os.path.basename(self.files[0]))
            if roi_mask is not None and roi_mask.shape != first_img.shape:
                raise ValueError(f"ROI shape {roi_mask.shape} does not match image shape {first_img.shape}")
            files_by_z = {
                int(extract_z_index(fpath, sequence_idx=i)): fpath
                for i, fpath in enumerate(self.files)
            }
            preprocess_context = build_stack_preprocess_context(self.files, roi_mask, params, exclusion_mask=exclusion_mask)
            save_stack_preprocess_context(preprocess_context, out_dir)
            if roi_mask is not None:
                tifffile.imwrite(os.path.join(out_dir, "roi_mask_used.tif"), roi_mask.astype(np.uint8) * 255)

            self.progress['value'] = 0
            self.progress['maximum'] = len(self.files)
            self.progress_post['value'] = 0
            self.progress_post['maximum'] = 100
            self.lbl_post_progress_val.config(text="0%", fg='orange')

            for idx, fpath in enumerate(self.files):
                z_idx = extract_z_index(fpath, sequence_idx=idx)

                pct = int(((idx + 1) / len(self.files)) * 100)
                self.progress['value'] = idx + 1
                self.lbl_progress_val.config(text=f"{pct}%")
                self.root.update()

                print(f"[{idx+1}/{len(self.files)}] Processing Z={z_idx:02d}...")

                img_raw = robust_imread(fpath)
                process_img = ensure_2d_image(img_raw, os.path.basename(fpath))
                print(f"DEBUG batch image {os.path.basename(fpath)} shape: {process_img.shape}, ndim={process_img.ndim}")

                full_img = process_img
                crop_oy, crop_ox = 0, 0
                unet_context = _make_unet_context_from_paths(files_by_z, z_idx)
                seg = segment_slice(process_img, params, z_idx=z_idx,
                                    roi_mask=roi_mask,
                                    preprocess_context=preprocess_context,
                                    exclusion_mask=exclusion_mask,
                                    unet_context_stack=unet_context)
                meas = measure_spermatids(seg, params)
                res = meas['results']
                sl_full = meas['skel_label']

                for r in res:
                    r['centroid_x'] += crop_ox
                    r['centroid_y'] += crop_oy

                um = params['UM_PER_PX_XY']
                ls_um = [r['length_px_geodesic']*um for r in res]
                ws_um = [r['width_px']*um for r in res]

                all_rows.extend(rows_from_results(res, z_idx, um))
                summaries.append({
                    "z_slice": z_idx,
                    "n_spermatids": len(res),
                    "mean_length_um": round(float(np.mean(ls_um)), 3) if ls_um else 0,
                    "median_length_um": round(float(np.median(ls_um)), 3) if ls_um else 0,
                    "mean_width_um": round(float(np.mean(ws_um)), 3) if ws_um else 0,
                })

                if params['SAVE_OVERLAYS']:
                    ov = make_overlay(full_img, sl_full)
                    # Create side-by-side panel
                    orig_rgb = (normalize_display(full_img) * 255).astype(np.uint8)
                    if orig_rgb.ndim == 2:
                        orig_rgb = np.stack([orig_rgb]*3, axis=-1)
                    panel = np.hstack([orig_rgb, ov])
                    _imwrite(os.path.join(overlay_dir, f"z{z_idx:02d}_panel.png"), panel)

                    # ---- LIVE GUI UPDATE ----
                    # Show the side-by-side segmentation panel during batch execution
                    if hasattr(self, 'ax') and hasattr(self, 'canvas'):
                        try:
                            self.ax.clear()
                            self.ax.axis('off')
                            self.ax.imshow(panel)
                            self.canvas.draw()
                        except Exception:
                            pass

                    # Update Z-Projections
                    if max_proj_raw is None:
                        max_proj_raw = full_img.copy().astype(np.float32)
                        max_proj_ov = ov.copy().astype(np.float32)
                    else:
                        max_proj_raw = np.maximum(max_proj_raw, full_img.astype(np.float32))
                        max_proj_ov = np.maximum(max_proj_ov, ov.astype(np.float32))
                    slice_cache[int(z_idx)] = {
                        "image": full_img.copy(),
                        "skel_label": sl_full.copy().astype(np.int32),
                    }

                if params["SAVE_MASK_TIFS"]:
                    tifffile.imwrite(os.path.join(out_dir, f"z{z_idx:02d}_mask.tif"),
                                     (seg["mask_clean"] & roi_mask if roi_mask is not None else seg["mask_clean"]).astype(np.uint8) * 255)
                if params.get("UNET_SAVE_PROBABILITY_MAPS", True):
                    if seg.get("unet_probability") is not None and np.any(seg.get("unet_probability")):
                        tifffile.imwrite(os.path.join(out_dir, f"z{z_idx:02d}_unet_probability.tif"),
                                         seg["unet_probability"].astype(np.float32))

                # Update Progress Bar for each slice
                self.lbl_batch_op.config(text=f"Batch Segmenting: {idx+1} / {len(self.files)} slices...", fg='blue')
                self.progress['value'] = idx + 1
                self.root.update()

            df = pd.DataFrame(all_rows)
            df_sum = pd.DataFrame(summaries)
            df.to_csv(os.path.join(out_dir, "spermatid_measurements.csv"), index=False)
            df_sum.to_csv(os.path.join(out_dir, "slice_summary.csv"), index=False)

            if not df.empty:
                self.lbl_batch_op.config(text='Running 3D Tracking & Morphometrics...', fg='#e67e22') # Orange-ish

                # --- Sequential Progress Bar Swap ---
                # Hide 2D progress, Show Post-Analysis progress
                self.batch_p_frame.pack_forget()
                self.post_p_frame.pack(fill='x')

                self.progress_post['value'] = 25
                self.lbl_post_progress_val.config(text="25%")
                self.root.update()

                df_trk, ts = track_across_slices(df, params)

                # --- Advanced 3D Morphometrics ---
                self.lbl_batch_op.config(text='Calculating Advanced 3D Metrics...', fg='#e67e22')
                # Metrics are natively generated correctly and safely in track_across_slices

                self.progress_post['value'] = 60
                self.lbl_post_progress_val.config(text="60%")
                self.root.update()

                df_trk.to_csv(os.path.join(out_dir, "measurements_with_tracks.csv"), index=False)

                # ------ AUTO CANDIDATE AUDIT ----------------------------------------------------------------------------------------------------------------
                self.lbl_batch_op.config(text='Running Candidate Audit...', fg='#e67e22')
                self.root.update()

                ts = flag_quality_tracks(ts, params)

                # Save annotated track summary (with strict quality and biological-candidate flags)
                ts.to_csv(os.path.join(out_dir, "track_summary.csv"), index=False)

                # Save strict no-warning and biological-candidate tracks for downstream use
                ts_quality = ts[ts["is_quality_track"]].copy()
                ts_quality.to_csv(os.path.join(out_dir, "track_summary_quality.csv"), index=False)
                ts_candidates = ts[ts["is_biological_candidate"]].copy() if "is_biological_candidate" in ts.columns else ts_quality
                ts_candidates.to_csv(os.path.join(out_dir, "track_summary_biological_candidates.csv"), index=False)
                export_comparative_track_tables(out_dir, ts, None)

                # Generate candidate-coded overlays after audit.
                if params['SAVE_OVERLAYS']:
                    export_quality_overlays(out_dir, slice_cache, df_trk, ts)

                # Generate outlier_audit/ subfolder automatically
                export_outlier_audit(out_dir, ts, params)
                export_post_detection_qc(out_dir, df_trk, ts)

                n_quality = len(ts_quality)
                n_candidates = len(ts_candidates)
                n_hard_fail = len(ts) - n_candidates
                self.lbl_batch_op.config(
                    text=f'Candidates: {n_candidates} kept / {n_hard_fail} hard-fail / {n_quality} strict', fg='#27ae60')
                self.root.update()

                # Save Global Z-Projection
                if max_proj_raw is not None:
                    self.lbl_batch_op.config(text='Generating Global Z-Projection...', fg='#e67e22')
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
            self.lbl_batch_op.config(text='Batch Analysis Complete.', fg='green')
            self.lbl_progress_val.config(text="100% - Done", fg='green')
            self.root.update()

            # Generate High-Res Graphical Report and Excel Audit
            self.lbl_batch_op.config(text='Generating PDF & Excel Reports...', fg='#8e44ad') # Purple

            def update_cb(v):
                self.progress_post['value'] = v
                self.lbl_post_progress_val.config(text=f"{v}%")
                self.root.update()

            generate_batch_report(out_dir, df, df_sum, um, ts if not df.empty else None, update_cb, generate_pptx=False)
            generate_excel_report(out_dir, df, df_sum, ts if not df.empty else None)

            # --- Store batch data for AI button ---
            if not df.empty and ts is not None and len(ts) > 0:
                self._last_batch_ts = ts
                self._last_batch_out_dir = out_dir
                self.btn_ai.config(state='normal')
                print(f"AI READY: {len(ts)} tracks stored. Click 'Run AI Analysis' to interpret.")

            self.lbl_batch_op.config(text='Generating PowerPoint Dashboard...', fg='#c0392b') # Red-ish
            self.root.update()
            try:
                ok = generate_pptx_report(out_dir, df, df_sum, um, ts if not df.empty else None)
                if not ok:
                    print("WARNING: PPTX generation returned False - check console for traceback.")
            except Exception as pptx_err:
                import traceback as _tb
                print(f"ERROR generating PPTX: {pptx_err}")
                _tb.print_exc()

            msg = f"Batch complete in {elapsed:.1f}s!\nSaved to: {out_dir}"
            print(msg)
            self.lbl_batch_op.config(text='ALL OPERATIONS COMPLETE', fg='green')
            self.lbl_progress_val.config(text="100%", fg='green')
            self.lbl_post_progress_val.config(text="100% - DONE", fg='green')
            self.progress_post['value'] = 100
            self.root.update()

            # --- macOS STABILITY PATCH: Ensure all plot resources are freed before UI popup ---
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except: pass

            messagebox.showinfo('Batch Complete', msg)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.lbl_batch_op.config(text=f'Batch error: {e}', fg='red')
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
    import multiprocessing
    multiprocessing.freeze_support()
    print(f"\n{'='*60}")
    print(f" SPERMATID ANALYSIS PIPELINE - v{_VERSION} - INITIALIZING...")
    print(f" Mode: {'GUI' if '--gui' in sys.argv or len(sys.argv)<=1 else 'CLI'}")
    print(f" Features: [Incremental Folders] [Multi-Tab Excel] [3D Stats]")
    print(f"{'='*60}\n")

    ap = argparse.ArgumentParser(description=f"Spermatid segmentation {_VERSION} / ROI GUI")
    ap.add_argument("--batch",  action="store_true", help="Run batch processing")
    ap.add_argument("--single", action="store_true", help="Run a single-image analysis")
    ap.add_argument("--gui",    action="store_true", help="Launch ROI GUI")
    ap.add_argument("--z",      type=int, default=None, help="Choose z-index in single mode")
    ap.add_argument("--params", type=str, default=None, help="Path to tuned parameters JSON file to override CONFIG")
    ap.add_argument("--roi-mask", type=str, default=None, help="Optional .npy/.tif ROI mask for batch processing")
    args = ap.parse_args()

    # Load tuned parameters from JSON if provided
    if args.params:
        import json as _json
        params_path = os.path.abspath(args.params)
        if os.path.exists(params_path):
            with open(params_path, 'r') as _pf:
                tuned = _json.load(_pf)
            applied = 0
            for key, value in tuned.items():
                if key in CONFIG:
                    CONFIG[key] = value
                    applied += 1
            print(f"  Loaded {applied} tuned parameters from: {os.path.basename(params_path)}")
        else:
            print(f"  WARNING: Params file not found: {params_path}")

    if args.roi_mask:
        CONFIG["ROI_MASK_PATH"] = os.path.abspath(args.roi_mask)
        print(f"  Loaded ROI mask path: {CONFIG['ROI_MASK_PATH']}")

    validate_config(CONFIG)

    # Launch GUI by default if no explicit CLI flags are provided
    if args.gui or not (args.batch or args.single or args.z is not None):
        launch_gui()
        raise SystemExit

    if args.batch:
        CONFIG["RUN_MODE"] = "batch"
        # CLI Incremental Folder Logic - Anchor to INPUT_DIR
        base_parent = CONFIG["INPUT_DIR"]
        if not os.path.isabs(base_parent):
            base_parent = os.path.abspath(base_parent)

        CONFIG["OUTPUT_DIR"] = get_unique_batch_dir(base_parent)
        ensure_dir(CONFIG["OUTPUT_DIR"])
        print(f"CLI BATCH MODE: Results will be saved inside input folder: {CONFIG['OUTPUT_DIR']}")
    if args.single:
        CONFIG["RUN_MODE"] = "single"
    if args.z is not None:
        CONFIG["SINGLE_IMAGE_SELECTION_MODE"] = "z_index"
        CONFIG["SINGLE_Z_INDEX"] = args.z

    if CONFIG["RUN_MODE"] == "single":
        process_one_image(choose_single_image(CONFIG), CONFIG, CONFIG["OUTPUT_DIR"])
    else:
        process_batch(CONFIG)
