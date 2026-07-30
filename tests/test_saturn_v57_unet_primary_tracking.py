import pytest
import numpy as np
import pandas as pd

import importlib.util
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sperm_segmentation_saturnv5_7", os.path.join(PROJECT_ROOT, "sperm_segmentation_saturnv5.7.py"))
sperm_seg = importlib.util.module_from_spec(spec)
sys.modules["sperm_segmentation_saturnv5_7"] = sperm_seg
spec.loader.exec_module(sperm_seg)
rows_from_results = sperm_seg.rows_from_results

def test_unet_primary_area_logic():
    um = 0.5
    z_idx = 10
    
    # Mock result from classical engine
    classical_res = {
        "label": 1,
        "length_px_geodesic": 20.0,
        "length_px_count": 22.0,
        "width_px": 2.0,
        "length_width_ratio": 10.0,
        "tortuosity": 1.0,
        "n_endpoints": 2,
        "n_branch_nodes": 0,
        "centroid_x": 50.0,
        "centroid_y": 50.0,
        "area_px": 40.0, # geodesic * width
        "detection_source": "saturn_classical",
    }
    
    # Mock result from unet_primary
    unet_res = {
        "label": 2,
        "source_instance_key": "10_2",
        "length_px_geodesic": 20.0,
        "length_px_count": 22.0,
        "width_px": 2.0,
        "length_width_ratio": 10.0,
        "tortuosity": 1.0,
        "n_endpoints": 2,
        "n_branch_nodes": 0,
        "centroid_x": 100.0,
        "centroid_y": 100.0,
        "area_px": 40.0, # classical slender area
        "instance_mask_area_px": 55.0, # filled mask area
        "detection_source": "unet_primary",
        "unet_mean_probability": 0.90,
        "unet_max_probability": 0.98,
        "morphology_warning": True,
        "morphology_warning_reasons": "short",
        "technical_failure": False,
        "technical_failure_reason": ""
    }
    
    rows = rows_from_results([classical_res, unet_res], z_idx, um)
    df = pd.DataFrame(rows)
    
    # Assert classical area uses estimated_slender_area_px
    classical_row = df[df["sperm_id"] == 1].iloc[0]
    assert classical_row["area_px"] == 40.0
    assert classical_row["estimated_slender_area_px"] == 40.0
    
    # Assert unet_primary area uses instance_mask_area_px
    unet_row = df[df["sperm_id"] == 2].iloc[0]
    assert unet_row["area_px"] == 55.0
    assert unet_row["estimated_slender_area_px"] == 40.0
    assert unet_row["instance_mask_area_px"] == 55.0
    
    # Assert other unet_primary fields are preserved
    assert unet_row["source_instance_key"] == "10_2"
    assert unet_row["unet_mean_probability"] == 0.90
    assert unet_row["morphology_warning"] == True
    assert unet_row["detection_source"] == "unet_primary"
