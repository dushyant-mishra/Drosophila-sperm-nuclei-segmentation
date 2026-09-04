import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "saturn_v571_tuner_body_width_test",
    ROOT / "utils" / "tune_parameters_Saturnv5_7_1.py",
)
tuner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tuner)


def test_tuner_primary_width_uses_body_chord_and_labels_legacy():
    cfg = tuner.CONFIG.copy()
    cfg["UM_PER_PX_XY"] = 0.5
    rows = [
        {
            "length_px_geodesic": 10.0,
            "body_width_px": 4.0,
            "length_body_width_ratio": 2.5,
            "width_px": 8.0,
            "length_width_ratio": 1.25,
        },
        {
            "length_px_geodesic": 12.0,
            "body_width_px": 6.0,
            "length_body_width_ratio": 2.0,
            "width_px": 10.0,
            "length_width_ratio": 1.2,
        },
    ]
    zero = np.zeros((5, 5), dtype=bool)
    segs = [
        (
            {
                "mask_clean": zero,
                "mask_hyst": zero,
                "bridge_stats": {
                    "skeleton_pixels_before": 1,
                    "skeleton_pixels_after": 1,
                },
            },
            {"results": [row]},
        )
        for row in rows
    ]

    summary = tuner.summarize_candidate(rows, segs, cfg)

    assert summary["median_width_um"] == 2.5
    assert summary["median_length_width_ratio"] == 2.25
    assert summary["median_width_um_dt_legacy"] == 4.5
    assert summary["median_length_width_ratio_dt_legacy"] == 1.225
    assert summary["body_width_available_count"] == 2
    assert summary["body_width_missing_fraction"] == 0.0


def test_tuner_does_not_substitute_legacy_width_when_body_width_is_unavailable():
    cfg = tuner.CONFIG.copy()
    rows = [
        {
            "length_px_geodesic": 10.0,
            "body_width_px": np.nan,
            "length_body_width_ratio": np.nan,
            "width_px": 3.0,
            "length_width_ratio": 3.0,
        }
    ]
    zero = np.zeros((5, 5), dtype=bool)
    segs = [(
        {
            "mask_clean": zero,
            "mask_hyst": zero,
            "bridge_stats": {
                "skeleton_pixels_before": 1,
                "skeleton_pixels_after": 1,
            },
        },
        {"results": rows},
    )]

    summary = tuner.summarize_candidate(rows, segs, cfg)

    assert np.isnan(summary["median_width_um"])
    assert np.isnan(summary["median_length_width_ratio"])
    assert summary["body_width_available_count"] == 0
    assert summary["body_width_missing_fraction"] == 1.0
    assert summary["median_width_um_dt_legacy"] == 3.0 * cfg["UM_PER_PX_XY"]
