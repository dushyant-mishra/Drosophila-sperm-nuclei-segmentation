"""Body-width contract for the classical (non-U-Net) detection path.

Saturn v5.7.1 measures the primary apparent body width as subpixel perpendicular
contour chords. Before this suite the chord was computed only for U-Net
instances, so classical detections silently fell back to the quantized
distance-transform (EDT) median while still being written under the unqualified
``width_px`` / ``width_um`` / ``length_width_ratio`` names. These tests pin the
corrected contract: unqualified width is the chord or it is unavailable, and the
EDT value is reachable only under an explicitly ``_dt_legacy`` name.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
from skimage import measure as skmeasure
from skimage.draw import polygon
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]

SHAPE = (128, 192)


def load_saturn_v571():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_classical_width_test",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classical_cfg(module, **updates):
    """Permissive classical config so morphology filters do not reject fixtures."""
    cfg = module.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "classical_saturn",
            "BODY_WIDTH_ENABLE": True,
            "BODY_WIDTH_ENDPOINT_TRIM_FRACTION": 0.125,
            "BODY_WIDTH_SAMPLE_SPACING_PX": 1.0,
            "BODY_WIDTH_SMOOTH_SIGMA_PX": 0.75,
            "BODY_WIDTH_MIN_SAMPLES": 5,
            "UM_PER_PX_XY": 1.0,
            # Micron keys win in resolve_pixel_parameters, so set them directly.
            "MIN_SKEL_LEN_UM": 1.0,
            "MAX_GEODESIC_LEN_UM": 500.0,
            "MAX_WIDTH_UM": 60.0,
            "MIN_LENGTH_WIDTH_RATIO": 1.0,
            "MAX_BRANCH_NODES": 50,
            "MAX_ENDPOINT_COUNT": 50,
            "MAX_TORTUOSITY": 100.0,
            "UNET_RESCUE_ENABLE": False,
        }
    )
    cfg.update(updates)
    return cfg


def rotated_rectangle(center, length, width, angle_deg):
    theta = np.deg2rad(angle_deg)
    tangent = np.array([np.sin(theta), np.cos(theta)])
    normal = np.array([-tangent[1], tangent[0]])
    corners = np.array(
        [
            center - tangent * length / 2 - normal * width / 2,
            center + tangent * length / 2 - normal * width / 2,
            center + tangent * length / 2 + normal * width / 2,
            center - tangent * length / 2 + normal * width / 2,
        ]
    )
    rr, cc = polygon(corners[:, 0], corners[:, 1], shape=SHAPE)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[rr, cc] = True
    return mask


def seg_from_mask(mask, skel_labeled=None):
    """Minimal classical segmentation dict for measure_spermatids."""
    skel = skeletonize(mask)
    if skel_labeled is None:
        skel_labeled = skmeasure.label(skel).astype(np.int32)
    return {
        "mask_clean": mask,
        "skel_pruned": skel,
        "skel_labeled": skel_labeled,
        "dist_clean": ndimage.distance_transform_edt(mask).astype(np.float32),
        "unet_probability": None,
    }


def measure_rows(module, seg, cfg, um=1.0):
    results = module.measure_spermatids(seg, cfg)["results"]
    return results, module.rows_from_results(results, z_idx=1, um=um)


def test_classical_detection_reports_subpixel_chord_not_dt_width():
    """An axis-aligned rod of known thickness must measure its true width."""
    saturn = load_saturn_v571()
    # Rows 40..46 inclusive -> exactly 7 px of material.
    mask = np.zeros(SHAPE, dtype=bool)
    mask[40:47, 30:150] = True

    results, rows = measure_rows(saturn, seg_from_mask(mask), classical_cfg(saturn))

    assert len(results) == 1
    row = rows[0]
    assert row["width_measurement_method"] == (
        "subpixel_mask_contour_perpendicular_chords_central_body"
    )
    # The chord recovers the true 7.0 px extent.
    assert row["width_px"] == pytest.approx(7.0, abs=0.25)
    # The legacy EDT median quantizes to 8.0 and must stay under the legacy name.
    assert row["width_px_dt_median_legacy"] == pytest.approx(8.0, abs=0.5)
    assert row["width_px"] != pytest.approx(row["width_px_dt_median_legacy"], abs=1e-9)


@pytest.mark.parametrize("angle", [0, 20, 45, 70, 90])
def test_classical_chord_is_rotation_stable(angle):
    """Apparent width must not depend on how the object lies on the pixel grid."""
    saturn = load_saturn_v571()
    mask = rotated_rectangle(np.array([64.0, 96.0]), 90, 9, angle)

    results, _ = measure_rows(saturn, seg_from_mask(mask), classical_cfg(saturn))

    assert len(results) == 1
    assert results[0]["body_width_px"] == pytest.approx(9.0, abs=1.0)


def test_rotation_spread_stays_within_subpixel_tolerance():
    """Spread across orientations is the geometry-fidelity bar for the claim."""
    saturn = load_saturn_v571()
    widths = []
    for angle in (0, 15, 30, 45, 60, 75, 90):
        mask = rotated_rectangle(np.array([64.0, 96.0]), 90, 9, angle)
        results, _ = measure_rows(saturn, seg_from_mask(mask), classical_cfg(saturn))
        widths.append(results[0]["body_width_px"])

    assert max(widths) - min(widths) < 1.0


def test_multi_centerline_component_reports_unavailable_not_merged_width():
    """A filled blob holding two centerlines must not report a fabricated width."""
    saturn = load_saturn_v571()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[36:45, 20:170] = True
    # Two disjoint pruned centerlines inside one filled component.
    skel_labeled = np.zeros(SHAPE, dtype=np.int32)
    skel_labeled[40, 25:80] = 1
    skel_labeled[40, 100:160] = 2

    seg = seg_from_mask(mask, skel_labeled=skel_labeled)
    results, rows = measure_rows(saturn, seg, classical_cfg(saturn))

    assert len(results) == 2
    for result, row in zip(results, rows):
        assert not np.isfinite(result["body_width_px"])
        assert row["width_measurement_method"] == "unavailable_multi_centerline_component"
        assert not np.isfinite(row["width_um"])
        assert not np.isfinite(row["length_width_ratio"])
        # The legacy measurement still exists, but only under its legacy name.
        assert np.isfinite(row["width_um_dt_median_legacy"])


def test_unqualified_width_never_equals_dt_value_when_they_differ():
    """The core audit invariant: unqualified width is never the EDT number."""
    saturn = load_saturn_v571()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[40:47, 30:150] = True

    _, rows = measure_rows(saturn, seg_from_mask(mask), classical_cfg(saturn))

    for row in rows:
        legacy = row["width_um_dt_median_legacy"]
        if np.isfinite(row["width_um"]) and np.isfinite(legacy):
            # They may coincide numerically by chance, but the method column must
            # always disclose which definition produced the unqualified value.
            assert row["width_measurement_method"] != "classical_dt_median"


def test_every_row_declares_a_width_measurement_method():
    """Provenance is mandatory, including when the width is unavailable."""
    saturn = load_saturn_v571()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[40:47, 30:150] = True
    mask[70:75, 30:120] = True

    _, rows = measure_rows(saturn, seg_from_mask(mask), classical_cfg(saturn))

    assert rows
    for row in rows:
        method = row["width_measurement_method"]
        assert isinstance(method, str) and method
        if np.isfinite(row["width_um"]):
            assert method.startswith("subpixel_")
        else:
            assert method.startswith("unavailable")


def test_slender_area_follows_primary_width_or_is_unavailable():
    """Area estimate must not mix a chord length with an EDT width."""
    saturn = load_saturn_v571()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[36:45, 20:170] = True
    skel_labeled = np.zeros(SHAPE, dtype=np.int32)
    skel_labeled[40, 25:80] = 1
    skel_labeled[40, 100:160] = 2

    _, rows = measure_rows(
        saturn, seg_from_mask(mask, skel_labeled=skel_labeled), classical_cfg(saturn)
    )

    for row in rows:
        # Width is unavailable for these merged-component objects, so the derived
        # area estimate must be unavailable too rather than falling back to EDT.
        assert not np.isfinite(row["estimated_slender_area_px"])


def test_degenerate_mask_yields_unavailable_rather_than_legacy_substitute():
    """Adversarial: too few chord samples must not resurrect the EDT value."""
    saturn = load_saturn_v571()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[40:47, 30:150] = True

    cfg = classical_cfg(saturn, BODY_WIDTH_MIN_SAMPLES=10_000)
    _, rows = measure_rows(saturn, seg_from_mask(mask), cfg)

    assert rows
    for row in rows:
        assert not np.isfinite(row["width_um"])
        assert row["width_measurement_method"] != "classical_dt_median"
        assert np.isfinite(row["width_um_dt_median_legacy"])


def test_disabled_body_width_does_not_fall_back_to_dt_under_primary_name():
    """Turning the chord off must yield unavailable, not a silent EDT swap."""
    saturn = load_saturn_v571()
    mask = np.zeros(SHAPE, dtype=bool)
    mask[40:47, 30:150] = True

    cfg = classical_cfg(saturn, BODY_WIDTH_ENABLE=False)
    _, rows = measure_rows(saturn, seg_from_mask(mask), cfg)

    assert rows
    for row in rows:
        assert not np.isfinite(row["width_um"])
        assert row["width_measurement_method"] == "disabled"
