"""Validate Saturn v5.7.1 body-width geometry on COCO instance masks."""

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage import measure
from skimage.draw import polygon
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "sperm_segmentation_saturnv5.7.1.py"
DEFAULT_COCO = (
    ROOT
    / "training_packages"
    / "v5_7_kj_wt_replay_finetune"
    / "annotations"
    / "_annotations.coco.json"
)
DEFAULT_OUTPUT = ROOT / "scratch" / "v5_7_1_body_width_validation"

PRIMARY_USER_FIELD = "representative_body_width_um"
MEASUREMENT_LABEL = "apparent central-body mask width"
CROSS_STRUCTURE = np.array(
    [[False, True, False], [True, True, True], [False, True, False]],
    dtype=bool,
)


def rotated_rectangle(shape, center, length, width, angle_deg):
    """Rasterize a known-width rectangle for implementation validation."""
    theta = np.deg2rad(float(angle_deg))
    tangent = np.array([np.sin(theta), np.cos(theta)])
    normal = np.array([-tangent[1], tangent[0]])
    center = np.asarray(center, dtype=float)
    corners = np.array(
        [
            center - tangent * length / 2 - normal * width / 2,
            center + tangent * length / 2 - normal * width / 2,
            center + tangent * length / 2 + normal * width / 2,
            center - tangent * length / 2 + normal * width / 2,
        ]
    )
    rr, cc = polygon(corners[:, 0], corners[:, 1], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    return mask


def measure_mask(saturn, mask, cfg):
    """Measure one mask, preserving unavailable results instead of inventing width."""
    mask = np.asarray(mask, dtype=bool)
    centerline = np.argwhere(skeletonize(mask))
    if centerline.shape[0] < 2:
        return saturn.measure_subpixel_body_width(mask, centerline, cfg)
    centerline = saturn.extract_geodesic_centerline_coords(
        centerline,
        mask.shape[1],
    )
    return saturn.measure_subpixel_body_width(mask, centerline, cfg)


def evaluate_synthetic_geometry(saturn, cfg):
    """Return deterministic known-geometry and rotation-stability evidence."""
    records = []
    for expected_width in (5.0, 9.0, 13.0):
        for angle in (0, 20, 45, 70, 90):
            mask = rotated_rectangle(
                (160, 160),
                np.array([80.0, 80.0]),
                90.0,
                expected_width,
                angle,
            )
            result = measure_mask(saturn, mask, cfg)
            measured = float(result["body_width_px"])
            records.append(
                {
                    "geometry": "rectangle",
                    "expected_width_px": expected_width,
                    "angle_deg": angle,
                    "measured_width_px": measured,
                    "absolute_error_px": abs(measured - expected_width),
                    "sample_count": int(result["body_width_sample_count"]),
                }
            )
    frame = pd.DataFrame(records)
    spreads = frame.groupby("expected_width_px")["measured_width_px"].agg(
        lambda values: float(values.max() - values.min())
    )
    return frame, {
        "case_count": int(len(frame)),
        "all_cases_measured": bool(frame["measured_width_px"].notna().all()),
        "maximum_absolute_error_px": float(frame["absolute_error_px"].max()),
        "maximum_rotation_spread_px": float(spreads.max()),
    }


def engineering_verdict(synthetic, rasterized, measured):
    """Assess software behavior without claiming biological diameter accuracy."""
    criteria = {
        "known_geometry_max_error_at_most_1px": bool(
            synthetic["maximum_absolute_error_px"] <= 1.0
        ),
        "rotation_spread_at_most_1px": bool(
            synthetic["maximum_rotation_spread_px"] <= 1.0
        ),
        "coco_measurement_success_at_least_95pct": bool(
            len(rasterized) > 0
            and len(measured) / len(rasterized) >= 0.95
        ),
        "one_pixel_expansion_increases_width": bool(
            pd.to_numeric(
                measured["mask_dilate1_width_delta_px"], errors="coerce"
            ).dropna().median()
            > 0
        ),
        "one_pixel_erosion_decreases_width": bool(
            pd.to_numeric(
                measured["mask_erode1_width_delta_px"], errors="coerce"
            ).dropna().median()
            < 0
        ),
    }
    return criteria, "pass" if all(criteria.values()) else "fail"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_v571_width_validation", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata():
    """Return the software environment needed to reproduce this QC run."""
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-image", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }


def prepare_output_dir(output_dir):
    """Reserve a new evidence directory; never reuse or overwrite one."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new audit run path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def atomic_write_bytes(path, payload):
    """Atomically publish one completed artifact in its destination directory."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_dataframe(path, frame):
    atomic_write_text(path, frame.to_csv(index=False))


def annotation_mask(annotation, image_record):
    segmentation = annotation.get("segmentation", [])
    if not isinstance(segmentation, list) or not segmentation:
        return None
    polygon_points = []
    for coordinates in segmentation:
        points = np.asarray(coordinates, dtype=float).reshape(-1, 2)
        if points.shape[0] >= 3:
            polygon_points.append(points)
    if not polygon_points:
        return None
    if len(annotation.get("bbox", [])) == 4:
        x, y, width, height = annotation["bbox"]
    else:
        combined = np.vstack(polygon_points)
        x = float(np.min(combined[:, 0]))
        y = float(np.min(combined[:, 1]))
        width = float(np.max(combined[:, 0]) - x)
        height = float(np.max(combined[:, 1]) - y)
    margin = 3
    x0 = max(0, int(np.floor(x)) - margin)
    y0 = max(0, int(np.floor(y)) - margin)
    x1 = min(int(image_record["width"]), int(np.ceil(x + width)) + margin)
    y1 = min(int(image_record["height"]), int(np.ceil(y + height)) + margin)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for points in polygon_points:
        rr, cc = polygon(
            points[:, 1] - y0,
            points[:, 0] - x0,
            shape=mask.shape,
        )
        mask[rr, cc] = True
    return mask if np.any(mask) else None


def finite_correlation(first, second):
    first = pd.to_numeric(first, errors="coerce")
    second = pd.to_numeric(second, errors="coerce")
    valid = first.notna() & second.notna()
    return float(first[valid].corr(second[valid])) if valid.sum() >= 3 else np.nan


def evaluate(coco_path, output_dir):
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    output_dir = prepare_output_dir(output_dir)
    saturn = load_pipeline()
    payload = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload.get("images", [])}
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "BODY_WIDTH_ENABLE": True,
            "BODY_WIDTH_ENDPOINT_TRIM_FRACTION": 0.125,
            "BODY_WIDTH_SAMPLE_SPACING_PX": 1.0,
            "BODY_WIDTH_SMOOTH_SIGMA_PX": 1.0,
            "BODY_WIDTH_MIN_SAMPLES": 5,
        }
    )
    synthetic_frame, synthetic_summary = evaluate_synthetic_geometry(
        saturn,
        cfg,
    )
    records = []
    started = time.perf_counter()
    for annotation in payload.get("annotations", []):
        image_record = images.get(int(annotation.get("image_id", -1)))
        if image_record is None:
            continue
        mask = annotation_mask(annotation, image_record)
        if mask is None:
            records.append({"annotation_id": annotation.get("id"), "status": "unsupported_or_empty"})
            continue
        skeleton = skeletonize(mask)
        raw_coords = np.argwhere(skeleton)
        if raw_coords.shape[0] < 2:
            records.append({"annotation_id": annotation.get("id"), "status": "no_centerline"})
            continue
        centerline = saturn.extract_geodesic_centerline_coords(raw_coords, mask.shape[1])
        topology = saturn.measure_topology(centerline, mask.shape[1], allow_loops=True)
        if topology is None or topology["geo_len"] <= 0:
            records.append({"annotation_id": annotation.get("id"), "status": "invalid_centerline"})
            continue
        result = saturn.measure_subpixel_body_width(mask, centerline, cfg)
        dilated_mask = binary_dilation(
            mask,
            structure=CROSS_STRUCTURE,
        )
        eroded_mask = binary_erosion(
            mask,
            structure=CROSS_STRUCTURE,
            border_value=0,
        )
        dilated_result = measure_mask(saturn, dilated_mask, cfg)
        eroded_result = measure_mask(saturn, eroded_mask, cfg)
        distance = distance_transform_edt(mask)
        legacy_values = 2.0 * distance[centerline[:, 0], centerline[:, 1]]
        prop = measure.regionprops(mask.astype(np.uint8))[0]
        geodesic = float(topology["geo_len"])
        records.append(
            {
                "annotation_id": int(annotation.get("id", -1)),
                "image_id": int(annotation.get("image_id", -1)),
                "file_name": image_record.get("file_name", ""),
                "status": "measured" if np.isfinite(result["body_width_px"]) else "insufficient_chords",
                "mask_area_px": int(np.count_nonzero(mask)),
                "centerline_length_px": geodesic,
                "legacy_dt_median_width_px": float(np.median(legacy_values)),
                "body_width_px": result["body_width_px"],
                "body_width_p90_px": result["body_width_p90_px"],
                "body_width_iqr_px": result["body_width_iqr_px"],
                "body_width_sample_count": result["body_width_sample_count"],
                "mask_dilate1_body_width_px": dilated_result["body_width_px"],
                "mask_dilate1_width_delta_px": (
                    dilated_result["body_width_px"] - result["body_width_px"]
                    if np.isfinite(dilated_result["body_width_px"])
                    and np.isfinite(result["body_width_px"])
                    else np.nan
                ),
                "mask_erode1_body_width_px": eroded_result["body_width_px"],
                "mask_erode1_width_delta_px": (
                    eroded_result["body_width_px"] - result["body_width_px"]
                    if np.isfinite(eroded_result["body_width_px"])
                    and np.isfinite(result["body_width_px"])
                    else np.nan
                ),
                "mask_boundary_span_px": (
                    dilated_result["body_width_px"] - eroded_result["body_width_px"]
                    if np.isfinite(dilated_result["body_width_px"])
                    and np.isfinite(eroded_result["body_width_px"])
                    else np.nan
                ),
                "area_length_width_px": float(np.count_nonzero(mask)) / geodesic,
                "minor_axis_length_px": float(prop.axis_minor_length),
                "orientation_rad": float(prop.orientation),
            }
        )
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(records)
    atomic_write_dataframe(
        output_dir / "coco_mask_width_validation.csv", frame
    )
    atomic_write_dataframe(
        output_dir / "synthetic_width_validation.csv", synthetic_frame
    )
    measured = frame[frame["status"] == "measured"].copy()
    rasterized = frame[frame["status"] != "unsupported_or_empty"].copy()
    status_counts = {
        str(key): int(value)
        for key, value in frame["status"].value_counts().to_dict().items()
    }
    body_area_difference = (
        measured["body_width_px"] - measured["area_length_width_px"]
    ).abs()
    dilation_delta = pd.to_numeric(
        measured["mask_dilate1_width_delta_px"], errors="coerce"
    ).dropna()
    erosion_delta = pd.to_numeric(
        measured["mask_erode1_width_delta_px"], errors="coerce"
    ).dropna()
    boundary_span = pd.to_numeric(
        measured["mask_boundary_span_px"], errors="coerce"
    ).dropna()
    criteria, verdict = engineering_verdict(
        synthetic_summary,
        rasterized,
        measured,
    )
    summary = {
        "schema_version": "1.0",
        "generated_at_utc": generated_at_utc,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "environment": environment_metadata(),
        "primary_user_field": PRIMARY_USER_FIELD,
        "primary_measurement_label": MEASUREMENT_LABEL,
        "engineering_validation_status": verdict,
        "absolute_biological_accuracy_status": "not_established",
        "engineering_acceptance_criteria": criteria,
        "synthetic_geometry": synthetic_summary,
        "pipeline": str(PIPELINE),
        "coco_source": str(Path(coco_path).resolve()),
        "annotation_count": int(len(frame)),
        "rasterized_mask_count": int(len(rasterized)),
        "measured_count": int(len(measured)),
        "measurement_success_fraction": float(len(measured) / max(len(frame), 1)),
        "measurement_success_fraction_of_rasterized_masks": float(
            len(measured) / max(len(rasterized), 1)
        ),
        "status_counts": status_counts,
        "elapsed_seconds": elapsed,
        "milliseconds_per_annotation": 1000.0 * elapsed / max(len(frame), 1),
        "unique_legacy_widths_rounded_0_001": int(
            measured["legacy_dt_median_width_px"].round(3).nunique()
        ),
        "unique_body_widths_rounded_0_001": int(
            measured["body_width_px"].round(3).nunique()
        ),
        "body_vs_area_length_correlation": finite_correlation(
            measured["body_width_px"], measured["area_length_width_px"]
        ),
        "body_vs_minor_axis_correlation": finite_correlation(
            measured["body_width_px"], measured["minor_axis_length_px"]
        ),
        "median_body_minus_area_length_px": float(
            (measured["body_width_px"] - measured["area_length_width_px"]).median()
        ),
        "median_absolute_body_vs_area_length_difference_px": float(
            body_area_difference.median()
        ),
        "p90_absolute_body_vs_area_length_difference_px": float(
            body_area_difference.quantile(0.90)
        ),
        "median_width_increase_from_one_pixel_mask_dilation_px": float(
            dilation_delta.median()
        ),
        "p10_p90_width_increase_from_one_pixel_mask_dilation_px": [
            float(dilation_delta.quantile(0.10)),
            float(dilation_delta.quantile(0.90)),
        ],
        "median_width_change_from_one_pixel_erosion_px": float(
            erosion_delta.median()
        ),
        "p10_p90_width_change_from_one_pixel_erosion_px": [
            float(erosion_delta.quantile(0.10)),
            float(erosion_delta.quantile(0.90)),
        ],
        "median_erode_to_dilate_boundary_span_px": float(
            boundary_span.median()
        ),
        "validation_scope": (
            "Engineering validation of apparent filled-mask width. Synthetic masks test formula "
            "and rotation behavior; COCO masks test coverage and mask-boundary sensitivity. "
            "This does not establish true physical nucleus diameter."
        ),
    }
    atomic_write_text(
        output_dir / "coco_mask_width_validation.json",
        json.dumps(summary, indent=2),
    )
    pd.DataFrame(
        [
            {
                "primary_user_field": PRIMARY_USER_FIELD,
                "display_name": MEASUREMENT_LABEL,
                "engineering_validation_status": verdict,
                "absolute_biological_accuracy_status": "not_established",
                "routine_report_action": (
                    "use primary field when available; keep alternatives in QC"
                ),
            }
        ]
    ).pipe(
        lambda decision: atomic_write_dataframe(
            output_dir / "width_measurement_decision.csv", decision
        )
    )
    validation_report = f"""# Saturn v5.7.1 Width Stability Check

## Scope

This automated check evaluates the versioned central-body contour-chord
measurement on known synthetic geometry and manually annotated COCO masks. It
validates engineering behavior, not true physical nucleus diameter.

## User Decision

- Primary biological-analysis field: `{PRIMARY_USER_FIELD}`
- Display name: {MEASUREMENT_LABEL}
- Engineering status: **{verdict.upper()}**
- Absolute biological accuracy: **NOT ESTABLISHED**
- Alternate width calculations remain technical QC and should not be selected
  by end users.

## Results

- COCO annotations: {summary['annotation_count']}
- Rasterized masks: {summary['rasterized_mask_count']}
- Widths measured: {summary['measured_count']}
- Measurement success: {summary['measurement_success_fraction_of_rasterized_masks']:.2%}
- Maximum synthetic absolute error: {synthetic_summary['maximum_absolute_error_px']:.3f} px
- Maximum synthetic rotation spread: {synthetic_summary['maximum_rotation_spread_px']:.3f} px
- Distinct legacy EDT widths at 0.001 px: {summary['unique_legacy_widths_rounded_0_001']}
- Distinct central-body widths at 0.001 px: {summary['unique_body_widths_rounded_0_001']}
- Correlation with filled-mask area/length: {summary['body_vs_area_length_correlation']:.3f}
- Median absolute difference from area/length: {summary['median_absolute_body_vs_area_length_difference_px']:.3f} px
- P90 absolute difference from area/length: {summary['p90_absolute_body_vs_area_length_difference_px']:.3f} px
- Median width increase after one-pixel mask dilation: {summary['median_width_increase_from_one_pixel_mask_dilation_px']:.3f} px
- P10-P90 mask-dilation width increase: {summary['p10_p90_width_increase_from_one_pixel_mask_dilation_px'][0]:.3f} to {summary['p10_p90_width_increase_from_one_pixel_mask_dilation_px'][1]:.3f} px
- Median width change after one-pixel mask erosion: {summary['median_width_change_from_one_pixel_erosion_px']:.3f} px
- Median erosion-to-dilation sensitivity span: {summary['median_erode_to_dilate_boundary_span_px']:.3f} px

## Interpretation

The subpixel contour-chord measurement removes the severe pixel-grid banding of
the legacy centerline distance-transform median. The area/length comparison is
an independent mask-derived consistency check, not a competing user-facing
measurement and not proof of absolute biological accuracy. The erosion and
dilation results quantify how strongly the answer follows the learned mask
boundary; they are QC evidence only.

Model C uses `train_mask_dilate_px: 0`. The erosion and dilation measurements
above are deliberate validation perturbations only; neither operation is applied
during production inference. The field remains an apparent mask width because
its boundary is learned from annotations, and no unvalidated fixed subtraction
is applied.

## Limits

- Do not call this a PSF-corrected or molecular diameter.
- Do not subtract a fixed number from widths to force agreement with expected WT
  morphology.
- A later held-out boundary study may supersede this claim, but users should not
  choose among the QC variants in routine analysis.
"""
    atomic_write_text(
        output_dir / "V5_7_1_BODY_WIDTH_VALIDATION.md", validation_report
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(measured["legacy_dt_median_width_px"], bins=40, alpha=0.65, label="Legacy EDT median")
    axes[0].hist(measured["body_width_px"], bins=40, alpha=0.65, label="Body chord")
    axes[0].set(xlabel="Width (pixels)", ylabel="Annotated nuclei", title="Width distributions")
    axes[0].legend(fontsize=8)
    axes[1].scatter(measured["area_length_width_px"], measured["body_width_px"], s=5, alpha=0.25)
    axes[1].set(xlabel="Area / centerline length (px)", ylabel="Body chord width (px)", title="Independent mask cross-check")
    axes[2].scatter(measured["minor_axis_length_px"], measured["body_width_px"], s=5, alpha=0.25)
    axes[2].set(xlabel="Ellipse minor axis (px)", ylabel="Body chord width (px)", title="Shape-geometry cross-check")
    fig.tight_layout()
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", dpi=180)
    plt.close(fig)
    atomic_write_bytes(
        output_dir / "coco_mask_width_validation.png", png_buffer.getvalue()
    )
    evidence_files = [
        "coco_mask_width_validation.csv",
        "coco_mask_width_validation.json",
        "synthetic_width_validation.csv",
        "width_measurement_decision.csv",
        "V5_7_1_BODY_WIDTH_VALIDATION.md",
        "coco_mask_width_validation.png",
    ]
    manifest = {
        "schema_version": "1.0",
        "claim_id": "MEAS-BODY-WIDTH-001",
        "generated_at_utc": generated_at_utc,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "environment": environment_metadata(),
        "pipeline": str(PIPELINE.resolve()),
        "pipeline_sha256": sha256(PIPELINE),
        "coco_source": str(Path(coco_path).resolve()),
        "coco_source_sha256": sha256(coco_path),
        "engineering_validation_status": verdict,
        "absolute_biological_accuracy_status": "not_established",
        "artifacts": [
            {
                "path": name,
                "sha256": sha256(output_dir / name),
            }
            for name in evidence_files
        ],
    }
    manifest_path = output_dir / "evidence_manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    completion = {
        "schema_version": "1.0",
        "status": "complete",
        "generated_at_utc": generated_at_utc,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
        "artifact_count": len(evidence_files),
        "qc_only": True,
        "absolute_biological_accuracy_status": "not_established",
    }
    atomic_write_text(
        output_dir / "COMPLETED.json", json.dumps(completion, indent=2)
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", default=str(DEFAULT_COCO))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    summary = evaluate(args.coco, args.output)
    print(json.dumps(summary, indent=2))
    if summary["engineering_validation_status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
