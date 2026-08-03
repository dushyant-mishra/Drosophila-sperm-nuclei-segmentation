"""Validate Saturn v5.7.1 body-width geometry on COCO instance masks."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, distance_transform_edt
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


def load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_v571_width_validation", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            structure=np.array(
                [[False, True, False], [True, True, True], [False, True, False]],
                dtype=bool,
            ),
        )
        dilated_centerline = np.argwhere(skeletonize(dilated_mask))
        dilated_result = saturn.measure_subpixel_body_width(
            dilated_mask,
            dilated_centerline,
            cfg,
        )
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
                "train_target_dilate1_body_width_px": dilated_result["body_width_px"],
                "train_target_dilate1_width_delta_px": (
                    dilated_result["body_width_px"] - result["body_width_px"]
                    if np.isfinite(dilated_result["body_width_px"])
                    and np.isfinite(result["body_width_px"])
                    else np.nan
                ),
                "area_length_width_px": float(np.count_nonzero(mask)) / geodesic,
                "minor_axis_length_px": float(prop.axis_minor_length),
                "orientation_rad": float(prop.orientation),
            }
        )
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "coco_mask_width_validation.csv", index=False)
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
        measured["train_target_dilate1_width_delta_px"], errors="coerce"
    ).dropna()
    summary = {
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
        "median_width_increase_from_training_target_dilation_px": float(
            dilation_delta.median()
        ),
        "p10_p90_width_increase_from_training_target_dilation_px": [
            float(dilation_delta.quantile(0.10)),
            float(dilation_delta.quantile(0.90)),
        ],
        "validation_scope": (
            "COCO masks test measurement coverage and agreement among mask-derived widths. "
            "Absolute biological width error requires independent manual perpendicular-width labels."
        ),
    }
    (output_dir / "coco_mask_width_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    validation_report = f"""# Saturn v5.7.1 Body-Width Validation

## Scope

This validation evaluates the versioned central-body contour-chord measurement on
manually annotated COCO instance masks. It does not replace independent manual
perpendicular-width measurements or optical PSF calibration.

## Results

- COCO annotations: {summary['annotation_count']}
- Rasterized masks: {summary['rasterized_mask_count']}
- Widths measured: {summary['measured_count']}
- Measurement success: {summary['measurement_success_fraction_of_rasterized_masks']:.2%}
- Distinct legacy EDT widths at 0.001 px: {summary['unique_legacy_widths_rounded_0_001']}
- Distinct central-body widths at 0.001 px: {summary['unique_body_widths_rounded_0_001']}
- Correlation with filled-mask area/length: {summary['body_vs_area_length_correlation']:.3f}
- Median absolute difference from area/length: {summary['median_absolute_body_vs_area_length_difference_px']:.3f} px
- P90 absolute difference from area/length: {summary['p90_absolute_body_vs_area_length_difference_px']:.3f} px
- Median width increase after the exact one-pixel training-target dilation: {summary['median_width_increase_from_training_target_dilation_px']:.3f} px
- P10-P90 target-dilation width increase: {summary['p10_p90_width_increase_from_training_target_dilation_px'][0]:.3f} to {summary['p10_p90_width_increase_from_training_target_dilation_px'][1]:.3f} px

## Interpretation

The subpixel contour-chord measurement removes the severe pixel-grid banding of
the legacy centerline distance-transform median. The area/length agreement is an
independent mask-derived consistency check, not proof of absolute biological
accuracy. The reported width remains sensitive to focus, PSF, and mask boundary
quality.

The checkpoint used for the current KJ/WT work was trained with
`train_mask_dilate_px: 1`. That training-only target expansion can teach wider
boundaries, although it does not directly dilate inference output. Therefore this
field is reported as an apparent mask width; no unvalidated fixed subtraction is
applied.

## Remaining Validation

- Compare against independently drawn manual perpendicular-width labels.
- Calibrate probability thresholds and boundary targets on a held-out width set;
  do not select them to reproduce an expected WT width.
- Review narrow, wide, curved, short, and fused-looking WT and mutant examples.
- Confirm representative-plane selection on linked observations of the same nucleus.
- Recalculate group comparisons only after those visual checks pass.
"""
    (output_dir / "V5_7_1_BODY_WIDTH_VALIDATION.md").write_text(
        validation_report,
        encoding="utf-8",
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
    fig.savefig(output_dir / "coco_mask_width_validation.png", dpi=180)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", default=str(DEFAULT_COCO))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    summary = evaluate(args.coco, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
