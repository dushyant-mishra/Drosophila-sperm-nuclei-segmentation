"""Check whether signal-width availability is biased, and whether bias differs by group.

Hardening the intensity profile made it refuse to measure any object whose
half-maximum crossing is not observed on both sides inside its own instance
mask. That is the correct fail-closed behaviour, but it drops roughly a fifth of
detections, so the measured set is a subset rather than the whole. If the objects
that drop out are systematically different, and if they drop out at different
rates in the groups being compared, the subset alone could manufacture or hide a
group difference.

This is technical quality control on measurement availability. It deliberately
reports no biological morphology comparison, and nothing here may be used to
tune a parameter toward a genotype outcome.

Usage:
    python scripts/validate_v571_width_availability_bias.py --output <dir>
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "sperm_segmentation_saturnv5.7.1.py"
PROFILE = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
DEFAULT_MANIFEST = (
    ROOT / "scratch" / "aborted_v57_full_20260730_0630" / "study_manifest.csv"
)


def load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_width_availability", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampled_planes(z_min, z_max, count):
    """Evenly spaced planes, avoiding the extreme top and bottom of the stack."""
    span = int(z_max) - int(z_min)
    low = int(z_min) + max(1, int(0.2 * span))
    high = int(z_max) - max(1, int(0.2 * span))
    if high <= low:
        low, high = int(z_min), int(z_max)
    return sorted({int(round(v)) for v in np.linspace(low, high, count)})


def measure_specimen(saturn, row, planes_per_specimen):
    """Return one row per detection for a few sampled planes of one specimen."""
    input_dir = Path(str(row["input_dir"]))
    files = sorted(input_dir.glob(str(row["file_pattern"])))
    if not files:
        return pd.DataFrame(), "no_source_files"
    files_by_z = {}
    for path in files:
        token = path.stem.split("_z")[-1].split("_")[0]
        try:
            files_by_z[int(token)] = str(path)
        except ValueError:
            continue
    if not files_by_z:
        return pd.DataFrame(), "unparsable_z_indices"

    cfg, _ = saturn.load_analysis_profile(PROFILE, saturn.CONFIG)
    ordered = [files_by_z[z] for z in sorted(files_by_z)]
    saturn.resolve_stack_microscope_calibration(cfg, ordered, input_dir=input_dir)
    um = float(cfg["UM_PER_PX_XY"])
    first = saturn.ensure_2d_image(
        saturn.robust_imread(ordered[0]), Path(ordered[0]).name
    )
    roi = saturn.load_roi_mask_file(
        Path(str(row["roi_path"])), expected_shape=first.shape
    )
    context = saturn.build_stack_preprocess_context(ordered, roi, cfg)
    roi_area_um2 = float(np.count_nonzero(roi)) * (um ** 2)

    records = []
    available_z = sorted(files_by_z)
    planes = sampled_planes(min(available_z), max(available_z), planes_per_specimen)
    for z_index in planes:
        if z_index not in files_by_z:
            continue
        raw = saturn.ensure_2d_image(
            saturn.robust_imread(files_by_z[z_index]), Path(files_by_z[z_index]).name
        )
        seg = saturn.segment_slice(
            raw,
            cfg,
            z_idx=z_index,
            roi_mask=roi,
            preprocess_context=context,
            unet_context_stack=saturn._make_unet_context_from_paths(
                files_by_z, z_index
            ),
        )
        rows = saturn.rows_from_results(
            saturn.measure_spermatids(seg, cfg)["results"], z_index, um
        )
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame["z_slice"] = z_index
        frame["sample_id"] = str(row["sample_id"])
        frame["group"] = str(row["group"])
        frame["roi_area_um2"] = roi_area_um2
        frame["um_per_px_xy"] = um
        records.append(frame)
    if not records:
        return pd.DataFrame(), "no_detections"
    return pd.concat(records, ignore_index=True), "ok"


def nearest_neighbour_um(frame):
    """Median nearest-neighbour centroid distance as a local crowding proxy."""
    distances = []
    for _, plane in frame.groupby("z_slice"):
        xs = pd.to_numeric(plane.get("centroid_x"), errors="coerce").to_numpy(float)
        ys = pd.to_numeric(plane.get("centroid_y"), errors="coerce").to_numpy(float)
        um = float(plane["um_per_px_xy"].iloc[0])
        keep = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[keep], ys[keep]
        if xs.size < 2:
            continue
        points = np.column_stack([ys, xs])
        deltas = points[:, None, :] - points[None, :, :]
        separation = np.sqrt((deltas ** 2).sum(axis=2))
        np.fill_diagonal(separation, np.inf)
        distances.append(float(np.median(separation.min(axis=1))) * um)
    return float(np.median(distances)) if distances else np.nan


def summarize(frame):
    """Per-specimen availability and the selection bias it implies."""
    out = []
    for (sample_id, group), block in frame.groupby(["sample_id", "group"]):
        width = pd.to_numeric(block.get("intensity_fwhm_width_um"), errors="coerce")
        mask_um = pd.to_numeric(
            block.get("body_width_px"), errors="coerce"
        ) * pd.to_numeric(block["um_per_px_xy"], errors="coerce")
        available = width.notna()
        planes = int(block["z_slice"].nunique())
        roi_area = float(block["roi_area_um2"].iloc[0])
        methods = block.get("intensity_width_method", pd.Series(dtype=str)).astype(str)
        out.append(
            {
                "sample_id": sample_id,
                "group": group,
                "planes_sampled": planes,
                "detections": int(len(block)),
                "width_available": int(available.sum()),
                "width_unavailable_fraction": float(1.0 - available.mean()),
                "boundary_clipped_fraction": float(
                    (methods == "unavailable_boundary_clipped_profiles").mean()
                ),
                "median_signal_width_um": float(width.median()),
                "median_mask_width_um_available": float(mask_um[available].median()),
                "median_mask_width_um_unavailable": float(mask_um[~available].median()),
                "mask_width_selection_bias_um": float(
                    mask_um[~available].median() - mask_um[available].median()
                ),
                "detections_per_1000um2": float(
                    len(block) / max(planes, 1) / max(roi_area, 1e-9) * 1000.0
                ),
                "median_nearest_neighbour_um": nearest_neighbour_um(block),
                "unavailable_reasons": json.dumps(
                    methods[~available].value_counts().to_dict(), sort_keys=True
                ),
            }
        )
    return pd.DataFrame(out).sort_values(["group", "sample_id"]).reset_index(drop=True)


def group_contrast(summary, column):
    """Descriptive contrast of a technical QC quantity between the two groups."""
    groups = sorted(summary["group"].unique())
    if len(groups) != 2:
        return {"note": "expected two groups, found " + ", ".join(groups)}
    first = summary.loc[summary["group"] == groups[0], column].astype(float).dropna()
    second = summary.loc[summary["group"] == groups[1], column].astype(float).dropna()
    if len(first) < 2 or len(second) < 2:
        return {"note": "insufficient specimens"}
    test = stats.ttest_ind(second, first, equal_var=False)
    return {
        "column": column,
        groups[0]: {
            "n": int(len(first)),
            "median": float(first.median()),
            "mean": float(first.mean()),
            "sd": float(first.std(ddof=1)),
        },
        groups[1]: {
            "n": int(len(second)),
            "median": float(second.median()),
            "mean": float(second.mean()),
            "sd": float(second.std(ddof=1)),
        },
        "difference_of_means": float(second.mean() - first.mean()),
        "welch_p_value": float(test.pvalue),
    }


def spearman_or_none(left, right, minimum=4):
    valid = left.notna() & right.notna()
    if int(valid.sum()) < minimum:
        return None
    return float(stats.spearmanr(left[valid], right[valid]).statistic)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", required=True)
    parser.add_argument("--planes-per-specimen", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 means all specimens")
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("refusing to overwrite existing evidence directory: " + str(output))
    output.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    if "include" in manifest.columns:
        manifest = manifest[manifest["include"].astype(str).str.lower() == "true"]
    if args.limit:
        per_group = max(1, args.limit // 2)
        manifest = pd.concat(
            [block.head(per_group) for _, block in manifest.groupby("group")]
        )

    saturn = load_pipeline()
    frames = []
    skipped = []
    for _, row in manifest.iterrows():
        frame, status = measure_specimen(saturn, row, args.planes_per_specimen)
        print(
            "  {0:28s} {1:3s} {2} n={3}".format(
                str(row["sample_id"]), str(row["group"]), status, len(frame)
            ),
            flush=True,
        )
        if status != "ok":
            skipped.append({"sample_id": str(row["sample_id"]), "reason": status})
            continue
        frames.append(frame)
    if not frames:
        raise SystemExit("no specimen produced detections")

    detections = pd.concat(frames, ignore_index=True)
    summary = summarize(detections)
    summary.to_csv(output / "width_availability_by_specimen.csv", index=False)

    availability = summary["width_unavailable_fraction"].astype(float)
    crowding = summary["median_nearest_neighbour_um"].astype(float)
    density = summary["detections_per_1000um2"].astype(float)

    report = {
        "schema_version": "1.0",
        "claim_id": "MEAS-INTENSITY-WIDTH-001",
        "scope": (
            "Technical quality control on signal-width availability. This is not a "
            "biological morphology comparison and must not be used to tune any "
            "parameter toward a genotype outcome."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "pipeline_sha256": sha256(PIPELINE),
        "profile_sha256": sha256(PROFILE),
        "planes_per_specimen": int(args.planes_per_specimen),
        "specimens": int(len(summary)),
        "skipped": skipped,
        "total_detections": int(len(detections)),
        "overall_unavailable_fraction": float(
            1.0
            - pd.to_numeric(detections["intensity_fwhm_width_um"], errors="coerce")
            .notna()
            .mean()
        ),
        "group_contrast_unavailable_fraction": group_contrast(
            summary, "width_unavailable_fraction"
        ),
        "group_contrast_selection_bias": group_contrast(
            summary, "mask_width_selection_bias_um"
        ),
        "group_contrast_median_signal_width": group_contrast(
            summary, "median_signal_width_um"
        ),
        "availability_vs_crowding_spearman": spearman_or_none(availability, crowding),
        "availability_vs_density_spearman": spearman_or_none(availability, density),
    }
    (output / "width_availability_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "# Signal-width availability bias (technical QC)",
        "",
        report["scope"],
        "",
        "- Specimens: {0}".format(report["specimens"]),
        "- Planes per specimen: {0}".format(report["planes_per_specimen"]),
        "- Detections: {0}".format(report["total_detections"]),
        "- Overall unavailable fraction: {0:.4f}".format(
            report["overall_unavailable_fraction"]
        ),
        "- Availability vs crowding (Spearman): {0}".format(
            report["availability_vs_crowding_spearman"]
        ),
        "- Availability vs density (Spearman): {0}".format(
            report["availability_vs_density_spearman"]
        ),
        "",
        "## Group contrast of the unavailable fraction",
        "",
        "```json",
        json.dumps(report["group_contrast_unavailable_fraction"], indent=2),
        "```",
        "",
        "## Group contrast of the mask-width selection bias",
        "",
        "```json",
        json.dumps(report["group_contrast_selection_bias"], indent=2),
        "```",
        "",
        "A positive selection bias means the dropped objects had wider masks than",
        "the measured ones, so the measured subset under-represents wide objects.",
        "",
        "The signal-width contrast is included only so a reviewer can judge whether",
        "an availability difference is large enough to matter relative to it. It is",
        "a technical readout on sampled planes, not a biological result.",
    ]
    (output / "WIDTH_AVAILABILITY_BIAS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\nwrote evidence to " + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
