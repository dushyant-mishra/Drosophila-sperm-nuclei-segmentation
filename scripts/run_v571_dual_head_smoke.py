"""Run a compact Model C dual-head smoke test on representative stack slices."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from skimage import color, segmentation


ROOT = Path(__file__).resolve().parents[1]


def load_saturn():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_dual_head_smoke",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_z_values(text):
    values = [int(value.strip()) for value in str(text).split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("--z-values must contain unique comma-separated integers")
    return values


def display_image(saturn, image):
    return np.clip(saturn.normalize_display(image), 0, 1)


def instance_overlay(saturn, image, labels):
    base = np.dstack([display_image(saturn, image)] * 3)
    boundaries = segmentation.find_boundaries(labels, mode="outer")
    base[boundaries] = (0.0, 1.0, 1.0)
    return base


def centerline_overlay(saturn, image, labels):
    base = np.dstack([display_image(saturn, image)] * 3)
    base[labels > 0] = (0.2, 1.0, 0.2)
    return base


def save_review_page(pdf, saturn, image, seg, results, specimen, z_value):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
    panels = [
        (display_image(saturn, image), "Raw image", "gray", None, None),
        (seg["unet_probability"], "Foreground probability", "magma", 0, 1),
        (seg["unet_core_probability"], "Core probability", "viridis", 0, 1),
        (
            instance_overlay(saturn, image, seg["unet_primary_instance_labels"]),
            "Watershed instances",
            None,
            None,
            None,
        ),
        (
            centerline_overlay(saturn, image, seg["skel_labeled"]),
            "Measured centerlines",
            None,
            None,
            None,
        ),
    ]
    for axis, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    lengths = [float(row["length_px_geodesic"]) for row in results]
    widths = [float(row.get("body_width_px", np.nan)) for row in results]
    valid_widths = [value for value in widths if np.isfinite(value)]
    fig.suptitle(
        f"{specimen} z{z_value:02d}: {len(results)} 2D instances | "
        f"median length {np.median(lengths) if lengths else np.nan:.1f} px | "
        f"median body width {np.median(valid_widths) if valid_widths else np.nan:.1f} px",
        fontsize=13,
        fontweight="bold",
    )
    pdf.savefig(fig, dpi=180)
    plt.close(fig)


def run_specimen(saturn, specimen, input_dir, roi_path, z_values, profile, output):
    cfg, _ = saturn.load_analysis_profile(profile)
    saturn.validate_analysis_runtime_config(cfg)
    cfg.update({
        "SAVE_DEBUG_IMAGES": False,
        "SAVE_OVERLAYS": False,
        "SAVE_DETAIL_FIGURE": False,
        "SHOW_PREVIEW_WINDOW": False,
        "SHOW_DEBUG_PREVIEW": False,
    })
    files, indices = saturn.load_batch_files(str(input_dir), cfg["FILE_PATTERN"])
    files_by_z = {int(z): path for path, z in zip(files, indices)}
    missing = [z for z in z_values if z not in files_by_z]
    if missing:
        raise ValueError(f"{specimen} is missing requested slices: {missing}")
    saturn.resolve_stack_microscope_calibration(
        cfg, files, input_dir=str(input_dir), require_metadata=False
    )
    first = saturn.robust_imread(files_by_z[z_values[0]])
    roi = saturn.load_roi_mask_file(roi_path, expected_shape=first.shape)
    exclusion = np.zeros(first.shape, dtype=bool)
    preprocess = saturn.build_stack_preprocess_context(
        files, roi, cfg, exclusion_mask=exclusion
    )

    specimen_dir = output / specimen
    specimen_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pdf_path = specimen_dir / f"{specimen}_dual_head_review.pdf"
    with PdfPages(pdf_path) as pdf:
        for z_value in z_values:
            image = saturn.robust_imread(files_by_z[z_value])
            context = saturn._make_unet_context_from_paths(files_by_z, z_value)
            seg = saturn.segment_slice(
                image,
                cfg,
                z_idx=z_value,
                roi_mask=roi,
                exclusion_mask=exclusion,
                preprocess_context=preprocess,
                unet_context_stack=context,
            )
            measured = saturn.measure_spermatids(seg, cfg)
            slice_rows = saturn.rows_from_results(
                measured["results"], z_value, cfg["UM_PER_PX_XY"]
            )
            rows.extend(slice_rows)
            save_review_page(
                pdf, saturn, image, seg, measured["results"], specimen, z_value
            )
            np.savez_compressed(
                specimen_dir / f"z{z_value:02d}_dual_head_stages.npz",
                foreground_probability=seg["unet_probability"],
                core_probability=seg["unet_core_probability"],
                instance_labels=seg["unet_primary_instance_labels"],
                centerline_labels=seg["skel_labeled"],
                roi=roi.astype(np.uint8),
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(specimen_dir / "slice_measurements.csv", index=False)
    summary_rows = []
    for z_value in z_values:
        part = frame[frame["z_slice"] == z_value] if not frame.empty else frame
        summary_rows.append({
            "specimen": specimen,
            "z_slice": z_value,
            "instance_2d_count": int(len(part)),
            "median_length_um": float(part["length_um_geodesic"].median()) if not part.empty else np.nan,
            "median_body_width_um": float(part["body_width_um"].median()) if not part.empty else np.nan,
            "median_legacy_width_um": float(part["width_um_dt_median_legacy"].median()) if not part.empty else np.nan,
            "body_width_available_fraction": float(part["body_width_um"].notna().mean()) if not part.empty else 0.0,
            "objects_15_to_20_um": int(((part["length_um_geodesic"] >= 15) & (part["length_um_geodesic"] <= 20)).sum()) if not part.empty else 0,
            "objects_over_20_um": int((part["length_um_geodesic"] > 20).sum()) if not part.empty else 0,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(specimen_dir / "slice_summary.csv", index=False)
    saturn.save_calibration_provenance(specimen_dir, cfg)
    saturn.save_analysis_settings_bundle(specimen_dir, cfg)
    return summary, pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--z-values", default="33,35,37")
    parser.add_argument(
        "--specimen",
        action="append",
        required=True,
        help="NAME|INPUT_DIR|ROI_PATH; repeat for each specimen",
    )
    args = parser.parse_args()

    saturn = load_saturn()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    z_values = parse_z_values(args.z_values)
    all_summaries = []
    artifacts = []
    for value in args.specimen:
        parts = value.split("|", 2)
        if len(parts) != 3:
            raise ValueError("--specimen must use NAME|INPUT_DIR|ROI_PATH")
        summary, pdf_path = run_specimen(
            saturn,
            parts[0],
            Path(parts[1]),
            Path(parts[2]),
            z_values,
            Path(args.profile),
            output,
        )
        all_summaries.append(summary)
        artifacts.append(str(pdf_path))
    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(output / "combined_smoke_summary.csv", index=False)
    manifest = {
        "pipeline": "sperm_segmentation_saturnv5.7.1.py",
        "profile": str(Path(args.profile).resolve()),
        "z_values": z_values,
        "review_pdfs": artifacts,
        "status": "first candidate for visual inspection",
    }
    (output / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(combined.to_string(index=False))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
