"""Generate commit-bound, identical-framing v5.7.1 segmentation stage panels."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "sperm_segmentation_saturnv5.7.1.py"
DEFAULT_PROFILE = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_v571_stage_evidence", PIPELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def roi_crop(roi, padding=20):
    yy, xx = np.where(roi)
    return (
        slice(max(int(yy.min()) - padding, 0), min(int(yy.max()) + padding + 1, roi.shape[0])),
        slice(max(int(xx.min()) - padding, 0), min(int(xx.max()) + padding + 1, roi.shape[1])),
    )


def render_panel(raw, segmentation, centerline_labels, overlay, roi, destination, title):
    crop = roi_crop(roi)
    instances = np.asarray(segmentation["unet_primary_instance_labels"])
    stages = [
        ("Raw image", raw, "gray", None, None),
        ("Foreground probability", segmentation["unet_probability"], "magma", 0, 1),
        ("Core probability", segmentation["unet_core_probability"], "magma", 0, 1),
        ("Filled foreground mask", segmentation["mask_clean"], "gray", 0, 1),
        ("Instance boundaries", find_boundaries(instances, mode="inner"), "gray", 0, 1),
        ("Measured centerlines", centerline_labels > 0, "gray", 0, 1),
        ("Final production overlay", overlay, None, None, None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for axis, stage in zip(axes.ravel(), stages):
        name, image, cmap, vmin, vmax = stage
        axis.imshow(np.asarray(image)[crop], cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(name, fontsize=10)
        axis.axis("off")
    axes.ravel()[-1].text(
        0.5,
        0.5,
        "All panels use the same ROI crop\nand identical pixel framing.",
        ha="center",
        va="center",
        fontsize=12,
    )
    axes.ravel()[-1].axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--z", type=int, default=35)
    parser.add_argument("--overlong-split-trigger-um", type=float)
    args = parser.parse_args()

    saturn = load_pipeline()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(args.pilot_manifest).read_text(encoding="utf-8"))
    evidence = []
    for specimen in records:
        cfg, _ = saturn.load_analysis_profile(args.profile, saturn.CONFIG)
        if args.overlong_split_trigger_um is not None:
            cfg["UNET_PRIMARY_OVERLONG_SPLIT_TRIGGER_UM"] = float(
                args.overlong_split_trigger_um
            )
        input_dir = Path(specimen["input_dir"])
        roi_path = Path(specimen["roi_path"])
        parsed_files = [
            (saturn._study_parse_source_name(path.name), str(path))
            for path in input_dir.iterdir()
            if path.is_file()
        ]
        parsed_files = [item for item in parsed_files if item[0] is not None]
        parsed_files.sort(key=lambda item: int(item[0]["z"]))
        files = [path for _parsed, path in parsed_files]
        z_values = [int(parsed["z"]) for parsed, _path in parsed_files]
        files_by_z = {int(z): path for path, z in zip(files, z_values)}
        if args.z not in files_by_z:
            nearest = min(files_by_z, key=lambda z: abs(z - args.z))
        else:
            nearest = args.z
        saturn.resolve_stack_microscope_calibration(cfg, files, input_dir=input_dir)
        raw = saturn.ensure_2d_image(
            saturn.robust_imread(files_by_z[nearest]),
            Path(files_by_z[nearest]).name,
        )
        roi = saturn.load_roi_mask_file(roi_path, expected_shape=raw.shape)
        context = saturn.build_stack_preprocess_context(files, roi, cfg)
        unet_context = saturn._make_unet_context_from_paths(files_by_z, nearest)
        segmentation = saturn.segment_slice(
            raw,
            cfg,
            z_idx=nearest,
            roi_mask=roi,
            preprocess_context=context,
            unet_context_stack=unet_context,
        )
        measured = saturn.measure_spermatids(segmentation, cfg)
        measured_lengths_um = np.asarray(
            [
                float(row["length_px_geodesic"]) * float(cfg["UM_PER_PX_XY"])
                for row in measured["results"]
            ],
            dtype=np.float64,
        )
        overlay = saturn.make_overlay(raw, measured["skel_label"])
        destination = output_dir / f"{specimen['specimen']}_z{nearest:03d}_stages.png"
        render_panel(
            raw,
            segmentation,
            measured["skel_label"],
            overlay,
            roi,
            destination,
            f"{specimen['specimen']} z{nearest:03d}: Saturn v5.7.1 production stages",
        )
        evidence.append(
            {
                "specimen": specimen["specimen"],
                "z_index": nearest,
                "source_image": str(Path(files_by_z[nearest]).resolve()),
                "source_sha256": sha256(files_by_z[nearest]),
                "roi_path": str(roi_path.resolve()),
                "roi_sha256": sha256(roi_path),
                "profile_path": str(Path(args.profile).resolve()),
                "profile_sha256": sha256(args.profile),
                "overlong_split_trigger_um": cfg[
                    "UNET_PRIMARY_OVERLONG_SPLIT_TRIGGER_UM"
                ],
                "checkpoint_sha256": cfg["UNET_CHECKPOINT_SHA256"],
                "artifact": str(destination),
                "artifact_sha256": sha256(destination),
                "detection_count": len(measured["results"]),
                "median_2d_length_um": (
                    float(np.median(measured_lengths_um))
                    if measured_lengths_um.size
                    else None
                ),
                "below_2_um_count": int(np.sum(measured_lengths_um < 2.0)),
                "from_15_to_20_um_count": int(
                    np.sum(
                        (measured_lengths_um >= 15.0)
                        & (measured_lengths_um <= 20.0)
                    )
                ),
                "above_20_um_count": int(np.sum(measured_lengths_um > 20.0)),
                "clear_multi_object_merge_count": int(
                    sum(
                        bool(row.get("suspected_multi_object_merge", False))
                        for row in measured["results"]
                    )
                ),
                "xy_um_per_pixel": cfg["UM_PER_PX_XY"],
                "z_um_per_slice": cfg["UM_PER_SLICE_Z"],
            }
        )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_commit = ""
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_generation": git_commit,
        "pipeline_sha256": sha256(PIPELINE),
        "generator_sha256": sha256(Path(__file__)),
        "records": evidence,
    }
    (output_dir / "visual_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
