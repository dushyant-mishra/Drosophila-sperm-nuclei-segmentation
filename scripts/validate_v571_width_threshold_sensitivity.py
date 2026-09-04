"""QC-only v5.7.1 apparent-width sensitivity to foreground threshold.

This bounded diagnostic uses one retained KJ specimen and one retained WT
specimen. It runs U-Net inference once for each of three selected planes per
specimen, repeats exactly one slice to verify bit-identical deterministic
dual-head inference, then rebuilds instances from the cached foreground and
core probability arrays at fixed foreground thresholds 0.55, 0.60, and 0.65.

The output assesses numerical sensitivity of the apparent filled-mask body
width and area. It does not validate biological accuracy and does not change
production parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "sperm_segmentation_saturnv5.7.1.py"
PROFILE = ROOT / "production_profiles/saturn_v5_7_1_model_c_epoch003.json"
EVIDENCE_ROOT = ROOT / "audits/evidence/v571_rc6_candidate"
REPLAY = EVIDENCE_ROOT / "provenance/tracking_replay_inputs_outputs.zip"
DEFAULT_OUTPUT = ROOT / "scratch/v571_width_threshold_sensitivity"

SPECIMENS = {
    "KJ-01": "kj_sv_40xx0.75-1",
    "WT-01": "w1118_sv_feb_40xx0.75-1",
}
FOREGROUND_THRESHOLDS = (0.55, 0.60, 0.65)
BASELINE_FOREGROUND_THRESHOLD = 0.60
CORE_THRESHOLD = 0.50
TARGET_Z = 35
EXPECTED_BASELINE_INFERENCE_CALLS = 6
EXPECTED_REPEAT_INFERENCE_CALLS = 1
EXPECTED_TOTAL_INFERENCE_CALLS = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def environment_metadata() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "torch", "scipy", "scikit-image"):
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


def prepare_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new audit run path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_text(path, frame.to_csv(index=False))


def configure_deterministic_torch(torch_module=None) -> dict:
    """Enforce deterministic Torch execution before model inference."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch_module is None:
        import torch as torch_module

    torch_module.manual_seed(0)
    cuda_available = bool(torch_module.cuda.is_available())
    if cuda_available:
        torch_module.cuda.manual_seed_all(0)
    torch_module.use_deterministic_algorithms(True)
    if hasattr(torch_module, "set_deterministic_debug_mode"):
        torch_module.set_deterministic_debug_mode("error")
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True
    return {
        "seed": 0,
        "deterministic_algorithms": True,
        "deterministic_debug_mode": "error",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_available": cuda_available,
    }


def verify_repeated_probabilities(
    foreground: np.ndarray,
    core: np.ndarray,
    repeated_foreground: np.ndarray,
    repeated_core: np.ndarray,
) -> dict:
    """Require bit-identical dual-head outputs from one repeated inference."""
    foreground = np.asarray(foreground)
    core = np.asarray(core)
    repeated_foreground = np.asarray(repeated_foreground)
    repeated_core = np.asarray(repeated_core)
    foreground_equal = bool(np.array_equal(foreground, repeated_foreground))
    core_equal = bool(np.array_equal(core, repeated_core))
    record = {
        "foreground_identical": foreground_equal,
        "core_identical": core_equal,
        "foreground_sha256": sha256_array(foreground),
        "repeated_foreground_sha256": sha256_array(repeated_foreground),
        "core_sha256": sha256_array(core),
        "repeated_core_sha256": sha256_array(repeated_core),
    }
    if not foreground_equal or not core_equal:
        raise RuntimeError(
            "Repeated deterministic inference produced non-identical dual-head arrays"
        )
    return record


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_width_threshold_sensitivity", PIPELINE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def retained_settings_dir(stem: str) -> Path:
    return EVIDENCE_ROOT / "provenance/specimens" / stem / "settings"


def build_specimen_provenance(
    specimen: str,
    stem: str,
    files_by_z: dict[int, str],
    roi_path: Path,
    cfg: dict,
) -> dict:
    """Bind every specimen input that can affect cached probabilities."""
    settings_dir = retained_settings_dir(stem)
    source_manifest = settings_dir / "source_image_manifest.json"
    metadata_path = settings_dir / "microscope_metadata_used.xml"
    calibration_path = settings_dir / "calibration_used.json"
    required = [roi_path, source_manifest, metadata_path, calibration_path]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing retained provenance inputs for {specimen}: {missing}"
        )
    stack_sources = [
        {
            "z_index": int(z_index),
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(Path(path)),
        }
        for z_index, path in sorted(files_by_z.items())
    ]
    return {
        "specimen": specimen,
        "source_stem": stem,
        "roi_path": str(Path(roi_path).resolve()),
        "roi_sha256": sha256_file(roi_path),
        "source_image_manifest": str(source_manifest.resolve()),
        "source_image_manifest_sha256": sha256_file(source_manifest),
        "microscope_metadata": str(metadata_path.resolve()),
        "microscope_metadata_sha256": sha256_file(metadata_path),
        "calibration_record": str(calibration_path.resolve()),
        "calibration_record_sha256": sha256_file(calibration_path),
        "resolved_xy_um_per_pixel": float(cfg["UM_PER_PX_XY"]),
        "resolved_z_um_per_slice": float(cfg["UM_PER_SLICE_Z"]),
        "stack_preprocess_source_count": len(stack_sources),
        "stack_preprocess_sources": stack_sources,
    }


def load_source_files(stem: str) -> dict[int, str]:
    manifest_path = retained_settings_dir(stem) / "source_image_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_by_z = {}
    for record in payload["ordered_source_images"]:
        z_index = int(record["z_index"])
        if int(record["channel"]) != 0:
            raise ValueError(f"Nonzero source channel at z{z_index}")
        source = Path(record["path"])
        if not source.is_file():
            raise FileNotFoundError(f"Missing retained source image z{z_index}: {source}")
        if z_index in files_by_z:
            raise ValueError(f"Duplicate retained Z index: {z_index}")
        files_by_z[z_index] = str(source)
    return files_by_z


def read_replay_table(archive: zipfile.ZipFile, stem: str, name: str) -> pd.DataFrame:
    with archive.open(f"{stem}/{name}.csv") as handle:
        return pd.read_csv(handle)


def select_track(track_summary: pd.DataFrame, detections: pd.DataFrame) -> pd.Series:
    """Select the same type of clean persistent track used by width evidence."""
    frame = track_summary.copy()
    frame = frame[
        (pd.to_numeric(frame["n_slices"], errors="coerce") >= 3)
        & pd.to_numeric(
            frame["representative_body_width_um"], errors="coerce"
        ).notna()
        & ~frame["suspected_multi_object_merge"].fillna(False).astype(bool)
    ].copy()
    frame["_z_distance"] = (
        pd.to_numeric(frame["representative_width_z"], errors="coerce") - TARGET_Z
    ).abs()

    representative = detections.copy()
    representative["z_slice"] = pd.to_numeric(
        representative["z_slice"], errors="coerce"
    )
    representative = representative.merge(
        frame[["track_id", "representative_width_z"]], on="track_id", how="inner"
    )
    representative = representative[
        representative["z_slice"]
        == pd.to_numeric(representative["representative_width_z"], errors="coerce")
    ]
    clean_ids = set(
        representative.loc[
            (pd.to_numeric(representative["n_branch_nodes"], errors="coerce") == 0)
            & pd.to_numeric(
                representative["instance_mask_area_px"], errors="coerce"
            ).between(60, 180)
            & (
                pd.to_numeric(
                    representative["length_body_width_ratio"], errors="coerce"
                )
                > 3.0
            )
            & ~representative["morphology_warning"].fillna(False).astype(bool),
            "track_id",
        ]
    )
    frame = frame[frame["track_id"].isin(clean_ids)]
    if frame.empty:
        raise ValueError("No eligible retained three-plane width target")
    return frame.sort_values(
        ["_z_distance", "track_id"], kind="mergesort"
    ).iloc[0]


def select_three_observations(
    detections: pd.DataFrame, track_id: int, representative_z: int
) -> pd.DataFrame:
    """Return exactly the representative plane and its two adjacent planes."""
    observations = detections[detections["track_id"] == track_id].copy()
    observations["z_slice"] = pd.to_numeric(
        observations["z_slice"], errors="coerce"
    )
    required = [representative_z - 1, representative_z, representative_z + 1]
    observations = observations[observations["z_slice"].isin(required)].copy()
    observations = observations.sort_values("z_slice", kind="mergesort")
    selected = [int(value) for value in observations["z_slice"].tolist()]
    if selected != required:
        raise ValueError(
            f"Track {track_id} does not have the required consecutive planes: "
            f"expected {required}, found {selected}"
        )
    return observations


def match_label_by_max_iou(
    candidate_labels: np.ndarray, baseline_mask: np.ndarray
) -> dict[str, float | int]:
    """Match one baseline mask to a candidate label using deterministic max IoU."""
    labels = np.asarray(candidate_labels)
    target = np.asarray(baseline_mask, dtype=bool)
    if labels.shape != target.shape:
        raise ValueError("Candidate labels and baseline mask must have matching shapes")
    if not np.any(target):
        raise ValueError("Baseline target mask is empty")

    candidate_ids = np.unique(labels[target])
    candidate_ids = candidate_ids[candidate_ids > 0]
    matches = []
    baseline_area = int(np.count_nonzero(target))
    for label in candidate_ids:
        candidate = labels == label
        intersection = int(np.count_nonzero(candidate & target))
        union = int(np.count_nonzero(candidate | target))
        iou = float(intersection / union) if union else 0.0
        matches.append((iou, intersection, -int(label), int(label), union))
    if not matches:
        return {
            "matched_label": 0,
            "iou": 0.0,
            "intersection_px": 0,
            "union_px": baseline_area,
        }
    iou, intersection, _negative_label, label, union = max(matches)
    return {
        "matched_label": label,
        "iou": iou,
        "intersection_px": intersection,
        "union_px": union,
    }


def relative_change_percent(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    return float(100.0 * (value - baseline) / baseline)


def max_abs_finite(values) -> float:
    finite = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(np.abs(finite))) if finite.size else float("nan")


def build_sensitivity_row(
    *,
    threshold: float,
    core_threshold: float,
    match: dict,
    baseline_area_px: float,
    matched_area_px: float,
    baseline_width_um: float,
    matched_width_um: float,
) -> dict:
    """Create deterministic threshold-relative metrics for one matched target."""
    return {
        "foreground_threshold": float(threshold),
        "core_threshold": float(core_threshold),
        "match_found": bool(int(match["matched_label"]) > 0),
        "matched_instance_id": int(match["matched_label"]),
        "match_iou": float(match["iou"]),
        "intersection_px": int(match["intersection_px"]),
        "union_px": int(match["union_px"]),
        "baseline_mask_area_px": float(baseline_area_px),
        "matched_mask_area_px": float(matched_area_px),
        "mask_area_change_px": float(matched_area_px - baseline_area_px),
        "mask_area_change_percent": relative_change_percent(
            matched_area_px, baseline_area_px
        ),
        "baseline_body_width_um": float(baseline_width_um),
        "matched_body_width_um": float(matched_width_um),
        "body_width_change_um": float(matched_width_um - baseline_width_um),
        "body_width_change_percent": relative_change_percent(
            matched_width_um, baseline_width_um
        ),
    }


def result_for_label(measured: dict, label: int) -> dict | None:
    return next(
        (row for row in measured["results"] if int(row["label"]) == int(label)),
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    output_dir = prepare_output_dir(Path(args.output_dir).resolve())
    deterministic_settings = configure_deterministic_torch()

    saturn = load_pipeline()
    records = []
    inference_records = []
    specimen_planes = {}
    specimen_provenance = {}
    deterministic_repeat = None
    total_inference_call_count = 0

    with zipfile.ZipFile(REPLAY) as archive:
        for specimen, stem in SPECIMENS.items():
            files_by_z = load_source_files(stem)
            files = [files_by_z[z] for z in sorted(files_by_z)]
            input_dir = Path(files[0]).parent
            roi_path = retained_settings_dir(stem) / "roi_mask_source.npy"
            cfg, _ = saturn.load_analysis_profile(PROFILE, saturn.CONFIG)
            if not np.isclose(
                float(cfg["UNET_FOREGROUND_THRESHOLD"]),
                BASELINE_FOREGROUND_THRESHOLD,
            ):
                raise ValueError("Production profile baseline foreground threshold changed")
            if not np.isclose(float(cfg["UNET_CORE_THRESHOLD"]), CORE_THRESHOLD):
                raise ValueError("Production profile core threshold changed")
            saturn.resolve_stack_microscope_calibration(
                cfg, files, input_dir=input_dir
            )
            specimen_provenance[specimen] = build_specimen_provenance(
                specimen,
                stem,
                files_by_z,
                roi_path,
                cfg,
            )
            raw0 = saturn.ensure_2d_image(
                saturn.robust_imread(files[0]), Path(files[0]).name
            )
            roi = saturn.load_roi_mask_file(roi_path, expected_shape=raw0.shape)
            exclusion = np.zeros(raw0.shape, dtype=bool)
            valid_mask = np.asarray(roi, dtype=bool) & ~exclusion
            preprocess = saturn.build_stack_preprocess_context(
                files, roi, cfg, exclusion_mask=exclusion
            )

            tracks = read_replay_table(archive, stem, "track_summary")
            detections = read_replay_table(archive, stem, "tracked_detections")
            selected_track = select_track(tracks, detections)
            track_id = int(selected_track["track_id"])
            representative_z = int(selected_track["representative_width_z"])
            observations = select_three_observations(
                detections, track_id, representative_z
            )
            specimen_planes[specimen] = [
                int(value) for value in observations["z_slice"]
            ]

            for observation in observations.itertuples(index=False):
                z_index = int(observation.z_slice)
                source_path = Path(files_by_z[z_index])
                raw = saturn.ensure_2d_image(
                    saturn.robust_imread(str(source_path)), source_path.name
                )

                # This is the only inference call for this specimen/plane.
                baseline_segmentation = saturn.segment_slice(
                    raw,
                    cfg,
                    z_idx=z_index,
                    roi_mask=roi,
                    exclusion_mask=exclusion,
                    preprocess_context=preprocess,
                    unet_context_stack=saturn._make_unet_context_from_paths(
                        files_by_z, z_index
                    ),
                )
                total_inference_call_count += 1
                foreground_probability = np.asarray(
                    baseline_segmentation["unet_probability"], dtype=np.float32
                )
                core_probability = np.asarray(
                    baseline_segmentation["unet_core_probability"],
                    dtype=np.float32,
                )
                if deterministic_repeat is None:
                    repeated_segmentation = saturn.segment_slice(
                        raw,
                        cfg,
                        z_idx=z_index,
                        roi_mask=roi,
                        exclusion_mask=exclusion,
                        preprocess_context=preprocess,
                        unet_context_stack=saturn._make_unet_context_from_paths(
                            files_by_z, z_index
                        ),
                    )
                    total_inference_call_count += 1
                    repeated_foreground = np.asarray(
                        repeated_segmentation["unet_probability"], dtype=np.float32
                    )
                    repeated_core = np.asarray(
                        repeated_segmentation["unet_core_probability"],
                        dtype=np.float32,
                    )
                    deterministic_repeat = verify_repeated_probabilities(
                        foreground_probability,
                        core_probability,
                        repeated_foreground,
                        repeated_core,
                    )
                    deterministic_repeat.update(
                        {
                            "specimen": specimen,
                            "z_index": z_index,
                            "source_path": str(source_path),
                            "repeat_inference_call_count": 1,
                        }
                    )
                baseline_label = int(observation.sperm_id)
                baseline_labels = np.asarray(
                    baseline_segmentation["unet_primary_instance_labels"]
                )
                baseline_mask = baseline_labels == baseline_label
                if not np.any(baseline_mask):
                    raise ValueError(
                        f"Retained target label {baseline_label} was not reproduced "
                        f"for {specimen} z{z_index}"
                    )
                baseline_measured = saturn.measure_spermatids(
                    baseline_segmentation, cfg
                )
                baseline_result = result_for_label(
                    baseline_measured, baseline_label
                )
                if baseline_result is None:
                    raise ValueError(
                        f"Baseline target {specimen} z{z_index} label "
                        f"{baseline_label} was not measurable"
                    )
                baseline_area_px = float(np.count_nonzero(baseline_mask))
                baseline_width_um = float(
                    baseline_result["body_width_px"] * cfg["UM_PER_PX_XY"]
                )
                inference_records.append(
                    {
                        "specimen": specimen,
                        "z_index": z_index,
                        "source_path": str(source_path),
                        "source_sha256": sha256_file(source_path),
                        "foreground_probability_sha256": sha256_array(
                            foreground_probability
                        ),
                        "core_probability_sha256": sha256_array(core_probability),
                        "inference_call_count": 1,
                    }
                )

                for threshold in FOREGROUND_THRESHOLDS:
                    threshold_cfg = cfg.copy()
                    threshold_cfg["UNET_FOREGROUND_THRESHOLD"] = float(threshold)
                    threshold_cfg["UNET_CORE_THRESHOLD"] = CORE_THRESHOLD
                    rebuilt = saturn._build_unet_primary_segmentation(
                        foreground_probability,
                        valid_mask,
                        threshold_cfg,
                        core_probability=core_probability,
                    )
                    match = match_label_by_max_iou(
                        rebuilt["unet_primary_instance_labels"], baseline_mask
                    )
                    matched_label = int(match["matched_label"])
                    matched_mask = np.zeros(baseline_mask.shape, dtype=bool)
                    if matched_label > 0:
                        matched_mask = (
                            np.asarray(rebuilt["unet_primary_instance_labels"])
                            == matched_label
                        )
                    measured = saturn.measure_spermatids(rebuilt, threshold_cfg)
                    matched_result = result_for_label(measured, matched_label)
                    matched_area_px = float(np.count_nonzero(matched_mask))
                    matched_width_um = (
                        float(
                            matched_result["body_width_px"]
                            * threshold_cfg["UM_PER_PX_XY"]
                        )
                        if matched_result is not None
                        else float("nan")
                    )
                    row = build_sensitivity_row(
                        threshold=threshold,
                        core_threshold=CORE_THRESHOLD,
                        match=match,
                        baseline_area_px=baseline_area_px,
                        matched_area_px=matched_area_px,
                        baseline_width_um=baseline_width_um,
                        matched_width_um=matched_width_um,
                    )
                    row.update(
                        {
                            "specimen": specimen,
                            "source_stem": stem,
                            "track_id": track_id,
                            "z_index": z_index,
                            "representative_z": representative_z,
                            "baseline_instance_id": baseline_label,
                            "matched_body_width_sample_count": int(
                                matched_result.get("body_width_sample_count", 0)
                            )
                            if matched_result is not None
                            else 0,
                            "body_width_method": str(
                                matched_result.get(
                                    "body_width_method", "unavailable"
                                )
                            )
                            if matched_result is not None
                            else "unavailable",
                            "matched_mask_area_um2": float(
                                matched_area_px * cfg["UM_PER_PX_XY"] ** 2
                            ),
                            "qc_only": True,
                        }
                    )
                    records.append(row)

    if (
        sum(row["inference_call_count"] for row in inference_records)
        != EXPECTED_BASELINE_INFERENCE_CALLS
    ):
        raise RuntimeError("Expected exactly six one-time slice inference calls")
    if (
        deterministic_repeat is None
        or total_inference_call_count != EXPECTED_TOTAL_INFERENCE_CALLS
    ):
        raise RuntimeError("Expected six baseline calls plus exactly one repeat call")

    details = pd.DataFrame(records).sort_values(
        ["specimen", "z_index", "foreground_threshold"], kind="mergesort"
    )
    details_path = output_dir / "width_threshold_sensitivity_qc.csv"
    atomic_write_dataframe(details_path, details)
    summary = (
        details.groupby(["specimen", "foreground_threshold"], sort=True)
        .agg(
            plane_count=("z_index", "count"),
            median_match_iou=("match_iou", "median"),
            median_body_width_change_percent=(
                "body_width_change_percent",
                "median",
            ),
            max_abs_body_width_change_percent=(
                "body_width_change_percent",
                max_abs_finite,
            ),
            median_mask_area_change_percent=(
                "mask_area_change_percent",
                "median",
            ),
            max_abs_mask_area_change_percent=(
                "mask_area_change_percent",
                max_abs_finite,
            ),
        )
        .reset_index()
    )
    summary_path = output_dir / "width_threshold_sensitivity_summary_qc.csv"
    atomic_write_dataframe(summary_path, summary)

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": generated_at_utc,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "environment": environment_metadata(),
        "deterministic_torch_settings": deterministic_settings,
        "purpose": "QC-only apparent filled-mask width and area threshold sensitivity",
        "biological_accuracy_claim": False,
        "production_parameters_changed": False,
        "user_facing_biological_report": False,
        "pipeline": str(PIPELINE),
        "pipeline_sha256": sha256_file(PIPELINE),
        "profile": str(PROFILE),
        "profile_sha256": sha256_file(PROFILE),
        "checkpoint_sha256": sha256_file(
            ROOT / "model_checkpoints/v571_model_c_dual_head_epoch003.pt"
        ),
        "replay_archive": str(REPLAY.resolve()),
        "replay_archive_sha256": sha256_file(REPLAY),
        "foreground_thresholds": list(FOREGROUND_THRESHOLDS),
        "baseline_foreground_threshold": BASELINE_FOREGROUND_THRESHOLD,
        "fixed_core_threshold": CORE_THRESHOLD,
        "specimen_planes": specimen_planes,
        "specimen_provenance": specimen_provenance,
        "baseline_inference_call_count": EXPECTED_BASELINE_INFERENCE_CALLS,
        "repeat_inference_call_count": EXPECTED_REPEAT_INFERENCE_CALLS,
        "total_inference_call_count": total_inference_call_count,
        "inference_records": inference_records,
        "deterministic_repeat": deterministic_repeat,
        "details_csv": str(details_path),
        "details_csv_sha256": sha256_file(details_path),
        "summary_csv": str(summary_path),
        "summary_csv_sha256": sha256_file(summary_path),
        "interpretation_limit": (
            "These measurements quantify threshold sensitivity of apparent "
            "mask-derived width and area only; they do not establish true "
            "physical nucleus width or biological accuracy."
        ),
    }
    manifest_path = output_dir / "width_threshold_sensitivity_manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    completion = {
        "schema_version": "1.0",
        "status": "complete",
        "generated_at_utc": generated_at_utc,
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_count": 2,
        "qc_only": True,
        "biological_accuracy_claim": False,
    }
    atomic_write_text(
        output_dir / "COMPLETED.json", json.dumps(completion, indent=2)
    )
    print(summary.to_string(index=False))
    print(f"QC-only artifacts: {output_dir}")


if __name__ == "__main__":
    main()
