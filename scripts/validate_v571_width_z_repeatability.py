"""Audit v5.7.1 apparent body-width repeatability across observed Z planes.

This is repeatability QC for retained replay data. It does not validate true
physical nucleus width and it does not change the analysis population.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
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
DEFAULT_ARCHIVE = (
    ROOT
    / "audits"
    / "evidence"
    / "v571_rc6_candidate"
    / "provenance"
    / "tracking_replay_inputs_outputs.zip"
)
DEFAULT_OUTPUT = ROOT / "scratch" / "v571_width_z_repeatability_qc"

SPECIMENS = {
    "KJ-01": "kj_sv_40xx0.75-1",
    "WT-01": "w1118_sv_feb_40xx0.75-1",
}
MIN_WIDTH_PLANES = 3
DEFAULT_MATERIAL_ABSOLUTE_UM = 0.5
DEFAULT_MATERIAL_RELATIVE_FRACTION = 0.20

TRACK_REQUIRED = {
    "track_id",
    "technical_valid",
    "observed_slice_count",
    "representative_width_z",
}
DETECTION_REQUIRED = {
    "track_id",
    "z_slice",
    "body_width_um",
    "instance_mask_area_px",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata() -> dict:
    packages = {}
    for name in ("numpy", "pandas"):
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


def boolean_series(series: pd.Series) -> pd.Series:
    """Parse persisted booleans without treating non-empty strings as true."""
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def _read_csv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise ValueError(f"Replay archive is missing required member: {member}") from exc
    return pd.read_csv(io.BytesIO(payload))


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _finite_positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.isfinite(numeric) & (numeric > 0)


def _track_record(
    specimen_id: str,
    specimen_path: str,
    summary_row: pd.Series,
    observations: pd.DataFrame,
    material_absolute_um: float,
    material_relative_fraction: float,
) -> dict | None:
    observations = observations.copy()
    observations["z_slice"] = pd.to_numeric(observations["z_slice"], errors="coerce")
    observations["body_width_um"] = pd.to_numeric(
        observations["body_width_um"], errors="coerce"
    )
    observations["instance_mask_area_px"] = pd.to_numeric(
        observations["instance_mask_area_px"], errors="coerce"
    )
    observations = observations[
        np.isfinite(observations["z_slice"])
        & _finite_positive(observations["body_width_um"])
    ].copy()
    if observations["z_slice"].duplicated().any():
        track_id = summary_row["track_id"]
        raise ValueError(
            f"{specimen_id} track {track_id} has multiple width observations "
            "from the same Z plane"
        )
    if observations["z_slice"].nunique() < MIN_WIDTH_PLANES:
        return None

    representative_z = pd.to_numeric(
        pd.Series([summary_row["representative_width_z"]]), errors="coerce"
    ).iloc[0]
    if not np.isfinite(representative_z):
        return None
    representative = observations[observations["z_slice"] == representative_z]
    if len(representative) != 1:
        return None
    representative_row = representative.iloc[0]
    representative_width = float(representative_row["body_width_um"])

    widths = observations["body_width_um"].astype(float)
    mean_width = float(widths.mean())
    sd_width = float(widths.std(ddof=1))
    width_cv = sd_width / mean_width if mean_width > 0 else math.nan
    adjacent = observations[
        observations["z_slice"].isin([representative_z - 1, representative_z + 1])
    ].sort_values("z_slice")
    adjacent_absolute = (adjacent["body_width_um"] - representative_width).abs()
    adjacent_relative = adjacent_absolute / representative_width
    material_flags = (
        (adjacent_absolute >= material_absolute_um)
        & (adjacent_relative >= material_relative_fraction)
    )

    adjacent_z = ";".join(str(int(value)) for value in adjacent["z_slice"])
    adjacent_widths = ";".join(
        f"{float(value):.6g}" for value in adjacent["body_width_um"]
    )
    return {
        "specimen_id": specimen_id,
        "archive_specimen_path": specimen_path,
        "track_id": summary_row["track_id"],
        "observed_plane_count": int(
            pd.to_numeric(summary_row["observed_slice_count"], errors="coerce")
        ),
        "width_available_plane_count": int(observations["z_slice"].nunique()),
        "width_mean_um": mean_width,
        "width_sd_um": sd_width,
        "width_cv": width_cv,
        "width_min_um": float(widths.min()),
        "width_max_um": float(widths.max()),
        "width_range_um": float(widths.max() - widths.min()),
        "representative_z": int(representative_z),
        "representative_width_um": representative_width,
        "representative_instance_mask_area_px": float(
            representative_row["instance_mask_area_px"]
        ),
        "adjacent_width_plane_count": int(len(adjacent)),
        "adjacent_z": adjacent_z,
        "adjacent_widths_um": adjacent_widths,
        "max_adjacent_absolute_difference_um": (
            float(adjacent_absolute.max()) if len(adjacent) else math.nan
        ),
        "max_adjacent_relative_difference": (
            float(adjacent_relative.max()) if len(adjacent) else math.nan
        ),
        "representative_materially_differs_from_adjacent": bool(
            material_flags.any()
        ),
    }


def _quantile(series: pd.Series, value: float) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.quantile(value)) if len(numeric) else None


def _specimen_summary(frame: pd.DataFrame) -> dict:
    adjacent = frame[frame["adjacent_width_plane_count"] > 0]
    material_count = int(
        adjacent["representative_materially_differs_from_adjacent"].sum()
    )
    return {
        "eligible_track_count": int(len(frame)),
        "tracks_with_adjacent_width_observation": int(len(adjacent)),
        "tracks_without_adjacent_width_observation": int(len(frame) - len(adjacent)),
        "median_width_cv": _quantile(frame["width_cv"], 0.5),
        "p90_width_cv": _quantile(frame["width_cv"], 0.9),
        "median_width_range_um": _quantile(frame["width_range_um"], 0.5),
        "p90_width_range_um": _quantile(frame["width_range_um"], 0.9),
        "median_max_adjacent_absolute_difference_um": _quantile(
            adjacent["max_adjacent_absolute_difference_um"], 0.5
        ),
        "p90_max_adjacent_absolute_difference_um": _quantile(
            adjacent["max_adjacent_absolute_difference_um"], 0.9
        ),
        "material_adjacent_difference_track_count": material_count,
        "material_adjacent_difference_track_fraction": (
            material_count / len(adjacent) if len(adjacent) else None
        ),
    }


def _primary_conclusion(overall: dict) -> str:
    fraction = overall["material_adjacent_difference_track_fraction"]
    if fraction is None:
        return (
            "Repeatability QC is inconclusive because no eligible representative "
            "planes had an adjacent width observation."
        )
    if fraction <= 0.10:
        return (
            "In this retained replay, largest-area representative-plane widths "
            "were usually repeatable against adjacent observed planes; material "
            f"QC differences occurred in {fraction:.1%} of evaluable tracks."
        )
    if fraction <= 0.25:
        return (
            "In this retained replay, representative-plane widths were generally "
            "repeatable, with a minority of tracks requiring width-sensitivity "
            f"attention ({fraction:.1%} materially different)."
        )
    return (
        "In this retained replay, representative-plane width frequently differed "
        "from adjacent observations; width should remain sensitivity-qualified "
        f"({fraction:.1%} materially different)."
    )


def analyze_archive(
    archive_path: Path,
    material_absolute_um: float = DEFAULT_MATERIAL_ABSOLUTE_UM,
    material_relative_fraction: float = DEFAULT_MATERIAL_RELATIVE_FRACTION,
) -> tuple[pd.DataFrame, dict]:
    """Analyze the two retained specimens without modifying production data."""
    archive_path = Path(archive_path)
    if material_absolute_um <= 0 or not 0 < material_relative_fraction < 1:
        raise ValueError("Material-difference thresholds must be positive and bounded")

    records = []
    eligible_before_width = {}
    with zipfile.ZipFile(archive_path) as archive:
        for specimen_id, specimen_path in SPECIMENS.items():
            tracked = _read_csv(
                archive, f"{specimen_path}/tracked_detections.csv"
            )
            summaries = _read_csv(archive, f"{specimen_path}/track_summary.csv")
            _require_columns(tracked, DETECTION_REQUIRED, "tracked_detections.csv")
            _require_columns(summaries, TRACK_REQUIRED, "track_summary.csv")
            observed_count = pd.to_numeric(
                summaries["observed_slice_count"], errors="coerce"
            )
            valid = summaries[
                boolean_series(summaries["technical_valid"])
                & (observed_count >= MIN_WIDTH_PLANES)
            ].copy()
            eligible_before_width[specimen_id] = int(len(valid))
            grouped = {key: part for key, part in tracked.groupby("track_id")}
            for _, summary_row in valid.iterrows():
                observations = grouped.get(summary_row["track_id"])
                if observations is None:
                    continue
                record = _track_record(
                    specimen_id,
                    specimen_path,
                    summary_row,
                    observations,
                    material_absolute_um,
                    material_relative_fraction,
                )
                if record is not None:
                    records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No technically valid tracks met the width-repeatability criteria")
    specimens = {
        specimen_id: _specimen_summary(frame[frame["specimen_id"] == specimen_id])
        for specimen_id in SPECIMENS
    }
    overall = _specimen_summary(frame)
    summary = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "environment": environment_metadata(),
        "analysis_type": "repeatability_qc",
        "biological_truth_status": "not_established",
        "population": (
            "technical_valid tracks with at least 3 distinct observed planes "
            "having finite positive body_width_um"
        ),
        "width_field": "body_width_um",
        "representative_plane_definition": (
            "recorded v5.7.1 largest-filled-mask-area representative width plane"
        ),
        "adjacent_plane_definition": "same track at representative_z - 1 or + 1",
        "material_difference_definition": (
            "absolute difference >= threshold AND relative difference >= threshold; "
            "descriptive QC only, never an exclusion rule"
        ),
        "material_absolute_threshold_um": float(material_absolute_um),
        "material_relative_threshold_fraction": float(material_relative_fraction),
        "archive": str(archive_path.resolve()),
        "archive_sha256": sha256(archive_path),
        "technical_valid_tracks_with_at_least_3_observed_planes": eligible_before_width,
        "specimens": specimens,
        "overall": overall,
    }
    summary["primary_conclusion"] = _primary_conclusion(overall)
    return frame.sort_values(["specimen_id", "track_id"]), summary


def write_outputs(frame: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir = prepare_output_dir(output_dir)
    summary = dict(summary)
    csv_path = output_dir / "v571_width_z_repeatability_tracks.csv"
    json_path = output_dir / "v571_width_z_repeatability_summary.json"
    markdown_path = output_dir / "V5_7_1_WIDTH_Z_REPEATABILITY_QC.md"

    rows = []
    for specimen_id, values in summary["specimens"].items():
        fraction = values["material_adjacent_difference_track_fraction"]
        rows.append(
            "| {specimen} | {eligible} | {adjacent} | {cv:.3f} | {range_um:.3f} | {fraction} |".format(
                specimen=specimen_id,
                eligible=values["eligible_track_count"],
                adjacent=values["tracks_with_adjacent_width_observation"],
                cv=values["median_width_cv"],
                range_um=values["median_width_range_um"],
                fraction=(f"{fraction:.1%}" if fraction is not None else "NA"),
            )
        )
    report = f"""# Saturn v5.7.1 Width Z-Repeatability QC

## Primary conclusion

{summary['primary_conclusion']}

## Compact results

| Specimen | Eligible tracks | With adjacent plane | Median width CV | Median range (um) | Material adjacent difference |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Scope and formulas

- Scope: **repeatability QC**, not biological truth or group inference.
- Population: {summary['population']}.
- Per-track CV: sample standard deviation of observed-plane `body_width_um` divided by its mean.
- Per-track range: maximum minus minimum observed-plane `body_width_um`.
- Adjacent comparison: {summary['adjacent_plane_definition']}.
- Material QC flag: absolute difference >= {summary['material_absolute_threshold_um']:.3f} um **and** relative difference >= {summary['material_relative_threshold_fraction']:.0%} of representative width.
- The flag is descriptive only. It does not invalidate a track, remove unusual morphology, or establish physical diameter accuracy.
"""
    atomic_write_dataframe(csv_path, frame)
    atomic_write_text(markdown_path, report)
    summary["artifacts"] = [
        {"path": csv_path.name, "sha256": sha256(csv_path)},
        {"path": markdown_path.name, "sha256": sha256(markdown_path)},
    ]
    atomic_write_text(json_path, json.dumps(summary, indent=2))
    completion = {
        "schema_version": "1.0",
        "status": "complete",
        "generated_at_utc": summary["generated_at_utc"],
        "summary": json_path.name,
        "summary_sha256": sha256(json_path),
        "artifact_count": len(summary["artifacts"]),
        "qc_only": True,
        "biological_truth_status": "not_established",
    }
    atomic_write_text(
        output_dir / "COMPLETED.json", json.dumps(completion, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--material-absolute-um",
        type=float,
        default=DEFAULT_MATERIAL_ABSOLUTE_UM,
    )
    parser.add_argument(
        "--material-relative-fraction",
        type=float,
        default=DEFAULT_MATERIAL_RELATIVE_FRACTION,
    )
    args = parser.parse_args()
    frame, summary = analyze_archive(
        args.archive,
        material_absolute_um=args.material_absolute_um,
        material_relative_fraction=args.material_relative_fraction,
    )
    write_outputs(frame, summary, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
