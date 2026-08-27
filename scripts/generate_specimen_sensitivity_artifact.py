"""Create a compact specimen-level sensitivity/QC artifact.

This is an audit-facing sidecar. It does not alter the biological PDF or run
between-group inference.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


OUTPUT_FIELDS = [
    "sample_id", "group", "status", "analysis_included", "technical_valid_count",
    "technical_failure_count", "all_reconstructed_count",
    "technical_exclusion_fraction", "width_available_fraction", "width_missing",
    "width_available_estimated_count", "width_missing_estimated_count",
    "roi_area_um2", "sampled_depth_um", "sampled_roi_volume_um3",
    "primary_count_per_1000_um2", "all_reconstructed_count_per_1000_um2",
    "primary_count_per_100000_um3", "all_reconstructed_count_per_100000_um3",
    "area_denominator_valid", "volume_denominator_valid", "normalization_valid",
    "normalization_warning",
]


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def build_rows(records: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    """Retain every specimen and mark, rather than silently drop, exclusions."""
    rows = []
    for record in records:
        area = _number(record, "roi_area_um2")
        volume = _number(record, "sampled_roi_volume_um3")
        width_fraction = _number(record, "body_width_available_fraction")
        included = record.get("status", "").lower() == "complete"
        technical_valid = int(
            _number(record, "qc_analysis_population_unet_track_count") or 0
        )
        technical_failure = int(
            _number(record, "qc_technical_failure_track_count") or 0
        )
        all_reconstructed = technical_valid + technical_failure
        width_available = (
            int(round(technical_valid * width_fraction))
            if width_fraction is not None
            else None
        )
        rows.append({
            "sample_id": record.get("sample_id", ""),
            "group": record.get("group", ""),
            "status": record.get("status", ""),
            "analysis_included": included,
            "technical_valid_count": technical_valid,
            "technical_failure_count": technical_failure,
            "all_reconstructed_count": all_reconstructed,
            "technical_exclusion_fraction": (
                technical_failure / all_reconstructed if all_reconstructed else None
            ),
            "width_available_fraction": width_fraction,
            "width_missing": width_fraction is None or width_fraction < 1.0,
            "width_available_estimated_count": width_available,
            "width_missing_estimated_count": (
                technical_valid - width_available
                if width_available is not None
                else None
            ),
            "roi_area_um2": area,
            "sampled_depth_um": _number(record, "sampled_depth_um"),
            "sampled_roi_volume_um3": volume,
            "primary_count_per_1000_um2": (
                technical_valid / area * 1000.0 if area and area > 0 else None
            ),
            "all_reconstructed_count_per_1000_um2": (
                all_reconstructed / area * 1000.0 if area and area > 0 else None
            ),
            "primary_count_per_100000_um3": (
                technical_valid / volume * 100000.0
                if volume and volume > 0
                else None
            ),
            "all_reconstructed_count_per_100000_um3": (
                all_reconstructed / volume * 100000.0
                if volume and volume > 0
                else None
            ),
            "area_denominator_valid": area is not None and area > 0,
            "volume_denominator_valid": volume is not None and volume > 0,
            "normalization_valid": record.get("normalization_valid", "").lower() == "true",
            "normalization_warning": record.get("normalization_warning", ""),
        })
    return rows


def write_artifact(input_csv: Path, output_csv: Path, output_json: Path) -> None:
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = build_rows(csv.DictReader(handle))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "1.0",
        "analysis_unit": "biological_specimen",
        "inference_performed": False,
        "inference_note": "Sensitivity rows are descriptive; no group inference is computed.",
        "source": str(input_csv),
        "specimen_count": len(rows),
        "included_specimen_count": sum(row["analysis_included"] for row in rows),
        "excluded_or_missing_specimen_count": sum(
            not row["analysis_included"] for row in rows
        ),
        "formulas": {
            "technical_exclusion_fraction": (
                "technical_failure_count / all_reconstructed_count"
            ),
            "primary_count_per_1000_um2": (
                "technical_valid_count / roi_area_um2 * 1000"
            ),
            "all_reconstructed_count_per_1000_um2": (
                "all_reconstructed_count / roi_area_um2 * 1000"
            ),
            "primary_count_per_100000_um3": (
                "technical_valid_count / sampled_roi_volume_um3 * 100000"
            ),
            "all_reconstructed_count_per_100000_um3": (
                "all_reconstructed_count / sampled_roi_volume_um3 * 100000"
            ),
        },
        "scenario_interpretation": {
            "primary": "Technical-valid reconstructed nuclei only.",
            "include_technical_failures_upper_bound": (
                "Descriptive upper-bound sensitivity; not a biological result."
            ),
            "width_missingness": (
                "Reports how much of the technical-valid population lacks the "
                "new apparent body-width measurement."
            ),
            "normalization": (
                "Area and sampled-volume denominators are evaluated separately; "
                "sampled ROI volume is not anatomical organ volume."
            ),
        },
        "rows": rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    write_artifact(args.input_csv, args.output_csv, args.output_json)


if __name__ == "__main__":
    main()
