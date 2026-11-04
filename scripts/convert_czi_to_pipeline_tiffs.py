"""Convert Zeiss CZI stacks into the per-slice TIFFs the Saturn pipeline reads.

The pipeline ingests one TIFF per optical slice, named so that its source parser
can recover the stack identity, the Z index and the channel. Zeiss CZI packs the
whole stack plus its metadata into a single container, so nothing in the
pipeline can open one directly.

Naming. Slices are written as ``<specimen>_z##_ch00.tif``, which
``_study_parse_source_name`` recognises as its ``explicit_z`` form. That keeps
the collaborator's own specimen names visible in the filenames rather than
renumbering them into Leica ``Series###`` form, which would make the stacks
harder to tell apart when drawing regions.

Calibration. The CZI XML is copied beside each stack unchanged, and the values
the pipeline actually needs are additionally written to a small JSON file and
gathered into a study manifest template. The Leica XML reader cannot parse Zeiss
metadata, so calibration has to be supplied through the manifest rather than
discovered from the files.

Usage:
    python scripts/convert_czi_to_pipeline_tiffs.py --input "<folder of .czi>"
"""

import argparse
import json
import re
from pathlib import Path

import czifile
import numpy as np
import tifffile


def _scalar(xml, tag, default=None):
    found = re.findall(rf"<{tag}>(.*?)</{tag}>", xml)
    return found[0] if found else default


def read_calibration(xml):
    """Pull the acquisition values the pipeline needs out of the CZI metadata."""

    def microns(tag):
        raw = _scalar(xml, tag)
        return round(float(raw) * 1e6, 6) if raw else None

    return {
        "xy_um_per_pixel": microns("ScalingX"),
        "xy_um_per_pixel_y": microns("ScalingY"),
        "z_um_per_slice": microns("ScalingZ"),
        "size_x": int(_scalar(xml, "SizeX", 0) or 0),
        "size_y": int(_scalar(xml, "SizeY", 0) or 0),
        "size_z": int(_scalar(xml, "SizeZ", 0) or 0),
        "bit_depth": int(_scalar(xml, "ComponentBitCount", 0) or 0),
        "pixel_type": _scalar(xml, "PixelType"),
        "objective_na": _scalar(xml, "LensNA"),
        "objective_magnification": _scalar(xml, "NominalMagnification"),
        "immersion": _scalar(xml, "Immersion"),
        "pinhole_airy": _scalar(xml, "PinholeSizeAiry"),
        "excitation_nm": _scalar(xml, "ExcitationWavelength"),
    }


def convert(czi_path, output_root):
    """Write one CZI out as a slice stack plus its metadata, and describe it."""
    stem = czi_path.stem
    destination = output_root / stem
    destination.mkdir(parents=True, exist_ok=True)

    with czifile.CziFile(str(czi_path)) as handle:
        xml = handle.metadata()
        stack = np.squeeze(handle.asarray())

    if stack.ndim == 2:  # a single plane still gets the same layout
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(
            f"{czi_path.name}: expected a Z stack, got shape {stack.shape}. "
            "A multi-channel or tiled CZI needs its own handling."
        )

    calibration = read_calibration(xml)
    if calibration["size_z"] and calibration["size_z"] != stack.shape[0]:
        raise ValueError(
            f"{czi_path.name}: metadata declares {calibration['size_z']} slices "
            f"but {stack.shape[0]} were decoded"
        )

    written = []
    for index in range(stack.shape[0]):
        out = destination / f"{stem}_z{index:02d}_ch00.tif"
        # No compression: the stacks are small and every reader opens them.
        tifffile.imwrite(str(out), stack[index], photometric="minisblack")
        written.append(out)

    (destination / f"{stem}_czi_metadata.xml").write_text(xml, encoding="utf-8")
    calibration_record = dict(calibration)
    calibration_record.update(
        {
            "specimen": stem,
            "source_czi": str(czi_path),
            "slices_written": len(written),
            "file_pattern": f"{stem}_z[0-9]*_ch00.tif",
            "note": (
                "Zeiss CZI converted for the Saturn pipeline. The Leica XML "
                "reader cannot parse this metadata, so xy_um_per_pixel and "
                "z_um_per_slice must be supplied through the study manifest."
            ),
        }
    )
    (destination / f"{stem}_calibration.json").write_text(
        json.dumps(calibration_record, indent=2), encoding="utf-8"
    )
    return calibration_record, destination


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="folder holding the .czi files")
    parser.add_argument("--output", default="", help="defaults to the input folder")
    args = parser.parse_args(argv)

    source = Path(args.input).resolve()
    output_root = Path(args.output).resolve() if args.output else source
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for czi in sorted(source.glob("*.czi")):
        record, destination = convert(czi, output_root)
        records.append(record)
        print(
            f"  {czi.name:34s} -> {destination.name}/  "
            f"{record['slices_written']:3d} slices  "
            f"{record['xy_um_per_pixel']} um/px  {record['z_um_per_slice']} um/z"
        )

    if records:
        import csv

        manifest = output_root / "study_manifest_template.csv"
        columns = [
            "include", "sample_id", "group", "group_role", "input_dir",
            "roi_path", "file_pattern", "slice_count",
            "xy_um_per_pixel", "z_um_per_slice", "calibration_metadata_path",
        ]
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in records:
                specimen = record["specimen"]
                writer.writerow(
                    {
                        "include": "TRUE",
                        "sample_id": specimen,
                        # Deliberately blank: group identity and direction are the
                        # owner's to declare, never inferred from a filename.
                        "group": "",
                        "group_role": "",
                        "input_dir": str(output_root / specimen),
                        "roi_path": "",
                        "file_pattern": record["file_pattern"],
                        "slice_count": record["slices_written"],
                        "xy_um_per_pixel": record["xy_um_per_pixel"],
                        "z_um_per_slice": record["z_um_per_slice"],
                        "calibration_metadata_path": str(
                            output_root / specimen / f"{specimen}_calibration.json"
                        ),
                    }
                )
        print(f"\nmanifest template: {manifest}")
    print(f"{len(records)} stacks converted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
