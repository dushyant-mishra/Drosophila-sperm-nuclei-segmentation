import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def discover_stack(directory, series):
    pattern = re.compile(
        rf"^Project001_{re.escape(series)}_z(\d+)_ch00\.tif{{1,2}}$",
        re.IGNORECASE,
    )
    by_z = {}
    for path in Path(directory).iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            by_z[int(match.group(1))] = path
    if not by_z:
        raise FileNotFoundError(f"No {series} TIFF stack found in {directory}")
    return by_z


def read_2d(path):
    image = np.asarray(tifffile.imread(path))
    if image.ndim > 2:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image: {path}")
    return image


def display_image(image, roi):
    values = image[roi].astype(np.float32)
    lo, hi = np.percentile(values, [1.0, 99.5])
    normalized = np.clip(
        (image.astype(np.float32) - lo) / max(float(hi - lo), 1.0),
        0.0,
        1.0,
    )
    return np.round(normalized * 255).astype(np.uint8)


def roi_guide(image, roi):
    edge = roi & ~(
        np.roll(roi, 1, axis=0)
        & np.roll(roi, -1, axis=0)
        & np.roll(roi, 1, axis=1)
        & np.roll(roi, -1, axis=1)
    )
    rgb = np.repeat(image[..., None], 3, axis=2)
    rgb[edge] = (0, 255, 80)
    return rgb


def blank_project(images_dir, records):
    images = []
    paths = {}
    for record in records:
        file_name = record["annotation_file"]
        path = (images_dir / file_name).resolve()
        with Image.open(path) as image:
            width, height = image.size
        images.append(
            {
                "file_name": file_name,
                "width": width,
                "height": height,
                "is_multi_slice": True,
                "slices": [
                    {
                        "name": Path(file_name).stem,
                        "annotations": {"sperm_nucleus": []},
                    }
                ],
                "dimensions": ["H", "W"],
                "shape": [height, width],
            }
        )
        paths[file_name] = str(path).replace("\\", "/")
    now = datetime.now().isoformat()
    return {
        "classes": [{"name": "sperm_nucleus", "color": "#1f77b4"}],
        "images": images,
        "image_paths": paths,
        "notes": (
            "Contiguous overlapping-tissue add-on. Annotate every visible sperm "
            "nucleus inside the ROI; do not treat the image as empty background."
        ),
        "creation_date": now,
        "last_modified": now,
    }


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare contiguous stack planes as a Sreeni fine-tuning add-on."
    )
    parser.add_argument("--stack-dir", required=True, type=Path)
    parser.add_argument("--series", required=True)
    parser.add_argument("--roi", required=True, type=Path)
    parser.add_argument("--center-z", required=True, nargs="+", type=int)
    parser.add_argument("--synthetic-start", type=int, default=500)
    parser.add_argument("--synthetic-spacing", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    annotation_dir = output / "annotation_images"
    guide_dir = output / "roi_guides"
    raw_dir = output / "raw_tiffs"
    roi_dir = output / "roi_masks"
    annotations_dir = output / "annotations"
    for directory in (
        annotation_dir,
        guide_dir,
        raw_dir,
        roi_dir,
        annotations_dir,
    ):
        directory.mkdir()

    by_z = discover_stack(args.stack_dir, args.series)
    roi = np.load(args.roi).astype(bool)
    first = read_2d(next(iter(by_z.values())))
    if roi.shape != first.shape:
        raise ValueError(f"ROI {roi.shape} does not match stack {first.shape}")

    records = []
    source_rows = []
    for index, source_z in enumerate(args.center_z):
        if source_z not in by_z:
            raise FileNotFoundError(f"Missing source z{source_z:02d}")
        synthetic_z = args.synthetic_start + index * args.synthetic_spacing
        name = f"Project001_Series002_z{synthetic_z:04d}_ch00.png"
        display = display_image(read_2d(by_z[source_z]), roi)
        Image.fromarray(display).save(annotation_dir / name)
        Image.fromarray(display).save(output / name)
        Image.fromarray(roi_guide(display, roi)).save(
            guide_dir / name.replace(".png", "_roi_guide.png")
        )

        context_sources = []
        for offset in (-1, 0, 1):
            requested = source_z + offset
            context_z = (
                requested
                if requested in by_z
                else min(by_z, key=lambda z: (abs(z - requested), z))
            )
            context_sources.append(context_z)
            shutil.copy2(
                by_z[context_z],
                raw_dir
                / f"Project001_Series002_z{synthetic_z + offset:04d}_ch00.tif",
            )
        np.save(
            roi_dir / f"Project001_Series002_z{synthetic_z:04d}_ch00.npy",
            roi,
        )
        records.append(
            {
                "annotation_file": name,
                "synthetic_z": synthetic_z,
                "split": "train",
            }
        )
        source_rows.append(
            {
                "source_directory": str(args.stack_dir.resolve()),
                "source_series": args.series,
                "source_center_z": source_z,
                "synthetic_center_z": synthetic_z,
                "synthetic_file": name,
                "context_source_z": "|".join(str(z) for z in context_sources),
                "roi_path": str(args.roi.resolve()),
            }
        )

    project = blank_project(output, records)
    project_path = output / "sreeni_overlap_tissue_addon_blank.iap"
    project_path.write_text(json.dumps(project, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "split_manifest.csv", records)
    write_csv(output / "source_mapping_private.csv", source_rows)
    instructions = """# Overlapping-Tissue Annotation Add-On

1. Open `sreeni_overlap_tissue_addon_blank.iap` in Sreeni.
2. Use `roi_guides/` to check the ROI boundary.
3. Annotate every visible sperm nucleus inside the ROI on all three images.
4. Include faint and partial nuclei; keep touching nuclei separate.
5. Do not annotate broad overlapping tissue as a sperm nucleus.
6. Export COCO to `annotations/_annotations.coco.json`.

These images are not blank negatives. They contain nucleus-like structures and
must not enter training until their positive nuclei have been annotated.
"""
    (output / "ANNOTATION_INSTRUCTIONS.md").write_text(
        instructions,
        encoding="utf-8",
    )
    archive = shutil.make_archive(
        str(output),
        "zip",
        root_dir=output.parent,
        base_dir=output.name,
    )
    print(
        json.dumps(
            {
                "images": len(records),
                "source_z": args.center_z,
                "project": str(project_path),
                "archive": archive,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
