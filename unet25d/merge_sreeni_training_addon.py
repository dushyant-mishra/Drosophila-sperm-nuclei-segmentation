import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    if not rows and not fieldnames:
        raise ValueError(f"Cannot write an empty CSV without columns: {path}")
    columns = fieldnames or list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_unique(source, destination):
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise FileExistsError(f"Conflicting destination file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def locate_project(package):
    projects = sorted((package / "annotation_images").glob("*.iap"))
    if len(projects) != 1:
        raise ValueError(
            f"Expected exactly one Sreeni project in {package / 'annotation_images'}, "
            f"found {projects}"
        )
    return projects[0]


def merge_project(target_project, addon, addon_rows):
    project = json.loads(target_project.read_text(encoding="utf-8"))
    existing = {image["file_name"] for image in project["images"]}
    image_dir = target_project.parent / "images"

    added = []
    for row in addon_rows:
        file_name = row["annotation_file"]
        if file_name in existing:
            raise ValueError(f"Image is already present in Sreeni project: {file_name}")
        source = addon / "annotation_images" / file_name
        destination = image_dir / file_name
        copy_unique(source, destination)
        with Image.open(destination) as image:
            width, height = image.size
        project["images"].append(
            {
                "file_name": file_name,
                "width": width,
                "height": height,
                "is_multi_slice": False,
                "annotations": {"sperm_nucleus": []},
            }
        )
        project["image_paths"][file_name] = str(destination.resolve()).replace(
            "\\", "/"
        )
        added.append(file_name)

    project["notes"] = (
        str(project.get("notes", "")).rstrip()
        + "\nAdded overlapping-tissue positive examples. Annotate every visible "
        "sperm nucleus inside the ROI; leave broad overlapping tissue unlabeled."
    ).strip()
    project["last_modified"] = datetime.now().isoformat()
    target_project.write_text(
        json.dumps(project, indent=2) + "\n",
        encoding="utf-8",
    )
    return added, len(project["images"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge a contiguous positive-image add-on into an existing Sreeni "
            "training package without losing its annotations."
        )
    )
    parser.add_argument("--target-package", required=True, type=Path)
    parser.add_argument("--addon-package", required=True, type=Path)
    args = parser.parse_args()

    target = args.target_package.resolve()
    addon = args.addon_package.resolve()
    target_project = locate_project(target)
    addon_rows = read_csv(addon / "split_manifest.csv")
    if not addon_rows:
        raise ValueError("The add-on has no images")
    if any(row["split"] != "train" for row in addon_rows):
        raise ValueError("All overlapping-tissue add-on images must be training-only")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = target / ".project_backups"
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / f"{target_project.stem}_before_overlap_{timestamp}.iap"
    shutil.copy2(target_project, backup)

    for directory, pattern in (
        ("raw_tiffs", "*.tif*"),
        ("roi_masks", "*.npy"),
        ("roi_guides", "*.png"),
    ):
        for source in (addon / directory).glob(pattern):
            copy_unique(source, target / directory / source.name)

    added, total_images = merge_project(target_project, addon, addon_rows)

    split_path = target / "split_manifest.csv"
    split_rows = read_csv(split_path)
    existing_z = {int(row["synthetic_z"]) for row in split_rows}
    for row in addon_rows:
        if int(row["synthetic_z"]) in existing_z:
            raise ValueError(f"Duplicate synthetic Z index: {row['synthetic_z']}")
    write_csv(split_path, split_rows + addon_rows)

    source_mapping = read_csv(addon / "source_mapping_private.csv")
    target_mapping = target / "overlap_tissue_source_mapping_private.csv"
    write_csv(target_mapping, source_mapping)

    result = {
        "target_project": str(target_project),
        "backup_project": str(backup),
        "added_images": added,
        "added_count": len(added),
        "total_project_images": total_images,
        "annotation_status": "new images intentionally blank; annotate before COCO export",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
