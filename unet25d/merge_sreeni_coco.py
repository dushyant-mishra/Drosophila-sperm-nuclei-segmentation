import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_image_path(images_dir, file_name):
    image_path = Path(images_dir) / file_name
    if image_path.exists():
        return image_path

    alt = Path(images_dir) / Path(file_name).name
    if alt.exists():
        return alt

    matches = list(Path(images_dir).rglob(Path(file_name).name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous image path for {file_name!r}: {matches}")
    raise FileNotFoundError(f"Could not find image {file_name!r} under {images_dir}")


def normalize_category_name(name):
    name = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if name in {"sperm_nuclei", "sperm_nucleus", "sperm_nucleus_mask"}:
        return "sperm_nucleus"
    if name == "background":
        return "background"
    return name


def polygon_bbox(poly):
    xs = [float(poly[i]) for i in range(0, len(poly), 2)]
    ys = [float(poly[i]) for i in range(1, len(poly), 2)]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def polygon_area(poly):
    if len(poly) < 6:
        return 0.0
    xs = [float(poly[i]) for i in range(0, len(poly), 2)]
    ys = [float(poly[i]) for i in range(1, len(poly), 2)]
    area = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) * 0.5


def coco_from_iap(iap_path):
    project = load_json(iap_path)
    images = []
    annotations = []
    ann_id = 1
    image_id = 1

    for image in project.get("images", []):
        file_name = image["file_name"]
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "height": int(image["height"]),
                "width": int(image["width"]),
            }
        )

        for slice_info in image.get("slices", []):
            for class_name, class_annotations in slice_info.get("annotations", {}).items():
                if not is_foreground_category(class_name):
                    continue
                for ann in class_annotations:
                    poly = ann.get("segmentation", [])
                    if len(poly) < 6:
                        continue
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "area": polygon_area(poly),
                            "iscrowd": 0,
                            "segmentation": [poly],
                            "bbox": polygon_bbox(poly),
                            "source_number": ann.get("number"),
                            "source_category_name": ann.get("category_name", class_name),
                        }
                    )
                    ann_id += 1
        image_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "sperm_nucleus", "supercategory": "cellular_structure"}
        ],
    }


def load_annotation_document(path):
    path = Path(path)
    if path.suffix.lower() == ".iap":
        return coco_from_iap(path)
    return load_json(path)


def is_foreground_category(category_name):
    return normalize_category_name(category_name) != "background"


def annotation_signature(ann):
    segmentation = ann.get("segmentation", [])
    if not isinstance(segmentation, list):
        return None

    rounded = []
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        rounded.append(tuple(round(float(v), 1) for v in poly))
    if not rounded:
        return None
    rounded.sort()
    return tuple(rounded)


def copy_image_once(src_path, out_images_dir, used_names):
    base_name = src_path.name
    stem = src_path.stem
    suffix = src_path.suffix
    out_name = base_name
    counter = 2
    while out_name.lower() in used_names:
        out_name = f"{stem}__dup{counter}{suffix}"
        counter += 1
    used_names.add(out_name.lower())
    dst_path = out_images_dir / out_name
    shutil.copy2(src_path, dst_path)
    return out_name, dst_path


def load_sources(source_specs):
    sources = []
    for spec in source_specs:
        parts = spec.split("|")
        if len(parts) != 3:
            raise ValueError(
                "Each --source must be formatted as name|coco_json|images_dir"
            )
        name, annotation_path, images_dir = parts
        annotation_path = Path(annotation_path)
        images_dir = Path(images_dir)
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
        if not images_dir.exists():
            raise FileNotFoundError(images_dir)
        sources.append({"name": name, "annotation_path": annotation_path, "images_dir": images_dir})
    return sources


def merge_sources(sources, output_dir, drop_empty_images=False):
    output_dir = Path(output_dir)
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    merged_images = []
    merged_annotations = []
    manifest_rows = []
    duplicate_rows = []

    hash_to_image = {}
    used_names = set()
    next_image_id = 1
    next_annotation_id = 1

    category_by_id = {1: {"id": 1, "name": "sperm_nucleus", "supercategory": "cellular_structure"}}

    for source in sources:
        coco = load_annotation_document(source["annotation_path"])
        source_name = source["name"]
        categories = {
            int(cat["id"]): normalize_category_name(cat.get("name", ""))
            for cat in coco.get("categories", [])
        }
        images_by_id = {int(im["id"]): im for im in coco.get("images", [])}
        anns_by_image = {image_id: [] for image_id in images_by_id}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        for old_image_id, image in images_by_id.items():
            src_path = find_image_path(source["images_dir"], image["file_name"])
            digest = file_hash(src_path)
            source_ann_count = len(anns_by_image.get(old_image_id, []))

            if digest in hash_to_image:
                merged_image = hash_to_image[digest]
                duplicate_rows.append(
                    {
                        "source": source_name,
                        "source_file": str(src_path),
                        "matched_output_file": merged_image["file_name"],
                        "reason": "same_sha256",
                        "source_annotations": source_ann_count,
                    }
                )
            else:
                out_name, dst_path = copy_image_once(src_path, out_images_dir, used_names)
                merged_image = {
                    "id": next_image_id,
                    "file_name": out_name,
                    "height": int(image["height"]),
                    "width": int(image["width"]),
                    "source_files": [str(src_path)],
                    "source_projects": [source_name],
                    "sha256": digest,
                }
                hash_to_image[digest] = merged_image
                merged_images.append(merged_image)
                next_image_id += 1
                manifest_rows.append(
                    {
                        "output_file": out_name,
                        "sha256": digest,
                        "width": int(image["width"]),
                        "height": int(image["height"]),
                        "source": source_name,
                        "source_file": str(src_path),
                        "source_annotations": source_ann_count,
                    }
                )

            existing_signatures = {
                annotation_signature(ann)
                for ann in merged_annotations
                if ann["image_id"] == merged_image["id"]
            }
            for ann in anns_by_image.get(old_image_id, []):
                cat_name = categories.get(int(ann.get("category_id", 1)), "sperm_nucleus")
                if not is_foreground_category(cat_name):
                    continue

                signature = annotation_signature(ann)
                if signature is None or signature in existing_signatures:
                    if signature is not None:
                        duplicate_rows.append(
                            {
                                "source": source_name,
                                "source_file": str(src_path),
                                "matched_output_file": merged_image["file_name"],
                                "reason": "duplicate_annotation_polygon",
                                "source_annotations": 1,
                            }
                        )
                    continue

                new_ann = dict(ann)
                new_ann["id"] = next_annotation_id
                new_ann["image_id"] = merged_image["id"]
                new_ann["category_id"] = 1
                new_ann["source_project"] = source_name
                new_ann["source_annotation_id"] = ann.get("id")
                merged_annotations.append(new_ann)
                existing_signatures.add(signature)
                next_annotation_id += 1

            if source_name not in merged_image["source_projects"]:
                merged_image["source_projects"].append(source_name)
                merged_image["source_files"].append(str(src_path))

    dropped_empty_images = 0
    if drop_empty_images:
        annotation_counts = {}
        for ann in merged_annotations:
            annotation_counts[ann["image_id"]] = annotation_counts.get(ann["image_id"], 0) + 1
        keep_ids = {image_id for image_id, count in annotation_counts.items() if count > 0}
        dropped_empty_images = len([im for im in merged_images if im["id"] not in keep_ids])
        merged_images = [im for im in merged_images if im["id"] in keep_ids]

    merged = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": list(category_by_id.values()),
    }

    write_json(output_dir / "_annotations.coco.json", merged)

    with open(output_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "output_file",
            "sha256",
            "width",
            "height",
            "source",
            "source_file",
            "source_annotations",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(output_dir / "duplicate_report.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source",
            "source_file",
            "matched_output_file",
            "reason",
            "source_annotations",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(duplicate_rows)

    return {
        "output_dir": str(output_dir),
        "image_count": len(merged_images),
        "annotation_count": len(merged_annotations),
        "duplicates_or_skipped": len(duplicate_rows),
        "dropped_empty_images": dropped_empty_images,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge Sreeni/COCO exports by image hash and union foreground annotations."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source formatted as name|coco_json|images_dir. Repeat for each project/split.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Exclude images with no foreground annotations from the merged COCO JSON.",
    )
    args = parser.parse_args()

    summary = merge_sources(load_sources(args.source), args.output, drop_empty_images=args.drop_empty_images)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
