import argparse
import json
from datetime import datetime
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def normalize_class_name(name):
    name = str(name).strip()
    if name.lower().replace(" ", "_") in {"sperm_nuclei", "sperm_nucleus"}:
        return "sperm_nucleus"
    return name or "sperm_nucleus"


def convert_coco_to_iap(coco_path, images_dir, output_path, class_name="sperm_nucleus"):
    coco_path = Path(coco_path)
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    coco = load_json(coco_path)

    class_name = normalize_class_name(class_name)
    images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
    anns_by_image = {image_id: [] for image_id in images_by_id}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    iap_images = []
    image_paths = {}
    for image_id in sorted(images_by_id):
        image = images_by_id[image_id]
        file_name = image["file_name"]
        image_path = (images_dir / file_name).resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        annotations = []
        for number, ann in enumerate(anns_by_image.get(image_id, []), start=1):
            segmentation = ann.get("segmentation", [])
            if not isinstance(segmentation, list):
                continue
            for poly in segmentation:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                annotations.append(
                    {
                        "segmentation": [float(v) if isinstance(v, float) else int(v) for v in poly],
                        "category_id": 1,
                        "category_name": class_name,
                        "number": number,
                    }
                )

        iap_images.append(
            {
                "file_name": file_name,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "is_multi_slice": True,
                "slices": [
                    {
                        "name": Path(file_name).stem,
                        "annotations": {class_name: annotations},
                    }
                ],
                "dimensions": ["H", "W"],
                "shape": [int(image["height"]), int(image["width"])],
            }
        )
        image_paths[file_name] = str(image_path).replace("\\", "/")

    now = datetime.now().isoformat()
    project = {
        "classes": [{"name": class_name, "color": "#1f77b4"}],
        "images": iap_images,
        "image_paths": image_paths,
        "notes": f"Merged from {coco_path.name} for continued annotation.",
        "creation_date": now,
        "last_modified": now,
        "dino_config": {
            "phrases": {class_name: [class_name.replace("_", " ")]},
            "thresholds": {class_name: {"box": 0.25, "txt": 0.25, "nms": 0.5}},
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, project)
    return {
        "output_path": str(output_path),
        "image_count": len(iap_images),
        "annotation_count": sum(
            len(image["slices"][0]["annotations"][class_name]) for image in iap_images
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert merged COCO back to Sreeni .iap JSON.")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-name", default="sperm_nucleus")
    args = parser.parse_args()

    summary = convert_coco_to_iap(args.coco, args.images_dir, args.output, args.class_name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
