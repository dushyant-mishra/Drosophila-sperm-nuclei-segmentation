import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_prepare_dataset():
    spec = importlib.util.spec_from_file_location(
        "unet25d_prepare_dataset_test",
        ROOT / "unet25d" / "prepare_dataset.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_unet25d_module(name):
    module_path = ROOT / "unet25d" / f"{name}.py"
    sys.path.insert(0, str(module_path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"unet25d_{name}_test", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_context_loader_supports_synthetic_indices_above_87(tmp_path):
    prepare = load_prepare_dataset()
    pattern = "Project001_Series002_z{z:04d}_ch00.tif"
    for z, value in ((109, 10), (110, 20), (111, 30)):
        tifffile.imwrite(tmp_path / pattern.format(z=z), np.full((8, 9), value, np.uint16))

    context = prepare.load_context(tmp_path, pattern, 110)

    assert context.shape == (3, 8, 9)
    assert float(context[0].mean()) < float(context[1].mean()) < float(context[2].mean())


def test_per_sample_roi_is_loaded_and_shape_checked(tmp_path):
    prepare = load_prepare_dataset()
    pattern = "Project001_Series002_z{z:04d}_ch00.npy"
    roi = np.zeros((7, 8), dtype=bool)
    roi[2:5, 3:6] = True
    np.save(tmp_path / pattern.format(z=120), roi)
    cfg = {"roi_mask_dir": str(tmp_path), "roi_mask_pattern": pattern}

    loaded = prepare.load_sample_roi(cfg, 120, roi.shape)

    np.testing.assert_array_equal(loaded, roi)


def test_training_dataset_can_balance_selected_synthetic_z_indices(tmp_path):
    train = load_unet25d_module("train_unet25d")
    for z in (110, 120, 300, 310):
        (tmp_path / f"Project001_Series002_z{z:04d}_ch00.npz").touch()

    dataset = train.SpermPatchDataset(
        tmp_path,
        patch_size=32,
        patches_per_image=4,
        augment=False,
        seed=7,
        repeat_z_indices=[110, 120],
        repeat_factor=2,
    )

    assert len(dataset.paths) == 6
    assert len(dataset) == 24
    repeated = [path.name for path in dataset.paths]
    assert repeated.count("Project001_Series002_z0110_ch00.npz") == 2
    assert repeated.count("Project001_Series002_z0300_ch00.npz") == 1


def test_photometric_augmentation_changes_images_not_masks(tmp_path):
    train = load_unet25d_module("train_unet25d")
    path = tmp_path / "Project001_Series002_z0110_ch00.npz"
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3:5, 3:5] = 1
    np.savez_compressed(
        path,
        image=np.full((3, 8, 8), 0.5, dtype=np.float32),
        mask=mask,
        supervision_mask=np.ones((8, 8), dtype=np.uint8),
    )
    dataset = train.SpermPatchDataset(
        tmp_path,
        patch_size=8,
        patches_per_image=1,
        augment=True,
        seed=11,
        photometric_augment_probability=1.0,
        photometric_gain_range=(0.5, 0.5),
        photometric_gamma_range=(1.0, 1.0),
        photometric_noise_std_max=0.0,
    )

    image, target, supervision = dataset[0]

    np.testing.assert_allclose(image.numpy(), 0.25)
    assert int(target.sum()) == int(mask.sum())
    assert int(supervision.sum()) == mask.size


def test_replay_builder_accepts_sreeni_manifest(tmp_path):
    builder = load_unet25d_module("build_kj_wt_replay_finetune_package")
    tifffile.imwrite(
        tmp_path / "Project001_Series002_z05_ch00.tif",
        np.zeros((12, 14), dtype=np.uint16),
    )
    manifest = {
        "classes": ["sperm_nucleus"],
        "images": [
            {
                "image": "images/Project001_Series002_z05_ch00.png",
                "instances": [
                    {
                        "class": "sperm_nucleus",
                        "segmentation": [2, 2, 8, 2, 8, 5, 2, 5],
                    }
                ],
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    coco = builder.load_replay_annotations(path, tmp_path)

    assert len(coco["images"]) == 1
    assert coco["images"][0]["width"] == 14
    assert coco["images"][0]["height"] == 12
    assert len(coco["annotations"]) == 1
    assert coco["annotations"][0]["segmentation"][0] == manifest["images"][0]["instances"][0]["segmentation"]


def test_instance_targets_preserve_touching_annotation_ids_and_cores():
    prepare = load_prepare_dataset()
    annotations = [
        {"id": 101, "segmentation": [[2, 3, 8, 3, 8, 7, 2, 7]]},
        {"id": 202, "segmentation": [[9, 3, 15, 3, 15, 7, 9, 7]]},
    ]

    labels, audit = prepare.rasterize_instances(20, 12, annotations)
    cores, missing = prepare.make_instance_core_labels(
        labels,
        {"instance_core_distance_fraction": 0.55, "instance_core_min_distance_px": 1.0},
    )

    assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
    assert sorted(np.unique(cores).tolist()) == [0, 1, 2]
    assert audit["source_instance_ids"] == {1: 101, 2: 202}
    assert audit["touching_pairs"] == [(1, 2)]
    assert not missing


def test_annotation_tolerant_boundary_keeps_thin_instance_core():
    prepare = load_prepare_dataset()
    annotations = [
        {"id": 1, "segmentation": [[3, 5, 18, 5, 18, 6, 3, 6]]},
    ]
    labels, _ = prepare.rasterize_instances(24, 12, annotations)
    boundary = prepare.make_boundary_ignore_mask(labels, radius=1)
    cores, missing = prepare.make_instance_core_labels(
        labels,
        {"instance_core_distance_fraction": 0.55, "instance_core_min_distance_px": 1.0},
    )
    weights = prepare.make_loss_weight_mask(
        np.ones(labels.shape, dtype=np.uint8),
        boundary,
        {"boundary_loss_weight": 0.1},
    )

    assert np.any(cores == 1)
    assert not missing
    assert np.all(weights[boundary] == np.float32(0.1))
    assert np.all(weights[~boundary] == np.float32(1.0))


def test_instance_target_hashes_are_deterministic():
    prepare = load_prepare_dataset()
    annotations = [
        {"id": 9, "segmentation": [[2, 2, 7, 2, 7, 6, 2, 6]]},
        {"id": 3, "segmentation": [[10, 3, 16, 3, 16, 8, 10, 8]]},
    ]

    first, first_audit = prepare.rasterize_instances(20, 12, annotations)
    second, second_audit = prepare.rasterize_instances(20, 12, list(reversed(annotations)))

    assert prepare.array_sha256(first) == prepare.array_sha256(second)
    assert first_audit == second_audit


def test_overlapping_instances_retain_separate_rle_records_and_overlap_depth():
    prepare = load_prepare_dataset()
    annotations = [
        {"id": 10, "segmentation": [[2, 2, 10, 2, 10, 8, 2, 8]]},
        {"id": 20, "segmentation": [[7, 3, 15, 3, 15, 9, 7, 9]]},
    ]

    labels, audit, overlap = prepare.rasterize_instances(
        20, 12, annotations, return_overlap_map=True
    )
    decoded = [
        prepare.decode_binary_mask_rle(record["rle"])
        for record in audit["instance_records"]
    ]

    assert len(decoded) == 2
    assert np.any(decoded[0] & decoded[1])
    assert int(overlap.max()) == 2
    np.testing.assert_array_equal(overlap, decoded[0].astype(int) + decoded[1])
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2]


def test_fractional_boundary_weight_is_applied_once_in_dice_loss():
    train = load_unet25d_module("train_unet25d")
    logits = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    target = torch.tensor([[[[1.0, 0.0]]]])
    weights = torch.tensor([[[[0.1, 1.0]]]])

    result = train.dice_loss(logits, target, weights)
    probability = torch.sigmoid(logits)
    expected = 1.0 - (
        2.0 * (probability * target * weights).sum() + 1e-6
    ) / (
        (probability * weights).sum()
        + (target * weights).sum()
        + 1e-6
    )

    assert float(result) == pytest.approx(float(expected))


def test_zero_weight_boundary_pixel_contributes_no_bce_loss():
    train = load_unet25d_module("train_unet25d")
    target = torch.tensor([[[[1.0, 0.0]]]])
    valid = torch.tensor([[[[1.0, 0.0]]]])
    first = torch.tensor([[[[0.0, -100.0]]]])
    second = torch.tensor([[[[0.0, 100.0]]]])

    first_loss = train.masked_bce_loss(first, target, valid)
    second_loss = train.masked_bce_loss(second, target, valid)

    assert float(first_loss) == pytest.approx(float(second_loss))


def test_dual_head_model_returns_foreground_and_core_logits():
    train = load_unet25d_module("train_unet25d")
    model = train.build_model(
        {
            "architecture": "dual_head_residual_attention_unet",
            "base_channels": 4,
            "deep_supervision": False,
        }
    )

    outputs = model(torch.zeros((1, 3, 32, 32), dtype=torch.float32))

    assert set(outputs) == {"foreground", "core"}
    assert outputs["foreground"].shape == (1, 1, 32, 32)
    assert outputs["core"].shape == (1, 1, 32, 32)


def test_dual_head_dataset_returns_distinct_touching_instance_cores(tmp_path):
    train = load_unet25d_module("train_unet25d")
    core_labels = np.zeros((16, 16), dtype=np.int32)
    core_labels[6:8, 4:6] = 1
    core_labels[6:8, 9:11] = 2
    mask = core_labels > 0
    np.savez_compressed(
        tmp_path / "Project001_Series002_z0001_ch00.npz",
        image=np.zeros((3, 16, 16), dtype=np.float32),
        mask=mask.astype(np.uint8),
        supervision_mask=np.ones((16, 16), dtype=np.float32),
        instance_core_labels=core_labels,
    )
    dataset = train.SpermPatchDataset(
        tmp_path,
        patch_size=16,
        patches_per_image=1,
        augment=False,
        seed=1,
        return_core_target=True,
    )

    _, _, _, core = dataset[0]

    assert int(core.sum()) == int(mask.sum())
    assert np.count_nonzero(core.numpy()[0, 6:8, 4:6]) == 4
    assert np.count_nonzero(core.numpy()[0, 6:8, 9:11]) == 4


def test_instance_evaluator_reports_a_merged_prediction():
    evaluator = load_unet25d_module("evaluate_annotation_tolerant_ab")
    reference = np.zeros((12, 16), dtype=np.int32)
    reference[3:8, 2:7] = 1
    reference[3:8, 7:12] = 2
    predicted = np.zeros_like(reference)
    predicted[3:8, 2:12] = 1

    result = evaluator.instance_metrics(
        reference,
        predicted,
        iou_threshold=0.20,
        touching_ids={1, 2},
    )

    assert result["reference_count"] == 2
    assert result["predicted_count"] == 1
    assert result["merged_prediction_count"] == 1
    assert result["merged_reference_count"] == 2
    assert result["merge_prediction_rate"] == pytest.approx(1.0)
    assert result["instance_true_positive"] == 1


def test_instance_evaluator_uses_overlapping_reference_rles():
    prepare = load_prepare_dataset()
    evaluator = load_unet25d_module("evaluate_annotation_tolerant_ab")
    first = np.zeros((12, 16), dtype=bool)
    second = np.zeros_like(first)
    first[3:8, 2:9] = True
    second[3:8, 7:14] = True
    records = [
        {"local_instance_id": 1, "rle": prepare.encode_binary_mask_rle(first)},
        {"local_instance_id": 2, "rle": prepare.encode_binary_mask_rle(second)},
    ]
    predicted = np.zeros(first.shape, dtype=np.int32)
    predicted[first] = 1
    predicted[second & ~first] = 2

    result = evaluator.instance_metrics_from_masks(
        records, predicted, iou_threshold=0.20, touching_ids={1, 2}
    )

    assert result["reference_count"] == 2
    assert result["instance_true_positive"] == 2


def test_core_marker_watershed_splits_connected_foreground():
    evaluator = load_unet25d_module("evaluate_annotation_tolerant_ab")
    foreground = np.zeros((20, 24), dtype=np.float32)
    foreground[6:14, 3:21] = 0.9
    core = np.zeros_like(foreground)
    core[8:12, 5:8] = 0.9
    core[8:12, 16:19] = 0.9

    labels = evaluator.marker_controlled_instances(
        foreground,
        core,
        foreground_threshold=0.5,
        core_threshold=0.5,
        roi=np.ones(foreground.shape, dtype=bool),
        minimum_area=3,
    )

    assert int(labels.max()) == 2
    assert labels[10, 6] != labels[10, 17]

    labels, foreground_mask, core_mask, markers = (
        evaluator.marker_controlled_instances(
            foreground,
            core,
            foreground_threshold=0.5,
            core_threshold=0.5,
            roi=np.ones(foreground.shape, dtype=bool),
            minimum_area=3,
            return_diagnostics=True,
        )
    )
    assert int(labels.max()) == 2
    assert int(markers.max()) == 2
    assert np.all(core_mask <= foreground_mask)


def test_partial_label_audit_separates_unknown_predictions():
    evaluator = load_unet25d_module("evaluate_annotation_tolerant_ab")
    labels = np.zeros((10, 12), dtype=np.int32)
    labels[2:5, 1:4] = 1
    labels[6:9, 8:11] = 2
    target = np.zeros(labels.shape, dtype=bool)
    target[2:5, 1:4] = True
    supervision = np.ones(labels.shape, dtype=bool)
    supervision[5:, 7:] = False

    audit = evaluator.partial_label_audit(
        labels, target, supervision, np.ones(labels.shape, dtype=bool)
    )

    assert audit["prediction_count_predominantly_ignored"] == 1
    assert audit["unmatched_prediction_count_predominantly_ignored"] == 1
    assert audit["unmatched_prediction_count_supervised"] == 0


def test_model_selection_table_is_review_only():
    evaluator = load_unet25d_module("evaluate_annotation_tolerant_ab")
    pixel_row = {
        "model": "model_b:epoch_003",
        "threshold": 0.3,
        "instance_method": "connected_components",
        "pixel_precision": 0.8,
        "pixel_recall": 0.9,
        "pixel_dice": 0.85,
        "pixel_iou": 0.74,
        "predicted_area_over_annotated_area": 1.05,
        "boundary_f1_tolerance_1px": 0.8,
        "mean_symmetric_contour_distance_px": 0.5,
    }
    instance_row = {
        "model": "model_b:epoch_003",
        "threshold": 0.3,
        "instance_method": "connected_components",
        "instance_precision": 0.8,
        "partial_adjusted_instance_precision": 0.82,
        "instance_recall": 0.9,
        "instance_f1": 0.85,
        "count_error": 1,
        "partial_adjusted_count_error": 0,
        "partial_unknown_prediction_count": 1,
        "merged_prediction_count": 2,
        "merged_reference_count": 3,
        "merge_prediction_rate": 0.2,
        "split_reference_count": 1,
        "split_prediction_count": 2,
        "split_reference_rate": 0.1,
        "missed_reference_count": 3,
        "duplicate_prediction_count": 4,
        "touching_instance_recall": 0.75,
    }

    table = evaluator.build_model_selection_table([pixel_row], [instance_row])

    assert len(table) == 1
    assert table[0]["selection_status"] == "candidate_for_visual_review_only"
    assert table[0]["instance_recall"] == pytest.approx(0.9)
