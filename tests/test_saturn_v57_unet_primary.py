import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage import measure


ROOT = Path(__file__).resolve().parents[1]


def load_saturn():
    spec = importlib.util.spec_from_file_location(
        "saturn_v57_unet_primary_test",
        ROOT / "sperm_segmentation_saturnv5.7.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "saturn_v57_unet_primary_runner_test",
        ROOT / "scratch" / "run_v57_unet_primary_smoke.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def primary_cfg(saturn, **updates):
    cfg = saturn.CONFIG.copy()
    cfg.update({
        "SEGMENTATION_ENGINE": "unet_primary",
        "UNET_CANDIDATE_THRESHOLD": 0.05,
        "UNET_SEED_THRESHOLD": 0.30,
        "UNET_PRIMARY_MIN_COMPONENT_PX": 1,
        "UNET_PRIMARY_CLASSICAL_ADDITIONS_ENABLE": False,
        "UM_PER_PX_XY": 1.0,
    })
    cfg.update(updates)
    return saturn.cfg_with_resolved_pixels(cfg)


def test_load_analysis_profile_binds_checkpoint_and_full_settings(tmp_path):
    saturn = load_saturn()
    checkpoint = tmp_path / "epoch_003.pt"
    checkpoint.write_bytes(b"test checkpoint")
    profile = tmp_path / "combined.json"
    profile.write_text(
        """
        {
          "SEGMENTATION_ENGINE": "unet_primary",
          "TRACKING_BACKEND": "global_assignment",
          "UNET_MODEL_PATH": "epoch_003.pt",
          "UNET_CANDIDATE_THRESHOLD": 0.05,
          "UNET_SEED_THRESHOLD": 0.30,
          "not_a_config_key": 99
        }
        """,
        encoding="utf-8",
    )

    cfg, applied = saturn.load_analysis_profile(profile)

    assert cfg["SEGMENTATION_ENGINE"] == "unet_primary"
    assert cfg["TRACKING_BACKEND"] == "global_assignment"
    assert cfg["UNET_MODEL_PATH"] == str(checkpoint.resolve())
    assert cfg["_ACTIVE_PROFILE_NAME"] == "combined.json"
    assert cfg["_ACTIVE_PROFILE_APPLIED_KEY_COUNT"] == 5
    assert "not_a_config_key" not in applied


def test_load_analysis_profile_accepts_nested_tuner_payload(tmp_path):
    saturn = load_saturn()
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"checkpoint")
    profile = tmp_path / "nested.json"
    profile.write_text(
        """
        {
          "score": 123,
          "best_parameters": {
            "SEGMENTATION_ENGINE": "unet_primary",
            "UNET_MODEL_PATH": "missing_original.pt",
            "TRACKING_BACKEND": "global_assignment"
          }
        }
        """,
        encoding="utf-8",
    )

    cfg, applied = saturn.load_analysis_profile(
        profile,
        checkpoint_override=checkpoint,
    )

    assert len(applied) == 3
    assert cfg["UNET_MODEL_PATH"] == str(checkpoint.resolve())
    assert cfg["TRACKING_BACKEND"] == "global_assignment"


def test_analysis_runtime_validation_requires_real_unet_checkpoint(tmp_path):
    saturn = load_saturn()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "unet_primary",
            "UNET_MODEL_PATH": str(tmp_path / "missing.pt"),
        }
    )
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        saturn.validate_analysis_runtime_config(cfg)

    checkpoint = tmp_path / "epoch_003.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg["UNET_MODEL_PATH"] = str(checkpoint)
    status = saturn.validate_analysis_runtime_config(cfg)
    assert status["unet_required"]
    assert status["checkpoint_path"] == str(checkpoint.resolve())
    assert "epoch_003.pt" in saturn.analysis_profile_summary(cfg)


def test_analysis_runtime_validation_allows_probability_cache_without_checkpoint():
    saturn = load_saturn()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "unet_primary",
            "UNET_MODEL_PATH": "",
            "_UNET_PROBABILITY_CACHE": {5: np.zeros((4, 4), dtype=np.float32)},
        }
    )
    status = saturn.validate_analysis_runtime_config(cfg)
    assert status["unet_required"]
    assert status["checkpoint_path"] == ""


def test_settings_bundle_copies_profile_checkpoint_and_checksums(tmp_path):
    saturn = load_saturn()
    profile = tmp_path / "reviewed.json"
    profile.write_text(
        '{"SEGMENTATION_ENGINE": "unet_primary"}',
        encoding="utf-8",
    )
    checkpoint = tmp_path / "epoch_003.pt"
    checkpoint.write_bytes(b"checkpoint contents")
    output = tmp_path / "analysis_output"
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "unet_primary",
            "UNET_MODEL_PATH": str(checkpoint),
            "_ACTIVE_PROFILE_PATH": str(profile),
            "_ACTIVE_PROFILE_NAME": profile.name,
        }
    )

    paths = saturn.save_analysis_settings_bundle(output, cfg)

    settings = output / "settings"
    assert paths["settings_dir"] == settings.resolve()
    assert (settings / "analysis_profile_used.json").read_bytes() == (
        profile.read_bytes()
    )
    assert (settings / "epoch_003.pt").read_bytes() == checkpoint.read_bytes()
    manifest = json.loads(
        (settings / "settings_manifest.json").read_text(encoding="utf-8")
    )
    records = {item["role"]: item for item in manifest["files"]}
    assert len(records["unet_checkpoint"]["sha256"]) == 64
    assert records["unet_checkpoint"]["size_bytes"] == len(
        b"checkpoint contents"
    )
    runtime = json.loads(
        (settings / "runtime_parameters.json").read_text(encoding="utf-8")
    )
    assert runtime["UNET_MODEL_PATH"] == str(checkpoint)
    assert "_ACTIVE_PROFILE_PATH" not in runtime


def test_settings_bundle_fails_before_run_when_checkpoint_is_missing(tmp_path):
    saturn = load_saturn()
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "SEGMENTATION_ENGINE": "unet_primary",
            "UNET_MODEL_PATH": str(tmp_path / "missing.pt"),
        }
    )
    with pytest.raises(FileNotFoundError, match="Cannot archive missing"):
        saturn.save_analysis_settings_bundle(tmp_path / "output", cfg)


def test_seed_connected_low_probability_pixels_are_retained():
    saturn = load_saturn()
    prob = np.zeros((12, 12), dtype=np.float32)
    prob[5, 2:9] = 0.10
    prob[5, 5] = 0.80
    mask, seeds, _, _ = saturn._build_unet_primary_foreground(
        prob, np.ones_like(prob, dtype=bool), primary_cfg(saturn)
    )
    assert mask[5, 2:9].all()
    assert seeds[5, 5]


def test_seedless_low_probability_component_is_rejected():
    saturn = load_saturn()
    prob = np.zeros((10, 10), dtype=np.float32)
    prob[3, 2:7] = 0.12
    mask, _, rejected, audit = saturn._build_unet_primary_foreground(
        prob, np.ones_like(prob, dtype=bool), primary_cfg(saturn)
    )
    assert not mask.any()
    assert np.any(
        rejected == saturn._UNET_PRIMARY_REASON_CODES[
            "no_high_confidence_seed"
        ]
    )
    assert audit[0]["technical_reason"] == "no_high_confidence_seed"


def test_roi_and_exclusion_are_strictly_respected():
    saturn = load_saturn()
    prob = np.full((12, 12), 0.8, dtype=np.float32)
    valid = np.zeros_like(prob, dtype=bool)
    valid[2:10, 2:10] = True
    valid[5:7, 5:7] = False
    seg = saturn._build_unet_primary_segmentation(
        prob, valid, primary_cfg(saturn)
    )
    assert not np.any(seg["mask_clean"] & ~valid)
    assert not np.any(seg["skel_pruned"] & ~valid)


def test_touching_objects_with_two_seed_regions_keep_two_labels():
    saturn = load_saturn()
    prob = np.zeros((20, 24), dtype=np.float32)
    prob[8:12, 3:21] = 0.10
    prob[8:12, 4:8] = 0.85
    prob[8:12, 16:20] = 0.90
    cfg = primary_cfg(saturn, UNET_PRIMARY_MIN_COMPONENT_PX=3)
    seg = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), cfg
    )
    labels = seg["unet_primary_instance_labels"]
    assert set(np.unique(labels)) == {0, 1, 2}


def test_watershed_labels_are_not_collapsed_by_binary_relabeling():
    saturn = load_saturn()
    prob = np.zeros((18, 24), dtype=np.float32)
    foreground = np.zeros_like(prob, dtype=bool)
    foreground[7:11, 2:22] = True
    prob[foreground] = 0.1
    seeds = np.zeros_like(foreground)
    seeds[8:10, 3:7] = True
    seeds[8:10, 17:21] = True
    prob[seeds] = 0.9
    labels, _, _, _ = saturn._split_unet_probability_instances(
        prob, foreground, seeds, 2
    )
    assert np.max(labels) == 2
    assert np.any(labels == 1)
    assert np.any(labels == 2)


def test_one_continuous_seed_stays_one_elongated_instance():
    saturn = load_saturn()
    prob = np.zeros((14, 28), dtype=np.float32)
    prob[6:9, 3:19] = 0.60
    seg = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), primary_cfg(saturn)
    )
    assert np.max(seg["unet_primary_instance_labels"]) == 1


def test_primary_splitter_uses_dedicated_instance_seed_threshold():
    saturn = load_saturn()
    prob = np.zeros((18, 42), dtype=np.float32)
    prob[7:11, 3:39] = 0.35
    prob[7:11, 4:14] = 0.80
    prob[7:11, 28:38] = 0.85
    cfg = primary_cfg(
        saturn,
        UNET_SEED_THRESHOLD=0.30,
        UNET_INSTANCE_SEED_THRESHOLD=0.50,
        UNET_PRIMARY_OVERLONG_SPLIT_ENABLE=False,
    )
    seg = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), cfg
    )
    assert np.max(seg["unet_primary_instance_labels"]) == 2
    assert np.max(measure.label(seg["unet_seed_mask"])) == 1
    assert np.max(measure.label(seg["unet_instance_seed_mask"])) == 2


def test_overlong_refinement_partitions_mask_without_changing_coverage():
    saturn = load_saturn()
    prob = np.zeros((20, 58), dtype=np.float32)
    prob[8:12, 4:54] = 0.90
    cfg = primary_cfg(
        saturn,
        UNET_INSTANCE_SEED_THRESHOLD=0.50,
        UNET_PRIMARY_OVERLONG_SPLIT_ENABLE=True,
        UNET_PRIMARY_OVERLONG_SPLIT_TRIGGER_UM=20.0,
        UNET_PRIMARY_OVERLONG_SPLIT_TARGET_UM=10.0,
        UNET_PRIMARY_OVERLONG_SPLIT_MIN_CHILD_UM=2.0,
    )
    seg = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), cfg
    )
    labels = seg["unet_primary_instance_labels"]
    assert np.max(labels) >= 4
    assert np.array_equal(labels > 0, prob >= 0.05)
    measured = saturn.measure_spermatids(seg, cfg)["results"]
    lengths = [row["length_px_geodesic"] for row in measured]
    assert lengths
    assert max(lengths) <= 20.0


def test_width_is_measured_from_each_instance_mask():
    saturn = load_saturn()
    labels = np.zeros((18, 18), dtype=np.int32)
    labels[4:14, 3:6] = 1
    labels[4:14, 6:11] = 2
    centerlines, metadata, failures = (
        saturn._centerline_unet_primary_instances(labels)
    )
    seg = {
        "unet_primary_instance_labels": labels,
        "unet_primary_centerline_labels": centerlines,
        "unet_probability": np.where(labels > 0, 0.8, 0).astype(np.float32),
        "unet_primary_parent_by_instance": {1: 1, 2: 1},
        "unet_primary_instance_sources": {1: "unet_primary", 2: "unet_primary"},
        "unet_primary_centerline_metadata": metadata,
        "unet_primary_technical_failures": failures,
        "unet_primary_rejected_reason": np.zeros_like(labels, dtype=np.uint8),
    }
    measured = saturn._measure_unet_primary_instances(
        seg, primary_cfg(saturn)
    )
    widths = {row["label"]: row["width_px"] for row in measured["results"]}
    assert widths[1] < widths[2]
    assert widths[1] <= 4.0


def test_unet_length_is_measured_from_final_mask_centerline_and_annotated():
    saturn = load_saturn()
    prob = np.zeros((20, 42), dtype=np.float32)
    prob[8:12, 3:39] = 0.9
    cfg = primary_cfg(
        saturn,
        UM_PER_PX_XY=0.5,
        UNET_PRIMARY_OVERLONG_SPLIT_ENABLE=False,
    )
    seg = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), cfg
    )
    measured = saturn.measure_spermatids(seg, cfg)["results"]
    assert len(measured) == 1
    result = measured[0]
    assert result["length_measurement_method"] == (
        "final_instance_mask_centerline"
    )
    assert result["centerline_within_instance_mask"]
    assert result["length_review_band"] == "15_to_20_um_long_review"
    assert not result["suspected_multi_object_merge"]


def test_unresolved_above_20_um_component_is_a_technical_merge_review():
    saturn = load_saturn()
    prob = np.zeros((20, 42), dtype=np.float32)
    prob[8:12, 3:39] = 0.9
    cfg = primary_cfg(
        saturn,
        UM_PER_PX_XY=1.0,
        UNET_PRIMARY_OVERLONG_SPLIT_ENABLE=False,
    )
    seg = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), cfg
    )
    result = saturn.measure_spermatids(seg, cfg)["results"][0]
    assert result["length_review_band"] == (
        "above_20_um_fused_component_review"
    )
    assert result["suspected_multi_object_merge"]

    tracks = saturn.flag_quality_tracks(
        pd.DataFrame([{
            "track_id": 1,
            "total_3d_length_um": result["length_px_geodesic"],
            "suspected_multi_object_merge": True,
        }]),
        cfg,
    )
    assert not bool(tracks.loc[0, "technical_valid"])
    assert "clear_multi_object_connected_component" in str(
        tracks.loc[0, "technical_failure_reasons"]
    )


def test_morphology_outliers_are_warning_only():
    saturn = load_saturn()
    prob = np.zeros((20, 20), dtype=np.float32)
    prob[5:15, 5:15] = 0.9
    cfg = primary_cfg(
        saturn,
        MAX_WIDTH_UM=1.0,
        MIN_LENGTH_WIDTH_RATIO=10.0,
        MAX_GEODESIC_LEN_UM=3.0,
    )
    seg = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), cfg
    )
    measured = saturn.measure_spermatids(seg, cfg)
    assert len(measured["results"]) == 1
    assert measured["results"][0]["morphology_warning"]
    assert not measured["results"][0]["technical_failure"]


def test_every_accepted_instance_contains_a_seed():
    saturn = load_saturn()
    prob = np.zeros((15, 20), dtype=np.float32)
    prob[3:6, 2:8] = 0.10
    prob[4, 4] = 0.8
    prob[9:12, 10:17] = 0.10
    seg = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), primary_cfg(saturn)
    )
    labels = seg["unet_primary_instance_labels"]
    seeds = prob >= 0.30
    for value in np.unique(labels):
        if value:
            assert np.any(seeds & (labels == value))


@pytest.mark.parametrize(
    ("engine", "expected_action"),
    [
        ("classical_saturn", None),
        ("unet_assisted", "replace_with_unet_candidate"),
        ("hybrid", "none"),
    ],
)
def test_existing_engine_candidate_support_behavior_is_preserved(
    engine, expected_action
):
    saturn = load_saturn()
    shape = (8, 8)
    mask = np.zeros(shape, dtype=bool)
    mask[1, 1] = True
    ridge = np.zeros(shape, dtype=np.float32)
    prob = np.zeros(shape, dtype=np.float32)
    prob[3, 3] = 0.8
    cfg = saturn.CONFIG.copy()
    cfg.update({
        "SEGMENTATION_ENGINE": engine,
        "_UNET_PROBABILITY_CACHE": {5: prob},
        "UNET_THRESHOLD_MODE": "soft",
    })
    result = saturn._apply_unet_candidate_support(
        mask.copy(),
        ridge,
        np.ones(shape, dtype=bool),
        shape,
        (0, 8, 0, 8),
        np.ones(shape, dtype=bool),
        cfg,
        None,
        z_idx=5,
    )
    if engine == "classical_saturn":
        assert np.array_equal(result[0], mask)
        assert not result[5]["unet_enabled"]
    else:
        assert result[5]["unet_mask_action"] == expected_action


def test_unet_primary_fails_without_cache_or_checkpoint():
    saturn = load_saturn()
    shape = (8, 8)
    cfg = primary_cfg(saturn, UNET_MODEL_PATH="")
    with pytest.raises(RuntimeError, match="requires UNET_MODEL_PATH"):
        saturn._apply_unet_candidate_support(
            np.zeros(shape, dtype=bool),
            np.zeros(shape, dtype=np.float32),
            np.ones(shape, dtype=bool),
            shape,
            (0, 8, 0, 8),
            np.ones(shape, dtype=bool),
            cfg,
            np.zeros((3, *shape), dtype=np.float32),
            z_idx=5,
        )


def test_repeated_instance_builds_are_deterministic():
    saturn = load_saturn()
    rng = np.random.default_rng(57)
    prob = rng.uniform(0, 0.2, size=(25, 25)).astype(np.float32)
    prob[10:14, 4:9] = 0.8
    prob[10:14, 16:21] = 0.9
    prob[11:13, 9:16] = 0.1
    cfg = primary_cfg(saturn, UNET_PRIMARY_MIN_COMPONENT_PX=2)
    first = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), cfg
    )
    second = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), cfg
    )
    assert np.array_equal(
        first["unet_primary_instance_labels"],
        second["unet_primary_instance_labels"],
    )


def test_filled_and_centerline_ids_map_one_to_one():
    saturn = load_saturn()
    prob = np.zeros((20, 24), dtype=np.float32)
    prob[4:7, 3:10] = 0.8
    prob[13:16, 14:21] = 0.9
    seg = saturn._build_unet_primary_segmentation(
        prob, np.ones_like(prob, dtype=bool), primary_cfg(saturn)
    )
    instance_ids = set(np.unique(seg["unet_primary_instance_labels"])) - {0}
    centerline_ids = set(np.unique(seg["unet_primary_centerline_labels"])) - {0}
    assert instance_ids == centerline_ids


def test_segment_slice_compatibility_fields_are_unet_primary_population():
    saturn = load_saturn()
    shape = (32, 32)
    image = np.zeros(shape, dtype=np.float32)
    image[14:18, 6:26] = 100.0
    probability = np.zeros(shape, dtype=np.float32)
    probability[14:18, 6:26] = 0.10
    probability[14:18, 12:20] = 0.80
    roi = np.zeros(shape, dtype=bool)
    roi[2:30, 2:30] = True
    exclusion = np.zeros(shape, dtype=bool)
    exclusion[2:5, 2:5] = True
    cfg = primary_cfg(
        saturn,
        _UNET_PROBABILITY_CACHE={5: probability},
        SAVE_DEBUG_IMAGES=False,
    )
    seg = saturn.segment_slice(
        image,
        cfg,
        z_idx=5,
        roi_mask=roi,
        exclusion_mask=exclusion,
        unet_context_stack=None,
    )
    assert np.array_equal(
        seg["mask_clean"], seg["unet_primary_instance_labels"] > 0
    )
    assert np.array_equal(
        seg["skel_labeled"], seg["unet_primary_centerline_labels"]
    )
    assert not np.any(seg["unet_probability"][~roi])
    assert not np.any(seg["unet_probability"][exclusion])


def test_optional_classical_addition_cannot_overwrite_unet_instance():
    saturn = load_saturn()
    prob = np.zeros((20, 30), dtype=np.float32)
    prob[7:11, 3:12] = 0.8
    classical = np.zeros_like(prob, dtype=bool)
    classical[7:11, 3:12] = True
    classical[7:11, 20:27] = True
    cfg = primary_cfg(
        saturn,
        UNET_PRIMARY_CLASSICAL_ADDITIONS_ENABLE=True,
        UNET_PRIMARY_CLASSICAL_EXCLUDE_DILATION_PX=1,
    )
    seg = saturn._build_unet_primary_segmentation(
        prob,
        np.ones_like(prob, dtype=bool),
        cfg,
        classical_mask=classical,
    )
    labels = seg["unet_primary_instance_labels"]
    assert np.all(labels[7:11, 3:12] == 1)
    assert seg["unet_primary_debug"]["saturn_only_additions"] == 1
    assert seg["unet_primary_instance_sources"][2] == "saturn_only_addition"


def test_smoke_runner_rejects_large_target_set_without_override():
    runner = load_runner()
    with pytest.raises(ValueError, match="allow-large-run"):
        runner.validate_target_values(list(range(7)), allow_large_run=False)


def test_smoke_runner_keeps_exact_requested_targets():
    runner = load_runner()
    targets = runner.validate_target_values(
        [5, 6, 12], allow_large_run=False
    )
    mocked_stack = {z: f"z{z:03d}.tif" for z in range(88)}
    selected = runner.resolve_target_files(mocked_stack, targets)
    assert list(selected) == [5, 6, 12]
    assert len(selected) == 3
