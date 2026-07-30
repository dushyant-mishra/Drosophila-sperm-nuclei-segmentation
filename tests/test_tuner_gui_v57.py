import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tuner_gui():
    spec = importlib.util.spec_from_file_location(
        "saturn_v57_tuner_gui_test",
        ROOT / "utils" / "tuner_gui_Saturnv5_7.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mixed_runner():
    spec = importlib.util.spec_from_file_location(
        "saturn_v57_mixed_runner_test",
        ROOT / "scripts" / "run_v57_mixed_unet_primary_tuning.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_stack_unet_command_binds_all_analysis_inputs(tmp_path):
    gui = load_tuner_gui()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    roi = tmp_path / "roi.npy"
    roi.write_bytes(b"roi")
    profile = tmp_path / "base.json"
    profile.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "epoch_003.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "output"

    command = gui.build_single_stack_command(
        {
            "mode_label": "U-Net-primary 2D segmentation",
            "image_dir": str(image_dir),
            "roi_path": str(roi),
            "exclusion_mask": "",
            "metadata_xml": "",
            "base_profile": str(profile),
            "checkpoint": str(checkpoint),
            "output_dir": str(output),
            "slices": "5,12,35",
            "candidates": 16,
            "review_candidates": 8,
            "seed": 12345,
            "auto_calibration": True,
            "rebuild_cache": True,
        },
        python_executable="python",
    )

    assert command[:4] == [
        "python",
        "-u",
        str(gui.TUNER),
        "--mode",
    ]
    assert command[4] == "unet_primary"
    assert command[command.index("--unet-model") + 1] == str(
        checkpoint.resolve()
    )
    assert command[command.index("--base-params") + 1] == str(profile.resolve())
    assert command[command.index("--slices") + 1] == "5,12,35"
    assert "--auto-calibration" in command
    assert "--rebuild-unet-cache" in command


def test_single_stack_unet_command_rejects_missing_checkpoint(tmp_path):
    gui = load_tuner_gui()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    roi = tmp_path / "roi.npy"
    roi.write_bytes(b"roi")
    with pytest.raises(ValueError, match="checkpoint not found"):
        gui.build_single_stack_command(
            {
                "mode_label": "U-Net-primary 2D segmentation",
                "image_dir": str(image_dir),
                "roi_path": str(roi),
                "output_dir": str(tmp_path / "output"),
                "checkpoint": str(tmp_path / "missing.pt"),
            }
        )


def test_mixed_command_uses_production_orchestrator(tmp_path):
    gui = load_tuner_gui()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("specimen_id\n", encoding="utf-8")
    checkpoint = tmp_path / "epoch_003.pt"
    checkpoint.write_bytes(b"checkpoint")
    profile = tmp_path / "base.json"
    profile.write_text("{}", encoding="utf-8")

    command = gui.build_mixed_command(
        {
            "manifest": str(manifest),
            "checkpoint": str(checkpoint),
            "base_profile": str(profile),
            "output_root": str(tmp_path / "output"),
            "segmentation_candidates": 24,
            "tracking_candidates": 20,
            "tracking_slice_count": 5,
            "seed": 12345,
            "validate_only": True,
        },
        python_executable="python",
    )

    assert command[1:3] == ["-u", str(gui.MIXED_RUNNER)]
    assert command[command.index("--checkpoint") + 1] == str(
        checkpoint.resolve()
    )
    assert command[command.index("--tracking-candidates") + 1] == "20"
    assert "--validate-only" in command


def test_mixed_campaign_archives_manifest_profile_and_checkpoint(tmp_path):
    runner = load_mixed_runner()
    manifest = tmp_path / "source_manifest.csv"
    manifest.write_text("specimen_id,group\nA,Control\n", encoding="utf-8")
    profile = tmp_path / "source_profile.json"
    profile.write_text('{"SEGMENTATION_ENGINE":"unet_primary"}', encoding="utf-8")
    checkpoint = tmp_path / "epoch_003.pt"
    checkpoint.write_bytes(b"model")
    run_root = tmp_path / "run"

    settings = runner.archive_tuning_inputs(
        run_root,
        manifest,
        checkpoint,
        profile,
    )

    assert (settings / "mixed_tuner_manifest.csv").read_bytes() == (
        manifest.read_bytes()
    )
    assert (settings / "analysis_profile_used.json").read_bytes() == (
        profile.read_bytes()
    )
    assert (settings / "epoch_003.pt").read_bytes() == checkpoint.read_bytes()
    assert (settings / "settings_manifest.json").is_file()
