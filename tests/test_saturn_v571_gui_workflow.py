import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[1]


def load_saturn_v571():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_gui_workflow_test",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def custom_profile(tmp_path, module):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"v571-test-checkpoint")
    profile = tmp_path / "analysis_profile.json"
    profile.write_text(
        json.dumps(
            {
                "SEGMENTATION_ENGINE": "unet_primary",
                "UNET_OUTPUT_MODE": "dual_head",
                "UNET_MODEL_PATH": checkpoint.name,
                "UNET_CHECKPOINT_SHA256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    cfg, _applied = module.load_analysis_profile(profile, module.CONFIG)
    return cfg


def test_arbitrary_profile_is_custom_and_cannot_claim_reviewed_status(tmp_path):
    saturn = load_saturn_v571()
    cfg = custom_profile(tmp_path, saturn)

    assert saturn.analysis_profile_review_state(cfg) == "custom"
    assert "[Custom - review required]" in saturn.analysis_profile_summary(cfg)


def test_repository_production_profile_is_reviewed_and_drift_is_detected():
    saturn = load_saturn_v571()
    profile = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
    cfg, _applied = saturn.load_analysis_profile(profile, saturn.CONFIG)

    assert saturn.analysis_profile_review_state(cfg) == "reviewed"
    assert "[Reviewed]" in saturn.analysis_profile_summary(cfg)

    cfg["THRESHOLD_HI"] = float(cfg["THRESHOLD_HI"]) - 1.0
    assert saturn.analysis_profile_review_state(cfg) == "custom"
    assert "[Custom - review required]" in saturn.analysis_profile_summary(cfg)


def test_mark_reviewed_profile_custom_records_reason():
    saturn = load_saturn_v571()
    profile = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
    cfg, _applied = saturn.load_analysis_profile(profile, saturn.CONFIG)

    saturn.mark_analysis_profile_custom(cfg, "Advanced parameters changed.")

    assert saturn.analysis_profile_review_state(cfg) == "custom"
    assert "[Custom - review required]" in saturn.analysis_profile_summary(cfg)
    assert cfg["_ACTIVE_PROFILE_DIRTY_REASON"] == "Advanced parameters changed."


def test_profile_resolution_drops_stale_manual_analysis_settings(tmp_path):
    saturn = load_saturn_v571()
    stale = saturn.CONFIG.copy()
    stale["THRESHOLD_HI"] = 12.345
    stale["INPUT_DIR"] = str(tmp_path / "selected-input")

    cfg = custom_profile(tmp_path, saturn)
    profile_path = Path(cfg["_ACTIVE_PROFILE_PATH"])
    resolved, _applied = saturn.load_analysis_profile(profile_path, stale)

    assert resolved["THRESHOLD_HI"] == saturn.ANALYSIS_CONFIG_DEFAULTS["THRESHOLD_HI"]
    assert resolved["INPUT_DIR"] == stale["INPUT_DIR"]
    assert len(resolved["_ACTIVE_PROFILE_RESOLVED_SHA256"]) == 64


def test_preflight_collects_actionable_blockers_in_one_report():
    saturn = load_saturn_v571()
    cfg = saturn.CONFIG.copy()

    report = saturn.build_gui_preflight_report(
        cfg,
        operation="Batch Analysis",
        image_count=0,
        roi_ready=False,
        calibration_ready=False,
        output_safe=False,
        require_reviewed_profile=True,
    )

    assert report.ready is False
    codes = {issue.code for issue in report.blocking_issues}
    assert {
        "IMAGES_MISSING",
        "ROI_MISSING",
        "PROFILE_NOT_REVIEWED",
        "CALIBRATION_MISSING",
        "OUTPUT_UNSAFE",
    }.issubset(codes)
    assert all(issue.action for issue in report.blocking_issues)


def test_reviewed_profile_identity_passes_but_failed_scientific_audit_blocks_run(tmp_path):
    saturn = load_saturn_v571()
    profile = ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
    cfg, _applied = saturn.load_analysis_profile(profile, saturn.CONFIG)

    report = saturn.build_gui_preflight_report(
        cfg,
        operation="Batch Analysis",
        image_count=3,
        roi_ready=True,
        calibration_ready=True,
        output_safe=True,
        require_reviewed_profile=True,
    )

    assert report.ready is False
    codes = {issue.code for issue in report.issues}
    assert "PROFILE_READY" in codes
    assert "RUNTIME_READY" in codes
    assert "AUDIT_GATE_BLOCKED" in codes


def test_rows_receive_stable_source_instance_key_when_measurement_lacks_one():
    saturn = load_saturn_v571()
    result = {
        "label": 7,
        "length_px_geodesic": 10.0,
        "length_px_count": 9.0,
        "width_px": 2.0,
        "length_width_ratio": 5.0,
        "tortuosity": 1.1,
        "n_endpoints": 2,
        "n_branch_nodes": 0,
        "centroid_x": 12.0,
        "centroid_y": 15.0,
    }

    row = saturn.rows_from_results([result], z_idx=12, um=0.5)[0]

    assert row["source_instance_key"] == "z0012:instance:7"


def test_preview_no_longer_contains_skeleton_only_manual_corrector():
    source = (ROOT / "sperm_segmentation_saturnv5.7.1.py").read_text(
        encoding="utf-8"
    )

    assert "class ManualCorrector" not in source
    assert "WORKFLOW-MANUAL-CORRECTION-001" in source


def test_study_gui_can_start_incomplete_study_and_cancel_report_refresh():
    saturn = load_saturn_v571()
    run_source = inspect.getsource(saturn.SpermGUI._study_run)
    refresh_source = inspect.getsource(
        saturn.SpermGUI._study_refresh_analysis_package
    )
    stop_source = inspect.getsource(saturn.SpermGUI._study_request_stop)

    assert "_validated_complete_study_report_inputs" not in run_source
    assert "stop_requested=self._study_stop_event" in refresh_source
    assert "progress_callback=report_progress" in refresh_source
    assert "_study_report_running" in stop_source


def test_study_gui_exposes_preview_and_metadata_locked_calibration():
    source = (ROOT / "sperm_segmentation_saturnv5.7.1.py").read_text(
        encoding="utf-8"
    )
    edit_source = inspect.getsource(load_saturn_v571().SpermGUI._study_edit_cell)

    assert '"Preview Selected"' in source
    assert "_study_preview_selected_specimen" in source
    assert 'CONFIG["_CALIBRATION_METADATA_SHA256"]' in source
    assert "Calibration Comes From Metadata" in edit_source


def test_report_refresh_stop_dispatches_stopped_callback(
    tmp_path, monkeypatch
):
    saturn = load_saturn_v571()

    class Var:
        def __init__(self):
            self.value = ""

        def set(self, value):
            self.value = value

    class Button:
        def config(self, **_kwargs):
            return None

    class Root:
        def after(self, _delay, callback):
            callback()

    class ImmediateThread:
        def __init__(self, target, daemon=True):
            self.target = target

        def start(self):
            self.target()

    gui = saturn.SpermGUI.__new__(saturn.SpermGUI)
    gui._study_running = False
    gui._study_report_running = False
    gui._study_stop_event = saturn.threading.Event()
    gui.study_output_dir = str(tmp_path)
    gui.study_rows = [{"include": True}]
    gui.study_window = None
    gui.study_stop_button = Button()
    gui.study_status_var = Var()
    gui.root = Root()
    monkeypatch.setattr(saturn.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        saturn,
        "generate_study_between_sample_analysis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            saturn.AnalysisStopRequested("synthetic report stop")
        ),
    )
    monkeypatch.setattr(saturn.messagebox, "showinfo", lambda *_a, **_k: None)

    gui._study_refresh_analysis_package()

    assert gui._study_report_running is False
    assert gui.study_status_var.value == "Analysis package generation stopped"


def test_corrupt_resume_state_fails_visibly(tmp_path):
    saturn = load_saturn_v571()
    state = tmp_path / "study_run_state.json"
    state.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="resume state is unreadable"):
        saturn._load_study_resume_state(state, "expected")


def test_resume_state_blocks_changed_analysis_settings(tmp_path):
    saturn = load_saturn_v571()
    state = tmp_path / "study_run_state.json"
    state.write_text(
        json.dumps(
            {
                "pipeline_version": saturn._VERSION,
                "config_hash": "old",
                "samples": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="settings differ"):
        saturn._load_study_resume_state(state, "new")


def test_study_config_fingerprint_changes_with_runtime_code_identity(monkeypatch):
    saturn = load_saturn_v571()
    cfg = {"THRESHOLD_HI": 90}
    monkeypatch.setattr(
        saturn,
        "_pipeline_runtime_identity",
        lambda: {"runtime_identity_sha256": "a" * 64},
    )
    first = saturn._study_config_fingerprint(cfg)
    monkeypatch.setattr(
        saturn,
        "_pipeline_runtime_identity",
        lambda: {"runtime_identity_sha256": "b" * 64},
    )
    second = saturn._study_config_fingerprint(cfg)
    assert first != second


def test_runtime_identity_covers_executing_unet_and_report_sources():
    saturn = load_saturn_v571()
    sources = saturn._pipeline_runtime_identity()["source_sha256"]

    assert "utils/saturn_unet25d_bridge.py" in sources
    assert "scripts/generate_v571_biological_comparison.py" in sources
    assert "scripts/generate_v57_biological_comparison.py" in sources


def test_resume_attempt_must_remain_under_expected_specimen_folder(tmp_path):
    saturn = load_saturn_v571()
    valid = tmp_path / "study" / "samples" / "sample-01" / "attempt_001"
    valid.mkdir(parents=True)
    assert saturn._validated_resume_output_path(
        tmp_path / "study", "sample-01", valid
    ) == valid.resolve()
    with pytest.raises(ValueError, match="outside its study specimen folder"):
        saturn._validated_resume_output_path(
            tmp_path / "study",
            "sample-01",
            tmp_path / "other" / "attempt_001",
        )


def test_authoritative_instance_evidence_is_saved_once_with_hashes(tmp_path):
    saturn = load_saturn_v571()
    instances = np.zeros((12, 14), dtype=np.int32)
    instances[2:6, 3:8] = 1
    centerlines = np.zeros_like(instances)
    centerlines[4, 3:8] = 1
    cfg = saturn.CONFIG.copy()
    cfg.update(
        {
            "UM_PER_PX_XY": 0.5,
            "UM_PER_SLICE_Z": 1.0,
            "UNET_CHECKPOINT_SHA256": "a" * 64,
        }
    )

    metadata = saturn.save_authoritative_instance_evidence(
        tmp_path,
        4,
        {
            "unet_primary_instance_labels": instances,
            "unet_primary_centerline_labels": centerlines,
        },
        cfg,
    )

    evidence = tmp_path / "raw_evidence" / "instance_labels" / "z0004"
    assert metadata["instance_count"] == 1
    assert len(metadata["instance_labels_sha256"]) == 64
    assert np.array_equal(
        tifffile.imread(evidence / "instance_labels.tif"),
        instances,
    )
    assert (evidence / "centerline_labels.tif").is_file()
    assert (evidence / "evidence.json").is_file()
    assert not list(evidence.parent.glob(".z0004.tmp-*"))
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        saturn.save_authoritative_instance_evidence(
            tmp_path,
            4,
            {
                "unet_primary_instance_labels": instances,
                "unet_primary_centerline_labels": centerlines,
            },
            cfg,
        )


def test_rows_from_results_reject_duplicate_source_instance_ids():
    saturn = load_saturn_v571()
    row = {
        "label": 1,
        "source_instance_key": "z0004:instance:1",
        "length_px_geodesic": 10.0,
        "length_px_count": 10.0,
        "width_px": 2.0,
        "length_width_ratio": 5.0,
        "tortuosity": 1.0,
        "n_endpoints": 2,
        "n_branch_nodes": 0,
        "centroid_x": 5.0,
        "centroid_y": 6.0,
        "area_px": 20.0,
    }
    with pytest.raises(ValueError, match="Duplicate source_instance_key"):
        saturn.rows_from_results([row, dict(row)], 4, 0.5)


def test_completion_marker_validates_inventory_and_hashes(tmp_path):
    saturn = load_saturn_v571()
    output = tmp_path / "attempt_001"
    (output / "biologist_results").mkdir(parents=True)
    (output / "settings").mkdir(parents=True)
    (output / "analysis_summary.csv").write_text("metric,value\ncount,1\n", encoding="utf-8")
    (output / "specimen_input_manifest.json").write_text("{}", encoding="utf-8")
    (output / "settings" / "settings_manifest.json").write_text("{}", encoding="utf-8")
    (output / "settings" / "source_image_manifest.json").write_text(
        json.dumps({"ordered_source_images": []}), encoding="utf-8"
    )
    (output / "settings" / "roi_mask_source.npy").write_bytes(b"roi")
    (output / "biologist_results" / "sample_summary.csv").write_text(
        "metric,value\ncount,1\n", encoding="utf-8"
    )
    (output / f"track_summary_{saturn._VERSION}.csv").write_text(
        "track_id\n1\n", encoding="utf-8"
    )
    inventory = saturn._study_completion_artifact_inventory(output)
    marker = {
        "schema_version": "1.0",
        "pipeline_version": saturn._VERSION,
        "sample_id": "sample-01",
        "config_hash": "config-hash",
        "specimen_input_hash": "input-hash",
        "artifact_inventory": inventory,
    }
    saturn._study_atomic_json(output / "sample_complete.json", marker)

    restored = saturn._validate_study_completion_marker(
        output / "sample_complete.json", "sample-01", "config-hash", "input-hash"
    )
    assert restored["artifact_inventory"] == inventory

    (output / "analysis_summary.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact (size|hash) changed"):
        saturn._validate_study_completion_marker(
            output / "sample_complete.json", "sample-01", "config-hash", "input-hash"
        )


def test_specimen_fingerprint_changes_with_tiff_roi_exclusion_or_manifest_row(tmp_path):
    saturn = load_saturn_v571()
    images = tmp_path / "images"
    images.mkdir()
    image_path = images / "Project001_Series002_z00_ch00.tif"
    tifffile.imwrite(image_path, np.zeros((8, 9), dtype=np.uint16))
    roi_path = tmp_path / "roi.npy"
    np.save(roi_path, np.ones((8, 9), dtype=bool))
    exclusion_path = tmp_path / "exclusion.npy"
    np.save(exclusion_path, np.zeros((8, 9), dtype=bool))
    row = {
        "sample_id": "sample-01",
        "group": "control",
        "group_role": "reference",
        "input_dir": str(images),
        "file_pattern": "*.tif",
        "roi_path": str(roi_path),
        "exclusion_mask_path": str(exclusion_path),
        "xy_um_per_pixel": 0.5,
        "z_um_per_slice": 1.0,
        "acquisition_class": "test",
        "include": True,
    }

    original, _payload = saturn._study_specimen_input_fingerprint(row)
    np.save(roi_path, np.zeros((8, 9), dtype=bool))
    changed_roi, _payload = saturn._study_specimen_input_fingerprint(row)
    np.save(roi_path, np.ones((8, 9), dtype=bool))
    np.save(exclusion_path, np.ones((8, 9), dtype=bool))
    changed_exclusion, _payload = saturn._study_specimen_input_fingerprint(row)
    np.save(exclusion_path, np.zeros((8, 9), dtype=bool))
    tifffile.imwrite(image_path, np.ones((8, 9), dtype=np.uint16))
    changed_image, _payload = saturn._study_specimen_input_fingerprint(row)
    changed_row = dict(row, group="mutant")
    changed_manifest, _payload = saturn._study_specimen_input_fingerprint(
        changed_row
    )

    assert len(
        {original, changed_roi, changed_exclusion, changed_image, changed_manifest}
    ) == 5


def test_unet_completion_requires_authoritative_evidence_for_every_slice(tmp_path):
    saturn = load_saturn_v571()
    output = tmp_path / "attempt_001"
    settings = output / "settings"
    settings.mkdir(parents=True)
    (output / "analysis_summary.csv").write_text("metric,value\n", encoding="utf-8")
    (output / "specimen_input_manifest.json").write_text("{}", encoding="utf-8")
    (settings / "settings_manifest.json").write_text("{}", encoding="utf-8")
    (settings / "source_image_manifest.json").write_text(
        json.dumps({"ordered_source_images": [{"z_index": 0}]}),
        encoding="utf-8",
    )
    (settings / "roi_mask_source.npy").write_bytes(b"roi")

    with pytest.raises(ValueError, match="instance evidence covers 0 of 1"):
        saturn._study_completion_artifact_inventory(
            output,
            {"SEGMENTATION_ENGINE": "unet_primary"},
        )


def test_completion_inventory_authenticates_every_settings_manifest_file(tmp_path):
    saturn = load_saturn_v571()
    output = tmp_path / "attempt_001"
    settings = output / "settings"
    settings.mkdir(parents=True)
    runtime = settings / "runtime_parameters.json"
    runtime.write_text("{}", encoding="utf-8")
    (output / "analysis_summary.csv").write_text("metric,value\n", encoding="utf-8")
    (output / "specimen_input_manifest.json").write_text("{}", encoding="utf-8")
    (settings / "source_image_manifest.json").write_text(
        json.dumps({"ordered_source_images": []}), encoding="utf-8"
    )
    (settings / "roi_mask_source.npy").write_bytes(b"roi")
    (settings / "settings_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "copied_path": str(runtime),
                        "size_bytes": runtime.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Settings manifest artifact hash changed"):
        saturn._study_completion_artifact_inventory(output)


def test_batch_lifecycle_marker_records_complete_and_stopped(tmp_path, monkeypatch):
    saturn = load_saturn_v571()
    monkeypatch.setattr(
        saturn,
        "_pipeline_runtime_identity",
        lambda: {"runtime_identity_sha256": "a" * 64},
    )
    cfg = {"OUTPUT_DIR": str(tmp_path / "complete")}
    monkeypatch.setattr(
        saturn,
        "_process_batch_impl",
        lambda *args, **kwargs: {"elapsed_seconds": 1.25},
    )
    saturn.process_batch(cfg)
    complete = json.loads(
        (tmp_path / "complete" / "run_status.json").read_text(encoding="utf-8")
    )
    assert complete["status"] == "complete"

    def stop_impl(*args, **kwargs):
        raise saturn.AnalysisStopRequested("synthetic stop")

    monkeypatch.setattr(saturn, "_process_batch_impl", stop_impl)
    with pytest.raises(saturn.AnalysisStopRequested):
        saturn.process_batch({"OUTPUT_DIR": str(tmp_path / "stopped")})
    stopped = json.loads(
        (tmp_path / "stopped" / "run_status.json").read_text(encoding="utf-8")
    )
    assert stopped["status"] == "stopped"


def test_required_report_generators_propagate_failure(tmp_path, monkeypatch):
    saturn = load_saturn_v571()

    class BrokenPdf:
        def __init__(self, *_args, **_kwargs):
            raise OSError("synthetic PDF failure")

    monkeypatch.setattr(saturn, "PdfPages", BrokenPdf)
    with pytest.raises(RuntimeError, match="PDF report generation failed"):
        saturn.generate_batch_report(
            tmp_path,
            pd.DataFrame(),
            pd.DataFrame(),
            {"xy": 1.0, "z": 1.0},
            generate_pptx=False,
        )


def test_partial_study_cannot_generate_biological_package(tmp_path):
    saturn = load_saturn_v571()
    output = tmp_path / "study"
    output.mkdir()
    (output / "study_run_state.json").write_text(
        json.dumps({"run_status": "complete_with_failures", "samples": {}}),
        encoding="utf-8",
    )
    (output / "study_manifest.csv").write_text(
        "include,sample_id\nTrue,sample-01\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="group comparison is blocked"):
        saturn._validated_complete_study_report_inputs(output)
    with pytest.raises(ValueError, match="group comparison is blocked"):
        saturn.generate_study_between_sample_analysis(output)


def test_complete_study_report_generation_honors_scientific_gate(
    tmp_path, monkeypatch
):
    saturn = load_saturn_v571()
    output = tmp_path / "study"
    output.mkdir()
    monkeypatch.setattr(
        saturn,
        "_validated_complete_study_report_inputs",
        lambda _path: {"state_sha256": "state"},
    )
    monkeypatch.setattr(
        saturn,
        "production_audit_gate_state",
        lambda: (False, "MEAS-BODY-WIDTH-001 is not accepted"),
    )

    with pytest.raises(RuntimeError, match="scientific audit gate"):
        saturn.generate_study_between_sample_analysis(output)

    assert not (output / "between_sample_analysis").exists()


def test_stale_analysis_package_is_moved_out_of_openable_location(tmp_path):
    saturn = load_saturn_v571()
    output = tmp_path / "study"
    package = output / "between_sample_analysis"
    package.mkdir(parents=True)
    (package / "old.pdf").write_bytes(b"old")
    comparison_pdf = output / "specimen_group_comparison.pdf"
    comparison_pdf.write_bytes(b"old-comparison")

    destination = saturn._invalidate_stale_study_analysis_outputs(output)

    assert destination is not None
    assert not package.exists()
    assert not comparison_pdf.exists()
    assert (destination / "between_sample_analysis" / "old.pdf").is_file()
    assert (destination / "specimen_group_comparison.pdf").is_file()
    invalidation = json.loads(
        (destination / "invalidation.json").read_text(encoding="utf-8")
    )
    assert invalidation["inference_status"] == "blocked"


def test_analysis_package_binding_rejects_stale_state_hash(tmp_path, monkeypatch):
    saturn = load_saturn_v571()
    output = tmp_path / "study"
    package = output / "between_sample_analysis"
    package.mkdir(parents=True)
    (package / "package_source_binding.json").write_text(
        json.dumps(
            {
                "study_state_sha256": "old-state",
                "study_manifest_sha256": "manifest-hash",
                "sample_completion_marker_sha256": {"sample-01": "marker-hash"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        saturn,
        "_validated_complete_study_report_inputs",
        lambda _path: {
            "state_sha256": "current-state",
            "manifest_sha256": "manifest-hash",
            "sample_completion_marker_sha256": {"sample-01": "marker-hash"},
        },
    )

    current, reason = saturn._study_analysis_package_binding_status(output)
    assert current is False
    assert "predates the current study state" in reason


def test_analysis_package_binding_rejects_changed_report_artifact(
    tmp_path, monkeypatch
):
    saturn = load_saturn_v571()
    output = tmp_path / "study"
    package = output / "between_sample_analysis"
    package.mkdir(parents=True)
    artifact = package / "report.pdf"
    artifact.write_bytes(b"original")
    inputs = {
        "state_sha256": "state-hash",
        "manifest_sha256": "manifest-hash",
        "sample_completion_marker_sha256": {"sample-01": "marker-hash"},
        "aggregate_source_sha256": {
            "specimen_summary.csv": "specimen-hash",
            "study_track_records.csv": "tracks-hash",
        },
    }
    runtime_identity = {"runtime_identity_sha256": "a" * 64}
    monkeypatch.setattr(
        saturn, "_pipeline_runtime_identity", lambda: runtime_identity
    )
    binding = {
        "study_state_sha256": inputs["state_sha256"],
        "study_manifest_sha256": inputs["manifest_sha256"],
        "sample_completion_marker_sha256": inputs[
            "sample_completion_marker_sha256"
        ],
        "aggregate_source_sha256": inputs["aggregate_source_sha256"],
        "report_runtime_identity": runtime_identity,
        "report_artifacts": saturn._study_report_artifact_inventory(package),
    }
    (package / "package_source_binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    monkeypatch.setattr(
        saturn, "_validated_complete_study_report_inputs", lambda _path: inputs
    )

    current, _reason = saturn._study_analysis_package_binding_status(output)
    assert current is True
    artifact.write_bytes(b"changed")
    current, reason = saturn._study_analysis_package_binding_status(output)
    assert current is False
    assert "artifacts" in reason


def test_analysis_package_binding_rejects_changed_report_code(
    tmp_path, monkeypatch
):
    saturn = load_saturn_v571()
    output = tmp_path / "study"
    package = output / "between_sample_analysis"
    package.mkdir(parents=True)
    (package / "report.pdf").write_bytes(b"report")
    inputs = {
        "state_sha256": "state-hash",
        "manifest_sha256": "manifest-hash",
        "sample_completion_marker_sha256": {"sample-01": "marker-hash"},
        "aggregate_source_sha256": {
            "specimen_summary.csv": "specimen-hash",
            "study_track_records.csv": "tracks-hash",
        },
    }
    binding = {
        "study_state_sha256": inputs["state_sha256"],
        "study_manifest_sha256": inputs["manifest_sha256"],
        "sample_completion_marker_sha256": inputs[
            "sample_completion_marker_sha256"
        ],
        "aggregate_source_sha256": inputs["aggregate_source_sha256"],
        "report_runtime_identity": {"runtime_identity_sha256": "a" * 64},
        "report_artifacts": saturn._study_report_artifact_inventory(package),
    }
    (package / "package_source_binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    monkeypatch.setattr(
        saturn, "_validated_complete_study_report_inputs", lambda _path: inputs
    )
    monkeypatch.setattr(
        saturn,
        "_pipeline_runtime_identity",
        lambda: {"runtime_identity_sha256": "b" * 64},
    )

    current, reason = saturn._study_analysis_package_binding_status(output)

    assert current is False
    assert "different report code" in reason


def test_analysis_package_binding_rejects_changed_aggregate_source(
    tmp_path, monkeypatch
):
    saturn = load_saturn_v571()
    output = tmp_path / "study"
    package = output / "between_sample_analysis"
    package.mkdir(parents=True)
    (package / "report.pdf").write_bytes(b"report")
    runtime = {"runtime_identity_sha256": "a" * 64}
    inputs = {
        "state_sha256": "state",
        "manifest_sha256": "manifest",
        "sample_completion_marker_sha256": {"sample-01": "marker"},
        "aggregate_source_sha256": {
            "specimen_summary.csv": "new-specimens",
            "study_track_records.csv": "tracks",
        },
    }
    binding = {
        "study_state_sha256": "state",
        "study_manifest_sha256": "manifest",
        "sample_completion_marker_sha256": {"sample-01": "marker"},
        "aggregate_source_sha256": {
            "specimen_summary.csv": "old-specimens",
            "study_track_records.csv": "tracks",
        },
        "report_runtime_identity": runtime,
        "report_artifacts": saturn._study_report_artifact_inventory(package),
    }
    (package / "package_source_binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    monkeypatch.setattr(
        saturn, "_validated_complete_study_report_inputs", lambda _path: inputs
    )
    monkeypatch.setattr(saturn, "_pipeline_runtime_identity", lambda: runtime)

    current, reason = saturn._study_analysis_package_binding_status(output)

    assert current is False
    assert "aggregate tables" in reason


def test_preview_output_directory_never_reuses_prior_run(tmp_path):
    saturn = load_saturn_v571()

    first = Path(saturn.get_unique_named_dir(tmp_path, "z0004_preview"))
    first.mkdir()
    second = Path(saturn.get_unique_named_dir(tmp_path, "z0004_preview"))

    assert first.name == "z0004_preview"
    assert second.name == "z0004_preview_1"


def test_stopped_slice_never_marks_specimen_complete(tmp_path, monkeypatch):
    saturn = load_saturn_v571()
    row = {
        "include": True,
        "sample_id": "sample-01",
        "group": "group-a",
        "group_role": "reference",
        "input_dir": str(tmp_path / "input"),
        "roi_path": str(tmp_path / "roi.npy"),
        "exclusion_mask_path": str(tmp_path / "sample_exclusion.npy"),
        "file_pattern": "*.tif",
        "slice_count": 3,
        "z_min": 0,
        "z_max": 2,
        "xy_um_per_pixel": 0.5,
        "z_um_per_slice": 1.0,
        "calibration_metadata_path": "",
        "calibration_metadata_sha256": "",
        "acquisition_class": "test",
        "status": "validated",
        "message": "",
    }
    monkeypatch.setattr(
        saturn,
        "validate_multisample_manifest",
        lambda rows, cfg=None: ([dict(row)], []),
    )
    monkeypatch.setattr(saturn, "save_multisample_manifest", lambda *args: None)
    monkeypatch.setattr(saturn, "save_study_exclusion_ledger", lambda *args, **kwargs: None)
    monkeypatch.setattr(saturn, "save_analysis_settings_bundle", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        saturn,
        "_write_study_aggregates",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        saturn,
        "_study_specimen_input_fingerprint",
        lambda row: ("a" * 64, {"sample_id": row["sample_id"]}),
    )

    observed_cfg = {}

    def stopped_runner(cfg, progress_callback=None, stop_requested=None):
        observed_cfg.update(cfg)
        raise saturn.AnalysisStopRequested("stopped after slice 1")

    monkeypatch.setattr(saturn, "process_batch", stopped_runner)

    state, _summary = saturn.run_multisample_study(
        [row],
        tmp_path / "output",
        base_cfg=saturn.CONFIG.copy(),
        resume=False,
    )

    record = state["samples"]["sample-01"]
    assert state["run_status"] == "stopped"
    assert record["status"] == "stopped"
    assert observed_cfg["EXCLUSION_MASK_PATH"] == row["exclusion_mask_path"]
    assert not (
        Path(record["output_dir"]) / "sample_complete.json"
    ).exists()


def test_failed_specimen_blocks_group_comparison_and_complete_status(
    tmp_path, monkeypatch
):
    saturn = load_saturn_v571()
    row = {
        "include": True,
        "sample_id": "sample-01",
        "group": "control",
        "group_role": "reference",
        "input_dir": str(tmp_path / "input"),
        "roi_path": str(tmp_path / "roi.npy"),
        "file_pattern": "*.tif",
        "slice_count": 1,
        "z_min": 0,
        "z_max": 0,
        "xy_um_per_pixel": 0.5,
        "z_um_per_slice": 1.0,
        "calibration_metadata_path": "",
        "calibration_metadata_sha256": "",
        "acquisition_class": "test",
        "status": "validated",
        "message": "",
    }
    monkeypatch.setattr(
        saturn,
        "validate_multisample_manifest",
        lambda rows, cfg=None: ([dict(row)], []),
    )
    monkeypatch.setattr(saturn, "save_multisample_manifest", lambda *args: None)
    monkeypatch.setattr(
        saturn, "save_study_exclusion_ledger", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        saturn, "save_analysis_settings_bundle", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        saturn,
        "_study_specimen_input_fingerprint",
        lambda row: ("a" * 64, {"sample_id": row["sample_id"]}),
    )
    aggregate_flags = []
    monkeypatch.setattr(
        saturn,
        "_write_study_aggregates",
        lambda *args, **kwargs: aggregate_flags.append(
            kwargs.get("allow_group_comparison")
        ) or [],
    )

    def failed_runner(_cfg):
        raise RuntimeError("synthetic specimen failure")

    state, _summary = saturn.run_multisample_study(
        [row],
        tmp_path / "output",
        base_cfg=saturn.CONFIG.copy(),
        resume=False,
        batch_runner=failed_runner,
    )

    assert state["run_status"] == "complete_with_failures"
    assert state["samples"]["sample-01"]["status"] == "failed"
    assert aggregate_flags
    assert all(flag is False for flag in aggregate_flags)


def test_group_roles_must_be_consistent_within_each_group():
    saturn = load_saturn_v571()
    frame = pd.DataFrame(
        {
            "group": ["control", "control", "mutant"],
            "group_role": ["reference", "comparison", "comparison"],
        }
    )
    with pytest.raises(ValueError, match="conflicting study roles"):
        saturn._study_explicit_group_pair(frame)


def test_gui_batch_has_one_execution_path_and_calls_process_batch():
    source = (ROOT / "sperm_segmentation_saturnv5.7.1.py").read_text(
        encoding="utf-8"
    )
    assert source.count("    def run_batch_analysis(self):") == 1
    method = source.split("    def run_batch_analysis(self):", 1)[1].split(
        "    def _batch_progress_event", 1
    )[0]
    assert "process_batch(" in method
    assert "segment_slice(" not in method
    assert "filedialog.askdirectory(" in method
    assert "output_path_is_separate_from_source" in method


def test_batch_output_must_be_outside_source_tree(tmp_path):
    saturn = load_saturn_v571()
    source = tmp_path / "images"
    source.mkdir()

    assert not saturn.output_path_is_separate_from_source(source, source)
    assert not saturn.output_path_is_separate_from_source(
        source, source / "batch_output"
    )
    assert saturn.output_path_is_separate_from_source(
        source, tmp_path / "results"
    )
    assert not saturn.output_path_is_separate_from_source(
        source, tmp_path
    )


def test_profile_summary_exposes_version_and_provenance_hashes(tmp_path):
    saturn = load_saturn_v571()
    cfg = custom_profile(tmp_path, saturn)
    summary = saturn.analysis_profile_summary(cfg)

    assert f"Saturn {saturn._VERSION}" in summary
    assert "profile SHA " in summary
    assert "model SHA " in summary
    assert "calibration " in summary
    assert "scientific audit BLOCKED" in summary


def test_gui_exposes_metadata_repair_and_read_only_review_notice():
    source = (ROOT / "sperm_segmentation_saturnv5.7.1.py").read_text(
        encoding="utf-8"
    )
    assert "Locate Microscope Metadata XML" in source
    assert "Set Metadata XML" in source
    assert "Reference controls report direction" in source
    assert "Overlay review is read-only" in source
