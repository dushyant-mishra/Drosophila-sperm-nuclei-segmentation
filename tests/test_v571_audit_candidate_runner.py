import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_v571_audit_candidate.py"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "v571_audit_candidate_runner_test", RUNNER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakePipeline:
    def __init__(self, registry_path, profile_path, checkpoint_path):
        self.PROJECT_ROOT = ROOT
        self._registry_path = Path(registry_path)
        self._profile_path = Path(profile_path)
        self._checkpoint_path = Path(checkpoint_path)
        self.gate_calls = []
        self.require_production_audit_gate = self._closed_gate

    @staticmethod
    def _closed_gate(operation):
        raise RuntimeError(f"{operation} is blocked")

    @staticmethod
    def production_audit_gate_state():
        return False, "synthetic closed gate"

    def discover_multisample_study(self, _study_root, base_cfg=None):
        return [
            {
                "sample_id": "WT-01",
                "group": "WT",
                "include": True,
                "slice_count": 2,
                "xy_um_per_pixel": 0.5,
                "z_um_per_slice": 1.0,
            }
        ]

    def run_multisample_study(self, rows, output_root, **_kwargs):
        self.require_production_audit_gate("Multi-sample study")
        self.require_production_audit_gate("Batch analysis")
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"sample_id": "WT-01", "group": "WT", "status": "complete"}]
        ).to_csv(output_root / "specimen_summary.csv", index=False)
        pd.DataFrame([{"sample_id": "WT-01", "track_id": 1}]).to_csv(
            output_root / "study_track_records.csv", index=False
        )
        settings = output_root / "settings"
        settings.mkdir()
        (settings / "settings_manifest.json").write_text(
            '{"files": []}', encoding="utf-8"
        )
        return {"run_status": "complete"}, pd.DataFrame()


def make_args(tmp_path, acknowledgement):
    profile = tmp_path / "profile.json"
    checkpoint = tmp_path / "checkpoint.pt"
    registry = tmp_path / "claims_registry.json"
    profile.write_text("{}", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    registry.write_text('{"claims": []}', encoding="utf-8")
    return (
        profile,
        checkpoint,
        registry,
        {
            "study_root": tmp_path / "study",
            "output_root": tmp_path / "run_AUDIT_CANDIDATE_ONLY",
            "params": profile,
            "checkpoint": checkpoint,
            "sample_id": ["WT-01"],
            "acknowledgement": acknowledgement,
            "no_resume": True,
            "lean_output": True,
        },
    )


def test_runner_refuses_without_exact_acknowledgement_before_writing(tmp_path):
    runner = load_runner()
    profile, checkpoint, registry, arguments = make_args(tmp_path, "yes")
    pipeline = FakePipeline(registry, profile, checkpoint)

    with pytest.raises(ValueError, match="exact acknowledgement"):
        runner.run_audit_candidate(arguments, pipeline=pipeline, registry_path=registry)

    assert not arguments["output_root"].exists()


def test_runner_records_bypass_stamps_tables_and_preserves_registry(tmp_path):
    runner = load_runner()
    profile, checkpoint, registry, arguments = make_args(
        tmp_path, runner.REQUIRED_ACKNOWLEDGEMENT
    )
    pipeline = FakePipeline(registry, profile, checkpoint)
    original_gate = pipeline.require_production_audit_gate
    registry_before = sha256(registry)

    record = runner.run_audit_candidate(
        arguments, pipeline=pipeline, registry_path=registry
    )

    assert pipeline.require_production_audit_gate == original_gate
    assert sha256(registry) == registry_before
    assert record["audit_candidate_only"] is True
    assert record["production_gate_ready"] is False
    assert record["production_gate_detail"] == "synthetic closed gate"
    assert record["claims_registry_sha256_before"] == registry_before
    assert record["claims_registry_sha256_after"] == registry_before
    assert record["profile_sha256"] == sha256(profile)
    assert record["checkpoint_sha256"] == sha256(checkpoint)
    assert record["settings_manifest_sha256"] == sha256(
        arguments["output_root"] / "settings" / "settings_manifest.json"
    )
    assert record["bypassed_operations"] == ["Multi-sample study", "Batch analysis"]

    saved = json.loads(
        (arguments["output_root"] / "AUDIT_CANDIDATE_ONLY.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved == record
    for name in ("specimen_summary.csv", "study_track_records.csv"):
        table = pd.read_csv(arguments["output_root"] / name)
        assert table["audit_candidate_only"].eq(True).all()
        assert table["production_gate_status"].eq("closed").all()


def test_runner_is_not_referenced_by_normal_gui_or_production_cli():
    runner_name = RUNNER_PATH.name
    for path in (
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
        ROOT / "scripts" / "run_v571_study.py",
        ROOT / "scripts" / "generate_v571_biological_comparison.py",
    ):
        assert runner_name not in path.read_text(encoding="utf-8")
