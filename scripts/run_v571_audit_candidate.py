"""Generate v5.7.1 acceptance evidence while the production gate is closed.

This entry point is intentionally separate from the GUI and production CLI.
Its outputs are audit candidates, never production biological results.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "sperm_segmentation_saturnv5.7.1.py"
DEFAULT_PROFILE = (
    ROOT / "production_profiles" / "saturn_v5_7_1_model_c_epoch003.json"
)
DEFAULT_REGISTRY = ROOT / "audits" / "claims_registry.json"
REQUIRED_ACKNOWLEDGEMENT = (
    "I understand this bypasses the closed production gate for audit evidence only"
)
AUDIT_TABLES = (
    "specimen_summary.csv",
    "study_track_records.csv",
    "specimen_technical_qc.csv",
    "group_summary.csv",
    "specimen_group_comparisons.csv",
    "common_depth_sensitivity.csv",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("saturn_v571_audit", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config(module, profile_path, checkpoint_path, lean_output):
    cfg, _ = module.load_analysis_profile(Path(profile_path), module.CONFIG)
    cfg.update(
        {
            "RUN_MODE": "batch",
            "ANALYSIS_MODE": "comparative",
            "SEGMENTATION_ENGINE": "unet_primary",
            "TRACKING_BACKEND": "global_assignment",
            "AUTO_LEICA_CALIBRATION": True,
            "DO_TRACKING": True,
            "SHOW_PREVIEW_WINDOW": False,
            "UNET_MODEL_PATH": str(Path(checkpoint_path).resolve()),
        }
    )
    if lean_output:
        cfg.update(
            {
                "SAVE_DETAIL_FIGURE": False,
                "SAVE_MASK_TIFS": False,
                "SAVE_LABEL_TIFS": True,
                "UNET_SAVE_PROBABILITY_MAPS": False,
                "SAVE_TECHNICAL_REVIEW_OVERLAYS": False,
                "REPORT_MAX_SLICE_PAGES": 6,
            }
        )
    module.validate_analysis_runtime_config(cfg)
    return cfg


def _git_identity():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return commit, bool(tracked_status), tracked_status.splitlines()


def _stamp_audit_tables(output_root):
    stamped = []
    empty = []
    for name in AUDIT_TABLES:
        path = Path(output_root) / name
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            empty.append(path)
            continue
        frame["audit_candidate_only"] = True
        frame["production_gate_status"] = "closed"
        frame.to_csv(path, index=False)
        stamped.append(path)
    return stamped, empty


def run_audit_candidate(arguments, pipeline=None, registry_path=DEFAULT_REGISTRY):
    """Run an explicitly acknowledged, provenance-bound audit candidate."""
    acknowledgement = str(arguments.get("acknowledgement", ""))
    if acknowledgement != REQUIRED_ACKNOWLEDGEMENT:
        raise ValueError(
            "Audit execution requires the exact acknowledgement: "
            + REQUIRED_ACKNOWLEDGEMENT
        )

    output_root = Path(arguments["output_root"]).resolve()
    if "audit_candidate_only" not in output_root.name.lower():
        raise ValueError("The output directory name must contain AUDIT_CANDIDATE_ONLY")
    profile_path = Path(arguments["params"]).resolve()
    checkpoint_path = Path(arguments["checkpoint"]).resolve()
    registry_path = Path(registry_path).resolve()
    for label, path in (
        ("analysis profile", profile_path),
        ("U-Net checkpoint", checkpoint_path),
        ("claims registry", registry_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    module = _load_pipeline() if pipeline is None else pipeline
    gate_ready, gate_detail = module.production_audit_gate_state()
    if gate_ready:
        raise RuntimeError(
            "The production gate is open; use the ordinary study runner instead."
        )

    registry_before = _sha256(registry_path)
    commit, tracked_dirty, tracked_changes = (
        _git_identity() if pipeline is None else ("test-double", False, [])
    )
    if tracked_dirty:
        raise RuntimeError(
            "Audit evidence requires a clean tracked worktree. Commit or revert: "
            + "; ".join(tracked_changes)
        )

    cfg = (
        _load_config(
            module,
            profile_path,
            checkpoint_path,
            bool(arguments.get("lean_output", False)),
        )
        if pipeline is None
        else {
            "_ACTIVE_PROFILE_PATH": str(profile_path),
            "UNET_MODEL_PATH": str(checkpoint_path),
        }
    )
    rows = module.discover_multisample_study(arguments["study_root"], base_cfg=cfg)
    selected = set(arguments.get("sample_id", []))
    if selected:
        discovered = {row["sample_id"] for row in rows}
        missing = sorted(selected - discovered)
        if missing:
            raise ValueError(f"Unknown sample IDs: {', '.join(missing)}")
        for row in rows:
            row["include"] = row["sample_id"] in selected

    output_root.mkdir(parents=True, exist_ok=True)
    record_path = output_root / "AUDIT_CANDIDATE_ONLY.json"
    record = {
        "schema_version": "1.0",
        "audit_candidate_only": True,
        "production_use_prohibited": True,
        "acknowledgement": acknowledgement,
        "production_gate_ready": False,
        "production_gate_detail": gate_detail,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "tracked_worktree_dirty": False,
        "pipeline_path": str(PIPELINE_PATH),
        "pipeline_sha256": _sha256(PIPELINE_PATH),
        "profile_path": str(profile_path),
        "profile_sha256": _sha256(profile_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "claims_registry_path": str(registry_path),
        "claims_registry_sha256_before": registry_before,
        "study_root": str(Path(arguments["study_root"]).resolve()),
        "output_root": str(output_root),
        "selected_sample_ids": sorted(selected),
        "bypassed_operations": [],
    }
    _atomic_json(record_path, record)

    original_gate = module.require_production_audit_gate

    def acknowledge_bypass(operation):
        allowed = {"Multi-sample study", "Batch analysis"}
        if operation not in allowed:
            raise RuntimeError(f"Audit runner cannot bypass operation: {operation}")
        record["bypassed_operations"].append(operation)

    module.require_production_audit_gate = acknowledge_bypass
    try:
        state, _summary = module.run_multisample_study(
            rows,
            output_root,
            base_cfg=cfg,
            resume=not bool(arguments.get("no_resume", False)),
            study_root=arguments["study_root"],
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(record_path, record)
        raise
    finally:
        module.require_production_audit_gate = original_gate

    run_status = str(state.get("run_status", "unknown"))
    try:
        stamped, empty = _stamp_audit_tables(output_root)
        registry_after = _sha256(registry_path)
        if registry_after != registry_before:
            raise RuntimeError(
                "The claims registry changed during audit-candidate execution"
            )
        settings_manifest = output_root / "settings" / "settings_manifest.json"
        if not settings_manifest.is_file():
            raise RuntimeError("The study did not emit its provenance settings manifest")

        record.update(
            {
                "status": run_status,
                "acceptance_evidence_ready": run_status == "complete",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "claims_registry_sha256_after": registry_after,
                "settings_manifest_sha256": _sha256(settings_manifest),
                "stamped_table_sha256": {
                    path.relative_to(output_root).as_posix(): _sha256(path)
                    for path in stamped
                },
                "empty_audit_table_sha256": {
                    path.relative_to(output_root).as_posix(): _sha256(path)
                    for path in empty
                },
            }
        )
        _atomic_json(record_path, record)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "study_run_status": run_status,
                "acceptance_evidence_ready": False,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(record_path, record)
        raise

    if run_status != "complete":
        raise RuntimeError(
            f"Audit candidate did not complete successfully: {run_status}"
        )
    return record


def _parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--params", default=str(DEFAULT_PROFILE))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--lean-output", action="store_true")
    parser.add_argument("--acknowledgement", required=True)
    return vars(parser.parse_args(arguments))


def main(arguments=None):
    record = run_audit_candidate(_parse_args(arguments))
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
