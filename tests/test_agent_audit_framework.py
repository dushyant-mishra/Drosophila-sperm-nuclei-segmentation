import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "saturn_agent_audit_validator",
        ROOT / "scripts" / "validate_agent_audit.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_run(tmp_path, *, verdict="pass", mode="acceptance_candidate"):
    run = tmp_path / "run"
    role = "measurement_geometry"
    commit = "abcdef0123456789"
    registry = tmp_path / "registry.json"
    write_json(
        registry,
        {
            "schema_version": "1.0",
            "claims": [
                {
                    "claim_id": "TEST-001",
                    "required_roles": [role],
                }
            ],
        },
    )
    write_json(
        run / "manifest.json",
        {
            "schema_version": "1.0",
            "audit_run_id": "run-001",
            "claim_id": "TEST-001",
            "reviewed_commit": commit,
            "mode": mode,
            "dirty_worktree": mode != "acceptance_candidate",
            "roles": [role],
        },
    )
    write_json(
        run / "reviews" / f"{role}.json",
        {
            "schema_version": "1.0",
            "audit_run_id": "run-001",
            "claim_id": "TEST-001",
            "role": role,
            "reviewed_commit": commit,
            "verdict": verdict,
            "confidence": "high",
            "summary": "Independent check complete.",
            "checks": [
                {
                    "check_id": "geometry",
                    "status": "pass",
                    "evidence": ["tests/test_example.py::test_geometry"],
                }
            ],
            "findings": [],
            "limitations": [],
            "recommended_follow_up": [],
        },
    )
    return run, registry


def test_complete_independent_review_passes_gate(tmp_path):
    validator = load_validator()
    run, registry = make_run(tmp_path)

    result = validator.validate_run(run, registry)

    assert result["gate_passed"] is True
    assert result["required_roles_not_passed"] == []


def test_conditional_verdict_blocks_gate(tmp_path):
    validator = load_validator()
    run, registry = make_run(tmp_path, verdict="conditional")

    result = validator.validate_run(run, registry)

    assert result["gate_passed"] is False
    assert result["required_roles_not_passed"] == ["measurement_geometry"]


def test_dirty_precommit_review_is_diagnostic_only(tmp_path):
    validator = load_validator()
    run, registry = make_run(tmp_path, mode="pre_commit")

    result = validator.validate_run(run, registry)

    assert result["gate_passed"] is False
    assert any("diagnostic" in item for item in result["validation_errors"])


def test_missing_evidence_blocks_gate(tmp_path):
    validator = load_validator()
    run, registry = make_run(tmp_path)
    path = run / "reviews" / "measurement_geometry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checks"][0]["evidence"] = []
    write_json(path, payload)

    result = validator.validate_run(run, registry)

    assert result["gate_passed"] is False
    assert any("no evidence" in item for item in result["validation_errors"])


def test_blocking_finding_blocks_even_pass_verdict(tmp_path):
    validator = load_validator()
    run, registry = make_run(tmp_path)
    path = run / "reviews" / "measurement_geometry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["findings"] = [
        {
            "severity": "blocking",
            "title": "Formula mismatch",
            "impact": "Reported units are wrong.",
            "evidence": ["pipeline.py:10"],
            "recommendation": "Correct and rerun.",
        }
    ]
    write_json(path, payload)

    result = validator.validate_run(run, registry)

    assert result["gate_passed"] is False
    assert result["blocking_findings"][0]["title"] == "Formula mismatch"
