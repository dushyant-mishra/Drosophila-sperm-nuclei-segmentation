"""The production gate must fail closed on every malformed registry shape.

A gate that raises instead of returning a verdict is not fail-closed: the caller
sees a crash rather than a refusal, and a crash in a different call path could be
caught and mistaken for something recoverable. Every input here must produce a
verdict, and only a genuinely complete registry may produce True.
"""

import json
from pathlib import Path

import pytest

from utils import saturn_v571_gui_services as SERVICES


ROOT = Path(__file__).resolve().parents[1]


def write_registry(root, payload):
    audits = root / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    (audits / "claims_registry.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return root


def accepted_claim(claim_id):
    return {
        "claim_id": claim_id,
        "status": "accepted",
        "latest_audit": {"gate_passed": True, "decision": "accepted"},
    }


def test_a_complete_registry_opens_the_gate(tmp_path):
    root = write_registry(
        tmp_path,
        {"claims": [accepted_claim(c) for c in SERVICES.PRODUCTION_REQUIRED_CLAIM_IDS]},
    )
    ready, detail = SERVICES.production_audit_gate_state(root)
    assert ready is True
    assert "accepted" in detail.lower()


def test_the_new_width_claim_is_required(tmp_path):
    """The width actually presented biologically must gate production."""
    assert "MEAS-INTENSITY-WIDTH-001" in SERVICES.PRODUCTION_REQUIRED_CLAIM_IDS
    required = [
        c for c in SERVICES.PRODUCTION_REQUIRED_CLAIM_IDS
        if c != "MEAS-INTENSITY-WIDTH-001"
    ]
    root = write_registry(tmp_path, {"claims": [accepted_claim(c) for c in required]})
    ready, detail = SERVICES.production_audit_gate_state(root)
    assert ready is False
    assert "MEAS-INTENSITY-WIDTH-001" in detail


@pytest.mark.parametrize(
    "payload",
    [
        [],                                   # a bare list, not a mapping
        None,                                 # JSON null
        "claims",                             # a bare string
        42,                                   # a bare number
        {"claims": "not-a-list"},             # claims of the wrong type
        {"claims": [None, 7, "x"]},           # non-mapping claim entries
        {},                                   # no claims key at all
    ],
)
def test_malformed_registry_shapes_return_a_verdict_rather_than_raising(
    tmp_path, payload
):
    root = write_registry(tmp_path, payload)
    ready, detail = SERVICES.production_audit_gate_state(root)
    assert ready is False
    assert isinstance(detail, str) and detail


@pytest.mark.parametrize("audit_value", ["accepted", ["accepted"], 1, 0.5])
def test_a_non_mapping_latest_audit_fails_closed(tmp_path, audit_value):
    """`latest_audit` of the wrong type must not be read as a passing gate."""
    claims = [accepted_claim(c) for c in SERVICES.PRODUCTION_REQUIRED_CLAIM_IDS]
    claims[0]["latest_audit"] = audit_value
    root = write_registry(tmp_path, {"claims": claims})
    ready, _ = SERVICES.production_audit_gate_state(root)
    assert ready is False


def test_missing_registry_file_fails_closed(tmp_path):
    ready, detail = SERVICES.production_audit_gate_state(tmp_path)
    assert ready is False
    assert "unavailable" in detail.lower()


def test_unparsable_registry_fails_closed(tmp_path):
    audits = tmp_path / "audits"
    audits.mkdir(parents=True)
    (audits / "claims_registry.json").write_text("{not json", encoding="utf-8")
    ready, detail = SERVICES.production_audit_gate_state(tmp_path)
    assert ready is False
    assert "unavailable" in detail.lower()


def test_a_missing_required_claim_is_named(tmp_path):
    required = list(SERVICES.PRODUCTION_REQUIRED_CLAIM_IDS)
    root = write_registry(
        tmp_path, {"claims": [accepted_claim(c) for c in required[1:]]}
    )
    ready, detail = SERVICES.production_audit_gate_state(root)
    assert ready is False
    assert required[0] in detail
    assert "missing" in detail


def test_partial_acceptance_blocks_and_names_every_blocker(tmp_path):
    required = list(SERVICES.PRODUCTION_REQUIRED_CLAIM_IDS)
    claims = [accepted_claim(c) for c in required]
    claims[1]["status"] = "implemented"
    claims[2]["latest_audit"] = None
    root = write_registry(tmp_path, {"claims": claims})
    ready, detail = SERVICES.production_audit_gate_state(root)
    assert ready is False
    assert required[1] in detail
    assert required[2] in detail


def test_accepted_status_with_a_failed_gate_is_still_blocked(tmp_path):
    """Status and gate verdict must both hold; neither alone is enough."""
    claims = [accepted_claim(c) for c in SERVICES.PRODUCTION_REQUIRED_CLAIM_IDS]
    claims[0]["latest_audit"] = {"gate_passed": False, "decision": "not_accepted"}
    root = write_registry(tmp_path, {"claims": claims})
    ready, detail = SERVICES.production_audit_gate_state(root)
    assert ready is False
    assert "not_accepted" in detail


def test_a_passing_gate_without_accepted_status_is_still_blocked(tmp_path):
    claims = [accepted_claim(c) for c in SERVICES.PRODUCTION_REQUIRED_CLAIM_IDS]
    claims[0]["status"] = "validated"
    root = write_registry(tmp_path, {"claims": claims})
    ready, _ = SERVICES.production_audit_gate_state(root)
    assert ready is False


def test_the_real_repository_registry_is_readable_and_currently_blocked():
    """A guard against the gate silently opening on the live registry."""
    ready, detail = SERVICES.production_audit_gate_state(ROOT)
    assert isinstance(ready, bool)
    assert isinstance(detail, str) and detail
    # Nothing has been through an acceptance audit yet, so it must be closed.
    assert ready is False
