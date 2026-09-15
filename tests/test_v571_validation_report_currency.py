"""The validation report must not contradict the live gate or the live registry.

A validation report is read as a statement of what the pipeline currently does.
When a measurement definition changes and the report is not updated, the stale
numbers keep circulating and get cited. These checks are deliberately about
agreement with the repository's own durable records rather than about exact
figures, so they do not break every time a test is added.
"""

import json
from pathlib import Path

from utils.saturn_v571_gui_services import production_audit_gate_state


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "V5_7_1_VALIDATION_REPORT.md"
REGISTRY = ROOT / "audits" / "claims_registry.json"


def report_text():
    return REPORT.read_text(encoding="utf-8")


def unaccepted_required_claims():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    claims = {
        str(claim.get("claim_id", "")): claim
        for claim in registry.get("claims", [])
        if isinstance(claim, dict)
    }
    from utils.saturn_v571_gui_services import PRODUCTION_REQUIRED_CLAIM_IDS

    blocked = []
    for claim_id in PRODUCTION_REQUIRED_CLAIM_IDS:
        claim = claims.get(claim_id, {})
        audit = claim.get("latest_audit")
        audit = audit if isinstance(audit, dict) else {}
        if not (
            str(claim.get("status", "")).lower() == "accepted"
            and bool(audit.get("gate_passed", False))
        ):
            blocked.append(claim_id)
    return blocked


def test_the_report_exists_and_is_not_empty():
    assert REPORT.is_file()
    assert len(report_text()) > 500


def test_a_closed_gate_is_disclosed_in_the_report():
    """A reader must not take the report as evidence production is unblocked."""
    ready, _ = production_audit_gate_state(ROOT)
    text = report_text().lower()
    if not ready:
        assert "gate is closed" in text or "not accepted" in text, (
            "the production gate is closed but the validation report does not say so"
        )


def test_every_currently_blocking_claim_is_named_in_the_report():
    blocked = unaccepted_required_claims()
    text = report_text()
    for claim_id in blocked:
        assert claim_id in text, (
            f"{claim_id} blocks production but is not mentioned in the "
            "validation report"
        )


def test_superseded_measurement_sections_are_marked():
    """Changed definitions must be flagged where the old numbers appear."""
    text = report_text()
    assert "Superseded" in text
    # The replay table and the width section are the two that changed.
    assert text.count("Superseded") >= 2


def test_the_report_does_not_present_mask_width_as_the_primary_biological_width():
    """Mask chord width is a technical diagnostic once the signal width exists."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    ids = {
        str(claim.get("claim_id", ""))
        for claim in registry.get("claims", [])
        if isinstance(claim, dict)
    }
    if "MEAS-INTENSITY-WIDTH-001" not in ids:
        return
    text = report_text()
    assert "The primary track width is the subpixel perpendicular contour-chord" not in text
    assert "signal-profile" in text.lower() or "half maximum" in text.lower()


def test_the_absolute_diameter_caveat_appears_wherever_width_is_reported():
    text = report_text().lower()
    assert "absolute nucleus diameter is not established" in text


def test_the_report_points_at_the_current_handover_state():
    """A stale report should at least route a reader to what is current."""
    text = report_text()
    handover = ROOT / "docs" / "plans" / "2026-09-14-v571-handover-state.md"
    assert handover.is_file()
    assert "2026-09-14-v571-handover-state.md" in text
