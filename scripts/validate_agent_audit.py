#!/usr/bin/env python3
"""Validate independent Saturn audit outputs and compute the release gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VERDICTS = {"pass", "conditional", "fail", "abstain"}
SEVERITIES = {"blocking", "high", "medium", "low", "note"}


def _load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _claim_by_id(registry, claim_id):
    matches = [item for item in registry.get("claims", []) if item.get("claim_id") == claim_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one registry claim {claim_id!r}; found {len(matches)}")
    return matches[0]


def validate_review(payload, *, manifest, role):
    errors = []
    required = {
        "schema_version", "audit_run_id", "claim_id", "role",
        "reviewed_commit", "verdict", "confidence", "summary", "checks",
        "findings", "limitations", "recommended_follow_up",
    }
    missing = sorted(required - set(payload))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
        return errors
    expected = {
        "schema_version": "1.0",
        "audit_run_id": manifest["audit_run_id"],
        "claim_id": manifest["claim_id"],
        "role": role,
        "reviewed_commit": manifest["reviewed_commit"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            errors.append(f"{key}={payload.get(key)!r}; expected {value!r}")
    if payload.get("verdict") not in VERDICTS:
        errors.append(f"invalid verdict: {payload.get('verdict')!r}")
    if not isinstance(payload.get("checks"), list) or not payload["checks"]:
        errors.append("checks must be a non-empty list")
    else:
        for index, check in enumerate(payload["checks"]):
            if not check.get("check_id"):
                errors.append(f"check {index} has no check_id")
            if check.get("status") in {"pass", "fail"} and not check.get("evidence"):
                errors.append(f"check {index} has no evidence")
    if not isinstance(payload.get("findings"), list):
        errors.append("findings must be a list")
    else:
        for index, finding in enumerate(payload["findings"]):
            if finding.get("severity") not in SEVERITIES:
                errors.append(f"finding {index} has invalid severity")
            if not finding.get("evidence"):
                errors.append(f"finding {index} has no evidence")
    return errors


def validate_run(run_dir: Path, registry_path: Path):
    manifest = _load_json(run_dir / "manifest.json")
    registry = _load_json(registry_path)
    claim = _claim_by_id(registry, manifest["claim_id"])
    required_roles = list(claim.get("required_roles", []))
    declared_roles = list(manifest.get("roles", []))
    errors = []
    reviews = {}

    missing_roles = sorted(set(required_roles) - set(declared_roles))
    if missing_roles:
        errors.append(f"manifest missing required roles: {', '.join(missing_roles)}")
    duplicate_roles = sorted({role for role in declared_roles if declared_roles.count(role) > 1})
    if duplicate_roles:
        errors.append(f"duplicate roles: {', '.join(duplicate_roles)}")

    for role in declared_roles:
        path = run_dir / "reviews" / f"{role}.json"
        if not path.exists():
            errors.append(f"missing review: {path}")
            continue
        try:
            payload = _load_json(path)
        except Exception as exc:
            errors.append(f"invalid JSON for {role}: {exc}")
            continue
        role_errors = validate_review(payload, manifest=manifest, role=role)
        errors.extend(f"{role}: {message}" for message in role_errors)
        reviews[role] = payload

    if manifest.get("mode") != "acceptance_candidate":
        errors.append("pre-commit audits are diagnostic and cannot pass the acceptance gate")
    if manifest.get("mode") == "acceptance_candidate" and manifest.get("dirty_worktree"):
        errors.append("acceptance candidate records a dirty working tree")

    blocking = []
    required_not_passed = []
    for role in required_roles:
        review = reviews.get(role)
        if not review or review.get("verdict") != "pass":
            required_not_passed.append(role)
        if review:
            for finding in review.get("findings", []):
                if finding.get("severity") == "blocking":
                    blocking.append({"role": role, "title": finding.get("title", "")})

    gate_passed = not errors and not blocking and not required_not_passed
    summary = {
        "schema_version": "1.0",
        "audit_run_id": manifest["audit_run_id"],
        "claim_id": manifest["claim_id"],
        "reviewed_commit": manifest["reviewed_commit"],
        "mode": manifest.get("mode"),
        "gate_passed": gate_passed,
        "required_roles": required_roles,
        "received_roles": sorted(reviews),
        "verdicts": {role: review.get("verdict") for role, review in sorted(reviews.items())},
        "required_roles_not_passed": required_not_passed,
        "blocking_findings": blocking,
        "validation_errors": errors,
    }
    with (run_dir / "gate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "audits" / "claims_registry.json",
    )
    args = parser.parse_args()
    summary = validate_run(args.run.resolve(), args.registry.resolve())
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["gate_passed"] else 1)


if __name__ == "__main__":
    main()
