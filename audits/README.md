# Saturn Independent Audit Framework

This directory turns scientific review into durable, machine-checkable
evidence. Each reviewer runs in a separate read-only Codex session and returns
a structured verdict. The purpose is not to produce consensus by repetition;
it is to expose different failure modes before a measurement or workflow is
treated as production-ready.

## Reviewer roles

| Role | Primary responsibility |
| --- | --- |
| `measurement_geometry` | Formula, coordinate system, units, pixel-grid effects, synthetic geometry, legacy compatibility |
| `biological_validity` | Whether technical rules preserve plausible WT and mutant phenotypes without encoding expected biology |
| `calibration_provenance` | Leica metadata, ROI meaning, dimensional alignment, source identity, checkpoint/profile provenance |
| `software_reproducibility` | Runtime paths, deterministic behavior, tests, schema compatibility, failure handling, cross-entry-point consistency |
| `statistics_reporting` | Statistical unit, estimand, denominator, plot/table/PDF agreement, uncertainty, understandable nomenclature |
| `visual_evidence` | Overlay truthfulness, mask/centerline alignment, representative examples, clipping, legends, visual failure cases |
| `repository_release` | Git status, committed scope, remote synchronization, versioning, documentation, immutable release evidence |

The role charters live under `audits/agent_charters/`.

## Claims registry

`claims_registry.json` is the canonical list of scientific and engineering
claims. A claim includes:

- its biological meaning and exact computation;
- units, population, calibration and source dependencies;
- known limitations and explicit non-claims;
- acceptance criteria;
- required independent roles;
- links to implementation and validation evidence;
- current lifecycle status.

Status values are `proposed`, `implemented`, `validated`, `accepted`,
`rejected`, `deprecated`, and `superseded`. Only the audit gate may move a
high-risk claim from `validated` to `accepted`.

## Running an audit

From PowerShell at the repository root:

```powershell
.\scripts\run_multi_agent_audit.ps1 `
  -ClaimId MEAS-BODY-WIDTH-001 `
  -RunId 20260821-body-width-independent-review
```

Reviewers are launched as independent, read-only, ephemeral Codex sessions.
Their JSON outputs are written to:

```text
audits/runs/<run-id>/reviews/<role>.json
```

The launcher snapshots the Git commit, working-tree state, claim definition,
roles, prompts, and timestamps. It does not edit production code, commit, push,
or alter claim status.

Validate the completed run:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_agent_audit.py `
  --run audits\runs\20260821-body-width-independent-review
```

Use `-Parallel` to run reviewers concurrently. Use `-AllowDirty` only for an
explicit pre-commit review; acceptance audits should use a clean commit.

## Decision process

1. The coordinator defines or updates the claim and acceptance criteria.
2. The implementation owner supplies evidence but does not approve the claim.
3. Required agents review independently without seeing other verdicts.
4. The validator checks role coverage, evidence, commit identity, and blocking
   findings.
5. The coordinator writes `decision.json` from the template only after the
   gate passes or records why the claim remains blocked.
6. The repository/release steward verifies the accepted decision before any
   publication action.

Disagreement is useful evidence. Do not average conflicting conclusions. Open
a new issue/claim, collect the missing evidence, and rerun the affected roles.

## Release gate

A production release must have:

- a clean, identified Git commit;
- all high-risk claims accepted or explicitly deferred as non-production;
- passing automated tests and audit validation;
- exact profile and checkpoint identity, including hashes;
- report-to-table consistency evidence;
- a repository/release steward verdict;
- explicit authorization before commit, push, merge, or tag operations.
