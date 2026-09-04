import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_entrypoint():
    path = ROOT / "scripts" / "generate_v571_biological_comparison.py"
    spec = importlib.util.spec_from_file_location("v571_report_entrypoint_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v571_report_requires_both_explicit_group_roles():
    module = load_entrypoint()
    with pytest.raises(SystemExit, match="requires explicit"):
        module._require_explicit_group_direction([])
    with pytest.raises(SystemExit, match="requires explicit"):
        module._require_explicit_group_direction(
            ["--reference-group", "Control"]
        )


def test_v571_report_accepts_neutral_explicit_group_names():
    module = load_entrypoint()
    module._require_explicit_group_direction(
        [
            "--reference-group",
            "Group_A",
            "--comparison-group=Group_B",
        ]
    )


def test_v571_report_gate_blocks_before_generator(monkeypatch):
    module = load_entrypoint()
    called = []
    monkeypatch.setattr(
        module,
        "production_audit_gate_state",
        lambda _root: (False, "MEAS-BODY-WIDTH-001 is not accepted"),
    )
    monkeypatch.setattr(module, "_generate_report_main", lambda: called.append(True))

    with pytest.raises(SystemExit, match="scientific audit gate"):
        module._main(
            [
                "--reference-group",
                "Group_A",
                "--comparison-group",
                "Group_B",
            ]
        )

    assert called == []


def test_v571_direct_report_rejects_partial_cohort(tmp_path):
    module = load_entrypoint()
    output = tmp_path / "study"
    output.mkdir()
    (output / "study_run_state.json").write_text(
        '{"run_status":"complete_with_failures","samples":{}}',
        encoding="utf-8",
    )
    (output / "study_manifest.csv").write_text(
        "include,sample_id\nTrue,sample-01\n", encoding="utf-8"
    )

    with pytest.raises(SystemExit, match="fully complete validated cohort"):
        module._require_complete_cohort(["--study-output", str(output)])


def test_v571_entrypoint_forces_concise_metric_profile(monkeypatch):
    module = load_entrypoint()
    calls = []
    monkeypatch.setattr(module, "_require_production_audit_gate", lambda: None)
    monkeypatch.setattr(module, "_require_complete_cohort", lambda _args: None)
    monkeypatch.setattr(module, "_generate_report_main", lambda args: calls.append(args))
    arguments = [
        "--study-output",
        "study",
        "--reference-group",
        "Group_A",
        "--comparison-group",
        "Group_B",
    ]

    module._main(arguments)

    assert calls == [arguments + ["--metric-profile", "concise_v571"]]
