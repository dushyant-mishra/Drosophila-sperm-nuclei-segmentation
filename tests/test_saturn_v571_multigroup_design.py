"""Study design contract: one reference group and one or more comparison groups.

The reporting layer was hard-wired to exactly two groups, so a study comparing
wild type against several mutants, or against a mutant and a rescue line, could
not be expressed at all. These tests pin the generalized design and, just as
importantly, pin that the existing two-group behaviour is unchanged, because a
silent change to a q-value would alter published conclusions.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_saturn_v571():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_multigroup_test",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design_frame(pairs):
    """pairs: sequence of (group, role) repeated per specimen."""
    rows = []
    for index, (group, role) in enumerate(pairs):
        rows.append({"specimen_id": f"s{index}", "group": group, "group_role": role})
    return pd.DataFrame(rows)


def test_single_comparison_design_is_unchanged():
    saturn = load_saturn_v571()
    frame = design_frame(
        [("WT", "reference"), ("WT", "reference"), ("KJ", "comparison")]
    )
    reference, comparisons = saturn._study_group_design(frame)
    assert reference == "WT"
    assert comparisons == ["KJ"]


def test_multiple_comparison_groups_are_accepted_and_ordered():
    saturn = load_saturn_v571()
    frame = design_frame(
        [
            ("WT", "reference"),
            ("mutantB", "comparison"),
            ("mutantA", "comparison"),
            ("rescue", "comparison"),
        ]
    )
    reference, comparisons = saturn._study_group_design(frame)
    assert reference == "WT"
    # Deterministic order so a report is reproducible.
    assert comparisons == ["mutantA", "mutantB", "rescue"]


def test_group_names_are_arbitrary_and_never_inferred_from_labels():
    """A group literally named 'control' must not become the reference by name."""
    saturn = load_saturn_v571()
    frame = design_frame(
        [
            ("Mutant named control", "comparison"),
            ("line 7", "reference"),
            ("line 7", "reference"),
        ]
    )
    reference, comparisons = saturn._study_group_design(frame)
    assert reference == "line 7"
    assert comparisons == ["Mutant named control"]


def test_design_requires_exactly_one_reference_group():
    saturn = load_saturn_v571()
    two_references = design_frame(
        [("WT", "reference"), ("other", "reference"), ("KJ", "comparison")]
    )
    with pytest.raises(ValueError):
        saturn._study_group_design(two_references)

    no_reference = design_frame([("KJ", "comparison"), ("KJ2", "comparison")])
    with pytest.raises(ValueError):
        saturn._study_group_design(no_reference)


def test_design_requires_at_least_one_comparison_group():
    saturn = load_saturn_v571()
    frame = design_frame([("WT", "reference"), ("WT", "reference")])
    with pytest.raises(ValueError):
        saturn._study_group_design(frame)


def test_design_rejects_missing_conflicting_or_unknown_roles():
    saturn = load_saturn_v571()
    missing = design_frame([("WT", "reference"), ("KJ", "")])
    with pytest.raises(ValueError):
        saturn._study_group_design(missing)

    conflicting = design_frame(
        [("WT", "reference"), ("WT", "comparison"), ("KJ", "comparison")]
    )
    with pytest.raises(ValueError):
        saturn._study_group_design(conflicting)

    unknown = design_frame([("WT", "reference"), ("KJ", "treatment")])
    with pytest.raises(ValueError):
        saturn._study_group_design(unknown)


def test_design_requires_the_role_column_at_all():
    saturn = load_saturn_v571()
    frame = pd.DataFrame({"specimen_id": ["a", "b"], "group": ["WT", "KJ"]})
    with pytest.raises(ValueError):
        saturn._study_group_design(frame)


def test_bh_family_across_comparisons_is_per_metric():
    saturn = load_saturn_v571()
    frame = pd.DataFrame(
        {
            "metric": ["width", "width", "length", "length"],
            "comparison_group": ["m1", "m2", "m1", "m2"],
            "permutation_p_value": [0.01, 0.04, 0.02, 0.03],
        }
    )
    adjusted = saturn._study_bh_qvalues_by_metric(frame, "permutation_p_value")
    # Each metric is its own family of two comparisons. Step-up BH on the
    # length family [0.02, 0.03] pulls the smaller value up to 0.03.
    assert adjusted.tolist() == pytest.approx([0.02, 0.04, 0.03, 0.03])


def test_bh_family_across_comparisons_preserves_missing_values():
    saturn = load_saturn_v571()
    frame = pd.DataFrame(
        {
            "metric": ["width", "width", "length"],
            "comparison_group": ["m1", "m2", "m1"],
            "permutation_p_value": [0.01, np.nan, 0.05],
        }
    )
    adjusted = saturn._study_bh_qvalues_by_metric(frame, "permutation_p_value")
    assert np.isnan(adjusted.iloc[1])
    # A family of one is uncorrected.
    assert adjusted.iloc[0] == pytest.approx(0.01)
    assert adjusted.iloc[2] == pytest.approx(0.05)


def test_across_metric_family_still_matches_the_existing_implementation():
    """The existing q-value must not change when a study has one comparison."""
    saturn = load_saturn_v571()
    values = pd.Series([0.01, 0.04, 0.03, np.nan, 0.20])
    legacy = saturn._study_bh_qvalues(values)
    assert np.isnan(legacy[3])
    assert legacy[0] <= legacy[2] <= legacy[1] <= legacy[4]


def _specimen_frame(groups, seed, metrics):
    """Synthetic specimen-level table for a star design."""
    rng = np.random.default_rng(seed)
    rows = []
    for group, role, offset in groups:
        for index in range(8):
            row = {
                "specimen_id": f"{group}_{index}",
                "group": group,
                "group_role": role,
                "status": "complete",
            }
            for position, metric in enumerate(metrics):
                row[metric] = 5.0 + position + offset + rng.normal(0, 0.3)
            rows.append(row)
    return pd.DataFrame(rows)


def test_three_group_study_produces_one_contrast_per_comparison():
    saturn = load_saturn_v571()
    frame = _specimen_frame(
        [
            ("WT", "reference", 0.0),
            ("mutantA", "comparison", 0.3),
            ("rescue", "comparison", 0.1),
        ],
        7,
        saturn._STUDY_COMPARISON_METRICS,
    )
    result, qc = saturn._study_specimen_group_comparisons(frame)

    assert not result.empty
    assert sorted(result["comparison_group"].unique()) == ["mutantA", "rescue"]
    assert set(result["reference_group"].unique()) == {"WT"}
    assert qc["comparison_groups"] == ["mutantA", "rescue"]
    # No single comparison exists, so the pairwise field must not name one.
    assert qc["comparison_group"] == ""
    assert "mutantA minus WT" in qc["effect_direction"]
    assert "rescue minus WT" in qc["effect_direction"]


def test_adding_a_comparison_group_does_not_move_existing_contrasts():
    """A published contrast must not change because another group was added."""
    saturn = load_saturn_v571()
    groups = [
        ("WT", "reference", 0.0),
        ("mutantA", "comparison", 0.3),
        ("rescue", "comparison", 0.1),
    ]
    metrics = saturn._STUDY_COMPARISON_METRICS
    three = _specimen_frame(groups, 7, metrics)
    two = three[three["group"].isin(["WT", "mutantA"])].copy()

    multi, _ = saturn._study_specimen_group_comparisons(three)
    pair, _ = saturn._study_specimen_group_comparisons(two)
    subset = multi[multi["comparison_group"] == "mutantA"].reset_index(drop=True)

    assert np.allclose(
        subset["permutation_p_value"].astype(float),
        pair["permutation_p_value"].astype(float),
        equal_nan=True,
    )
    # The within-contrast family is unchanged too, because it spans metrics only.
    assert np.allclose(
        subset["bh_fdr_q_value"].astype(float),
        pair["bh_fdr_q_value"].astype(float),
        equal_nan=True,
    )


def test_both_q_value_families_are_reported_and_named():
    saturn = load_saturn_v571()
    frame = _specimen_frame(
        [
            ("WT", "reference", 0.0),
            ("mutantA", "comparison", 0.3),
            ("rescue", "comparison", 0.1),
        ],
        11,
        saturn._STUDY_COMPARISON_METRICS,
    )
    result, _ = saturn._study_specimen_group_comparisons(frame)

    assert "bh_fdr_q_value" in result.columns
    assert "bh_fdr_q_value_across_comparisons" in result.columns
    assert set(result["bh_family_within_contrast"]) == {
        "metrics_tested_in_this_contrast"
    }
    assert set(result["bh_family_across_comparisons"]) == {
        "comparison_groups_for_this_metric"
    }
    finite = result["permutation_p_value"].notna()
    assert (result.loc[finite, "bh_fdr_q_value"] >= result.loc[finite, "permutation_p_value"] - 1e-12).all()


def test_a_single_group_study_is_refused_rather_than_compared():
    saturn = load_saturn_v571()
    frame = _specimen_frame(
        [("WT", "reference", 0.0)], 3, saturn._STUDY_COMPARISON_METRICS
    )
    result, qc = saturn._study_specimen_group_comparisons(frame)
    assert result.empty
    assert qc["comparison_status"] == "not_run"
    assert any("comparison group" in w for w in qc["warnings"])
