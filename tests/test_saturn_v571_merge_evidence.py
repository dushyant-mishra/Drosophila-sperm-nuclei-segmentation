"""Merge evidence must not be gated behind a length threshold.

The flag previously required an object to be both longer than 20 um and
branched. Median instance length is about 8 um, so the length gate almost never
opened and objectively branched structures were counted as single nuclei. These
tests pin that branching alone is sufficient evidence, with a tolerance so a
single skeletonisation spur does not reclassify a valid nucleus.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_saturn_v571():
    spec = importlib.util.spec_from_file_location(
        "saturn_v571_merge_evidence_test",
        ROOT / "sperm_segmentation_saturnv5.7.1.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_branching_alone_is_sufficient_merge_evidence():
    """A short but clearly branched structure is still several objects."""
    saturn = load_saturn_v571()
    # The KJ panel that prompted this: 19 branch nodes at 18.91 um, previously
    # unflagged because it fell under the 20 um gate.
    assert saturn.suspected_multi_object_merge_evidence(
        geodesic_um=18.91, branch_count=19, profile_merge=False
    )
    # The WT panel: 8 branch nodes at 6.24 um.
    assert saturn.suspected_multi_object_merge_evidence(
        geodesic_um=6.24, branch_count=8, profile_merge=False
    )


def test_a_single_spur_does_not_reclassify_a_valid_nucleus():
    """One or two branch nodes can be a skeletonisation artifact."""
    saturn = load_saturn_v571()
    assert not saturn.suspected_multi_object_merge_evidence(
        geodesic_um=8.0, branch_count=1, profile_merge=False
    )
    assert not saturn.suspected_multi_object_merge_evidence(
        geodesic_um=8.0, branch_count=2, profile_merge=False
    )
    assert saturn.suspected_multi_object_merge_evidence(
        geodesic_um=8.0, branch_count=3, profile_merge=False
    )


def test_profile_bimodality_is_independent_evidence():
    """Two nuclei side by side are unbranched but still two objects."""
    saturn = load_saturn_v571()
    assert saturn.suspected_multi_object_merge_evidence(
        geodesic_um=6.0, branch_count=0, profile_merge=True
    )


def test_a_clean_unbranched_nucleus_is_never_flagged():
    saturn = load_saturn_v571()
    assert not saturn.suspected_multi_object_merge_evidence(
        geodesic_um=7.91, branch_count=0, profile_merge=False
    )
    # Length alone is morphology, never merge evidence, per the design ledger.
    assert not saturn.suspected_multi_object_merge_evidence(
        geodesic_um=45.0, branch_count=0, profile_merge=False
    )


def test_spur_tolerance_is_configurable_and_validated():
    saturn = load_saturn_v571()
    assert saturn.suspected_multi_object_merge_evidence(
        geodesic_um=8.0, branch_count=2, profile_merge=False, min_branch_nodes=2
    )
    with pytest.raises(ValueError):
        saturn.suspected_multi_object_merge_evidence(
            geodesic_um=8.0, branch_count=2, profile_merge=False, min_branch_nodes=0
        )


def test_missing_or_nonfinite_inputs_fail_closed_to_not_flagged():
    """A missing measurement is not evidence of a merge."""
    saturn = load_saturn_v571()
    assert not saturn.suspected_multi_object_merge_evidence(
        geodesic_um=float("nan"), branch_count=0, profile_merge=False
    )
    assert not saturn.suspected_multi_object_merge_evidence(
        geodesic_um=8.0, branch_count=None, profile_merge=False
    )


def test_branch_node_count_on_known_skeleton_geometry():
    """A T-shaped filament has exactly one branch point."""
    saturn = load_saturn_v571()
    component = np.zeros((41, 41), dtype=bool)
    component[19:22, 6:34] = True   # horizontal bar
    component[6:20, 19:22] = True   # vertical stem meeting it
    assert saturn.component_branch_node_count(component) >= 1

    straight = np.zeros((41, 41), dtype=bool)
    straight[19:22, 6:34] = True
    assert saturn.component_branch_node_count(straight) == 0

    empty = np.zeros((41, 41), dtype=bool)
    assert saturn.component_branch_node_count(empty) == 0


def test_the_previously_documented_overlong_branched_case_is_retained():
    """This change must add evidence, never unflag a case flagged before."""
    saturn = load_saturn_v571()
    # One branch node is a spur at typical length, but above the overlong
    # review length the design ledger already called it a technical merge.
    assert not saturn.suspected_multi_object_merge_evidence(
        geodesic_um=8.0, branch_count=1, profile_merge=False
    )
    assert saturn.suspected_multi_object_merge_evidence(
        geodesic_um=25.0, branch_count=1, profile_merge=False
    )
    # Length still never suffices on its own.
    assert not saturn.suspected_multi_object_merge_evidence(
        geodesic_um=25.0, branch_count=0, profile_merge=False
    )
