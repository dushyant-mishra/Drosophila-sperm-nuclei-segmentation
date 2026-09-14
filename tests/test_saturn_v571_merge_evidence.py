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


def _split_cfg(saturn, **overrides):
    cfg = {
        **saturn.CONFIG,
        "UM_PER_PX_XY": 0.37841796875,
        "UNET_PRIMARY_OVERLONG_SPLIT_ENABLE": True,
        "UNET_PRIMARY_OVERLONG_SPLIT_TRIGGER_UM": 20.0,
    }
    cfg.update(overrides)
    return saturn.cfg_with_resolved_pixels(cfg)


def _branched_component():
    """A stem with one arm: objectively two filaments, well under 20 um."""
    mask = np.zeros((44, 44), dtype=bool)
    mask[21:24, 6:30] = True
    mask[10:22, 16:19] = True
    return mask


def _split_inputs(saturn, mask):
    probability = np.where(mask, 0.9, 0.0).astype(np.float32)
    # A core head that sees a single object, so only topology can supply evidence.
    core = np.zeros_like(probability)
    core[mask] = 0.8
    labels = saturn.measure.label(mask).astype(np.int32)
    return probability, core, labels


def test_a_short_branched_merge_is_split_without_any_length_trigger():
    """The case the length gate missed: a merge near the median instance length."""
    saturn = load_saturn_v571()
    mask = _branched_component()
    cfg = _split_cfg(saturn)
    probability, core, labels = _split_inputs(saturn, mask)

    _, length_px = saturn._longest_centerline_for_mask(mask)
    assert length_px * cfg["UM_PER_PX_XY"] < 20.0, "fixture must sit under the trigger"

    refined, _, audit = saturn._refine_merged_unet_instances(
        probability, labels, {1: 1}, cfg, core_probability=core
    )

    assert int(refined.max()) >= 2
    assert audit[0]["candidate_reason"] == "branch_topology"
    assert audit[0]["disposition"] == "branch_topology_watershed_split"
    assert audit[0]["objective_core_marker_count"] < 2
    # Splitting redistributes pixels, it never adds or removes them.
    assert np.array_equal(refined > 0, mask)


def test_a_clean_nucleus_is_not_a_candidate_and_is_never_split():
    saturn = load_saturn_v571()
    straight = np.zeros((70, 70), dtype=bool)
    straight[33:37, 10:40] = True
    cfg = _split_cfg(saturn)
    probability, core, labels = _split_inputs(saturn, straight)

    refined, _, audit = saturn._refine_merged_unet_instances(
        probability, labels, {1: 1}, cfg, core_probability=core
    )

    assert int(refined.max()) == 1
    assert audit == []
    assert np.array_equal(refined > 0, straight)


def test_a_split_whose_children_fail_the_checks_leaves_the_parent_intact():
    """If the constituents do not pass, treat the object as any other detection."""
    saturn = load_saturn_v571()
    mask = _branched_component()
    # Demand children longer than either filament can be.
    cfg = _split_cfg(saturn, UNET_PRIMARY_OVERLONG_SPLIT_MIN_CHILD_UM=8.0)
    probability, core, labels = _split_inputs(saturn, mask)

    refined, _, audit = saturn._refine_merged_unet_instances(
        probability, labels, {1: 1}, cfg, core_probability=core
    )

    assert int(refined.max()) == 1
    assert audit[0]["disposition"] == "unchanged"
    assert np.array_equal(refined > 0, mask)


def test_length_alone_never_makes_a_split_without_objective_evidence():
    """A genuinely long straight nucleus is morphology, not a merge."""
    saturn = load_saturn_v571()
    long_straight = np.zeros((40, 200), dtype=bool)
    long_straight[19:22, 5:195] = True
    cfg = _split_cfg(saturn)
    probability, core, labels = _split_inputs(saturn, long_straight)

    _, length_px = saturn._longest_centerline_for_mask(long_straight)
    assert length_px * cfg["UM_PER_PX_XY"] > 20.0, "fixture must exceed the trigger"

    refined, _, audit = saturn._refine_merged_unet_instances(
        probability, labels, {1: 1}, cfg, core_probability=core
    )

    assert int(refined.max()) == 1
    assert audit[0]["split_evidence"] == "no_objective_split_evidence_not_split"
    assert np.array_equal(refined > 0, long_straight)


def test_branch_markers_require_two_filaments_worth_keeping():
    saturn = load_saturn_v571()
    cfg = _split_cfg(saturn)
    straight = np.zeros((44, 44), dtype=bool)
    straight[21:24, 6:30] = True
    _, count = saturn._branch_topology_watershed_markers(straight, cfg)
    assert count == 0

    markers, count = saturn._branch_topology_watershed_markers(
        _branched_component(), cfg
    )
    assert count >= 2
    assert int(markers.max()) == count
