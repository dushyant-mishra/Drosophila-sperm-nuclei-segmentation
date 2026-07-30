import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_v57_biological_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("v57_biological_comparison", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _specimens():
    rows = []
    for group, offset in (("Reference line", 0.0), ("Experimental line", 2.0)):
        for specimen_index in range(6):
            row = {
                "specimen_id": f"{group}_{specimen_index}",
                "group": group,
            }
            for metric_index, metric in enumerate(MODULE.METRICS):
                row[metric] = 5.0 + metric_index + specimen_index * 0.1 + offset
            rows.append(row)
    return pd.DataFrame(rows)


def test_reference_group_uses_generic_control_labels():
    assert MODULE.reference_group(["Experimental line", "Reference line"]) == (
        "Reference line"
    )
    assert MODULE.reference_group(["mutant", "control"]) == "control"


def test_statistics_use_specimens_and_preserve_effect_direction():
    statistics = MODULE.compute_statistics(
        _specimens(),
        reference="Reference line",
        comparison="Experimental line",
        seed=123,
    )

    assert len(statistics) == len(MODULE.METRICS)
    assert set(statistics["analysis_unit"]) == {"biological specimen"}
    assert set(statistics["reference_n"]) == {6}
    assert set(statistics["comparison_n"]) == {6}
    assert np.allclose(
        statistics["median_difference_comparison_minus_reference"], 2.0
    )
    assert (statistics["cliffs_delta_comparison_minus_reference"] > 0).all()
    assert statistics["permutation_median_test_p"].between(0, 1).all()
    assert statistics["mann_whitney_p"].between(0, 1).all()
    assert statistics["welch_t_p"].between(0, 1).all()


def test_bh_qvalues_are_bounded_and_not_smaller_than_input_p_values():
    p_values = np.array([0.001, 0.02, 0.03, 0.5, np.nan])
    q_values = MODULE.bh_qvalues(p_values)

    finite = np.isfinite(p_values)
    assert np.isnan(q_values[-1])
    assert ((q_values[finite] >= p_values[finite]) & (q_values[finite] <= 1)).all()


def test_biological_and_qc_metric_sets_are_complete_and_disjoint():
    assert set(MODULE.BIOLOGICAL_METRICS).isdisjoint(MODULE.QC_METRICS)
    assert set(MODULE.BIOLOGICAL_METRICS) | set(MODULE.QC_METRICS) == set(
        MODULE.METRICS
    )
    assert "estimated_nuclei_per_1000_um2" in MODULE.QC_METRICS
    assert "median_3d_z_span_um" in MODULE.QC_METRICS
    assert "median_2d_length_um" in MODULE.BIOLOGICAL_METRICS
