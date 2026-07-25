# Saturn v5.6 2D Graph Segmentation

Canonical module:

`utils/saturn_v56_2d_graph_segmentation.py`

Canonical runner:

`scratch/run_v56_consolidated_2d_segmentation.py`

This workflow converts raw microscopy slices plus four-class ilastik
probability maps into auditable, raw-supported 2D centerline candidates. It does
not run tracking, cross-Z linking, genotype analysis, model retraining, or
full-stack segmentation.

## Inputs

- Raw top-level TIFFs: `Project001_Series002_z##_ch00.tif`
- ROI: `C:\Users\dmishra\Desktop\sperm images\roi_z28.1.npy`
- Ilastik maps: `scratch\v5_6_ilastik_pilot\probability_maps`
- HDF5 key: `exported_data`
- Channels: sperm nucleus, structured tissue edge, punctum/ring, diffuse background

## Historical Baseline

The historical raw regression baseline is named
`historical_raw_regression_baseline_v5_6`.

Current regression counts:

- z05: 266
- z35: 318

These are engineering provenance checks, not biological ground truth and not
count targets for the new graph method.

## Weighted Ridge

The selected transparent formulation is:

`raw_ridge * nucleus_probability * max(nucleus_probability - max(tissue_probability, punctum_probability), 0)`

The runner also reports two alternatives for comparison. Formulations are not
selected by detection count.

## Outputs

All outputs are written under:

`scratch/v5_6_consolidated_2d`

Important files:

- `configuration/consolidated_2d_config_v5_6.json`
- `weighted_ridge/weighted_ridge_comparison_v5_6.csv`
- `seed_graphs/seed_audit_v5_6.csv`
- `endpoint_extensions/endpoint_extension_audit_v5_6.csv`
- `join_matching/join_proposal_audit_v5_6.csv`
- `join_matching/join_matching_audit_v5_6.csv`
- `completed_paths/completed_path_audit_v5_6.csv`
- `pass2_recovery/pass2_recovery_audit_v5_6.csv`
- `historical_mapping/historical_to_new_mapping_v5_6.csv`
- `reports/development_quality_gates_v5_6.json`
- `manual_review/v5_6_2d_candidate_review.xlsx`
- `reports/consolidated_2d_summary_v5_6.json`

The method remains a development candidate until manual crop review is complete.
