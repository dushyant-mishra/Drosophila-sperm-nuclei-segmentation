# Sperm Project Layout

The repository root contains only the active Saturn v5.7 workflow and its
durable supporting material.

## Active Runtime

- `sperm_segmentation_saturnv5.7.py`: active GUI and analysis pipeline.
- `utils/tune_parameters_Saturnv5_7.py`: active tuner.
- `utils/saturn_unet25d_bridge.py`: active tiled U-Net inference bridge.
- `unet25d/`: U-Net preparation, training, inference, and review code.
- `requirements.txt`: core Python dependencies.
- `.venv/`: current local environment.

## Active Data and Documentation

- `parameter_tuning_results_v5_7/`: reviewed v5.7 candidate parameters and
  tuning summaries.
- `docs/v5_7_illustrated_workflow/`: figures, metadata, and report-generation
  assets.
- `Saturn_V5.7_Illustrated_Analysis_Workflow_FINAL.docx`: current illustrated
  analysis report.
- `USER_GUIDE.md`: operating and interpretation guidance.
- `scratch/`: small active v5.7 validation and report scripts only.
- `tests/test_saturn_v57_multisample.py`: active automated regression tests.
- `pytest.ini`: limits test discovery to active tests and excludes archived
  historical suites.

## Archive

- `archive/historical_source/saturn_v5_series/`: v5.1-v5.6 pipelines, tuners,
  tests, presets, documentation, and historical tuning results.
- `archive/project_history/pre_v5_7/`: large generated pre-v5.7 experiments.
- `archive/project_history/v5_7/`: reviewed or superseded v5.7 run evidence.
- `archive/distributions/`: old packaged applications and local build output.
- `archive/manifests/`: reversible cleanup records.

## Local-Only Items

- `gemini_api_key.txt`: active local key used by the optional report assistant;
  ignored by Git.
- `.agents/`: empty tool-managed workspace directory; it may be recreated or
  protected by the development environment.

New generated batch outputs should remain outside the source tree when
possible. Reviewed development outputs can be moved into
`archive/project_history/v5_7/`.
