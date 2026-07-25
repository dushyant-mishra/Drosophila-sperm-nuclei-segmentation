# Saturn Project Archive

This directory preserves historical source, generated validation evidence, and
legacy distributions that are no longer part of the active Saturn v5.7
workspace.

## Layout

- `historical_source/pipelines/`: superseded pipeline and GUI source files.
- `historical_source/legacy_backups/`: older combined pipelines and tuning
  history retained in their original internal layout.
- `historical_source/tuning/`: historical parameter files and tracking notes.
- `historical_source/utilities/`: temporary comparison and patch utilities.
- `historical_source/saturn_v5_series/`: v5.1-v5.6 pipelines, matching tuners,
  regression tests, presets, documentation, tuning results, and old build
  tooling moved from the active project root. The archived macOS workflow
  targeted v5.1 and is retained as history rather than active automation.
- `distributions/legacy_builds/`: old packaged application artifacts.
- `distributions/local_build_2026-07-25/`: generated local `build/` and `dist/`
  trees removed from the active project root.
- `private_config/`: ignored local configuration or secrets. Never commit this
  directory's contents.
- `project_history/pre_v5_7/`: generated runs, scripts, and notes from work
  before Saturn v5.7.
- `project_history/v5_7/development_runs/`: superseded v5.7 smoke and
  development runs, including preliminary tuning smoke folders.
- `project_history/v5_7/validation_runs/`: useful v5.7 validation evidence,
  including the hybrid multisample and WT/SATNull pilot runs.
- `manifests/`: records of archive reorganizations.

## Active Files

The current pipeline, tuner, U-Net implementation, parameter results, and
illustrated report remain outside this archive. Generated project-history data
is intentionally ignored by Git because it is large and may contain research
images.

Nothing in the 2026-07-25 cleanup was deleted except Python bytecode cache.
The later disk-cleanup pass also deleted two reproducible obsolete virtual
environments (`venv/` and `.venv-ai-v56/`); the active `.venv/` was retained.
