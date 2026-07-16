# Saturn v5.6 Comparative Analysis

Saturn v5.6 now supports an explicit `ANALYSIS_MODE = "comparative"` for WT-versus-mutant studies.

The purpose of comparative mode is to avoid silently forcing mutant morphology toward WT-like expectations. Segmentation should answer:

> Is this an independently resolved image object that could be a nucleus?

It should not answer:

> Does this object resemble the expected WT morphology?

## Output Populations

Use this table for statistical WT-versus-mutant comparisons:

```text
track_summary_technical_valid_v5.6-roi-adaptive.csv
```

The full annotated table is:

```text
track_summary_all_v5.6-roi-adaptive.csv
```

Additional diagnostic tables:

```text
track_summary_reference_morphology_v5.6-roi-adaptive.csv
track_summary_morphology_warning_v5.6-roi-adaptive.csv
track_summary_technical_failures_v5.6-roi-adaptive.csv
```

`reference_morphology` is a diagnostic subset only. It is not the biologically correct population.

Morphology warnings are retained in the comparative population because they may represent genuine genotype-dependent phenotypes.

## Technical Validity Versus Morphology

Technical failures are clear processing failures, such as invalid coordinates, zero length, segmentation leakage, exclusion-mask overlap, outside-ROI detections, gross branch networks, clear multi-object connected components, and impossible software-generated geometry.

Morphology warnings include long, short, wide, thin, high tortuosity, high taper, low taper, low length-to-width ratio, unusual pitch, unusual volume, unusual Z-span, and unusual nearest-neighbor distance.

In comparative mode, morphology warnings do not remove tracks from the main comparative population.

## Why WT-Only Tuning Can Bias Mutant Measurements

If a tuner rewards a fixed WT-like length, width, taper, tortuosity, count, volume, or Z-span, it can preferentially remove real mutant phenotypes. A mutant that genuinely has longer, wider, more tapered, more tortuous, or lower-count nuclei should remain longer, wider, more tapered, more tortuous, or lower count after segmentation and audit unless there is independent evidence of a technical failure.

## Why Separate Genotype Tuning Is Prohibited

Do not tune WT and mutant independently. Genotype-specific morphology thresholds can convert biological differences into parameter artifacts. Use the same segmentation parameter set and the same technical-failure rules for all groups.

Stack-specific photometric normalization is allowed because imaging brightness can differ between stacks. Genotype-specific morphology adaptation is not allowed.

## Sensitivity Presets

Four comparative presets are provided for sensitivity analysis:

```text
comparative_presets/comparative_conservative_v5_6.json
comparative_presets/comparative_selected_v5_6.json
comparative_presets/comparative_intermediate_v5_6.json
comparative_presets/comparative_permissive_v5_6.json
```

These are not final production parameters. They are intended to show whether the biological conclusion is stable under conservative, selected, intermediate, and permissive segmentation settings.

For each preset, compare:

- technical-valid count
- length
- width
- taper
- tortuosity
- volume
- Z-span
- pitch
- nearest-neighbor distance
- morphology-warning fraction
- technical-failure fraction
- overlap or matching between presets

Report how many objects are detected by all presets, only by permissive settings, lost by conservative settings, classified differently by morphology warnings, or technically rejected differently.

## Blinded Validation Workflow

Prepare a manifest with one row per dataset or stack. Recommended columns:

```text
dataset_path,roi_path,exclusion_mask_path,genotype,dataset_label
```

Before segmentation, create a blinded manifest:

- assign anonymized dataset IDs
- remove genotype/group labels from the segmentation input
- save the reveal table separately
- use the same segmentation parameters for every group
- use the same technical-failure rules for every group
- allow stack-specific photometric normalization
- do not retune WT and mutant independently

The manual review workbook should hide genotype identity and include blank columns for:

- true detection
- missed nucleus
- split nucleus
- merged nuclei
- tissue-edge false positive
- puncta/ring false positive
- uncertain

Reveal genotype labels only after review is complete.

## Differential-Error Checks

For each anonymized group, report:

- technical-failure fraction
- morphology-warning fraction
- short-fragment fraction
- suspected-merge fraction
- branch-network fraction
- ROI-edge fraction
- permissive-only detection fraction
- conservative-loss fraction

Warn if one group has substantially higher technical rejection or parameter sensitivity. Do not correct distributions to make groups agree.

## Recommended Interpretation

Use `technical_valid` as the primary analysis population.

Use morphology warnings to understand which phenotypes are unusual, not to remove them by default.

Use the reference-morphology subset as a diagnostic view of WT-like tracks only.

Use sensitivity presets to ask whether conclusions survive reasonable segmentation uncertainty.

## Limitations

Comparative mode does not prove every technical-valid detection is true. It prevents the audit from discarding morphology solely because it is unusual. Manual blinded review is still required for representative WT and mutant images.
