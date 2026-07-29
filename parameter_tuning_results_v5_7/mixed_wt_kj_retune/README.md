# Saturn v5.7 mixed WT/KJ tuner input

This is a balanced, ROI-aware staged tuner manifest. It intentionally
references the original complete stacks instead of copying selected TIFF files
into one flat folder.

A flat mixed image folder is not valid for the current tuner because each
specimen has a different ROI. It would also remove the neighboring Z planes
needed by the 2.5D U-Net.

## Composition

- Two KJ specimens: KJ-01 and KJ-13
- Two WT specimens: WT-01 and WT-13
- Six representative planes per specimen
- Early, intermediate, and late optical planes
- The specimen-specific `analysis_roi_v5_7.npy`
- The epoch-3 fine-tuned U-Net checkpoint
- One shared starting preset for both biological groups

The four strata use the same random seed and candidate count. This makes
candidate roles identical across specimens. The runner first aggregates a
shared classical 2D candidate, then uses that unchanged preset as the base for
U-Net rescue tuning and final shared hybrid aggregation.

## Run

From the project root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  ".\parameter_tuning_results_v5_7\mixed_wt_kj_retune\run_mixed_tuner.ps1" `
  -ValidateOnly

powershell -NoProfile -ExecutionPolicy Bypass -File `
  ".\parameter_tuning_results_v5_7\mixed_wt_kj_retune\run_mixed_tuner.ps1"
```

Outputs are written beneath:

```text
parameter_tuning_results_v5_7/mixed_wt_kj_retune/results/<timestamp>
```

Each run contains:

```text
01_classical_2d/
02_unet_rescue/
run_metadata.json
completed_run.json
```

Both aggregated presets are candidates for visual inspection, not automatic
biological finals. Review all eight stratum PDFs before loading the final
hybrid preset into a full study.
