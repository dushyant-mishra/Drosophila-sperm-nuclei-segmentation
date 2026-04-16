# Saturn V3: Native Conservative Tracking Guide

## Executive Summary

The Saturn pipeline has evolved to its **V3 architecture**, natively integrating **Conservative Tracking**. 
Previously, tracking anomalies (monster fusions, multi-strand zig-zags) were handled by a reactive "Patch V1" overlay that attempted to repair bad links after they formed. This approach proved unstable and generated excessive tortuosity outliers.

Saturn V3 permanently eliminates "Patch V1" and bakes conservative tracking directly into the core `sperm_segmentation_saturnv3.py` and `sperm_segmentation_v12_combined.py` engines. 

## The Native Conservative Philosophy

**PREVENT bad tracks from forming, don't repair them after.**

When the V3 engine extends a track to a new Z-slice, it evaluates:
- ✅ **Centroid Displacement**: Is jump < 10 µm?
- ✅ **Width Consistency**: Is width change < 40%?
- ✅ **Length Consistency**: Is length change < 60%?
- ✅ **Area Consistency**: Is area change < 70%?

If ALL checks pass → the filament is extended into the Z-slice.
If ANY check fails → track extension is cleanly aborted, and the new slice begins a new, disjoint track.

This is fundamentally safer: breaking a legitimate track into two pieces (fragments) is vastly superior to artificially fusing two disjoint cells (monster fusions).

---

## File System & Architecture

The ecosystem has been cleaned up. The following files govern the new native V3 pipeline:

1. **`sperm_segmentation_saturnv3.py` (Core Engine)**
   The core tracking algorithm. All thresholds are defined directly in the `CONFIG` dictionary under the `# ── conservative tracking stop-rules ──` section.

2. **`sperm_segmentation_v12_combined.py` (Combined Engine)**
   The hybrid engine synced with V3 conservative logic + advanced GUI/Reports.

3. **`tune_universal_parameters.py` (Biological Tuner)**
   Automatically mutates configuration files to mass-test biological consistency thresholds and emits optimal scores. Used to run grid searches to minimize tortuous tracking errors.

4. **`tuning_results_optimal.json`**
   The finalized optimized target file from the tuner.

---

## How to Test the Pipeline

### Step 1: Run the Pipeline

```bash
# Headless batch processing using V3
python sperm_segmentation_saturnv3.py --batch --dir "path/to/your/images"

# Or using the Combined GUI
python sperm_segmentation_v12_combined.py
```

Outputs will be correctly labeled with the internal version tag: `track_summary_v3.csv` or `track_summary_v12.csv`.

### Step 2: Audit Tracking Quality

The auditor automatically discovers versioned CSV outputs using wildcard logic.

```bash
python audit_sperm_outliers.py --dir "path/to/batch_output"
```

### Step 3: Compare Against Legacy Runs

To quantify improvements against older V2 runs, use the comparison script:

```bash
python compare_sperm_outlier_audits.py \
  --old "C:/path/batch_output_legacy_v2/outlier_audit" \
  --new "C:/path/batch_output_new_v3/outlier_audit" \
  --label-old "V2" \
  --label-new "V3 Conservative"
```

---

## Tuning the Parameters (Grid Search)

If the pipeline is failing to link cells correctly (fragmentation) or still linking distinct cells (fusions), you can run an automated tuning grid.

```bash
python tune_universal_parameters.py --dir "path/to/images" --quick
```

This will:
1. Fire up a dozen batch runs of Saturn V3 using slightly perturbed parameter limits.
2. Automatically run the auditor on each output dataset.
3. Collate the errors.
4. Output the statistically optimal constraints to `tuning_results_optimal.json`.

You can then manually transcribe these discovered constraints into the top of `sperm_segmentation_saturnv3.py`.
