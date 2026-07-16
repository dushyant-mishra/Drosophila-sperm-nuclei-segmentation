# Saturn V5.4 Hybrid Repair Prototype

V5.4 is an experimental tracking branch created from V5.3. It keeps V5.2-style
legacy tracking as the first pass, then applies a conservative repair pass to
short fragments only.

## New Files

- `sperm_segmentation_saturnv5.4.py`
- `utils/tune_parameters_Saturnv5_4.py`
- `parameter_presets/saturnv5.4_hybrid_repair_prototype.json`

## Tracking Backends

`TRACKING_BACKEND` supports:

- `legacy` - original conservative overlap/centroid tracking.
- `global_assignment` - V5.3 whole-population assignment prototype.
- `hybrid_repair` - V5.4 default: legacy tracking plus conservative fragment repair.

## Hybrid Repair Logic

The repair pass only considers tracks separated by a small z-gap. At least one
side of the proposed merge must be a short fragment, and the candidate is
rejected if its cost is too high, the link distance is too large, overlap is too
weak, or the estimated merged 3D length would exceed the configured maximum.

This is intentionally conservative: ambiguous objects remain split instead of
being forced into longer tracks.

## Initial Dry Run

Using the existing 2D detections from `batch_output_2`, the default V5.4 hybrid
settings produced:

- Total tracks: 17,745
- Quality tracks: 5,717
- Biological candidate tracks: 11,652
- Warning-only candidate tracks: 5,935
- Hard-fail tracks: 6,093
- Single-slice fraction: 46.8%
- Median 3D length: 11.399 um
- Mean 3D length: 12.945 um
- Median z-span: 1.04 um
- Hybrid repairs accepted: 220

This is not a final tuned result, but it suggests the hybrid approach is less
aggressive than pure V5.3 global assignment while still reducing fragmentation.

## Quality Overlays

V5.4 writes an additional `quality_overlays/` folder after tracking and audit.
These panels are generated after the two-tier audit is known:

- green = biological candidate with no warning flags
- yellow = biological candidate with warning-only PSF-sensitive flags
- red = hard-failed track
- gray = untracked or unmapped detection

The PDF summary prefers `quality_global_z_projection.png` when it exists, so the
report visual reflects the candidate/hard-fail population instead of only raw 2D
detection.

## Two-Tier Audit

V5.4 keeps the old strict audit as `is_quality_track`, but adds a softer
biological-candidate tier:

- `is_quality_track`: strict no-warning subset. Any long, tortuous, thick,
  taper, or shallow flag fails.
- `is_biological_candidate`: softer review population. Long, tortuous,
  extreme-thick, extreme-taper, or shallow tracks hard-fail; ordinary thick and
  taper flags are kept as warning-only because they are PSF-sensitive.
- `hard_flags`: structural or extreme failures.
- `warning_flags`: PSF-sensitive warnings such as thick/taper.
- `has_warning_only`: candidate track that carries only warning flags.

The report, GUI status text, AI analysis export, and v5.4 tuner now treat
`is_biological_candidate` as the main analysis population. The strict
`is_quality_track` subset is retained as a conservative diagnostic/no-warning
population, not as the primary biological count.

The v5.4 tuner therefore penalizes ordinary thick/taper tracks only softly.
Extreme thickness/taper, excessive length, and tortuosity remain hard negatives
because those are more likely to represent over-merges or unstable tracks.

## Suggested Use

First load the tuned segmentation JSON, then load:

`parameter_presets/saturnv5.4_hybrid_repair_prototype.json`

For tuning:

```powershell
.\.venv\Scripts\python.exe .\utils\tune_parameters_Saturnv5_4.py --mode tracking --dir "C:\Users\dmishra\Desktop\sperm images" --slices 28-32 --roi-mask "C:\Users\dmishra\Desktop\sperm images\roi_z28.1.npy" --params .\parameter_tuning_results\roi_z28_middle_segmentation\best_segmentation_params_v5_2_001_run_sperm_images.json --maxiter 0 --popsize 1 --no-polish --outdir .\parameter_tuning_results\roi_z28_middle_tracking_v5_4_hybrid
```
