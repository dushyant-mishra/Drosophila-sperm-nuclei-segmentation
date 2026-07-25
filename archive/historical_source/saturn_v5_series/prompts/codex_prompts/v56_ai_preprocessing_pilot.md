Implement a limited AI-assisted preprocessing pilot for Saturn v5.6.

The purpose is to determine whether a small self-supervised microscopy
denoising model can improve visibility and recovery of faint sperm nuclei
without hallucinating structures, deleting unusual morphology, or increasing
tissue-edge and puncta false positives.

Dataset:

C:\Users\dmishra\Desktop\sperm images

ROI:

C:\Users\dmishra\Desktop\sperm images\roi_z28.1.npy

Raw source pattern, top level only:

^Project001_Series002_z(\d+)_ch00\.tif{1,2}$

The stack has exactly 88 uint8 slices with Z indices 0 through 87.

Use these held-out test slices:

z05, z06, z12, z35, z60, z87

Do not use genotype information.

===============================================================================
STRICT SCOPE
===============================================================================

Do not:

- modify any v5.5 file
- replace the existing raw-image workflow
- enable AI preprocessing by default
- process the full stack through segmentation
- run tracking
- perform genotype analysis
- optimize toward a target count
- optimize toward a target length, width, taper, or tortuosity
- commit
- push
- silently change to a different AI method if N2V2 is unavailable

All AI outputs are experimental candidate-proposal inputs.

Raw images remain the source of truth for:

- length
- width
- intensity
- taper
- tortuosity
- final acceptance review
- publication images

===============================================================================
1. ISOLATED AI ENVIRONMENT
===============================================================================

Do not destabilize the existing .venv.

Create an isolated environment:

.venv-ai-v56

Use the same compatible Python major/minor version as the working project
environment whenever possible.

Install the current project requirements in that environment, followed by the
minimum optional packages required for CAREamics/N2V2.

Create:

requirements-ai-v5.6.txt

Record exact installed versions with:

python -m pip freeze

Save them as:

scratch/v5_6_ai_pilot/environment/pip_freeze_ai_v5_6.txt

Check whether CUDA is available and record:

- Python version
- PyTorch version
- CUDA version
- GPU name
- CAREamics version
- NumPy version
- scikit-image version

If CAREamics or N2V2 is incompatible with the available Python/PyTorch
environment, stop and report the exact incompatibility. Do not change the
working .venv.

===============================================================================
2. IMPLEMENT OPTIONAL AI PREPROCESSING MODULE
===============================================================================

Create:

utils/ai_preprocessing_v5_6.py

Provide a small, version-tolerant interface such as:

- discover_n2v2_backend()
- build_n2v2_training_config()
- train_n2v2_model()
- load_n2v2_model()
- predict_n2v2()
- normalize_ai_input()
- restore_output_scale()
- validate_ai_output()
- calculate_raw_support_metrics()

Do not tightly couple the main v5.6 pipeline to one CAREamics API version.

Use an adapter layer and record the discovered backend/API version.

The production pipeline must continue to work when CAREamics is not installed.

AI-related imports must be lazy and optional.

===============================================================================
3. DATA SPLITTING WITHOUT TEST LEAKAGE
===============================================================================

The six representative test slices must not be used for model training.

Also exclude their immediate neighboring slices from training when they exist.

For each test slice z, exclude:

z - 1
z
z + 1

Use a deterministic seed.

From the remaining slices, create:

- training set
- validation set

Select validation slices deterministically and keep them separate from the
training slices.

Save:

scratch/v5_6_ai_pilot/data_split_v5_6.json

Include:

- training Z indices
- validation Z indices
- test Z indices
- excluded buffer Z indices
- random seed

Do not train using the ROI-exterior region.

===============================================================================
4. ROI-BASED PATCH EXTRACTION
===============================================================================

Train a small 2D N2V2 model using patches sampled only from the ROI.

Requirements:

- preserve native 1024 x 1024 source data
- use float32 internally
- retain the original intensity scale metadata
- sample nucleus-rich and background-containing ROI patches
- avoid patches dominated by zero-filled space
- do not use outside-ROI pixels as biological training content
- fill outside-ROI areas using the same robust median strategy used by v5.6
- use moderate augmentation only
- do not geometrically distort nuclei
- record patch size, batch size, epochs, learning rate, and patch counts

Use conservative training limits suitable for a pilot.

Do not perform a long hyperparameter search.

Use early stopping when supported.

Save training artifacts under:

scratch/v5_6_ai_pilot/n2v2_model

Save:

- model checkpoint
- training configuration
- normalization metadata
- training and validation losses
- training log
- representative input/output patches

===============================================================================
5. N2V2 OUTPUT VALIDATION
===============================================================================

For every held-out test slice produce:

- raw image
- robust-normalized raw image
- N2V2-denoised image
- raw minus denoised residual
- ridge response from raw
- ridge response from denoised
- intensity-line comparisons across selected nuclei
- Fourier/noise summaries where useful

Check for:

- hallucinated elongated structures
- deleted faint nuclei
- merged adjacent nuclei
- excessive smoothing
- width inflation
- boundary artifacts
- checkerboard artifacts
- intensity clipping
- changes outside the ROI

Save full-resolution TIFFs and reviewable PNG panels.

Do not assess quality only through global PSNR or loss.

===============================================================================
6. SATURN SEGMENTATION COMPARISON
===============================================================================

Create:

scratch/run_v56_ai_preprocessing_pilot.py

Compare these conditions on exactly:

z05, z06, z12, z35, z60, z87

Conditions:

A. raw_selected
   Existing Saturn v5.6 selected preset on the raw image.

B. n2v2_selected
   The same Saturn selected preset using the N2V2 image only for candidate
   generation.

C. raw_plus_n2v2_residual_recovery
   Raw selected segmentation as Pass 1, then use the N2V2 image only in
   residual, unclaimed regions for candidate proposals.

Do not measure morphology from the N2V2 image.

For conditions B and C:

- calculate candidate geometry from the raw image and raw-derived mask
- require corresponding signal support in the raw image
- label every AI-assisted detection with:
  ai_assisted = true
  ai_method = N2V2
  raw_support_score
  recovery_pass
  recovery_reason

Reject an AI proposal when it has no sufficient signal support in the original
raw image.

===============================================================================
7. RAW-SIGNAL SUPPORT GATE
===============================================================================

Implement a configurable raw-signal validation check.

For each AI-proposed object calculate:

- raw centerline intensity
- local raw background
- raw local contrast
- raw ridge-response support
- fraction of the proposed centerline supported in raw data
- orientation consistency between raw and AI-derived ridges
- overlap with raw hysteresis or subthreshold ridge evidence
- distance from ROI edge
- punctate or ring likelihood
- branch-network likelihood

The gate must not require WT-like length or morphology.

Save all accepted and rejected AI proposals with rejection reasons.

===============================================================================
8. COMPARISON METRICS
===============================================================================

For every slice and condition calculate:

- total detections
- technical-valid detections
- morphology-warning detections
- new detections relative to raw_selected
- detections lost relative to raw_selected
- matched-object fraction
- centroid displacement
- length difference for matched objects
- width difference measured from raw support
- split rate indicator
- merge rate indicator
- clean-mask occupancy
- skeleton occupancy
- branch-network fraction
- loop fraction
- ROI-edge fraction
- outside-ROI leakage
- puncta/ring candidate fraction
- broad-tissue candidate fraction
- raw-support failure fraction

Do not reward higher detection counts automatically.

===============================================================================
9. MANUAL REVIEW PANELS
===============================================================================

Create identical-coordinate panels for:

- upper bulb
- central bulb
- transition region
- lower shaft
- faint nuclei
- dense parallel nuclei
- crossing nuclei
- broad tissue edges
- bright puncta
- ROI boundary

For each crop show:

1. raw
2. N2V2
3. raw ridge
4. N2V2 ridge
5. raw-selected overlay
6. N2V2-selected overlay
7. residual-recovery overlay
8. AI proposals rejected by the raw-support gate

Use consistent display limits and identical crop coordinates.

Create a blank workbook:

scratch/v5_6_ai_pilot/ai_pilot_manual_review_v5_6.xlsx

Columns:

- Z index
- crop ID
- condition
- genuine nucleus recovered
- faint nucleus recovered
- genuine nucleus deleted
- hallucinated object
- split nucleus
- merged nuclei
- tissue-edge false positive
- puncta/ring false positive
- excessive smoothing
- width distortion
- uncertain
- reviewer notes

Leave subjective columns blank.

===============================================================================
10. ILASTIK INTEGRATION SUPPORT
===============================================================================

Do not install or automate the ilastik GUI in this phase.

Create an export utility:

scratch/export_v56_ilastik_training_data.py

Export neutral training images for the six representative slices under:

scratch/v5_6_ai_pilot/ilastik_export

Export:

- raw ROI image
- robust-normalized ROI image
- optional N2V2 image
- ROI mask
- metadata JSON

Do not export existing Saturn detections as ground truth.

Create documentation for manually labeling these ilastik pixel classes:

1. sperm_nucleus
2. structured_tissue_edge
3. punctum_or_ring
4. diffuse_background

Also implement an optional probability-map importer:

load_ilastik_probability_map()

It should validate:

- image dimensions
- channel count
- probability range
- ROI alignment
- Z index
- metadata consistency

Do not use an ilastik probability map unless the user explicitly supplies one.

===============================================================================
11. BIT-DEPTH HANDLING
===============================================================================

The earlier robustness test showed poor matching after uint8-to-uint16 scaling.

For this AI pilot:

- normalize uint8 and uint16 inputs through one explicit float32 pathway
- record source dtype and source intensity range
- do not feed raw integer values directly into N2V2
- restore output to a clearly documented normalized float32 range
- do not silently cast AI output back to uint8 before segmentation

Add a synthetic equivalence test demonstrating that numerically equivalent
uint8 and scaled uint16 inputs produce nearly equivalent normalized AI inputs.

Do not claim the full production pipeline is bit-depth invariant unless its
separate tests pass.

===============================================================================
12. REQUIRED OUTPUTS
===============================================================================

Use:

scratch/v5_6_ai_pilot

Subdirectories:

- environment
- data_split
- training
- n2v2_model
- predictions
- segmentation_comparison
- rejection_audit
- review_panels
- ilastik_export
- reports

Create:

ai_pilot_summary_v5_6.json
ai_pilot_slice_metrics_v5_6.csv
ai_proposal_audit_v5_6.csv
ai_object_matching_v5_6.csv
ai_training_history_v5_6.csv
ai_pilot_manual_review_v5_6.xlsx
ai_preprocessing_pilot_report_v5_6.pdf

The report must state clearly:

- AI restoration is experimental
- raw images remain the measurement source
- increased count is not evidence of increased accuracy
- AI-generated candidates require raw-image support
- visual review is required before production integration

===============================================================================
13. TESTS
===============================================================================

Add tests for:

1. CAREamics is optional and the production pipeline imports without it.
2. Test slices and neighboring buffer slices are excluded from training.
3. Patch extraction stays inside the ROI.
4. Outside-ROI changes do not alter AI input patches.
5. AI predictions preserve image dimensions.
6. Equivalent uint8 and uint16 images have equivalent float32 AI inputs.
7. AI proposals without raw support are rejected.
8. Genuine synthetic faint rods with raw support may be recovered.
9. AI recovery does not alter existing Pass-1 object IDs.
10. AI recovery remains inside the ROI.
11. AI recovery does not use a target count.
12. AI recovery does not use target length or width.
13. Morphology measurements come from raw-derived data.
14. ilastik probability-map dimension mismatch is rejected.
15. Existing v5.6 tests continue to pass.

===============================================================================
14. EXECUTION ORDER
===============================================================================

First:

1. Inspect the existing environment.
2. Create the isolated AI environment.
3. Confirm N2V2 availability.
4. Implement modules and tests.
5. Run synthetic tests.
6. Export the planned data split.
7. Train one small pilot model.
8. Predict only the six held-out test slices.
9. Run the three-condition segmentation comparison.
10. Generate review panels and report.

Stop before production integration.

===============================================================================
15. VALIDATION COMMANDS
===============================================================================

Run production checks using the normal .venv:

.\.venv\Scripts\python.exe -m py_compile `
  .\sperm_segmentation_saturnv5.6.py `
  .\utils\ai_preprocessing_v5_6.py `
  .\scratch\run_v56_ai_preprocessing_pilot.py `
  .\scratch\export_v56_ilastik_training_data.py

.\.venv\Scripts\python.exe -m pytest -q

.\.venv\Scripts\python.exe `
  .\utils\tune_parameters_Saturnv5_6.py --self-check

Run AI-specific checks and pilot execution using .venv-ai-v56.

Then run:

git diff --check
git status --short

Confirm:

- no v5.5 files changed
- no full-stack segmentation occurred
- no tracking occurred
- no genotype analysis occurred
- no target count or morphology prior was used
- raw images remained the measurement source
- AI preprocessing was not enabled by default
- no commit occurred
- no push occurred

Return:

1. AI environment versions
2. training/validation/test split
3. training duration and loss history
4. six-slice N2V2 prediction paths
5. raw versus N2V2 segmentation metrics
6. residual-recovery metrics
7. rejected AI-proposal reasons
8. manual-review workbook path
9. PDF report path
10. ilastik export path
11. automated test results
12. unresolved risks

Do not merge the AI branch into the production segmentation path automatically.
