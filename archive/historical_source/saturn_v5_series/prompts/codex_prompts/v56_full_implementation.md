You are working inside the existing Git repository:

dushyant-mishra/Drosophila-sperm-nuclei-segmentation

Implement Saturn v5.6 and the corresponding Saturn v5.6 parameter tuner.

The current Git branch must be:

feature/saturn-v5.6-roi-adaptive

Do not modify, rename, delete, or overwrite any Saturn v5.5 source file.
Do not commit or push changes automatically.
Do not run the complete microscopy batch automatically.

The primary objective is to make segmentation robust to moderate changes in:

- image brightness
- image contrast
- detector gain
- integer bit depth
- gradual Z-stack bleaching
- bright structures outside the selected ROI

The v5.6 design must ensure that the ROI participates directly in
normalization, contrast estimation, ridge thresholding, cleanup, skeletonization,
and parameter tuning.

Applying the ROI only after full-frame segmentation is not acceptable.

===============================================================================
PHASE 1 â€” INSPECT THE REPOSITORY BEFORE EDITING
===============================================================================

Before making any changes:

1. Run and inspect:

   git branch --show-current
   git status --short
   git rev-parse HEAD

2. Stop with a clear error if the current branch is main or master.

3. Confirm that the current branch is:

   feature/saturn-v5.6-roi-adaptive

4. Locate the exact paths of:

   - the Saturn v5.5 main segmentation pipeline
   - the Saturn v5.5 tuner
   - the outlier-audit utility
   - requirements files
   - existing tests
   - GitHub Actions workflows
   - README and other project documentation

5. Search the entire repository for every call to:

   segment_slice(

6. Search the entire repository for all ROI-related code, including:

   - ROI drawing
   - ROI loading
   - ROI saving
   - ROI filtering
   - ROI use during batch analysis
   - ROI use during GUI preview
   - ROI use during single-image analysis
   - ROI use during tuner evaluation
   - result filtering after segmentation

7. Determine whether v5.5:

   - preprocesses the complete frame
   - calculates normalization or ridge percentiles from the complete frame
   - calls segment_slice without an ROI
   - filters detections to the ROI only afterward
   - performs a second ROI-specific pass
   - unions full-frame and ROI-specific results

8. Record SHA-256 hashes of all v5.5 Python source files before editing.

9. Preserve those hashes for final verification.

10. Report the discovered v5.5 source paths before proceeding internally, then
    continue without asking for confirmation unless required files are missing.

===============================================================================
PHASE 2 â€” CREATE VERSIONED V5.6 FILES
===============================================================================

Create versioned copies of the v5.5 pipeline and tuner.

Expected names:

- sperm_segmentation_saturnv5.6.py
- tune_parameters_Saturnv5_6.py

Place each new file in the same directory as its corresponding v5.5 file.

Do not edit the v5.5 originals.

Update visible version strings, output filenames, default output directories,
module names, report titles, GUI labels, documentation strings, and tuner import
paths to v5.6.

Use a clear version label such as:

v5.6-roi-adaptive

Preserve compatibility with the existing:

- CSV outputs
- Excel outputs
- PDF outputs
- PowerPoint outputs
- tracking tables
- biological-candidate audit
- strict-quality audit
- GUI workflow
- parameter JSON loading

Do not change the mathematical definitions of existing 2D or 3D measurements
unless explicitly required below.

===============================================================================
PHASE 3 â€” V5.6 CONFIGURATION
===============================================================================

Add the following v5.6 configuration keys with these defaults:

PREPROCESS_MODE = "roi_adaptive"
LEGACY_TWO_PASS_ROI = False

STACK_QC_SAMPLE_COUNT = 12

NORM_LOW_PERCENTILE = 1.0
NORM_HIGH_PERCENTILE = 99.5
NORM_STACK_WEIGHT = 0.80

CLAHE_MODE = "auto_stack"
CLAHE_CLIP_HIGH_CONTRAST = 0.010
CLAHE_CLIP_STANDARD = 0.025
CLAHE_CLIP_LOW_CONTRAST = 0.035
CLAHE_KERNEL = 128

AUTO_CONTRAST_HIGH_THRESHOLD = 0.45
AUTO_CONTRAST_LOW_THRESHOLD = 0.25

ROI_CROP_PADDING_PX = 16
ROI_THRESHOLD_PERCENTILES_ONLY = True
EXCLUSION_MASK_PATH = ""

DENOISE_SIGMA_UM = 0.45
BG_SIGMA_UM = 6.0
RIDGE_SIGMAS_UM = [0.60, 0.90, 1.20, 1.50]

THRESHOLD_HI = 91.0
THRESHOLD_LO = 83.0

CLOSE_RADIUS = 0
MIN_HOLE_AREA = 0
MIN_OBJ_PX = 8

MIN_SKEL_LEN_UM = 6.0
MAX_GEODESIC_LEN_UM = 20.0
MAX_WIDTH_UM = 4.2
MAX_BRIDGE_UM = 1.5
MAX_BRANCH_LEN_UM = 2.3
MAX_BRIDGE_ANGLE_DEG = 35.0

MIN_LENGTH_WIDTH_RATIO = 2.5
MAX_BRANCH_NODES = 0
MAX_ENDPOINT_COUNT = 4
MAX_TORTUOSITY = 2.5
ALLOW_LOOPS = False
AUTO_LOCAL_REANALYSIS = False

ROI_EDGE_QC_DISTANCE_UM = 1.0

Keep legacy pixel-valued keys for loading older parameter JSON files.

When both a physical-unit key and its legacy pixel equivalent are available,
the physical-unit key must take precedence.

Examples:

- MAX_BRIDGE_UM takes precedence over MAX_BRIDGE_PX
- MIN_SKEL_LEN_UM takes precedence over MIN_SKEL_LEN_PX
- MAX_WIDTH_UM takes precedence over MAX_WIDTH_PX
- MAX_GEODESIC_LEN_UM takes precedence over MAX_GEODESIC_LEN_PX
- MAX_BRANCH_LEN_UM takes precedence over MAX_BRANCH_LEN_PX

Add a function similar to:

resolve_pixel_parameters(cfg)

It must convert physical values to pixel values using UM_PER_PX_XY.

Return a structured dictionary containing both:

- physical-unit values
- resolved pixel values

Use safe rounding rules:

- minimum and maximum length thresholds may remain floating point
- morphology structuring-element radii should resolve to nonnegative integers
- bridge distance should resolve to a nonnegative integer
- Gaussian and ridge sigmas may remain floating point

Include resolved values in output provenance and preprocessing QC.

===============================================================================
PHASE 4 â€” STACK-LEVEL PREPROCESSING CONTEXT
===============================================================================

Add a dataclass named:

StackPreprocessContext

It must contain at least:

- normalization_low
- normalization_high
- selected_clahe_clip
- selected_clahe_profile
- contrast_score
- sampled_z_indices
- roi_percentiles
- saturation_fraction
- slice_brightness_statistics
- source_dtype
- inferred_bit_depth
- resolved_pixel_parameters
- configuration_provenance
- image_shape
- roi_pixel_count
- excluded_pixel_count

Add a function similar to:

build_stack_preprocess_context(
    image_files,
    roi_mask,
    cfg,
    exclusion_mask=None
)

Requirements:

1. Select representative slices distributed across the full stack.

2. Use approximately evenly spaced indices.

3. Do not select only one consecutive middle block.

4. Avoid duplicate indices in short stacks.

5. Read representative images using the pipeline's robust image-reading
   function.

6. Validate that all sampled images have compatible dimensions.

7. Pool only finite pixels that are:

   - inside the ROI
   - outside the exclusion mask

8. Calculate robust pooled statistics:

   - p1
   - p20
   - p50
   - p95
   - p99.5

9. Calculate:

   - robust dynamic range
   - slice-level median intensity
   - slice-level p95 intensity
   - slice-to-slice brightness coefficient of variation
   - saturation fraction
   - source dtype
   - inferred bit depth

10. The saturation calculation must account for integer and floating-point
    images appropriately.

11. Select exactly one CLAHE profile for the full stack:

    - no_clahe
    - high_contrast
    - standard
    - low_signal

12. The automatic profile should use a documented robust contrast metric.

13. The CLAHE profile must not be selected independently for every slice.

14. Emit clear warnings for:

    - missing ROI
    - empty ROI
    - ROI with too few pixels
    - empty valid pixels after exclusion
    - very low dynamic range
    - excessive saturation
    - nonfinite data
    - inconsistent image dimensions
    - invalid calibration

15. Save the selected stack preprocessing context as:

    stack_preprocessing_qc.json
    stack_preprocessing_qc.csv

16. Ensure all values written to JSON are serializable Python scalar types.

===============================================================================
PHASE 5 â€” ROI-AWARE SLICE SEGMENTATION
===============================================================================

Change the v5.6 segment_slice interface to support:

segment_slice(
    img_raw,
    cfg,
    z_idx=None,
    debug_dir=None,
    roi_mask=None,
    preprocess_context=None,
    exclusion_mask=None
)

Maintain compatibility with existing callers where practical.

When an ROI is supplied, implement this sequence:

1. Validate that roi_mask has the same full-image shape as img_raw.

2. Validate the exclusion mask when present.

3. Combine valid pixels as:

   valid_mask = roi_mask AND NOT exclusion_mask

4. Determine the bounding box of the ROI.

5. Expand the bounding box by ROI_CROP_PADDING_PX.

6. Clip the bounding box safely to the image dimensions.

7. Crop:

   - raw image
   - ROI mask
   - exclusion mask

8. Do not set all outside-ROI crop pixels to zero before filtering.

9. Fill pixels outside the ROI polygon, but inside the crop, using a robust
   median derived from valid ROI pixels.

10. The purpose of median filling is to prevent the artificial high-gradient
    ridge that would be produced by a zero-valued ROI boundary.

11. Keep the exact ROI mask separately and reapply it after thresholding and
    after every subsequent operation that could add pixels.

12. Normalize using stack-level robust limits from StackPreprocessContext.

13. Allow limited slice adaptation using NORM_STACK_WEIGHT.

For example:

    blended_low =
        NORM_STACK_WEIGHT * stack_low
        + (1 - NORM_STACK_WEIGHT) * slice_low

    blended_high =
        NORM_STACK_WEIGHT * stack_high
        + (1 - NORM_STACK_WEIGHT) * slice_high

14. Calculate slice_low and slice_high using only valid ROI pixels.

15. Clip normalized values safely to [0, 1].

16. Handle near-zero normalization ranges without division errors.

17. Apply Gaussian denoising using DENOISE_SIGMA_UM converted to pixels.

18. Apply the single stack-selected CLAHE profile.

19. Apply broad background estimation using BG_SIGMA_UM converted to pixels.

20. Calculate a foreground image by subtracting the broad background.

21. Normalize foreground response robustly where needed.

22. Run Meijering ridge enhancement using only the narrow scales resolved from:

    RIDGE_SIGMAS_UM = [0.60, 0.90, 1.20, 1.50]

23. Do not silently reintroduce the broad sigma 3 and sigma 4 scales from v5.5.

24. Calculate hysteresis percentile values using ridge pixels only from:

    ROI AND NOT exclusion mask

25. Use:

    THRESHOLD_HI = 91
    THRESHOLD_LO = 83

as the default starting values, while allowing loaded or tuned values.

26. Enforce:

    THRESHOLD_LO < THRESHOLD_HI

27. Apply hysteresis thresholding.

28. Immediately set all pixels outside the ROI or inside the exclusion mask to
    False.

29. Reapply the valid mask after:

    - hysteresis
    - morphological closing
    - hole filling
    - small-object removal
    - skeletonization
    - skeleton bridging
    - branch pruning
    - junction breaking
    - recursive or local repair, when enabled

30. CLOSE_RADIUS should default to zero.

31. MIN_HOLE_AREA should default to zero.

32. AUTO_LOCAL_REANALYSIS must default to False.

33. ALLOW_LOOPS must default to False.

34. Map all returned arrays back into full-image coordinates.

This includes:

- normalized image
- denoised image
- CLAHE image
- estimated background
- foreground image
- ridge response
- hysteresis mask
- cleaned mask
- clean skeleton
- bridged skeleton
- pruned skeleton
- label image
- distance map
- measurement coordinates
- centroids
- endpoints

35. Do not return crop-relative centroids to downstream tracking.

36. Do not union separate full-frame and ROI-crop segmentations in the default
    v5.6 path.

37. Keep any legacy two-pass behavior only behind:

    LEGACY_TWO_PASS_ROI = True

38. Clearly mark legacy mode in logs and output provenance.

39. When no ROI is supplied, retain a full-frame fallback using robust
    percentile normalization rather than absolute image min-max normalization.

===============================================================================
PHASE 6 â€” BRIDGING AND SKELETON CONTROL
===============================================================================

Update endpoint bridging.

Requirements:

1. Use MAX_BRIDGE_UM converted to pixels.

2. Do not bridge beyond that physical distance.

3. Never bridge outside the ROI.

4. Never bridge through the exclusion mask.

5. Estimate local tangent direction near each endpoint.

6. Reject bridge candidates when the endpoint tangent directions are
   inconsistent.

7. Use:

   MAX_BRIDGE_ANGLE_DEG = 35

as the default maximum angular incompatibility.

8. Preserve existing bridge-cost or geometric logic where it remains useful.

9. Log:

   - skeleton pixels before bridging
   - skeleton pixels after bridging
   - number of proposed bridges
   - number rejected by distance
   - number rejected by orientation
   - number rejected by ROI
   - number rejected by exclusion mask
   - number accepted

10. Preserve BREAK_JUNCTIONS behavior if currently used.

11. Keep MAX_BRANCH_NODES at zero by default for accepted final detections.

12. Keep MAX_ENDPOINT_COUNT at four by default.

13. Do not allow branching or bridging logic to add pixels outside the valid
    mask.

===============================================================================
PHASE 7 â€” BATCH, GUI, SINGLE-IMAGE, AND PREVIEW INTEGRATION
===============================================================================

Update every v5.6 call to segment_slice.

This includes:

- batch processing
- single-image analysis
- GUI preview
- debug-image generation
- ROI preview
- tuner segmentation evaluation
- tracking preparation
- any audit or review helper that resegments images

For batch mode:

1. Load or draw the ROI before building StackPreprocessContext.

2. Load EXCLUSION_MASK_PATH when supplied.

3. Validate the exclusion-mask shape.

4. Build one StackPreprocessContext for the complete stack.

5. Pass the same context to every segment_slice call.

6. Pass roi_mask to every segment_slice call.

7. Pass exclusion_mask to every segment_slice call.

8. Do not build a new preprocessing context separately for each slice.

9. Save:

   roi_mask_used.tif
   exclusion_mask_used.tif, when supplied
   stack_preprocessing_qc.json
   stack_preprocessing_qc.csv

10. Include preprocessing provenance in:

    - text report
    - Excel workbook
    - PDF report
    - PowerPoint report
    - run logs
    - parameter JSON output

11. Record:

    - selected CLAHE profile
    - CLAHE clip
    - normalization percentiles
    - representative Z indices
    - resolved physical-unit parameters
    - ROI area
    - excluded area
    - preprocessing mode
    - legacy-mode state

For GUI preview:

1. Preview must use the selected ROI during preprocessing.

2. Preview must not display results generated by full-frame thresholding followed
   only by ROI clipping.

3. Where a full stack context is unavailable, build a temporary context from the
   currently loaded stack or selected representative images.

4. Clearly identify temporary preview context in logs.

===============================================================================
PHASE 8 â€” DEBUG OUTPUTS
===============================================================================

For each requested debug slice, save these stages:

01_raw_robust_normalized
02_denoised
03_clahe
04_background
05_foreground
06_ridge
07_hysteresis
08_clean
09_skeleton_clean
10_skeleton_bridged
11_skeleton_pruned
12_final_detections

Use lossless TIFF or PNG as appropriate.

Also create one labeled debug montage per slice.

The montage must include:

- raw image
- normalized image
- denoised image
- CLAHE image
- foreground
- ridge response
- hysteresis mask
- clean mask
- clean skeleton
- bridged skeleton
- pruned skeleton
- final overlay

Save a debug JSON record for each slice containing:

- z index
- stack normalization low
- stack normalization high
- slice normalization low
- slice normalization high
- blended normalization low
- blended normalization high
- selected CLAHE profile
- selected CLAHE clip
- denoise sigma in micrometers
- denoise sigma in pixels
- background sigma in micrometers
- background sigma in pixels
- ridge sigmas in micrometers
- ridge sigmas in pixels
- numeric ridge high threshold
- numeric ridge low threshold
- valid ROI pixel count
- foreground occupancy inside ROI
- foreground occupancy outside ROI
- foreground occupancy inside exclusion mask
- skeleton pixels before bridging
- skeleton pixels after bridging
- bridge inflation fraction
- final detection count
- median final detection length
- median final detection width
- ROI-edge detection fraction
- exclusion-mask-overlap detection count

After the ROI is applied:

- outside-ROI foreground occupancy must be zero
- exclusion-mask foreground occupancy must be zero
- outside-ROI skeleton occupancy must be zero
- exclusion-mask skeleton occupancy must be zero

Debug visualization may display the outside-ROI region as black.

Actual filtering must use the median-filled crop rather than a zero-valued
polygon edge.

===============================================================================
PHASE 9 â€” V5.6 TUNER
===============================================================================

Modify only the new v5.6 tuner file.

It must import the v5.6 pipeline rather than v5.5.

Update:

- dynamic import path
- module alias
- output names
- result directories
- summary labels
- ROI filenames
- version strings
- debug filenames

Default output directory:

parameter_tuning_results_v5_6

Support these modes:

--mode profile
--mode segmentation
--mode tracking

Keep existing useful arguments and add:

--slices auto
--auto-slice-count 6
--roi-mask PATH
--exclusion-mask PATH
--profile standard
--profile low_signal
--profile high_contrast
--profile no_clahe
--profile auto
--base-params PATH
--save-all-debug-candidates
--review-candidates INTEGER

Allow --base-params to be repeated.

Use argparse action="append" or equivalent.

Merge repeated parameter files in the order supplied, where later files
override earlier files.

Example:

--base-params preprocessing.json
--base-params segmentation.json
--base-params tracking.json

===============================================================================
PHASE 10 â€” TUNER ROI AND CONTEXT REQUIREMENTS
===============================================================================

Every tuner segmentation call must pass:

- roi_mask_global
- preprocess_context_global
- exclusion_mask_global

Remove the v5.5 pattern that:

- calls segment_slice with roi_mask=None
- segments the full frame
- filters detections to the ROI afterward

Build one StackPreprocessContext before optimization begins.

Reuse the identical context for every optimization candidate within one run.

Do not recalculate stack normalization or automatically select a different
CLAHE profile for every candidate.

This is required for:

- fair candidate comparison
- deterministic behavior
- speed
- reproducibility

===============================================================================
PHASE 11 â€” AUTOMATIC REPRESENTATIVE SLICE SELECTION
===============================================================================

When:

--slices auto

is used:

1. Select slices distributed across the full stack.

2. Default to six slices.

3. Approximate:

   - first
   - 20%
   - 40%
   - 60%
   - 80%
   - last

4. Avoid duplicate indices.

5. Save selected Z indices in all tuner summaries.

6. Print them before optimization begins.

7. Nonconsecutive slices are acceptable for:

   - profile mode
   - segmentation mode

8. Tracking mode must use consecutive slices.

9. Warn and stop or require explicit override when nonconsecutive slices are
   supplied for tracking optimization.

===============================================================================
PHASE 12 â€” PREPROCESSING PROFILE MODE
===============================================================================

Implement profile mode.

Compare these stack-wide profiles:

1. no_clahe
2. high_contrast, clip 0.010
3. standard, clip 0.025
4. low_signal, clip 0.035
5. auto, selected by stack QC

During profile comparison:

- keep ridge scales fixed
- keep background sigma fixed
- keep physical morphology parameters fixed
- use the same ROI
- use the same exclusion mask
- use the same representative slices
- use the same stack normalization context where applicable

For each profile calculate:

- per-slice detection count
- count median
- count coefficient of variation
- median geodesic length
- mean geodesic length
- median width
- median length-to-width ratio
- short-fragment fraction
- long-object fraction
- very-long-object fraction
- wide-object fraction
- low-length-width-ratio fraction
- high-tortuosity fraction
- loop fraction
- branch fraction
- ROI-edge fraction
- exclusion-mask-overlap count
- hysteresis occupancy
- clean-mask occupancy
- bridge inflation
- score

Do not reward a profile solely for producing more detections.

Strongly penalize:

- high binary occupancy
- broad tissue-edge detections
- large bridge inflation
- loops
- branching
- ROI-edge artifacts
- detections overlapping exclusion masks
- excessive short fragments
- giant connected components
- unstable slice counts

Save:

best_preprocessing_profile_v5_6_###.json
profile_comparison_v5_6_###.csv
profile_comparison_v5_6_###.json
profile_review_v5_6_###.pdf

For each profile, save representative review panels containing:

- raw
- normalized
- CLAHE
- ridge
- hysteresis
- final overlay
- length distribution
- QC summary

===============================================================================
PHASE 13 â€” SEGMENTATION TUNING SPACE
===============================================================================

Use this default v5.6 segmentation search space:

SEGMENTATION_PARAM_SPACE = [
    ("THRESHOLD_HI",              88.0, 94.0, False),
    ("THRESHOLD_LO",              80.0, 87.0, False),
    ("MIN_OBJ_PX",                 6,   12,   True),
    ("MAX_BRIDGE_UM",              0.0,  2.0, False),
    ("MIN_SKEL_LEN_UM",            5.5,  8.5, False),
    ("MAX_WIDTH_UM",               3.0,  5.0, False),
    ("MIN_LENGTH_WIDTH_RATIO",     2.2,  3.2, False),
    ("MAX_TORTUOSITY",             1.8,  3.0, False)
]

Enforce THRESHOLD_LO < THRESHOLD_HI.

Do not tune CLAHE independently in segmentation mode.

CLAHE must come from:

- a selected preprocessing profile
- a loaded preprocessing JSON
- the automatic stack profile

Do not tune BG_SIGMA_UM or RIDGE_SIGMAS_UM during normal per-dataset
segmentation tuning.

Those parameters describe expected biological scale and should remain fixed
unless a future explicit profile-development mode is added.

===============================================================================
PHASE 14 â€” SEGMENTATION OBJECTIVE
===============================================================================

Update the segmentation objective.

Raw detection count must be a guardrail, not the dominant reward.

Reward:

- median 2D length near approximately 9 to 10 micrometers
- stable counts across representative slices
- reasonable median width
- high median length-to-width ratio
- low short-fragment fraction
- low long-merge fraction
- low very-long-object fraction
- low wide-object fraction
- low branch fraction
- low loop fraction
- low ROI-edge fraction
- low bridge inflation
- zero exclusion-mask overlap
- consistent morphology across representative slices

Penalize strongly:

- monster components
- broad tissue-edge detections
- excessive binary-mask occupancy
- excessive ridge occupancy
- excessive loops
- excessive branching
- large skeleton increase during bridging
- objects touching the exclusion mask
- excessive detections near the ROI border
- unstable counts
- low length-to-width ratio
- excessive width
- extreme tortuosity
- optimization values resting exactly on search-space boundaries when
  morphology is poor

Keep count limits as sanity bounds.

A large count must not compensate for poor morphology.

Save all objective subcomponents in the tuning CSV and JSON results so the score
is auditable.

===============================================================================
PHASE 15 â€” TOP-CANDIDATE REVIEW OUTPUT
===============================================================================

For each top segmentation candidate, save a multi-stage review panel.

Include:

- raw image
- robust normalized image
- CLAHE image
- ridge response
- hysteresis mask
- clean mask
- final overlay
- length histogram
- candidate parameters
- preprocessing profile
- all score penalties
- count
- median length
- median width
- length-to-width ratio
- mask occupancy
- bridge inflation
- ROI-edge fraction

Do not save only raw plus final overlay.

Make visual comparison between top candidates straightforward.

===============================================================================
PHASE 16 â€” TRACKING TUNING
===============================================================================

Preserve the existing useful v5.5 tracking tuner behavior and safe-candidate
rules unless required by the v5.6 interface.

Tracking mode must:

1. Use consecutive slices.

2. Use one fixed preprocessing profile.

3. Use one fixed segmentation parameter set.

4. Segment all tracking-tuning images once.

5. Cache segmentation results.

6. Tune only tracking parameters after segmentation caching.

7. Do not rerun segmentation for every tracking candidate.

8. Save preprocessing provenance.

9. Save segmentation provenance.

10. Save selected Z indices.

11. Preserve biological-candidate and quality-audit summaries.

12. Preserve deterministic random seed behavior.

===============================================================================
PHASE 17 â€” CONFIGURATION VALIDATION
===============================================================================

Extend validate_config or equivalent to verify:

- calibration values are positive
- normalization percentiles are between 0 and 100
- normalization low percentile is less than high percentile
- hysteresis low percentile is less than high percentile
- CLAHE mode is valid
- preprocessing mode is valid
- stack weight is between 0 and 1
- CLAHE clips are nonnegative
- ROI crop padding is nonnegative
- ridge-sigma list is nonempty
- ridge sigmas are positive
- background sigma is positive
- denoise sigma is nonnegative
- physical morphology values are nonnegative
- maximum lengths exceed minimum lengths where relevant
- maximum bridge angle is between 0 and 180
- exclusion-mask shape matches the image
- ROI shape matches the image
- physical-unit keys override pixel keys deterministically
- no ambiguous duplicate values remain after parameter resolution

Warnings should be informative and identify the offending parameter.

===============================================================================
PHASE 18 â€” AUTOMATED TESTS
===============================================================================

Create:

tests/test_saturn_v56_preprocessing.py

Add any additional minimal test files required.

Tests must use synthetic images and small arrays so they run quickly.

Required tests:

1. OFF-ROI BRIGHT-OBJECT INVARIANCE

   Construct an ROI containing synthetic elongated nuclei.
   Add an extremely bright elongated object outside the ROI.
   Segmentation inside the ROI must remain unchanged or nearly unchanged.

2. ROI ZERO-LEAK TEST

   Hysteresis mask, clean mask, skeleton, bridged skeleton, pruned skeleton, and
   final labels must contain zero pixels outside the ROI.

3. EXCLUSION-MASK ZERO-LEAK TEST

   The same arrays must contain zero pixels inside an exclusion mask.

4. ROI-BOUNDARY ARTIFACT TEST

   Confirm that median filling outside the polygon does not create a strong
   artificial ridge along the ROI boundary.

5. MULTIPLICATIVE BRIGHTNESS INVARIANCE

   Compare equivalent images after multiplicative scaling.

   Detection count and median length must remain within documented tolerance.

6. ADDITIVE BRIGHTNESS INVARIANCE

   Compare equivalent images after additive intensity offset.

7. BIT-DEPTH INVARIANCE

   Compare equivalent uint8 and uint16 representations.

8. STACK-WIDE CLAHE TEST

   Confirm that CLAHE profile selection occurs once per stack context and not
   independently per slice.

9. CROP MAPPING TEST

   Confirm full-image array shapes and correct full-image centroid coordinates
   after ROI-crop processing.

10. PHYSICAL-UNIT CONVERSION TEST

    Test UM_PER_PX_XY at multiple values and verify resolved bridge, width, and
    length thresholds.

11. EXCLUSION-MASK DETECTION TEST

    Confirm detections crossing or centered inside the exclusion mask are
    rejected.

12. BRIDGE-DISTANCE TEST

    Confirm endpoints beyond MAX_BRIDGE_UM are not joined.

13. BRIDGE-ORIENTATION TEST

    Confirm geometrically close endpoints with incompatible tangent directions
    are not joined.

14. BRIDGE-ROI TEST

    Confirm no bridge crosses outside the ROI.

15. TUNER ROI-PASSING TEST

    Monkeypatch or instrument segment_slice and confirm the tuner passes:

    - ROI mask
    - preprocessing context
    - exclusion mask

16. CONTEXT-REUSE TEST

    Confirm the same preprocessing-context instance or equivalent immutable
    context is reused for optimization candidates.

17. BASE-PARAMETER MERGE TEST

    Confirm repeated --base-params files merge in supplied order.

18. AUTO-SLICE-SELECTION TEST

    Confirm representative slices cover the stack and contain no duplicates.

19. DETERMINISTIC-SEED TEST

    Confirm identical inputs and random seed produce repeatable candidate
    ordering or scores within numerical tolerance.

20. VERSION-ISOLATION TEST

    Verify all v5.5 source hashes match their pre-edit values.

===============================================================================
PHASE 19 â€” SELF-CHECK MODE
===============================================================================

Extend the v5.6 tuner --self-check so it verifies:

- v5.6 module import
- v5.6 version string
- ROI passed during segmentation
- preprocessing context passed during segmentation
- exclusion mask honored
- physical parameters resolve correctly
- threshold ordering
- automatic slice selection
- repeated base-parameter merge order
- profile output naming
- segmentation output naming
- deterministic seed
- no v5.5 imports remain in normal v5.6 execution

The self-check must not start differential evolution.

===============================================================================
PHASE 20 â€” DOCUMENTATION
===============================================================================

Create these files at an appropriate repository documentation location or root:

V5_6_PIPELINE_IMPLEMENTATION.md
V5_6_TUNER_IMPLEMENTATION.md
V5_6_VALIDATION_REPORT.md

V5_6_PIPELINE_IMPLEMENTATION.md must describe:

- exact source file created
- major changed functions
- ROI-aware preprocessing sequence
- StackPreprocessContext
- new configuration keys
- physical-unit conversion
- exclusion-mask support
- bridging changes
- debug outputs
- backward compatibility
- known limitations

V5_6_TUNER_IMPLEMENTATION.md must describe:

- exact tuner file created
- profile mode
- segmentation mode
- tracking mode
- representative-slice selection
- new parameter search space
- scoring changes
- repeated base-parameter loading
- ROI/context reuse
- example commands

V5_6_VALIDATION_REPORT.md must include:

- branch name
- starting commit SHA
- modified and created files
- every validation command run
- compile results
- test results
- self-check result
- failures or skipped tests
- confirmation that the full microscopy batch was not run
- confirmation that v5.5 hashes remained unchanged
- unresolved assumptions
- readiness assessment for representative-slice testing

Do not claim real-image validation if the raw TIFF stack is unavailable.

===============================================================================
PHASE 21 â€” COMMANDS TO RUN
===============================================================================

After implementation, run only:

python -m py_compile <exact-v5.6-pipeline-path>

python -m py_compile <exact-v5.6-tuner-path>

python -m pytest -q

python <exact-v5.6-tuner-path> --self-check

git diff --check

git status --short

git diff --stat

git diff --name-only

Do not run the complete 88-slice batch.

Do not run a long differential-evolution optimization.

Do not overwrite any existing batch output.

===============================================================================
PHASE 22 â€” FINAL VERSION-ISOLATION VERIFICATION
===============================================================================

At the end:

1. Recalculate SHA-256 hashes for all v5.5 Python files.

2. Compare them with the pre-edit hashes.

3. Fail validation if any v5.5 source changed.

4. Confirm no v5.5 file is listed as modified by Git.

5. Confirm the current branch remains:

   feature/saturn-v5.6-roi-adaptive

6. Do not commit.

7. Do not push.

===============================================================================
PHASE 23 â€” FINAL RESPONSE
===============================================================================

Return a concise implementation report containing:

1. Repository paths discovered.

2. Files created.

3. Files modified.

4. Major v5.6 architectural changes.

5. Test results.

6. Self-check results.

7. Compile results.

8. Confirmation that v5.5 remained unchanged.

9. Any failed or skipped checks.

10. Remaining risks before running on real microscopy images.

11. Exact commands for:

    - profile comparison
    - representative-slice segmentation tuning
    - consecutive-slice tracking tuning
    - a non-destructive representative-slice smoke test

Do not state that v5.6 is ready for production unless all automated checks pass.

Do not commit or push changes automatically.
