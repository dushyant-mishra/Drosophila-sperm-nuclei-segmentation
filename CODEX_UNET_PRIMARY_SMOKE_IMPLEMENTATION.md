# Saturn U-Net-primary implementation and three-image smoke test

Work only in the current repository and current feature branch.

## Safety constraints

1. Do not modify, delete, rename, or reformat historical Saturn versions.
2. Do not modify files under archive/ unless a test imports one accidentally and
   the import must instead be corrected.
3. Do not alter v5.5 behavior, hashes, outputs, or frozen files.
4. Preserve existing behavior for:
   - SEGMENTATION_ENGINE = classical_saturn
   - SEGMENTATION_ENGINE = unet_assisted
   - SEGMENTATION_ENGINE = hybrid
5. Do not run a full-stack analysis.
6. Do not run parameter tuning.
7. Do not run biological genotype comparisons.
8. Do not commit or push.
9. Do not optimize toward a desired nucleus count, length, width, shape, WT
   phenotype, or mutant phenotype.
10. Begin by inspecting git status and the relevant functions. Never use git
    reset, checkout --, restore, clean, or another destructive command.

Primary files:

- sperm_segmentation_saturnv5.7.py
- utils/tune_parameters_Saturnv5_7.py

Create:

- tests/test_saturn_v57_unet_primary.py
- scratch/run_v57_unet_primary_smoke.py
- docs/UNET_PRIMARY_SMOKE_TEST.md

## Scientific objective

Implement a real:

    SEGMENTATION_ENGINE = "unet_primary"

The U-Net probability map must be the authoritative foreground evidence.

The initial thresholds are:

    low candidate threshold = 0.05
    high seed threshold      = 0.30

These are starting smoke-test values, not finalized biological parameters.

The main flow must be:

    U-Net probability
    -> true low/high hysteresis
    -> ROI and exclusion restriction
    -> tiny technical-noise removal
    -> instance separation
    -> one filled mask per instance
    -> one centerline per instance
    -> measurement
    -> optional classical additions

The classical pipeline must not veto a U-Net-supported instance merely because
it is short, wide, unusually curved, tortuous, or has a low length-to-width
ratio.

## Part 1: engine and configuration

Add "unet_primary" as a supported SEGMENTATION_ENGINE value everywhere needed.

Update configuration validation so that:

1. SEGMENTATION_ENGINE is one of:
   classical_saturn, unet_assisted, hybrid, unet_primary.

2. For unet_primary:

       0 <= UNET_CANDIDATE_THRESHOLD
         < UNET_SEED_THRESHOLD
         <= 1

3. U-Net inference fails explicitly when neither a valid checkpoint nor a
   compatible cached probability map is available.

4. No silent classical-only fallback is permitted.

Add minimal configuration controls:

    UNET_PRIMARY_MIN_COMPONENT_PX
    UNET_PRIMARY_CLASSICAL_ADDITIONS_ENABLE
    UNET_PRIMARY_CLASSICAL_EXCLUDE_DILATION_PX
    UNET_PRIMARY_RETAIN_MORPHOLOGY_WARNINGS
    UNET_PRIMARY_SAVE_FILLED_MASK_OVERLAY
    UNET_PRIMARY_SAVE_INSTANCE_OVERLAY

Use existing candidate and seed threshold keys rather than creating duplicate
threshold keys.

For the smoke test, default:

    UNET_PRIMARY_CLASSICAL_ADDITIONS_ENABLE = False

This isolates U-Net-primary behavior. Classical additions can be tested later.

## Part 2: probability inference

Refactor only as much as needed so U-Net probability loading works for:

- unet_assisted
- hybrid
- unet_primary

Both model inference and _UNET_PROBABILITY_CACHE must remain supported.

Do not make U-Net inference depend on whether the classical ridge mask is
nonempty.

The returned probability map must:

- have the same shape as the raw image;
- be finite;
- be in [0, 1];
- be zero outside the ROI;
- be zero inside the exclusion mask.

## Part 3: authoritative hysteresis foreground

Implement a focused helper, with a clear name such as:

    _build_unet_primary_foreground(...)

Use:

    apply_hysteresis_threshold(
        unet_probability,
        UNET_CANDIDATE_THRESHOLD,
        UNET_SEED_THRESHOLD,
    )

Then intersect strictly with:

    roi_mask & ~exclusion_mask

Remove only components below UNET_PRIMARY_MIN_COMPONENT_PX.

Every retained hysteresis component must contain at least one pixel at or above
UNET_SEED_THRESHOLD. Record components that fail this requirement as
"no_high_confidence_seed".

Do not use expected biological length, width, tortuosity, or aspect ratio when
building this foreground.

## Part 4: correct instance splitting

Correct _split_unet_rescue_instances or replace it with a general helper usable
by both U-Net rescue and U-Net-primary.

Important bug:

Do not perform this after watershed:

    labels = measure.label(labels > 0)

That converts distinct touching watershed labels back into one connected
foreground object.

Preserve the integer watershed labels returned by skimage.segmentation.watershed.

Required behavior:

1. Process each connected hysteresis component separately.
2. Build deterministic markers from connected high-confidence seed regions.
3. Remove seed regions below the technical minimum size.
4. When there is exactly one valid marker:
   - retain the component as one instance unless there is strong technical
     evidence of an unresolved multi-object merge;
   - do not invent multiple instances merely to increase counts.
5. When there are multiple valid markers:
   - run watershed within that component;
   - preserve each watershed label;
   - assign globally unique sequential labels.
6. Every retained child instance must:
   - contain at least one valid high-confidence seed;
   - remain inside the parent component;
   - remain inside ROI and outside exclusion;
   - meet only the technical minimum component size.
7. Discarding a tiny watershed child must not erase or relabel the other valid
   children.
8. Output an audit table containing:
   - parent component ID
   - marker count
   - child instance count
   - child area
   - maximum probability
   - mean probability
   - contains seed
   - disposition
   - technical reason
9. Ensure deterministic output for identical input.

Do not force peak_local_max to create many markers along one long nucleus.
Use connected high-confidence seed regions as the preferred markers.
Peak generation may be used only as a conservative fallback for a component
already identified as technically complex, and the result must be auditable.

## Part 5: per-instance centerlines

For each filled U-Net-primary instance:

1. Skeletonize that instance independently.
2. Never skeletonize all touching instances as one binary union.
3. Preserve the mapping:

       filled instance ID
       -> centerline label ID
       -> measurement row ID

4. When the skeleton has a branch:
   - extract a deterministic longest geodesic centerline using the existing
     centerline/geodesic utilities where possible;
   - retain the instance;
   - record a morphology/topology warning.
5. Treat failure to obtain any valid centerline as a technical failure.
6. Do not hard-reject merely because the instance is:
   - short
   - wide
   - low aspect ratio
   - curved
   - tortuous

Create compatible fields in the segmentation dictionary:

    unet_primary_hysteresis_mask
    unet_primary_instance_labels
    unet_primary_centerline_labels
    unet_primary_rejected_reason
    unet_primary_component_audit
    unet_primary_debug

Keep the existing fields required by downstream code:

    mask_hyst
    mask_clean
    skel_clean
    skel_pruned
    skel_labeled
    dist_clean

For unet_primary, those compatibility fields must represent the U-Net-primary
population and must not silently revert to the classical ridge population.

## Part 6: measurements

Add a U-Net-primary measurement path while preserving existing classical and
hybrid behavior.

For each U-Net-primary instance:

1. Compute width from the filled mask of that individual instance, not from the
   distance transform of a union containing neighboring instances.
2. Compute the centerline/geodesic length from its mapped centerline.
3. Record:
   - filled mask area in pixels
   - geodesic length
   - median width
   - length-to-width ratio
   - tortuosity
   - endpoints
   - branch count
   - centroid
   - bounding box
   - orientation
   - mean U-Net probability
   - maximum U-Net probability
4. Preserve the existing output schema where possible.
5. Add fields rather than silently changing historical field meaning:

       instance_mask_area_px
       detection_source = "unet_primary"
       morphology_warning
       morphology_warning_reasons
       technical_failure
       technical_failure_reason
       parent_hysteresis_component_id

6. Short, wide, low-ratio, and tortuous status must be warning-only.
7. Hard technical failure may include only:
   - no high-confidence seed
   - tiny isolated noise
   - outside ROI/exclusion leakage
   - empty or invalid mask
   - no valid centerline
   - duplicate instance
   - unresolved multi-instance merge
   - invalid geometry or nonfinite measurement

Do not automatically reject a long instance solely because it exceeds a
biological length threshold. Flag it for merge review instead.

## Part 7: optional Saturn additions

Implement the contract for optional classical additions but leave it disabled
during the first smoke test.

When enabled:

1. Run classical Saturn only in residual valid space not represented by accepted
   U-Net-primary masks.
2. Dilate accepted U-Net masks by
   UNET_PRIMARY_CLASSICAL_EXCLUDE_DILATION_PX before searching residual space.
3. Classical detections may add new objects.
4. Classical detections must never remove, shorten, overwrite, or relabel an
   accepted U-Net-primary instance.
5. Use detection_source = "saturn_only_addition".
6. Deduplicate against U-Net instances before acceptance.

Do not implement aggressive classical bridging or extension in this first pass.
Those repairs must wait until the three-image instance-splitting smoke test is
visually reviewed.

## Part 8: overlays

Create separate overlays because filled masks and centerlines serve different
purposes.

For every smoke-test slice save:

1. raw image;
2. U-Net probability;
3. low/high hysteresis support;
4. uniquely colored filled instance labels;
5. accepted filled masks:
   - cyan = U-Net-primary
   - green = Saturn-only addition
   - yellow = repaired U-Net instance, reserved for later
   - red = hard technical failure
6. measured centerlines;
7. current hybrid result;
8. side-by-side hybrid versus U-Net-primary comparison.

Do not compare a filled U-Net mask only against a thin classical centerline
without labeling that distinction.

Overlay dilation must remain display-only.

## Part 9: three-image smoke runner

Create:

    scratch/run_v57_unet_primary_smoke.py

Required command-line arguments:

    --input-dir
    --unet-model
    --base-params
    --roi-mask
    --exclusion-mask
    --z-values
    --outdir
    --engines
    --repeat

Requirements:

1. Load the complete source stack index only to obtain z-1/z/z+1 context.
2. Process exactly the target Z values supplied by --z-values.
3. Default engines:

       hybrid,unet_primary

4. Default repeat = 2 to test determinism.
5. Do not run cross-slice tracking when target values are nonconsecutive.
6. Do not run the tuner.
7. Do not process unrequested target slices.
8. Reuse each probability map between the hybrid and U-Net-primary arms.
9. Write:

       smoke_summary_v5_7.json
       smoke_summary_v5_7.csv
       instance_audit_v5_7.csv
       technical_failures_v5_7.csv
       review_panels/
       probability_maps/

10. Per slice and per engine report:

       candidate pixels
       seed pixels
       hysteresis component count
       marker count
       split instance count
       accepted instance count
       morphology-warning count
       hard technical-failure count
       unresolved-merge count
       Saturn-only additions
       final measured object count
       outside-ROI pixels
       exclusion-mask pixels
       deterministic label hash

11. Quality gates:

       exactly requested target slices processed
       no outside-ROI output
       no exclusion-mask output
       no accepted instance lacking a >= seed-threshold pixel
       no duplicate instance IDs
       no collapsed watershed labels in synthetic regression test
       identical label hashes across repeat runs
       no silent U-Net failure or classical fallback

Do not fail a smoke test because object counts differ from historical counts.
Do not fail based on length, width, morphology, or genotype expectations.

## Part 10: tests

Create tests/test_saturn_v57_unet_primary.py with synthetic tests for:

1. Low-threshold pixels connected to a high seed are retained.
2. Low-threshold pixels without a high seed are removed.
3. ROI and exclusion masks are strictly respected.
4. Two touching objects with two seed regions remain two watershed labels.
5. The original watershed labels are not collapsed by binary relabeling.
6. One elongated object with one continuous seed remains one instance.
7. Separate instance distance transforms are used for width.
8. Morphology outliers are warning-only in unet_primary.
9. A seedless object cannot be accepted.
10. Classical, unet_assisted, and hybrid behavior remains unchanged.
11. U-Net-primary fails explicitly if neither cache nor checkpoint is available.
12. Repeated runs are deterministic.
13. Filled instance IDs and centerline IDs remain mapped one-to-one.
14. The smoke runner refuses more than a small explicitly supplied target set
    unless an explicit --allow-large-run flag is given.
15. The smoke runner processes exactly three targets in a mocked test.

## Validation commands

Run only:

    python -m py_compile sperm_segmentation_saturnv5.7.py
    python -m py_compile utils/tune_parameters_Saturnv5_7.py
    python -m py_compile scratch/run_v57_unet_primary_smoke.py
    python -m pytest -q tests/test_saturn_v57_unet_primary.py
    python -m pytest -q
    python utils/tune_parameters_Saturnv5_7.py --self-check
    git diff --check

Do not run real image inference until the user supplies the explicit smoke-test
command and paths.

At completion report:

- files changed;
- functions added or modified;
- confirmation that watershed labels are preserved;
- confirmation that only hard technical failures exclude U-Net-primary objects;
- all test results;
- any unresolved limitations;
- exact smoke-test command syntax.

Do not commit or push.
