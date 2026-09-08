# Saturn v5.7.1 Design Decisions

This ledger records the intended production semantics that reviewers must test.
It explains why a behavior exists; it does not override contradictory evidence
or excuse a defect.

## Comparative biological population

- The primary WT-versus-mutant population is `technical_valid`.
- Short, long, wide, thin, curved, tortuous, irregular, and single-slice
  morphology remains measurable and may receive a morphology warning.
- WT-like length, width, ratio, count, or shape must not be an optimization
  target or a technical acceptance rule.
- A 15-20 um object is retained with a review annotation. Length above 20 um is
  not sufficient evidence to delete or split an object. Objective fusion or
  merge evidence is required before a technical intervention.

## U-Net-primary segmentation

- The dual-head U-Net is the primary segmentation source. Classical morphology
  may annotate supported instances but must not veto them for unusual shape.
- Foreground probability defines supported mask extent; learned core components
  provide independent separation evidence for touching objects.
- Instance splitting must not be driven solely by a desired biological length.
- The original filled parent mask, parent identity, and split evidence must
  remain auditable whenever an objective split is applied.

## Cross-slice tracking and gaps

- A short 2D observation may be the optical tip of a valid multi-plane nucleus
  and remains eligible for tracking.
- One missing Z plane may be bridged when calibrated position, motion, and
  overlap/support evidence are compatible. This handles a faint or missed
  intermediate optical section.
- Gap linking does not invent a 2D detection, mask, area, or width on the
  missing plane. Observed-mask volume sums observed masks only.
- Single-slice tracks remain valid because specimen orientation and Z spacing
  can make a complete nucleus visible primarily in one plane.
- A proposed impossible join is rejected without deleting its original 2D
  detections.

## Width, area, and volume are measured from intensity, not from the mask edge

- The mask boundary reproduces the training annotation convention rather than the
  nucleus. Annotations in this project have a median width of 1.606 um while the
  optical width of a nucleus is about 0.643 um, so the learned mask is roughly
  2.4 times too wide. The bias is not constant: it was 2.32x in KJ-01 against
  2.51x in WT-01 and rises with brightness, which is larger than the 0.169 um
  width difference between those specimens. Mask width therefore cannot carry a
  genotype comparison.
- Primary width is the half-maximum extent of the background-corrected intensity
  profile along centerline normals. It is independent of where a boundary was
  drawn: inflating a mask from 1.9 to 6.4 um leaves it unchanged at 0.797 um.
- Integrated signal per profile is recorded alongside it. Blur conserves light,
  so integration does not saturate below the resolution limit and is about 26
  times more sensitive to a real width change. It is, however, proportional to
  staining brightness, which the half-maximum is immune to. The two are reported
  together: agreement is evidence, disagreement is a flag.
- Area and volume are derived as centerline length times profile width, because a
  filament's footprint is length times width. Summing mask pixels inflated volume
  by 2.10x on KJ-01. The mask pixel count remains available as
  `instance_mask_area_px` for diagnostics only.
- This replaces the previous mask-derived area and volume outright rather than
  retaining them as legacy fields, which departs from the usual rule in
  `AGENTS.md` about preserving prior measurements. The owner authorized the
  replacement because the pipeline has not yet produced a real biological run, so
  no result depends on the superseded values, and carrying an inflated duplicate
  would risk it being reported. Width retains its legacy fields as usual.
- Dilation is never used in a measurement path. Profile background is read from
  the far tails of the same profile, taking the quieter side, because a dilated
  ring would let the chosen radius set the background, the half maximum and
  therefore the width. Dilation remains acceptable only for display overlays.
- Merges are detected from the profile: a cross-section through two filaments is
  bimodal. Peak counting is restricted to pixels inside the instance mask so a
  neighbouring nucleus is not mistaken for a merge.
- Absolute nucleus diameter is not established and must never be reported. At
  0.378 um per pixel against a 0.23 um point spread function the image is 3.3
  times below Nyquist, and the deconvolved estimate saturates near 0.6 um.
  Comparison between groups is supported; an absolute width claim is not. This
  caveat travels with the metric wherever it is calculated or plotted.

## Measurements

- Primary length follows the final instance-mask centerline and remains separate
  from centroid trajectory and legacy fields.
- Primary apparent body width uses subpixel perpendicular contour chords after
  endpoint trimming. Legacy distance-transform width remains explicitly
  labelled and must not drive biological reports.
- A reconstructed track receives representative width from the technically
  valid observed plane with the largest filled-mask area. Missing width remains
  unavailable; it is not fabricated from a gap.
- Width is apparent mask width and is sensitive to segmentation boundary,
  annotation thickness, focus, and lateral PSF. It is not a deconvolved
  molecular diameter.

## Calibration and ROI

- Leica calibration must be resolved before any physical threshold or
  measurement is applied.
- Organized filenames may differ from Leica source names, so the retained
  manifest-to-XML mapping is authoritative and must be hashed and archived.
- Every run archives the exact ordered source files, applied ROI/exclusion mask,
  profile, checkpoint identity, and resolved calibration.
- A repeated 2D ROI supports sampled-area normalization. It does not establish
  anatomical seminal-vesicle volume; that requires slice-specific 3D organ
  masks.

## Reporting and review burden

- Biological reports show one primary technical-valid population and
  specimen-level outcomes. Internal rescue lanes and detailed audit categories
  belong in technical QC, not the main PDF.
- Individual nuclei are nested observations. Biological inference uses
  specimens as replicates and is unavailable when group sample size is
  insufficient.
- Detailed visual evidence exists for software validation and adversarial audit;
  it must not become a routine manual-review queue for the biologist.
