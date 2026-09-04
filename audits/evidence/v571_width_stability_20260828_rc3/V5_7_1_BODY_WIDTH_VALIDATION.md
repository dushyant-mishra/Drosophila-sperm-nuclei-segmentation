# Saturn v5.7.1 Width Stability Check

## Scope

This automated check evaluates the versioned central-body contour-chord
measurement on known synthetic geometry and manually annotated COCO masks. It
validates engineering behavior, not true physical nucleus diameter.

## User Decision

- Primary biological-analysis field: `representative_body_width_um`
- Display name: apparent central-body mask width
- Engineering status: **PASS**
- Absolute biological accuracy: **NOT ESTABLISHED**
- Alternate width calculations remain technical QC and should not be selected
  by end users.

## Results

- COCO annotations: 5273
- Rasterized masks: 5273
- Widths measured: 5263
- Measurement success: 99.81%
- Maximum synthetic absolute error: 0.435 px
- Maximum synthetic rotation spread: 0.441 px
- Distinct legacy EDT widths at 0.001 px: 8
- Distinct central-body widths at 0.001 px: 1572
- Correlation with filled-mask area/length: 0.945
- Median absolute difference from area/length: 0.315 px
- P90 absolute difference from area/length: 1.045 px
- Median width increase after one-pixel mask dilation: 1.825 px
- P10-P90 mask-dilation width increase: 1.449 to 2.005 px
- Median width change after one-pixel mask erosion: -1.815 px
- Median erosion-to-dilation sensitivity span: 3.561 px

## Interpretation

The subpixel contour-chord measurement removes the severe pixel-grid banding of
the legacy centerline distance-transform median. The area/length comparison is
an independent mask-derived consistency check, not a competing user-facing
measurement and not proof of absolute biological accuracy. The erosion and
dilation results quantify how strongly the answer follows the learned mask
boundary; they are QC evidence only.

Model C uses `train_mask_dilate_px: 0`. The erosion and dilation measurements
above are deliberate validation perturbations only; neither operation is applied
during production inference. The field remains an apparent mask width because
its boundary is learned from annotations, and no unvalidated fixed subtraction
is applied.

## Limits

- Do not call this a PSF-corrected or molecular diameter.
- Do not subtract a fixed number from widths to force agreement with expected WT
  morphology.
- A later held-out boundary study may supersede this claim, but users should not
  choose among the QC variants in routine analysis.
