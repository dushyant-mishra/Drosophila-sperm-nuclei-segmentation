# v5.7.1 Annotation-Tolerant Target Audit

## Scope

This audit used the existing mixed KJ/WT plus replay package. No annotations
were redrawn. The production Saturn pipeline, inference thresholds, tracking,
and biological reporting were not changed.

## Dataset Accounting

| Item | Result |
|---|---:|
| Training images | 25 |
| Validation images | 4 |
| Source COCO annotations | 5,273 |
| Generated instance labels | 5,273 |
| Generated instance cores | 5,273 |
| Missing instances | 0 |
| Missing cores | 0 |
| Touching annotation pairs | 1,489 |
| Overlapping annotation pairs | 1,052 |
| Strict audit | PASS |

The overlap counts document polygons that share pixels; they are not silently
discarded. Pixel ownership is deterministic, and each source annotation retains
an instance and a core.

## Model A Versus Model B Targets

| Split | Model A foreground pixels | Model B foreground pixels | Pixels added by historical dilation |
|---|---:|---:|---:|
| Train | 669,535 | 438,134 | 231,401 |
| Validation | 130,615 | 92,291 | 38,324 |
| Total | 800,150 | 530,425 | 269,725 |

Model A and Model B use identical source images, instance labels, core labels,
split, seed, architecture, augmentation, optimizer, and epoch count. The
controlled target difference is:

- Model A: one-pixel positive-mask dilation;
- Model B: no positive dilation and a one-pixel boundary uncertainty band with
  loss weight 0.1.

The 269,725-pixel difference shows that training dilation was substantial enough
to test directly. It was training-only; it was never an inference dilation.

## Experimental Model C

Model C uses the Model B targets and adds a second output head for confident
instance cores. During evaluation, core components seed marker-controlled
watershed inside the predicted foreground. A foreground component lacking a
core receives one deterministic fallback marker so it is not discarded merely
because the experimental core head is uncertain.

Optional clDice and deep supervision are implemented but disabled in the Model C
configuration. They should be evaluated only after Models A, B, and C establish
whether boundary-aware supervision and core-based instance separation help.

## Validation Completed

- Full repository suite: 165 tests passed.
- Focused U-Net target/model suite: 14 tests passed.
- Real-data strict target preparation: passed for all 29 images.
- Real-sample dual-head optimization smoke: finite loss and successful update.
- Production Saturn files were not changed by this experiment.

## Remaining Work

GPU training and A/B/C evaluation must be run on Kaggle. Model selection must
consider instance recall, count error, merges, false splits, boundary behavior,
threshold stability, brightness, touching objects, and KJ/WT balance. The
highest Dice score alone is not sufficient. See
`V571_ANNOTATION_TOLERANT_AB_KAGGLE.md` for the exact notebook workflow.
