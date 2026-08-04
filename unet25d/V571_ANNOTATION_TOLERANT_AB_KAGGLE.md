# Saturn v5.7.1 Annotation-Tolerant U-Net Experiments

This workflow reuses the existing 5,273 COCO instances. It does not require
redrawing the hand annotations and does not change Saturn production inference.

The canonical runnable workflow is the generated notebook at
`notebooks/v571_annotation_tolerant_unet_kaggle.ipynb`. Upload
`Kaggle notebook inputs/v571_annotation_tolerant_code_bundle.zip` as a Kaggle
dataset; it contains the tested code and the verified `epoch_003.pt` warm start.
The cells below are a shorter command reference.

## Experiment Arms

- Model A: replay control, current residual-attention U-Net, one-pixel positive
  target dilation, current loss.
- Model B: same architecture and training settings, no positive dilation,
  fractional one-pixel boundary weight, preserved instance labels.
- Model C: experimental dual-head residual-attention U-Net using Model B targets.
  It predicts full foreground plus a confident core map. clDice and deep
  supervision are implemented but disabled for this comparison.

Models A and B are the required controlled comparison. Train Model C only after
their target audits pass. Do not select a model from validation Dice alone.

## Kaggle Cell 1: Locate Inputs

```python
from pathlib import Path

INPUT = Path("/kaggle/input")
packages = list(INPUT.rglob("v5_7_kj_wt_replay_finetune"))
checkpoints = list(INPUT.rglob("epoch_003.pt"))
if not checkpoints:
    checkpoints = list(INPUT.rglob("best.pt"))

assert packages, "Training package directory not found under /kaggle/input"
assert checkpoints, "Warm-start checkpoint not found under /kaggle/input"

PACKAGE = packages[0]
WARM_START = checkpoints[0]
print("Package:", PACKAGE)
print("Warm start:", WARM_START)
```

Use the previously selected `epoch_003.pt` when it is available. Every arm must
start from the same checkpoint file.

## Kaggle Cell 2: Get This Experiment Branch

Enable Internet for the notebook, then run:

```bash
cd /kaggle/working
git clone --branch feature/v5.7.1-annotation-tolerant-unet \
  https://github.com/dushyant-mishra/Drosophila-sperm-nuclei-segmentation.git repo
pip install -q -r /kaggle/working/repo/unet25d/requirements.txt
```

If Internet is unavailable, upload the repository as a Kaggle dataset and set
`REPO` below to that extracted directory.

## Kaggle Cell 3: Materialize Identical Configs

```python
from pathlib import Path
import yaml

REPO = Path("/kaggle/working/repo")
config_dir = REPO / "unet25d" / "configs"

def localize(source_name):
    cfg = yaml.safe_load((config_dir / source_name).read_text())
    cfg["project_root"] = str(PACKAGE)
    cfg["stack_image_dir"] = str(PACKAGE / "raw_tiffs")
    cfg["annotation_manifest"] = str(PACKAGE / "annotations" / "_annotations.coco.json")
    cfg["roi_mask_dir"] = str(PACKAGE / "roi_masks")
    destination = Path("/kaggle/working") / source_name
    destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return destination

CONFIG_A = localize("v571_model_a_replay_control_kaggle.yaml")
CONFIG_B = localize("v571_model_b_annotation_tolerant_kaggle.yaml")
CONFIG_C = localize("v571_model_c_dual_head_kaggle.yaml")
print(CONFIG_A, CONFIG_B, CONFIG_C, sep="\n")
```

## Kaggle Cell 4: Prepare and Audit Targets

```python
import json
import subprocess
import sys

for config in (CONFIG_A, CONFIG_B, CONFIG_C):
    subprocess.run(
        [sys.executable, str(REPO / "unet25d" / "prepare_dataset.py"),
         "--config", str(config)],
        check=True,
    )

for output_name in (
    "outputs/model_a_replay_control",
    "outputs/model_b_boundary_tolerant",
    "outputs/model_c_dual_head",
):
    path = Path("/kaggle/working") / output_name / "dataset" / "target_generation_audit_summary.json"
    audit = json.loads(path.read_text())
    print(output_name, {
        "annotations": audit["source_annotation_count"],
        "instances": audit["generated_instance_count"],
        "cores": audit["core_instance_count"],
        "failures": audit["audit_failure_count"],
        "pass": audit["audit_pass"],
    })
    assert audit["audit_pass"]
    assert audit["source_annotation_count"] == 5273
    assert audit["generated_instance_count"] == 5273
    assert audit["core_instance_count"] == 5273
```

Stop here if any audit fails.

## Kaggle Cell 5: Train Models A and B

```python
trainer = REPO / "unet25d" / "train_unet25d.py"
for config in (CONFIG_A, CONFIG_B):
    subprocess.run(
        [sys.executable, str(trainer), "--config", str(config),
         "--warm-start", str(WARM_START)],
        check=True,
    )
```

## Kaggle Cell 6: Train Experimental Model C

Model C adds a core head, so it must use a partial architecture warm start.

```python
subprocess.run(
    [sys.executable, str(trainer), "--config", str(CONFIG_C),
     "--warm-start", str(WARM_START), "--allow-partial-warm-start"],
    check=True,
)
```

## Kaggle Cell 7: Evaluate A, B, and C

```python
evaluator = REPO / "unet25d" / "evaluate_annotation_tolerant_ab.py"
subprocess.run(
    [
        sys.executable, str(evaluator),
        "--config-a", str(CONFIG_A),
        "--checkpoint-a", "/kaggle/working/outputs/model_a_replay_control/checkpoints/best.pt",
        "--config-b", str(CONFIG_B),
        "--checkpoint-b", "/kaggle/working/outputs/model_b_boundary_tolerant/checkpoints/best.pt",
        "--config-c", str(CONFIG_C),
        "--checkpoint-c", "/kaggle/working/outputs/model_c_dual_head/checkpoints/best.pt",
        "--reference-dataset", "/kaggle/working/outputs/model_b_boundary_tolerant/dataset/valid",
        "--group-key", str(PACKAGE / "annotation_key_private.csv"),
        "--output", "/kaggle/working/v571_annotation_tolerant_comparison",
    ],
    check=True,
)
```

The evaluator writes per-pixel, per-image, per-object, and aggregate CSV files,
plus continuous probability maps. Models A and B use connected components for
this fixed diagnostic. Model C uses its core map as markers for watershed of the
foreground map.

Before ordering model candidates, require all of these outputs:

- `core_watershed_diagnostic_manifest.csv`;
- `core_watershed_diagnostics/**/*.png`, showing the raw plane, foreground
  probability, core probability, thresholded foreground, thresholded core,
  core markers, watershed labels, and reference/prediction boundaries;
- `core_watershed_diagnostics/**/*.npz`, containing the corresponding marker
  and instance-label arrays;
- `merge_split_audit.csv`, with normalized merge and split rates;
- `partial_label_audit.csv`, separating evaluable false positives from
  predictions that lie predominantly in intentionally unsupervised regions.

The evaluator records both unadjusted and partial-label-adjusted precision and
count error. Predictions in ignored regions are unknown, not automatically
false. Review faint/intermediate/bright, touching, genotype/group, and
partial-label strata separately. Only after the diagnostic gate passes should
`model_selection_table.csv` be used to order candidates for visual review.

## Kaggle Cell 8: Archive Results

```python
from pathlib import Path
import zipfile

archive = Path("/kaggle/working/v571_annotation_tolerant_experiment.zip")
folders = [
    Path("/kaggle/working/outputs/model_a_replay_control"),
    Path("/kaggle/working/outputs/model_b_boundary_tolerant"),
    Path("/kaggle/working/outputs/model_c_dual_head"),
    Path("/kaggle/working/v571_annotation_tolerant_comparison"),
]
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
    for folder in folders:
        for path in folder.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to("/kaggle/working"))
print(archive, archive.stat().st_size)
```

## Acceptance Gates

Model B or C is a candidate for further review only when:

- nucleus recall remains stable in both KJ and WT;
- faint-object recall remains stable;
- touching-object recall improves or remains stable;
- merge rate improves or remains stable;
- false splitting does not materially increase;
- count error remains stable across thresholds;
- boundary F1 or contour distance improves;
- predicted area and width are less threshold-sensitive;
- unusual KJ objects are not preferentially lost.

The numerically best checkpoint is the first candidate for visual inspection,
not an automatic production replacement. Production Saturn thresholds,
tracking, reporting, and biological results remain frozen until this comparison
passes visual and morphometry review.

## Blinded Epoch 3 Versus Epoch 12 Review

Build the event-level review from cached Model C outputs only:

```powershell
.\.venv\Scripts\python.exe .\unet25d\build_blinded_checkpoint_review.py `
  --cache-dir .\scratch\v571_blinded_cached_outputs\outputs\model_comparison `
  --reference-dataset .\scratch\v571_local_posthoc_inputs\outputs\model_b_boundary_tolerant\dataset\valid `
  --group-key .\training_packages\v5_7_kj_wt_replay_finetune\annotation_key_private.csv `
  --output .\scratch\v571_blinded_epoch003_vs_epoch012_review `
  --foreground-threshold 0.60 `
  --core-threshold 0.50 `
  --minimum-component-px 3 `
  --seed 5710312
```

The reviewer package contains Method A and Method B only. Its master CSV links
every event to a thumbnail, and its PDFs group split, merge, tiny-child,
supervised-background, ignored-region, and difficult-KJ examples. The method
key is written outside the reviewer package and remains sealed until all review
fields are complete. No numerical score selects a checkpoint before reveal.

After review, the required order is: freeze the selected checkpoint and current
watershed settings; integrate dual-head inference and apparent-width fields;
run representative-slice smoke tests; run one full WT and one full KJ pilot;
validate calibrated deterministic tracking; then rerun the biological study.

PSF correction remains behind a disabled experimental switch. Apparent
perpendicular-chord width is the production width even when PSF records are
unavailable. Lateral PSF is relevant only to lateral width; axial PSF belongs
to Z and 3D interpretation. Objects from 15 to 20 um trigger technical review
for possible merging or fusion but are not automatically rejected, and unusual
KJ morphology remains measurable.
