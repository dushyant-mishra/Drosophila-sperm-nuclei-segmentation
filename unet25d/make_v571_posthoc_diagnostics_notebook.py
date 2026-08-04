"""Generate a Kaggle notebook that audits saved models without retraining."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "v571_posthoc_watershed_diagnostics_kaggle.ipynb"


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    markdown(
        """# Saturn v5.7.1 post-training watershed diagnostics

This notebook reuses completed Model A/B/C checkpoints. It does not train any
model. It generates core-marker, watershed-instance, merge/split, partial-label,
and candidate-review outputs before any model ordering is considered.
"""
    ),
    code(
        """# Cell 1 - Locate the two already-extracted Kaggle datasets.
from pathlib import Path
import hashlib, shutil

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')

package_markers = list(INPUT.rglob('v5_7_kj_wt_replay_finetune/annotations/_annotations.coco.json'))
repo_markers = list(INPUT.rglob('repo/unet25d/evaluate_annotation_tolerant_ab.py'))
experiment_markers = list(INPUT.rglob('outputs/model_a_replay_control/checkpoints/best.pt'))
assert package_markers, 'Updated complete v5.7.1 package dataset is missing'
assert repo_markers, 'Updated diagnostic evaluator is missing'
assert experiment_markers, 'Saved v571 experiment output dataset is missing'

PACKAGE = package_markers[0].parents[1]
INPUT_REPO = repo_markers[0].parents[1]
SAVED_OUTPUTS = experiment_markers[0].parents[2]
REPO = WORK/'repo'
if REPO.exists(): shutil.rmtree(REPO)
shutil.copytree(INPUT_REPO, REPO)
EVAL_OUT = WORK/'v571_posthoc_watershed_diagnostics'
if EVAL_OUT.exists(): shutil.rmtree(EVAL_OUT)

print('Training package:', PACKAGE)
print('Saved model outputs:', SAVED_OUTPUTS)
print('Updated evaluator:', REPO/'unet25d'/'evaluate_annotation_tolerant_ab.py')
"""
    ),
    code(
        """# Cell 2 - Install the tested dependencies and verify the GPU.
import subprocess, sys, torch
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(REPO/'unet25d'/'requirements.txt')], check=True)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator'
"""
    ),
    code(
        """# Cell 3 - Localize configs and rerun evaluation only.
import yaml
CONFIG_ROOT = REPO/'unet25d'/'configs'
def localize(name):
    cfg = yaml.safe_load((CONFIG_ROOT/name).read_text())
    cfg['project_root'] = str(PACKAGE)
    cfg['stack_image_dir'] = str(PACKAGE/'raw_tiffs')
    cfg['annotation_manifest'] = str(PACKAGE/'annotations'/'_annotations.coco.json')
    cfg['roi_mask_dir'] = str(PACKAGE/'roi_masks')
    destination = WORK/name
    destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return destination

CONFIG_A = localize('v571_model_a_replay_control_kaggle.yaml')
CONFIG_B = localize('v571_model_b_annotation_tolerant_kaggle.yaml')
CONFIG_C = localize('v571_model_c_dual_head_kaggle.yaml')
WARM_START = REPO/'warm_start'/'epoch_003.pt'

def available(folder, include_baseline=False):
    specs = [('baseline', WARM_START)] if include_baseline else []
    for label, name in [('best','best.pt'), ('epoch_003','epoch_003.pt'),
                        ('epoch_006','epoch_006.pt'), ('epoch_009','epoch_009.pt'),
                        ('epoch_012','epoch_012.pt')]:
        path = folder/'checkpoints'/name
        if path.exists(): specs.append((label, path))
    return specs

evaluator = REPO/'unet25d'/'evaluate_annotation_tolerant_ab.py'
args = [sys.executable, '-u', str(evaluator),
        '--config-a', str(CONFIG_A), '--config-b', str(CONFIG_B),
        '--config-c', str(CONFIG_C),
        '--reference-dataset', str(SAVED_OUTPUTS/'model_b_boundary_tolerant'/'dataset'/'valid'),
        '--group-key', str(PACKAGE/'annotation_key_private.csv'),
        '--output', str(EVAL_OUT)]
for label, path in available(SAVED_OUTPUTS/'model_a_replay_control', True):
    args += ['--checkpoint-a', f'{label}={path}']
for label, path in available(SAVED_OUTPUTS/'model_b_boundary_tolerant', True):
    args += ['--checkpoint-b', f'{label}={path}']
for label, path in available(SAVED_OUTPUTS/'model_c_dual_head'):
    args += ['--checkpoint-c', f'{label}={path}']
subprocess.run(args, check=True)
"""
    ),
    code(
        """# Cell 4 - Enforce diagnostic completeness before reviewing candidates.
import json, pandas as pd
metadata = json.loads((EVAL_OUT/'evaluation_metadata.json').read_text())
required = [EVAL_OUT/'core_watershed_diagnostic_manifest.csv',
            EVAL_OUT/'merge_split_audit.csv', EVAL_OUT/'partial_label_audit.csv']
assert all(path.exists() for path in required), required
assert metadata['diagnostic_gate_pass'], metadata

diagnostics = pd.read_csv(required[0])
merge_split = pd.read_csv(required[1])
partial = pd.read_csv(required[2])
selection = pd.read_csv(EVAL_OUT/'model_selection_table.csv')
print('Diagnostic panels:', len(diagnostics))
display(merge_split.groupby(['model','threshold'])[
    ['merge_prediction_rate','split_reference_rate','touching_instance_recall']
].mean().reset_index())
display(partial.groupby(['model','threshold'])[
    ['ignored_roi_fraction','unmatched_prediction_count_predominantly_ignored',
     'unmatched_prediction_count_supervised']
].mean().reset_index())
display(selection.sort_values(
    ['instance_recall','merge_prediction_rate','split_reference_rate'],
    ascending=[False,True,True]).head(30))
"""
    ),
    code(
        """# Cell 5 - Display Model C best/epoch-003 watershed evidence.
from IPython.display import Image as NotebookImage, display
review = diagnostics[
    diagnostics['model'].isin(['model_c_dual_head:best','model_c_dual_head:epoch_003'])
    & diagnostics['threshold'].isin([0.30, 0.60])
].sort_values(['model','threshold','image'])
assert not review.empty
for relative_path in review['panel_relative_path']:
    display(NotebookImage(filename=str(EVAL_OUT/relative_path)))
"""
    ),
    code(
        """# Cell 6 - Archive the diagnostics for download.
import zipfile
archive = WORK/'v571_posthoc_watershed_diagnostics.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
    for path in EVAL_OUT.rglob('*'):
        if path.is_file(): handle.write(path, path.relative_to(WORK))
print('Archive:', archive)
print('Size MB:', archive.stat().st_size/1024**2)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(OUTPUT)
