"""Generate the reproducible Kaggle A/B/C experiment notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks" / "v571_annotation_tolerant_unet_kaggle.ipynb"


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
        """# Saturn v5.7.1 annotation-tolerant U-Net comparison

This notebook runs a controlled Model A versus Model B comparison, then an
experimental dual-head Model C. It reuses all 5,273 existing annotations and
does not modify Saturn production inference, tracking, calibration, or
biological thresholds. The four held-out images represent four specimens;
individual nuclei are not independent biological replicates.
"""
    ),
    code(
        """# Cell 1 - Locate Kaggle's already-extracted training package and code.
from pathlib import Path
import hashlib, shutil

INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working')
RUN_MODEL_C = True

package_markers = list(INPUT.rglob('v5_7_kj_wt_replay_finetune/annotations/_annotations.coco.json'))
repo_markers = list(INPUT.rglob('repo/unet25d/prepare_dataset.py'))
assert package_markers, 'Extracted training package was not found under /kaggle/input'
assert repo_markers, 'Extracted repo/unet25d code was not found under /kaggle/input'

PACKAGE = package_markers[0].parents[1]
INPUT_REPO = repo_markers[0].parents[1]
REPO = WORK / 'repo'
if REPO.exists(): shutil.rmtree(REPO)
shutil.copytree(INPUT_REPO, REPO)
bundled_checkpoint = REPO/'warm_start'/'epoch_003.pt'
WARM_START = bundled_checkpoint
assert WARM_START.exists(), 'The known epoch_003.pt warm start is missing'
EXPECTED_WARM_START_SHA256 = 'afe88f52e1c679d133a4755f4b4c51d17f8b2bef8a9c565e687cc74be0fbaeaf'
actual_checkpoint_hash = hashlib.sha256(WARM_START.read_bytes()).hexdigest()
assert actual_checkpoint_hash == EXPECTED_WARM_START_SHA256, (actual_checkpoint_hash, EXPECTED_WARM_START_SHA256)
print('Package:', PACKAGE)
print('Warm start:', WARM_START)
print('Extracted input code:', INPUT_REPO)
print('Writable code copy:', REPO)
"""
    ),
    code(
        """# Cell 2 - Install only the experiment requirements.
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(REPO/'unet25d'/'requirements.txt')], check=True)
"""
    ),
    code(
        """# Cell 3 - Record the software and GPU environment.
import platform, numpy as np, scipy, skimage, PIL, torch
environment = {
    'python': platform.python_version(), 'torch': torch.__version__,
    'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'numpy': np.__version__, 'scipy': scipy.__version__,
    'scikit_image': skimage.__version__, 'pillow': PIL.__version__,
}
print(environment)
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator before training'
"""
    ),
    code(
        """# Cell 4 - Localize the three immutable experiment configurations.
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
print(CONFIG_A, CONFIG_B, CONFIG_C, sep='\\n')
"""
    ),
    code(
        """# Cell 5 - Run source hashes and train/validation leakage preflight.
PREFLIGHT = WORK/'outputs'/'preflight'
configs = [CONFIG_A, CONFIG_B] + ([CONFIG_C] if RUN_MODEL_C else [])
command = [sys.executable, str(REPO/'unet25d'/'preflight_annotation_tolerant_experiment.py'),
           '--package', str(PACKAGE), '--checkpoint', str(WARM_START),
           '--repo', str(REPO), '--output', str(PREFLIGHT)]
for cfg in configs: command += ['--config', str(cfg)]
subprocess.run(command, check=True)
"""
    ),
    code(
        """# Cell 6 - Run the focused unit suite before spending GPU time.
subprocess.run([sys.executable, '-m', 'pytest', '-q',
                str(REPO/'tests'/'test_unet25d_prepare_dataset.py')], check=True)
"""
    ),
    code(
        """# Cell 7 - Generate separate A/B/C targets and require strict audits.
import json
PREPARER = REPO/'unet25d'/'prepare_dataset.py'
for cfg in configs:
    subprocess.run([sys.executable, '-u', str(PREPARER), '--config', str(cfg)], check=True)

output_dirs = [WORK/'outputs'/'model_a_replay_control', WORK/'outputs'/'model_b_boundary_tolerant']
if RUN_MODEL_C: output_dirs.append(WORK/'outputs'/'model_c_dual_head')
for folder in output_dirs:
    audit_path = folder/'dataset'/'target_generation_audit_summary.json'
    audit = json.loads(audit_path.read_text())
    print(folder.name, audit)
    assert audit['audit_pass']
    assert audit['source_annotation_count'] == 5273
    assert audit['generated_instance_count'] == 5273
    assert audit['core_instance_count'] == 5273
    assert audit['outside_roi_pixel_count'] == 0
    assert audit['incorrect_dimension_count'] == 0
"""
    ),
    code(
        """# Cell 8 - Re-run preflight to hash generated manifests and target arrays.
subprocess.run(command, check=True)
"""
    ),
    code(
        """# Cell 9 - Display real Model A and B targets before training.
import matplotlib.pyplot as plt
sample_a = sorted((output_dirs[0]/'dataset'/'train').glob('*.npz'))[0]
sample_b = output_dirs[1]/'dataset'/'train'/sample_a.name
with np.load(sample_a) as a, np.load(sample_b) as b:
    panels = [a['image'][0], a['image'][1], a['image'][2], b['raw_annotation_mask'],
              a['foreground_target'], b['foreground_target'], b['boundary_ignore_mask'],
              b['loss_weight_mask'], b['instance_core_labels']>0, b['overlap_count_map']>1]
titles = ['z-1','z','z+1','Original annotation','A: dilated foreground','B: raw foreground',
          'B: uncertain boundary','B: loss weights','Instance cores','Overlapping pixels']
fig, axes = plt.subplots(2,5,figsize=(20,8))
for ax, panel, title in zip(axes.ravel(), panels, titles):
    ax.imshow(panel, cmap='gray'); ax.set_title(title); ax.axis('off')
plt.tight_layout(); plt.savefig(WORK/'outputs'/'target_example.png', dpi=180); plt.show()
"""
    ),
    code(
        """# Cell 10 - Strict A/B checkpoint load and finite forward/backward smoke tests.
import sys
sys.path.insert(0, str(REPO/'unet25d'))
from train_unet25d import SpermPatchDataset, build_model, run_epoch, masked_bce_loss
from torch.utils.data import DataLoader

payload = torch.load(WARM_START, map_location='cpu')
for cfg_path, folder, partial in [(CONFIG_A, output_dirs[0], False), (CONFIG_B, output_dirs[1], False)] + ([(CONFIG_C, output_dirs[2], True)] if RUN_MODEL_C else []):
    cfg = yaml.safe_load(cfg_path.read_text())
    model = build_model(cfg)
    state = payload.get('model', payload)
    if partial:
        current = model.state_dict(); compatible = {k:v for k,v in state.items() if k in current and current[k].shape==v.shape}
        current.update(compatible); model.load_state_dict(current, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    dual = cfg['architecture'].startswith('dual_head')
    ds = SpermPatchDataset(folder/'dataset'/'train', 64, 1, False, cfg['seed'], return_core_target=dual)
    ds.paths = ds.paths[:1]; ds.positive_loss_weight = cfg['positive_loss_weight']
    loader = DataLoader(ds, batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    loss, dice = run_epoch(model, loader, optimizer, torch.device('cpu'), True, cfg)
    assert np.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(folder.name, {'loss':loss, 'foreground_dice':dice, 'finite_gradients':True})

# A zero-weight uncertain pixel must not alter BCE.
target=torch.tensor([[[[1.,0.]]]]); weight=torch.tensor([[[[1.,0.]]]])
assert torch.allclose(masked_bce_loss(torch.tensor([[[[0.,-100.]]]]),target,weight),
                      masked_bce_loss(torch.tensor([[[[0.,100.]]]]),target,weight))
"""
    ),
    code(
        """# Cell 11 - Train Model A control. Output is streamed and snapshots are retained.
TRAINER = REPO/'unet25d'/'train_unet25d.py'
subprocess.run([sys.executable, '-u', str(TRAINER), '--config', str(CONFIG_A),
                '--warm-start', str(WARM_START)], check=True)
"""
    ),
    code(
        """# Cell 12 - Train Model B boundary-tolerant model from the identical checkpoint.
subprocess.run([sys.executable, '-u', str(TRAINER), '--config', str(CONFIG_B),
                '--warm-start', str(WARM_START)], check=True)
"""
    ),
    code(
        """# Cell 13 - Train optional Model C dual-head model.
if RUN_MODEL_C:
    subprocess.run([sys.executable, '-u', str(TRAINER), '--config', str(CONFIG_C),
                    '--warm-start', str(WARM_START), '--allow-partial-warm-start'], check=True)
"""
    ),
    code(
        """# Cell 14 - Evaluate baseline and every planned snapshot across thresholds.
EVAL_OUT = WORK/'outputs'/'model_comparison'
evaluator = REPO/'unet25d'/'evaluate_annotation_tolerant_ab.py'
args = [sys.executable, '-u', str(evaluator), '--config-a', str(CONFIG_A), '--config-b', str(CONFIG_B),
        '--reference-dataset', str(output_dirs[1]/'dataset'/'valid'),
        '--group-key', str(PACKAGE/'annotation_key_private.csv'), '--output', str(EVAL_OUT)]
for label, path in [('baseline',WARM_START), ('best',output_dirs[0]/'checkpoints'/'best.pt')] + [(f'epoch_{n:03d}',output_dirs[0]/'checkpoints'/f'epoch_{n:03d}.pt') for n in (3,6,9,12)]:
    args += ['--checkpoint-a', f'{label}={path}']
for label, path in [('baseline',WARM_START), ('best',output_dirs[1]/'checkpoints'/'best.pt')] + [(f'epoch_{n:03d}',output_dirs[1]/'checkpoints'/f'epoch_{n:03d}.pt') for n in (3,6,9,12)]:
    args += ['--checkpoint-b', f'{label}={path}']
if RUN_MODEL_C:
    args += ['--config-c', str(CONFIG_C)]
    for label, path in [('best',output_dirs[2]/'checkpoints'/'best.pt')] + [(f'epoch_{n:03d}',output_dirs[2]/'checkpoints'/f'epoch_{n:03d}.pt') for n in (3,6,9,12)]:
        args += ['--checkpoint-c', f'{label}={path}']
subprocess.run(args, check=True)
"""
    ),
    code(
        """# Cell 15 - Enforce diagnostic gates before reviewing model candidates.
import json, pandas as pd
metadata = json.loads((EVAL_OUT/'evaluation_metadata.json').read_text())
required_diagnostics = [
    EVAL_OUT/'core_watershed_diagnostic_manifest.csv',
    EVAL_OUT/'merge_split_audit.csv',
    EVAL_OUT/'partial_label_audit.csv',
]
assert all(path.exists() for path in required_diagnostics), required_diagnostics
assert metadata['diagnostic_gate_pass'], metadata

diagnostics = pd.read_csv(required_diagnostics[0])
merge_split = pd.read_csv(required_diagnostics[1])
partial_labels = pd.read_csv(required_diagnostics[2])
assert len(diagnostics) == metadata['core_watershed_diagnostic_count']
assert len(merge_split) == metadata['merge_split_audit_row_count']
assert len(partial_labels) == metadata['partial_label_audit_row_count']

print('Core-marker/watershed diagnostics:', len(diagnostics))
display(merge_split.groupby(['model','threshold'])[
    ['merge_prediction_rate','split_reference_rate','touching_instance_recall']
].mean().reset_index())
display(partial_labels.groupby(['model','threshold'])[
    ['ignored_roi_fraction','unmatched_prediction_count_predominantly_ignored',
     'unmatched_prediction_count_supervised']
].mean().reset_index())

# Candidate ordering is shown only after all diagnostic gates pass.
selection = pd.read_csv(EVAL_OUT/'model_selection_table.csv')
display(selection.sort_values(
    ['instance_recall','merge_prediction_rate','split_reference_rate'],
    ascending=[False,True,True],
).head(30))
print('No automatic winner: inspect marker and watershed panels before selection.')
print('Validation limitation: four images from four specimens.')
"""
    ),
    code(
        """# Cell 16 - Display the required core-marker and watershed-instance panels.
from IPython.display import Image as NotebookImage, display
review = diagnostics[
    diagnostics['model'].isin(['model_c_dual_head:best','model_c_dual_head:epoch_003'])
    & diagnostics['threshold'].isin([0.30, 0.60])
].sort_values(['model','threshold','image'])
assert not review.empty, 'Required Model C diagnostic panels were not generated'
for path in review['panel_relative_path']:
    display(NotebookImage(filename=str(EVAL_OUT/path)))
"""
    ),
    code(
        """# Cell 17 - Hash checkpoints and archive all reproducibility artifacts.
import hashlib, csv, zipfile
hash_rows=[]
for path in sorted((WORK/'outputs').rglob('*.pt')):
    digest=hashlib.sha256(path.read_bytes()).hexdigest(); hash_rows.append({'path':str(path.relative_to(WORK)),'sha256':digest})
with open(WORK/'outputs'/'checkpoint_hashes.csv','w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=['path','sha256']); writer.writeheader(); writer.writerows(hash_rows)

archive=WORK/'v571_annotation_tolerant_experiment.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as handle:
    for folder in (WORK/'outputs', REPO/'unet25d'):
        for path in folder.rglob('*'):
            if path.is_file(): handle.write(path,path.relative_to(WORK))
    for path in configs: handle.write(path,path.relative_to(WORK))
print('Archive:',archive,'bytes:',archive.stat().st_size)
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
