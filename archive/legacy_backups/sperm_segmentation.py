"""
Drosophila Spermatid Segmentation Pipeline  v2
===============================================
Detects elongated spermatid threads (SINGLE CLASS) from confocal Z-slices.

Key improvements vs v1
-----------------------
FALSE POSITIVES FIXED - oval/round blobs no longer pass:
  Old filter: eccentricity >= 0.80, minor <= 20 px  (let ratio~1.7 blobs through)
  New filter: eccentricity >= 0.90, minor <= 6 px, AND axis_ratio >= 3.0

FALSE NEGATIVES FIXED - dim/broken threads now recovered:
  1. CLAHE before ridge filter boosts local contrast of dim threads
  2. Hysteresis threshold: seeds at 97th pct, grows at 87th pct
     -> recovers full length of threads with uneven brightness along their length
  3. 3-pass bridge: strict filter -> dilate thin mask -> re-filter -> erode back
     -> merges nearby fragments of the same thread before measuring

Calibration (from Project001_Series002.xml):
  XY: 0.378788 um/px  |  Z: 0.346184 um/slice

Usage:
  python sperm_segmentation.py
  python sperm_segmentation.py --input_dir /path --output_dir /path/out
  python sperm_segmentation.py --threshold_hi 96 --threshold_lo 85
"""

import argparse, os, glob, re, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage import measure, morphology, exposure
from skimage.filters import meijering, gaussian, apply_hysteresis_threshold
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


# ─── PARAMETERS ───────────────────────────────────────────────────────────────
DEFAULTS = dict(
    input_dir        = r"C:\Users\dmishra\Desktop\sperm images",
    output_dir       = r"C:\Users\dmishra\Desktop\output",
    file_pattern     = "Project001_Series002_z*_ch00.tif",
    um_per_px_xy     = 0.378788,
    um_per_slice_z   = 0.346184,

    # Enhancement
    clahe_clip       = 0.10,   # Boosted for faint threads (was 0.04)
    clahe_kernel     = 64,     # CLAHE tile size (px)
    bg_sigma         = 20,     # Gaussian sigma for tissue background subtraction

    # Ridge filter
    ridge_sigmas     = [1, 2], # Meijering scales matching spermatid width (~1-2 px)

    # Hysteresis threshold (percentiles of ridge image)
    threshold_hi     = 95,     # Lowered to capture faint seeds (was 97)
    threshold_lo     = 75,     # Lowered to grow into faint tails (was 87)

    # Morphological cleanup
    close_radius     = 1,
    min_hole_area    = 50,
    min_obj_px       = 20,

    # ── Shape filter — ALL THREE must pass ───────────────────────────────────
    min_eccentricity = 0.90,   # needle-like; 0.90 allows slightly curved threads
    max_minor_px     = 6.0,    # spermatid threads ~1.5-4 px wide at this resolution
    min_axis_ratio   = 2.5,    # Relaxed slightly for fragmented segments (was 3.0)
    min_major_px     = 12,     # minimum length ~ 4.5 um

    # Bridge: dilation radius to merge nearby fragments of the same thread
    bridge_radius    = 3,      # px; increase if threads still appear fragmented

    # Skeleton
    min_skel_len_px  = 8,
    
    # 3D Tracking
    max_3d_dist_um   = 10.0,   # Max distance to link centroids across Z-slices
)


def load_slices(input_dir, pattern):
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in '{input_dir}'")
    def z_key(path):
        m = re.search(r'_z(\d+)_', os.path.basename(path))
        return int(m.group(1)) if m else 0
    files     = sorted(files, key=z_key)
    z_indices = [z_key(f) for f in files]
    print(f"Found {len(files)} slices  Z = {z_indices}")
    return files, z_indices



def segment_slice(img_raw, p, roi_mask=None):
    """
    Full 2D pipeline for one slice.
    """
    img      = img_raw.astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-9)

    # Apply ROI mask if provided (set outside to 0)
    if roi_mask is not None:
        img_norm[~roi_mask] = 0


    # 1. CLAHE
    img_eq = exposure.equalize_adapthist(
        img_norm, clip_limit=p['clahe_clip'], kernel_size=p['clahe_kernel'])

    # 2. Background subtraction
    bg  = gaussian(img_eq, sigma=p['bg_sigma'])
    fg  = np.clip(img_eq - bg, 0, None)
    fgn = fg / (fg.max() + 1e-9)

    # 3. Ridge filter
    ridge = meijering(fgn, sigmas=p['ridge_sigmas'], black_ridges=False)

    # 4. Hysteresis threshold
    th_hi = np.percentile(ridge, p['threshold_hi'])
    th_lo = np.percentile(ridge, p['threshold_lo'])
    mask  = apply_hysteresis_threshold(ridge, th_lo, th_hi)

    # 5. Cleanup
    mask = morphology.binary_closing(mask, morphology.disk(p['close_radius']))
    mask = morphology.remove_small_holes(mask, area_threshold=p['min_hole_area'])
    mask = morphology.remove_small_objects(mask, max_size=p['min_obj_px'])

    # 6. PASS 1: strict shape - only needle-like objects pass
    labeled1 = measure.label(mask)
    keep1 = []
    for prop in measure.regionprops(labeled1):
        ratio = prop.major_axis_length / (prop.minor_axis_length + 1e-9)
        if prop.eccentricity      < p['min_eccentricity']: continue
        if prop.minor_axis_length > p['max_minor_px']:     continue
        if ratio                  < p['min_axis_ratio']:   continue
        if prop.major_axis_length < p['min_major_px']:     continue
        keep1.append(prop.label)
    thin_mask = np.isin(labeled1, keep1)

    # 7. Dilate to bridge nearby fragments of same thread
    br      = p['bridge_radius']
    bridged = morphology.binary_dilation(thin_mask, morphology.disk(br))
    bridged = morphology.remove_small_objects(bridged, max_size=p['min_obj_px'])

    # 8. PASS 2: re-filter on bridged mask (relax slightly for dilated width)
    labeled2 = measure.label(bridged)
    keep2 = []
    for prop in measure.regionprops(labeled2):
        ratio = prop.major_axis_length / (prop.minor_axis_length + 1e-9)
        if ratio                  < (p['min_axis_ratio'] - 0.5): continue
        if prop.major_axis_length < p['min_major_px']:           continue
        keep2.append(prop.label)
    bridged_filtered = np.isin(labeled2, keep2)

    # 9. Erode back to restore true width
    final_mask = morphology.binary_erosion(bridged_filtered, morphology.disk(br))
    final_mask = morphology.remove_small_objects(final_mask, max_size=p['min_obj_px'])

    # 10. Skeletonize + measure
    skel         = skeletonize(final_mask)
    dist         = distance_transform_edt(final_mask)
    skel_labeled = measure.label(skel)

    raw_results = []
    for sp in measure.regionprops(skel_labeled):
        length_px = sp.coords.shape[0]
        if length_px < p['min_skel_len_px']:
            continue
        width_px = float(np.median(2.0 * dist[sp.coords[:, 0], sp.coords[:, 1]]))
        cy, cx   = sp.centroid
        raw_results.append(dict(label=sp.label, length_px=length_px,
                                width_px=width_px, centroid_x=cx, centroid_y=cy))

    # Re-index 1..N
    keep_skel     = {r['label'] for r in raw_results}
    clean_skel    = np.isin(skel_labeled, list(keep_skel))
    clean_labeled = measure.label(clean_skel)
    final_results = []
    for i, sp in enumerate(measure.regionprops(clean_labeled), start=1):
        length_px = sp.coords.shape[0]
        width_px  = float(np.median(2.0 * dist[sp.coords[:, 0], sp.coords[:, 1]]))
        cy, cx    = sp.centroid
        final_results.append(dict(label=i, length_px=length_px,
                                  width_px=width_px, centroid_x=cx, centroid_y=cy))

    return dict(mask=final_mask, skel=clean_skel,
                skel_label=clean_labeled, results=final_results)


def make_overlay(img_raw, seg, um):
    img  = img_raw.astype(np.float32)
    base = np.clip((img - np.percentile(img, 1)) /
                   (np.percentile(img, 99.5) - np.percentile(img, 1) + 1e-9), 0, 1)
    rgb  = np.stack([base] * 3, axis=-1)
    n    = max(len(seg['results']), 1)
    cols = plt.cm.gist_rainbow(np.linspace(0, 1, n))[:, :3]
    for idx, r in enumerate(seg['results']):
        tmp = morphology.binary_dilation(seg['skel_label'] == r['label'],
                                         morphology.disk(1))
        c = cols[idx % n]
        rgb[tmp, 0] = c[0];  rgb[tmp, 1] = c[1];  rgb[tmp, 2] = c[2]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def run_pipeline(params):
    os.makedirs(params['output_dir'], exist_ok=True)
    overlay_dir = os.path.join(params['output_dir'], 'overlays')
    os.makedirs(overlay_dir, exist_ok=True)

    files, z_indices = load_slices(params['input_dir'], params['file_pattern'])
    um = params['um_per_px_xy']
    all_rows, summaries = [], []

    for fpath, z_idx in zip(files, z_indices):
        print(f"\n--- Z={z_idx:02d}  {os.path.basename(fpath)} ---")
        img_raw = tifffile.imread(fpath)
        seg     = segment_slice(img_raw, params)
        results = seg['results']
        n       = len(results)

        lengths_um = [r['length_px'] * um for r in results]
        widths_um  = [r['width_px']  * um for r in results]
        print(f"  Spermatids: {n}")
        if lengths_um:
            print(f"  Length um: min={min(lengths_um):.2f}  "
                  f"median={np.median(lengths_um):.2f}  max={max(lengths_um):.2f}")

        for i, r in enumerate(results):
            all_rows.append(dict(
                z_slice=z_idx, sperm_id=i+1,
                length_px=round(r['length_px'], 1),
                length_um=round(r['length_px'] * um, 3),
                width_px=round(r['width_px'], 2),
                width_um=round(r['width_px'] * um, 3),
                centroid_x=round(r['centroid_x'], 1),
                centroid_y=round(r['centroid_y'], 1),
            ))

        summaries.append(dict(
            z_slice=z_idx, n_spermatids=n,
            mean_length_um  =round(np.mean(lengths_um),   3) if lengths_um else 0,
            median_length_um=round(np.median(lengths_um), 3) if lengths_um else 0,
            mean_width_um   =round(np.mean(widths_um),    3) if widths_um  else 0,
        ))

        overlay_rgb = make_overlay(img_raw, seg, um)
        plt.imsave(os.path.join(overlay_dir, f'z{z_idx:02d}_overlay.png'), overlay_rgb)

        # 3-panel detail figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        img_disp = np.clip(
            (img_raw.astype(float) - np.percentile(img_raw, 1)) /
            (np.percentile(img_raw, 99.5) - np.percentile(img_raw, 1) + 1e-9), 0, 1)
        axes[0].imshow(img_disp, cmap='gray')
        axes[0].set_title(f'Z={z_idx:02d} - Original');  axes[0].axis('off')
        axes[1].imshow(overlay_rgb)
        axes[1].set_title(f'Z={z_idx:02d} - Spermatids (N={n})');  axes[1].axis('off')
        for r in results:
            axes[1].text(r['centroid_x'], r['centroid_y'],
                         f"{r['length_px']*um:.1f}", color='white', fontsize=4,
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.4, lw=0))
        if lengths_um:
            axes[2].hist(lengths_um, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
            axes[2].axvline(np.median(lengths_um), color='orange', lw=2,
                            label=f'Median={np.median(lengths_um):.1f} um')
            axes[2].set_xlabel('Length (um)');  axes[2].set_ylabel('Count')
            axes[2].set_title(f'Z={z_idx:02d} - Length distribution')
            axes[2].legend(fontsize=9)
        else:
            axes[2].text(0.5, 0.5, 'No spermatids detected',
                         transform=axes[2].transAxes, ha='center', va='center')
            axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(overlay_dir, f'z{z_idx:02d}_detail.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()

        tifffile.imwrite(os.path.join(params['output_dir'], f'z{z_idx:02d}_mask.tif'),
                         seg['mask'].astype(np.uint8) * 255)
        tifffile.imwrite(os.path.join(params['output_dir'], f'z{z_idx:02d}_labels.tif'),
                         seg['skel_label'].astype(np.uint16))

    df = pd.DataFrame(all_rows)
    
    # --- 3D Tracking ---
    print(f"\n--- Running 3D Tracking (max_dist={params['max_3d_dist_um']} um) ---")
    df = track_3d_objects(df, params['max_3d_dist_um'], params['um_per_px_xy'])
    
    # Save per-detection CSV (now with 3D ID)
    df.to_csv(os.path.join(params['output_dir'], 'spermatid_measurements.csv'), index=False)

    # 3D Summary CSV
    # Group by 3D ID to get max length, start/end Z, etc.
    if 'sperm_3d_id' in df.columns:
        summary_3d = []
        for pid, grp in df.groupby('sperm_3d_id'):
            summary_3d.append(dict(
                sperm_3d_id = pid,
                n_slices    = len(grp),
                start_z     = grp['z_slice'].min(),
                end_z       = grp['z_slice'].max(),
                avg_length_um = round(grp['length_um'].mean(), 3),
                max_length_um = round(grp['length_um'].max(), 3),
                avg_width_um  = round(grp['width_um'].mean(), 3),
                centroid_x    = round(grp['centroid_x'].mean(), 1),
                centroid_y    = round(grp['centroid_y'].mean(), 1),
            ))
        df_3d = pd.DataFrame(summary_3d)
        df_3d.to_csv(os.path.join(params['output_dir'], 'spermatid_3d_summary.csv'), index=False)
        print(f"[OK] Identified {len(df_3d)} unique 3D objects")
        print(df_3d.describe().loc[['count', 'mean', 'min', 'max']])

    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(os.path.join(params['output_dir'], 'slice_summary.csv'), index=False)
    print(f"\n[OK] {len(df)} 2D detections saved")
    print(df_sum.to_string(index=False))

    if len(summaries) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        zs   = [s['z_slice'] for s in summaries]
        axes[0].bar(zs, [s['n_spermatids'] for s in summaries], color='steelblue')
        axes[0].set_xlabel('Z slice');  axes[0].set_ylabel('N spermatids (2D)')
        axes[0].set_title('Raw 2D Detections per Slice')
        axes[1].plot(zs, [s['median_length_um'] for s in summaries], 'o-', color='darkorange')
        axes[1].set_xlabel('Z slice');  axes[1].set_ylabel('Median length (um)')
        axes[1].set_title('Median spermatid length per slice')
        plt.tight_layout()
        plt.savefig(os.path.join(params['output_dir'], 'summary_across_slices.png'),
                    dpi=120, bbox_inches='tight')
        plt.close()

    print(f"\nDONE - {params['output_dir']}")
    return df, df_sum


def track_3d_objects(df, max_dist_um, um_per_px):
    """
    Links 2D detections across adjacent Z-slices into unique 3D objects.
    Simple greedy matching: object in Z is linked to closest object in Z-1.
    """
    if df.empty:
        return df
    
    # Sort by Z
    df = df.sort_values(['z_slice', 'sperm_id']).reset_index(drop=True)
    df['sperm_3d_id'] = -1
    
    # Initialize ID counter
    next_id = 1
    
    # Get list of Z slices
    z_slices = sorted(df['z_slice'].unique())
    
    # Keep track of active objects in previous slice
    # keys: (z_slice, sperm_id) -> 3D_ID
    # We also need their coordinates for distance calc
    prev_objects = {} # {id_in_slice: {'3d_id': X, 'pos': (x,y)}}
    
    for z in z_slices:
        current_rows = df[df['z_slice'] == z]
        curr_objects = {}
        
        for idx, row in current_rows.iterrows():
            best_match = None
            min_dist   = float('inf')
            
            cx, cy = row['centroid_x'], row['centroid_y']
            
            # Check against objects in immediately previous slice
            # We look at z-1. If we wanted to handle gaps, we'd check z-1, z-2 etc.
            # For now, strict adjacency.
            if prev_objects:
                for old_local_id, data in prev_objects.items():
                    ox, oy = data['pos']
                    dist = np.sqrt((cx-ox)**2 + (cy-oy)**2) * um_per_px
                    if dist < max_dist_um and dist < min_dist:
                        min_dist = dist
                        best_match = data['3d_id']
            
            # Assign ID
            if best_match is not None:
                new_id = best_match
            else:
                new_id = next_id
                next_id += 1
                
            df.at[idx, 'sperm_3d_id'] = new_id
            curr_objects[row['sperm_id']] = {'3d_id': new_id, 'pos': (cx, cy)}
            
        # Update prev_objects for next iteration
        prev_objects = curr_objects
        
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if not isinstance(v, list):
            parser.add_argument(f'--{k}', type=type(v), default=v)
    args   = parser.parse_args()
    params = {**DEFAULTS, **{k: v for k, v in vars(args).items()
                              if k in DEFAULTS and not isinstance(DEFAULTS[k], list)}}
    run_pipeline(params)
