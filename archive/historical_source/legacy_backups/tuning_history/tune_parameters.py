import os
import sys
import argparse
import numpy as np
import tifffile
import time
import json
import matplotlib
import glob

# Force TkAgg backend before importing pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

from scipy.optimize import differential_evolution

# Import the core segmentation and measuring functions from our main script
try:
    from sperm_segmentation_v9_combined import (
        CONFIG,
        segment_slice,
        measure_spermatids,
        normalize_display,
        make_overlay
    )
except ImportError:
    print("Error: Could not import 'sperm_segmentation_v9_combined.py'. Make sure it's in the same directory.")
    sys.exit(1)

ROI_SAVE_PATH = "last_drawn_roi.tif"

# Global tracking for the optimization progress
eval_count = 0
best_global_score = -999999
results_list = []
images_to_eval = []
roi_mask_global = None

def objective_fn(x):
    global eval_count, best_global_score, results_list, images_to_eval, roi_mask_global
    
    # Unpack parameters
    th_hi, th_lo, bg_sigma, clahe_clip, min_len, max_wid, min_rat, max_tort, max_branch, max_endpt = x
    
    # Ensure LO is strictly less than HI
    if th_lo >= th_hi:
        th_lo = th_hi - 1.0
        
    p_comb = {
        "THRESHOLD_HI": float(th_hi),
        "THRESHOLD_LO": float(th_lo),
        "BG_SIGMA": float(bg_sigma),
        "CLAHE_CLIP": float(clahe_clip),
        "MIN_SKEL_LEN_PX": float(min_len),
        "MAX_WIDTH_PX": float(max_wid),
        "MIN_LENGTH_WIDTH_RATIO": float(min_rat),
        "MAX_TORTUOSITY": float(max_tort),
        "MAX_BRANCH_NODES": int(round(max_branch)),
        "MAX_ENDPOINT_COUNT": int(round(max_endpt))
    }
    
    cfg = CONFIG.copy()
    cfg.update(p_comb)
    
    # Hide measure spam
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    total_score = 0.0
    total_count = 0
    all_medians = []
    
    try:
        for img in images_to_eval:
            seg = segment_slice(img, cfg, z_idx=0, debug_dir=None, roi_mask=roi_mask_global)
            meas = measure_spermatids(seg, cfg)
            
            raw_results = meas['results']
            
            # Since segment_slice inherently masks the ridges via roi_mask, 
            # we don't need to manually filter the centroids anymore.
            count = len(raw_results)
            total_count += count
            if count > 0:
                lengths = [r['length_px_geodesic'] for r in raw_results]
                median_length_px = float(np.median(lengths))
                median_um = median_length_px * 0.2
                all_medians.append(median_um)
            else:
                median_um = 0.0
                
            # Count-Scaled Penalty
            penalty = 0.0
            if count > 0:
                if median_um < 7.5:
                    penalty = (7.5 - median_um) * count * 3.0
                elif median_um > 13.0:
                    penalty = (median_um - 13.0) * count * 2.0
                    
            img_score = count - penalty
            total_score += img_score
            
        avg_median = float(np.mean(all_medians)) if all_medians else 0.0
        
    except Exception as e:
        total_count = 0
        avg_median = 0.0
        total_score = -999999
        
    sys.stdout = old_stdout
    
    eval_count += 1
    if total_score > best_global_score:
        best_global_score = total_score
        sys.stdout.write(f"\rEval {eval_count:5d} | New Best Score: {best_global_score:8.2f} (Total Count: {total_count}, Avg MedLen: {avg_median:.2f} um)")
        sys.stdout.flush()
        
    results_list.append({
        "params": p_comb,
        "count": total_count,
        "median_length_um": round(avg_median, 2),
        "score": round(total_score, 2)
    })
    
    return -total_score


def apply_best_config(img, roi_mask, best_cfg):
    seg = segment_slice(img, best_cfg, z_idx=0, roi_mask=roi_mask)
    meas = measure_spermatids(seg, best_cfg)
    
    overlay = make_overlay(img, meas['skel_label'])
    
    from skimage.segmentation import find_boundaries
    b = find_boundaries(roi_mask)
    overlay[b] = [255, 0, 0]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(f"Optimized Overlay (Total N={len(meas['results'])})")
    plt.axis('off')
    plt.show(block=False)

def cb_generation(xk, convergence):
    print(f"\nGeneration complete. Population convergence: {convergence:.3f}")

def main():
    global images_to_eval, roi_mask_global, results_list
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=r"C:\Users\dmishra\Desktop\sperm images", help="Directory containing .tif slices")
    parser.add_argument("--slices", default="15,40,49,54", help="Comma-separated Z-slices to use for tuning (default: 15,40,49,54)")
    parser.add_argument("--new-roi", action="store_true", help="Force drawing a new ROI, overriding saved ones.")
    args = parser.parse_args()
    
    z_list = [int(x.strip()) for x in args.slices.split(",")]
    files = glob.glob(os.path.join(args.dir, "*.tif")) + glob.glob(os.path.join(args.dir, "*.tiff"))
    
    if not files:
        print(f"No .tif files found in {args.dir}")
        sys.exit(1)
        
    found_paths = []
    for f in files:
        import re
        m = re.search(r'z(\d+)', os.path.basename(f), re.IGNORECASE)
        if m:
            z_val = int(m.group(1))
            if z_val in z_list:
                found_paths.append(f)
                
    if not found_paths:
        print("Could not find any of the requested slices. Falling back to the first 4 slices.")
        found_paths = files[:4]
        
    for f in found_paths:
        print(f"Loading {os.path.basename(f)}...")
        img = tifffile.imread(f)
        if img.ndim > 2:
            img = img[0]
            if img.ndim > 2: img = img[:, :, 0]
        images_to_eval.append(img)
        
    print(f"\nSuccessfully loaded {len(images_to_eval)} images for multi-slice optimization.")
    
    roi_img = images_to_eval[0]
    roi_mask = None
    
    if not args.new_roi and os.path.exists(ROI_SAVE_PATH):
        try:
            print(f"\nLoaded previously drawn ROI from {ROI_SAVE_PATH}.")
            print("Tip: If you want to draw a new ROI, run this again with --new-roi")
            roi_mask = tifffile.imread(ROI_SAVE_PATH).astype(bool)
            if roi_mask.shape != roi_img.shape:
                print("Warning: Saved ROI dimensions do not match current images. Forcing redraw.")
                roi_mask = None
        except Exception:
            roi_mask = None

    if roi_mask is None:
        print("\nPlease draw a GLOBAL ROI in the popup window.")
        print("This ROI will be applied to ALL loaded slices, so draw a slightly loose perimeter.")
        print("- Left click to place points\n- Right click to remove point\n- Press ENTER to finalize.")
        
        polygon_pts = []
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(normalize_display(roi_img), cmap='gray')
        ax.set_title("Draw global tubule ROI, then press ENTER on your keyboard")
        
        def on_select(verts):
            nonlocal polygon_pts
            polygon_pts = verts
            
        def on_key(event):
            if event.key == 'enter':
                if len(polygon_pts) > 2:
                    plt.close(fig)
                else:
                    print("Please draw at least 3 points before pressing Enter.")
                
        fig.canvas.mpl_connect('key_press_event', on_key)
        selector = PolygonSelector(ax, on_select)
        
        # Force the Tkinter Window to steal focus and pop to the exact front
        try:
            fig.canvas.manager.window.attributes('-topmost', 1)
            fig.canvas.manager.window.attributes('-topmost', 0)
        except Exception:
            pass
            
        plt.show(block=True)
        
        if len(polygon_pts) < 3:
            print("Invalid or no ROI drawn. Exiting.")
            sys.exit(1)
            
        H, W = roi_img.shape
        y, x = np.mgrid[:H, :W]
        points = np.column_stack((x.ravel(), y.ravel()))
        path = Path(polygon_pts)
        roi_mask = path.contains_points(points).reshape(H, W)
        
        tifffile.imwrite(ROI_SAVE_PATH, roi_mask.astype(np.uint8) * 255)
        print(f"Saved drawn ROI to {ROI_SAVE_PATH} for future runs!")
    
    # In earlier versions we cropped the image, but doing so across 4 wildly different slices is bad.
    # Instead, we just pass the full image and the `roi_mask` into segment_slice, which natively handles it!
    roi_mask_global = roi_mask
    
    bounds = [
        (65.0, 99.0), # THRESHOLD_HI (Lower limit widened to allow dimmer structures inside the cyst to be captured)
        (55.0, 95.0), # THRESHOLD_LO (Lower limit widened)
        (2.0, 30.0),  # BG_SIGMA 
        (0.01, 0.15), # CLAHE_CLIP 
        (2.0, 15.0),  # MIN_SKEL_LEN_PX
        (10.0, 35.0), # MAX_WIDTH_PX 
        (1.0, 4.0),   # MIN_LENGTH_WIDTH_RATIO 
        (3.0, 999.0), # MAX_TORTUOSITY (Removed ceiling to allow infinite webs)
        (10.0, 999.0),# MAX_BRANCH_NODES (Removed ceiling)
        (5.0, 999.0)  # MAX_ENDPOINT_COUNT (Removed ceiling)
    ]
    
    print("\n--- Starting Deep Differential Evolution Multi-Slice Optimization ---")
    t_start = time.time()
    
    res = differential_evolution(
        func=objective_fn,
        bounds=bounds,
        maxiter=12,        # Generations
        popsize=10,        # Population
        mutation=(0.5, 1.0),
        recombination=0.7,
        callback=cb_generation,
        disp=False
    )
    
    t_end = time.time()
    print(f"\nOptimization Finished in {t_end - t_start:.1f} seconds. Analyzed {eval_count} configurations.")
    
    results_list.sort(key=lambda x: (x['score'], x['count']), reverse=True)
    
    best = results_list[0]
    print("\n--- OPTIMAL GLOBAL CONFIGURATION ---")
    print(f"  Max Output Score:  {best['score']:.2f}")
    print(f"  Spermatid Count:   {best['count']}")
    print(f"  Average MedLen:    {best['median_length_um']} um")
    print(f"  Found Parameters:  {best['params']}")
        
    with open("tuning_results.json", "w") as f:
        json.dump(results_list, f, indent=2)
    print("Full search history saved to 'tuning_results.json'.")
    
    if len(results_list) > 0:
        best_cfg = CONFIG.copy()
        best_cfg.update(best["params"])
        
        # Display overlay preview for the first and last image in the batch
        apply_best_config(images_to_eval[0], roi_mask_global, best_cfg)
        if len(images_to_eval) > 1:
            apply_best_config(images_to_eval[-1], roi_mask_global, best_cfg)
        
        plt.show(block=True)

if __name__ == "__main__":
    main()
