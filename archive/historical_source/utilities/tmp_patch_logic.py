import math

file_path = r'c:\Users\dmishra\Desktop\sperm_project\sperm_segmentation_saturnv3.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_logic = '''def check_extension_consistency(prev_state, candidate_detection, cfg, overlap_exists=False):
    """
    Check if extending a track with this detection would be biologically consistent.
    Implements 'Continue Unless Implausible' logic for overlapping footprints.
    
    Stage 2b: All thresholds are now driven by CONFIG for hyperparameter tuning.
    """
    um_xy = cfg["UM_PER_PX_XY"]
    
    # Read tunable Stage 2 parameters from CONFIG
    stab_thresh = cfg.get("OVERLAP_STABILITY_THRESHOLD", 0.08)
    ori_deg     = cfg.get("OVERLAP_ORIENTATION_DEG", 15.0)
    ovl_mult    = cfg.get("OVERLAP_MULTIPLIER", 1.35)
    min_stable  = cfg.get("OVERLAP_MIN_STABLE_COUNT", 1)
    
    # Extract previous track state
    prev_x = prev_state["last_x"]
    prev_y = prev_state["last_y"]
    prev_width = prev_state.get("last_width")
    prev_length = prev_state.get("last_length")
    prev_area = prev_state.get("last_area")
    prev_ori = prev_state.get("last_orientation")
    
    # Extract candidate detection features
    cand_x = candidate_detection["centroid_x"]
    cand_y = candidate_detection["centroid_y"]
    cand_width = candidate_detection.get("width_um")
    cand_length = candidate_detection.get("length_um_geodesic")
    cand_area = candidate_detection.get("area_px")
    cand_ori = candidate_detection.get("orientation")
    
    # Logic: 
    # If overlap_exists, we allow the track to continue IF enough primary metrics are stable.
    # This prevents 'monster merges' where a track jumps onto a totally different cell.
    if overlap_exists:
        stable_count = 0
        
        # 1. Width stability
        if prev_width and cand_width:
            if (abs(cand_width - prev_width) / max(prev_width, 1e-9)) < stab_thresh:
                stable_count += 1
            
        # 2. Area stability
        if prev_area and cand_area:
            if (abs(cand_area - prev_area) / max(prev_area, 1e-9)) < stab_thresh:
                stable_count += 1
            
        # 3. Orientation stability
        if prev_ori is not None and cand_ori is not None:
            diff_rad = abs(cand_ori - prev_ori)
            if diff_rad > math.pi / 2:
                diff_rad = math.pi - diff_rad
            if diff_rad < (ori_deg * math.pi / 180):
                stable_count += 1
        
        # 4. Length stability
        if prev_length and cand_length:
            if (abs(cand_length - prev_length) / max(prev_length, 1e-9)) < stab_thresh:
                stable_count += 1
            
        # Require minimum stable metrics to continue an overlapping track
        if stable_count < min_stable:
            return False, f"overlap_but_{stable_count}_stable"
        
        # Even with stable metrics, still apply capped multiplier for fallback checks
        multiplier = ovl_mult
    else:
        multiplier = 1.0

    # 1. Check centroid jump
    dx = cand_x - prev_x
    dy = cand_y - prev_y
    centroid_jump_um = math.sqrt(dx*dx + dy*dy) * um_xy
    
    if not overlap_exists and centroid_jump_um > cfg["CONSERVATIVE_MAX_CENTROID_JUMP_UM"]:
        return False, f"centroid_jump={centroid_jump_um:.2f}um"
    
    # 2. Check width consistency
    if prev_width is not None and cand_width is not None:
        width_ratio = abs(cand_width - prev_width) / max(prev_width, 1e-9)
        if width_ratio > cfg["CONSERVATIVE_MAX_WIDTH_JUMP_RATIO"] * multiplier:
            return False, f"width_jump={width_ratio:.2f}"
            
    # 3. Check length consistency
    if prev_length is not None and cand_length is not None:
        length_ratio = abs(cand_length - prev_length) / max(prev_length, 1e-9)
        if length_ratio > cfg["CONSERVATIVE_MAX_LENGTH_JUMP_RATIO"] * multiplier:
            return False, f"length_jump={length_ratio:.2f}"
            
    # 4. Check area consistency
    if prev_area is not None and cand_area is not None:
        area_ratio = abs(cand_area - prev_area) / max(prev_area, 1e-9)
        if area_ratio > cfg["CONSERVATIVE_MAX_AREA_JUMP_RATIO"] * multiplier:
            return False, f"area_jump={area_ratio:.2f}"
    
    return True, "ok"
'''

# Find the function range
start_line = -1
end_line = -1
for i, line in enumerate(lines):
    if 'def check_extension_consistency' in line:
        start_line = i
    if start_line != -1 and i > start_line and line.strip().startswith('def '):
        end_line = i
        break

if start_line != -1 and end_line != -1:
    new_content = lines[:start_line] + [new_logic + '\n'] + lines[end_line:]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_content)
    print(f'Successfully patched (lines {start_line+1}-{end_line})')
else:
    print(f'Failed. start={start_line}, end={end_line}')
    exit(1)
