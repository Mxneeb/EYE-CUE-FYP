"""
Path Planner Module (PPM) — Fuzzy-logic navigation guidance with
depth-gated obstacle masking.

Pipeline:
  1. Depth-gated masking:  seg_mask AND (depth < 3m) → valid_obstacle_mask
  2. 6-sector grid (2×3, full-coverage, no blind spots)
  3. OStatus computation per sector
  4. Prominent obstacle identification from mid sectors
  5. Overlapping trapezoidal fuzzy membership → 3-rule engine
  6. Centroid (Mamdani) defuzzification → continuous [-1, +1]
  7. 5-level action classification + STOP safety fallback
"""

import cv2
import numpy as np

from nav_assist.config import (
    ADE20K_CLASSES, PATH_CLASS_INDICES,
    ALERT_ZONE_PORTRAIT_X_MARGIN, ALERT_ZONE_LANDSCAPE_X_MARGIN,
    ALERT_ZONE_Y_FRACTION,
    PPM_OVERHEAD_TOP, PPM_OVERHEAD_BOT,
    PPM_GROUND_TOP, PPM_COL_LEFT, PPM_COL_RIGHT,
    DEPTH_3M_RATIO,
)

SECTOR_NAMES = [
    'top_left', 'top_mid', 'top_right',
    'bot_left', 'bot_mid', 'bot_right',
]


# ════════════════════════════════════════════════════════════════════════════
# Alert Zone (Fig. 5)
# ════════════════════════════════════════════════════════════════════════════

def compute_alert_zone(h, w):
    """
    Alert zone mask per Fig. 5: bottom 40% of image, horizontally narrowed
    to the forward-facing cone (~40deg cane swing angle).

    Portrait (h > w): center 60% of width  (0.2W margin each side)
    Landscape (w >= h): center 40% of width (0.3W margin each side)

    Returns (y_start, x_start, x_end) — the rectangular alert region.
    """
    is_portrait = h > w
    x_margin = (ALERT_ZONE_PORTRAIT_X_MARGIN if is_portrait
                else ALERT_ZONE_LANDSCAPE_X_MARGIN)

    y_start = int((1.0 - ALERT_ZONE_Y_FRACTION) * h)
    x_start = int(x_margin * w)
    x_end = int((1.0 - x_margin) * w)
    return y_start, x_start, x_end


# ════════════════════════════════════════════════════════════════════════════
# Depth-Gated Obstacle Masking
# ════════════════════════════════════════════════════════════════════════════

def create_depth_gated_mask(seg_mask, depth_map, depth_ratio=None):
    """
    Create a strict obstacle mask by logical AND of semantic detection
    and depth proximity.

    Only pixels that (a) belong to a non-walkable semantic class AND
    (b) fall within the ~3-metre depth threshold are marked as obstacles.
    Everything further away is strictly zeroed.

    Parameters
    ----------
    seg_mask : np.ndarray (H, W) uint8
        Semantic segmentation class indices (0..149, ADE20K).
    depth_map : np.ndarray (H, W) float32
        Disparity / inverse-depth map (higher = nearer).
    depth_ratio : float, optional
        Fraction of max disparity for the near-gate.  Defaults to
        ``DEPTH_3M_RATIO`` from config.

    Returns
    -------
    valid_mask : np.ndarray (H, W) bool
        True where a near, non-walkable obstacle exists.
    obstacle_labels : np.ndarray (H, W) int16
        Semantic class ID per obstacle pixel; -1 elsewhere.
    """
    if depth_ratio is None:
        depth_ratio = DEPTH_3M_RATIO

    h, w = seg_mask.shape[:2]

    # Align depth resolution to segmentation
    if depth_map.shape[:2] != (h, w):
        depth = cv2.resize(depth_map.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
    else:
        depth = depth_map.astype(np.float32)

    max_d = depth.max()
    if max_d < 1e-6:
        return (np.zeros((h, w), dtype=bool),
                np.full((h, w), -1, dtype=np.int16))

    # Depth gate: keep only pixels within the ~3 m alert radius
    near_mask = depth >= (depth_ratio * max_d)

    # Semantic gate: exclude walkable surfaces (floor, road, sidewalk …)
    path_mask = np.isin(seg_mask, list(PATH_CLASS_INDICES))

    # Valid obstacle = near AND not-walkable
    valid_mask = near_mask & ~path_mask

    # Per-pixel labels
    obstacle_labels = np.full((h, w), -1, dtype=np.int16)
    obstacle_labels[valid_mask] = seg_mask[valid_mask].astype(np.int16)

    return valid_mask, obstacle_labels


# ════════════════════════════════════════════════════════════════════════════
# 6-Sector Grid (Fig. 7)
# ════════════════════════════════════════════════════════════════════════════

def compute_sector_bounds(h, w):
    """
    Divide the image into a 2×3 grid (6 sectors) — **no blind spots**.

    Vertical (image coords, y=0 at top):
      Overhead (Sectors 1,2,3): 0 → 0.5H
      Ground   (Sectors 4,5,6): 0.5H → H

    Horizontal (landscape orientation):
      Left (Sectors 1,4): 0 → 0.3W
      Mid  (Sectors 2,5): 0.3W → 0.7W
      Right(Sectors 3,6): 0.7W → W

    Returns dict: sector_name -> (y0, y1, x0, x1).
    """
    oh_top = int(PPM_OVERHEAD_TOP * h)
    oh_bot = int(PPM_OVERHEAD_BOT * h)
    gr_top = int(PPM_GROUND_TOP * h)

    c_left = int(PPM_COL_LEFT * w)
    c_right = int(PPM_COL_RIGHT * w)

    return {
        'top_left':  (oh_top, oh_bot, 0,       c_left),
        'top_mid':   (oh_top, oh_bot, c_left,  c_right),
        'top_right': (oh_top, oh_bot, c_right, w),
        'bot_left':  (gr_top, h,     0,       c_left),
        'bot_mid':   (gr_top, h,     c_left,  c_right),
        'bot_right': (gr_top, h,     c_right, w),
    }


def compute_ostatus(obstacle_mask, obstacle_labels, sector_bounds):
    """
    Compute OStatus (obstacle pixel density) and track semantic labels
    per sector (Fig. 7 / Fig. 9).

    Parameters
    ----------
    obstacle_mask : (H, W) bool
    obstacle_labels : (H, W) int16   (-1 = no obstacle, else class_id)
    sector_bounds : dict from compute_sector_bounds()

    Returns
    -------
    ostatus : dict  sector_name -> float (0..1)
    sector_labels : dict  sector_name -> {class_id: pixel_count}
    """
    ostatus = {}
    sector_labels = {}

    for name, (y0, y1, x0, x1) in sector_bounds.items():
        region_mask = obstacle_mask[y0:y1, x0:x1]
        total = region_mask.size
        obs_pixels = int(region_mask.sum())
        ostatus[name] = obs_pixels / total if total > 0 else 0.0

        # Semantic labels in this sector (obstacle pixels only)
        region_labels = obstacle_labels[y0:y1, x0:x1]
        obs_label_vals = region_labels[region_mask]
        labels_dict = {}
        if len(obs_label_vals) > 0:
            unique, counts = np.unique(obs_label_vals, return_counts=True)
            for cls_id, cnt in zip(unique, counts):
                if cls_id >= 0:
                    labels_dict[int(cls_id)] = int(cnt)
        sector_labels[name] = labels_dict

    return ostatus, sector_labels


def find_prominent_obstacle(sector_labels):
    """
    Identify the prominent obstacle: the class with the maximum pixel
    strength in the mid sectors (top_mid + bot_mid), per Fig. 7.

    Also determines position (ahead vs overhead) based on which mid
    sector contributes more pixels of that class (Fig. 6).

    Returns
    -------
    class_name : str or None
    class_id : int or -1
    position : 'ahead' or 'overhead'
    """
    mid_counts = {}
    for sector in ('top_mid', 'bot_mid'):
        for cls_id, count in sector_labels.get(sector, {}).items():
            mid_counts[cls_id] = mid_counts.get(cls_id, 0) + count

    if not mid_counts:
        return None, -1, 'ahead'

    prominent_id = max(mid_counts, key=mid_counts.get)
    class_name = (ADE20K_CLASSES[prominent_id]
                  if prominent_id < len(ADE20K_CLASSES) else 'unknown')

    top_count = sector_labels.get('top_mid', {}).get(prominent_id, 0)
    bot_count = sector_labels.get('bot_mid', {}).get(prominent_id, 0)
    position = 'overhead' if top_count > bot_count else 'ahead'

    return class_name, prominent_id, position


# ════════════════════════════════════════════════════════════════════════════
# Trapezoidal Membership Functions (Fig. 9)
# ════════════════════════════════════════════════════════════════════════════

def _trapz(x, a, b, c, d):
    """Trapezoidal MF: 0 outside [a,d], rises a->b, flat b->c, falls c->d."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


# Input MFs for OStatus (0..1) — overlapping, crossover at ~0.25
def mu_free(v):
    """Sector is FREE of obstacles.  1.0 at v≤0.10, drops to 0.0 at v=0.40."""
    return _trapz(v, -0.01, 0.0, 0.10, 0.40)


def mu_blocked(v):
    """Sector is BLOCKED by obstacles.  0.0 at v≤0.10, rises to 1.0 at v=0.40."""
    return _trapz(v, 0.10, 0.40, 1.0, 1.01)


# Output MFs over navigation space [-1, 1]  (-1=left, 0=ahead, +1=right)
def _mu_out_left(x):
    return _trapz(x, -1.01, -1.0, -0.5, -0.1)


def _mu_out_ahead(x):
    return _trapz(x, -0.35, -0.05, 0.05, 0.35)


def _mu_out_right(x):
    return _trapz(x, 0.1, 0.5, 1.0, 1.01)


# ════════════════════════════════════════════════════════════════════════════
# Fuzzy Rule Evaluation (Fig. 9)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_rules(ostatus):
    """
    Evaluate three fuzzy rules using both overhead and ground sectors.

    Rule 1 (Move Ahead):  IF top_mid AND bot_mid are "no-obstacle"
    Rule 2 (Move Left):   IF (top_mid OR bot_mid is "obstacle")
                           AND (top_left AND bot_left are "no-obstacle")
    Rule 3 (Move Right):  IF (top_mid OR bot_mid is "obstacle")
                           AND (top_right AND bot_right are "no-obstacle")

    Returns dict with firing strengths for each rule.
    """
    tm = ostatus['top_mid']
    bm = ostatus['bot_mid']
    tl = ostatus['top_left']
    bl = ostatus['bot_left']
    tr = ostatus['top_right']
    br = ostatus['bot_right']

    # Rule 1: Move Ahead — both mid sectors are free
    r1 = min(mu_free(tm), mu_free(bm))

    # Rule 2: Move Left — mid is blocked AND left side is free
    mid_blocked = max(mu_blocked(tm), mu_blocked(bm))
    left_free = min(mu_free(tl), mu_free(bl))
    r2 = min(mid_blocked, left_free)

    # Rule 3: Move Right — mid is blocked AND right side is free
    right_free = min(mu_free(tr), mu_free(br))
    r3 = min(mid_blocked, right_free)

    return {'move_ahead': r1, 'move_left': r2, 'move_right': r3}


def defuzzify(rule_strengths):
    """
    Centroid defuzzification (Mamdani) over [-1, 1] navigation space.

    Clips each output MF to the rule's firing strength, aggregates
    with max, then computes the centroid.
    """
    x_pts = np.linspace(-1.0, 1.0, 201)
    aggregated = np.zeros_like(x_pts)

    for i, x in enumerate(x_pts):
        ml = min(rule_strengths['move_left'],  _mu_out_left(x))
        ma = min(rule_strengths['move_ahead'], _mu_out_ahead(x))
        mr = min(rule_strengths['move_right'], _mu_out_right(x))
        aggregated[i] = max(ml, ma, mr)

    total = aggregated.sum()
    if total < 1e-8:
        return 0.0
    return float((x_pts * aggregated).sum() / total)


def classify_action(centroid):
    """Map defuzzified centroid to a 5-level action string."""
    if centroid < -0.40:
        return 'MOVE LEFT'
    elif centroid < -0.15:
        return 'MOVE SLIGHT LEFT'
    elif centroid <= 0.15:
        return 'MOVE AHEAD'
    elif centroid <= 0.40:
        return 'MOVE SLIGHT RIGHT'
    else:
        return 'MOVE RIGHT'


# ════════════════════════════════════════════════════════════════════════════
# Main Interface
# ════════════════════════════════════════════════════════════════════════════

def plan_path(seg_mask, depth_map):
    """
    Full PPM pipeline: depth-gate → sectors → fuzzy → instruction.

    Parameters
    ----------
    seg_mask : np.ndarray (H, W) uint8
        Semantic segmentation class indices (ADE20K, 0..149).
    depth_map : np.ndarray (H, W) float32
        Disparity / inverse-depth map (higher values = nearer).

    Returns
    -------
    instruction : str
        e.g. "Desk Ahead. Move Slight Left"
    details : dict
        action, prominent_obstacle, position, ostatus, sector_bounds,
        sector_labels, rule_strengths, centroid, valid_mask.
    """
    # ── Depth-gated obstacle mask ──────────────────────────────────────
    valid_mask, obstacle_labels = create_depth_gated_mask(seg_mask, depth_map)

    h, w = valid_mask.shape[:2]

    # ── 6-sector OStatus (Fig. 7) ──────────────────────────────────────
    sector_bounds = compute_sector_bounds(h, w)
    ostatus, sector_labels = compute_ostatus(
        valid_mask, obstacle_labels, sector_bounds)

    # ── Prominent obstacle from mid sectors (Fig. 7) ───────────────────
    prominent_name, prominent_id, position = find_prominent_obstacle(
        sector_labels)

    # ── Fuzzy rule evaluation (Fig. 9) ─────────────────────────────────
    rule_strengths = evaluate_rules(ostatus)

    # ── Centroid defuzzification ───────────────────────────────────────
    centroid = defuzzify(rule_strengths)
    action = classify_action(centroid)

    # ── STOP override: all bottom sectors heavily blocked ──────────────
    if all(ostatus[s] > 0.50 for s in ('bot_left', 'bot_mid', 'bot_right')):
        action = 'STOP'

    # ── Format instruction ────────────────────────────────────────────
    action_display = action.title()   # e.g. "Move Slight Left"
    if action == 'STOP':
        instruction = 'Stop'
    elif prominent_name:
        obs_display = prominent_name.capitalize()
        pos_display = 'Ahead' if position == 'ahead' else 'overhead'
        instruction = f'{obs_display} {pos_display}. {action_display}'
    else:
        instruction = action_display

    details = {
        'action': action,
        'prominent_obstacle': prominent_name,
        'position': position,
        'ostatus': ostatus,
        'sector_bounds': sector_bounds,
        'sector_labels': sector_labels,
        'rule_strengths': rule_strengths,
        'centroid': centroid,
        'valid_mask': valid_mask,
    }

    return instruction, details
