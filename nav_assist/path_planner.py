"""
Path Planner Module (PPM) — Paper-accurate fuzzy-logic navigation guidance.

Architecture (from Obs-tackle paper):
  Fig. 5  — Alert zone: bottom 40% of image, horizontally constrained
  Fig. 6  — Ground (bottom half) vs overhead (top half) vertical split
  Fig. 7  — 6-sector grid: [top_left, top_mid, top_right,
                             bot_left, bot_mid, bot_right]
            Prominent obstacle = class with max pixel strength in mid sectors
  Fig. 9  — Trapezoidal fuzzy membership, 3 rules (Move Ahead/Left/Right),
            centroid defuzzification
  Fig. 10 — Output: "{obstacle} {ahead|overhead}, Move {left|right|ahead}"
"""

import numpy as np

from nav_assist.config import (
    ADE20K_CLASSES,
    ALERT_ZONE_PORTRAIT_X_MARGIN, ALERT_ZONE_LANDSCAPE_X_MARGIN,
    ALERT_ZONE_Y_FRACTION,
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
# 6-Sector Grid (Fig. 7)
# ════════════════════════════════════════════════════════════════════════════

def compute_sector_bounds(h, w):
    """
    Divide the image into a 3x2 grid (Fig. 7):
        [1:top_left | 2:top_mid | 3:top_right]   ← overhead region
        [4:bot_left | 5:bot_mid | 6:bot_right]   ← ground region

    Returns dict: sector_name -> (y0, y1, x0, x1).
    """
    row_split = h // 2
    c1, c2 = w // 3, 2 * w // 3

    return {
        'top_left':  (0, row_split, 0,  c1),
        'top_mid':   (0, row_split, c1, c2),
        'top_right': (0, row_split, c2, w),
        'bot_left':  (row_split, h, 0,  c1),
        'bot_mid':   (row_split, h, c1, c2),
        'bot_right': (row_split, h, c2, w),
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


# Input MFs for OStatus (0..1)
def mu_free(v):
    """Sector is FREE of obstacles."""
    return _trapz(v, -0.01, 0.0, 0.15, 0.30)


def mu_blocked(v):
    """Sector is BLOCKED by obstacles."""
    return _trapz(v, 0.30, 0.50, 1.0, 1.01)


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
    Evaluate the three fuzzy rules from Fig. 9.

    Rule 1 (Move Ahead): IF bot_mid is FREE
    Rule 2 (Move Left):  IF right/center BLOCKED AND left FREE AND center NOT FREE
    Rule 3 (Move Right): IF left/center BLOCKED AND right FREE AND center NOT FREE

    Returns dict with firing strengths for each rule.
    """
    bl = ostatus['bot_left']
    bm = ostatus['bot_mid']
    br = ostatus['bot_right']

    not_free_mid = 1.0 - mu_free(bm)

    # Rule 1: Move Ahead — center ground path is free
    r1 = mu_free(bm)

    # Rule 2: Move Left — obstacles on right, left is free
    r2a = min(mu_blocked(br), mu_free(bl), not_free_mid)
    r2b = min(mu_blocked(bm), mu_free(bl))
    r2 = max(r2a, r2b)

    # Rule 3: Move Right — obstacles on left, right is free
    r3a = min(mu_blocked(bl), mu_free(br), not_free_mid)
    r3b = min(mu_blocked(bm), mu_free(br))
    r3 = max(r3a, r3b)

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
    """Map defuzzified centroid to a discrete action."""
    if centroid < -0.20:
        return 'MOVE LEFT'
    elif centroid > 0.20:
        return 'MOVE RIGHT'
    else:
        return 'MOVE AHEAD'


# ════════════════════════════════════════════════════════════════════════════
# Main Interface
# ════════════════════════════════════════════════════════════════════════════

def plan_path(obstacle_mask, obstacle_labels, obstacle_info=None):
    """
    Full path planner pipeline per paper architecture (Fig. 5-10).

    Parameters
    ----------
    obstacle_mask : np.ndarray (H, W) bool
        Binary obstacle mask from ODM.
    obstacle_labels : np.ndarray (H, W) int16
        Per-pixel semantic class ID (-1 = non-obstacle).
    obstacle_info : list[dict], optional
        Obstacle metadata sorted nearest-first.

    Returns
    -------
    instruction : str
        e.g. "MOVE LEFT — person ahead"
    details : dict
        action, prominent_obstacle, position, ostatus, rule_strengths, centroid.
    """
    if obstacle_info is None:
        obstacle_info = []

    h, w = obstacle_mask.shape[:2]

    # ── 6-sector OStatus (Fig. 7) ──────────────────────────────────────
    sector_bounds = compute_sector_bounds(h, w)
    ostatus, sector_labels = compute_ostatus(
        obstacle_mask, obstacle_labels, sector_bounds)

    # ── Prominent obstacle from mid sectors (Fig. 7) ───────────────────
    prominent_name, prominent_id, position = find_prominent_obstacle(
        sector_labels)

    # ── Fuzzy rule evaluation (Fig. 9) ─────────────────────────────────
    rule_strengths = evaluate_rules(ostatus)

    # ── Centroid defuzzification ───────────────────────────────────────
    centroid = defuzzify(rule_strengths)
    action = classify_action(centroid)

    # ── STOP override: all bottom sectors heavily blocked ──────────────
    if all(ostatus[s] > 0.45 for s in ('bot_left', 'bot_mid', 'bot_right')):
        action = 'STOP'

    # ── Format instruction (Fig. 10) ──────────────────────────────────
    if prominent_name:
        instruction = f'{action} — {prominent_name} {position}'
    elif action == 'MOVE AHEAD':
        instruction = f'{action} — path clear'
    else:
        instruction = action

    details = {
        'action': action,
        'prominent_obstacle': prominent_name,
        'position': position,
        'ostatus': ostatus,
        'sector_labels': sector_labels,
        'rule_strengths': rule_strengths,
        'centroid': centroid,
    }

    return instruction, details
