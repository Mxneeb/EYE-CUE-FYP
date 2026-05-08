"""
Stair hazard detection for the navigation pipeline.

Scans ADE20K class 54 (stairs;steps) and 60 (stairway;staircase) from the
TopFormer segmentation map and classifies direction (up / down) using the
Depth Anything V2 disparity map.

The function is **stateless** — all repeat-interval and speech logic lives
in the caller (PathPlanner). The function either returns a confident alert
dict or None. Nothing in between.

Anti-false-positive layers (any failure → None):
  1. Depth proximity gate — same 0.40 ratio as the obstacle pipeline;
     stair labels beyond ~3 m are ignored entirely.
  2. Absolute minimum pixel count — scatter / single-pixel noise is rejected
     before any geometry check runs.
  3. Per-cell pixel-ratio threshold (3%, per spec) — must be met in at least
     one grid cell after depth gating.
  4. Lower-two-thirds spatial constraint — stair pixels above y = H/3 are not
     an immediate walking hazard; ignored for all subsequent checks.
  5. Absolute minimum after spatial filtering — re-checked after restricting
     to the lower two-thirds.
  6. Spatial compactness — stair pixels must fill ≥ 10 % of their bounding
     box; scattered misclassifications cannot form a compact region.
  7. Minimum vertical span — the stair region must span ≥ 4 % of frame height;
     single-row artefacts cannot pass.
  8. Depth gradient R² ≥ 0.45 — direction is reported only when the linear
     depth-vs-row fit is statistically meaningful.
  9. Gradient magnitude ≥ 6 % of max disparity — depth noise cannot mimic
     the disparity change expected across a real staircase.
"""

import cv2
import numpy as np

# ── ADE20K stair class IDs ─────────────────────────────────────────────────
STAIR_CLASS_IDS: frozenset = frozenset({54, 60})

# ── Detection thresholds ───────────────────────────────────────────────────
_DEPTH_NEARBY_RATIO    = 0.40   # mirrors DEPTH_3M_RATIO in analyzer.py
_MIN_STAIR_PIXELS      = 100    # absolute lower bound before any geometry check
_CELL_RATIO_THRESHOLD  = 0.03   # 3 % of a grid cell's area (spec requirement)
_MIN_STAIR_LOWER_PX    = 100    # absolute lower bound after lower-2/3 filter
_MIN_COMPACTNESS       = 0.10   # stair_px / bounding_box_area
_MIN_HEIGHT_SPAN_FRAC  = 0.04   # stair region vertical extent / frame height
_MIN_R2                = 0.45   # minimum R² to commit to up / down direction
_MIN_GRADIENT_FRAC     = 0.06   # depth range within stair region / max depth


def check_for_stairs(
    seg_map: np.ndarray,
    depth_map: np.ndarray,
    grid_cells: dict,
) -> "dict | None":
    """
    Return a stair alert dict if stairs are confidently detected, else None.

    Parameters
    ----------
    seg_map : (H, W) uint8
        ADE20K class-index map produced by TopFormer each frame.
    depth_map : (H, W) float32
        Disparity map from Depth Anything V2 (higher value = closer).
        Must not be all-zeros; if depth is unavailable, returns None.
    grid_cells : dict
        Zone dict from ``compute_zones()``:  {name: (y0, y1, x0, x1)}.

    Returns
    -------
    dict or None
        On detection::

            {
                'detected':          True,
                'direction':         'down' | 'up' | 'unknown',
                'message':           str,         # ready to pass to speaker
                'flagged_zones':     list[str],   # zone names that exceeded 3 %
                'stair_pixel_count': int,         # depth-gated pixel count
                'gradient_r2':       float,       # R² of depth regression
            }

        None if any anti-false-positive check fails.
    """
    h, w = seg_map.shape[:2]

    # ── Layer 1: depth proximity gate ─────────────────────────────────────
    if depth_map.shape[:2] != (h, w):
        depth = cv2.resize(
            depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        depth = depth_map.astype(np.float32)

    max_d = float(depth.max())
    if max_d < 1e-6:
        return None   # depth not yet available; never fire without depth

    near_mask = depth >= (_DEPTH_NEARBY_RATIO * max_d)

    # ── Layer 2: absolute pixel count (pre-geometry) ──────────────────────
    stair_semantic = (seg_map == 54) | (seg_map == 60)
    stair_mask = stair_semantic & near_mask

    total_stair_px = int(stair_mask.sum())
    if total_stair_px < _MIN_STAIR_PIXELS:
        return None

    # ── Layer 3: per-cell pixel ratio (3%, spec requirement) ─────────────
    flagged_zones = []
    for zone_name, (y0, y1, x0, x1) in grid_cells.items():
        cell = stair_mask[y0:y1, x0:x1]
        cell_area = cell.size
        if cell_area == 0:
            continue
        if int(cell.sum()) / cell_area >= _CELL_RATIO_THRESHOLD:
            flagged_zones.append(zone_name)

    if not flagged_zones:
        return None

    # ── Layer 4: lower-two-thirds spatial constraint ──────────────────────
    ys, xs = np.where(stair_mask)
    lower_boundary_y = h // 3          # y >= h//3  →  lower 2/3 of frame
    in_lower = ys >= lower_boundary_y

    if not np.any(in_lower):
        return None

    ys_l = ys[in_lower]
    xs_l = xs[in_lower]

    # ── Layer 5: absolute pixel count after spatial filter ────────────────
    if len(ys_l) < _MIN_STAIR_LOWER_PX:
        return None

    # ── Layer 6: spatial compactness ─────────────────────────────────────
    y_lo, y_hi = int(ys_l.min()), int(ys_l.max())
    x_lo, x_hi = int(xs_l.min()), int(xs_l.max())
    bbox_area   = max(1, (y_hi - y_lo + 1) * (x_hi - x_lo + 1))
    compactness = len(ys_l) / bbox_area

    if compactness < _MIN_COMPACTNESS:
        return None

    # ── Layer 7: minimum vertical span ───────────────────────────────────
    if (y_hi - y_lo) / h < _MIN_HEIGHT_SPAN_FRAC:
        return None

    # ── Layers 8-9: depth gradient direction ─────────────────────────────
    direction, r2 = _gradient_direction(ys_l, xs_l, depth, h, max_d)

    # ── Build result ──────────────────────────────────────────────────────
    if direction == 'down':
        message = 'Caution, stairs going down ahead.'
    elif direction == 'up':
        message = 'Caution, stairs going up ahead.'
    else:
        message = 'Caution, stairs detected ahead.'

    return {
        'detected':          True,
        'direction':         direction,
        'message':           message,
        'flagged_zones':     flagged_zones,
        'stair_pixel_count': total_stair_px,
        'gradient_r2':       r2,
    }


# ── Internal helper ────────────────────────────────────────────────────────

def _gradient_direction(
    ys: np.ndarray,
    xs: np.ndarray,
    depth: np.ndarray,
    frame_h: int,
    max_depth: float,
) -> "tuple[str, float]":
    """
    Fit  depth ~ a·ŷ + b  over the stair pixels via OLS.

    Convention (Depth Anything V2 disparity, higher = closer):
        slope > 0  →  depth increases downward  →  stairs going DOWN
                       (you're at the top; bottom steps are farther away)
        slope < 0  →  depth decreases downward  →  stairs going UP
                       (you're at the bottom; upper steps are farther away)

    Returns ('down' | 'up' | 'unknown', r2_score).
    'unknown' is returned when R² < _MIN_R2 or gradient magnitude < threshold.
    """
    stair_depths = depth[ys, xs].astype(np.float64)
    valid = stair_depths > 0.0
    n_valid = int(valid.sum())
    if n_valid < 20:
        return 'unknown', 0.0

    y_f = ys[valid].astype(np.float64) / frame_h   # normalise y to [0, 1]
    d_f = stair_depths[valid]

    # OLS: slope = Cov(y,d) / Var(y)
    y_mean = y_f.mean()
    d_mean = d_f.mean()
    ss_xy  = float(np.dot(y_f - y_mean, d_f - d_mean))
    ss_xx  = float(np.dot(y_f - y_mean, y_f - y_mean))

    if ss_xx < 1e-10:
        return 'unknown', 0.0

    slope  = ss_xy / ss_xx
    d_pred = slope * (y_f - y_mean) + d_mean
    ss_res = float(np.dot(d_f - d_pred, d_f - d_pred))
    ss_tot = float(np.dot(d_f - d_mean, d_f - d_mean))
    r2     = (1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    # Layer 8: R² threshold
    if r2 < _MIN_R2:
        return 'unknown', float(r2)

    # Layer 9: gradient magnitude (rules out near-flat depth across stairs)
    d_range = float(d_f.max() - d_f.min())
    if d_range / max_depth < _MIN_GRADIENT_FRAC:
        return 'unknown', float(r2)

    return ('down' if slope > 0 else 'up'), float(r2)
