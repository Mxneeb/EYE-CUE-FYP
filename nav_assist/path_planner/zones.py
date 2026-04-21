"""
Zone definitions for the Path Planner Module.

Based on the research paper:
  - Fig 5: Horizontal zone division (left/center/right) with
    portrait/landscape adaptation and 3m alert radius
  - Fig 6: Vertical split — ground obstacles (bottom 40% of image)
    and overhead obstacles (10%-40% from top)

The frame is divided into a 2x3 grid of zones representing the
walking space ahead of the user.
"""

import numpy as np


# ── Fig 5: Horizontal zone margins ──────────────────────────────────────
# Portrait (h > w): center 60% of width  (0.2W margin each side)
# Landscape (w >= h): center 40% of width (0.3W margin each side)
PORTRAIT_X_MARGIN = 0.20
LANDSCAPE_X_MARGIN = 0.30

# ── Fig 6: Vertical zone boundaries ────────────────────────────────────
# Paper coords have 0 at bottom, H at top.
# Ground obstacles: paper 0→0.4H  →  image y: 0.6H→H  (bottom 40%)
# Overhead obstacles: paper 0.6H→0.9H → image y: 0.1H→0.4H
GROUND_Y_START_FRAC = 0.60     # image y = 0.6 * H
OVERHEAD_Y_START_FRAC = 0.10   # image y = 0.1 * H
OVERHEAD_Y_END_FRAC = 0.40     # image y = 0.4 * H

# Alert zone: bottom 40% of image (≈3m walking radius)
ALERT_ZONE_Y_FRAC = 0.40


def compute_zones(h, w):
    """
    Divide the frame into 6 zones (3 columns x 2 rows) per Fig 5 & Fig 6.

    Columns (horizontal, adaptive to orientation):
      Portrait:  LEFT 0-0.2W | CENTER 0.2W-0.8W | RIGHT 0.8W-W
      Landscape: LEFT 0-0.3W | CENTER 0.3W-0.7W | RIGHT 0.7W-W

    Rows (vertical, per Fig 6):
      Overhead: image y from 0.1H to 0.4H  (paper 0.6H-0.9H)
      Ground:   image y from 0.6H to H     (paper 0-0.4H)

    Parameters
    ----------
    h, w : int
        Frame dimensions.

    Returns
    -------
    dict : zone_name -> (y0, y1, x0, x1)
    """
    is_portrait = h > w
    x_margin = PORTRAIT_X_MARGIN if is_portrait else LANDSCAPE_X_MARGIN

    x_left_end = int(x_margin * w)
    x_right_start = int((1.0 - x_margin) * w)

    # Ground zone (bottom of image, ~3m radius)
    gy0 = int(GROUND_Y_START_FRAC * h)
    gy1 = h

    # Overhead zone (upper portion of image)
    oy0 = int(OVERHEAD_Y_START_FRAC * h)
    oy1 = int(OVERHEAD_Y_END_FRAC * h)

    return {
        'ground_left':     (gy0, gy1, 0, x_left_end),
        'ground_center':   (gy0, gy1, x_left_end, x_right_start),
        'ground_right':    (gy0, gy1, x_right_start, w),
        'overhead_left':   (oy0, oy1, 0, x_left_end),
        'overhead_center': (oy0, oy1, x_left_end, x_right_start),
        'overhead_right':  (oy0, oy1, x_right_start, w),
    }


def compute_alert_zone(h, w):
    """
    3m alert zone from Fig 5: bottom 40% of frame, narrowed
    horizontally to the forward-facing cone.

    Returns (y_start, x_start, x_end).
    """
    is_portrait = h > w
    x_margin = PORTRAIT_X_MARGIN if is_portrait else LANDSCAPE_X_MARGIN

    y_start = int((1.0 - ALERT_ZONE_Y_FRAC) * h)
    x_start = int(x_margin * w)
    x_end = int((1.0 - x_margin) * w)
    return y_start, x_start, x_end
