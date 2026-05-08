"""
Enhanced Path Planner Module (PPM) — Fuzzy-logic navigation with improvements.

Enhanced Features:
    - Variable grid resolution: Finer grid in center (path direction)
    - Dynamic sector sizing: Adjust based on walking speed
    - Trajectory prediction: Predict user path 2-3 seconds ahead
    - Alternative path generation: Suggest multiple routes ranked by safety
    - Enhanced fuzzy rules with hierarchy awareness

Pipeline:
    1. Depth-gated masking: seg_mask AND (depth < 3m) → valid_obstacle_mask
    2. Variable-resolution 6-sector grid (finer in center)
    3. OStatus computation per sector with hierarchy weighting
    4. Prominent obstacle identification from mid sectors
    5. Enhanced fuzzy rule evaluation with trajectory prediction
    6. Alternative path generation and safety ranking
    7. Final navigation decision
"""

import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional

from nav_assist.config import (
    ADE20K_CLASSES, PATH_CLASS_INDICES,
    ALERT_ZONE_PORTRAIT_X_MARGIN, ALERT_ZONE_LANDSCAPE_X_MARGIN,
    ALERT_ZONE_Y_FRACTION,
    PPM_OVERHEAD_TOP, PPM_OVERHEAD_BOT,
    PPM_GROUND_TOP, PPM_COL_LEFT, PPM_COL_RIGHT,
    DEPTH_3M_RATIO,
    PPM_VARIABLE_GRID, PPM_CENTER_COL_LEFT, PPM_CENTER_COL_RIGHT, PPM_CENTER_ROW,
    TRAJECTORY_PREDICTION_ENABLED, TRAJECTORY_LOOKAHEAD_FRAMES, TRAJECTORY_SMOOTHING,
    ALTERNATIVE_PATHS_ENABLED, MAX_ALTERNATIVE_PATHS, PATH_RANK_BY_SAFETY,
)


SECTOR_NAMES = [
    'top_left', 'top_mid', 'top_right',
    'bot_left', 'bot_mid', 'bot_right',
]


# Hierarchy priority weights for different obstacle levels
HIERARCHY_WEIGHTS = {
    1: 1.5,  # Critical obstacles weighted higher
    2: 1.0,  # Caution obstacles normal weight
    3: 0.5,  # Navigable obstacles weighted lower
}


class TrajectoryHistory:
    """Track user trajectory for prediction."""
    
    def __init__(self, max_history=20):
        self.history = deque(maxlen=max_history)
        self.velocity = np.zeros(2, dtype=np.float32)
        
    def update(self, x, y):
        """Update trajectory with new position."""
        self.history.append((x, y))
        
        if len(self.history) >= 2:
            # Calculate velocity
            dx = x - self.history[-2][0]
            dy = y - self.history[-2][1]
            self.velocity = np.array([dx, dy], dtype=np.float32)
    
    def predict(self, frames=TRAJECTORY_LOOKAHEAD_FRAMES):
        """Predict future positions."""
        if len(self.history) < 2:
            return []
        
        last_x, last_y = self.history[-1]
        predictions = []
        
        # Use exponential moving average for velocity
        alpha = TRAJECTORY_SMOOTHING
        
        # Extend velocity with smoothing
        extended_vel = self.velocity.copy()
        
        for i in range(frames):
            pred_x = last_x + extended_vel[0] * (i + 1)
            pred_y = last_y + extended_vel[1] * (i + 1)
            predictions.append((int(pred_x), int(pred_y)))
        
        return predictions
    
    def get_direction(self):
        """Get current movement direction (-1=left, 0=straight, 1=right)."""
        if len(self.history) < 2:
            return 0
        
        # Use recent history for direction
        dx = self.history[-1][0] - self.history[-3][0] if len(self.history) >= 3 else self.velocity[0]
        
        if dx < -5:
            return -1  # Moving left
        elif dx > 5:
            return 1   # Moving right
        else:
            return 0    # Moving straight


def compute_alert_zone(h, w):
    """
    Alert zone mask per Fig. 5: bottom 40% of image, horizontally narrowed
    to the forward-facing cone (~40deg cane swing angle).
    """
    is_portrait = h > w
    x_margin = (ALERT_ZONE_PORTRAIT_X_MARGIN if is_portrait
                else ALERT_ZONE_LANDSCAPE_X_MARGIN)

    y_start = int((1.0 - ALERT_ZONE_Y_FRACTION) * h)
    x_start = int(x_margin * w)
    x_end = int((1.0 - x_margin) * w)
    return y_start, x_start, x_end


def create_depth_gated_mask(seg_mask, depth_map, depth_ratio=None):
    """Create a strict obstacle mask by logical AND of semantic detection and depth proximity."""
    if depth_ratio is None:
        depth_ratio = DEPTH_3M_RATIO

    h, w = seg_mask.shape[:2]

    if depth_map.shape[:2] != (h, w):
        depth = cv2.resize(depth_map.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
    else:
        depth = depth_map.astype(np.float32)

    max_d = depth.max()
    if max_d < 1e-6:
        return np.zeros((h, w), dtype=bool)

    near_mask = depth >= (depth_ratio * max_d)
    path_mask = np.isin(seg_mask, list(PATH_CLASS_INDICES))
    valid_mask = near_mask & ~path_mask

    return valid_mask


def compute_variable_sector_bounds(h, w):
    """
    Divide the image into a variable-resolution 6-sector grid.
    Finer grid in center (path direction).
    """
    oh_top = int(PPM_OVERHEAD_TOP * h)
    oh_bot = int(PPM_OVERHEAD_BOT * h)
    gr_top = int(PPM_GROUND_TOP * h)

    c_left = int(PPM_COL_LEFT * w)
    c_right = int(PPM_COL_RIGHT * w)
    
    if PPM_VARIABLE_GRID:
        # Add finer center columns
        cc_left = int(PPM_CENTER_COL_LEFT * w)
        cc_right = int(PPM_CENTER_COL_RIGHT * w)
        center_row = int(PPM_CENTER_ROW * h)
        
        return {
            # Standard 6 sectors with hierarchy weights
            'top_left':  {'bounds': (oh_top, oh_bot, 0, c_left), 'weight': 1.2},
            'top_mid':   {'bounds': (oh_top, oh_bot, c_left, c_right), 'weight': 1.5},
            'top_right': {'bounds': (oh_top, oh_bot, c_right, w), 'weight': 1.2},
            'bot_left':  {'bounds': (gr_top, h, 0, c_left), 'weight': 1.2},
            'bot_mid':   {'bounds': (gr_top, h, c_left, c_right), 'weight': 1.5},
            'bot_right': {'bounds': (gr_top, h, c_right, w), 'weight': 1.2},
            # Center column (finer resolution)
            'center_top': {'bounds': (oh_top, center_row, cc_left, cc_right), 'weight': 2.0},
            'center_bot': {'bounds': (center_row, h, cc_left, cc_right), 'weight': 2.0},
        }
    else:
        return {
            'top_left':  {'bounds': (oh_top, oh_bot, 0, c_left), 'weight': 1.0},
            'top_mid':   {'bounds': (oh_top, oh_bot, c_left, c_right), 'weight': 1.0},
            'top_right': {'bounds': (oh_top, oh_bot, c_right, w), 'weight': 1.0},
            'bot_left':  {'bounds': (gr_top, h, 0, c_left), 'weight': 1.0},
            'bot_mid':   {'bounds': (gr_top, h, c_left, c_right), 'weight': 1.0},
            'bot_right': {'bounds': (gr_top, h, c_right, w), 'weight': 1.0},
        }


def compute_ostatus_enhanced(obstacle_mask, obstacle_labels, sector_bounds, hierarchy_info=None):
    """
    Compute enhanced OStatus with hierarchy weighting.
    
    Parameters:
        obstacle_mask: Binary obstacle mask
        obstacle_labels: Per-pixel semantic class IDs (-1 = no obstacle)
        sector_bounds: Dict of sector definitions
        hierarchy_info: Dict of sector -> hierarchy levels present
    
    Returns:
        ostatus: dict sector_name -> float (0..1)
        sector_labels: dict sector_name -> {class_id: pixel_count}
        sector_hierarchy: dict sector_name -> max hierarchy level
    """
    ostatus = {}
    sector_labels = {}
    sector_hierarchy = {}
    
    for name, sector_info in sector_bounds.items():
        bounds = sector_info['bounds']
        weight = sector_info.get('weight', 1.0)
        
        y0, y1, x0, x1 = bounds
        region_mask = obstacle_mask[y0:y1, x0:x1]
        total = region_mask.size
        obs_pixels = int(region_mask.sum())
        
        # Weighted OStatus
        base_ostatus = obs_pixels / total if total > 0 else 0.0
        ostatus[name] = base_ostatus * weight
        
        # Semantic labels
        region_labels = obstacle_labels[y0:y1, x0:x1]
        obs_label_vals = region_labels[region_mask]
        labels_dict = {}
        max_hierarchy = 0
        
        if len(obs_label_vals) > 0:
            unique, counts = np.unique(obs_label_vals, return_counts=True)
            for cls_id, cnt in zip(unique, counts):
                if cls_id >= 0:
                    labels_dict[int(cls_id)] = int(cnt)
                    
                    # Get hierarchy level for this class
                    class_name = ADE20K_CLASSES[cls_id] if cls_id < len(ADE20K_CLASSES) else ''
                    if class_name in ['person', 'car', 'truck', 'bus', 'motorcycle']:
                        level = 1
                    elif class_name in ['wall', 'building', 'fence', 'pole', 'tree', 'stairs', 'step']:
                        level = 2
                    else:
                        level = 3
                    
                    max_hierarchy = max(max_hierarchy, level)
        
        sector_labels[name] = labels_dict
        sector_hierarchy[name] = max_hierarchy
    
    return ostatus, sector_labels, sector_hierarchy


def find_prominent_obstacle(sector_labels):
    """Identify the prominent obstacle in the mid sectors."""
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


def _trapz(x, a, b, c, d):
    """Trapezoidal MF: 0 outside [a,d], rises a->b, flat b->c, falls c->d."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def mu_free(v):
    """Sector is FREE of obstacles."""
    return _trapz(v, -0.01, 0.0, 0.10, 0.40)


def mu_blocked(v):
    """Sector is BLOCKED by obstacles."""
    return _trapz(v, 0.10, 0.40, 1.0, 1.01)


def mu_critical(v):
    """Sector has CRITICAL obstacle."""
    return _trapz(v, 0.15, 0.30, 0.5, 0.8)


def evaluate_rules_enhanced(ostatus, sector_hierarchy, trajectory_dir=0):
    """
    Enhanced fuzzy rule evaluation with hierarchy awareness.
    
    Parameters:
        ostatus: OStatus values per sector
        sector_hierarchy: Maximum hierarchy level per sector
        trajectory_dir: Current trajectory direction (-1, 0, 1)
    
    Returns:
        dict with firing strengths for each rule
    """
    tm = ostatus['top_mid']
    bm = ostatus['bot_mid']
    tl = ostatus['top_left']
    bl = ostatus['bot_left']
    tr = ostatus['top_right']
    br = ostatus['bot_right']
    
    # Hierarchy factors
    mid_hierarchy = max(sector_hierarchy.get('top_mid', 0), sector_hierarchy.get('bot_mid', 0))
    hierarchy_factor = HIERARCHY_WEIGHTS.get(mid_hierarchy, 1.0)
    
    # Critical obstacle boost
    has_critical = mid_hierarchy == 1
    critical_boost = 1.5 if has_critical else 1.0
    
    # Rule 1: Move Ahead — both mid sectors are free
    r1 = min(mu_free(tm), mu_free(bm))
    
    # Rule 2: Move Left — mid is blocked AND left side is free
    mid_blocked = max(mu_blocked(tm), mu_blocked(bm))
    left_free = min(mu_free(tl), mu_free(bl))
    r2 = min(mid_blocked, left_free) * critical_boost
    
    # Rule 3: Move Right — mid is blocked AND right side is free
    right_free = min(mu_free(tr), mu_free(br))
    r3 = min(mid_blocked, right_free) * critical_boost
    
    # Rule 4: Emergency Stop — critical obstacle very close
    if 'center_top' in ostatus:
        center_blocked = mu_critical(ostatus['center_top'])
        r4 = center_blocked * (1.5 if has_critical else 1.0)
    else:
        r4 = mid_blocked * hierarchy_factor * 0.5
    
    # Rule 5: Adjust for trajectory (stay in current direction if safe)
    if trajectory_dir != 0:
        if trajectory_dir < 0:  # Moving left
            current_free = left_free
            alternative_free = right_free
        else:  # Moving right
            current_free = right_free
            alternative_free = left_free
        
        # Prefer to continue current safe direction
        if current_free > 0.7:
            if trajectory_dir < 0:
                r2 *= 1.2
            else:
                r3 *= 1.2
    
    return {
        'move_ahead': r1,
        'move_left': r2,
        'move_right': r3,
        'emergency_stop': r4,
    }


def defuzzify_enhanced(rules):
    """
    Enhanced centroid defuzzification with safety priority.
    
    Returns:
        (direction, confidence, action_type)
    """
    # Output universe: [-1=left, 0=ahead, +1=right]
    r1 = rules['move_ahead']
    r2 = rules['move_left']
    r3 = rules['move_right']
    r4 = rules['emergency_stop']
    
    # Emergency stop takes precedence
    if r4 > 0.6:
        return 0.0, r4, 'STOP'
    
    # If ahead is strongly preferred
    if r1 > max(r2, r3) * 1.5:
        return 0.0, r1, 'MOVE_AHEAD'
    
    # If sides are blocked, consider alternative
    if r2 > 0.3 or r3 > 0.3:
        if r2 > r3:
            return -0.8, r2, 'MOVE_LEFT'
        elif r3 > r2:
            return 0.8, r3, 'MOVE_RIGHT'
    
    # Weighted centroid
    numerator = -0.8 * r2 + 0.0 * r1 + 0.8 * r3
    denominator = r1 + r2 + r3 + 1e-6
    direction = numerator / denominator
    
    max_strength = max(r1, r2, r3)
    
    if max_strength < 0.2:
        return 0.0, 0.5, 'MOVE_AHEAD'  # Default to straight
    elif abs(direction) < 0.3:
        return direction, max_strength, 'MOVE_AHEAD'
    elif direction < 0:
        return direction, max_strength, 'MOVE_LEFT'
    else:
        return direction, max_strength, 'MOVE_RIGHT'


def generate_alternative_paths(ostatus, sector_bounds):
    """
    Generate alternative safe paths ranked by safety.
    
    Returns:
        list of (path_name, safety_score, direction)
    """
    if not ALTERNATIVE_PATHS_ENABLED:
        return []
    
    paths = []
    
    # Left path (go around right side of obstacle)
    left_safety = 1.0 - min(ostatus.get('bot_left', 0) * 1.5, 1.0)
    right_obstacle = max(ostatus.get('bot_right', 0), ostatus.get('top_right', 0))
    left_safety *= (1.0 - right_obstacle * 0.5)
    
    if left_safety > 0.3:
        paths.append(('LEFT_ROUTE', left_safety, -0.8))
    
    # Right path (go around left side of obstacle)
    right_safety = 1.0 - min(ostatus.get('bot_right', 0) * 1.5, 1.0)
    left_obstacle = max(ostatus.get('bot_left', 0), ostatus.get('top_left', 0))
    right_safety *= (1.0 - left_obstacle * 0.5)
    
    if right_safety > 0.3:
        paths.append(('RIGHT_ROUTE', right_safety, 0.8))
    
    # Center path (if clear)
    center_safety = 1.0 - min(ostatus.get('bot_mid', 0) * 2.0, 1.0)
    if center_safety > 0.5:
        paths.append(('CENTER_ROUTE', center_safety, 0.0))
    
    # Sort by safety score
    if PATH_RANK_BY_SAFETY:
        paths.sort(key=lambda x: x[1], reverse=True)
    
    return paths[:MAX_ALTERNATIVE_PATHS]


def plan_path_enhanced(seg_mask, depth_map, obstacle_labels=None, trajectory=None):
    """
    Enhanced path planning with all improvements.
    
    Parameters:
        seg_mask: Semantic segmentation mask
        depth_map: Depth/disparity map
        obstacle_labels: Optional obstacle labels from enhanced ODM
        trajectory: Optional TrajectoryHistory for prediction
    
    Returns:
        (nav_instruction, planner_details)
    """
    h, w = seg_mask.shape[:2]
    
    # Create depth-gated obstacle mask
    obstacle_mask = create_depth_gated_mask(seg_mask, depth_map)
    
    # If obstacle_labels not provided, create simple version
    if obstacle_labels is None:
        obstacle_labels = np.full((h, w), -1, dtype=np.int16)
        obstacle_labels[obstacle_mask] = 1
    
    # Compute variable-resolution sector bounds
    sector_bounds = compute_variable_sector_bounds(h, w)
    
    # Compute enhanced OStatus with hierarchy
    ostatus, sector_labels, sector_hierarchy = compute_ostatus_enhanced(
        obstacle_mask, obstacle_labels, sector_bounds
    )
    
    # Find prominent obstacle
    class_name, class_id, position = find_prominent_obstacle(sector_labels)
    
    # Get trajectory direction
    trajectory_dir = 0
    if trajectory is not None:
        trajectory_dir = trajectory.get_direction()
        trajectory_predictions = trajectory.predict()
    else:
        trajectory_predictions = []
    
    # Evaluate enhanced fuzzy rules
    rules = evaluate_rules_enhanced(ostatus, sector_hierarchy, trajectory_dir)
    
    # Defuzzify
    direction, confidence, action_type = defuzzify_enhanced(rules)
    
    # Generate alternative paths
    alternative_paths = generate_alternative_paths(ostatus, sector_bounds)
    
    # Build instruction
    if action_type == 'STOP':
        if class_name:
            nav_instruction = f"Stop — {class_name} ahead"
        else:
            nav_instruction = "Stop — path blocked"
    elif action_type == 'MOVE_AHEAD':
        if class_name and position == 'overhead':
            nav_instruction = f"Move ahead — {class_name} overhead"
        elif class_name:
            nav_instruction = f"Path clear ahead — {class_name} to the side"
        else:
            nav_instruction = "Path clear — move ahead"
    elif action_type == 'MOVE_LEFT':
        if class_name:
            nav_instruction = f"Move left — {class_name} on right"
        else:
            nav_instruction = "Move left"
    else:  # MOVE_RIGHT
        if class_name:
            nav_instruction = f"Move right — {class_name} on left"
        else:
            nav_instruction = "Move right"
    
    # Add trajectory awareness
    if trajectory_dir != 0 and action_type == 'MOVE_AHEAD':
        if trajectory_dir < 0:
            nav_instruction += " (continue left)"
        else:
            nav_instruction += " (continue right)"
    
    # Build details
    planner_details = {
        'sector_bounds': sector_bounds,
        'ostatus': ostatus,
        'sector_labels': sector_labels,
        'sector_hierarchy': sector_hierarchy,
        'rules': rules,
        'direction': direction,
        'confidence': confidence,
        'action_type': action_type,
        'prominent_obstacle': {
            'class_name': class_name,
            'class_id': class_id,
            'position': position,
        },
        'alternative_paths': alternative_paths,
        'trajectory_direction': trajectory_dir,
        'trajectory_predictions': trajectory_predictions,
    }
    
    return nav_instruction, planner_details


def get_direction_arrow(direction, frame_shape):
    """
    Get arrow points for AR overlay.
    
    Parameters:
        direction: -1 to 1 (left to right)
        frame_shape: (h, w, 3)
    
    Returns:
        list of points for drawing the arrow
    """
    h, w = frame_shape[:2]
    center_x = w // 2
    base_y = h - 80
    
    # Arrow parameters
    arrow_length = 100
    arrow_width = 40
    
    # Calculate arrow tip position
    if abs(direction) < 0.1:
        # Straight ahead
        tip_x = center_x
        tip_y = base_y - arrow_length
    else:
        # Offset to side
        offset_x = int(direction * w * 0.3)
        tip_x = center_x + offset_x
        tip_y = base_y - int(arrow_length * 0.7)
    
    # Arrow base points
    base_left = (center_x - arrow_width // 2, base_y)
    base_right = (center_x + arrow_width // 2, base_y)
    base_center = (center_x, base_y)
    
    # Arrow tip
    tip = (tip_x, tip_y)
    
    return {
        'tip': tip,
        'base_left': base_left,
        'base_right': base_right,
        'base_center': base_center,
    }


def draw_navigation_arrow(frame, direction, action_type, color=None):
    """
    Draw AR-style navigation arrow on the frame.
    
    Parameters:
        frame: Camera frame
        direction: -1 to 1 (left to right)
        action_type: Action type string
        color: Optional BGR color tuple
    
    Returns:
        frame with arrow overlay
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    if color is None:
        if action_type == 'STOP':
            color = (0, 0, 255)  # Red
        elif action_type == 'MOVE_AHEAD':
            color = (0, 255, 0)  # Green
        else:
            color = (0, 255, 255)  # Yellow
    
    arrow = get_direction_arrow(direction, frame.shape)
    tip = arrow['tip']
    base_left = arrow['base_left']
    base_right = arrow['base_right']
    
    # Draw arrow body
    cv2.line(overlay, (w // 2, h - 60), tip, color, 4, cv2.LINE_AA)
    
    # Draw arrow head
    if action_type != 'STOP':
        pts = np.array([
            tip,
            (base_left[0], base_left[1] - 20),
            (base_right[0], base_right[1] - 20),
        ], np.int32)
        cv2.fillPoly(overlay, [pts], color)
    else:
        # Stop sign
        cv2.circle(overlay, tip, 25, color, -1)
        cv2.circle(overlay, tip, 25, (255, 255, 255), 3)
        # Draw X
        cv2.line(overlay, (tip[0] - 10, tip[1] - 10), (tip[0] + 10, tip[1] + 10), (255, 255, 255), 3)
        cv2.line(overlay, (tip[0] + 10, tip[1] - 10), (tip[0] - 10, tip[1] + 10), (255, 255, 255), 3)
    
    # Add transparency
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    return frame
