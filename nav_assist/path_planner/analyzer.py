"""
Obstacle analysis: depth-gated masking and per-zone occupancy.

Creates a strict obstacle mask using both semantic segmentation
(non-walkable classes) and depth proximity (~3m gate), then
computes obstacle pixel density per zone.
"""

import cv2
import numpy as np

from nav_assist.config import PATH_CLASS_INDICES, ADE20K_CLASSES

# Fraction of max disparity for the ~3m proximity gate.
# Pixels below this ratio are considered too far and discarded.
DEPTH_3M_RATIO = 0.40


def create_depth_gated_mask(seg_mask, depth_map, depth_ratio=DEPTH_3M_RATIO):
    """
    Create obstacle mask: semantic AND depth proximity.

    Only pixels that (a) belong to a non-walkable class AND
    (b) have disparity >= depth_ratio * max_disparity are obstacles.

    Returns
    -------
    obstacle_mask : (H, W) bool
    obstacle_labels : (H, W) int16 — class ID per obstacle pixel, -1 elsewhere
    """
    h, w = seg_mask.shape[:2]

    if depth_map.shape[:2] != (h, w):
        depth = cv2.resize(depth_map.astype(np.float32), (w, h),
                           interpolation=cv2.INTER_LINEAR)
    else:
        depth = depth_map.astype(np.float32)

    max_d = depth.max()
    if max_d < 1e-6:
        return (np.zeros((h, w), dtype=bool),
                np.full((h, w), -1, dtype=np.int16))

    near_mask = depth >= (depth_ratio * max_d)
    path_mask = np.isin(seg_mask, list(PATH_CLASS_INDICES))
    obstacle_mask = near_mask & ~path_mask

    obstacle_labels = np.full((h, w), -1, dtype=np.int16)
    obstacle_labels[obstacle_mask] = seg_mask[obstacle_mask].astype(np.int16)

    return obstacle_mask, obstacle_labels


def compute_zone_occupancy(obstacle_mask, obstacle_labels, zones):
    """
    Compute obstacle pixel density (OStatus) and semantic labels per zone.

    Parameters
    ----------
    obstacle_mask : (H, W) bool
    obstacle_labels : (H, W) int16
    zones : dict from compute_zones()

    Returns
    -------
    occupancy : dict zone_name -> float (0..1)
    zone_labels : dict zone_name -> {class_id: pixel_count}
    """
    occupancy = {}
    zone_labels = {}

    for name, (y0, y1, x0, x1) in zones.items():
        region = obstacle_mask[y0:y1, x0:x1]
        total = region.size
        occupancy[name] = int(region.sum()) / total if total > 0 else 0.0

        region_lbl = obstacle_labels[y0:y1, x0:x1]
        obs_vals = region_lbl[region]
        labels_dict = {}
        if len(obs_vals) > 0:
            unique, counts = np.unique(obs_vals, return_counts=True)
            for cls_id, cnt in zip(unique, counts):
                if cls_id >= 0:
                    labels_dict[int(cls_id)] = int(cnt)
        zone_labels[name] = labels_dict

    return occupancy, zone_labels


def find_prominent_obstacle(zone_labels):
    """
    Find the dominant obstacle class in the center zones
    (ground_center + overhead_center).

    Returns
    -------
    class_name : str or None
    class_id : int or -1
    position : 'ahead' (ground) or 'overhead'
    """
    center_counts = {}
    for zone_name in ('ground_center', 'overhead_center'):
        for cls_id, count in zone_labels.get(zone_name, {}).items():
            center_counts[cls_id] = center_counts.get(cls_id, 0) + count

    if not center_counts:
        return None, -1, 'ahead'

    prominent_id = max(center_counts, key=center_counts.get)
    class_name = (ADE20K_CLASSES[prominent_id]
                  if prominent_id < len(ADE20K_CLASSES) else 'unknown')

    ground_count = zone_labels.get('ground_center', {}).get(prominent_id, 0)
    overhead_count = zone_labels.get('overhead_center', {}).get(prominent_id, 0)
    position = 'overhead' if overhead_count > ground_count else 'ahead'

    return class_name, prominent_id, position
