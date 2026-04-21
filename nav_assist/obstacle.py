"""
Obstacle Detection Module (ODM) — Algorithm 1 from the Obs-tackle paper.

Fuses depth estimation and semantic segmentation to identify nearby obstacles.

Algorithm:
    For each semantic label in the segmentation image:
        For each connected component of that label:
            Compute max disparity (depth) value within the component
            If max_disparity <= threshold  OR  segment is a path class:
                Discard (not a nearby obstacle)
            Else:
                Add to obstacle image

The disparity threshold is set at 60% of the maximum disparity value
in the entire depth image (per the paper's experimental findings).
"""

import cv2
import numpy as np
from skimage import measure

from nav_assist.config import (
    ADE20K_PALETTE, ADE20K_CLASSES, PATH_CLASS_INDICES,
    DISPARITY_THRESHOLD_RATIO, MIN_COMPONENT_AREA,
)


def detect_obstacles(seg_mask, depth_map, threshold_ratio=None,
                     min_area=None):
    """
    Core ODM function: fuse segmentation + depth to produce obstacle image.

    Parameters
    ----------
    seg_mask : np.ndarray, shape (H, W), dtype uint8
        Semantic segmentation mask (class indices 0..149).
    depth_map : np.ndarray, shape (H, W), dtype float32
        Depth/disparity map (higher values = closer objects).
    threshold_ratio : float, optional
        Fraction of max disparity used as the nearness threshold.
        Defaults to DISPARITY_THRESHOLD_RATIO (0.60).
    min_area : int, optional
        Minimum pixel area for a connected component to be considered.
        Defaults to MIN_COMPONENT_AREA.

    Returns
    -------
    obstacle_bgr : np.ndarray, shape (H, W, 3), dtype uint8
        BGR image with obstacles coloured by their semantic class,
        background is black.
    obstacle_mask : np.ndarray, shape (H, W), dtype bool
        Binary mask — True where obstacles are detected.
    obstacle_info : list[dict]
        Per-obstacle metadata: class_id, class_name, disparity, bbox, area.
    """
    if threshold_ratio is None:
        threshold_ratio = DISPARITY_THRESHOLD_RATIO
    if min_area is None:
        min_area = MIN_COMPONENT_AREA

    h, w = seg_mask.shape[:2]

    # Resize depth to match segmentation if needed
    if depth_map.shape[:2] != (h, w):
        depth_resized = cv2.resize(depth_map.astype(np.float32), (w, h),
                                   interpolation=cv2.INTER_LINEAR)
    else:
        depth_resized = depth_map.astype(np.float32)

    # Compute disparity threshold: 60% of max disparity value
    max_disparity = depth_resized.max()
    if max_disparity < 1e-6:
        # No meaningful depth data — return empty
        return (np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w), dtype=bool),
                [])

    threshold_d = threshold_ratio * max_disparity

    # Output arrays
    obstacle_mask = np.zeros((h, w), dtype=bool)
    obstacle_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    obstacle_info = []

    # Get unique semantic labels present in the image
    unique_labels = np.unique(seg_mask)

    for class_id in unique_labels:
        class_id = int(class_id)

        # Skip path/walkable classes per Algorithm 1
        if class_id in PATH_CLASS_INDICES:
            continue

        # Binary mask for this semantic class
        class_mask = (seg_mask == class_id).astype(np.uint8)

        # Find connected components within this class
        labelled = measure.label(class_mask, connectivity=2)
        regions = measure.regionprops(labelled)

        for region in regions:
            # Skip tiny components (noise)
            if region.area < min_area:
                continue

            # Create mask for this specific component
            component_mask = (labelled == region.label)

            # Compute max disparity within this component
            component_disparity = depth_resized[component_mask].max()

            # Algorithm 1: discard if disparity <= threshold (too far away)
            if component_disparity <= threshold_d:
                continue

            # This component is a nearby obstacle — add to output
            obstacle_mask[component_mask] = True

            # Colour by semantic class
            if class_id < len(ADE20K_PALETTE):
                rgb = ADE20K_PALETTE[class_id]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            else:
                bgr = (255, 255, 255)
            obstacle_bgr[component_mask] = bgr

            # Collect metadata
            min_row, min_col, max_row, max_col = region.bbox
            class_name = (ADE20K_CLASSES[class_id]
                          if class_id < len(ADE20K_CLASSES) else 'unknown')
            obstacle_info.append({
                'class_id': class_id,
                'class_name': class_name,
                'disparity': float(component_disparity),
                'bbox': (min_col, min_row, max_col, max_row),  # x1,y1,x2,y2
                'area': int(region.area),
                'centroid': (int(region.centroid[1]), int(region.centroid[0])),
            })

    # Sort obstacles by disparity (nearest first)
    obstacle_info.sort(key=lambda o: o['disparity'], reverse=True)

    return obstacle_bgr, obstacle_mask, obstacle_info


def get_zone_obstacles(obstacle_mask, obstacle_info, image_width):
    """
    Split the scene into LEFT / CENTRE / RIGHT zones and compute
    obstacle density per zone.

    Returns
    -------
    zones : dict with keys 'left', 'center', 'right', each containing:
        - 'density': fraction of zone pixels that are obstacles (0..1)
        - 'obstacles': list of obstacle_info entries in that zone
        - 'max_disparity': max disparity of obstacles in the zone
    """
    col_w = image_width // 3
    boundaries = {
        'left':   (0, col_w),
        'center': (col_w, 2 * col_w),
        'right':  (2 * col_w, image_width),
    }

    zones = {}
    for zone_name, (x_start, x_end) in boundaries.items():
        zone_mask = obstacle_mask[:, x_start:x_end]
        zone_pixels = zone_mask.size
        obstacle_pixels = zone_mask.sum()
        density = obstacle_pixels / zone_pixels if zone_pixels > 0 else 0.0

        # Filter obstacles whose centroid falls in this zone
        zone_obs = [o for o in obstacle_info
                    if x_start <= o['centroid'][0] < x_end]
        max_disp = max((o['disparity'] for o in zone_obs), default=0.0)

        zones[zone_name] = {
            'density': density,
            'obstacles': zone_obs,
            'max_disparity': max_disp,
        }

    return zones
