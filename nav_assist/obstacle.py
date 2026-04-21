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
    min_area : int, optional
        Minimum pixel area for a connected component to be considered.

    Returns
    -------
    obstacle_bgr : np.ndarray, shape (H, W, 3), dtype uint8
        BGR image with obstacles coloured by their semantic class.
    obstacle_mask : np.ndarray, shape (H, W), dtype bool
        Binary mask — True where obstacles are detected.
    obstacle_info : list[dict]
        Per-obstacle metadata: class_id, class_name, disparity, bbox, area.
    obstacle_labels : np.ndarray, shape (H, W), dtype int16
        Per-pixel semantic class ID for obstacle pixels; -1 for non-obstacle.
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
        return (np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w), dtype=bool),
                [],
                np.full((h, w), -1, dtype=np.int16))

    threshold_d = threshold_ratio * max_disparity

    # Output arrays
    obstacle_mask = np.zeros((h, w), dtype=bool)
    obstacle_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    obstacle_labels = np.full((h, w), -1, dtype=np.int16)
    obstacle_info = []

    unique_labels = np.unique(seg_mask)

    for class_id in unique_labels:
        class_id = int(class_id)

        if class_id in PATH_CLASS_INDICES:
            continue

        class_mask = (seg_mask == class_id).astype(np.uint8)
        labelled = measure.label(class_mask, connectivity=2)
        regions = measure.regionprops(labelled)

        for region in regions:
            if region.area < min_area:
                continue

            component_mask = (labelled == region.label)
            component_disparity = depth_resized[component_mask].max()

            if component_disparity <= threshold_d:
                continue

            obstacle_mask[component_mask] = True
            obstacle_labels[component_mask] = class_id

            if class_id < len(ADE20K_PALETTE):
                rgb = ADE20K_PALETTE[class_id]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            else:
                bgr = (255, 255, 255)
            obstacle_bgr[component_mask] = bgr

            min_row, min_col, max_row, max_col = region.bbox
            class_name = (ADE20K_CLASSES[class_id]
                          if class_id < len(ADE20K_CLASSES) else 'unknown')
            obstacle_info.append({
                'class_id': class_id,
                'class_name': class_name,
                'disparity': float(component_disparity),
                'bbox': (min_col, min_row, max_col, max_row),
                'area': int(region.area),
                'centroid': (int(region.centroid[1]), int(region.centroid[0])),
            })

    obstacle_info.sort(key=lambda o: o['disparity'], reverse=True)

    return obstacle_bgr, obstacle_mask, obstacle_info, obstacle_labels
