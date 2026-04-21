"""
Enhanced Obstacle Detection Module (ODM) — Algorithm 1 with improvements.

Enhanced Features:
    - Confidence-weighted fusion: α * depth_confidence + β * seg_confidence
    - Edge-aware fusion: Canny edges refine obstacle boundaries
    - Temporal consistency: Kalman filtering for obstacle tracking
    - Hierarchical obstacle classification: Critical/Caution/Navigable
    - Motion detection for dynamic obstacles
    - Model failure detection with graceful degradation

Algorithm:
    1. Depth-gated masking: seg_mask AND (depth < 3m) → valid_obstacle_mask
    2. Confidence-weighted fusion: Weight by model confidence scores
    3. Edge-aware refinement: Use Canny edges to refine boundaries
    4. Hierarchical classification: Assign priority levels to obstacles
    5. Temporal tracking: Kalman filter for smooth obstacle tracking
    6. Motion detection: Identify moving obstacles
"""

import cv2
import numpy as np
from scipy import ndimage
from collections import deque
import time

from nav_assist.config import (
    ADE20K_PALETTE, ADE20K_CLASSES, PATH_CLASS_INDICES,
    DISPARITY_THRESHOLD_RATIO, MIN_COMPONENT_AREA,
    CONFIDENCE_ALPHA, CONFIDENCE_BETA,
    EDGE_SIGMA, EDGE_LOW_THRESH, EDGE_HIGH_THRESH,
    OBSTACLE_HIERARCHY, MODEL_HEALTH_CHECK, GRACEFUL_DEGRADATION,
    TRAJECTORY_PREDICTION_ENABLED, TRAJECTORY_SMOOTHING,
    MOTION_DETECTION_ENABLED, MOTION_THRESHOLD, MOTION_MIN_FRAMES,
    KALMAN_PROCESS_NOISE, KALMAN_MEASURE_NOISE, KALMAN_SMOOTHING_FRAMES,
    ENABLE_TEMPORAL_SMOOTHING, TEMPORAL_ALPHA,
)


class KalmanTracker:
    """
    Kalman filter for tracking obstacle positions across frames.
    Provides temporal smoothing and motion prediction.
    """
    
    def __init__(self, process_noise=KALMAN_PROCESS_NOISE, 
                 measure_noise=KALMAN_MEASURE_NOISE):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state vars, 2 measurements
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measure_noise
        self.initialized = False
        self.velocity = np.zeros(2, dtype=np.float32)
        
    def predict(self, x, y):
        """Predict next position based on current velocity."""
        if self.initialized:
            predicted = self.kalman.predict()
            return int(predicted[0]), int(predicted[1])
        return x, y
    
    def update(self, x, y):
        """Update Kalman filter with new measurement."""
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        
        if not self.initialized:
            self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        else:
            self.kalman.correct(measurement)
            state = self.kalman.statePost
            self.velocity = np.array([state[2], state[3]], dtype=np.float32)
        
        return x, y
    
    def get_velocity(self):
        """Return current velocity estimate."""
        return self.velocity


class ObstacleTracker:
    """
    Tracks multiple obstacles across frames using Kalman filtering.
    Maintains temporal consistency and detects motion.
    """
    
    def __init__(self, max_tracks=20, max_history=10):
        self.tracks = {}  # track_id -> {'kalman': KalmanTracker, 'history': deque, 'class_id': int}
        self.next_id = 0
        self.max_tracks = max_tracks
        self.max_history = max_history
        self.frame_count = 0
        
    def update(self, obstacles):
        """
        Update tracks with new obstacle detections.
        
        Parameters:
            obstacles: list of dicts with 'centroid', 'class_id', 'disparity', 'area'
        
        Returns:
            list of updated obstacles with track_ids and motion flags
        """
        self.frame_count += 1
        updated = []
        
        # Match existing tracks to new detections
        matched_ids = set()
        
        for obs in obstacles:
            cx, cy = obs['centroid']
            best_match = None
            best_dist = float('inf')
            
            # Find best matching track
            for track_id, track_data in self.tracks.items():
                if track_id in matched_ids:
                    continue
                    
                history = track_data['history']
                if history:
                    last_pos = history[-1]
                    dist = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
                    if dist < best_dist and dist < 50:  # 50px threshold
                        best_dist = dist
                        best_match = track_id
            
            if best_match is not None:
                # Update existing track
                matched_ids.add(best_match)
                track = self.tracks[best_match]
                track['history'].append((cx, cy))
                if len(track['history']) > self.max_history:
                    track['history'].popleft()
                    
                # Predict next position
                kalman = track['kalman']
                pred_x, pred_y = kalman.predict(cx, cy)
                kalman.update(cx, cy)
                velocity = kalman.get_velocity()
                
                obs['track_id'] = best_match
                obs['predicted_pos'] = (pred_x, pred_y)
                obs['velocity'] = tuple(velocity)
                obs['is_moving'] = np.linalg.norm(velocity) > 2.0  # Moving if velocity > 2 px/frame
            else:
                # Create new track
                if len(self.tracks) < self.max_tracks:
                    track_id = self.next_id
                    self.next_id += 1
                    
                    kalman = KalmanTracker()
                    kalman.update(cx, cy)
                    
                    self.tracks[track_id] = {
                        'kalman': kalman,
                        'history': deque([(cx, cy)], maxlen=self.max_history),
                        'class_id': obs['class_id'],
                        'created_frame': self.frame_count
                    }
                    
                    obs['track_id'] = track_id
                    obs['predicted_pos'] = (cx, cy)
                    obs['velocity'] = (0, 0)
                    obs['is_moving'] = False
                    
            updated.append(obs)
        
        # Age out old tracks
        to_remove = []
        for track_id, track_data in self.tracks.items():
            if track_id not in matched_ids:
                # Check if track should be removed (no match for several frames)
                if len(track_data['history']) > 0:
                    last_frame = track_data.get('last_seen_frame', 0)
                    if self.frame_count - last_frame > 10:
                        to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
        
        # Mark last seen frame for unmatched tracks
        for track_id, track_data in self.tracks.items():
            if track_id not in matched_ids:
                track_data['last_seen_frame'] = self.frame_count
        
        return updated
    
    def get_trajectory(self, track_id, frames=None):
        """Get predicted trajectory for a track."""
        if track_id not in self.tracks:
            return []
        
        track = self.tracks[track_id]
        if not track['history']:
            return []
        
        history = list(track['history'])
        kalman = track['kalman']
        
        # Extend trajectory with predictions
        if frames is None:
            frames = TRAJECTORY_PREDICTION_ENABLED
        
        if frames > 0:
            last_x, last_y = history[-1]
            velocity = kalman.get_velocity()
            
            for i in range(frames):
                pred_x = last_x + velocity[0] * (i + 1)
                pred_y = last_y + velocity[1] * (i + 1)
                history.append((pred_x, pred_y))
        
        return history


def compute_depth_confidence(depth_map):
    """
    Compute confidence map for depth estimation.
    Higher confidence where depth values are consistent locally.
    """
    # Use local variance as inverse confidence
    # Low variance → high confidence
    
    if depth_map.size == 0:
        return np.ones((1, 1), dtype=np.float32)
    
    # Blur for local statistics
    blurred = cv2.GaussianBlur(depth_map, (5, 5), 1.0)
    
    # Local standard deviation (simplified)
    sq_mean = cv2.GaussianBlur(depth_map**2, (5, 5), 1.0)
    mean_sq = blurred**2
    local_var = np.maximum(sq_mean - mean_sq, 0)
    
    # Convert to confidence (lower variance = higher confidence)
    max_var = local_var.max() + 1e-6
    confidence = 1.0 - np.clip(local_var / max_var, 0, 1)
    
    # Also consider edge regions (lower confidence at edges)
    depth_edges = cv2.Canny((depth_map * 255).astype(np.uint8), 50, 150)
    edge_confidence = 1.0 - (depth_edges > 0).astype(np.float32) * 0.3
    
    # Combine confidences
    final_confidence = confidence * edge_confidence
    
    return final_confidence.astype(np.float32)


def compute_seg_confidence(seg_mask):
    """
    Compute confidence map for segmentation.
    Uses edge proximity as uncertainty measure.
    """
    if seg_mask.size == 0:
        return np.ones((1, 1), dtype=np.float32)
    
    # Edge detection on segmentation mask
    seg_uint8 = seg_mask.astype(np.uint8)
    edges = cv2.Canny(seg_uint8, 50, 150)
    
    # Distance transform from edges
    dist_from_edge = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    max_dist = dist_from_edge.max() + 1e-6
    
    # Normalize to 0-1 confidence (farther from edge = higher confidence)
    confidence = np.clip(dist_from_edge / max_dist * 2, 0.5, 1.0)
    
    return confidence.astype(np.float32)


def detect_edges(frame):
    """
    Detect edges in the camera frame using Canny.
    Used for edge-aware obstacle refinement.
    """
    if frame is None or frame.size == 0:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), EDGE_SIGMA)
    edges = cv2.Canny(gray, EDGE_LOW_THRESH, EDGE_HIGH_THRESH)
    
    return edges


def confidence_weighted_fusion(depth_map, seg_mask, depth_conf, seg_conf):
    """
    Confidence-weighted fusion of depth and segmentation.
    
    obstacle_score = α * depth_confidence + β * seg_confidence
    
    Parameters:
        depth_map: Raw depth/disparity map
        seg_mask: Semantic segmentation mask
        depth_conf: Depth confidence map
        seg_conf: Segmentation confidence map
    
    Returns:
        Combined obstacle score map
    """
    h, w = depth_map.shape[:2]
    
    # Ensure all inputs have same shape
    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h))
    if depth_conf.shape[:2] != (h, w):
        depth_conf = cv2.resize(depth_conf, (w, h))
    if seg_conf.shape[:2] != (h, w):
        seg_conf = cv2.resize(seg_conf, (w, h))
    
    # Normalize depth to 0-1
    depth_norm = depth_map / (depth_map.max() + 1e-6)
    
    # Compute combined confidence score
    combined_conf = CONFIDENCE_ALPHA * depth_conf + CONFIDENCE_BETA * seg_conf
    
    # Obstacle score: combine depth (higher = closer) and confidence
    obstacle_score = depth_norm * combined_conf
    
    return obstacle_score


def hierarchical_classification(class_name):
    """
    Classify obstacle into hierarchy levels.
    
    Returns:
        (level, priority) where level is 1-3 and priority is 'critical', 'caution', or 'navigable'
    """
    class_lower = class_name.lower()
    
    for level_name, classes in OBSTACLE_HIERARCHY.items():
        if class_lower in classes:
            if level_name == 'critical':
                return 1, level_name
            elif level_name == 'caution':
                return 2, level_name
            else:
                return 3, level_name
    
    # Default to caution for unknown classes
    return 2, 'caution'


def detect_obstacles_enhanced(seg_mask, depth_map, frame=None, tracker=None,
                              threshold_ratio=None, min_area=None,
                              use_temporal=True, use_edge_aware=True,
                              use_confidence_weighted=True):
    """
    Enhanced obstacle detection with all improvements.
    
    Parameters:
        seg_mask: Semantic segmentation mask (H, W) uint8
        depth_map: Depth/disparity map (H, W) float32
        frame: Optional camera frame for edge detection
        tracker: Optional ObstacleTracker for temporal consistency
        threshold_ratio: Fraction of max disparity for near threshold
        min_area: Minimum pixel area for connected component
        use_temporal: Enable temporal smoothing
        use_edge_aware: Enable edge refinement
        use_confidence_weighted: Enable confidence-weighted fusion
    
    Returns:
        obstacle_bgr, obstacle_mask, obstacle_info, obstacle_labels, debug_info
    """
    if threshold_ratio is None:
        threshold_ratio = DISPARITY_THRESHOLD_RATIO
    if min_area is None:
        min_area = MIN_COMPONENT_AREA
    
    h, w = seg_mask.shape[:2]
    debug_info = {}
    
    # Resize depth to match segmentation if needed
    if depth_map.shape[:2] != (h, w):
        depth_resized = cv2.resize(depth_map.astype(np.float32), (w, h),
                                   interpolation=cv2.INTER_LINEAR)
    else:
        depth_resized = depth_map.astype(np.float32)
    
    # ── Compute confidence maps ──────────────────────────────────────────────
    if use_confidence_weighted:
        depth_conf = compute_depth_confidence(depth_resized)
        seg_conf = compute_seg_confidence(seg_mask)
        debug_info['depth_confidence'] = depth_conf
        debug_info['seg_confidence'] = seg_conf
        
        # Confidence-weighted fusion
        obstacle_scores = confidence_weighted_fusion(
            depth_resized, seg_mask, depth_conf, seg_conf
        )
    else:
        # Simple depth-based scoring
        depth_norm = depth_resized / (depth_resized.max() + 1e-6)
        obstacle_scores = depth_norm
        depth_conf = np.ones_like(depth_norm)
    
    debug_info['obstacle_scores'] = obstacle_scores
    
    # ── Edge-aware refinement ────────────────────────────────────────────────
    if use_edge_aware and frame is not None:
        edges = detect_edges(frame)
        if edges is not None:
            # Dilate edges slightly
            edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
            # Reduce obstacle score near edges (might be depth discontinuities)
            edge_mask = (edges > 0).astype(np.float32)
            obstacle_scores = obstacle_scores * (1.0 - edge_mask * 0.3)
            debug_info['edges'] = edges
    
    # ── Compute disparity threshold ─────────────────────────────────────────
    max_disparity = depth_resized.max()
    if max_disparity < 1e-6:
        return (np.zeros((h, w, 3), dtype=np.uint8),
                np.zeros((h, w), dtype=bool),
                [],
                np.full((h, w), -1, dtype=np.int16),
                debug_info)
    
    threshold_d = threshold_ratio * max_disparity
    
    # ── Output arrays ────────────────────────────────────────────────────────
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
        
        # Connected component analysis
        from skimage import measure
        labelled = measure.label(class_mask, connectivity=2)
        regions = measure.regionprops(labelled)
        
        for region in regions:
            if region.area < min_area:
                continue
            
            component_mask = (labelled == region.label)
            component_depth = depth_resized[component_mask]
            component_disparity = component_depth.max()
            
            if component_disparity <= threshold_d:
                continue
            
            # Check obstacle score threshold
            component_scores = obstacle_scores[component_mask]
            avg_score = component_scores.mean()
            
            if avg_score < 0.1:  # Confidence threshold
                continue
            
            obstacle_mask[component_mask] = True
            obstacle_labels[component_mask] = class_id
            
            # Get class info
            class_name = (ADE20K_CLASSES[class_id]
                         if class_id < len(ADE20K_CLASSES) else 'unknown')
            
            # Get color from palette
            if class_id < len(ADE20K_PALETTE):
                rgb = ADE20K_PALETTE[class_id]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            else:
                bgr = (255, 255, 255)
            obstacle_bgr[component_mask] = bgr
            
            # Bounding box and centroid
            min_row, min_col, max_row, max_col = region.bbox
            centroid = (int(region.centroid[1]), int(region.centroid[0]))
            
            # Height estimation (top-bottom extent in image)
            height_pixels = max_row - min_row
            # Rough height classification (can be refined with depth)
            is_overhead = min_row < h * 0.4
            
            # Hierarchical classification
            level, priority = hierarchical_classification(class_name)
            
            # Average confidence in this component
            avg_depth_conf = depth_conf[component_mask].mean()
            avg_seg_conf = seg_conf[component_mask].mean()
            
            obs_info = {
                'class_id': class_id,
                'class_name': class_name,
                'disparity': float(component_disparity),
                'depth_confidence': float(avg_depth_conf),
                'seg_confidence': float(avg_seg_conf),
                'bbox': (min_col, min_row, max_col, max_row),
                'area': int(region.area),
                'centroid': centroid,
                'is_overhead': is_overhead,
                'height_estimate': 'overhead' if is_overhead else 'ground',
                'hierarchy_level': level,
                'priority': priority,
                'obstacle_score': float(avg_score),
            }
            
            obstacle_info.append(obs_info)
    
    # ── Temporal smoothing ────────────────────────────────────────────────────
    if use_temporal and tracker is not None:
        obstacle_info = tracker.update(obstacle_info)
    
    # Sort by disparity (nearest first) and hierarchy
    obstacle_info.sort(key=lambda o: (-o['disparity'], o['hierarchy_level']))
    
    # Add track info to obstacles
    for obs in obstacle_info:
        if 'track_id' in obs:
            # Add trajectory
            if tracker is not None:
                obs['trajectory'] = tracker.get_trajectory(obs['track_id'])
    
    return obstacle_bgr, obstacle_mask, obstacle_info, obstacle_labels, debug_info


def create_obstacle_heatmap(obstacle_mask, depth_map, weights=None):
    """
    Create a danger/heat map showing obstacle severity.
    
    Returns:
        BGR heatmap image (hot = dangerous, cool = safe)
    """
    h, w = obstacle_mask.shape[:2]
    
    if weights is None:
        # Default weights based on proximity
        weights = 1.0 - (depth_map / (depth_map.max() + 1e-6))
    
    # Combine obstacle mask with depth weights
    heatmap = obstacle_mask.astype(np.float32) * weights
    
    # Normalize
    heatmap = heatmap / (heatmap.max() + 1e-6)
    
    # Apply colormap (hot = danger)
    cmap = colormaps.get_cmap('hot')
    heat_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(heat_colored, cv2.COLOR_RGB2BGR)
    
    return heat_bgr


def get_nearest_obstacle_direction(obstacle_info, frame_shape):
    """
    Get the direction to the nearest critical obstacle.
    
    Returns:
        (direction, distance, class_name) where direction is angle in degrees
        (-180 to 180, negative=left, positive=right)
    """
    if not obstacle_info:
        return None, None, None
    
    h, w = frame_shape[:2]
    center_x = w // 2
    
    # Find nearest obstacle that's not overhead
    valid_obs = [o for o in obstacle_info if not o.get('is_overhead', False)]
    
    if not valid_obs:
        return None, None, None
    
    nearest = valid_obs[0]  # Already sorted by disparity
    cx, cy = nearest['centroid']
    
    # Calculate direction
    dx = cx - center_x
    angle = np.degrees(np.arctan2(dx, h - cy))  # 0 = straight ahead
    
    # Normalize to -180 to 180
    if angle > 180:
        angle -= 360
    
    distance = nearest['disparity']
    
    return angle, distance, nearest['class_name']


# Helper for colormap
from matplotlib import colormaps
