"""
Enhanced Visualization Module.

Enhanced Features:
    - Depth histogram: Show distance distribution in real-time
    - Trajectory visualization: Draw predicted walking path
    - Hazard heatmap: Gradient showing danger levels
    - AR-style overlay: Navigation arrows on camera feed
    - Confidence overlay: Model uncertainty visualization
    - System metrics panel: Real-time performance graphs
    - 8-panel display support
    - Session recording

Panel Layout (8-panel):
    ┌─────────┬─────────┬─────────┬─────────┐
    │  PANEL  │  PANEL  │  PANEL  │  PANEL  │
    │    1    │    2    │    3    │    4    │
    │ CAMERA  │  DEPTH  │   SEG   │  FUSION │
    │  FEED   │   MAP   │  MASK   │ HEATMAP │
    ├─────────┼─────────┼─────────┼─────────┤
    │  PANEL  │  PANEL  │  PANEL  │  PANEL  │
    │    5    │    6    │    7    │    8    │
    │OBSTACLE │  PATH   │CONFIDENCE│SYSTEM  │
    │   MAP   │ PLANNER │ OVERLAY │METRICS  │
    └─────────┴─────────┴─────────┴─────────┘
"""

import cv2
import numpy as np
from collections import deque
from matplotlib import colormaps
import time
import threading

from nav_assist.config import (
    ADE20K_CLASSES, ADE20K_PALETTE, PANEL_W, PANEL_H, STATUS_H, WIN_W,
    PPM_OSTATUS_THRESHOLD, INSTRUCTION_BAR_H,
    PANEL_8_W, PANEL_8_H, GRID_8_COLS, GRID_8_ROWS, WIN_8_W, WIN_8_H,
    SHOW_DEPTH_HISTOGRAM, SHOW_TRAJECTORY, SHOW_HAZARD_HEATMAP,
    SHOW_AR_OVERLAY, SHOW_CONFIDENCE_OVERLAY,
    ENABLE_PERF_STATS, PERF_HISTORY_SIZE, SHOW_LATENCY_BREAKDOWN,
    SHOW_MEMORY_USAGE, ENABLE_RECORDING, RECORD_FPS, RECORD_CODEC,
)


# Colormap for visualization
_CMAP_SPECTRAL = colormaps.get_cmap('Spectral_r')
_CMAP_HOT = colormaps.get_cmap('hot')
_CMAP_VIRIDIS = colormaps.get_cmap('viridis')


# ════════════════════════════════════════════════════════════════════════════
# Drawing Primitives
# ════════════════════════════════════════════════════════════════════════════

def put_text(img, text, pos, scale=0.55, color=(255, 255, 255),
             thickness=1, bg_color=(0, 0, 0)):
    """Draw text with shadow for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (pos[0] + 1, pos[1] + 1), font,
                scale, bg_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def put_label_box(img, text, x, y, color_bgr, font_scale=0.45):
    """Colored legend box + label."""
    cv2.rectangle(img, (x, y), (x + 14, y + 14), color_bgr, -1)
    cv2.rectangle(img, (x, y), (x + 14, y + 14), (50, 50, 50), 1)
    put_text(img, text, (x + 18, y + 11), scale=font_scale,
             color=(230, 230, 230), bg_color=(20, 20, 20))


# ════════════════════════════════════════════════════════════════════════════
# Colourisation Functions
# ════════════════════════════════════════════════════════════════════════════

def colorize_depth(depth_np, cmap='Spectral_r'):
    """Raw depth array → colored BGR image."""
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth_np)
    else:
        norm = (depth_np - d_min) / (d_max - d_min)
    
    cmap_obj = colormaps.get_cmap(cmap)
    coloured = (cmap_obj(norm)[:, :, :3] * 255).astype(np.uint8)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)
    return coloured, norm


def colorize_seg(seg_mask, h, w):
    """Segmentation mask → colored BGR image."""
    colour_map = ADE20K_PALETTE[seg_mask]
    colour_bgr = colour_map[..., ::-1].astype(np.uint8)
    colour_bgr = cv2.resize(colour_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    return colour_bgr


def colorize_heatmap(scores, cmap='hot'):
    """Score array → colored heatmap."""
    scores_norm = scores / (scores.max() + 1e-6)
    cmap_obj = colormaps.get_cmap(cmap)
    colored = (cmap_obj(scores_norm)[:, :, :3] * 255).astype(np.uint8)
    colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    return colored


# ════════════════════════════════════════════════════════════════════════════
# Depth Histogram
# ════════════════════════════════════════════════════════════════════════════

def draw_depth_histogram(depth_np, panel_w, panel_h, bins=30):
    """
    Draw depth distribution histogram.
    
    Returns:
        BGR image of histogram
    """
    h, w = panel_h - 80, panel_w - 20
    
    # Create histogram
    depth_flat = depth_np.flatten()
    depth_flat = depth_flat[depth_flat > 0.01]  # Remove zeros
    
    if len(depth_flat) == 0:
        hist_img = np.zeros((h, w, 3), dtype=np.uint8)
        put_text(hist_img, "No depth data", (w//2 - 50, h//2), scale=0.5)
        return hist_img
    
    # Compute histogram
    hist, bin_edges = np.histogram(depth_flat, bins=bins)
    hist = hist.astype(float)
    hist /= hist.max() + 1e-6
    
    # Draw histogram
    bin_w = w // bins
    hist_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(bins):
        bar_h = int(hist[i] * h * 0.8)
        x1 = i * bin_w
        x2 = x1 + bin_w - 1
        
        # Color gradient from blue (far) to red (near)
        color_val = int(255 * (1 - i / bins))
        color = (color_val, 100, 255 - color_val)
        
        cv2.rectangle(hist_img, (x1, h - bar_h), (x2, h), color, -1)
    
    # Add labels
    put_text(hist_img, "FAR", (5, h - 5), scale=0.35, color=(255, 100, 100))
    put_text(hist_img, "NEAR", (w - 40, h - 5), scale=0.35, color=(100, 100, 255))
    
    return hist_img


def build_depth_histogram_panel(depth_np, width, height):
    """Build depth histogram panel with stats."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'DEPTH HISTOGRAM', (8, 18), scale=0.45, color=(200, 200, 200))
    
    # Draw histogram
    hist_img = draw_depth_histogram(depth_np, width, height - 28)
    panel[28:, :] = hist_img
    
    # Add statistics
    depth_flat = depth_np.flatten()
    depth_flat = depth_flat[depth_flat > 0.01]
    
    if len(depth_flat) > 0:
        mean_d = depth_flat.mean()
        std_d = depth_flat.std()
        min_d = depth_flat.min()
        max_d = depth_flat.max()
        
        stats_y = height - 25
        put_text(panel, f'M:{mean_d:.2f} S:{std_d:.2f}', (8, stats_y),
                 scale=0.35, color=(180, 180, 180))
    
    return panel


# ════════════════════════════════════════════════════════════════════════════
# Trajectory Visualization
# ════════════════════════════════════════════════════════════════════════════

def draw_trajectory(frame, trajectory, predicted=None, color=(0, 255, 0), 
                     predicted_color=(255, 255, 0)):
    """
    Draw trajectory path on frame.
    
    Parameters:
        frame: Camera frame
        trajectory: List of (x, y) points for actual path
        predicted: List of (x, y) points for predicted path
        color: Color for actual trajectory
        predicted_color: Color for predicted trajectory
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw actual trajectory
    if len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
            pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
            cv2.line(overlay, pt1, pt2, color, 2, cv2.LINE_AA)
            
            # Draw dots at intervals
            if i % 5 == 0:
                cv2.circle(overlay, pt2, 3, color, -1)
    
    # Draw predicted trajectory
    if predicted and len(predicted) > 0:
        # Connect last actual to first predicted
        if len(trajectory) > 0:
            last_actual = (int(trajectory[-1][0]), int(trajectory[-1][1]))
            first_pred = (int(predicted[0][0]), int(predicted[0][1]))
            cv2.line(overlay, last_actual, first_pred, predicted_color, 2, cv2.LINE_AA)
        
        for i in range(1, len(predicted)):
            pt1 = (int(predicted[i-1][0]), int(predicted[i-1][1]))
            pt2 = (int(predicted[i][0]), int(predicted[i][1]))
            cv2.line(overlay, pt1, pt2, predicted_color, 2, cv2.LINE_AA)
            
            # Draw larger dots for prediction
            if i % 3 == 0:
                cv2.circle(overlay, pt2, 5, predicted_color, -1)
    
    # Add trajectory legend
    legend_y = h - 40
    cv2.circle(overlay, (20, legend_y), 5, color, -1)
    put_text(overlay, 'Path', (30, legend_y + 5), scale=0.4, color=color)
    
    cv2.circle(overlay, (100, legend_y), 5, predicted_color, -1)
    put_text(overlay, 'Predicted', (110, legend_y + 5), scale=0.4, color=predicted_color)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    return frame


# ════════════════════════════════════════════════════════════════════════════
# Hazard Heatmap
# ════════════════════════════════════════════════════════════════════════════

def create_hazard_heatmap(obstacle_mask, depth_map, obstacle_info=None):
    """
    Create danger/heat map showing obstacle severity.
    
    Returns:
        BGR heatmap image
    """
    h, w = obstacle_mask.shape[:2]
    
    # Normalize depth (higher = closer = more dangerous)
    depth_norm = depth_map / (depth_map.max() + 1e-6)
    
    # Start with depth as base
    heatmap = depth_norm.copy()
    
    # Add obstacle mask as boost
    if obstacle_mask.any():
        heatmap = np.maximum(heatmap, obstacle_mask.astype(float) * 0.5)
    
    # Boost areas with critical obstacles
    if obstacle_info:
        for obs in obstacle_info:
            if obs.get('hierarchy_level', 3) == 1:  # Critical
                cx, cy = obs['centroid']
                # Create gradient around critical obstacle
                yy, xx = np.mgrid[0:h, 0:w]
                dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
                gaussian = np.exp(-dist / 50.0)
                heatmap = np.maximum(heatmap, gaussian * 0.8)
    
    # Normalize
    heatmap = heatmap / (heatmap.max() + 1e-6)
    
    # Apply hot colormap (red = dangerous, blue = safe)
    cmap_obj = colormaps.get_cmap('hot')
    heat_colored = (cmap_obj(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(heat_colored, cv2.COLOR_RGB2BGR)
    
    return heat_bgr


def build_hazard_heatmap_panel(obstacle_mask, depth_map, obstacle_info, width, height):
    """Build hazard heatmap panel."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'HAZARD HEATMAP', (8, 18), scale=0.45, color=(200, 200, 200))
    
    # Create heatmap
    heatmap = create_hazard_heatmap(obstacle_mask, depth_map, obstacle_info)
    heatmap_small = cv2.resize(heatmap, (width, height - 28))
    
    panel[28:, :] = heatmap_small
    
    # Add legend
    legend_y = height - 20
    cv2.putText(panel, 'SAFE', (5, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(panel, 'DANGER', (width - 55, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (100, 100, 255), 1, cv2.LINE_AA)
    
    return panel


# ════════════════════════════════════════════════════════════════════════════
# AR Navigation Overlay
# ════════════════════════════════════════════════════════════════════════════

def draw_ar_navigation(frame, direction, action_type, nav_instruction,
                       obstacle_info=None, confidence=1.0):
    """
    Draw AR-style navigation overlay on camera feed.
    
    Parameters:
        frame: Camera frame
        direction: -1 (left) to 1 (right)
        action_type: Action type string
        nav_instruction: Full instruction text
        obstacle_info: List of obstacles for highlighting
        confidence: Navigation confidence (0-1)
    
    Returns:
        Frame with AR overlay
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Determine color based on action type
    if action_type == 'STOP':
        arrow_color = (0, 0, 255)  # Red
    elif action_type == 'MOVE_AHEAD':
        arrow_color = (0, 255, 0)  # Green
    else:
        arrow_color = (0, 255, 255)  # Yellow
    
    # Draw navigation arrow
    center_x = w // 2
    arrow_base_y = h - 100
    arrow_tip_y = arrow_base_y - 80
    arrow_tip_x = int(center_x + direction * w * 0.3)
    
    if action_type != 'STOP':
        # Arrow shaft
        cv2.arrowedLine(overlay, (center_x, arrow_base_y),
                        (arrow_tip_x, arrow_tip_y),
                        arrow_color, 4, tipLength=0.3)
        
        # Arrow head
        pts = np.array([
            [arrow_tip_x, arrow_tip_y],
            [arrow_tip_x - 15, arrow_tip_y + 30],
            [arrow_tip_x + 15, arrow_tip_y + 30],
        ], np.int32)
        cv2.fillPoly(overlay, [pts], arrow_color)
    else:
        # Stop sign
        cv2.circle(overlay, (arrow_tip_x, arrow_tip_y - 30), 30, arrow_color, -1)
        cv2.circle(overlay, (arrow_tip_x, arrow_tip_y - 30), 30, (255, 255, 255), 3)
        # Draw X
        cv2.line(overlay, (arrow_tip_x - 12, arrow_tip_y - 42),
                 (arrow_tip_x + 12, arrow_tip_y - 18), (255, 255, 255), 3)
        cv2.line(overlay, (arrow_tip_x + 12, arrow_tip_y - 42),
                 (arrow_tip_x - 12, arrow_tip_y - 18), (255, 255, 255), 3)
    
    # Highlight nearby obstacles with circles
    if obstacle_info:
        for obs in obstacle_info[:3]:  # Top 3
            cx, cy = obs['centroid']
            disparity = obs['disparity']
            
            # Scale circle size based on distance
            radius = int(30 * disparity)
            if obs.get('hierarchy_level', 3) == 1:
                color = (0, 0, 255)  # Red for critical
            elif obs.get('hierarchy_level', 3) == 2:
                color = (0, 165, 255)  # Orange for caution
            else:
                color = (0, 255, 255)  # Yellow for navigable
            
            cv2.circle(overlay, (cx, cy), radius, color, 2)
            
            # Label
            class_name = obs['class_name'][:10]
            put_text(overlay, class_name, (cx + radius + 5, cy),
                     scale=0.4, color=color)
    
    # Draw confidence indicator
    conf_bar_w = 100
    conf_bar_h = 10
    conf_x = w - conf_bar_w - 10
    conf_y = h - 30
    
    cv2.rectangle(overlay, (conf_x, conf_y), (conf_x + conf_bar_w, conf_y + conf_bar_h),
                  (50, 50, 50), -1)
    cv2.rectangle(overlay, (conf_x, conf_y),
                  (conf_x + int(conf_bar_w * confidence), conf_y + conf_bar_h),
                  (0, 255, 0), -1)
    put_text(overlay, f'{confidence:.0%}', (conf_x - 45, conf_y + 9),
             scale=0.4, color=(180, 180, 180))
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    return frame


# ════════════════════════════════════════════════════════════════════════════
# Confidence Overlay
# ════════════════════════════════════════════════════════════════════════════

def build_confidence_overlay(depth_conf=None, seg_conf=None, obstacle_scores=None,
                             width=320, height=240):
    """
    Build model confidence visualization panel.
    
    Returns:
        BGR image showing confidence maps
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'CONFIDENCE', (8, 18), scale=0.45, color=(200, 200, 200))
    
    if depth_conf is not None and seg_conf is not None:
        # Resize confidences
        h, w = height - 28, width // 2 - 5
        
        if depth_conf is not None:
            depth_conf_resized = cv2.resize(depth_conf, (w, h))
            depth_conf_colored = colorize_heatmap(depth_conf_resized, 'viridis')
            panel[28:28+h, 5:5+w] = depth_conf_colored
            put_text(panel, 'Depth', (5, 28 + h + 15), scale=0.35, color=(200, 200, 200))
        
        if seg_conf is not None:
            seg_conf_resized = cv2.resize(seg_conf, (w, h))
            seg_conf_colored = colorize_heatmap(seg_conf_resized, 'viridis')
            panel[28:28+h, width//2 + 5:width//2 + 5 + w] = seg_conf_colored
            put_text(panel, 'Seg', (width//2 + 5, 28 + h + 15), scale=0.35, color=(200, 200, 200))
    
    # Show obstacle scores if available
    if obstacle_scores is not None:
        h, w = 60, width - 10
        scores_resized = cv2.resize(obstacle_scores, (w, h))
        scores_colored = colorize_heatmap(scores_resized, 'hot')
        panel[height - h - 25:height - 25, 5:5+w] = scores_colored
        put_text(panel, 'Obstacle Score', (5, height - 28), scale=0.35, color=(200, 200, 200))
    
    return panel


# ════════════════════════════════════════════════════════════════════════════
# System Metrics Panel
# ════════════════════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """Track and visualize system performance metrics."""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.timestamps = deque(maxlen=max_history)
        self.fps_history = {
            'camera': deque(maxlen=max_history),
            'depth': deque(maxlen=max_history),
            'segmentation': deque(maxlen=max_history),
            'fusion': deque(maxlen=max_history),
            'path_planning': deque(maxlen=max_history),
        }
        self.latency_history = {
            'total': deque(maxlen=max_history),
            'depth': deque(maxlen=max_history),
            'segmentation': deque(maxlen=max_history),
            'fusion': deque(maxlen=max_history),
            'path': deque(maxlen=max_history),
        }
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self._lock = threading.Lock()
        
        # Accuracy metrics
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
    def update(self, fps_dict, latency_dict):
        """Update metrics with new frame data."""
        with self._lock:
            self.frame_count += 1
            current_time = time.perf_counter() - self.start_time
            
            self.timestamps.append(current_time)
            
            for key, value in fps_dict.items():
                if key in self.fps_history:
                    self.fps_history[key].append(value)
            
            for key, value in latency_dict.items():
                if key in self.latency_history:
                    self.latency_history[key].append(value)
    
    def record_detection(self, tp=0, fp=0, fn=0):
        """Record detection accuracy."""
        with self._lock:
            self.true_positives += tp
            self.false_positives += fp
            self.false_negatives += fn
    
    def get_precision(self):
        """Calculate precision."""
        with self._lock:
            tp = self.true_positives
            fp = self.false_positives
            if tp + fp == 0:
                return 0
            return tp / (tp + fp)
    
    def get_recall(self):
        """Calculate recall."""
        with self._lock:
            tp = self.true_positives
            fn = self.false_negatives
            if tp + fn == 0:
                return 0
            return tp / (tp + fn)
    
    def get_f1(self):
        """Calculate F1 score."""
        p = self.get_precision()
        r = self.get_recall()
        if p + r == 0:
            return 0
        return 2 * p * r / (p + r)


def build_metrics_panel(perf_monitor, width=320, height=240):
    """
    Build system performance metrics panel.
    
    Returns:
        BGR image with performance graphs
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'SYSTEM METRICS', (8, 18), scale=0.45, color=(200, 200, 200))
    
    with perf_monitor._lock:
        # Draw FPS graph
        graph_y = 35
        graph_h = 60
        graph_w = width - 10
        
        # Background
        cv2.rectangle(panel, (5, graph_y), (graph_w + 5, graph_y + graph_h),
                      (40, 40, 40), -1)
        
        # Draw FPS lines
        colors = {
            'camera': (100, 255, 100),
            'depth': (100, 100, 255),
            'segmentation': (255, 200, 100),
            'fusion': (255, 100, 255),
            'path_planning': (100, 255, 255),
        }
        
        for key, color in colors.items():
            history = list(perf_monitor.fps_history.get(key, []))
            if len(history) > 1:
                points = []
                for i, val in enumerate(history[-50:]):
                    x = int(5 + (i / 50) * graph_w)
                    y = int(graph_y + graph_h - (val / 30) * graph_h)
                    y = max(graph_y, min(graph_y + graph_h, y))
                    points.append((x, y))
                
                for i in range(len(points) - 1):
                    cv2.line(panel, points[i], points[i+1], color, 1)
        
        # Labels
        label_y = graph_y + graph_h + 12
        put_text(panel, 'FPS (max 30)', (5, label_y), scale=0.35, color=(150, 150, 150))
        
        # Draw latency breakdown
        latency_y = label_y + 20
        latency_h = 50
        
        cv2.rectangle(panel, (5, latency_y), (graph_w + 5, latency_y + latency_h),
                      (40, 40, 40), -1)
        
        # Latency bars
        latencies = {
            'depth': (100, 100, 255),
            'segmentation': (255, 200, 100),
            'fusion': (255, 100, 255),
            'path': (100, 255, 255),
        }
        
        bar_width = (graph_w - 20) // 4
        for i, (key, color) in enumerate(latencies.items()):
            history = list(perf_monitor.latency_history.get(key, []))
            if history:
                avg_lat = np.mean(history[-10:])
                bar_h = int(min(avg_lat / 200, 1.0) * latency_h)
                
                x = 5 + i * (bar_width + 5)
                y = latency_y + latency_h - bar_h
                
                cv2.rectangle(panel, (x, y), (x + bar_width - 5, latency_y + latency_h),
                              color, -1)
                put_text(panel, f'{avg_lat:.0f}ms', (x, latency_y + latency_h + 12),
                         scale=0.3, color=color)
        
        # Accuracy metrics
        metrics_y = latency_y + latency_h + 25
        
        precision = perf_monitor.get_precision()
        recall = perf_monitor.get_recall()
        f1 = perf_monitor.get_f1()
        
        put_text(panel, f'Precision: {precision:.2f}', (5, metrics_y),
                 scale=0.4, color=(200, 200, 100))
        put_text(panel, f'Recall: {recall:.2f}', (width//2, metrics_y),
                 scale=0.4, color=(100, 200, 200))
        
        # F1 score with color
        if f1 > 0.8:
            f1_color = (0, 255, 0)
        elif f1 > 0.5:
            f1_color = (0, 255, 255)
        else:
            f1_color = (0, 0, 255)
        
        put_text(panel, f'F1: {f1:.2f}', (5, metrics_y + 20),
                 scale=0.5, color=f1_color)
        
        # Frame count
        put_text(panel, f'Frames: {perf_monitor.frame_count}', 
                 (5, height - 10), scale=0.35, color=(150, 150, 150))
    
    return panel


# ════════════════════════════════════════════════════════════════════════════
# Session Recorder
# ════════════════════════════════════════════════════════════════════════════

class SessionRecorder:
    """Record session for later analysis."""
    
    def __init__(self, output_dir='recordings', fps=RECORD_FPS, codec=RECORD_CODEC):
        self.output_dir = output_dir
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.recording = False
        self.frame_count = 0
        self.start_time = None
        self._lock = threading.Lock()
        
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def start_recording(self, frame_shape):
        """Start recording to file."""
        import os
        import datetime
        
        with self._lock:
            if self.recording:
                return
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.output_dir, f'session_{timestamp}.mp4')
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                filename, fourcc, self.fps,
                (frame_shape[1], frame_shape[0])
            )
            
            if self.writer.isOpened():
                self.recording = True
                self.start_time = time.time()
                self.frame_count = 0
                print(f'[Recorder] Started recording: {filename}')
            else:
                print('[Recorder] Failed to start recording')
    
    def write_frame(self, frame):
        """Write frame to recording."""
        with self._lock:
            if self.recording and self.writer is not None:
                self.writer.write(frame)
                self.frame_count += 1
    
    def stop_recording(self):
        """Stop recording."""
        with self._lock:
            if self.recording:
                self.recording = False
                if self.writer is not None:
                    self.writer.release()
                    self.writer = None
                
                duration = time.time() - self.start_time if self.start_time else 0
                print(f'[Recorder] Stopped. Recorded {self.frame_count} frames in {duration:.1f}s')
    
    def is_recording(self):
        """Check if currently recording."""
        with self._lock:
            return self.recording


# ════════════════════════════════════════════════════════════════════════════
# Panel Builders
# ════════════════════════════════════════════════════════════════════════════

def build_camera_panel(frame, cam_fps, width=PANEL_W, height=PANEL_H):
    """Camera feed panel with timestamp and AR overlay option."""
    panel = cv2.resize(frame, (width, height))
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'CAMERA FEED', (8, 18), scale=0.50, color=(200, 200, 200))
    put_text(panel, f'{cam_fps:.1f} fps', (width - 75, 18),
             scale=0.45, color=(100, 220, 100))
    
    # Timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    put_text(panel, timestamp, (width - 70, height - 10),
             scale=0.40, color=(180, 180, 180))
    
    return panel


def build_depth_panel(depth_np, depth_fps, width=PANEL_W, height=PANEL_H):
    """Depth map panel with colorbar and zone warnings."""
    coloured, norm = colorize_depth(depth_np)
    panel = cv2.resize(coloured, (width, height))
    norm_s = cv2.resize(norm.astype(np.float32), (width, height))
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'DEPTH MAP (red=NEAR)', (8, 18),
             scale=0.45, color=(200, 200, 200))
    put_text(panel, f'{depth_fps:.1f} fps', (width - 75, 18),
             scale=0.45, color=(100, 220, 100))
    
    # Near-obstacle zone warnings
    NEAR_THRESH = 0.20
    near_mask = (norm_s < NEAR_THRESH).astype(np.uint8) * 255
    col_w = width // 3
    labels = ['LEFT', 'CENTER', 'RIGHT']
    positions = [0, col_w, 2 * col_w]
    
    for label, cx in zip(labels, positions):
        region = near_mask[:, cx: cx + col_w]
        frac = region.mean() / 255.0
        if frac > 0.08:
            danger_colour = (0, 0, 200) if frac > 0.25 else (0, 100, 220)
            overlay = panel.copy()
            cv2.rectangle(overlay, (cx + 2, height - 50),
                          (cx + col_w - 2, height - 2), danger_colour, 2)
            cv2.addWeighted(overlay, 0.4, panel, 0.6, 0, panel)
            put_text(panel, f'OBS {label}', (cx + 5, height - 10),
                     scale=0.40, color=(50, 50, 255))
    
    # Color scale bar
    bar_h, bar_w = 12, 100
    bar_x, bar_y = width - bar_w - 10, height - 22
    bar = np.linspace(0, 1, bar_w)[None, :]
    bar_rgb = (_CMAP_SPECTRAL(bar)[0, :, :3] * 255).astype(np.uint8)
    bar_bgr = bar_rgb[:, ::-1]
    bar_bgr = np.repeat(bar_bgr[None], bar_h, axis=0)
    panel[bar_y: bar_y + bar_h, bar_x: bar_x + bar_w] = bar_bgr
    put_text(panel, 'NEAR', (bar_x - 38, bar_y + 9), scale=0.35, color=(50, 50, 255))
    put_text(panel, 'FAR', (bar_x + bar_w + 3, bar_y + 9), scale=0.35, color=(200, 100, 40))
    
    return panel


def build_seg_panel(frame, seg_mask, seg_fps, width=PANEL_W, height=PANEL_H):
    """Segmentation panel with class legend."""
    coloured = colorize_seg(seg_mask, height, width)
    cam_small = cv2.resize(frame, (width, height))
    panel = cv2.addWeighted(coloured, 0.65, cam_small, 0.35, 0)
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'SEMANTIC SEG (ADE20K)', (8, 18),
             scale=0.45, color=(200, 200, 200))
    put_text(panel, f'{seg_fps:.1f} fps', (width - 75, 18),
             scale=0.45, color=(100, 220, 100))
    
    # Legend
    seg_small = cv2.resize(seg_mask.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST)
    unique, counts = np.unique(seg_small, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_k = min(6, len(unique))
    
    leg_x, leg_y = 8, 36
    cv2.rectangle(panel, (leg_x - 2, leg_y - 2),
                  (leg_x + 145, leg_y + top_k * 18 + 2),
                  (20, 20, 20), -1)
    
    for rank in range(top_k):
        cls_id = int(unique[order[rank]])
        if cls_id >= len(ADE20K_CLASSES):
            continue
        rgb = ADE20K_PALETTE[cls_id].tolist()
        bgr = (rgb[2], rgb[1], rgb[0])
        name = ADE20K_CLASSES[cls_id][:18]
        pct = 100.0 * counts[order[rank]] / seg_small.size
        put_label_box(panel, f'{name} {pct:.0f}%', leg_x, leg_y + rank * 18, bgr)
    
    return panel


def build_obstacle_panel(obstacle_img, nav_instruction, obs_fps, obstacle_info=None,
                          width=PANEL_W, height=PANEL_H):
    """Obstacle detection panel with navigation instruction."""
    if obstacle_info is None:
        obstacle_info = []
    
    panel = cv2.resize(obstacle_img, (width, height),
                       interpolation=cv2.INTER_NEAREST)
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'OBSTACLES (ODM)', (8, 18),
             scale=0.45, color=(200, 200, 200))
    put_text(panel, f'{obs_fps:.1f} fps', (width - 75, 18),
             scale=0.45, color=(100, 220, 100))
    
    # Nearest obstacle label
    if obstacle_info:
        nearest = obstacle_info[0]
        label = nearest['class_name'].upper()
        level = nearest.get('hierarchy_level', 3)
        
        if level == 1:
            color = (0, 0, 255)  # Red for critical
        elif level == 2:
            color = (0, 165, 255)  # Orange for caution
        else:
            color = (0, 200, 255)  # Yellow for navigable
        
        cv2.rectangle(panel, (0, 28), (width, 52), (15, 15, 15), -1)
        put_text(panel, f'NEAREST: {label}', (8, 46), scale=0.48, color=color)
    else:
        cv2.rectangle(panel, (0, 28), (width, 52), (15, 15, 15), -1)
        put_text(panel, 'NO OBSTACLES', (8, 46),
                 scale=0.48, color=(0, 255, 100))
    
    # Navigation instruction overlay
    if nav_instruction:
        cv2.rectangle(panel, (0, height - 40), (width, height),
                      (0, 0, 0), -1)
        inst_lower = nav_instruction.lower()
        if 'stop' in inst_lower:
            color = (0, 0, 255)
        elif 'left' in inst_lower or 'right' in inst_lower:
            color = (0, 200, 255)
        else:
            color = (0, 255, 100)
        put_text(panel, nav_instruction, (8, height - 12),
                 scale=0.50, color=color, bg_color=(0, 0, 0))
    
    return panel


def build_path_planner_panel(sector_bounds, ostatus, direction, nav_instruction,
                              width=PANEL_W, height=PANEL_H):
    """Path planner visualization panel with sector grid."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'PATH PLANNER', (8, 18), scale=0.45, color=(200, 200, 200))
    
    # Draw sector grid
    grid_margin = 10
    grid_w = width - 2 * grid_margin
    grid_h = height - 80
    grid_top = 35
    
    # Grid boundaries
    col1 = grid_margin
    col2 = grid_margin + int(grid_w * 0.33)
    col3 = grid_margin + int(grid_w * 0.67)
    col4 = width - grid_margin
    
    row1 = grid_top
    row2 = grid_top + grid_h // 2
    row3 = height - 45
    
    # Draw grid lines
    cv2.rectangle(panel, (col1, row1), (col4, row3), (80, 80, 80), 1)
    cv2.line(panel, (col2, row1), (col2, row3), (80, 80, 80), 1)
    cv2.line(panel, (col3, row1), (col3, row3), (80, 80, 80), 1)
    cv2.line(panel, (col1, row2), (col4, row2), (80, 80, 80), 1)
    
    # Draw sectors with OStatus coloring
    sectors = [
        ('top_left', col1, row1, col2, row2),
        ('top_mid', col2, row1, col3, row2),
        ('top_right', col3, row1, col4, row2),
        ('bot_left', col1, row2, col2, row3),
        ('bot_mid', col2, row2, col3, row3),
        ('bot_right', col3, row2, col4, row3),
    ]
    
    for name, x1, y1, x2, y2 in sectors:
        ost = ostatus.get(name, 0)
        
        # Color based on OStatus (green=free, red=blocked)
        intensity = int(ost * 255)
        if intensity > 200:
            color = (0, 0, 200)  # Very blocked - red
        elif intensity > 100:
            color = (0, 100, 200)  # Blocked - orange
        elif intensity > 50:
            color = (0, 180, 100)  # Some obstacles - yellow-green
        else:
            color = (0, 200, 50)  # Free - green
        
        cv2.rectangle(panel, (x1+2, y1+2), (x2-2, y2-2), color, -1)
        
        # OStatus value
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        put_text(panel, f'{ost:.2f}', (center_x - 15, center_y + 5),
                 scale=0.45, color=(255, 255, 255))
    
    # Draw direction arrow
    center_x = width // 2
    arrow_y = height - 25
    
    if direction < -0.2:
        arrow_text = "← LEFT"
        color = (0, 200, 255)
    elif direction > 0.2:
        arrow_text = "RIGHT →"
        color = (0, 200, 255)
    else:
        arrow_text = "↑ AHEAD"
        color = (0, 255, 100)
    
    put_text(panel, arrow_text, (center_x - 40, arrow_y),
             scale=0.55, color=color)
    
    return panel


def build_fusion_panel(fusion_img, width=PANEL_W, height=PANEL_H):
    """Fusion heatmap panel."""
    panel = cv2.resize(fusion_img, (width, height))
    
    # Header
    cv2.rectangle(panel, (0, 0), (width, 28), (30, 30, 30), -1)
    put_text(panel, 'FUSION HEATMAP', (8, 18), scale=0.45, color=(200, 200, 200))
    
    return panel


def build_status_bar(cam_fps, depth_fps, seg_fps, obs_fps, width=WIN_W):
    """Bottom status bar with FPS counters and key hints."""
    bar = np.zeros((STATUS_H, width, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    cv2.line(bar, (0, 0), (width, 0), (80, 80, 80), 1)
    
    put_text(bar, f'Cam: {cam_fps:4.1f}', (12, 22),
             scale=0.45, color=(130, 220, 130))
    put_text(bar, f'Depth: {depth_fps:4.1f}', (140, 22),
             scale=0.45, color=(130, 180, 255))
    put_text(bar, f'Seg: {seg_fps:4.1f}', (290, 22),
             scale=0.45, color=(255, 180, 100))
    put_text(bar, f'ODM: {obs_fps:4.1f}', (420, 22),
             scale=0.45, color=(255, 130, 130))
    put_text(bar, 'Q/Esc: Quit | S: Screenshot | M: Mute | R: Record',
             (width - 380, 22), scale=0.42, color=(160, 160, 160))
    
    # Model labels row
    cv2.line(bar, (0, STATUS_H // 2), (width, STATUS_H // 2), (50, 50, 50), 1)
    put_text(bar, 'Depth Anything V2 (vitb)', (12, STATUS_H - 8),
             scale=0.38, color=(90, 140, 200))
    put_text(bar, 'TopFormer-Base ADE20K', (280, STATUS_H - 8),
             scale=0.38, color=(180, 130, 90))
    put_text(bar, 'ODM + Fuzzy Path Planner (Enhanced)', (520, STATUS_H - 8),
             scale=0.38, color=(200, 90, 90))
    
    return bar
