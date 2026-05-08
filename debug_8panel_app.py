"""
8-Panel Debug Navigation System
================================
Advanced debugging display with performance metrics and model analysis.

Panel Layout (2x4 Grid):
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

Panel Descriptions:
    1. Camera Feed          - Original input with timestamp
    2. Depth Map            - Depth Anything V2 with near/far indicators
    3. Segmentation         - TopFormer output with class legend
    4. Fusion Heatmap       - Depth+Seg combined visualization
    5. Obstacle Map         - ODM output with obstacle labels
    6. Path Planner         - 6-sector grid with navigation
    7. Confidence Overlay   - Model uncertainty visualization
    8. System Metrics       - Real-time performance graphs

Debug Features:
    - Real-time FPS graphs
    - Processing latency breakdown
    - Model confidence heatmaps
    - Memory usage tracking
    - GPU utilization display

Controls:
    Q/Esc   - Quit
    S       - Save screenshot
    M       - Mute/unmute audio
    F       - Toggle fullscreen
    R       - Toggle recording
    D       - Toggle debug overlay
    1-8     - Toggle individual panels
    +/-     - Adjust panel brightness
"""

import os
import sys
import time
import datetime
import threading
import collections

import cv2
import numpy as np
import torch
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DEPTH_SRC = os.path.join(ROOT, 'obs-tackle', 'third_party', 'Depth-Anything-V2')
DEPTH_CKPT = os.path.join(ROOT, 'depth_anything_v2_vitb.pth')
SEG_ONNX = os.path.join(ROOT, 'topformer.onnx')
SCREENSHOT_DIR = os.path.join(ROOT, 'screenshots')
RECORDING_DIR = os.path.join(ROOT, 'recordings')

sys.path.insert(0, DEPTH_SRC)

# ── Display Configuration ───────────────────────────────────────────────────
PANEL_W = 320
PANEL_H = 240
GRID_COLS = 4
GRID_ROWS = 2
MARGIN = 3
HEADER_H = 22
STATUS_H = 60

WIN_W = PANEL_W * GRID_COLS + MARGIN * (GRID_COLS + 1)
WIN_H = PANEL_H * GRID_ROWS + MARGIN * (GRID_ROWS + 1) + STATUS_H

# ── Model Configuration ─────────────────────────────────────────────────────
DEPTH_ENCODER = 'vitb'
DEPTH_FEATURES = 128
DEPTH_OUT_CHANNELS = [96, 192, 384, 768]
DEPTH_INPUT_SIZE = 308

# ── ADE20K Configuration ────────────────────────────────────────────────────
ADE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
ADE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SEG_SIZE = 512

ADE20K_CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door',
    'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
    'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
    'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard',
    'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
    'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
    'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge',
    'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
    'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer',
    'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel',
    'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth',
    'television', 'airplane', 'dirt track', 'apparel', 'pole', 'land',
    'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage',
    'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
    'pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike',
    'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave',
    'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
    'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan',
    'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower',
    'radiator', 'glass', 'clock', 'flag',
]

ADE20K_PALETTE = np.array([
    [120,120,120],[180,120,120],[  6,230,230],[ 80, 50, 50],[  4,200,  3],
    [120,120, 80],[140,140,140],[204,  5,255],[230,230,230],[  4,250,  7],
    [224,  5,255],[235,255,  7],[150,  5, 61],[120,120, 70],[  8,255, 51],
    [255,  6, 82],[143,255,140],[204,255,  4],[255, 51,  7],[204, 70,  3],
    [  0,102,200],[ 61,230,250],[255,  6, 51],[ 11,102,255],[255,  7, 71],
    [255,  9,224],[  9,  7,230],[220,220,220],[255,  9, 92],[112,  9,255],
    [  8,255,214],[  7,255,224],[255,184,  6],[ 10,255, 71],[255, 41, 10],
    [  7,255,255],[224,255,  8],[102,  8,255],[255, 61,  6],[255,194,  7],
    [255,122,  8],[  0,255, 20],[255,  8, 41],[255,  5,153],[  6, 51,255],
    [235, 12,255],[160,150, 20],[  0,163,255],[140,140,140],[250, 10, 15],
    [ 20,255,  0],[ 31,255,  0],[255, 31,  0],[255,224,  0],[153,255,  0],
    [  0,  0,255],[255, 71,  0],[  0,235,255],[  0,173,255],[ 31,  0,255],
    [ 11,200,200],[255, 82,  0],[  0,255,245],[  0, 61,255],[  0,255,112],
    [  0,255,133],[255,  0,  0],[255,163,  0],[255,102,  0],[194,255,  0],
    [  0,143,255],[ 51,255,  0],[  0, 82,255],[  0,255, 41],[  0,255,173],
    [ 10,  0,255],[173,255,  0],[  0,255,153],[255, 92,  0],[255,  0,255],
    [255,  0,245],[255,  0,102],[255,173, 0],[255,  0, 20],[255,184,184],
    [  0, 31,255],[  0,255, 61],[  0, 71,255],[255,  0,204],[  0,255,194],
    [  0,255, 82],[  0, 10,255],[  0,112,255],[ 51,  0,255],[  0,194,255],
    [  0,122,255],[  0,255,163],[255,153,  0],[  0,255, 10],[255,112,  0],
    [143,255,  0],[ 82,  0,255],[163,255,  0],[255,235,  0],[  8,184,170],
    [133,  0,255],[  0,255, 92],[184,  0,255],[255,  0, 31],[  0,184,255],
    [  0,255,255],[255,255,  0],[255,255,  0],[255,255,  0],[255,255,  0],
], dtype=np.uint8)

PATH_CLASS_INDICES = {3, 6, 11, 52, 55, 96}

_CMAP_SPECTRAL = colormaps.get_cmap('Spectral_r')
_CMAP_JET = colormaps.get_cmap('jet')
_CMAP_HOT = colormaps.get_cmap('hot')


# ═════════════════════════════════════════════════════════════════════════════
# Performance Monitor
# ═════════════════════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """Track and visualize system performance metrics."""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.timestamps = collections.deque(maxlen=max_history)
        self.fps_history = {
            'camera': collections.deque(maxlen=max_history),
            'depth': collections.deque(maxlen=max_history),
            'segmentation': collections.deque(maxlen=max_history),
            'fusion': collections.deque(maxlen=max_history),
            'path_planning': collections.deque(maxlen=max_history),
        }
        self.latency_history = collections.deque(maxlen=max_history)
        self.frame_count = 0
        self.start_time = time.perf_counter()
        
    def update(self, fps_dict, latency_ms):
        """Update metrics with new frame data."""
        self.frame_count += 1
        current_time = time.perf_counter() - self.start_time
        
        self.timestamps.append(current_time)
        for key, value in fps_dict.items():
            if key in self.fps_history:
                self.fps_history[key].append(value)
        self.latency_history.append(latency_ms)
    
    def draw_metrics_panel(self, width, height):
        """Draw performance metrics visualization."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)
        
        if len(self.timestamps) < 2:
            return panel
        
        # Draw FPS graphs
        graph_height = (height - 40) // 3
        colors = {
            'camera': (100, 255, 100),
            'depth': (100, 100, 255),
            'segmentation': (255, 200, 100),
            'fusion': (255, 100, 255),
            'path_planning': (100, 255, 255),
        }
        
        # Title
        cv2.putText(panel, 'FPS History', (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw each FPS line
        y_offset = 30
        for name, history in self.fps_history.items():
            if len(history) < 2:
                continue
            
            # Draw label
            color = colors.get(name, (200, 200, 200))
            cv2.putText(panel, f'{name[:4]}: {list(history)[-1]:.1f}',
                       (10, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw graph line
            history_list = list(history)
            x_scale = width / self.max_history
            y_scale = graph_height / 35  # Scale for 0-35 FPS
            
            for i in range(1, len(history_list)):
                x1 = int((len(history_list) - len(history_list) + i - 1) * x_scale)
                y1 = int(y_offset + graph_height - history_list[i-1] * y_scale)
                x2 = int((len(history_list) - len(history_list) + i) * x_scale)
                y2 = int(y_offset + graph_height - history_list[i] * y_scale)
                cv2.line(panel, (x1, y1), (x2, y2), color, 1)
            
            y_offset += graph_height + 5
        
        # Draw latency graph at bottom
        if len(self.latency_history) > 1:
            latency_list = list(self.latency_history)
            y_start = height - 60
            cv2.putText(panel, f'Latency: {latency_list[-1]:.1f}ms',
                       (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
            
            x_scale = width / self.max_history
            y_scale = 40 / 100  # Scale for 0-100ms
            
            for i in range(1, len(latency_list)):
                x1 = int((i - 1) * x_scale)
                y1 = int(y_start + 40 - latency_list[i-1] * y_scale)
                x2 = int(i * x_scale)
                y2 = int(y_start + 40 - latency_list[i] * y_scale)
                cv2.line(panel, (x1, y1), (x2, y2), (255, 255, 100), 1)
        
        return panel


# ═════════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ═════════════════════════════════════════════════════════════════════════════

def put_text(img, text, pos, scale=0.55, color=(255, 255, 255),
             thickness=1, bg_color=(0, 0, 0)):
    """Draw text with shadow for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (pos[0] + 1, pos[1] + 1), font,
                scale, bg_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def colorize_depth(depth_np, cmap='Spectral_r'):
    """Raw depth array → coloured BGR image."""
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
    """Segmentation mask → coloured BGR image."""
    colour_map = ADE20K_PALETTE[seg_mask]
    colour_bgr = colour_map[..., ::-1].astype(np.uint8)
    colour_bgr = cv2.resize(colour_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    return colour_bgr


def create_panel_header(title, fps=0.0, color=(200, 200, 200)):
    """Create a panel header bar."""
    header = np.zeros((HEADER_H, PANEL_W, 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    put_text(header, title, (6, 15), scale=0.45, color=color)
    if fps > 0:
        put_text(header, f'{fps:.1f}', (PANEL_W - 45, 15),
                scale=0.42, color=(100, 220, 100))
    return header


def build_camera_panel(frame, cam_fps):
    """Panel 1: Original camera feed with overlay."""
    panel = cv2.resize(frame, (PANEL_W, PANEL_H - HEADER_H))
    
    # Timestamp
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    put_text(panel, timestamp, (PANEL_W - 65, PANEL_H - HEADER_H - 10),
             scale=0.38, color=(200, 200, 200))
    
    header = create_panel_header('📷 CAMERA', cam_fps, (200, 200, 200))
    return np.vstack([header, panel])


def build_depth_panel(depth_np, depth_fps):
    """Panel 2: Depth Anything V2 output."""
    coloured, norm = colorize_depth(depth_np, 'Spectral_r')
    panel = cv2.resize(coloured, (PANEL_W, PANEL_H - HEADER_H))
    
    # Depth scale indicator
    d_min, d_max = depth_np.min(), depth_np.max()
    put_text(panel, f'R:{d_min:.2f}-{d_max:.2f}', (5, 20),
             scale=0.40, color=(255, 255, 200))
    
    header = create_panel_header('🔍 DEPTH', depth_fps, (100, 180, 255))
    return np.vstack([header, panel])


def build_seg_panel(frame, seg_mask, seg_fps):
    """Panel 3: TopFormer semantic segmentation."""
    coloured = colorize_seg(seg_mask, PANEL_H - HEADER_H, PANEL_W)
    cam_small = cv2.resize(frame, (PANEL_W, PANEL_H - HEADER_H))
    panel = cv2.addWeighted(coloured, 0.65, cam_small, 0.35, 0)
    
    # Top-3 classes only (compact)
    seg_small = cv2.resize(seg_mask.astype(np.uint8), (PANEL_W, PANEL_H - HEADER_H),
                           interpolation=cv2.INTER_NEAREST)
    unique, counts = np.unique(seg_small, return_counts=True)
    order = np.argsort(counts)[::-1]
    
    y_offset = 25
    for rank in range(min(3, len(unique))):
        cls_id = int(unique[order[rank]])
        if cls_id >= len(ADE20K_CLASSES):
            continue
        rgb = ADE20K_PALETTE[cls_id].tolist()
        bgr = (rgb[2], rgb[1], rgb[0])
        name = ADE20K_CLASSES[cls_id][:10]
        pct = 100.0 * counts[order[rank]] / seg_small.size
        
        cv2.rectangle(panel, (3, y_offset + rank * 14 - 10), (14, y_offset + rank * 14 + 1), bgr, -1)
        put_text(panel, f'{name} {pct:.0f}%', (18, y_offset + rank * 14), scale=0.35, color=(220, 220, 220))
    
    header = create_panel_header('🎨 SEG', seg_fps, (255, 180, 100))
    return np.vstack([header, panel])


def build_fusion_panel(depth_np, seg_mask, fusion_fps):
    """Panel 4: Depth-Segmentation Fusion."""
    h, w = seg_mask.shape[:2]
    
    if depth_np.shape[:2] != (h, w):
        depth_resized = cv2.resize(depth_np.astype(np.float32), (w, h))
    else:
        depth_resized = depth_np
    
    d_min, d_max = depth_resized.min(), depth_resized.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_resized - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_resized)
    
    seg_color = colorize_seg(seg_mask, h, w)
    seg_hsv = cv2.cvtColor(seg_color, cv2.COLOR_BGR2HSV).astype(np.float32)
    seg_hsv[:, :, 2] = seg_hsv[:, :, 2] * (0.4 + 0.6 * (1.0 - depth_norm))
    seg_hsv = np.clip(seg_hsv, 0, 255).astype(np.uint8)
    fusion_vis = cv2.cvtColor(seg_hsv, cv2.COLOR_HSV2BGR)
    
    panel = cv2.resize(fusion_vis, (PANEL_W, PANEL_H - HEADER_H))
    
    put_text(panel, 'Fusion: Depth+Seg', (5, 18), scale=0.40, color=(255, 255, 255))
    
    header = create_panel_header('🔥 FUSION', fusion_fps, (255, 100, 255))
    return np.vstack([header, panel])


def build_obstacle_panel(seg_mask, depth_np, obstacle_info, obs_fps):
    """Panel 5: Obstacle Detection Map."""
    from skimage import measure
    
    h, w = seg_mask.shape[:2]
    if depth_np.shape[:2] != (h, w):
        depth_resized = cv2.resize(depth_np.astype(np.float32), (w, h))
    else:
        depth_resized = depth_np.astype(np.float32)
    
    max_disparity = depth_resized.max()
    threshold_d = 0.60 * max_disparity
    
    obstacle_vis = np.zeros((h, w, 3), dtype=np.uint8)
    obstacle_count = 0
    
    unique_labels = np.unique(seg_mask)
    for class_id in unique_labels:
        if int(class_id) in PATH_CLASS_INDICES:
            continue
        
        class_mask = (seg_mask == class_id).astype(np.uint8)
        labelled = measure.label(class_mask, connectivity=2)
        regions = measure.regionprops(labelled)
        
        for region in regions:
            if region.area < 100:
                continue
            
            coords = region.coords
            depths = depth_resized[coords[:, 0], coords[:, 1]]
            max_d = depths.max()
            
            if max_d > threshold_d:
                obstacle_count += 1
                rgb = ADE20K_PALETTE[int(class_id)]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                for y, x in coords:
                    obstacle_vis[y, x] = bgr
    
    panel = cv2.resize(obstacle_vis, (PANEL_W, PANEL_H - HEADER_H))
    
    # Overlay info
    cv2.rectangle(panel, (0, 0), (PANEL_W, 35), (20, 20, 20), -1)
    put_text(panel, f'OBS: {obstacle_count}', (5, 18),
             scale=0.50, color=(0, 200, 255) if obstacle_count > 0 else (0, 255, 100))
    
    header = create_panel_header('⚠️ ODM', obs_fps, (100, 255, 150))
    return np.vstack([header, panel])


def build_path_panel(frame, seg_mask, depth_np, planner_details, path_fps):
    """Panel 6: Path Planning with 6-sector grid."""
    panel = cv2.resize(frame, (PANEL_W, PANEL_H - HEADER_H))
    h, w = panel.shape[:2]
    
    rows, cols = 2, 3
    sh, sw = h // rows, w // cols
    
    ostatus = planner_details.get('ostatus', {})
    action = planner_details.get('action', 'WAIT')
    
    sector_names = [
        ('top_left', 'OH-L'), ('top_mid', 'OH-M'), ('top_right', 'OH-R'),
        ('bot_left', 'GR-L'), ('bot_mid', 'GR-M'), ('bot_right', 'GR-R')
    ]
    
    for idx, (sector_key, label) in enumerate(sector_names):
        row, col = idx // 3, idx % 3
        x1, y1 = col * sw, row * sh
        x2, y2 = x1 + sw, y1 + sh
        
        status = ostatus.get(sector_key, 0.0)
        
        if status > 0.5:
            color = (0, 0, 255)
            alpha = 0.3
        elif status > 0.25:
            color = (0, 165, 255)
            alpha = 0.2
        else:
            color = (0, 255, 0)
            alpha = 0.1
        
        overlay = panel.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0, panel)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (80, 80, 80), 1)
        
        put_text(panel, label, (x1 + 4, y1 + 16), scale=0.40, color=(255, 255, 255))
        put_text(panel, f'{status:.2f}', (x1 + 4, y1 + 32), scale=0.38, color=(200, 200, 200))
    
    # Action indicator
    action_colors = {
        'STRONG_LEFT': (255, 0, 0), 'WEAK_LEFT': (200, 100, 0),
        'FORWARD': (0, 255, 0), 'WEAK_RIGHT': (0, 200, 100),
        'STRONG_RIGHT': (0, 0, 255), 'WAIT': (128, 128, 128),
        'STOP': (0, 0, 255)
    }
    color = action_colors.get(action, (200, 200, 200))
    cv2.rectangle(panel, (0, h - 30), (w, h), color, -1)
    put_text(panel, action, (5, h - 8), scale=0.50, color=(255, 255, 255))
    
    header = create_panel_header('🧭 PATH', path_fps, (150, 255, 180))
    return np.vstack([header, panel])


def build_confidence_panel(depth_np, seg_mask, conf_fps):
    """Panel 7: Confidence overlay showing model uncertainty."""
    h, w = seg_mask.shape[:2]
    
    # Create confidence map based on depth variance
    if depth_np.shape[:2] != (h, w):
        depth_resized = cv2.resize(depth_np.astype(np.float32), (w, h))
    else:
        depth_resized = depth_np
    
    # Compute local variance as uncertainty proxy
    depth_blur = cv2.GaussianBlur(depth_resized, (5, 5), 0)
    variance = np.abs(depth_resized - depth_blur)
    
    # Normalize to 0-1 (higher variance = lower confidence)
    v_max = variance.max()
    if v_max > 1e-6:
        confidence = 1.0 - (variance / v_max)
    else:
        confidence = np.ones_like(variance)
    
    # Colorize: Green = high confidence, Red = low confidence
    confidence_uint8 = (confidence * 255).astype(np.uint8)
    confidence_color = cv2.applyColorMap(confidence_uint8, cv2.COLORMAP_JET)
    
    panel = cv2.resize(confidence_color, (PANEL_W, PANEL_H - HEADER_H))
    
    # Add legend
    put_text(panel, 'Confidence:', (5, 18), scale=0.42, color=(255, 255, 255))
    put_text(panel, 'Green=High Red=Low', (5, 36), scale=0.38, color=(200, 200, 200))
    
    # Mean confidence
    mean_conf = confidence.mean()
    color = (0, 255, 0) if mean_conf > 0.7 else (0, 165, 255) if mean_conf > 0.4 else (0, 0, 255)
    put_text(panel, f'Mean: {mean_conf:.2f}', (5, PANEL_H - HEADER_H - 10),
             scale=0.42, color=color)
    
    header = create_panel_header('🔒 CONF', conf_fps, (200, 200, 255))
    return np.vstack([header, panel])


def build_status_bar(nav_instruction, audio_on, recording):
    """Build bottom status bar."""
    bar = np.zeros((STATUS_H, WIN_W, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)
    
    # Status indicators
    put_text(bar, '🔊 ON' if audio_on else '🔇 OFF', (10, 22),
             scale=0.55, color=(100, 255, 100) if audio_on else (100, 100, 100))
    put_text(bar, '🔴 REC' if recording else '○ REC', (80, 22),
             scale=0.55, color=(0, 0, 255) if recording else (100, 100, 100))
    
    # Navigation instruction
    if nav_instruction:
        inst_lower = nav_instruction.lower()
        if 'stop' in inst_lower:
            color = (0, 0, 255)
        elif 'left' in inst_lower or 'right' in inst_lower:
            color = (0, 200, 255)
        else:
            color = (0, 255, 100)
        put_text(bar, f'► {nav_instruction}', (200, 35), scale=0.65, color=color)
    
    # Controls
    put_text(bar, 'Q:Quit S:Screenshot M:Mute F:Fullscreen R:Record',
             (WIN_W - 450, 55), scale=0.40, color=(150, 150, 150))
    
    return bar


# ═════════════════════════════════════════════════════════════════════════════
# Worker Threads (Same as multi_panel_app.py)
# ═════════════════════════════════════════════════════════════════════════════

class DepthWorker(threading.Thread):
    def __init__(self, model, shared, lock, stop_event):
        super().__init__(daemon=True)
        self.model = model
        self.shared = shared
        self.lock = lock
        self.stop_event = stop_event
        self.frame_queue = []
        self.fps = 0.0
        
    def push_frame(self, frame):
        if len(self.frame_queue) < 2:
            self.frame_queue.append(frame.copy())
    
    def run(self):
        import torch.nn.functional as F
        frame_count = 0
        t_start = time.perf_counter()
        
        while not self.stop_event.is_set():
            if not self.frame_queue:
                time.sleep(0.005)
                continue
            
            frame = self.frame_queue.pop(0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE))
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            
            device = next(self.model.parameters()).device
            img = img.to(device)
            
            with torch.no_grad():
                depth = self.model(img)
            
            depth = F.interpolate(depth, size=(PANEL_H, PANEL_W),
                                  mode='bilinear', align_corners=False)
            depth_np = depth.squeeze().cpu().numpy()
            
            with self.lock:
                self.shared['depth'] = depth_np
                self.shared['depth_fps'] = self.fps
            
            frame_count += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 0.5:
                self.fps = frame_count / elapsed
                frame_count = 0
                t_start = time.perf_counter()


class SegWorker(threading.Thread):
    def __init__(self, session, shared, lock, stop_event):
        super().__init__(daemon=True)
        self.session = session
        self.shared = shared
        self.lock = lock
        self.stop_event = stop_event
        self.frame_queue = []
        self.fps = 0.0
    
    def push_frame(self, frame):
        if len(self.frame_queue) < 2:
            self.frame_queue.append(frame.copy())
    
    def run(self):
        input_name = self.session.get_inputs()[0].name
        frame_count = 0
        t_start = time.perf_counter()
        
        while not self.stop_event.is_set():
            if not self.frame_queue:
                time.sleep(0.005)
                continue
            
            frame = self.frame_queue.pop(0)
            h, w = frame.shape[:2]
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (SEG_SIZE, SEG_SIZE))
            img = img.astype(np.float32)
            img = (img - ADE_MEAN) / ADE_STD
            img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
            
            outputs = self.session.run(None, {input_name: img})
            seg = outputs[0].argmax(axis=1).squeeze().astype(np.uint8)
            seg_resized = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
            
            with self.lock:
                self.shared['seg'] = seg_resized
                self.shared['seg_fps'] = self.fps
            
            frame_count += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 0.5:
                self.fps = frame_count / elapsed
                frame_count = 0
                t_start = time.perf_counter()


# ═════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═════════════════════════════════════════════════════════════════════════════

def detect_obstacles_simple(seg_mask, depth_map):
    """Simplified obstacle detection."""
    from skimage import measure
    
    h, w = seg_mask.shape[:2]
    if depth_map.shape[:2] != (h, w):
        depth_resized = cv2.resize(depth_map.astype(np.float32), (w, h))
    else:
        depth_resized = depth_map.astype(np.float32)
    
    max_disparity = depth_resized.max()
    if max_disparity < 1e-6:
        return []
    
    threshold_d = 0.60 * max_disparity
    obstacle_info = []
    
    unique_labels = np.unique(seg_mask)
    for class_id in unique_labels:
        if int(class_id) in PATH_CLASS_INDICES:
            continue
        
        class_mask = (seg_mask == class_id).astype(np.uint8)
        labelled = measure.label(class_mask, connectivity=2)
        regions = measure.regionprops(labelled)
        
        for region in regions:
            if region.area < 100:
                continue
            
            coords = region.coords
            depths = depth_resized[coords[:, 0], coords[:, 1]]
            max_d = depths.max()
            
            if max_d > threshold_d:
                obstacle_info.append({
                    'class_id': int(class_id),
                    'class_name': ADE20K_CLASSES[int(class_id)],
                    'disparity': float(max_d),
                    'bbox': region.bbox,
                    'area': int(region.area)
                })
    
    obstacle_info.sort(key=lambda x: x['disparity'], reverse=True)
    return obstacle_info


def plan_path_simple(seg_mask, depth_map):
    """Simplified path planning."""
    h, w = seg_mask.shape[:2]
    
    if depth_map.shape[:2] != (h, w):
        depth = cv2.resize(depth_map.astype(np.float32), (w, h))
    else:
        depth = depth_map.astype(np.float32)
    
    max_d = depth.max()
    if max_d < 1e-6:
        return 'WAIT — No depth data', {'action': 'WAIT', 'ostatus': {}}
    
    depth_ratio = 0.40
    near_mask = depth > (depth_ratio * max_d)
    
    for path_idx in PATH_CLASS_INDICES:
        near_mask[seg_mask == path_idx] = False
    
    sectors = {
        'top_left': (0, 0, w//3, h//2),
        'top_mid': (w//3, 0, 2*w//3, h//2),
        'top_right': (2*w//3, 0, w, h//2),
        'bot_left': (0, h//2, w//3, h),
        'bot_mid': (w//3, h//2, 2*w//3, h),
        'bot_right': (2*w//3, h//2, w, h),
    }
    
    ostatus = {}
    for name, (x1, y1, x2, y2) in sectors.items():
        sector_mask = near_mask[y1:y2, x1:x2]
        ostatus[name] = float(sector_mask.mean()) if sector_mask.size > 0 else 0.0
    
    bot_mid_status = ostatus.get('bot_mid', 0.0)
    bot_left_status = ostatus.get('bot_left', 0.0)
    bot_right_status = ostatus.get('bot_right', 0.0)
    
    THRESHOLD = 0.25
    
    if bot_mid_status > THRESHOLD * 2:
        action = 'STOP'
        instruction = 'STOP — Obstacle ahead'
    elif bot_mid_status > THRESHOLD:
        if bot_left_status < bot_right_status:
            action = 'WEAK_LEFT'
            instruction = 'Turn slightly LEFT'
        else:
            action = 'WEAK_RIGHT'
            instruction = 'Turn slightly RIGHT'
    else:
        if bot_left_status > THRESHOLD * 1.5:
            action = 'WEAK_RIGHT'
            instruction = 'Veering RIGHT'
        elif bot_right_status > THRESHOLD * 1.5:
            action = 'WEAK_LEFT'
            instruction = 'Veering LEFT'
        else:
            action = 'FORWARD'
            instruction = 'Proceed FORWARD'
    
    return instruction, {'action': action, 'ostatus': ostatus}


class SimpleAudio:
    def __init__(self):
        self.enabled = True
    
    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
    
    def speak(self, text):
        pass
    
    def shutdown(self):
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Main Application
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('8-Panel Debug Navigation System')
    print('=' * 70)
    
    if not os.path.exists(DEPTH_CKPT):
        print(f'ERROR: Depth checkpoint not found: {DEPTH_CKPT}')
        sys.exit(1)
    if not os.path.exists(SEG_ONNX):
        print(f'ERROR: ONNX model not found: {SEG_ONNX}')
        sys.exit(1)
    
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)
    
    print('\n[1/4] Loading Depth Anything V2...')
    from depth_anything_v2.dpt import DepthAnythingV2
    
    DEVICE = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'      Device: {DEVICE}')
    
    depth_model = DepthAnythingV2(
        encoder=DEPTH_ENCODER,
        features=DEPTH_FEATURES,
        out_channels=DEPTH_OUT_CHANNELS,
    )
    depth_model.load_state_dict(
        torch.load(DEPTH_CKPT, map_location='cpu', weights_only=True))
    depth_model = depth_model.to(DEVICE).eval()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print('      Depth model ready.')
    
    print('[2/4] Loading TopFormer ONNX...')
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if DEVICE == 'cuda' else ['CPUExecutionProvider'])
    seg_session = ort.InferenceSession(SEG_ONNX, providers=providers)
    print(f'      Providers: {seg_session.get_providers()}')
    
    print('[3/4] Initializing audio...')
    audio = SimpleAudio()
    
    print('[4/4] Opening camera...')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Cannot open camera.')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'      Camera: {actual_w}x{actual_h}')
    
    print('\n' + '=' * 70)
    print('Controls: Q=Quit S=Screenshot M=Mute F=Fullscreen R=Record')
    print('=' * 70 + '\n')
    
    shared = {
        'depth': np.zeros((PANEL_H, PANEL_W), dtype=np.float32),
        'depth_fps': 0.0,
        'seg': np.zeros((actual_h, actual_w), dtype=np.uint8),
        'seg_fps': 0.0,
    }
    lock = threading.Lock()
    stop_event = threading.Event()
    
    depth_worker = DepthWorker(depth_model, shared, lock, stop_event)
    seg_worker = SegWorker(seg_session, shared, lock, stop_event)
    depth_worker.start()
    seg_worker.start()
    
    # Performance monitor
    perf_monitor = PerformanceMonitor(max_history=100)
    
    win_name = 'Navigation System - 8 Panel Debug Display'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIN_W, WIN_H)
    
    cam_fps = 0.0
    fusion_fps = 0.0
    obs_fps = 0.0
    path_fps = 0.0
    conf_fps = 30.0
    frame_count = 0
    t_fps_start = time.perf_counter()
    
    fullscreen = False
    recording = False
    video_writer = None
    nav_instruction = ''
    planner_details = {'action': 'WAIT', 'ostatus': {}}
    obstacle_info = []
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            frame_count += 1
            
            if frame_count % 2 == 0:
                depth_worker.push_frame(frame)
                seg_worker.push_frame(frame)
            
            elapsed = time.perf_counter() - t_fps_start
            if elapsed >= 0.5:
                cam_fps = frame_count / elapsed
                frame_count = 0
                t_fps_start = time.perf_counter()
            
            with lock:
                depth_np = shared['depth'].copy()
                seg_mask = shared['seg'].copy()
                depth_fps = shared['depth_fps']
                seg_fps = shared['seg_fps']
            
            # Process
            t_fusion = time.perf_counter()
            fusion_fps = 1.0 / max(time.perf_counter() - t_fusion, 1e-6)
            
            nav_instruction, planner_details = plan_path_simple(seg_mask, depth_np)
            
            t_obs = time.perf_counter()
            obstacle_info = detect_obstacles_simple(seg_mask, depth_np)
            obs_fps = 1.0 / max(time.perf_counter() - t_obs, 1e-6)
            
            # Build panels
            panel1 = build_camera_panel(frame, cam_fps)
            panel2 = build_depth_panel(depth_np, depth_fps)
            panel3 = build_seg_panel(frame, seg_mask, seg_fps)
            panel4 = build_fusion_panel(depth_np, seg_mask, fusion_fps)
            panel5 = build_obstacle_panel(seg_mask, depth_np, obstacle_info, obs_fps)
            panel6 = build_path_panel(frame, seg_mask, depth_np, planner_details, path_fps)
            panel7 = build_confidence_panel(depth_np, seg_mask, conf_fps)
            panel8 = perf_monitor.draw_metrics_panel(PANEL_W, PANEL_H - HEADER_H)
            header8 = create_panel_header('📊 METRICS', 0, (255, 255, 150))
            panel8 = np.vstack([header8, panel8])
            
            # Assemble 2x4 grid
            margin_v = np.zeros((PANEL_H, MARGIN, 3), dtype=np.uint8)
            
            row1 = np.hstack([panel1, margin_v, panel2, margin_v, panel3, margin_v, panel4])
            row2 = np.hstack([panel5, margin_v, panel6, margin_v, panel7, margin_v, panel8])
            
            margin_h = np.zeros((MARGIN, WIN_W, 3), dtype=np.uint8)
            grid = np.vstack([row1, margin_h, row2])
            
            # Status bar
            status_bar = build_status_bar(nav_instruction, audio.enabled, recording)
            display = np.vstack([grid, status_bar])
            
            # Recording
            if recording:
                if video_writer is None:
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    video_path = os.path.join(RECORDING_DIR, f'recording_{ts}.avi')
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (WIN_W, WIN_H))
                video_writer.write(display)
            elif video_writer is not None:
                video_writer.release()
                video_writer = None
            
            cv2.imshow(win_name, display)
            
            # Update performance monitor
            loop_latency = (time.perf_counter() - loop_start) * 1000  # ms
            fps_dict = {'camera': cam_fps, 'depth': depth_fps, 'segmentation': seg_fps,
                       'fusion': fusion_fps, 'path_planning': path_fps}
            perf_monitor.update(fps_dict, loop_latency)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('s'), ord('S')):
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(SCREENSHOT_DIR, f'8panel_{ts}.png')
                cv2.imwrite(path, display)
                print(f'Screenshot saved: {path}')
            elif key in (ord('m'), ord('M')):
                state = audio.toggle()
                print(f'Audio {"ON" if state else "OFF"}')
            elif key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else 0
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, prop)
            elif key in (ord('r'), ord('R')):
                recording = not recording
                print(f'Recording {"STARTED" if recording else "STOPPED"}')
    
    finally:
        print('\nShutting down...')
        stop_event.set()
        depth_worker.join(timeout=3)
        seg_worker.join(timeout=3)
        if video_writer is not None:
            video_writer.release()
        audio.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        print('Done.')


if __name__ == '__main__':
    main()
