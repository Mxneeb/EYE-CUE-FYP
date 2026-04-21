"""
Enhanced Navigation Assistance Application.

Integrates all improvements from FYP_IMPROVEMENTS.md:
    - Enhanced Depth-Segmentation Fusion (confidence-weighted)
    - Temporal consistency (Kalman filtering)
    - Variable grid path planning
    - Trajectory prediction
    - Alternative path generation
    - Spatial audio feedback
    - Audio icons and rhythm-based guidance
    - 8-panel display with metrics
    - Session recording
    - Performance monitoring

Controls:
    Q / Esc  — quit
    S        — save screenshot
    M        — mute/unmute audio
    F        — toggle fullscreen
    R        — toggle recording
    D        — toggle debug overlay
    1-8      — toggle individual panels
    +/-      — adjust panel brightness
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

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


# ── Import Enhanced Modules ─────────────────────────────────────────────────
from nav_assist.enhanced_obstacle import (
    detect_obstacles_enhanced, ObstacleTracker,
    compute_depth_confidence, compute_seg_confidence,
    get_nearest_obstacle_direction, create_obstacle_heatmap,
)
from nav_assist.enhanced_path_planner import (
    plan_path_enhanced, TrajectoryHistory,
    draw_navigation_arrow, compute_variable_sector_bounds,
)
from nav_assist.enhanced_visualization import (
    PerformanceMonitor, SessionRecorder,
    draw_depth_histogram, draw_trajectory, create_hazard_heatmap,
    draw_ar_navigation, build_confidence_overlay, build_metrics_panel,
    build_depth_histogram_panel, build_hazard_heatmap_panel,
    put_text, put_label_box, colorize_depth, colorize_seg, colorize_heatmap,
)
from nav_assist.enhanced_audio import EnhancedAudioFeedback, ComfortScorer

from matplotlib import colormaps
_CMAP_SPECTRAL = colormaps.get_cmap('Spectral_r')


# ════════════════════════════════════════════════════════════════════════════
# Inference Workers
# ════════════════════════════════════════════════════════════════════════════

class DepthWorker(threading.Thread):
    """Background depth estimation worker."""
    
    def __init__(self, model, shared, lock, stop_event):
        super().__init__(daemon=True)
        self.model = model
        self.shared = shared
        self.lock = lock
        self.stop_event = stop_event
        self.frame = None
        self.ready = threading.Event()
        
    def push_frame(self, frame):
        self.frame = frame.copy()
        self.ready.set()
    
    def run(self):
        from depth_anything_v2.dpt import DepthAnythingV2
        
        while not self.stop_event.is_set():
            self.ready.wait()
            if self.stop_event.is_set():
                break
            
            frame = self.frame
            self.ready.clear()
            
            if frame is None:
                continue
            
            # Preprocess
            h, w = frame.shape[:2]
            input_size = DEPTH_INPUT_SIZE
            
            # Resize and normalize
            resized = cv2.resize(frame, (input_size, input_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
            
            # Inference
            t_start = time.perf_counter()
            
            with torch.no_grad():
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                depth_pred = self.model(tensor)
                depth_pred = depth_pred.squeeze().cpu().numpy()
            
            # Resize back to original
            depth_map = cv2.resize(depth_pred, (w, h))
            
            t_end = time.perf_counter()
            
            with self.lock:
                self.shared['depth'] = depth_map
                self.shared['depth_fps'] = 1.0 / max(t_end - t_start, 1e-6)


class SegWorker(threading.Thread):
    """Background segmentation worker."""
    
    def __init__(self, session, shared, lock, stop_event):
        super().__init__(daemon=True)
        self.session = session
        self.shared = shared
        self.lock = lock
        self.stop_event = stop_event
        self.frame = None
        self.ready = threading.Event()
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        
    def push_frame(self, frame):
        self.frame = frame.copy()
        self.ready.set()
    
    def run(self):
        while not self.stop_event.is_set():
            self.ready.wait()
            if self.stop_event.is_set():
                break
            
            frame = self.frame
            self.ready.clear()
            
            if frame is None:
                continue
            
            # Preprocess
            h, w = frame.shape[:2]
            
            resized = cv2.resize(frame, (SEG_SIZE, SEG_SIZE))
            rgb = resized[:, :, ::-1]  # BGR to RGB
            
            # Normalize
            rgb = rgb.astype(np.float32)
            rgb = (rgb - ADE_MEAN) / ADE_STD
            
            # Transpose for CHW
            blob = rgb.transpose(2, 0, 1)[None, ...]
            
            # Inference
            t_start = time.perf_counter()
            
            seg_logits = self.session.run([self.output_name], {self.input_name: blob})[0]
            seg_pred = seg_logits[0].argmax(axis=0).astype(np.uint8)
            
            t_end = time.perf_counter()
            
            # Resize to original
            seg_mask = cv2.resize(seg_pred, (w, h), interpolation=cv2.INTER_NEAREST)
            
            with self.lock:
                self.shared['seg'] = seg_mask
                self.shared['seg_fps'] = 1.0 / max(t_end - t_start, 1e-6)


# ════════════════════════════════════════════════════════════════════════════
# Panel Drawing Functions
# ════════════════════════════════════════════════════════════════════════════

def build_camera_panel(frame, cam_fps):
    """Panel 1: Original camera feed."""
    panel = cv2.resize(frame, (PANEL_W, PANEL_H))
    
    cv2.rectangle(panel, (0, 0), (PANEL_W, HEADER_H), (30, 30, 30), -1)
    put_text(panel, '1 CAMERA FEED', (8, 16), scale=0.45, color=(200, 200, 200))
    put_text(panel, f'{cam_fps:.1f} fps', (PANEL_W - 65, 16),
             scale=0.40, color=(100, 220, 100))
    
    return panel


def build_depth_panel(depth_np, depth_fps):
    """Panel 2: Depth Anything V2 output."""
    coloured, norm = colorize_depth(depth_np, 'Spectral_r')
    panel = cv2.resize(coloured, (PANEL_W, PANEL_H))
    norm_s = cv2.resize(norm.astype(np.float32), (PANEL_W, PANEL_H))
    
    cv2.rectangle(panel, (0, 0), (PANEL_W, HEADER_H), (30, 30, 30), -1)
    put_text(panel, '2 DEPTH MAP', (8, 16), scale=0.45, color=(200, 200, 200))
    put_text(panel, f'{depth_fps:.1f} fps', (PANEL_W - 65, 16),
             scale=0.40, color=(100, 220, 100))
    
    # Near-obstacle warnings
    NEAR_THRESH = 0.20
    near_mask = (norm_s < NEAR_THRESH).astype(np.uint8) * 255
    col_w = PANEL_W // 3
    labels = ['L', 'C', 'R']
    positions = [0, col_w, 2 * col_w]
    
    for label, cx in zip(labels, positions):
        region = near_mask[:, cx: cx + col_w]
        frac = region.mean() / 255.0
        if frac > 0.08:
            danger = (0, 0, 200) if frac > 0.25 else (0, 100, 220)
            overlay = panel.copy()
            cv2.rectangle(overlay, (cx + 2, PANEL_H - 45),
                          (cx + col_w - 2, PANEL_H - 2), danger, 2)
            cv2.addWeighted(overlay, 0.4, panel, 0.6, 0, panel)
            put_text(panel, f'OBS {label}', (cx + 5, PANEL_H - 12),
                     scale=0.38, color=(50, 50, 255))
    
    # Color bar
    bar_h, bar_w = 10, 80
    bar_x, bar_y = PANEL_W - bar_w - 8, PANEL_H - 18
    bar = np.linspace(0, 1, bar_w)[None, :]
    bar_rgb = (_CMAP_SPECTRAL(bar)[0, :, :3] * 255).astype(np.uint8)
    bar_bgr = bar_rgb[:, ::-1]
    bar_bgr = np.repeat(bar_bgr[None], bar_h, axis=0)
    panel[bar_y: bar_y + bar_h, bar_x: bar_x + bar_w] = bar_bgr
    put_text(panel, 'N', (bar_x - 12, bar_y + 8), scale=0.30, color=(50, 50, 255))
    put_text(panel, 'F', (bar_x + bar_w + 2, bar_y + 8), scale=0.30, color=(200, 100, 40))
    
    return panel


def build_seg_panel(frame, seg_mask, seg_fps):
    """Panel 3: TopFormer segmentation."""
    coloured = colorize_seg(seg_mask, PANEL_H, PANEL_W)
    cam_small = cv2.resize(frame, (PANEL_W, PANEL_H))
    panel = cv2.addWeighted(coloured, 0.65, cam_small, 0.35, 0)
    
    cv2.rectangle(panel, (0, 0), (PANEL_W, HEADER_H), (30, 30, 30), -1)
    put_text(panel, '3 SEGMENTATION', (8, 16), scale=0.45, color=(200, 200, 200))
    put_text(panel, f'{seg_fps:.1f} fps', (PANEL_W - 65, 16),
             scale=0.40, color=(100, 220, 100))
    
    # Legend
    seg_small = cv2.resize(seg_mask.astype(np.uint8), (PANEL_W, PANEL_H),
                            interpolation=cv2.INTER_NEAREST)
    unique, counts = np.unique(seg_small, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_k = min(5, len(unique))
    
    leg_x, leg_y = 5, HEADER_H + 5
    cv2.rectangle(panel, (leg_x - 1, leg_y - 1),
                  (leg_x + 120, leg_y + top_k * 16 + 1),
                  (20, 20, 20), -1)
    
    for rank in range(top_k):
        cls_id = int(unique[order[rank]])
        if cls_id >= len(ADE20K_CLASSES):
            continue
        rgb = ADE20K_PALETTE[cls_id].tolist()
        bgr = (rgb[2], rgb[1], rgb[0])
        name = ADE20K_CLASSES[cls_id][:15]
        pct = 100.0 * counts[order[rank]] / seg_small.size
        put_label_box(panel, f'{name} {pct:.0f}%', leg_x, leg_y + rank * 16, bgr)
    
    return panel


def build_fusion_panel(obstacle_mask, depth_np, obstacle_info):
    """Panel 4: Fusion heatmap / hazard visualization."""
    heatmap = create_hazard_heatmap(obstacle_mask, depth_np, obstacle_info)
    panel = cv2.resize(heatmap, (PANEL_W, PANEL_H))
    
    cv2.rectangle(panel, (0, 0), (PANEL_W, HEADER_H), (30, 30, 30), -1)
    put_text(panel, '4 FUSION HEATMAP', (8, 16), scale=0.45, color=(200, 200, 200))
    
    # Legend
    put_text(panel, 'SAFE', (5, PANEL_H - 10), scale=0.35, color=(200, 200, 200))
    put_text(panel, 'DANGER', (PANEL_W - 60, PANEL_H - 10), scale=0.35, color=(200, 200, 200))
    
    return panel


def build_obstacle_panel(obstacle_bgr, obstacle_info, nav_instruction, planner_details):
    """Panel 5: Obstacle detection map."""
    panel = cv2.resize(obstacle_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)
    
    cv2.rectangle(panel, (0, 0), (PANEL_W, HEADER_H), (30, 30, 30), -1)
    put_text(panel, '5 OBSTACLE DETECTION', (8, 16), scale=0.45, color=(200, 200, 200))
    
    # Nearest obstacle label
    if obstacle_info:
        nearest = obstacle_info[0]
        label = nearest['class_name'].upper()
        level = nearest.get('hierarchy_level', 3)
        
        if level == 1:
            color = (0, 0, 255)
        elif level == 2:
            color = (0, 165, 255)
        else:
            color = (0, 200, 255)
        
        cv2.rectangle(panel, (0, HEADER_H), (PANEL_W, HEADER_H + 22), (15, 15, 15), -1)
        put_text(panel, f'NEAREST: {label}', (8, HEADER_H + 16), scale=0.42, color=color)
        
        # Show trajectory if available
        if 'trajectory' in nearest and nearest['trajectory']:
            trajectory = nearest['trajectory']
            for i in range(1, len(trajectory)):
                pt1 = (int(trajectory[i-1][0] * PANEL_W / 640),
                       int(trajectory[i-1][1] * PANEL_H / 480))
                pt2 = (int(trajectory[i][0] * PANEL_W / 640),
                       int(trajectory[i][1] * PANEL_H / 480))
                cv2.line(panel, pt1, pt2, (0, 255, 0), 2)
    else:
        cv2.rectangle(panel, (0, HEADER_H), (PANEL_W, HEADER_H + 22), (15, 15, 15), -1)
        put_text(panel, 'NO OBSTACLES', (8, HEADER_H + 16), scale=0.42, color=(0, 255, 100))
    
    # Navigation instruction
    if nav_instruction:
        cv2.rectangle(panel, (0, PANEL_H - 35), (PANEL_W, PANEL_H), (0, 0, 0), -1)
        inst_lower = nav_instruction.lower()
        if 'stop' in inst_lower:
            color = (0, 0, 255)
        elif 'left' in inst_lower or 'right' in inst_lower:
            color = (0, 200, 255)
        else:
            color = (0, 255, 100)
        put_text(panel, nav_instruction, (8, PANEL_H - 12), scale=0.45, color=color)
    
    return panel


def build_path_panel(sector_bounds, ostatus, planner_details):
    """Panel 6: Enhanced path planner visualization."""
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    
    cv2.rectangle(panel, (0, 0), (PANEL_W, HEADER_H), (30, 30, 30), -1)
    put_text(panel, '6 PATH PLANNER', (8, 16), scale=0.45, color=(200, 200, 200))
    
    # Draw sector grid
    grid_margin = 8
    grid_w = PANEL_W - 2 * grid_margin
    grid_h = PANEL_H - 75
    grid_top = HEADER_H + 5
    
    col1 = grid_margin
    col2 = grid_margin + int(grid_w * 0.33)
    col3 = grid_margin + int(grid_w * 0.67)
    col4 = PANEL_W - grid_margin
    
    row1 = grid_top
    row2 = grid_top + grid_h // 2
    row3 = PANEL_H - 30
    
    # Grid
    cv2.rectangle(panel, (col1, row1), (col4, row3), (80, 80, 80), 1)
    cv2.line(panel, (col2, row1), (col2, row3), (80, 80, 80), 1)
    cv2.line(panel, (col3, row1), (col3, row3), (80, 80, 80), 1)
    cv2.line(panel, (col1, row2), (col4, row2), (80, 80, 80), 1)
    
    # Sectors
    sectors = [
        ('top_left', col1, row1, col2, row2),
        ('top_mid', col2, row1, col3, row2),
        ('top_right', col3, row1, col4, row2),
        ('bot_left', col1, row2, col2, row3),
        ('bot_mid', col2, row2, col3, row3),
        ('bot_right', col3, row2, col4, row3),
    ]
    
    direction = planner_details.get('direction', 0)
    
    for name, x1, y1, x2, y2 in sectors:
        ost = ostatus.get(name, 0)
        
        # Color based on OStatus
        if ost > 0.5:
            color = (0, 0, 180)  # Red - blocked
        elif ost > 0.25:
            color = (0, 100, 180)  # Orange - caution
        elif ost > 0.1:
            color = (0, 150, 100)  # Yellow-green
        else:
            color = (0, 180, 50)  # Green - free
        
        cv2.rectangle(panel, (x1+2, y1+2), (x2-2, y2-2), color, -1)
        
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        put_text(panel, f'{ost:.2f}', (cx - 18, cy + 4), scale=0.38, color=(255, 255, 255))
    
    # Direction arrow
    center_x = PANEL_W // 2
    arrow_y = PANEL_H - 12
    
    if direction < -0.2:
        put_text(panel, '< LEFT', (center_x - 35, arrow_y), scale=0.50, color=(0, 200, 255))
    elif direction > 0.2:
        put_text(panel, 'RIGHT >', (center_x - 35, arrow_y), scale=0.50, color=(0, 200, 255))
    else:
        put_text(panel, '^ AHEAD', (center_x - 35, arrow_y), scale=0.50, color=(0, 255, 100))
    
    return panel


def build_confidence_panel(depth_conf, seg_conf, obstacle_scores):
    """Panel 7: Confidence overlay."""
    panel = build_confidence_overlay(depth_conf, seg_conf, obstacle_scores,
                                      PANEL_W, PANEL_H)
    
    cv2.rectangle(panel, (0, 0), (PANEL_W, HEADER_H), (30, 30, 30), -1)
    put_text(panel, '7 CONFIDENCE', (8, 16), scale=0.45, color=(200, 200, 200))
    
    return panel


def build_metrics_panel(perf_monitor):
    """Panel 8: System performance metrics."""
    panel = build_metrics_panel(perf_monitor, PANEL_W, PANEL_H)
    return panel


# ════════════════════════════════════════════════════════════════════════════
# Status Bar
# ════════════════════════════════════════════════════════════════════════════

def build_status_bar(cam_fps, depth_fps, seg_fps, obs_fps, is_recording=False):
    """Bottom status bar."""
    bar = np.zeros((STATUS_H, WIN_W, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    cv2.line(bar, (0, 0), (WIN_W, 0), (80, 80, 80), 1)
    
    # FPS counters
    put_text(bar, f'Cam: {cam_fps:4.1f}', (12, 20), scale=0.42, color=(130, 220, 130))
    put_text(bar, f'Depth: {depth_fps:4.1f}', (130, 20), scale=0.42, color=(130, 180, 255))
    put_text(bar, f'Seg: {seg_fps:4.1f}', (260, 20), scale=0.42, color=(255, 180, 100))
    put_text(bar, f'ODM: {obs_fps:4.1f}', (380, 20), scale=0.42, color=(255, 130, 130))
    
    # Recording indicator
    if is_recording:
        cv2.circle(bar, (WIN_W - 280, 15), 6, (0, 0, 255), -1)
        put_text(bar, 'REC', (WIN_W - 270, 20), scale=0.42, color=(0, 0, 255))
    
    # Controls hint
    put_text(bar, 'Q:Quit | S:Shot | M:Mute | R:Record | F:Full', 
             (WIN_W - 320, 20), scale=0.38, color=(150, 150, 150))
    
    # Model labels
    cv2.line(bar, (0, STATUS_H // 2), (WIN_W, STATUS_H // 2), (50, 50, 50), 1)
    put_text(bar, 'Depth Anything V2 (vitb)', (12, STATUS_H - 8),
             scale=0.35, color=(90, 140, 200))
    put_text(bar, 'TopFormer-Base ADE20K', (280, STATUS_H - 8),
             scale=0.35, color=(180, 130, 90))
    put_text(bar, 'Enhanced ODM + PPM', (520, STATUS_H - 8),
             scale=0.35, color=(200, 90, 90))
    
    return bar


# ════════════════════════════════════════════════════════════════════════════
# Main Application
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main enhanced application entry point."""
    
    # ── Pre-flight checks ───────────────────────────────────────────────────
    if not os.path.exists(DEPTH_CKPT):
        print(f'ERROR: Depth checkpoint not found:\n  {DEPTH_CKPT}')
        sys.exit(1)
    if not os.path.exists(SEG_ONNX):
        print(f'ERROR: ONNX model not found:\n  {SEG_ONNX}')
        sys.exit(1)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)

    # ── Load Depth Anything V2 ──────────────────────────────────────────────
    print('[1/5] Loading Depth Anything V2 (vitb)...')
    sys.path.insert(0, DEPTH_SRC)
    from depth_anything_v2.dpt import DepthAnythingV2

    DEVICE = ('cuda' if torch.cuda.is_available() else
              'mps'  if torch.backends.mps.is_available() else 'cpu')
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
    print(f'      Depth model ready.')

    # ── Load TopFormer ONNX ────────────────────────────────────────────────
    print('[2/5] Loading TopFormer ONNX session...')
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if DEVICE == 'cuda' else ['CPUExecutionProvider'])
    seg_session = ort.InferenceSession(SEG_ONNX, providers=providers)

    # ── Initialize enhanced modules ─────────────────────────────────────────
    print('[3/5] Initializing enhanced modules...')
    
    # Obstacle tracker for temporal consistency
    obstacle_tracker = ObstacleTracker(max_tracks=20, max_history=10)
    
    # Trajectory history
    trajectory_history = TrajectoryHistory(max_history=20)
    
    # Enhanced audio
    audio = EnhancedAudioFeedback(cooldown=3.0, enabled=True)
    
    # Comfort scorer
    comfort_scorer = ComfortScorer()
    
    # Performance monitor
    perf_monitor = PerformanceMonitor(max_history=100)
    
    # Session recorder
    recorder = SessionRecorder(output_dir=RECORDING_DIR)

    # ── Open camera ─────────────────────────────────────────────────────────
    print('[4/5] Opening camera...')
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

    print('\n[5/5] System ready!')
    print('Controls: Q=Quit | S=Screenshot | M=Mute | R=Record | F=Fullscreen\n')

    # ── Shared state ────────────────────────────────────────────────────────
    shared = {
        'depth': np.zeros((actual_h, actual_w), dtype=np.float32),
        'depth_fps': 0.0,
        'seg': np.zeros((actual_h, actual_w), dtype=np.uint8),
        'seg_fps': 0.0,
    }
    lock = threading.Lock()
    stop_event = threading.Event()

    # ── Spawn workers ───────────────────────────────────────────────────────
    depth_worker = DepthWorker(depth_model, shared, lock, stop_event)
    seg_worker = SegWorker(seg_session, shared, lock, stop_event)
    depth_worker.start()
    seg_worker.start()

    # ── Window ──────────────────────────────────────────────────────────────
    win_name = 'Enhanced Navigation System (FYP)'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIN_W, WIN_H)

    # ── Main loop ───────────────────────────────────────────────────────────
    cam_fps = 0.0
    obs_fps = 0.0
    frame_count = 0
    t_fps_start = time.perf_counter()
    PUSH_EVERY = 2
    nav_instruction = ''
    is_fullscreen = False
    is_recording = False
    debug_info = {}

    # Temporal depth buffer
    depth_buffer = collections.deque(maxlen=5)
    
    # Previous frame for motion detection
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera read failed — retrying...')
            time.sleep(0.05)
            continue

        frame_count += 1

        # Push frames to workers
        if frame_count % PUSH_EVERY == 0:
            depth_worker.push_frame(frame)
            seg_worker.push_frame(frame)

        # Camera FPS
        elapsed = time.perf_counter() - t_fps_start
        if elapsed >= 0.5:
            cam_fps = frame_count / elapsed
            frame_count = 0
            t_fps_start = time.perf_counter()

        # Get inference results
        with lock:
            depth_np = shared['depth'].copy()
            seg_mask = shared['seg'].copy()
            depth_fps = shared['depth_fps']
            seg_fps = shared['seg_fps']

        # ── Temporal depth smoothing ──────────────────────────────────────
        depth_buffer.append(depth_np.copy())
        if len(depth_buffer) > 1:
            # Exponential moving average
            smoothed_depth = np.mean(depth_buffer, axis=0)
        else:
            smoothed_depth = depth_np
        
        # ── Enhanced obstacle detection ────────────────────────────────────
        t_obs = time.perf_counter()
        
        (obstacle_bgr, obstacle_mask, obstacle_info,
         obstacle_labels, debug_info) = detect_obstacles_enhanced(
            seg_mask, smoothed_depth, frame=frame,
            tracker=obstacle_tracker,
            use_temporal=True,
            use_edge_aware=True,
            use_confidence_weighted=True,
        )
        
        obs_fps = 1.0 / max(time.perf_counter() - t_obs, 1e-6)
        
        # ── Enhanced path planning ──────────────────────────────────────────
        nav_instruction, planner_details = plan_path_enhanced(
            seg_mask, smoothed_depth,
            obstacle_labels=obstacle_labels,
            trajectory=trajectory_history,
        )
        
        # Update trajectory history
        if obstacle_info:
            nearest = obstacle_info[0]
            cx, cy = nearest['centroid']
            trajectory_history.update(cx, cy)
        
        # ── Audio feedback ─────────────────────────────────────────────────
        direction = planner_details.get('direction', 0)
        
        if obstacle_info:
            nearest = obstacle_info[0]
            # Calculate distance (normalized disparity)
            max_d = smoothed_depth.max() + 1e-6
            distance = nearest['disparity'] / max_d
            
            audio.speak_obstacle_warning(
                nearest['class_name'],
                direction * 90,  # Convert to degrees
                distance,
            )
        
        audio.speak_navigation(
            nav_instruction,
            planner_details.get('action_type', 'MOVE_AHEAD'),
            direction,
        )
        
        # Record comfort score
        if obstacle_info:
            nearest = obstacle_info[0]
            level = nearest.get('hierarchy_level', 3)
            comfort_scorer.record_warning(level, 2.0)
        
        # ── Performance monitoring ─────────────────────────────────────────
        perf_monitor.update(
            {'camera': cam_fps, 'depth': depth_fps,
             'segmentation': seg_fps, 'fusion': obs_fps,
             'path_planning': obs_fps},
            {'total': 1.0/max(obs_fps, 1e-6),
             'depth': 1.0/max(depth_fps, 1e-6),
             'segmentation': 1.0/max(seg_fps, 1e-6),
             'fusion': 1.0/max(obs_fps, 1e-6),
             'path': 0.001}
        )
        
        # ── Build panels ───────────────────────────────────────────────────
        panel1 = build_camera_panel(frame, cam_fps)
        panel2 = build_depth_panel(smoothed_depth, depth_fps)
        panel3 = build_seg_panel(frame, seg_mask, seg_fps)
        panel4 = build_fusion_panel(obstacle_mask, smoothed_depth, obstacle_info)
        panel5 = build_obstacle_panel(obstacle_bgr, obstacle_info, nav_instruction, planner_details)
        panel6 = build_path_panel(planner_details.get('sector_bounds', {}),
                                  planner_details.get('ostatus', {}),
                                  planner_details)
        
        depth_conf = debug_info.get('depth_confidence')
        seg_conf = debug_info.get('seg_confidence')
        obs_scores = debug_info.get('obstacle_scores')
        panel7 = build_confidence_panel(depth_conf, seg_conf, obs_scores)
        panel8 = build_metrics_panel(perf_monitor)
        
        # ── Arrange grid ────────────────────────────────────────────────────
        row1 = np.hstack([panel1, panel2, panel3, panel4])
        row2 = np.hstack([panel5, panel6, panel7, panel8])
        grid = np.vstack([row1, row2])
        
        # Status bar
        status = build_status_bar(cam_fps, depth_fps, seg_fps, obs_fps, is_recording)
        
        # Final display
        display = np.vstack([grid, status])
        
        # ── Recording ──────────────────────────────────────────────────────
        if is_recording:
            recorder.write_frame(display)
        
        cv2.imshow(win_name, display)

        # ── Key handling ────────────────────────────────────────────────────
        key = cv2.waitKey(16) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if key in (ord('s'), ord('S')):
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(SCREENSHOT_DIR, f'enhanced_{ts}.png')
            cv2.imwrite(path, display)
            print(f'Screenshot saved: {path}')
        if key in (ord('m'), ord('M')):
            state = audio.toggle()
            print(f'Audio {"ON" if state else "OFF"}')
        if key in (ord('r'), ord('R')):
            if is_recording:
                recorder.stop_recording()
                is_recording = False
            else:
                recorder.start_recording(display.shape)
                is_recording = True
        if key in (ord('f'), ord('F')):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

    # ── Cleanup ─────────────────────────────────────────────────────────────
    print('\nShutting down...')
    stop_event.set()
    depth_worker.join(timeout=3)
    seg_worker.join(timeout=3)
    audio.shutdown()
    if is_recording:
        recorder.stop_recording()
    cap.release()
    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
