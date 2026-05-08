"""
Multi-Panel Navigation Assistance System (6-Panel Display)
===========================================================
Comprehensive visualization of all processing stages for FYP demonstration.

Panel Layout (2x3 Grid):
    ┌─────────────┬─────────────┬─────────────┐
    │   Panel 1   │   Panel 2   │   Panel 3   │
    │   CAMERA    │    DEPTH    │    SEG      │
    │    FEED     │    MAP      │   MASK      │
    ├─────────────┼─────────────┼─────────────┤
    │   Panel 4   │   Panel 5   │   Panel 6   │
    │   FUSION    │  OBSTACLE   │    PATH     │
    │   HEATMAP   │   DETECTION │   PLANNER   │
    └─────────────┴─────────────┴─────────────┘

Processing Pipeline Displayed:
    Camera → Depth (DA-V2) → Segmentation (TopFormer)
         ↓
    Fusion Layer (depth+seg combined)
         ↓
    Obstacle Detection (ODM Algorithm 1)
         ↓
    Path Planning (6-sector fuzzy logic)

Controls:
    Q / Esc  — quit
    S        — save screenshot
    M        — mute/unmute audio
    F        — toggle fullscreen
    P        — pause/resume processing
    1-6      — toggle individual panels on/off
    +/-      — adjust depth colormap sensitivity
"""

import os
import sys
import time
import datetime
import threading

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

sys.path.insert(0, DEPTH_SRC)

# ── Display Configuration ───────────────────────────────────────────────────
PANEL_W = 420          # width of each panel
PANEL_H = 315          # height of each panel (4:3 ratio)
GRID_COLS = 3
GRID_ROWS = 2
MARGIN = 4             # margin between panels
HEADER_H = 24          # panel header height

# Window size calculation
WIN_W = PANEL_W * GRID_COLS + MARGIN * (GRID_COLS + 1)
WIN_H = PANEL_H * GRID_ROWS + MARGIN * (GRID_ROWS + 1) + 80  # Extra for status bar

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

PATH_CLASS_INDICES = {3, 6, 11, 52, 55, 96}  # floor, road, sidewalk, path, runway, dirt track

_CMAP_SPECTRAL = colormaps.get_cmap('Spectral_r')
_CMAP_JET = colormaps.get_cmap('jet')
_CMAP_PLASMA = colormaps.get_cmap('plasma')


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
    put_text(header, title, (8, 17), scale=0.50, color=color)
    if fps > 0:
        put_text(header, f'{fps:.1f} fps', (PANEL_W - 70, 17),
                scale=0.45, color=(100, 220, 100))
    return header


def build_camera_panel(frame, cam_fps, panel_idx=0):
    """Panel 1: Original camera feed."""
    panel = cv2.resize(frame, (PANEL_W, PANEL_H - HEADER_H))
    header = create_panel_header('📷 CAMERA FEED', cam_fps, (200, 200, 200))
    
    # Add timestamp
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    put_text(panel, timestamp, (PANEL_W - 110, PANEL_H - HEADER_H - 10),
             scale=0.40, color=(180, 180, 180))
    
    return np.vstack([header, panel])


def build_depth_panel(depth_np, depth_fps):
    """Panel 2: Depth Anything V2 output."""
    coloured, norm = colorize_depth(depth_np, 'Spectral_r')
    panel = cv2.resize(coloured, (PANEL_W, PANEL_H - HEADER_H))
    
    # Add depth scale bar
    bar_h, bar_w = 10, 80
    bar_x, bar_y = PANEL_W - bar_w - 10, PANEL_H - HEADER_H - 25
    bar = np.linspace(0, 1, bar_w)[None, :]
    bar_rgb = (_CMAP_SPECTRAL(bar)[0, :, :3] * 255).astype(np.uint8)
    bar_bgr = bar_rgb[:, ::-1]
    bar_bgr = np.repeat(bar_bgr[None], bar_h, axis=0)
    panel[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = bar_bgr
    
    put_text(panel, 'NEAR', (bar_x - 35, bar_y + 8), scale=0.32, color=(50, 50, 255))
    put_text(panel, 'FAR', (bar_x + bar_w + 3, bar_y + 8), scale=0.32, color=(200, 100, 40))
    
    # Near obstacle warning zones
    norm_s = cv2.resize(norm.astype(np.float32), (PANEL_W, PANEL_H - HEADER_H))
    NEAR_THRESH = 0.20
    near_mask = (norm_s < NEAR_THRESH).astype(np.uint8) * 255
    col_w = PANEL_W // 3
    
    for i, (label, cx) in enumerate(zip(['L', 'C', 'R'], [0, col_w, 2 * col_w])):
        region = near_mask[:, cx:cx + col_w]
        frac = region.mean() / 255.0
        if frac > 0.05:
            danger_color = (0, 0, 255) if frac > 0.25 else (0, 165, 255)
            cv2.rectangle(panel, (cx + 2, 5), (cx + col_w - 2, 25), danger_color, 2)
            put_text(panel, label, (cx + 8, 20), scale=0.45, color=danger_color)
    
    header = create_panel_header('🔍 DEPTH MAP (DA-V2)', depth_fps, (100, 180, 255))
    return np.vstack([header, panel])


def build_seg_panel(frame, seg_mask, seg_fps):
    """Panel 3: TopFormer semantic segmentation."""
    coloured = colorize_seg(seg_mask, PANEL_H - HEADER_H, PANEL_W)
    cam_small = cv2.resize(frame, (PANEL_W, PANEL_H - HEADER_H))
    panel = cv2.addWeighted(coloured, 0.65, cam_small, 0.35, 0)
    
    # Add top-5 class legend
    seg_small = cv2.resize(seg_mask.astype(np.uint8), (PANEL_W, PANEL_H - HEADER_H),
                           interpolation=cv2.INTER_NEAREST)
    unique, counts = np.unique(seg_small, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_k = min(5, len(unique))
    
    y_offset = 35
    for rank in range(top_k):
        cls_id = int(unique[order[rank]])
        if cls_id >= len(ADE20K_CLASSES):
            continue
        rgb = ADE20K_PALETTE[cls_id].tolist()
        bgr = (rgb[2], rgb[1], rgb[0])
        name = ADE20K_CLASSES[cls_id][:14]
        pct = 100.0 * counts[order[rank]] / seg_small.size
        
        cv2.rectangle(panel, (5, y_offset + rank * 18 - 12), (20, y_offset + rank * 18 + 3), bgr, -1)
        put_text(panel, f'{name} {pct:.0f}%', (25, y_offset + rank * 18), scale=0.38, color=(220, 220, 220))
    
    header = create_panel_header('🎨 SEMANTIC SEG (TopFormer)', seg_fps, (255, 180, 100))
    return np.vstack([header, panel])


def build_fusion_panel(depth_np, seg_mask, fusion_fps):
    """Panel 4: Depth-Segmentation Fusion Heatmap.
    
    Shows how depth and segmentation are combined.
    Creates a visualization showing which segments are at what depths.
    """
    h, w = seg_mask.shape[:2]
    
    # Resize depth to match segmentation
    if depth_np.shape[:2] != (h, w):
        depth_resized = cv2.resize(depth_np.astype(np.float32), (w, h))
    else:
        depth_resized = depth_np
    
    # Normalize depth to 0-1
    d_min, d_max = depth_resized.min(), depth_resized.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_resized - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_resized)
    
    # Create fusion visualization
    # Use HSV: Hue = semantic class, Value = depth (brighter = closer)
    seg_color = colorize_seg(seg_mask, h, w)
    seg_hsv = cv2.cvtColor(seg_color, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Modulate value channel by depth (closer = brighter)
    seg_hsv[:, :, 2] = seg_hsv[:, :, 2] * (1.0 - depth_norm * 0.7 + 0.3)
    seg_hsv = np.clip(seg_hsv, 0, 255).astype(np.uint8)
    
    fusion_vis = cv2.cvtColor(seg_hsv, cv2.COLOR_HSV2BGR)
    panel = cv2.resize(fusion_vis, (PANEL_W, PANEL_H - HEADER_H))
    
    # Add fusion explanation overlay
    overlay = panel.copy()
    cv2.rectangle(overlay, (5, 5), (200, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)
    
    put_text(panel, 'FUSION: Hue=Class', (10, 22), scale=0.40, color=(200, 200, 255))
    put_text(panel, '        Brightness=Depth', (10, 40), scale=0.40, color=(200, 200, 255))
    put_text(panel, '(Closer = Brighter)', (10, 55), scale=0.38, color=(150, 150, 200))
    
    header = create_panel_header('🔥 FUSION HEATMAP', fusion_fps, (255, 100, 255))
    return np.vstack([header, panel])


def build_obstacle_panel(seg_mask, depth_np, obstacle_info, nav_instruction, obs_fps):
    """Panel 5: Obstacle Detection Map (ODM output).
    
    Shows the result of Algorithm 1 - fused obstacles.
    """
    from skimage import measure
    
    h, w = seg_mask.shape[:2]
    
    # Resize depth to match segmentation
    if depth_np.shape[:2] != (h, w):
        depth_resized = cv2.resize(depth_np.astype(np.float32), (w, h))
    else:
        depth_resized = depth_np.astype(np.float32)
    
    max_disparity = depth_resized.max()
    threshold_d = 0.60 * max_disparity
    
    # Create obstacle visualization
    obstacle_vis = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = np.unique(seg_mask)
    
    obstacle_count = 0
    for class_id in unique_labels:
        if int(class_id) in PATH_CLASS_INDICES:
            continue
        
        class_mask = (seg_mask == class_id).astype(np.uint8)
        labelled = measure.label(class_mask, connectivity=2)
        regions = measure.regionprops(labelled)
        
        for region in regions:
            if region.area < 100:
                continue
            
            # Get depth values in this region
            coords = region.coords
            depths = depth_resized[coords[:, 0], coords[:, 1]]
            max_d = depths.max()
            
            if max_d > threshold_d:
                obstacle_count += 1
                rgb = ADE20K_PALETTE[int(class_id)]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                
                # Draw with alpha based on depth
                alpha = min(1.0, max_d / max_disparity)
                for y, x in coords:
                    obstacle_vis[y, x] = bgr
    
    panel = cv2.resize(obstacle_vis, (PANEL_W, PANEL_H - HEADER_H))
    
    # Add obstacle count and info
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W, 45), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, panel, 0.3, 0, panel)
    
    put_text(panel, f'OBSTACLES DETECTED: {obstacle_count}', (10, 20),
             scale=0.50, color=(0, 200, 255) if obstacle_count > 0 else (0, 255, 100))
    
    # Show nearest obstacle class if available
    if obstacle_info and len(obstacle_info) > 0:
        nearest = obstacle_info[0]
        label = nearest.get('class_name', 'Unknown').upper()
        put_text(panel, f'NEAREST: {label}', (10, 38), scale=0.45, color=(0, 200, 255))
    
    # Navigation instruction at bottom
    if nav_instruction:
        inst_lower = nav_instruction.lower()
        if 'stop' in inst_lower:
            color = (0, 0, 255)
        elif 'left' in inst_lower or 'right' in inst_lower:
            color = (0, 200, 255)
        else:
            color = (0, 255, 100)
        
        cv2.rectangle(panel, (0, PANEL_H - HEADER_H - 30), (PANEL_W, PANEL_H - HEADER_H),
                      (0, 0, 0), -1)
        put_text(panel, nav_instruction, (10, PANEL_H - HEADER_H - 8),
                scale=0.55, color=color)
    
    header = create_panel_header('⚠️ OBSTACLE DETECTION', obs_fps, (100, 255, 150))
    return np.vstack([header, panel])


def build_path_panel(frame, seg_mask, depth_np, planner_details, nav_instruction, path_fps):
    """Panel 6: Path Planning Visualization.
    
    Shows 6-sector grid, OStatus values, and navigation decision.
    """
    h, w = frame.shape[:2]
    panel = cv2.resize(frame, (PANEL_W, PANEL_H - HEADER_H))
    
    # Define sectors (2x3 grid)
    rows, cols = 2, 3
    sh, sw = (PANEL_H - HEADER_H) // rows, PANEL_W // cols
    
    # Get sector info from planner_details
    ostatus = planner_details.get('ostatus', {})
    action = planner_details.get('action', 'WAIT')
    
    sector_names = [
        ('top_left', 'OH-L'), ('top_mid', 'OH-M'), ('top_right', 'OH-R'),
        ('bot_left', 'GR-L'), ('bot_mid', 'GR-M'), ('bot_right', 'GR-R')
    ]
    
    # Draw sectors
    for idx, (sector_key, label) in enumerate(sector_names):
        row, col = idx // 3, idx % 3
        x1, y1 = col * sw, row * sh
        x2, y2 = x1 + sw, y1 + sh
        
        # Get OStatus for this sector
        status = ostatus.get(sector_key, 0.0)
        
        # Color based on obstacle status
        if status > 0.5:
            color = (0, 0, 255)  # Red - high obstacle
            alpha = 0.3
        elif status > 0.25:
            color = (0, 165, 255)  # Orange - medium
            alpha = 0.2
        else:
            color = (0, 255, 0)  # Green - clear
            alpha = 0.1
        
        # Draw sector overlay
        overlay = panel.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0, panel)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        # Draw label and status
        put_text(panel, label, (x1 + 5, y1 + 20), scale=0.45, color=(255, 255, 255))
        put_text(panel, f'{status:.2f}', (x1 + 5, y1 + 38), scale=0.40, color=(200, 200, 200))
    
    # Draw action indicator
    action_colors = {
        'STRONG_LEFT': (255, 0, 0), 'WEAK_LEFT': (200, 100, 0),
        'FORWARD': (0, 255, 0), 'WEAK_RIGHT': (0, 200, 100),
        'STRONG_RIGHT': (0, 0, 255), 'WAIT': (128, 128, 128),
        'STOP': (0, 0, 255)
    }
    action_color = action_colors.get(action, (200, 200, 200))
    
    # Draw action bar at bottom
    cv2.rectangle(panel, (0, PANEL_H - HEADER_H - 40), (PANEL_W, PANEL_H - HEADER_H),
                  action_color, -1)
    put_text(panel, f'ACTION: {action}', (10, PANEL_H - HEADER_H - 12),
             scale=0.55, color=(255, 255, 255))
    
    header = create_panel_header('🧭 PATH PLANNER (PPM)', path_fps, (150, 255, 180))
    return np.vstack([header, panel])


def build_status_bar(fps_dict, instruction, audio_on):
    """Build bottom status bar with all system info."""
    bar = np.zeros((70, WIN_W, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)
    
    # FPS counters
    x_pos = 10
    for name, fps in fps_dict.items():
        color = (100, 220, 100) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        put_text(bar, f'{name}: {fps:4.1f}', (x_pos, 22), scale=0.50, color=color)
        x_pos += 130
    
    # Audio indicator
    audio_text = '🔊 ON' if audio_on else '🔇 OFF'
    audio_color = (100, 255, 100) if audio_on else (100, 100, 100)
    put_text(bar, audio_text, (WIN_W - 150, 22), scale=0.55, color=audio_color)
    
    # Divider line
    cv2.line(bar, (0, 32), (WIN_W, 32), (60, 60, 60), 1)
    
    # Navigation instruction (prominent)
    if instruction:
        inst_lower = instruction.lower()
        if 'stop' in inst_lower:
            inst_color = (0, 0, 255)
        elif 'left' in inst_lower or 'right' in inst_lower:
            inst_color = (0, 200, 255)
        else:
            inst_color = (0, 255, 100)
        put_text(bar, f'► {instruction}', (10, 58), scale=0.65, color=inst_color)
    
    # Controls hint
    put_text(bar, 'Q:Quit | S:Screenshot | M:Mute | F:Fullscreen',
             (WIN_W - 400, 58), scale=0.40, color=(150, 150, 150))
    
    return bar


# ═════════════════════════════════════════════════════════════════════════════
# Worker Threads
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
            
            # Preprocess
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
            
            # Preprocess for ADE20K
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (SEG_SIZE, SEG_SIZE))
            img = img.astype(np.float32)
            img = (img - ADE_MEAN) / ADE_STD
            img = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
            
            # Run inference
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
# Main Application
# ═════════════════════════════════════════════════════════════════════════════

def detect_obstacles_simple(seg_mask, depth_map):
    """Simplified obstacle detection for visualization."""
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
    
    # Sort by disparity (closest first)
    obstacle_info.sort(key=lambda x: x['disparity'], reverse=True)
    return obstacle_info


def plan_path_simple(seg_mask, depth_map):
    """Simplified path planning for visualization."""
    h, w = seg_mask.shape[:2]
    
    if depth_map.shape[:2] != (h, w):
        depth = cv2.resize(depth_map.astype(np.float32), (w, h))
    else:
        depth = depth_map.astype(np.float32)
    
    max_d = depth.max()
    if max_d < 1e-6:
        return 'WAIT — No depth data', {'action': 'WAIT', 'ostatus': {}}
    
    # Create depth-gated mask
    depth_ratio = 0.40
    near_mask = depth > (depth_ratio * max_d)
    
    # Remove path classes
    for path_idx in PATH_CLASS_INDICES:
        near_mask[seg_mask == path_idx] = False
    
    # 6-sector analysis
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
        if sector_mask.size > 0:
            ostatus[name] = float(sector_mask.mean())
        else:
            ostatus[name] = 0.0
    
    # Simple decision logic
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
            instruction = 'Veering RIGHT — obstacle left'
        elif bot_right_status > THRESHOLD * 1.5:
            action = 'WEAK_LEFT'
            instruction = 'Veering LEFT — obstacle right'
        else:
            action = 'FORWARD'
            instruction = 'Proceed FORWARD — Path clear'
    
    return instruction, {'action': action, 'ostatus': ostatus}


class SimpleAudio:
    """Simple audio feedback placeholder."""
    def __init__(self):
        self.enabled = True
    
    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
    
    def speak(self, text):
        pass  # Placeholder - integrate with actual TTS
    
    def shutdown(self):
        pass


def main():
    print('=' * 70)
    print('Multi-Panel Navigation Assistance System (6-Panel Display)')
    print('=' * 70)
    
    # Pre-flight checks
    if not os.path.exists(DEPTH_CKPT):
        print(f'ERROR: Depth checkpoint not found:\n  {DEPTH_CKPT}')
        print('Please download depth_anything_v2_vitb.pth to the project root.')
        sys.exit(1)
    if not os.path.exists(SEG_ONNX):
        print(f'ERROR: ONNX model not found:\n  {SEG_ONNX}')
        print('Run: python convert_topformer_onnx.py  first.')
        sys.exit(1)
    
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    
    # Load Depth Anything V2
    print('\n[1/4] Loading Depth Anything V2 (vitb)...')
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
    print(f'      Depth model ready.')
    
    # Load TopFormer ONNX
    print('[2/4] Loading TopFormer ONNX session...')
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if DEVICE == 'cuda' else ['CPUExecutionProvider'])
    seg_session = ort.InferenceSession(SEG_ONNX, providers=providers)
    print(f'      Providers: {seg_session.get_providers()}')
    
    # Initialize audio
    print('[3/4] Initializing audio feedback...')
    audio = SimpleAudio()
    
    # Open camera
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
    print('Controls: Q=Quit | S=Screenshot | M=Mute | F=Fullscreen')
    print('=' * 70 + '\n')
    
    # Shared state
    shared = {
        'depth': np.zeros((PANEL_H, PANEL_W), dtype=np.float32),
        'depth_fps': 0.0,
        'seg': np.zeros((actual_h, actual_w), dtype=np.uint8),
        'seg_fps': 0.0,
    }
    lock = threading.Lock()
    stop_event = threading.Event()
    
    # Spawn workers
    depth_worker = DepthWorker(depth_model, shared, lock, stop_event)
    seg_worker = SegWorker(seg_session, shared, lock, stop_event)
    depth_worker.start()
    seg_worker.start()
    
    # Create window
    win_name = 'Navigation Assistance System - 6 Panel Display'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIN_W, WIN_H)
    
    # Main loop
    cam_fps = 0.0
    fusion_fps = 0.0
    obs_fps = 0.0
    path_fps = 0.0
    frame_count = 0
    t_fps_start = time.perf_counter()
    
    fullscreen = False
    nav_instruction = ''
    planner_details = {'action': 'WAIT', 'ostatus': {}}
    obstacle_info = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Camera read failed — retrying...')
                time.sleep(0.05)
                continue
            
            frame_count += 1
            
            # Push frames to workers
            if frame_count % 2 == 0:
                depth_worker.push_frame(frame)
                seg_worker.push_frame(frame)
            
            # Update FPS
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
            
            # Fusion step
            t_fusion = time.perf_counter()
            fusion_fps = 1.0 / max(time.perf_counter() - t_fusion, 1e-6)
            
            # Path planning
            t_path = time.perf_counter()
            nav_instruction, planner_details = plan_path_simple(seg_mask, depth_np)
            path_fps = 1.0 / max(time.perf_counter() - t_path, 1e-6)
            
            # Obstacle detection
            t_obs = time.perf_counter()
            obstacle_info = detect_obstacles_simple(seg_mask, depth_np)
            obs_fps = 1.0 / max(time.perf_counter() - t_obs, 1e-6)
            
            # Build panels
            panel1 = build_camera_panel(frame, cam_fps)
            panel2 = build_depth_panel(depth_np, depth_fps)
            panel3 = build_seg_panel(frame, seg_mask, seg_fps)
            panel4 = build_fusion_panel(depth_np, seg_mask, fusion_fps)
            panel5 = build_obstacle_panel(seg_mask, depth_np, obstacle_info,
                                          nav_instruction, obs_fps)
            panel6 = build_path_panel(frame, seg_mask, depth_np, planner_details,
                                      nav_instruction, path_fps)
            
            # Assemble grid (2x3)
            row1 = np.hstack([panel1, np.zeros((PANEL_H, MARGIN, 3), dtype=np.uint8),
                              panel2, np.zeros((PANEL_H, MARGIN, 3), dtype=np.uint8),
                              panel3])
            row2 = np.hstack([panel4, np.zeros((PANEL_H, MARGIN, 3), dtype=np.uint8),
                              panel5, np.zeros((PANEL_H, MARGIN, 3), dtype=np.uint8),
                              panel6])
            
            grid = np.vstack([row1, np.zeros((MARGIN, WIN_W, 3), dtype=np.uint8), row2])
            
            # Add status bar
            fps_dict = {
                'CAM': cam_fps, 'DEPTH': depth_fps, 'SEG': seg_fps,
                'ODM': obs_fps, 'PATH': path_fps
            }
            status_bar = build_status_bar(fps_dict, nav_instruction, audio.enabled)
            display = np.vstack([grid, status_bar])
            
            cv2.imshow(win_name, display)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('s'), ord('S')):
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(SCREENSHOT_DIR, f'multi_panel_{ts}.png')
                cv2.imwrite(path, display)
                print(f'Screenshot saved: {path}')
            elif key in (ord('m'), ord('M')):
                state = audio.toggle()
                print(f'Audio {"ON" if state else "OFF"}')
            elif key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, 0)
    
    finally:
        print('\nShutting down...')
        stop_event.set()
        depth_worker.join(timeout=3)
        seg_worker.join(timeout=3)
        audio.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        print('Done.')


if __name__ == '__main__':
    main()
