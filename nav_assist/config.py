"""
Centralized configuration for the Navigation Assistance System.
Paths, display dimensions, ADE20K class data, and model parameters.
"""

import os
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPTH_SRC = os.path.join(ROOT, 'obs-tackle', 'third_party', 'Depth-Anything-V2')
DEPTH_CKPT = os.path.join(ROOT, 'depth_anything_v2_vitb.pth')
SEG_ONNX = os.path.join(ROOT, 'topformer.onnx')
SCREENSHOT_DIR = os.path.join(ROOT, 'screenshots')

# ── Depth model configuration ─────────────────────────────────────────────
# Using DepthAnythingV2 ViT-Base (vitb) to match checkpoint file
DEPTH_ENCODER = 'vitb'
DEPTH_FEATURES = 128
DEPTH_OUT_CHANNELS = [96, 192, 384, 768]
DEPTH_INPUT_SIZE = 308    # smaller than 518 for faster CPU inference

# ── Display dimensions ─────────────────────────────────────────────────────
PANEL_W = 430          # width of each panel (4 panels now)
PANEL_H = 340          # height of each panel
STATUS_H = 60          # bottom status bar height
NUM_PANELS = 4         # camera, depth, segmentation, obstacles
WIN_W = PANEL_W * NUM_PANELS + (NUM_PANELS - 1) * 2  # 2px dividers
WIN_H = PANEL_H + STATUS_H

# ── Inference pacing ───────────────────────────────────────────────────────
# Cap both model workers to the same update cadence to reduce temporal skew
# between segmentation and depth during fusion.
MAX_MODEL_FPS = 5.0

# ── ADE20K normalisation (same as training) ────────────────────────────────
ADE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
ADE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SEG_SIZE = 512         # ONNX model fixed input size

# ── Obstacle detection parameters ──────────────────────────────────────────
DISPARITY_THRESHOLD_RATIO = 0.60   # 60% of max disparity (from paper)
MIN_COMPONENT_AREA = 100           # minimum pixel area for a component

# ── Alert zone geometry (Fig. 5) ───────────────────────────────────────────
ALERT_ZONE_PORTRAIT_X_MARGIN = 0.20   # 0.2W from each side in portrait
ALERT_ZONE_LANDSCAPE_X_MARGIN = 0.30  # 0.3W from each side in landscape
ALERT_ZONE_Y_FRACTION = 0.40          # bottom 40% of image height

# ── Path classes (ADE20K indices that represent walkable surfaces) ──────────
# These are discarded during obstacle detection per Algorithm 1
PATH_CLASS_NAMES = {'floor', 'road', 'sidewalk', 'path', 'runway',
                    'dirt track', 'grass', 'field', 'rug'}

# ── ADE20K class names (150 classes, 0-indexed) ───────────────────────────
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

# Build lookup: class name → index
ADE20K_CLASS_TO_IDX = {name: idx for idx, name in enumerate(ADE20K_CLASSES)}

# Build set of path-class indices for fast lookup
PATH_CLASS_INDICES = frozenset(
    ADE20K_CLASS_TO_IDX[name] for name in PATH_CLASS_NAMES
    if name in ADE20K_CLASS_TO_IDX
)

# ── ADE20K colour palette (RGB, 150 entries) ──────────────────────────────
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
    [255,  0,245],[255,  0,102],[255,173,  0],[255,  0, 20],[255,184,184],
    [  0, 31,255],[  0,255, 61],[  0, 71,255],[255,  0,204],[  0,255,194],
    [  0,255, 82],[  0, 10,255],[  0,112,255],[ 51,  0,255],[  0,194,255],
    [  0,122,255],[  0,255,163],[255,153,  0],[  0,255, 10],[255,112,  0],
    [143,255,  0],[ 82,  0,255],[163,255,  0],[255,235,  0],[  8,184,170],
    [133,  0,255],[  0,255, 92],[184,  0,255],[255,  0, 31],[  0,184,255],
    [  0,214,255],[255,  0,112],[ 92,255,  0],[  0,224,255],[112,224,255],
    [ 70,184,160],[163,  0,255],[153,  0,255],[ 71,255,  0],[255,  0,163],
    [255,204,  0],[255,  0,143],[  0,255,235],[133,255,  0],[255,  0,235],
    [245,  0,255],[255,  0,122],[255,245,  0],[ 10,190,212],[214,255,  0],
    [  0,204,255],[ 20,  0,255],[255,255,  0],[  0,153,255],[  0, 41,255],
    [  0,255,204],[ 41,  0,255],[ 41,255,  0],[173,  0,255],[  0,245,255],
    [ 71,  0,255],[122,  0,255],[  0,255,184],[  0, 92,255],[184,255,  0],
    [  0,133,255],[255,214,  0],[ 25,194,194],[102,255,  0],[ 92,  0,255],
], dtype=np.uint8)  # shape (150, 3) RGB
