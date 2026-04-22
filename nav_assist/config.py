"""
Centralized configuration for the Navigation Assistance System.
Paths, display dimensions, ADE20K class data, and model parameters.

Enhanced version with:
- Confidence-weighted fusion
- Temporal consistency (Kalman filtering)
- Variable grid resolution path planning
- Spatial audio feedback
- Multi-scale processing
- Environmental adaptation
"""

import os
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPTH_SRC = os.path.join(ROOT, 'obs-tackle', 'third_party', 'Depth-Anything-V2')
DEPTH_CKPT = os.path.join(ROOT, 'depth_anything_vits14.pth')
SEG_ONNX = os.path.join(ROOT, 'topformer.onnx')
SCREENSHOT_DIR     = os.path.join(ROOT, 'screenshots')
GEMMA_SNAPSHOT_DIR = os.path.join(ROOT, 'gemma_snapshots')
CLIP_DIR           = os.path.join(ROOT, 'video_clips')
CLIP_MAX_SECONDS   = 30.0
CLIP_FPS           = 15.0
RECORDING_DIR = os.path.join(ROOT, 'recordings')
CONFIG_DIR = os.path.join(ROOT, 'config')

# ── System Settings ────────────────────────────────────────────────────────
ENABLE_TEMPORAL_SMOOTHING = True  # Kalman filter for depth
TEMPORAL_ALPHA = 0.7  # Blend factor for temporal smoothing (0-1)
ENABLE_CONFIDENCE_WEIGHTING = True  # Confidence-weighted fusion
ENABLE_EDGE_AWARENESS = True  # Canny edge refinement
ENABLE_MULTI_SCALE = False  # Multi-scale processing (experimental)
ENABLE_MOTION_DETECTION = True  # Track dynamic obstacles
ENABLE_METRIC_DEPTH = False  # Convert to metric depth (requires calibration)

# ── Model Failure Detection ────────────────────────────────────────────────
MODEL_HEALTH_CHECK = True  # Monitor model outputs for failures
GRACEFUL_DEGRADATION = True  # Fall back to simpler algorithms on failure
SAFETY_MODE_THRESHOLD = 0.3  # Be extra cautious when confidence is low

# ── Depth model configuration ─────────────────────────────────────────────
# Using DepthAnythingV2 ViT-Small (vits) to match checkpoint file
DEPTH_ENCODER = 'vits'
DEPTH_FEATURES = 64
DEPTH_OUT_CHANNELS = [48, 96, 192, 384]
DEPTH_INPUT_SIZE = 308    # smaller than 518 for faster CPU inference

# ── Display dimensions ─────────────────────────────────────────────────────
PANEL_W = 430          # width of each panel (4 panels now)
PANEL_H = 340          # height of each panel
STATUS_H = 60          # bottom status bar height
NUM_PANELS = 4         # camera, depth, segmentation, obstacles
WIN_W = PANEL_W * NUM_PANELS + (NUM_PANELS - 1) * 2  # 2px dividers
WIN_H = PANEL_H + STATUS_H

# ── 8-Panel Display Configuration ───────────────────────────────────────────
PANEL_8_W = 320        # width for 8-panel mode
PANEL_8_H = 240        # height for 8-panel mode
GRID_8_COLS = 4        # 4 columns in 8-panel grid
GRID_8_ROWS = 2         # 2 rows in 8-panel grid
WIN_8_W = PANEL_8_W * GRID_8_COLS + (GRID_8_COLS + 1) * 3
WIN_8_H = PANEL_8_H * GRID_8_ROWS + (GRID_8_ROWS + 1) * 3 + STATUS_H

# ── Visualization Settings ───────────────────────────────────────────────────
SHOW_DEPTH_HISTOGRAM = True         # Show depth distribution
SHOW_TRAJECTORY = True               # Draw predicted path
SHOW_HAZARD_HEATMAP = True           # Gradient showing danger levels
SHOW_CLASS_ACTIVATION = False        # CAM visualization (slow)
SHOW_AR_OVERLAY = True               # AR-style navigation arrows
SHOW_CONFIDENCE_OVERLAY = True       # Model uncertainty visualization
SHOW_OPTICAL_FLOW = False            # Motion flow visualization

# ── Recording Settings ────────────────────────────────────────────────────────
ENABLE_RECORDING = True               # Save session recordings
RECORD_FPS = 15                       # Recording frame rate
RECORD_CODEC = 'mp4v'                 # Video codec
RECORD_QUALITY = 85                   # JPEG quality for screenshots

# ── Performance Monitoring ────────────────────────────────────────────────────
ENABLE_PERF_STATS = True              # Track performance metrics
PERF_HISTORY_SIZE = 100               # Number of frames to track
SHOW_LATENCY_BREAKDOWN = True         # Show per-component latency
SHOW_MEMORY_USAGE = True              # Track memory consumption
SHOW_GPU_UTILIZATION = True           # Track GPU usage (if available)

# ── Metrics Collection ───────────────────────────────────────────────────────
COLLECT_ACCURACY_METRICS = True       # Track detection accuracy
COLLECT_USER_COMFORT = False           # User comfort scoring (requires input)
METRICS_LOG_INTERVAL = 10             # Log metrics every N frames

# ── PPM Sector Grid — Full Coverage (No Blind Spots) ──────────────────────
# Overhead (Sectors 1,2,3): top of frame → 0.5H
PPM_OVERHEAD_TOP = 0.0
PPM_OVERHEAD_BOT = 0.50
# Ground (Sectors 4,5,6): 0.5H → bottom of frame
PPM_GROUND_TOP = 0.50

# Horizontal column boundaries (landscape orientation, Fig. 5)
PPM_COL_LEFT = 0.30        # Left column: 0 → 0.3W
PPM_COL_RIGHT = 0.70       # Right column: 0.7W → W

# ── Enhanced Path Planning (Variable Grid Resolution) ──────────────────────
PPM_VARIABLE_GRID = True       # Finer grid in center
PPM_CENTER_COL_LEFT = 0.40     # Finer center column: 0.4W → 0.6W
PPM_CENTER_COL_RIGHT = 0.60
PPM_CENTER_ROW = 0.50          # Center row split

# ── Trajectory Prediction ────────────────────────────────────────────────────
TRAJECTORY_PREDICTION_ENABLED = True
TRAJECTORY_LOOKAHEAD_FRAMES = 5    # Predict 2-3 seconds ahead
TRAJECTORY_SMOOTHING = 0.8          # Exponential smoothing factor

# ── Alternative Path Generation ──────────────────────────────────────────────
ALTERNATIVE_PATHS_ENABLED = True
MAX_ALTERNATIVE_PATHS = 2           # Generate 2 alternative routes
PATH_RANK_BY_SAFETY = True         # Rank paths by obstacle density

# ── Depth gating (3-metre alert radius) ────────────────────────────────────
# Approximate 3m in normalised disparity (higher = nearer).
# Pixels with disparity < DEPTH_3M_RATIO * max_disparity are discarded.
DEPTH_3M_RATIO = 0.40

# ── Multi-scale processing ───────────────────────────────────────────────────
MULTI_SCALE_SCALES = [1.0, 0.5, 0.25]  # Full, half, quarter resolution
MULTI_SCALE_FUSION = 'attention'        # How to fuse multi-scale results

# ── Environmental Adaptation ─────────────────────────────────────────────────
LOW_LIGHT_DETECTION = True            # Detect low light conditions
LOW_LIGHT_GAIN = 1.5                   # Boost depth in low light
BRIGHT_LIGHT_DETECTION = True         # Detect bright sunlight
GLARE_DETECTION = True                # Detect glare/reflections

# ── Motion Detection ─────────────────────────────────────────────────────────
MOTION_DETECTION_ENABLED = True
MOTION_THRESHOLD = 10                  # Pixel difference threshold
MOTION_MIN_FRAMES = 3                  # Minimum frames for motion detection

# ── Depth gating (3-metre alert radius) ───────────────────────────────────
# Approximate 3m in normalised disparity (higher = nearer).
# Pixels with disparity < DEPTH_3M_RATIO * max_disparity are discarded.
DEPTH_3M_RATIO = 0.40

# OStatus visual threshold (fuzzy crossover point of new MFs)
PPM_OSTATUS_THRESHOLD = 0.25

# ── Display: overlay mode ──────────────────────────────────────────────────
INSTRUCTION_BAR_H = 60     # Height of bottom instruction bar (px)

# ── ADE20K normalisation (same as training) ────────────────────────────────
ADE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
ADE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SEG_SIZE = 512         # ONNX model fixed input size

# ── Obstacle detection parameters ──────────────────────────────────────────
DISPARITY_THRESHOLD_RATIO = 0.60   # 60% of max disparity (from paper)
MIN_COMPONENT_AREA = 100           # minimum pixel area for a component

# ── Confidence-weighted fusion parameters ───────────────────────────────────
CONFIDENCE_ALPHA = 0.6  # Weight for depth confidence
CONFIDENCE_BETA = 0.4   # Weight for segmentation confidence
EDGE_SIGMA = 2.0        # Gaussian blur sigma for edge detection
EDGE_LOW_THRESH = 50    # Canny edge low threshold
EDGE_HIGH_THRESH = 150  # Canny edge high threshold

# ── Obstacle Hierarchy (Priority Levels) ────────────────────────────────────
OBSTACLE_HIERARCHY = {
    # Level 1: Critical (immediate danger)
    'critical': {'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'},
    # Level 2: Caution (potential hazard)
    'caution': {'wall', 'building', 'fence', 'pole', 'tree', 'stairs', 
                'step', 'stairs', 'escalator', 'railing'},
    # Level 3: Navigable (minor obstacles)
    'navigable': {'chair', 'table', 'cabinet', 'box', 'bag', 'stool'},
}

# ── Audio Feedback Configuration ─────────────────────────────────────────────
AUDIO_SPATIAL_ENABLED = True       # Enable stereo positioning
AUDIO_DISTANCE_VOLUME = True       # Closer obstacles = louder
AUDIO_ICON_ENABLED = True         # Distinct sounds for obstacle types
AUDIO_RHYTHM_ENABLED = True       # Beep frequency indicates urgency
AUDIO_SPATIAL_PAN_RANGE = 0.7      # How much to pan left/right (-0.7 to 0.7)
AUDIO_MIN_VOLUME = 0.3             # Minimum volume for distant obstacles
AUDIO_MAX_VOLUME = 1.0             # Maximum volume for close obstacles
AUDIO_SPEECH_RATE = 160            # TTS speech rate

# ── Audio icon frequencies (Hz) ─────────────────────────────────────────────
AUDIO_ICON_FREQUENCIES = {
    'wall': 200,          # Low beep - solid obstacle
    'person': 440,        # Gentle chime - living being
    'pole': 880,          # Sharp click - narrow obstacle
    'step': 600,          # Rising tone - height change
    'chair': 350,         # Medium tone - furniture
    'default': 500,       # Default tone
}

# ── Rhythm-based guidance (beeps per second based on distance) ───────────────
RHYTHM_BEEP_RATES = {
    'very_close': 4,    # < 1m - rapid beeps
    'close': 2,         # 1-2m - moderate beeps
    'medium': 1,        # 2-3m - slow beeps
    'far': 0.5,         # > 3m - occasional beeps
}

# ── Temporal Filtering (Kalman) ─────────────────────────────────────────────
KALMAN_PROCESS_NOISE = 0.01   # Q (uncertainty in motion model)
KALMAN_MEASURE_NOISE = 0.1    # R (measurement uncertainty)
KALMAN_SMOOTHING_FRAMES = 5   # Number of frames for exponential moving average

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

# ── Gemma 4 Multimodal Assistant ──────────────────────────────────────────
GEMMA_MODEL_PATH  = os.path.join(ROOT, 'gemma-4', 'gemma-4-E2B-it-Q3_K_M.gguf')
GEMMA_MMPROJ_PATH = os.path.join(ROOT, 'gemma-4', 'mmproj-F16.gguf')
GEMMA_N_GPU_LAYERS = -1     # full GPU offload; drop to 20 if OOM
GEMMA_CTX_SIZE    = 2048
GEMMA_IMAGE_SIZE  = 448     # Gemma 4 native patch resolution

GEMMA_RESPONSE_DISPLAY_SECS = 15.0
GEMMA_HINT_BAR_EXTRA_H      = 22   # px appended below INSTRUCTION_BAR_H

PIPER_VOICE_ONNX = os.path.join(ROOT, 'piper_voices', 'en_US-lessac-high.onnx')
PIPER_VOICE_JSON = os.path.join(ROOT, 'piper_voices', 'en_US-lessac-high.onnx.json')
