"""
Real-Time Navigation Assistance System for Visually Impaired Individuals
========================================================================
Dual-stream pipeline:
  Stream A — Depth Anything V2 (PyTorch, vits)  → depth map
  Stream B — TopFormer-Base   (ONNX runtime)    → ADE20K segmentation

Display: [Camera Feed | Depth Map | Segmentation Mask] side-by-side

Controls:
  Q / Esc  — quit
  S        — save screenshot to screenshots/
  +/-      — increase/decrease display brightness
"""

import os, sys, time, threading, queue, datetime
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort
import matplotlib

matplotlib.use('Agg')                     # headless backend — we use cv2 for display
import matplotlib.pyplot as plt
from matplotlib import colormaps

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
DEPTH_SRC  = os.path.join(ROOT, 'obs-tackle', 'third_party', 'Depth-Anything-V2')
DEPTH_CKPT = os.path.join(ROOT, 'depth_anything_v2_vits.pth')
SEG_ONNX   = os.path.join(ROOT, 'topformer.onnx')
SCREENSHOT_DIR = os.path.join(ROOT, 'screenshots')

sys.path.insert(0, DEPTH_SRC)

# ── Display dimensions ─────────────────────────────────────────────────────
PANEL_W = 540          # width of each of the 3 panels
PANEL_H = 405          # height of each panel  (4:3 ratio)
STATUS_H = 60          # bottom status bar height
WIN_W  = PANEL_W * 3 + 4  # 2px divider between each panel pair
WIN_H  = PANEL_H + STATUS_H

# ── ADE20K normalisation (same as training) ────────────────────────────────
ADE_MEAN = np.array([123.675, 116.28,  103.53 ], dtype=np.float32)
ADE_STD  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)
SEG_SIZE = 512          # ONNX model fixed input size

# ════════════════════════════════════════════════════════════════════════════
# ADE20K class names (150 classes)
# ════════════════════════════════════════════════════════════════════════════
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

# ── ADE20K colour palette (RGB) ────────────────────────────────────────────
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
], dtype=np.uint8)  # shape (150, 3)  RGB


# ════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ════════════════════════════════════════════════════════════════════════════

_CMAP_SPECTRAL = colormaps.get_cmap('Spectral_r')

def put_text(img, text, pos, scale=0.55, color=(255,255,255),
             thickness=1, bg_color=(0,0,0)):
    """Draw text with a semi-transparent shadow for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), font,
                scale, bg_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def put_label_box(img, text, x, y, color_bgr, font_scale=0.45):
    """Coloured legend box + label."""
    cv2.rectangle(img, (x, y), (x + 14, y + 14), color_bgr, -1)
    cv2.rectangle(img, (x, y), (x + 14, y + 14), (50, 50, 50), 1)
    put_text(img, text, (x + 18, y + 11), scale=font_scale,
             color=(230, 230, 230), bg_color=(20, 20, 20))


def colorize_depth(depth_np):
    """
    Convert raw depth array → coloured BGR uint8 image (Spectral_r).
    Returns coloured image + normalised [0,1] depth map.
    """
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth_np)
    else:
        norm = (depth_np - d_min) / (d_max - d_min)

    coloured = (_CMAP_SPECTRAL(norm)[:, :, :3] * 255).astype(np.uint8)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_RGB2BGR)
    return coloured, norm


def colorize_seg(seg_mask, h, w):
    """
    Convert (H, W) class-index array → coloured BGR uint8 image.
    seg_mask values are 0..149.
    """
    colour_map = ADE20K_PALETTE[seg_mask]               # (H, W, 3) RGB
    colour_bgr = colour_map[..., ::-1].astype(np.uint8) # RGB → BGR
    colour_bgr = cv2.resize(colour_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    return colour_bgr


# ════════════════════════════════════════════════════════════════════════════
# Inference workers
# ════════════════════════════════════════════════════════════════════════════

class DepthWorker(threading.Thread):
    """
    Runs Depth Anything V2 (vits, PyTorch) on the latest available frame.
    Writes results into `shared` dict under lock.
    """
    def __init__(self, model, shared, lock, stop_event):
        super().__init__(daemon=True)
        self.model       = model
        self.shared      = shared
        self.lock        = lock
        self.stop_event  = stop_event
        self._frame_ready = threading.Event()
        self._next_frame  = None
        self._frame_lock  = threading.Lock()

    def push_frame(self, frame):
        with self._frame_lock:
            self._next_frame = frame.copy()
        self._frame_ready.set()

    def run(self):
        while not self.stop_event.is_set():
            got = self._frame_ready.wait(timeout=0.5)
            if not got:
                continue
            self._frame_ready.clear()
            with self._frame_lock:
                frame = self._next_frame
            if frame is None:
                continue
            t0 = time.perf_counter()
            try:
                depth = self.model.infer_image(frame, input_size=518)
            except Exception as e:
                print(f'[DepthWorker] ERROR: {e}')
                continue
            elapsed = time.perf_counter() - t0
            with self.lock:
                self.shared['depth']     = depth
                self.shared['depth_fps'] = 1.0 / elapsed


class SegWorker(threading.Thread):
    """
    Runs TopFormer (ONNX) on the latest available frame.
    Writes results into `shared` dict under lock.
    """
    def __init__(self, session, shared, lock, stop_event):
        super().__init__(daemon=True)
        self.session     = session
        self.shared      = shared
        self.lock        = lock
        self.stop_event  = stop_event
        self._frame_ready = threading.Event()
        self._next_frame  = None
        self._frame_lock  = threading.Lock()

    def push_frame(self, frame):
        with self._frame_lock:
            self._next_frame = frame.copy()
        self._frame_ready.set()

    def _preprocess(self, bgr_frame):
        """BGR uint8 → normalised float32 NCHW tensor (ADE20K convention)."""
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb  = cv2.resize(rgb, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_LINEAR)
        norm = (rgb - ADE_MEAN) / ADE_STD                  # (512, 512, 3)
        nchw = norm.transpose(2, 0, 1)[None]               # (1, 3, 512, 512)
        return nchw.astype(np.float32)

    def run(self):
        while not self.stop_event.is_set():
            got = self._frame_ready.wait(timeout=0.5)
            if not got:
                continue
            self._frame_ready.clear()
            with self._frame_lock:
                frame = self._next_frame
            if frame is None:
                continue
            t0 = time.perf_counter()
            try:
                inp    = self._preprocess(frame)
                logits = self.session.run(None, {'input': inp})[0]  # (1,150,H',W')
                # Upsample to frame size
                h, w   = frame.shape[:2]
                logits_t = torch.from_numpy(logits)
                logits_up = F.interpolate(logits_t, size=(h, w),
                                          mode='bilinear', align_corners=False)
                seg_mask = logits_up[0].argmax(dim=0).numpy().astype(np.uint8)
            except Exception as e:
                print(f'[SegWorker] ERROR: {e}')
                continue
            elapsed = time.perf_counter() - t0
            with self.lock:
                self.shared['seg']     = seg_mask
                self.shared['seg_fps'] = 1.0 / elapsed


# ════════════════════════════════════════════════════════════════════════════
# Panel builders
# ════════════════════════════════════════════════════════════════════════════

def build_camera_panel(frame, cam_fps):
    """Original camera feed with FPS overlay."""
    panel = cv2.resize(frame, (PANEL_W, PANEL_H))
    # Header bar
    cv2.rectangle(panel, (0, 0), (PANEL_W, 28), (30, 30, 30), -1)
    put_text(panel, 'CAMERA FEED', (8, 18), scale=0.55,
             color=(200, 200, 200), bg_color=(0, 0, 0))
    put_text(panel, f'{cam_fps:.1f} fps', (PANEL_W - 75, 18),
             scale=0.50, color=(100, 220, 100))
    return panel


def build_depth_panel(frame, depth_np, depth_fps):
    """
    Depth Anything V2 output.
    Colour: Spectral_r  (red = NEAR, blue = FAR)
    Overlays: depth colour bar, near-obstacle warning zones.
    """
    h, w = frame.shape[:2]

    # ── Colorise ────────────────────────────────────────────────────────────
    coloured, norm = colorize_depth(depth_np)
    panel = cv2.resize(coloured, (PANEL_W, PANEL_H))
    norm_s = cv2.resize(norm.astype(np.float32), (PANEL_W, PANEL_H))

    # ── Header ──────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (PANEL_W, 28), (30, 30, 30), -1)
    put_text(panel, 'DEPTH MAP  (red=NEAR  blue=FAR)', (8, 18),
             scale=0.50, color=(200, 200, 200))
    put_text(panel, f'{depth_fps:.1f} fps', (PANEL_W - 75, 18),
             scale=0.50, color=(100, 220, 100))

    # ── Near-obstacle zones (split frame into left / centre / right) ────────
    # In Spectral_r: low norm value = near (warm red); high = far (cool blue).
    # We highlight regions where norm < 0.25 (closest quartile).
    NEAR_THRESH = 0.20
    near_mask = (norm_s < NEAR_THRESH).astype(np.uint8) * 255
    # Divide into 3 horizontal columns
    col_w = PANEL_W // 3
    labels, positions = ['LEFT', 'CENTRE', 'RIGHT'], [0, col_w, 2 * col_w]
    for col_i, (label, cx) in enumerate(zip(labels, positions)):
        region = near_mask[:, cx: cx + col_w]
        frac   = region.mean() / 255.0
        if frac > 0.08:          # > 8 % of column is "near"
            danger_colour = (0, 0, 200) if frac > 0.25 else (0, 100, 220)
            overlay = panel.copy()
            cv2.rectangle(overlay, (cx + 2, PANEL_H - 50),
                          (cx + col_w - 2, PANEL_H - 2), danger_colour, 2)
            cv2.addWeighted(overlay, 0.4, panel, 0.6, 0, panel)
            put_text(panel, f'OBS {label}',
                     (cx + 5, PANEL_H - 10), scale=0.45,
                     color=(50, 50, 255), bg_color=(0, 0, 0))

    # ── Colour scale bar ────────────────────────────────────────────────────
    bar_h, bar_w = 12, 120
    bar_x, bar_y = PANEL_W - bar_w - 10, PANEL_H - 22
    bar = np.linspace(0, 1, bar_w)[None, :]       # (1, bar_w)
    bar_rgb = (_CMAP_SPECTRAL(bar)[0, :, :3] * 255).astype(np.uint8)
    bar_bgr = bar_rgb[:, ::-1]                    # RGB→BGR
    bar_bgr = np.repeat(bar_bgr[None], bar_h, axis=0)  # (bar_h, bar_w, 3)
    panel[bar_y: bar_y + bar_h, bar_x: bar_x + bar_w] = bar_bgr
    put_text(panel, 'NEAR', (bar_x - 38, bar_y + 9), scale=0.38,
             color=(50, 50, 255))
    put_text(panel, 'FAR', (bar_x + bar_w + 3, bar_y + 9), scale=0.38,
             color=(200, 100, 40))

    return panel


def build_seg_panel(frame, seg_mask, seg_fps):
    """
    TopFormer segmentation output.
    Coloured ADE20K mask blended with camera frame.
    Legend shows top-5 most-present classes.
    """
    h, w = frame.shape[:2]
    coloured = colorize_seg(seg_mask, PANEL_H, PANEL_W)

    # Blend with resized camera for context
    cam_small = cv2.resize(frame, (PANEL_W, PANEL_H))
    panel = cv2.addWeighted(coloured, 0.65, cam_small, 0.35, 0)

    # ── Header ──────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (PANEL_W, 28), (30, 30, 30), -1)
    put_text(panel, 'SEMANTIC SEG (ADE20K · 150 cls)', (8, 18),
             scale=0.50, color=(200, 200, 200))
    put_text(panel, f'{seg_fps:.1f} fps', (PANEL_W - 75, 18),
             scale=0.50, color=(100, 220, 100))

    # ── Legend: top-5 classes ───────────────────────────────────────────────
    # seg_mask is at original resolution; resize to panel for pixel counts
    seg_small = cv2.resize(
        seg_mask.astype(np.uint8), (PANEL_W, PANEL_H),
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
        pct  = 100.0 * counts[order[rank]] / seg_small.size
        put_label_box(panel, f'{name} {pct:.0f}%',
                      leg_x, leg_y + rank * 18, bgr)
    return panel


def build_status_bar(cam_fps, depth_fps, seg_fps):
    """Bottom status bar with FPS counters and key hints."""
    bar = np.zeros((STATUS_H, WIN_W, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    cv2.line(bar, (0, 0), (WIN_W, 0), (80, 80, 80), 1)

    put_text(bar, f'Camera: {cam_fps:5.1f} fps', (12, 22),
             scale=0.52, color=(130, 220, 130))
    put_text(bar, f'Depth (DAv2-vits): {depth_fps:5.1f} fps', (190, 22),
             scale=0.52, color=(130, 180, 255))
    put_text(bar, f'Seg (TopFormer-B ONNX): {seg_fps:5.1f} fps', (480, 22),
             scale=0.52, color=(255, 180, 100))
    put_text(bar, 'Q/Esc: Quit  |  S: Screenshot', (WIN_W - 280, 22),
             scale=0.48, color=(160, 160, 160))

    # Model labels row
    cv2.line(bar, (0, STATUS_H // 2), (WIN_W, STATUS_H // 2), (50, 50, 50), 1)
    put_text(bar, 'Depth Anything V2 (vits · PyTorch)', (12, STATUS_H - 8),
             scale=0.42, color=(90, 140, 200))
    put_text(bar, 'TopFormer-Base · ADE20K · onnxruntime', (400, STATUS_H - 8),
             scale=0.42, color=(180, 130, 90))

    return bar


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── Pre-flight checks ───────────────────────────────────────────────────
    if not os.path.exists(DEPTH_CKPT):
        print(f'ERROR: Depth checkpoint not found:\n  {DEPTH_CKPT}')
        sys.exit(1)
    if not os.path.exists(SEG_ONNX):
        print(f'ERROR: ONNX model not found:\n  {SEG_ONNX}')
        print('Run  python convert_topformer_onnx.py  first.')
        sys.exit(1)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # ── Load Depth Anything V2 ──────────────────────────────────────────────
    print('[1/3] Loading Depth Anything V2 (vits)...')
    from depth_anything_v2.dpt import DepthAnythingV2
    DEVICE = ('cuda' if torch.cuda.is_available() else
              'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'      Device: {DEVICE}')

    depth_model = DepthAnythingV2(
        encoder='vits', features=64,
        out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(
        torch.load(DEPTH_CKPT, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()
    print('      Depth model ready.')

    # ── Load TopFormer ONNX ─────────────────────────────────────────────────
    print('[2/3] Loading TopFormer ONNX session...')
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if DEVICE == 'cuda' else ['CPUExecutionProvider'])
    seg_session = ort.InferenceSession(SEG_ONNX, providers=providers)
    print(f'      Providers: {seg_session.get_providers()}')

    # ── Open camera ─────────────────────────────────────────────────────────
    print('[3/3] Opening camera (index 0)...')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # CAP_DSHOW faster on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)              # fallback without backend hint
    if not cap.isOpened():
        print('ERROR: Cannot open camera. Check it is connected and not in use.')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,           30)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'      Camera: {actual_w}×{actual_h}')
    print('\nSystem ready — press Q or Esc to quit, S to screenshot.\n')

    # ── Shared state ────────────────────────────────────────────────────────
    shared = {
        'depth':     np.zeros((PANEL_H, PANEL_W), dtype=np.float32),
        'depth_fps': 0.0,
        'seg':       np.zeros((actual_h, actual_w), dtype=np.uint8),
        'seg_fps':   0.0,
    }
    lock       = threading.Lock()
    stop_event = threading.Event()

    # ── Spawn workers ───────────────────────────────────────────────────────
    depth_worker = DepthWorker(depth_model, shared, lock, stop_event)
    seg_worker   = SegWorker(seg_session,  shared, lock, stop_event)
    depth_worker.start()
    seg_worker.start()

    # ── Window ──────────────────────────────────────────────────────────────
    win_name = 'Navigation Assistance System'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WIN_W, WIN_H)

    # ── Main display loop ───────────────────────────────────────────────────
    cam_fps      = 0.0
    frame_count  = 0
    t_fps_start  = time.perf_counter()
    PUSH_EVERY   = 2   # push a new frame to workers every N camera frames
                       # (reduce CPU load; workers run at their own pace)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera read failed — retrying...')
            time.sleep(0.05)
            continue

        frame_count += 1

        # Push frames to inference workers (throttled)
        if frame_count % PUSH_EVERY == 0:
            depth_worker.push_frame(frame)
            seg_worker.push_frame(frame)

        # FPS counter (camera)
        elapsed = time.perf_counter() - t_fps_start
        if elapsed >= 0.5:
            cam_fps     = frame_count / elapsed
            frame_count = 0
            t_fps_start = time.perf_counter()

        # Grab latest inference results
        with lock:
            depth_np = shared['depth'].copy()
            seg_mask = shared['seg'].copy()
            depth_fps = shared['depth_fps']
            seg_fps   = shared['seg_fps']

        # ── Build panels ────────────────────────────────────────────────────
        cam_panel   = build_camera_panel(frame, cam_fps)
        depth_panel = build_depth_panel(frame, depth_np, depth_fps)
        seg_panel   = build_seg_panel(frame, seg_mask, seg_fps)

        # ── Compose display ─────────────────────────────────────────────────
        divider = np.full((PANEL_H, 2, 3), 60, dtype=np.uint8)
        top_row  = np.hstack([cam_panel, divider, depth_panel, divider, seg_panel])
        status   = build_status_bar(cam_fps, depth_fps, seg_fps)
        display  = np.vstack([top_row, status])

        cv2.imshow(win_name, display)

        # ── Key handling ────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q or Esc
            break
        if key in (ord('s'), ord('S')):
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(SCREENSHOT_DIR, f'nav_{ts}.png')
            cv2.imwrite(path, display)
            print(f'Screenshot saved: {path}')

    # ── Cleanup ─────────────────────────────────────────────────────────────
    print('\nShutting down...')
    stop_event.set()
    depth_worker.join(timeout=3)
    seg_worker.join(timeout=3)
    cap.release()
    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
