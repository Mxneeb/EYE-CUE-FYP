"""
Integrated Navigation Assistance Application.

Dual-stream pipeline:
  Stream A — Depth Anything V2 (PyTorch, vitb)  -> depth map
  Stream B — TopFormer-Base   (ONNX runtime)    -> ADE20K segmentation

Then:
  ODM    — Fuses depth + segmentation -> obstacle image (Algorithm 1)
  Planner — Fuzzy-logic path planning -> navigation instruction
  Audio  — TTS voice feedback

Display: [Camera | Depth Map | Segmentation | Obstacles] side-by-side

Controls:
  Q / Esc  — quit
  S        — save screenshot
  M        — mute/unmute audio
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

from nav_assist.config import (
    ROOT, DEPTH_SRC, DEPTH_CKPT, SEG_ONNX, SCREENSHOT_DIR,
    DEPTH_ENCODER, DEPTH_FEATURES, DEPTH_OUT_CHANNELS,
    PANEL_W, PANEL_H, INSTRUCTION_BAR_H,
)
from nav_assist.workers import DepthWorker, SegWorker
from nav_assist.path_planner import plan_path
from nav_assist.visualization import build_navigation_overlay
from nav_assist.audio import AudioFeedback


def main():
    # ── Pre-flight checks ───────────────────────────────────────────────
    if not os.path.exists(DEPTH_CKPT):
        print(f'ERROR: Depth checkpoint not found:\n  {DEPTH_CKPT}')
        print('Please download depth_anything_v2_vitb.pth to the project root.')
        sys.exit(1)
    if not os.path.exists(SEG_ONNX):
        print(f'ERROR: ONNX model not found:\n  {SEG_ONNX}')
        print('Run  python convert_topformer_onnx.py  first.')
        sys.exit(1)

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # ── Load Depth Anything V2 ──────────────────────────────────────────
    print('[1/4] Loading Depth Anything V2 (vitb)...')
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
    # Limit PyTorch threads so it doesn't starve the main loop
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f'      Depth model ready (threads={torch.get_num_threads()}).')

    # ── Load TopFormer ONNX ─────────────────────────────────────────────
    print('[2/4] Loading TopFormer ONNX session...')
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if DEVICE == 'cuda' else ['CPUExecutionProvider'])
    seg_session = ort.InferenceSession(SEG_ONNX, providers=providers)
    print(f'      Providers: {seg_session.get_providers()}')

    # ── Audio feedback ──────────────────────────────────────────────────
    print('[3/4] Initializing audio feedback...')
    audio = AudioFeedback(cooldown=3.0, enabled=True)

    # ── Open camera ─────────────────────────────────────────────────────
    print('[4/4] Opening camera (index 0)...')
    cap = cv2.VideoCapture(0)   # no CAP_DSHOW — works better with virtual cams
    if not cap.isOpened():
        print('ERROR: Cannot open camera.')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'      Camera: {actual_w}x{actual_h}')
    print('\nSystem ready — Q/Esc=quit, S=screenshot, M=mute/unmute\n')

    # ── Shared state ────────────────────────────────────────────────────
    shared = {
        'depth':     np.zeros((PANEL_H, PANEL_W), dtype=np.float32),
        'depth_fps': 0.0,
        'seg':       np.zeros((actual_h, actual_w), dtype=np.uint8),
        'seg_fps':   0.0,
    }
    lock       = threading.Lock()
    stop_event = threading.Event()

    # ── Spawn DTM workers ───────────────────────────────────────────────
    depth_worker = DepthWorker(depth_model, shared, lock, stop_event)
    seg_worker   = SegWorker(seg_session, shared, lock, stop_event)
    depth_worker.start()
    seg_worker.start()

    # ── Window ──────────────────────────────────────────────────────────
    win_name = 'Obs-tackle: Navigation Assistance System'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, actual_w, actual_h + INSTRUCTION_BAR_H)

    # ── Main display loop ───────────────────────────────────────────────
    cam_fps = 0.0
    obs_fps = 0.0
    frame_count = 0
    t_fps_start = time.perf_counter()
    PUSH_EVERY = 2
    nav_instruction = ''

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
            cam_fps = frame_count / elapsed
            frame_count = 0
            t_fps_start = time.perf_counter()

        # Grab latest inference results
        with lock:
            depth_np = shared['depth'].copy()
            seg_mask = shared['seg'].copy()
            depth_fps = shared['depth_fps']
            seg_fps = shared['seg_fps']

        # ── Path Planner (depth-gated + fuzzy logic) ──────────────────
        t_obs = time.perf_counter()
        nav_instruction, planner_details = plan_path(seg_mask, depth_np)

        obs_fps = 1.0 / max(time.perf_counter() - t_obs, 1e-6)

        # ── Audio feedback ──────────────────────────────────────────────
        speech_text = nav_instruction.replace(' — ', ', ')
        audio.speak(speech_text)

        # ── Build overlay display ───────────────────────────────────────
        display = build_navigation_overlay(
            frame, nav_instruction, planner_details,
            cam_fps=cam_fps, depth_fps=depth_fps,
            seg_fps=seg_fps, obs_fps=obs_fps)

        cv2.imshow(win_name, display)

        # ── Key handling ────────────────────────────────────────────────
        key = cv2.waitKey(16) & 0xFF   # ~60fps cap, gives OS time for UI
        if key in (ord('q'), ord('Q'), 27):
            break
        if key in (ord('s'), ord('S')):
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(SCREENSHOT_DIR, f'nav_{ts}.png')
            cv2.imwrite(path, display)
            print(f'Screenshot saved: {path}')
        if key in (ord('m'), ord('M')):
            state = audio.toggle()
            print(f'Audio {"ON" if state else "OFF"}')

    # ── Cleanup ─────────────────────────────────────────────────────────
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
