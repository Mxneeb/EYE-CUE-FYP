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
  D        — AI surrounding description
  W        — AI wardrobe suggestion
  T        — announce time and weather
  C        — announce current time and date (offline)
  A        — toggle distance sonar (ambient pitch beep for obstacle proximity)
  V        — save the last 30 seconds of camera feed as an MP4 clip
  E        — SOS alert (Telegram message + photo + location to emergency contact)
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
    ROOT, DEPTH_SRC, DEPTH_CKPT, SEG_ONNX, SCREENSHOT_DIR, GEMMA_SNAPSHOT_DIR,
    CLIP_DIR, CLIP_MAX_SECONDS, CLIP_FPS,
    DEPTH_ENCODER, DEPTH_FEATURES, DEPTH_OUT_CHANNELS,
    PANEL_W, PANEL_H, INSTRUCTION_BAR_H, GEMMA_HINT_BAR_EXTRA_H,
    GEMMA_RESPONSE_DISPLAY_SECS,
)
from nav_assist.workers import DepthWorker, SegWorker
from nav_assist.path_planner import PathPlanner
from nav_assist.visualization import build_navigation_overlay
from nav_assist.gemma_assistant import GemmaAssistant
from nav_assist.time_weather import get_weather_and_time, announce_time
from nav_assist.startup_briefing import announce_briefing
from nav_assist.sonar import Sonar
from nav_assist.clip_buffer import ClipBuffer
from sos.sos import trigger_sos


def _grab_hires_frame(cap, fallback, target_h=720):
    """Switch cap to 4K, grab one frame, restore to 640×480, resize to target_h.

    Falls back to `fallback` if the camera can't deliver a larger frame.
    The brief resolution switch (~5 buffer flushes ≈ 150 ms) is acceptable
    for a deliberate key press.
    """
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        for _ in range(3):      # flush stale low-res frames from the buffer
            cap.grab()
        ret, hi = cap.read()
    finally:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(2):      # flush any high-res frames before the loop resumes
            cap.grab()

    if not ret or hi is None:
        print('[App] High-res grab failed — using current frame.')
        return fallback

    h, w = hi.shape[:2]
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)   # what the camera actually gave
    print(f'[App] Grabbed {w}×{h} frame for Gemma (camera reported {int(actual_h)}px)')

    if h > target_h:
        scale = target_h / h
        hi = cv2.resize(hi, (int(w * scale), target_h), interpolation=cv2.INTER_AREA)
    return hi


def _save_gemma_snapshot(frame, mode):
    """Save the exact frame being sent to Gemma for later review."""
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(GEMMA_SNAPSHOT_DIR, f'{mode}_{ts}.jpg')
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f'[App] Gemma snapshot saved: {path}')


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
    os.makedirs(GEMMA_SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(CLIP_DIR, exist_ok=True)
    sonar       = Sonar()
    clip_buffer = ClipBuffer(max_seconds=CLIP_MAX_SECONDS, fps=CLIP_FPS)

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

    # ── Path planner + voice guidance ──────────────────────────────────
    print('[3/4] Initializing path planner with voice guidance...')
    planner = PathPlanner(speaker_enabled=True, speaker_cooldown=2.5)

    # ── Gemma 4 assistant (starts loading in background) ───────────────
    print('[3.5/4] Starting Gemma 4 assistant (loads in background)...')
    gemma = GemmaAssistant()

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
    print('\nSystem ready — Q/Esc=quit, S=screenshot, M=mute/unmute, D=describe, W=wardrobe\n')

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
    cv2.resizeWindow(win_name, actual_w,
                     actual_h + INSTRUCTION_BAR_H + GEMMA_HINT_BAR_EXTRA_H)

    # ── Startup briefing (greeting + time + weather + battery) ──────────
    threading.Thread(target=announce_briefing,
                     daemon=True, name='StartupBriefing').start()

    # ── Main display loop ───────────────────────────────────────────────
    _weather_thread = None   # guards against overlapping T presses
    cam_fps = 0.0
    obs_fps = 0.0
    frame_count = 0
    t_fps_start = time.perf_counter()
    PUSH_EVERY = 2
    nav_instruction = ''
    # Nav-voice ducking: silenced while Gemma/Piper is speaking
    _gemma_was_speaking = False
    _nav_enabled_before_duck = True

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

        # ── Path Planner (depth-gated + fuzzy logic + voice) ─────────
        t_obs = time.perf_counter()
        nav_instruction, planner_details = planner.plan(seg_mask, depth_np)

        obs_fps = 1.0 / max(time.perf_counter() - t_obs, 1e-6)

        # ── Ambient feeds: rolling clip buffer + sonar level ────────────
        clip_buffer.push(frame)
        _inst_lower = nav_instruction.lower() if nav_instruction else ''
        if 'stop' in _inst_lower:
            _sonar_level = 'critical'
        elif 'left' in _inst_lower or 'right' in _inst_lower or 'turn' in _inst_lower:
            _sonar_level = 'warning'
        else:
            _sonar_level = 'safe'
        sonar.update(_sonar_level)

        # ── Auto-duck nav voice while Gemma/Piper is speaking ──────────
        gemma_speaking = gemma.is_speaking()
        if gemma_speaking and not _gemma_was_speaking:
            # Piper just started — save nav state and silence it
            _nav_enabled_before_duck = (planner.speaker.enabled
                                        if planner.speaker else False)
            if planner.speaker:
                planner.speaker.enabled = False
            _gemma_was_speaking = True
        elif not gemma_speaking and _gemma_was_speaking:
            # Piper just finished — restore nav state
            if planner.speaker:
                planner.speaker.enabled = _nav_enabled_before_duck
            _gemma_was_speaking = False

        # ── Gemma assistant status ──────────────────────────────────────
        g_state, g_response, g_age = gemma.get_status()
        if g_response and g_age > GEMMA_RESPONSE_DISPLAY_SECS:
            gemma.dismiss()

        # ── Build overlay display ───────────────────────────────────────
        display = build_navigation_overlay(
            frame, nav_instruction, planner_details,
            cam_fps=cam_fps, depth_fps=depth_fps,
            seg_fps=seg_fps, obs_fps=obs_fps,
            gemma_state=g_state,
            gemma_response=g_response,
            gemma_response_age=g_age)

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
            state = planner.toggle_speaker()
            print(f'Voice guidance {"ON" if state else "OFF"}')
            threading.Thread(
                target=gemma.speak,
                args=('Guidance on.' if state else 'Guidance off.',),
                daemon=True, name='GuidanceToggleSpeak').start()
        if key in (ord('d'), ord('D')):
            if g_state == 'ready':
                print('[App] Requesting surrounding description (grabbing high-res)...')
                hi = _grab_hires_frame(cap, frame)
                _save_gemma_snapshot(hi, 'describe')
                gemma.request(hi, 'describe')
        if key in (ord('w'), ord('W')):
            if g_state == 'ready':
                print('[App] Requesting wardrobe suggestion (grabbing high-res)...')
                hi = _grab_hires_frame(cap, frame)
                _save_gemma_snapshot(hi, 'wardrobe')
                gemma.request(hi, 'wardrobe')
        if key in (ord('t'), ord('T')):
            if _weather_thread is None or not _weather_thread.is_alive():
                print('[App] Announcing time and weather...')
                _weather_thread = threading.Thread(
                    target=get_weather_and_time, daemon=True, name='WeatherAnnounce')
                _weather_thread.start()
        if key in (ord('c'), ord('C')):
            threading.Thread(target=announce_time, daemon=True, name='TimeAnnounce').start()
        if key in (ord('a'), ord('A')):
            on = sonar.toggle()
            print(f'[App] Sonar {"ON" if on else "OFF"}')
            threading.Thread(
                target=gemma.speak,
                args=('Sonar on.' if on else 'Sonar off.',),
                daemon=True, name='SonarSpeak').start()
        if key in (ord('v'), ord('V')):
            def _do_save_clip():
                path = clip_buffer.save(CLIP_DIR)
                if path:
                    gemma.speak('Clip saved.')
                else:
                    gemma.speak('Could not save clip.')
            threading.Thread(target=_do_save_clip,
                             daemon=True, name='ClipSave').start()
        if key in (ord('e'), ord('E')):
            print('[App] SOS triggered.')
            _sos_frame = frame.copy()
            threading.Thread(target=trigger_sos, kwargs={'frame': _sos_frame},
                             daemon=True, name='SOS').start()

    # ── Cleanup ─────────────────────────────────────────────────────────
    print('\nShutting down...')
    stop_event.set()
    depth_worker.join(timeout=6)
    seg_worker.join(timeout=6)
    planner.shutdown()
    gemma.shutdown()
    sonar.shutdown()
    cap.release()
    cv2.destroyAllWindows()
    print('Done.')


if __name__ == '__main__':
    main()
