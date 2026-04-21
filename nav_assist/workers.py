"""
Threaded inference workers for the Data Transformation Module (DTM).
  - DepthWorker: Depth Anything V2 (PyTorch) -> depth map
  - SegWorker:   TopFormer-Base (ONNX)       -> ADE20K segmentation
"""

import time
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from nav_assist.config import ADE_MEAN, ADE_STD, SEG_SIZE


class DepthWorker(threading.Thread):
    """
    Runs Depth Anything V2 on the latest available frame.
    Writes results into `shared` dict under lock.
    """

    def __init__(self, model, shared, lock, stop_event, max_fps=5.0):
        super().__init__(daemon=True)
        self.model = model
        self.shared = shared
        self.lock = lock
        self.stop_event = stop_event
        self.max_fps = float(max_fps) if max_fps else 0.0
        self._period_s = (1.0 / self.max_fps) if self.max_fps > 0 else 0.0
        self._frame_ready = threading.Event()
        self._next_frame = None
        self._frame_lock = threading.Lock()
        self._last_publish_t = None

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
            if self._period_s > 0:
                now = time.perf_counter()
                if self._last_publish_t is not None:
                    wait_s = self._period_s - (now - self._last_publish_t)
                    if wait_s > 0:
                        time.sleep(wait_s)
            t0 = time.perf_counter()
            try:
                from nav_assist.config import DEPTH_INPUT_SIZE
                depth = self.model.infer_image(frame, input_size=DEPTH_INPUT_SIZE)
            except Exception as e:
                print(f'[DepthWorker] ERROR: {e}')
                continue
            now = time.perf_counter()
            if self._last_publish_t is None:
                eff_fps = 1.0 / max(now - t0, 1e-6)
            else:
                eff_fps = 1.0 / max(now - self._last_publish_t, 1e-6)
            self._last_publish_t = now
            with self.lock:
                self.shared['depth'] = depth
                self.shared['depth_fps'] = eff_fps


class SegWorker(threading.Thread):
    """
    Runs TopFormer (ONNX) on the latest available frame.
    Writes results into `shared` dict under lock.
    """

    def __init__(self, session, shared, lock, stop_event, max_fps=5.0):
        super().__init__(daemon=True)
        self.session = session
        self.shared = shared
        self.lock = lock
        self.stop_event = stop_event
        self.max_fps = float(max_fps) if max_fps else 0.0
        self._period_s = (1.0 / self.max_fps) if self.max_fps > 0 else 0.0
        self._frame_ready = threading.Event()
        self._next_frame = None
        self._frame_lock = threading.Lock()
        self._last_publish_t = None

    def push_frame(self, frame):
        with self._frame_lock:
            self._next_frame = frame.copy()
        self._frame_ready.set()

    @staticmethod
    def preprocess(bgr_frame):
        """BGR uint8 -> normalised float32 NCHW tensor (ADE20K convention)."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = cv2.resize(rgb, (SEG_SIZE, SEG_SIZE),
                         interpolation=cv2.INTER_LINEAR)
        norm = (rgb - ADE_MEAN) / ADE_STD
        nchw = norm.transpose(2, 0, 1)[None]
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
            if self._period_s > 0:
                now = time.perf_counter()
                if self._last_publish_t is not None:
                    wait_s = self._period_s - (now - self._last_publish_t)
                    if wait_s > 0:
                        time.sleep(wait_s)
            t0 = time.perf_counter()
            try:
                inp = self.preprocess(frame)
                logits = self.session.run(None, {'input': inp})[0]
                h, w = frame.shape[:2]
                logits_t = torch.from_numpy(logits)
                logits_up = F.interpolate(logits_t, size=(h, w),
                                          mode='bilinear', align_corners=False)
                seg_mask = logits_up[0].argmax(dim=0).numpy().astype(np.uint8)
            except Exception as e:
                print(f'[SegWorker] ERROR: {e}')
                continue
            now = time.perf_counter()
            if self._last_publish_t is None:
                eff_fps = 1.0 / max(now - t0, 1e-6)
            else:
                eff_fps = 1.0 / max(now - self._last_publish_t, 1e-6)
            self._last_publish_t = now
            with self.lock:
                self.shared['seg'] = seg_mask
                self.shared['seg_fps'] = eff_fps
