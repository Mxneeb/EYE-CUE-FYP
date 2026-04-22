"""
Rolling video-clip buffer — always holds the last N seconds of camera
frames in memory; on demand, flushes them to an MP4 file.

Memory layout
-------------
Frames are stored JPEG-encoded (default quality 80) in a bounded
`collections.deque`. A 30-second buffer at 15 FPS of 640×480 frames is
roughly 25 MB instead of ~400 MB of raw BGR frames — important on
Jetson with shared GPU/CPU memory.

Public API
----------
    clip = ClipBuffer(max_seconds=30.0, fps=15.0)
    clip.push(frame)              # call from main loop each iteration
    path = clip.save(output_dir)  # returns MP4 path, or None if empty / failed
    clip.clear()                  # drop all buffered frames
"""

import datetime
import os
import threading
import time
from collections import deque

import cv2
import numpy as np


class ClipBuffer:
    def __init__(self, max_seconds=30.0, fps=15.0, jpeg_quality=80):
        self.max_seconds   = float(max_seconds)
        self.fps           = float(fps)
        self.jpeg_quality  = int(jpeg_quality)
        self._max_len      = max(1, int(self.max_seconds * self.fps))
        self._frames       = deque(maxlen=self._max_len)  # (jpeg_bytes, (h, w))
        self._lock         = threading.Lock()
        self._min_interval = 1.0 / self.fps
        self._last_push    = 0.0

    def push(self, frame):
        """Sample the frame into the buffer. Throttled to `fps`."""
        now = time.monotonic()
        if now - self._last_push < self._min_interval:
            return
        ok, buf = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        self._last_push = now
        with self._lock:
            self._frames.append((buf.tobytes(), frame.shape[:2]))

    def save(self, output_dir, prefix='clip'):
        """Write all currently-buffered frames as an MP4. Returns the file path."""
        with self._lock:
            snapshot = list(self._frames)
        if not snapshot:
            print('[ClipBuffer] Buffer empty; nothing to save.')
            return None

        os.makedirs(output_dir, exist_ok=True)
        h, w   = snapshot[0][1]
        ts     = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path   = os.path.join(output_dir, f'{prefix}_{ts}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            print(f'[ClipBuffer] VideoWriter failed to open for {path}')
            return None

        count = 0
        try:
            for jpg, shape in snapshot:
                arr   = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
                count += 1
        finally:
            writer.release()

        print(f'[ClipBuffer] Saved {count} frames '
              f'({count / self.fps:.1f} s) → {path}')
        return path

    def clear(self):
        with self._lock:
            self._frames.clear()
