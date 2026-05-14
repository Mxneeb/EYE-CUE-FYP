"""
Spatial Sonar — depth-map driven 3D audio for EyeCue.

Instead of 6 grid buckets, the raw Depth Anything V2 output
(H x W float32 numpy array) is scanned directly. Every column
of pixels contributes to the stereo pan and distance severity.

Behaviour
---------
  Pan         — weighted centroid of the danger mass across the
                full frame width. Sub-centimetre positioning
                rather than 3 coarse zones.

  Beep rate   — proportional to how close the nearest obstacle
                is. Far = 1 beep/s. Right in front = 8 beeps/s.

  Frequency   — warning tone rises slightly as obstacle closes.
                Critical tone stays fixed and urgent.

  Level       — derived automatically from the depth map.
                Caller no longer needs to pass a level string;
                just pass the depth frame every loop iteration.

Public API
----------
    sonar = Sonar()
    sonar.toggle()
    sonar.update_from_depth(depth_map)   # call every frame
    sonar.shutdown()
"""

import subprocess
import threading
import numpy as np

# ── Audio constants ────────────────────────────────────────────────────────
_SR          = 22050
_CHUNK_S     = 0.05          # 50 ms chunks for tighter pan updates
_VOLUME      = 0.50

# Tone ranges
_WARN_FREQ_NEAR  = 800       # Hz — warning when obstacle is close
_WARN_FREQ_FAR   = 520       # Hz — warning when obstacle is distant
_CRIT_FREQ       = 1050      # Hz — stop/blocked

# Beep rate range (beeps per second)
_RATE_FAR    = 1.0
_RATE_NEAR   = 8.0
_TICK_MS     = 65            # each beep duration in ms
_FADE_MS     = 4             # fade in/out per beep

# ── Depth analysis constants ───────────────────────────────────────────────
# Depth Anything V2 outputs relative inverse depth:
# higher value = CLOSER to camera.
# We normalise to [0,1] per frame so values are scale-independent.

_ROI_TOP     = 0.30   # ignore top 30% of frame (sky, ceiling, far background)
_ROI_BOTTOM  = 0.95   # ignore very bottom edge (belt/shoe noise)

# Threshold above which a depth value counts as an obstacle
_DANGER_THRESH   = 0.55   # fraction of normalised depth
_CRITICAL_THRESH = 0.80   # fraction of normalised depth

# Minimum fraction of columns that must exceed threshold to trigger a level
_WARN_COL_FRAC   = 0.15   # 15% of columns → warning
_CRIT_COL_FRAC   = 0.40   # 40% of columns → critical


# ── DSP helpers ───────────────────────────────────────────────────────────

def _equal_power_pan(mono: np.ndarray, pan: float) -> np.ndarray:
    """
    pan in [-1, +1].  -1 = full left ear,  +1 = full right ear.
    Returns interleaved stereo float32 array.
    """
    angle      = (pan + 1.0) * np.pi / 4.0
    left_gain  = np.cos(angle)
    right_gain = np.sin(angle)
    stereo     = np.column_stack(
        (mono * left_gain, mono * right_gain)
    ).flatten()
    return stereo.astype(np.float32)


def _sine_chunk(freq: float, n: int, phase: int) -> np.ndarray:
    t = (phase + np.arange(n)) / _SR
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


# ── Depth map analysis ────────────────────────────────────────────────────

def _analyse_depth(depth_map: np.ndarray):
    """
    Analyses a raw Depth Anything V2 output frame.

    Returns
    -------
    level      : str   'safe' | 'warning' | 'critical'
    pan        : float [-1, +1]  weighted obstacle centroid
    proximity  : float [0,  1]   0 = nothing close, 1 = right in front
    """
    h, w = depth_map.shape[:2]

    # ── 1. Crop to region of interest ──────────────────────────────────────
    y0 = int(h * _ROI_TOP)
    y1 = int(h * _ROI_BOTTOM)
    roi = depth_map[y0:y1, :].astype(np.float32)

    # ── 2. Normalise per-frame so relative depth works across environments ──
    d_min, d_max = roi.min(), roi.max()
    if d_max - d_min < 1e-5:
        return 'safe', 0.0, 0.0
    roi_norm = (roi - d_min) / (d_max - d_min)

    # ── 3. Per-column danger profile ────────────────────────────────────────
    # For each horizontal column take the MAX normalised depth
    # (the single closest point in that column).
    col_danger = roi_norm.max(axis=0)          # shape (W,)

    # ── 4. Level classification ─────────────────────────────────────────────
    crit_cols = np.sum(col_danger > _CRITICAL_THRESH) / w
    warn_cols = np.sum(col_danger > _DANGER_THRESH)   / w

    if crit_cols >= _CRIT_COL_FRAC:
        level = 'critical'
    elif warn_cols >= _WARN_COL_FRAC:
        level = 'warning'
    else:
        level = 'safe'

    if level == 'safe':
        return 'safe', 0.0, 0.0

    # ── 5. Continuous pan — weighted centroid ───────────────────────────────
    # Only columns above the danger threshold contribute to pan.
    mask    = col_danger > _DANGER_THRESH
    weights = col_danger * mask                # zero out safe columns

    total_w = weights.sum()
    if total_w < 1e-6:
        pan = 0.0
    else:
        # Column positions mapped from -1 (leftmost) to +1 (rightmost)
        positions = np.linspace(-1.0, 1.0, w)
        pan       = float((weights * positions).sum() / total_w)
        pan       = np.clip(pan, -1.0, 1.0)

    # ── 6. Proximity — how close is the single nearest obstacle ─────────────
    # Take 90th percentile of col_danger in the hot columns only
    hot_vals  = col_danger[mask]
    proximity = float(np.percentile(hot_vals, 90)) if hot_vals.size else 0.0

    return level, pan, proximity


# ── Sonar class ───────────────────────────────────────────────────────────

class Sonar:
    def __init__(self):
        self._enabled   = False
        self._level     = 'safe'
        self._pan       = 0.0
        self._proximity = 0.0          # 0–1, drives beep rate + frequency
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._aplay     = None
        self._thread    = threading.Thread(
            target=self._run, daemon=True, name='Sonar')
        self._thread.start()

    # ── Public API ──────────────────────────────────────────────────────────

    def toggle(self):
        with self._lock:
            self._enabled = not self._enabled
        return self._enabled

    def enabled(self):
        with self._lock:
            return self._enabled

    def update_from_depth(self, depth_map: np.ndarray):
        """
        Pass the raw Depth Anything V2 output every frame.
        The sonar analyses it and updates all parameters internally.
        depth_map : numpy array, shape (H, W) or (H, W, 1), float32/float64
        """
        if depth_map is None:
            return
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]

        level, pan, proximity = _analyse_depth(depth_map)

        with self._lock:
            self._level     = level
            self._pan       = pan
            self._proximity = proximity

    def shutdown(self):
        self._stop.set()
        self._thread.join(timeout=2)
        if self._aplay is not None:
            try:
                self._aplay.stdin.close()
            except Exception:
                pass
            try:
                self._aplay.terminate()
                self._aplay.wait(timeout=1)
            except Exception:
                pass

    # ── Audio thread ────────────────────────────────────────────────────────

    def _start_aplay(self):
        try:
            self._aplay = subprocess.Popen(
                ['aplay', '-q',
                 '-r', str(_SR), '-f', 'S16_LE',
                 '-c', '2',              # stereo
                 '-t', 'raw', '-'],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f'[Sonar] aplay unavailable ({exc}); sonar disabled.')
            self._aplay = None

    def _run(self):
        self._start_aplay()
        if self._aplay is None:
            return

        chunk_n       = int(_SR * _CHUNK_S)
        phase_samples = 0

        while not self._stop.is_set():
            with self._lock:
                enabled   = self._enabled
                level     = self._level
                pan       = self._pan
                proximity = self._proximity

            mono = np.zeros(chunk_n, dtype=np.float32)

            if enabled:

                if level == 'critical':
                    # ── Continuous urgent tone, panned to obstacle side ──
                    mono          = _sine_chunk(_CRIT_FREQ, chunk_n, phase_samples)
                    phase_samples += chunk_n

                elif level == 'warning':
                    # ── Pulsed beeps — rate and pitch scale with proximity ──
                    # Beep rate: 1 Hz (far) → 8 Hz (near)
                    beep_rate = _RATE_FAR + proximity * (_RATE_NEAR - _RATE_FAR)
                    period_n  = max(1, int(_SR / beep_rate))

                    # Frequency: 520 Hz (far) → 800 Hz (near)
                    freq = _WARN_FREQ_FAR + proximity * (
                        _WARN_FREQ_NEAR - _WARN_FREQ_FAR)

                    tick_n = min(int(_SR * _TICK_MS / 1000), period_n)
                    fade_n = max(1, int(_FADE_MS * _SR / 1000))

                    i = 0
                    while i < chunk_n:
                        offset = phase_samples % period_n
                        if offset < tick_n:
                            fit  = min(tick_n - offset, chunk_n - i)
                            idx  = offset + np.arange(fit)
                            wave = np.sin(2 * np.pi * freq * idx / _SR)
                            env  = np.minimum(
                                idx / fade_n,
                                (tick_n - idx) / fade_n)
                            mono[i:i + fit] = wave * np.clip(env, 0.0, 1.0)
                            i             += fit
                            phase_samples += fit
                        else:
                            fit            = min(period_n - offset, chunk_n - i)
                            i             += fit
                            phase_samples += fit

                else:
                    phase_samples = 0

            else:
                phase_samples = 0

            stereo = _equal_power_pan(mono, pan)
            pcm    = (stereo * _VOLUME * 32767).astype(np.int16)
            try:
                self._aplay.stdin.write(pcm.tobytes())
            except Exception:
                break
