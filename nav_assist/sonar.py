"""
Distance sonar — audio feedback tied to navigation severity level.

Levels (match the instruction bar text colour):
  'safe'     → silent              (green text)
  'warning'  → pulsed beep-beep   (yellow text — turning needed)
  'critical' → continuous tone     (red text — stop / blocked)

Public API
----------
    sonar = Sonar()
    sonar.toggle()               # returns new enabled state
    sonar.update('warning')      # call each main-loop iteration
    sonar.shutdown()
"""

import subprocess
import threading

import numpy as np

_SR           = 22050
_CHUNK_S      = 0.10
_VOLUME       = 0.30

# Warning: short pulsed beeps
_WARN_FREQ    = 660    # Hz
_WARN_RATE    = 3.0    # beeps per second
_WARN_TICK_MS = 80     # duration of each beep in ms

# Critical: continuous tone
_CRIT_FREQ    = 960    # Hz — higher and more urgent than warning


class Sonar:
    def __init__(self):
        self._enabled = False
        self._level   = 'safe'
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._aplay   = None
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name='Sonar')
        self._thread.start()

    # ── Public API ─────────────────────────────────────────────────────────

    def toggle(self):
        with self._lock:
            self._enabled = not self._enabled
        return self._enabled

    def enabled(self):
        with self._lock:
            return self._enabled

    def update(self, level: str):
        """Set alert level: 'safe', 'warning', or 'critical'."""
        if level not in ('safe', 'warning', 'critical'):
            level = 'safe'
        with self._lock:
            self._level = level

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

    # ── Internal ───────────────────────────────────────────────────────────

    def _start_aplay(self):
        try:
            self._aplay = subprocess.Popen(
                ['aplay', '-q', '-r', str(_SR), '-f', 'S16_LE',
                 '-c', '1', '-t', 'raw', '-'],
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
        phase_samples = 0   # shared phase counter for continuity across chunks

        while not self._stop.is_set():
            with self._lock:
                enabled = self._enabled
                level   = self._level

            chunk = np.zeros(chunk_n, dtype=np.float32)

            if enabled:
                if level == 'critical':
                    # ── Continuous tone ──────────────────────────────────
                    t     = (phase_samples + np.arange(chunk_n)) / _SR
                    chunk = np.sin(2 * np.pi * _CRIT_FREQ * t).astype(np.float32)
                    phase_samples += chunk_n

                elif level == 'warning':
                    # ── Pulsed beep-beep-beep (phase-continuous) ─────────
                    period_n = max(1, int(_SR / _WARN_RATE))
                    tick_n   = min(int(_SR * _WARN_TICK_MS / 1000), period_n)
                    fade_n   = max(1, int(0.004 * _SR))   # 4 ms fade in/out

                    i = 0
                    while i < chunk_n:
                        offset = phase_samples % period_n
                        if offset < tick_n:
                            fit = min(tick_n - offset, chunk_n - i)
                            idx = offset + np.arange(fit)
                            wave = np.sin(2 * np.pi * _WARN_FREQ * idx / _SR)
                            env  = np.minimum(idx / fade_n,
                                              (tick_n - idx) / fade_n)
                            chunk[i:i + fit] = wave * np.clip(env, 0.0, 1.0)
                            i             += fit
                            phase_samples += fit
                        else:
                            fit = min(period_n - offset, chunk_n - i)
                            i             += fit
                            phase_samples += fit
                else:
                    phase_samples = 0   # reset when safe so next warning starts clean

            else:
                phase_samples = 0

            pcm = (chunk * _VOLUME * 32767).astype(np.int16)
            try:
                self._aplay.stdin.write(pcm.tobytes())
            except Exception:
                break
