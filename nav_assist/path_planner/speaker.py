"""
Navigation speaker: Piper TTS voice guidance for visually impaired users.

Uses Piper for high-quality offline TTS. A WAV cache eliminates synthesis
latency for repeated and common phrases. On a cache miss, the action suffix
("Go left.", "Stop.", etc.) is played instantly from cache while the full
phrase is synthesized in the background and stored for next time.
"""

import io
import subprocess
import threading
import time
import queue
import wave

from nav_assist.config import PIPER_VOICE_ONNX, PIPER_VOICE_JSON

# Pre-synthesized at startup — covers the vast majority of nav instructions
_PRECACHE_PHRASES = [
    'Clear ahead.',
    'Go left.',
    'Go right.',
    'Stop.',
    'Carefully go left.',
    'Carefully go right.',
    'Stop. Path blocked on all sides.',
    'Stop. No clear path.',
    'Stop. Path completely blocked.',
    # Stair alerts — pre-warmed so speak_immediate() has zero synthesis delay
    'Caution, stairs going down ahead.',
    'Caution, stairs going up ahead.',
    'Caution, stairs detected ahead.',
]

# Maps lowercase suffix → canonical cached phrase used as instant fallback
# when the full dynamic instruction (e.g. "Chair ahead. Go left.") is not cached yet
_ACTION_SUFFIXES = {
    'go left.':               'Go left.',
    'go right.':              'Go right.',
    'carefully go left.':     'Carefully go left.',
    'carefully go right.':    'Carefully go right.',
    'stop. no clear path.':   'Stop. No clear path.',
    'stop. path blocked on all sides.': 'Stop. Path blocked on all sides.',
    'stop. path completely blocked.':   'Stop. Path completely blocked.',
    'stop.':                  'Stop.',
}


class NavigationSpeaker:
    """
    Speaks navigation instructions via Piper TTS with a WAV cache.

    Cache hits play with zero synthesis delay. On a cache miss, the action
    suffix is spoken immediately (from cache) while the full phrase is
    synthesized in a background thread and stored for subsequent calls.

    Parameters
    ----------
    cooldown : float
        Kept for API compatibility; not used for gating (change-only logic
        handles repetition suppression).
    enabled : bool
        Start with speaker enabled.
    """

    def __init__(self, cooldown=2.5, enabled=True, rate=170, volume=0.9):
        self.cooldown = cooldown
        self.enabled = enabled
        self._last_spoken_time = 0.0
        self._last_instruction = ''

        self._voice = None
        self._cache: dict[str, bytes] = {}   # text → WAV bytes
        self._cache_lock = threading.Lock()

        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='NavSpeaker')
        self._init_engine()

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_engine(self):
        """Load Piper voice, pre-warm cache in background, start consumer thread."""
        try:
            from piper.voice import PiperVoice
            self._voice = PiperVoice.load(
                PIPER_VOICE_ONNX,
                config_path=PIPER_VOICE_JSON,
                use_cuda=False,
            )
            self._thread.start()
            # Pre-synthesize common phrases in background so first use is instant
            threading.Thread(target=self._precache, daemon=True,
                             name='NavSpeakerWarmup').start()
            print('[Speaker] Piper voice guidance ready (warming cache).')
        except Exception as exc:
            print(f'[Speaker] Piper TTS unavailable: {exc}')
            print('[Speaker] Navigation will continue without voice.')
            self._voice = None

    def _precache(self):
        """Synthesize all known phrases at startup and store in cache."""
        for phrase in _PRECACHE_PHRASES:
            if self._stop.is_set():
                break
            try:
                wav = self._synthesize(phrase)
                with self._cache_lock:
                    self._cache[phrase] = wav
            except Exception:
                pass
        print('[Speaker] Cache warm-up complete.')

    # ── Synthesis helpers ──────────────────────────────────────────────────

    def _synthesize(self, text: str) -> bytes:
        """Synthesize text to raw WAV bytes (blocking)."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            self._voice.synthesize_wav(text, wf)
        return buf.getvalue()

    def _get_wav(self, text: str) -> bytes:
        """
        Return WAV bytes for text. Lookup order:
          1. Exact cache hit → instant
          2. Action-suffix shortcut in cache → instant; full phrase cached async
          3. Synthesize now → blocking (first occurrence of novel phrase)
        """
        with self._cache_lock:
            if text in self._cache:
                return self._cache[text]

        # Try action-suffix shortcut
        text_lower = text.lower()
        for suffix, canonical in _ACTION_SUFFIXES.items():
            if text_lower.endswith(suffix):
                with self._cache_lock:
                    cached = self._cache.get(canonical)
                if cached is not None:
                    # Synthesize full phrase in background for next time
                    threading.Thread(
                        target=self._cache_async, args=(text,),
                        daemon=True, name='NavSpeakerSynth').start()
                    return cached

        # No shortcut available — synthesize now and cache
        return self._synthesize_and_cache(text)

    def _synthesize_and_cache(self, text: str) -> bytes:
        wav = self._synthesize(text)
        with self._cache_lock:
            self._cache[text] = wav
        return wav

    def _cache_async(self, text: str):
        """Synthesize text in background and store in cache (no return value)."""
        try:
            self._synthesize_and_cache(text)
        except Exception:
            pass

    # ── Consumer thread ────────────────────────────────────────────────────

    def _run(self):
        """Pull WAV bytes from queue and play via aplay."""
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                proc = subprocess.Popen(
                    ['aplay', '-q', '-'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc.communicate(input=item['wav'])
            except Exception:
                pass

    # ── Public API ─────────────────────────────────────────────────────────

    def speak(self, instruction, severity='safe'):
        """
        Queue a navigation instruction for speech.

        Parameters
        ----------
        instruction : str
            Text to speak.
        severity : str
            'safe', 'warning', 'critical', or 'emergency'.
            Critical/emergency flush older queued messages.
        """
        if not self.enabled or self._voice is None:
            return

        if instruction == self._last_instruction:
            return

        now = time.time()
        if now - self._last_spoken_time < 1.0:
            return

        self._last_spoken_time = now
        self._last_instruction = instruction

        if severity in ('critical', 'emergency'):
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        try:
            wav = self._get_wav(instruction)
            self._queue.put_nowait({'wav': wav, 'severity': severity})
        except queue.Full:
            pass

    def speak_immediate(self, instruction, severity='emergency'):
        """
        Speak immediately, bypassing the change-only check and 1-second gap.

        Used by stair detection, which has its own repeat-interval timer and
        must never be silenced by the normal cooldown logic.
        Flushes any queued messages before enqueueing the new one.
        """
        if not self.enabled or self._voice is None:
            return

        # Flush queue so the stair alert isn't delayed by a queued nav phrase
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        # Update tracking so the regular speak() path stays coherent
        self._last_instruction = instruction
        self._last_spoken_time = time.time()

        try:
            wav = self._get_wav(instruction)
            self._queue.put_nowait({'wav': wav, 'severity': severity})
        except queue.Full:
            pass

    def toggle(self):
        """Toggle speaker on/off. Returns the new enabled state."""
        self.enabled = not self.enabled
        return self.enabled

    def shutdown(self):
        """Signal the background thread to stop and clean up."""
        self._stop.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=2)
