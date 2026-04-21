"""
Audio feedback module — text-to-speech navigation instructions.

Uses pyttsx3 for offline TTS. A dedicated background thread consumes
from a queue so speak() never blocks the caller.
"""

import time
import queue
import threading


class AudioFeedback:
    """
    Speaks navigation instructions with a cooldown between utterances.
    speak() is non-blocking — it drops the text into a queue consumed
    by a single background thread.
    """

    def __init__(self, cooldown=3.0, enabled=True):
        self.cooldown = cooldown
        self.enabled = enabled
        self._last_spoken_time = 0.0
        self._last_instruction = ''
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._engine = None
        self._thread = None
        self._init_engine()

    def _init_engine(self):
        """Initialize pyttsx3 engine. Fails gracefully if unavailable."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', 160)
            self._engine.setProperty('volume', 0.9)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f'[Audio] TTS engine unavailable: {e}')
            print('[Audio] Navigation will continue without voice feedback.')
            self._engine = None

    def _run(self):
        """Background thread: pull from queue, speak, repeat."""
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass

    def toggle(self):
        """Toggle audio on/off. Returns new state."""
        self.enabled = not self.enabled
        return self.enabled

    def speak(self, instruction):
        """
        Queue a navigation instruction for speech.
        Never blocks — drops the message if the queue is full
        (i.e. previous speech is still playing).
        """
        if not self.enabled or self._engine is None:
            return

        now = time.time()
        # Cooldown: skip if same instruction repeated too soon
        if (instruction == self._last_instruction and
                now - self._last_spoken_time < self.cooldown):
            return
        if now - self._last_spoken_time < 1.5:
            return

        self._last_spoken_time = now
        self._last_instruction = instruction

        # Non-blocking put — silently drop if queue full
        try:
            self._queue.put_nowait(instruction)
        except queue.Full:
            pass

    def shutdown(self):
        """Signal the background thread to stop."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
