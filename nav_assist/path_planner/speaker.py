"""
Navigation speaker: TTS voice guidance for visually impaired users.

Uses pyttsx3 for offline text-to-speech. Runs in a background thread
so speak() never blocks the main loop. Critical/emergency messages
get shorter cooldowns and can flush the queue.
"""

import time
import queue
import threading


class NavigationSpeaker:
    """
    Speaks navigation instructions via TTS with cooldown management.

    Critical messages (STOP, direction changes) use a shorter cooldown
    and can flush older queued messages. Identical instructions within
    the cooldown window are silently dropped.

    Parameters
    ----------
    cooldown : float
        Minimum seconds between identical instructions.
    enabled : bool
        Start with speaker enabled.
    rate : int
        TTS speech rate (words per minute).
    volume : float
        TTS volume (0.0 to 1.0).
    """

    def __init__(self, cooldown=2.5, enabled=True, rate=170, volume=0.9):
        self.cooldown = cooldown
        self.enabled = enabled
        self._rate = rate
        self._volume = volume
        self._last_spoken_time = 0.0
        self._last_instruction = ''
        self._queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._engine = None
        self._thread = None
        self._init_engine()

    def _init_engine(self):
        """Initialize pyttsx3 and start the background consumer thread."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self._rate)
            self._engine.setProperty('volume', self._volume)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            print('[Speaker] Voice guidance ready.')
        except Exception as e:
            print(f'[Speaker] TTS unavailable: {e}')
            print('[Speaker] Navigation will continue without voice.')
            self._engine = None

    def _run(self):
        """Background thread: pull from queue, speak, repeat."""
        while not self._stop.is_set():
            try:
                msg = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._engine:
                    self._engine.say(msg['text'])
                    self._engine.runAndWait()
            except Exception:
                pass

    def speak(self, instruction, severity='safe'):
        """
        Queue a navigation instruction for speech.

        Parameters
        ----------
        instruction : str
            Text to speak.
        severity : str
            'safe', 'warning', 'critical', or 'emergency'.
            Critical/emergency use a shorter cooldown and can
            flush older messages from the queue.
        """
        if not self.enabled or self._engine is None:
            return

        now = time.time()

        # Critical/emergency messages get a shorter cooldown
        effective_cooldown = self.cooldown
        if severity in ('critical', 'emergency'):
            effective_cooldown = min(1.5, self.cooldown)

        # Skip if same instruction repeated within cooldown
        if (instruction == self._last_instruction
                and now - self._last_spoken_time < effective_cooldown):
            return

        # Minimum gap between any two messages
        if now - self._last_spoken_time < 1.0:
            return

        self._last_spoken_time = now
        self._last_instruction = instruction

        # Flush queue for urgent messages
        if severity in ('critical', 'emergency'):
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        try:
            self._queue.put_nowait({
                'text': instruction,
                'severity': severity,
            })
        except queue.Full:
            pass

    def toggle(self):
        """Toggle speaker on/off. Returns the new enabled state."""
        self.enabled = not self.enabled
        return self.enabled

    def shutdown(self):
        """Signal the background thread to stop and clean up."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
