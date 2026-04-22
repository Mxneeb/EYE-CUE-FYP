"""
Gemma 4 multimodal assistant — surrounding description and wardrobe suggestion.

All heavy work (model loading, inference, TTS synthesis) runs in background
daemon threads. The main display loop is never blocked.

Public API
----------
    assistant = GemmaAssistant()          # starts background load immediately
    assistant.request(frame, 'describe')  # or 'wardrobe'
    state, text, age = assistant.get_status()
    assistant.dismiss()
    assistant.shutdown()
"""

import base64
import io
import queue
import re
import subprocess
import threading
import time
import wave

import cv2
import numpy as np

from nav_assist.config import (
    GEMMA_MODEL_PATH, GEMMA_MMPROJ_PATH,
    GEMMA_N_GPU_LAYERS, GEMMA_CTX_SIZE, GEMMA_IMAGE_SIZE,
    PIPER_VOICE_ONNX, PIPER_VOICE_JSON,
)

# ── State constants ────────────────────────────────────────────────────────
STATE_LOADING    = 'loading'
STATE_READY      = 'ready'
STATE_PROCESSING = 'processing'
STATE_ERROR      = 'error'

# ── Prompts (tailored for visually impaired users) ─────────────────────────
_PROMPT_DESCRIBE = (
    "You are a spatial guide speaking directly to a person who is blind or visually impaired. "
    "Describe their surroundings in exactly 4 sentences using this structure:\n"
    "Sentence 1 — Start with 'You are currently in what looks like a' and name the environment "
    "(for example: a kitchen, a corridor, an outdoor street, a living room).\n"
    "Sentence 2 — Describe what is on their left and right using natural spatial language "
    "such as 'To your left is', 'On your right you have'. "
    "Mention furniture, walls, open space, people, or objects.\n"
    "Sentence 3 — Describe what is directly ahead and whether the path looks clear or blocked. "
    "Use distances like 'a step away', 'arm's length', 'a few metres ahead', or 'across the room'.\n"
    "Sentence 4 — Give a single hazard callout starting with 'Watch out for' or 'Be careful of'. "
    "Mention the most likely obstacle to their movement: something on the floor like a bag, "
    "cable, or step; something at head or shoulder level like a shelf or open cupboard door; "
    "or a narrow gap they would need to squeeze through. "
    "If there is genuinely no hazard, note the clearest and safest direction to move instead.\n"
    "Never say 'the camera sees'. Do not speculate beyond what is clearly visible. "
    "IMPORTANT: Do not use any punctuation marks whatsoever in your response. "
    "No periods, no commas, no apostrophes, no dashes, no colons, no question marks, no exclamation marks. "
    "Use only plain words and spaces."
)

_PROMPT_WARDROBE = (
    "You are a wardrobe assistant for a blind user. "
    "Clothing items are laid flat in front of the camera — there is no person in the image.\n\n"
    "Identify each visible item by type (shirt, t-shirt, jeans, trousers, jacket, etc.), "
    "its exact color, and any pattern (plain, striped, checkered, printed). "
    "Label items by position if there are multiple: left, middle, right.\n\n"
    "Then apply these rules based on how many items are shown:\n"
    "ONE item — describe it and suggest what colors and garment types pair well with it, "
    "using color harmony (complementary, neutral, or tonal combinations).\n"
    "TWO items — give a direct yes or no on whether they match. Be brutally honest about color clashes "
    "or style mismatches. Explain why in one sentence.\n"
    "THREE or more — pick the single best outfit combination. State which items to wear and which to skip, "
    "and commit to one clear choice.\n\n"
    "Speak in plain natural sentences — no bullet points or lists, this will be read aloud. "
    "Be decisive, never vague. Maximum 5 to 6 sentences. "
    "If no clothing is visible, say so and ask the user to lay items flat in good lighting. "
    "IMPORTANT: Do not use any punctuation marks whatsoever in your response. "
    "No periods, no commas, no apostrophes, no dashes, no colons, no question marks, no exclamation marks. "
    "Use only plain words and spaces."
)


# ── Helpers ────────────────────────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

def _strip_punctuation(text: str) -> str:
    """Remove every punctuation character — hard guarantee regardless of model compliance."""
    return text.translate(_PUNCT_TABLE)


def _split_sentences(text: str, words_per_chunk: int = 18) -> list:
    """
    Split response text into TTS-sized chunks.

    First tries splitting on sentence-ending punctuation (fallback for any
    punctuation that slipped through). If that yields only one chunk (i.e. the
    text is already punctuation-free), falls back to fixed word-count chunks so
    the sentence-by-sentence TTS pipeline still fires.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    # Punctuation-free path: split into ~words_per_chunk word chunks
    words = text.split()
    return [' '.join(words[i:i + words_per_chunk])
            for i in range(0, len(words), words_per_chunk)]


def _frame_to_b64(frame: np.ndarray, size: int = GEMMA_IMAGE_SIZE) -> str:
    """Resize + center-crop BGR frame to square, encode as base64 JPEG."""
    h, w = frame.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    cropped = resized[y0:y0 + size, x0:x0 + size]
    _, buf = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


# ── Piper TTS speaker ──────────────────────────────────────────────────────

class PiperSpeaker:
    """
    Asynchronous neural TTS using Piper.
    Synthesizes to WAV in memory, plays via aplay. CPU-only so the GPU
    remains free for llama.cpp inference.
    Gracefully no-ops if voice files are missing.
    """

    def __init__(self, onnx_path: str, json_path: str):
        self._voice = None
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._speaking = threading.Event()   # set while aplay subprocess is running
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='PiperSpeaker')
        try:
            from piper.voice import PiperVoice
            self._voice = PiperVoice.load(onnx_path, config_path=json_path,
                                          use_cuda=False)
            print('[PiperSpeaker] Voice loaded.')
        except Exception as exc:
            print(f'[PiperSpeaker] Unavailable ({exc}); TTS will be silent.')
        self._thread.start()

    def is_speaking(self) -> bool:
        """True while Piper audio is actively playing through aplay."""
        return self._speaking.is_set()

    def speak(self, text: str) -> None:
        if self._voice is None:
            return
        # Replace any pending item so latest response is always spoken
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def _synthesize(self, text: str) -> bytes:
        """Synthesize one sentence to raw WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            self._voice.synthesize_wav(text, wf)
        return buf.getvalue()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            sentences = _split_sentences(text) or [text]
            try:
                self._speaking.set()
                # Pre-synthesize the first sentence before playback starts
                audio = [None] * len(sentences)
                audio[0] = self._synthesize(sentences[0])

                for i, _ in enumerate(sentences):
                    if self._stop.is_set():
                        break

                    # Synthesize next sentence in background while current plays
                    next_thread = None
                    if i + 1 < len(sentences):
                        def _synth(idx=i + 1):
                            try:
                                audio[idx] = self._synthesize(sentences[idx])
                            except Exception as exc:
                                print(f'[PiperSpeaker] Synthesis error: {exc}')
                        next_thread = threading.Thread(target=_synth, daemon=True)
                        next_thread.start()

                    try:
                        proc = subprocess.Popen(
                            ['aplay', '-q', '-'],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        proc.communicate(input=audio[i])
                    except Exception as exc:
                        print(f'[PiperSpeaker] Playback error: {exc}')

                    # Ensure next sentence is synthesized before we loop
                    if next_thread:
                        next_thread.join()
            finally:
                self._speaking.clear()

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)


# ── Gemma assistant ────────────────────────────────────────────────────────

class GemmaAssistant:
    """
    Wraps Gemma 4 multimodal GGUF via llama-cpp-python.
    Model loads in a background thread at construction time.
    All inference is serialised through a single background thread.
    """

    def __init__(self):
        self._state       = STATE_LOADING
        self._response    = ''
        self._resp_time   = 0.0
        self._error_msg   = ''
        self._lock        = threading.Lock()

        self._llm         = None
        self._pending     = None          # (frame_bgr, mode) or None
        self._request_evt = threading.Event()
        self._stop_evt    = threading.Event()

        self._speaker = PiperSpeaker(PIPER_VOICE_ONNX, PIPER_VOICE_JSON)

        self._load_thread = threading.Thread(
            target=self._load_model, daemon=True, name='GemmaLoader')
        self._infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True, name='GemmaInfer')
        self._load_thread.start()
        self._infer_thread.start()

    # ── Public API ─────────────────────────────────────────────────────────

    def request(self, frame: np.ndarray, mode: str) -> None:
        """Queue a new inference request. No-ops if model not ready."""
        with self._lock:
            if self._state not in (STATE_READY, STATE_PROCESSING):
                return
            self._pending = (frame.copy(), mode)
            self._state   = STATE_PROCESSING
        self._request_evt.set()

    def get_status(self):
        """Return (state, response_text, age_seconds)."""
        with self._lock:
            age = (time.monotonic() - self._resp_time
                   if self._resp_time > 0 else 0.0)
            return self._state, self._response, age

    def dismiss(self) -> None:
        """Clear the currently displayed response."""
        with self._lock:
            self._response  = ''
            self._resp_time = 0.0

    def borrow_for_vision(self):
        """Atomically check READY, set PROCESSING, return the Llama instance.

        Call release_vision() when the vision tool finishes.
        Returns None if the model is not ready.
        """
        with self._lock:
            if self._state != STATE_READY or self._llm is None:
                return None
            self._state = STATE_PROCESSING
            return self._llm

    def release_vision(self) -> None:
        """Return state to READY after a vision tool call."""
        with self._lock:
            if self._state == STATE_PROCESSING:
                self._state = STATE_READY

    def is_speaking(self) -> bool:
        """True while Piper is actively playing a response through aplay."""
        return self._speaker.is_speaking()

    def speak(self, text: str) -> None:
        """Speak text via the Piper TTS backend."""
        self._speaker.speak(text)

    def shutdown(self) -> None:
        self._stop_evt.set()
        self._request_evt.set()   # unblock infer thread
        self._infer_thread.join(timeout=5)
        self._speaker.shutdown()
        self._load_thread.join(timeout=3)
        # Explicitly free CUDA resources before Python's atexit handlers run
        with self._lock:
            if self._llm is not None:
                del self._llm
                self._llm = None

    # ── Internal threads ───────────────────────────────────────────────────

    def _set_state(self, state: str, response: str = '', error: str = '') -> None:
        with self._lock:
            self._state = state
            if response:
                self._response  = response
                self._resp_time = time.monotonic()
            if error:
                self._error_msg = error

    def _load_model(self) -> None:
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            class Gemma4ChatHandler(Llava15ChatHandler):
                """Gemma 4 multimodal chat format."""
                DEFAULT_SYSTEM_MESSAGE = None
                CHAT_FORMAT = (
                    "{% for message in messages %}"
                    "{% if message.role == 'user' %}"
                    "<start_of_turn>user\n"
                    "{% if message.content is iterable and message.content is not string %}"
                    "{% for content in message.content %}"
                    "{% if content.type == 'image_url' and content.image_url is mapping %}"
                    "{{ content.image_url.url }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% for content in message.content %}"
                    "{% if content.type == 'text' %}"
                    "{{ content.text }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% else %}"
                    "{{ message.content }}"
                    "{% endif %}"
                    "<end_of_turn>\n"
                    "{% endif %}"
                    "{% if message.role == 'assistant' and message.content is not none %}"
                    "<start_of_turn>model\n{{ message.content }}<end_of_turn>\n"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "<start_of_turn>model\n"
                    "{% endif %}"
                )

            chat_handler = Gemma4ChatHandler(
                clip_model_path=GEMMA_MMPROJ_PATH,
                verbose=False,
            )
            llm = Llama(
                model_path=GEMMA_MODEL_PATH,
                chat_handler=chat_handler,
                n_ctx=GEMMA_CTX_SIZE,
                n_gpu_layers=GEMMA_N_GPU_LAYERS,
                n_threads=2,
                n_batch=512,
                logits_all=False,
                verbose=False,
            )
            with self._lock:
                self._llm = llm
            self._set_state(STATE_READY)
            print('[GemmaAssistant] Model ready.')
        except Exception as exc:
            self._set_state(STATE_ERROR, error=str(exc))
            print(f'[GemmaAssistant] Load failed: {exc}')

    def _infer_loop(self) -> None:
        while not self._stop_evt.is_set():
            fired = self._request_evt.wait(timeout=1.0)
            if not fired:
                continue
            self._request_evt.clear()
            if self._stop_evt.is_set():
                break

            with self._lock:
                pending = self._pending
                llm     = self._llm

            if pending is None or llm is None:
                continue

            frame_bgr, mode = pending
            prompt = _PROMPT_DESCRIBE if mode == 'describe' else _PROMPT_WARDROBE

            try:
                b64 = _frame_to_b64(frame_bgr, GEMMA_IMAGE_SIZE)
                messages = [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/jpeg;base64,{b64}'
                                },
                            },
                            {'type': 'text', 'text': prompt},
                        ],
                    }
                ]
                resp = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=256,
                    temperature=0.3,
                    stop=['<end_of_turn>', '<eos>'],
                )
                text = _strip_punctuation(resp['choices'][0]['message']['content'].strip())
                self._set_state(STATE_READY, response=text)
                self._speaker.speak(text)
                print(f'[GemmaAssistant] {mode}: {text[:80]}...')
            except Exception as exc:
                self._set_state(STATE_READY,
                                response=f'Analysis error: {exc}')
                print(f'[GemmaAssistant] Inference error: {exc}')
