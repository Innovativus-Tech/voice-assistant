"""Text-to-speech via Supertonic (local, on-device, 44.1 kHz)."""

import os
import queue
import re
import threading
from typing import Callable, Iterator, Optional

import numpy as np
import sounddevice as sd
from supertonic import TTS as SupertonicTTS

SR = 44100

# ElevenLabs-equivalent tuning for supertonic-3:
#   total_steps=8   — the model's true default; we were using 5 (degraded quality)
#   speed=1.0       — natural neutral pace (1.05 was slightly rushed)
#   silence_duration=0.0 — we split sentences ourselves; no intra-chunk gaps wanted
#   lang=None       — lets supertonic-3 auto-resolve to "na" (multilingual fallback),
#                     which handles contractions and punctuation more robustly than "en"
_STEPS    = int(os.getenv("TTS_STEPS", "8"))
_SPEED    = float(os.getenv("TTS_SPEED", "0.8"))

_tts:   Optional[SupertonicTTS] = None
_style  = None

# Sentence boundary: .!? followed by whitespace, or double newline
_SENT_RE = re.compile(r'(?<=[.!?])\s+|\n\n')


def _get() -> tuple:
    global _tts, _style
    if _tts is None:
        _tts   = SupertonicTTS(auto_download=True)
        _style = _tts.get_voice_style(voice_name=os.getenv("VOICE_STYLE", "F1"))
    return _tts, _style


def _synth(tts, style, text: str) -> np.ndarray:
    """Synthesise text → float32 waveform array."""
    wav, _ = tts.synthesize(
        text,
        voice_style=style,
        total_steps=_STEPS,
        speed=_SPEED,
        silence_duration=0.0,  # we handle inter-sentence pacing ourselves
    )
    return wav[0].astype(np.float32)


def speak(text: str) -> None:
    """Synthesise text and play it synchronously (legacy/fallback path)."""
    tts, style = _get()
    sd.play(_synth(tts, style, text), samplerate=SR)
    sd.wait()


def speak_sentences(
    text: str,
    stop_event: threading.Event,
    on_sentence_start: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Speak an already-complete reply with no gap between sentences.

    A producer thread synthesises sentences into a small queue while the
    playback thread plays them — sentence N+1 is ready the instant N ends.
    on_sentence_start(sentence) fires the moment each sentence begins playing.
    """
    tts, style = _get()
    sentences = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    if not sentences:
        return

    wav_q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=2)

    def _producer() -> None:
        for sentence in sentences:
            if stop_event.is_set():
                break
            try:
                wav_q.put((sentence, _synth(tts, style, sentence)))
            except Exception:
                break
        wav_q.put(None)

    threading.Thread(target=_producer, daemon=True).start()

    while True:
        item = wav_q.get()
        if item is None or stop_event.is_set():
            break
        sentence, wav = item
        if on_sentence_start:
            on_sentence_start(sentence)
        sd.play(wav, samplerate=SR)
        sd.wait()


def speak_stream(
    token_iter: Iterator[str],
    stop_event: threading.Event,
    on_first_sentence: Optional[Callable] = None,
    on_sentence_start: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Consume LLM tokens and speak sentence-by-sentence as they arrive.

    A background worker synthesises and plays each sentence while the main
    thread continues pulling LLM tokens — generation and playback overlap.
    on_sentence_start(sentence) fires the instant each sentence begins playing.
    """
    tts, style = _get()
    play_q: "queue.Queue[Optional[str]]" = queue.Queue()
    _first = [False]

    def _worker() -> None:
        while True:
            sentence = play_q.get()
            if sentence is None:
                break
            if stop_event.is_set():
                while True:
                    try:
                        play_q.get_nowait()
                    except queue.Empty:
                        break
                break
            if not _first[0]:
                if on_first_sentence:
                    on_first_sentence()
                _first[0] = True
            try:
                wav = _synth(tts, style, sentence)
                if not stop_event.is_set():
                    if on_sentence_start:
                        on_sentence_start(sentence)
                    sd.play(wav, samplerate=SR)
                    sd.wait()
            except Exception:
                pass

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    buf = ""
    for token in token_iter:
        if stop_event.is_set():
            break
        buf += token
        parts = _SENT_RE.split(buf)
        for sentence in parts[:-1]:
            sentence = sentence.strip()
            if len(sentence) > 6:
                play_q.put(sentence)
        buf = parts[-1]

    if buf.strip() and not stop_event.is_set():
        play_q.put(buf.strip())

    play_q.put(None)
    worker.join()


def stop_audio() -> None:
    """Immediately halt any ongoing playback."""
    sd.stop()
