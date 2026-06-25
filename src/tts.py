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


def _synth_play(tts, style, text: str, stop_event: threading.Event) -> None:
    wav, _ = tts.synthesize(text, voice_style=style, lang="en",
                            total_steps=5, speed=1.05)
    if not stop_event.is_set():
        sd.play(wav[0].astype(np.float32), samplerate=SR)
        sd.wait()


def speak(text: str, speed: float = 1.05) -> None:
    """Synthesise text and play it synchronously (used by fallback/legacy paths)."""
    tts, style = _get()
    wav, _ = tts.synthesize(text, voice_style=style, lang="en",
                             total_steps=5, speed=speed)
    sd.play(wav[0].astype(np.float32), samplerate=SR)
    sd.wait()


def speak_stream(
    token_iter: Iterator[str],
    stop_event: threading.Event,
    on_first_sentence: Optional[Callable] = None,
) -> None:
    """
    Consume LLM tokens and speak sentence-by-sentence as they arrive.

    A background worker thread synthesises and plays each sentence while
    the main thread continues pulling tokens from the LLM — so generation
    and playback overlap, drastically reducing time-to-first-audio.
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
                # drain remaining items so the queue unblocks
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
                _synth_play(tts, style, sentence, stop_event)
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
        # all parts except the last are complete sentences
        for sentence in parts[:-1]:
            sentence = sentence.strip()
            if len(sentence) > 6:   # skip very short fragments / stray punctuation
                play_q.put(sentence)
        buf = parts[-1]

    # flush any remaining text
    if buf.strip() and not stop_event.is_set():
        play_q.put(buf.strip())

    play_q.put(None)   # signal worker to exit
    worker.join()


def stop_audio() -> None:
    """Immediately halt any ongoing playback."""
    sd.stop()
