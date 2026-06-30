"""Speech-to-text using faster-whisper (local, free, no API key)."""

import os
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000

_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        # tiny.en is ~2× faster than base for English voice utterances with
        # negligible accuracy loss; override via WHISPER_MODEL env var.
        size = os.getenv("WHISPER_MODEL", "tiny.en")
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a float32 mono 16 kHz numpy array and return the text.

    Greedy decoding (beam_size=1, temperature=0) — 3-4× faster than beam=5
    with negligible accuracy loss for short voice utterances. VAD filter
    strips leading/trailing silence before inference.
    """
    model = _get_model()
    segments, _ = model.transcribe(
        audio,
        language="en",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        without_timestamps=True,
    )
    return " ".join(seg.text for seg in segments).strip()


def warmup() -> None:
    """Pre-load model + prime CT2 kernels so the first turn doesn't pay the cost."""
    model = _get_model()
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    try:
        list(model.transcribe(silence, language="en", beam_size=1,
                              without_timestamps=True, vad_filter=False)[0])
    except Exception:
        pass
