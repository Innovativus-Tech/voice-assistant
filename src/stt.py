"""Speech-to-text using faster-whisper (local, free, no API key)."""

import os
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        size = os.getenv("WHISPER_MODEL", "base")
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a float32 mono 16 kHz numpy array and return the text.

    Uses greedy decoding (beam_size=1, temperature=0) — 3-4× faster than
    beam_size=5 with negligible accuracy loss for short voice utterances.
    VAD filter strips leading/trailing silence before inference.
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
