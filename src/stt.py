"""Speech-to-text using faster-whisper (local, free, no API key)."""

import os
from typing import Optional
from faster_whisper import WhisperModel

_model: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        size = os.getenv("WHISPER_MODEL", "base")
        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


def transcribe(audio_path: str) -> str:
    """Transcribe a WAV file and return the text."""
    model = _get_model()
    segments, _ = model.transcribe(audio_path, beam_size=5, language="en")
    return " ".join(seg.text for seg in segments).strip()
