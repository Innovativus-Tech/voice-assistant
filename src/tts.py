"""Text-to-speech via Supertonic (local, on-device, 44.1 kHz)."""

import os
from typing import Optional

import numpy as np
import sounddevice as sd
from supertonic import TTS as SupertonicTTS

SR = 44100

_tts:   Optional[SupertonicTTS] = None
_style  = None


def _get() -> tuple:
    global _tts, _style
    if _tts is None:
        _tts  = SupertonicTTS(auto_download=True)
        _style = _tts.get_voice_style(voice_name=os.getenv("VOICE_STYLE", "F1"))
    return _tts, _style


def speak(text: str, speed: float = 1.05) -> None:
    """Synthesise text and play it synchronously."""
    tts, style = _get()
    wav, _ = tts.synthesize(text, voice_style=style, lang="en",
                             total_steps=8, speed=speed)
    sd.play(wav[0].astype(np.float32), samplerate=SR)
    sd.wait()


def stop_audio() -> None:
    """Immediately halt any ongoing playback."""
    sd.stop()
