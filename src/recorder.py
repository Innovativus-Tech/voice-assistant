"""Microphone recording with automatic silence detection."""

import threading
from typing import Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SIZE  = 1024


def record_until_silence(
    silence_threshold: float = 0.01,
    silence_duration:  float = 0.8,   # was 1.5 — cuts ~0.7s per turn
    max_duration:      float = 30.0,
    pre_speech_chunks: int   = 8,
    stop_event:   Optional[threading.Event] = None,
    commit_event: Optional[threading.Event] = None,
) -> Optional[np.ndarray]:
    """
    Record until the user stops speaking or stop_event is set.
    Returns a float32 mono numpy array at 16 kHz, or None if no speech.
    No temp file is written — the array is passed directly to the STT engine.
    """
    max_silent = int(silence_duration * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks = int(max_duration     * SAMPLE_RATE / CHUNK_SIZE)

    chunks: list  = []
    pre_buf: list = []
    silent_n      = 0
    speaking      = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SIZE
    ) as stream:
        for _ in range(max_chunks):
            if stop_event and stop_event.is_set():
                return None
            if commit_event and commit_event.is_set():
                break

            data, _ = stream.read(CHUNK_SIZE)
            rms = float(np.sqrt(np.mean(data ** 2)))

            if rms > silence_threshold:
                if not speaking:
                    chunks.extend(pre_buf)
                    pre_buf.clear()
                    speaking = True
                silent_n = 0
                chunks.append(data.copy())
            else:
                if speaking:
                    silent_n += 1
                    chunks.append(data.copy())
                    if silent_n >= max_silent:
                        break
                else:
                    pre_buf.append(data.copy())
                    if len(pre_buf) > pre_speech_chunks:
                        pre_buf.pop(0)

    if not chunks:
        return None

    # Flatten to 1-D float32 — what faster-whisper expects
    return np.concatenate(chunks, axis=0).flatten()
