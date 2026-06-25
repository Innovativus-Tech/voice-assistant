"""Microphone recording with automatic silence detection."""

import tempfile
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CHUNK_SIZE  = 1024


def record_until_silence(
    silence_threshold: float = 0.01,
    silence_duration:  float = 1.5,
    max_duration:      float = 30.0,
    pre_speech_chunks: int   = 8,
    stop_event:   Optional[threading.Event] = None,
    commit_event: Optional[threading.Event] = None,
) -> Optional[str]:
    """
    Record until the user stops speaking or stop_event is set.
    If commit_event is set mid-recording, return what was captured so far.
    Returns path to a temp WAV file, or None if no speech / cancelled.
    """
    max_silent = int(silence_duration * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks = int(max_duration     * SAMPLE_RATE / CHUNK_SIZE)

    chunks: list    = []
    pre_buf: list   = []
    silent_n        = 0
    speaking        = False

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SIZE
    ) as stream:
        for _ in range(max_chunks):
            if stop_event and stop_event.is_set():
                return None
            if commit_event and commit_event.is_set():
                break   # process whatever we have so far

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

    audio   = np.concatenate(chunks, axis=0)
    tmp     = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, audio, SAMPLE_RATE)
    return tmp
