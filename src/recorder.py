"""Microphone recording with automatic silence detection.

Supports early transcription: when the first silent chunk after speech is
detected, an optional callback fires with an audio snapshot so the caller
can start STT immediately — overlapping transcription with the silence window.
"""

import threading
from typing import Any, Callable, Optional, Tuple

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHUNK_SIZE  = 1024


def record_until_silence(
    silence_threshold: float = 0.01,
    silence_duration:  float = 0.5,
    max_duration:      float = 30.0,
    pre_speech_chunks: int   = 8,
    stop_event:        Optional[threading.Event] = None,
    commit_event:      Optional[threading.Event] = None,
    early_transcribe:  Optional[Callable[[np.ndarray], Any]] = None,
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """
    Record until silence, with optional overlap between the silence window
    and transcription.

    If early_transcribe is provided, it is called in a background thread
    the instant the first silent chunk is detected after speech — so STT
    runs during the silence window rather than after it.  If the user
    resumes speaking the early result is discarded and transcription runs
    normally after recording ends.

    Returns (audio_array, early_transcription_result).
    early_transcription_result is None if early_transcribe was not given,
    or if the user resumed speaking after the first silence (stale result).
    """
    max_silent = int(silence_duration * SAMPLE_RATE / CHUNK_SIZE)
    max_chunks = int(max_duration     * SAMPLE_RATE / CHUNK_SIZE)

    chunks: list  = []
    pre_buf: list = []
    silent_n      = 0
    speaking      = False

    # early transcription state
    _early_thread: Optional[threading.Thread] = None
    _early_result: list = [None]
    _early_valid:  list = [False]

    def _launch_early(snapshot: np.ndarray) -> None:
        _early_valid[0] = True
        _early_result[0] = None

        def _run() -> None:
            _early_result[0] = early_transcribe(snapshot)

        nonlocal _early_thread
        _early_thread = threading.Thread(target=_run, daemon=True)
        _early_thread.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SIZE
    ) as stream:
        for _ in range(max_chunks):
            if stop_event and stop_event.is_set():
                return None, None
            if commit_event and commit_event.is_set():
                break

            data, _ = stream.read(CHUNK_SIZE)
            rms = float(np.sqrt(np.mean(data ** 2)))

            if rms > silence_threshold:
                if not speaking:
                    chunks.extend(pre_buf)
                    pre_buf.clear()
                    speaking = True
                elif _early_valid[0]:
                    # User resumed speaking — early result is stale
                    _early_valid[0] = False
                silent_n = 0
                chunks.append(data.copy())
            else:
                if speaking:
                    silent_n += 1
                    chunks.append(data.copy())
                    # First silent chunk — kick off transcription immediately
                    if silent_n == 1 and early_transcribe and not _early_valid[0]:
                        snapshot = np.concatenate(chunks, axis=0).flatten()
                        _launch_early(snapshot)
                    if silent_n >= max_silent:
                        break
                else:
                    pre_buf.append(data.copy())
                    if len(pre_buf) > pre_speech_chunks:
                        pre_buf.pop(0)

    if not chunks:
        return None, None

    audio = np.concatenate(chunks, axis=0).flatten()

    # Wait for early transcription (should already be done — ran during silence window)
    if _early_thread and _early_valid[0]:
        _early_thread.join(timeout=3.0)
        return audio, _early_result[0]

    return audio, None
