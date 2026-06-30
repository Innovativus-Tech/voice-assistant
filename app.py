"""
Voice Assistant — Web UI
Run:  python3 app.py
Stop: Ctrl+C
"""

import os
import signal
import threading
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template

load_dotenv()

from src.brain    import VoiceBrain
from src.recorder import record_until_silence
from src.stt      import transcribe, warmup as stt_warmup
from src.tts      import speak_sentences, stop_audio, warmup as tts_warmup

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

_brain      = None
_brain_lock = threading.Lock()

_stop_event   = threading.Event()
_commit_event = threading.Event()

# True during barge-in so the pipeline's finally block doesn't flip to idle.
_barge_in = False

_lock  = threading.Lock()
_state = {
    "status":         "idle",   # idle|listening|transcribing|thinking|speaking
    "messages":       [],
    "streaming_text": "",       # live LLM tokens — shown while generating
    "speaking_text":  "",       # text revealed sentence-by-sentence in sync with voice
    "active_model":   "",
    "error":          None,
}


def get_brain() -> VoiceBrain:
    global _brain
    with _brain_lock:
        if _brain is None:
            _brain = VoiceBrain()
    return _brain


# Bump this whenever behavior changes so you can confirm the running code.
BUILD = "v9-warmup"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    voice = os.getenv("VOICE_STYLE", "F1")
    try:
        model = get_brain().provider_info()
    except Exception:
        model = "—"
    return render_template("index.html", model=model, voice=voice, build=BUILD)


@app.route("/ping")
def ping():
    return jsonify({"ok": True, "build": BUILD})


@app.route("/version")
def version():
    return jsonify({"build": BUILD})


@app.route("/status")
def status():
    with _lock:
        resp = jsonify(_state.copy())
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    return resp


@app.after_request
def _no_cache_html(resp):
    """Force fresh HTML/JS every load — prevents stale cached templates."""
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"]        = "no-cache"
    return resp


@app.route("/toggle", methods=["POST"])
def toggle():
    """
    Smart toggle button:
      idle        → start listening
      listening   → commit (process what was said so far, don't wait for silence)
      transcribing/thinking → cancel → idle
      speaking    → barge-in (stop speech, restart listening)
    """
    global _barge_in

    with _lock:
        current = _state["status"]

    if current == "idle":
        with _lock:
            _stop_event.clear()
            _commit_event.clear()
            _state["status"]         = "listening"
            _state["error"]          = None
            _state["streaming_text"] = ""
            _state["speaking_text"]  = ""
        threading.Thread(target=_pipeline, daemon=True).start()
        return jsonify({"ok": True, "action": "started"})

    elif current == "listening":
        _commit_event.set()
        return jsonify({"ok": True, "action": "commit"})

    elif current == "speaking":
        with _lock:
            _barge_in = True
        _stop_event.set()
        stop_audio()
        threading.Thread(target=_barge_in_restart, daemon=True).start()
        return jsonify({"ok": True, "action": "barge_in"})

    else:
        _stop_event.set()
        stop_audio()
        with _lock:
            _state["status"]         = "idle"
            _state["streaming_text"] = ""
            _state["speaking_text"]  = ""
            _state["error"]          = None
        return jsonify({"ok": True, "action": "stopped"})


@app.route("/stop", methods=["POST"])
def stop_route():
    """Hard stop — always returns to idle."""
    global _barge_in
    with _lock:
        _barge_in = False
    _stop_event.set()
    stop_audio()
    with _lock:
        _state["status"]         = "idle"
        _state["streaming_text"] = ""
        _state["speaking_text"]  = ""
        _state["error"]          = None
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset():
    global _brain, _barge_in
    _stop_event.set()
    stop_audio()
    with _lock:
        _barge_in = False
        _state["status"]         = "idle"
        _state["messages"]       = []
        _state["streaming_text"] = ""
        _state["speaking_text"]  = ""
        _state["error"]          = None
    with _brain_lock:
        if _brain:
            _brain.reset()
    return jsonify({"ok": True})


# ── Pipeline ─────────────────────────────────────────────────────────────────

def _barge_in_restart() -> None:
    """Background thread: wait for old pipeline to exit, restart listen cycle."""
    global _barge_in
    time.sleep(0.15)
    _stop_event.clear()
    _commit_event.clear()
    with _lock:
        _barge_in = False
        _state["status"]         = "listening"
        _state["error"]          = None
        _state["streaming_text"] = ""
        _state["speaking_text"]  = ""
    _pipeline()


def _pipeline() -> None:
    try:
        # 1. Record + transcribe overlap: STT runs during the silence window.
        audio, early_text = record_until_silence(
            silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.01")),
            silence_duration=float(os.getenv("SILENCE_DURATION", "0.5")),
            stop_event=_stop_event,
            commit_event=_commit_event,
            early_transcribe=transcribe,
        )
        _commit_event.clear()

        if _stop_event.is_set() or audio is None:
            with _lock:
                if not _stop_event.is_set():
                    _state["error"] = "No speech detected — try again."
            return

        # 2. Use early transcription if ready (almost always), else run now.
        with _lock:
            _state["status"] = "transcribing"

        user_text = early_text if early_text else transcribe(audio)

        if not user_text.strip() or _stop_event.is_set():
            with _lock:
                if not _stop_event.is_set():
                    _state["error"] = "Could not understand — try again."
            return

        with _lock:
            _state["messages"].append({"role": "user", "content": user_text})
            _state["status"]         = "thinking"
            _state["streaming_text"] = ""
            _state["speaking_text"]  = ""

        # 3. Stream LLM tokens. Batch state updates by character count so we
        # don't acquire _lock on every single token (was ~hundreds of lock
        # acquisitions per reply, competing with /status polling).
        full_llm_text = ""
        last_pushed   = 0
        for token in get_brain().stream_chat(user_text):
            if _stop_event.is_set():
                break
            full_llm_text += token
            if len(full_llm_text) - last_pushed >= 12:   # ~every 2-3 tokens
                with _lock:
                    _state["streaming_text"] = full_llm_text
                last_pushed = len(full_llm_text)

        # final flush — ensure UI shows the complete text
        if full_llm_text:
            with _lock:
                _state["streaming_text"] = full_llm_text

        if not full_llm_text or _stop_event.is_set():
            with _lock:
                _state["streaming_text"] = ""
                _state["speaking_text"]  = ""
            return

        # 4. Speak sentence-by-sentence; speaking_text grows in sync with voice.
        with _lock:
            _state["status"]        = "speaking"
            _state["speaking_text"] = ""
            _state["active_model"]  = get_brain().active_model

        def _on_sentence_start(sentence: str) -> None:
            with _lock:
                prev = _state["speaking_text"]
                _state["speaking_text"] = (prev + " " + sentence).strip() if prev else sentence

        speak_sentences(full_llm_text, _stop_event, on_sentence_start=_on_sentence_start)

        with _lock:
            _state["messages"].append({"role": "assistant", "content": full_llm_text})
            _state["streaming_text"] = ""
            _state["speaking_text"]  = ""

    except Exception as exc:
        with _lock:
            _state["error"] = str(exc)

    finally:
        with _lock:
            if not _barge_in:
                _state["status"] = "idle"


# ── Entry ─────────────────────────────────────────────────────────────────────

def _quit(sig, frame):
    stop_audio()
    os._exit(0)


def _warmup_models() -> None:
    """Background: pre-load Whisper + Supertonic so first turn isn't slow."""
    t0 = time.time()
    threads = [
        threading.Thread(target=stt_warmup, daemon=True),
        threading.Thread(target=tts_warmup, daemon=True),
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"  ✓ Models warmed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _quit)

    port     = int(os.getenv("PORT", "5001"))
    groq_key = os.getenv("GROQ_API_KEY", "")
    hf_token = os.getenv("HF_TOKEN",    "")
    key_ok   = (groq_key and not groq_key.startswith("your_")) or \
               (hf_token and not hf_token.startswith("your_"))

    print()
    print("  ┌──────────────────────────────────────┐")
    print("  │        Voice Assistant  🎙            │")
    print("  └──────────────────────────────────────┘")
    print(f"  Open  →  http://localhost:{port}")
    print(f"  TTS   →  Supertonic {os.getenv('VOICE_STYLE','F1')}")
    if not key_ok:
        print()
        print("  ⚠  No LLM key found in .env")
        print("     Add GROQ_API_KEY (free at https://console.groq.com)")
    print()
    print("  Warming models in background...")

    # Pre-warm in a background thread so the server starts accepting connections
    # immediately; models are usually ready before the user clicks the mic.
    threading.Thread(target=_warmup_models, daemon=True).start()

    print("  Ctrl+C to stop.\n")

    app.run(host="127.0.0.1", port=port,
            debug=False, threaded=True, use_reloader=False)
