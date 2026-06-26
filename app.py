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
from src.stt      import transcribe
from src.tts      import speak_sentences, stop_audio

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

_brain      = None
_brain_lock = threading.Lock()

_stop_event   = threading.Event()
_commit_event = threading.Event()

# When True the running pipeline's finally block won't reset status to idle
# (used during barge-in so the UI stays consistent).
_barge_in = False

_lock  = threading.Lock()
_state = {
    "status":         "idle",   # idle|listening|transcribing|thinking|speaking
    "messages":       [],
    "streaming_text": "",       # live LLM tokens — shown while generating
    "active_model":   "",       # updated each reply; changes on fallback
    "error":          None,
}


def get_brain() -> VoiceBrain:
    global _brain
    with _brain_lock:
        if _brain is None:
            _brain = VoiceBrain()
    return _brain


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    voice = os.getenv("VOICE_STYLE", "F1")
    try:
        model = get_brain().provider_info()
    except Exception:
        model = "—"
    return render_template("index.html", model=model, voice=voice)


@app.route("/ping")
def ping():
    return jsonify({"ok": True})


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


@app.route("/record", methods=["POST"])
def record():
    """Legacy start-recording endpoint (also called internally)."""
    with _lock:
        if _state["status"] != "idle":
            return jsonify({"error": "Already processing"}), 400
        _stop_event.clear()
        _commit_event.clear()
        _state["status"]         = "listening"
        _state["error"]          = None
        _state["streaming_text"] = ""

    threading.Thread(target=_pipeline, daemon=True).start()
    return jsonify({"ok": True})


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
        threading.Thread(target=_pipeline, daemon=True).start()
        return jsonify({"ok": True, "action": "started"})

    elif current == "listening":
        # Process whatever the user has said so far — don't wait for silence
        _commit_event.set()
        return jsonify({"ok": True, "action": "commit"})

    elif current == "speaking":
        # Barge-in: stop playback, restart the whole listen→speak cycle
        with _lock:
            _barge_in = True
        _stop_event.set()
        stop_audio()
        threading.Thread(target=_barge_in_restart, daemon=True).start()
        return jsonify({"ok": True, "action": "barge_in"})

    else:
        # transcribing / thinking — hard cancel
        _stop_event.set()
        stop_audio()
        with _lock:
            _state["status"]         = "idle"
            _state["streaming_text"] = ""
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
        _state["error"]          = None
    with _brain_lock:
        if _brain:
            _brain.reset()
    return jsonify({"ok": True})


# ── Pipeline ─────────────────────────────────────────────────────────────────

def _barge_in_restart() -> None:
    """
    Called in a background thread after barge-in.
    Waits for the old pipeline to fully exit, then restarts the listen cycle.
    """
    global _barge_in
    time.sleep(0.15)   # pipeline's finally block has ~0ms of work after sd.stop()
    _stop_event.clear()
    _commit_event.clear()
    with _lock:
        _barge_in = False
        _state["status"]         = "listening"
        _state["error"]          = None
        _state["streaming_text"] = ""
    _pipeline()


def _pipeline() -> None:
    try:
        # 1. Record — stops on silence, stop_event (cancel), or commit_event (early process)
        audio_path = record_until_silence(
            silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.01")),
            stop_event=_stop_event,
            commit_event=_commit_event,
        )
        _commit_event.clear()   # consume the commit signal

        if _stop_event.is_set() or not audio_path:
            with _lock:
                if not _stop_event.is_set():
                    _state["error"] = "No speech detected — try again."
            return

        # 2. Transcribe
        with _lock:
            _state["status"] = "transcribing"

        user_text = transcribe(audio_path)
        os.unlink(audio_path)

        if not user_text.strip() or _stop_event.is_set():
            with _lock:
                if not _stop_event.is_set():
                    _state["error"] = "Could not understand — try again."
            return

        with _lock:
            _state["messages"].append({"role": "user", "content": user_text})
            _state["status"]         = "thinking"
            _state["streaming_text"] = ""

        # 3. Stream the FULL reply into the UI first — status stays "thinking"
        # and streaming_text grows token-by-token as the LLM writes (~0.5-1s).
        # This runs with NO TTS competing for the GIL, so the text reliably
        # appears on screen before any audio starts.
        full_llm_text = ""
        for token in get_brain().stream_chat(user_text):
            if _stop_event.is_set():
                break
            full_llm_text += token
            with _lock:
                _state["streaming_text"] = full_llm_text

        if not full_llm_text or _stop_event.is_set():
            with _lock:
                _state["streaming_text"] = ""
            return

        # 4. Text is now fully visible. Flip to "speaking" and speak it
        # sentence-by-sentence (first audio starts after the 1st sentence
        # synthesises, not the whole paragraph).
        with _lock:
            _state["status"]       = "speaking"
            _state["active_model"] = get_brain().active_model

        speak_sentences(full_llm_text, _stop_event)

        with _lock:
            _state["messages"].append({"role": "assistant", "content": full_llm_text})
            _state["streaming_text"] = ""
            _state["active_model"]   = get_brain().active_model

    except Exception as exc:
        with _lock:
            _state["error"] = str(exc)

    finally:
        with _lock:
            if not _barge_in:
                _state["status"] = "idle"
            # streaming_text is intentionally NOT cleared here.
            # The frontend clears it as soon as it renders the final bubble,
            # so the stream row stays visible right up until the bubble appears.


# ── Entry ─────────────────────────────────────────────────────────────────────

def _quit(sig, frame):
    stop_audio()
    os._exit(0)


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
    print("  Ctrl+C to stop.\n")

    app.run(host="127.0.0.1", port=port,
            debug=False, threaded=True, use_reloader=False)
