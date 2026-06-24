"""
Voice Assistant — Web UI
Run:  python3 app.py
Stop: Ctrl+C
"""

import os
import signal
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template

load_dotenv()

from src.brain    import VoiceBrain
from src.recorder import record_until_silence
from src.stt      import transcribe
from src.tts      import speak, stop_audio

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
_brain = None
_brain_lock = threading.Lock()
_stop_event = threading.Event()

_lock  = threading.Lock()
_state = {
    "status":         "idle",   # idle|listening|transcribing|thinking|speaking
    "messages":       [],
    "streaming_text": "",       # live LLM tokens — shown while thinking
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
        return jsonify(_state.copy())


@app.route("/record", methods=["POST"])
def record():
    with _lock:
        if _state["status"] != "idle":
            return jsonify({"error": "Already processing"}), 400
        _stop_event.clear()
        _state["status"]         = "listening"
        _state["error"]          = None
        _state["streaming_text"] = ""

    threading.Thread(target=_pipeline, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop_route():
    _stop_event.set()
    stop_audio()
    with _lock:
        _state["status"]         = "idle"
        _state["streaming_text"] = ""
        _state["error"]          = None
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset():
    global _brain
    _stop_event.set()
    stop_audio()
    with _lock:
        if _state["status"] != "idle":
            _state["status"] = "idle"
        _state["messages"]       = []
        _state["streaming_text"] = ""
        _state["error"]          = None
    with _brain_lock:
        if _brain:
            _brain.reset()
    return jsonify({"ok": True})


# ── Pipeline ─────────────────────────────────────────────────────────────────

def _pipeline() -> None:
    try:
        # 1. Record
        audio_path = record_until_silence(
            silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.01")),
            stop_event=_stop_event,
        )

        if _stop_event.is_set() or not audio_path:
            with _lock:
                _state["status"] = "idle"
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
                _state["status"] = "idle"
                if not _stop_event.is_set():
                    _state["error"] = "Could not understand — try again."
            return

        with _lock:
            _state["messages"].append({"role": "user", "content": user_text})
            _state["status"]         = "thinking"
            _state["streaming_text"] = ""

        # 3. Stream LLM tokens into state (frontend polls & shows live)
        full_reply = ""
        for token in get_brain().stream_chat(user_text):
            if _stop_event.is_set():
                break
            full_reply += token
            with _lock:
                _state["streaming_text"] = full_reply

        if not full_reply or _stop_event.is_set():
            with _lock:
                _state["status"]         = "idle"
                _state["streaming_text"] = ""
            return

        with _lock:
            _state["messages"].append({"role": "assistant", "content": full_reply})
            _state["streaming_text"] = ""
            _state["active_model"]   = get_brain().active_model
            _state["status"]         = "speaking"

        # 4. Speak via Supertonic
        speak(full_reply)

    except Exception as exc:
        with _lock:
            _state["error"] = str(exc)
    finally:
        with _lock:
            _state["status"]         = "idle"
            _state["streaming_text"] = ""


# ── Entry ─────────────────────────────────────────────────────────────────────

def _quit(sig, frame):
    stop_audio()
    os._exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _quit)

    port     = int(os.getenv("PORT", "5001"))
    groq_key = os.getenv("GROQ_API_KEY", "")
    hf_token = os.getenv("HF_TOKEN",    "")
    key_ok   = (groq_key and groq_key != "your_groq_key_here") or \
               (hf_token and hf_token != "your_huggingface_token_here")

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
