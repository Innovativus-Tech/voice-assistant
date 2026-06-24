"""
Voice Assistant — ChatGPT-style voice conversation
  STT  : faster-whisper (local Whisper)
  Brain: Hugging Face Inference API (free)
  TTS  : Supertonic (local on-device)
"""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

load_dotenv()

from src.recorder import record_until_silence
from src.stt import transcribe
from src.brain import VoiceBrain
from src.tts import speak

console = Console()

BANNER = """
[bold cyan]  Voice Assistant[/bold cyan]
  [dim]STT:[/dim]   Whisper (local)
  [dim]Brain:[/dim] Hugging Face — {model}
  [dim]TTS:[/dim]   Supertonic (local · voice {voice})
""".format(
    model=os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta"),
    voice=os.getenv("VOICE_STYLE", "F1"),
)

HELP_TEXT = (
    "[dim]Commands:[/dim]  [bold]Enter[/bold] = speak  "
    "[bold]r[/bold] = reset chat  "
    "[bold]q[/bold] = quit"
)

SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.01"))


def run() -> None:
    console.print(Panel(BANNER.strip(), expand=False))
    console.print(HELP_TEXT)

    brain = VoiceBrain()

    # Warm up models on first run so the first response isn't slow
    console.print("\n[dim]Loading models (first run may download ~400 MB for Supertonic)...[/dim]")
    try:
        from src.tts import _get_tts
        from src.stt import _get_model
        _get_model()
        _get_tts()
        console.print("[green]Models ready.[/green]\n")
    except Exception as e:
        console.print(f"[red]Warning: model pre-load failed — {e}[/red]\n")

    while True:
        console.rule("[dim]ready[/dim]")
        cmd = Prompt.ask(
            "[bold green]Press Enter to speak[/bold green] [dim](r=reset, q=quit)[/dim]",
            default="",
        ).strip().lower()

        if cmd == "q":
            console.print("[dim]Goodbye.[/dim]")
            break

        if cmd == "r":
            brain.reset()
            console.print("[yellow]Conversation reset.[/yellow]")
            continue

        # ── Record ────────────────────────────────────────────────
        console.print("[bold cyan]Listening...[/bold cyan] [dim](speak, then pause)[/dim]")
        audio_path = record_until_silence(silence_threshold=SILENCE_THRESHOLD)

        if audio_path is None:
            console.print("[yellow]No speech detected — try again.[/yellow]")
            continue

        # ── STT ───────────────────────────────────────────────────
        console.print("[dim]Transcribing...[/dim]")
        user_text = transcribe(audio_path)
        os.unlink(audio_path)

        if not user_text:
            console.print("[yellow]Couldn't make out speech — try again.[/yellow]")
            continue

        rprint(f"\n[bold white]You:[/bold white] {user_text}")

        # ── LLM ───────────────────────────────────────────────────
        console.print("[dim]Thinking...[/dim]")
        reply = brain.chat(user_text)
        rprint(f"[bold magenta]Assistant:[/bold magenta] {reply}\n")

        # ── TTS ───────────────────────────────────────────────────
        console.print("[dim]Speaking...[/dim]")
        try:
            speak(reply)
        except Exception as e:
            console.print(f"[red]TTS error: {e}[/red]")


if __name__ == "__main__":
    run()
