"""
LLM brain — Groq (primary) with automatic fallback chain on rate limits.
Falls back: 70b → 8b-instant → gemma2-9b → 3b → HF (if token set).
"""

import os
from typing import List, Dict, Optional, Iterator

SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. "
    "Give clear, concise spoken answers — no markdown, no bullet points, "
    "no lists. Keep responses under 3 sentences unless asked for more."
)

# Groq fallback chain — tried in order when previous model is rate-limited.
# groq/compound and groq/compound-mini have NO daily token limit.
GROQ_CHAIN = [
    "llama-3.3-70b-versatile",  # primary — best quality, 1K req/day
    "groq/compound",            # fallback 1 — no token/day limit
    "groq/compound-mini",       # fallback 2 — no token/day limit (smaller/faster)
]


def _is_rate_limit(exc: Exception) -> bool:
    """Return True for any rate-limit / quota-exceeded error."""
    try:
        import groq
        if isinstance(exc, groq.RateLimitError):
            return True
    except ImportError:
        pass
    s = str(exc).lower()
    return "rate_limit" in s or "rate limit" in s or "429" in s or "quota" in s


class VoiceBrain:
    def __init__(self) -> None:
        self._client         = None
        self._provider: Optional[str] = None
        self._primary_model: Optional[str] = None
        self._active_model:  Optional[str] = None   # may change on fallback
        self._history: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── lazy init ───────────────────────────────────────────────

    def _init(self) -> None:
        groq_key = os.getenv("GROQ_API_KEY", "")
        hf_token  = os.getenv("HF_TOKEN", "")

        if groq_key and not groq_key.startswith("your_"):
            from groq import Groq
            self._client        = Groq(api_key=groq_key)
            self._provider      = "groq"
            preferred           = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            # make sure preferred is first in chain
            chain = [preferred] + [m for m in GROQ_CHAIN if m != preferred]
            self._primary_model = chain[0]
            self._active_model  = chain[0]
            self._chain         = chain

        elif hf_token and not hf_token.startswith("your_"):
            from huggingface_hub import InferenceClient
            model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
            self._client        = InferenceClient(model=model, token=hf_token)
            self._provider      = "hf"
            self._primary_model = model
            self._active_model  = model
            self._chain         = [model]

        else:
            raise RuntimeError(
                "No LLM key found. Add GROQ_API_KEY to .env "
                "(free at https://console.groq.com)"
            )

    def _ensure(self) -> None:
        if self._client is None:
            self._init()

    # ── public ──────────────────────────────────────────────────

    @property
    def active_model(self) -> str:
        self._ensure()
        return self._active_model or "—"

    def provider_info(self) -> str:
        self._ensure()
        return f"{self._provider}/{self._active_model}"

    def stream_chat(self, user_text: str) -> Iterator[str]:
        """Stream reply tokens. Automatically falls back on rate-limit."""
        self._ensure()
        self._history.append({"role": "user", "content": user_text})
        full = ""

        try:
            if self._provider == "groq":
                full = yield from self._groq_stream()
            else:
                full = yield from self._hf_stream()
        finally:
            if full:
                self._history.append({"role": "assistant", "content": full})
                if len(self._history) > 21:
                    self._history = [self._history[0]] + self._history[-20:]

    def _groq_stream(self) -> Iterator[str]:
        """Try each model in the chain; step down on rate-limit."""
        # start from active model (may already be a fallback from a prior call)
        chain = self._chain
        start = chain.index(self._active_model) if self._active_model in chain else 0

        last_err = None
        for model in chain[start:]:
            try:
                stream = self._client.chat.completions.create(
                    model=model,
                    messages=self._history,
                    max_tokens=256,
                    temperature=0.7,
                    stream=True,
                )
                self._active_model = model   # record which model actually responded
                full = ""
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    full += token
                    yield token
                return full

            except Exception as exc:
                if _is_rate_limit(exc):
                    last_err = exc
                    print(f"  [brain] {model} rate-limited → trying next model")
                    continue
                raise   # non-rate-limit error — bubble up

        # exhausted every fallback
        raise RuntimeError(
            f"All Groq models rate-limited. Last error: {last_err}"
        ) from last_err

    def _hf_stream(self) -> Iterator[str]:
        full = ""
        stream = self._client.chat_completion(
            messages=self._history,
            max_tokens=256,
            temperature=0.7,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full += token
            yield token
        return full

    def reset(self) -> None:
        self._history      = [self._history[0]]
        # reset active model back to primary on conversation reset
        if self._primary_model:
            self._active_model = self._primary_model
