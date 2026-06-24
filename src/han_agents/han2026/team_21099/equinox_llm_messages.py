"""Stance-aware natural-language messages for Equinox (HAN human track).

The negotiation *policy* owns every decision (what to offer, accept, or reject).
This module only renders the chat text that a human counterpart reads alongside
those actions.

Hard privacy rule
-----------------
Nothing that could identify our utility function may reach the model or the
human: no numbers, no "utility" / "reserved value" / limits, no issue
valuations or priority rankings. Internal code chooses a qualitative *stance*
from those private quantities; only the stance and coarse, non-numeric dynamics
are ever rendered.

Robustness
----------
Every path degrades to a deterministic template. The LLM call is best-effort
with a short timeout and is wrapped so that a missing/slow/odd model can never
slow down or crash the negotiation. ``_sanitize`` is the real privacy guarantee
- it does not trust the 4B model to obey the prompt.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

__all__ = [
    "Stance",
    "Dynamics",
    "classify_stance",
    "time_phase",
    "try_llm_message",
    "template_message",
    "compose_message",
]


class Stance(Enum):
    """The negotiation posture a message should convey."""

    OPENING = "opening"
    HOLDING = "holding"
    RECIPROCATING = "reciprocating"
    PRESSING = "pressing"
    CLOSING = "closing"
    WALKING = "walking"
    ACCEPT = "accept"


#: Past this share of the deadline we treat the negotiation as late-game.
LATE_THRESHOLD = 0.85
#: Minimum normalized-utility change that counts as "someone moved".
_MOVE_EPS = 1e-3


@dataclass(frozen=True)
class Dynamics:
    """Coarse, non-numeric description of the round (safe to render)."""

    we_moved: bool = False
    they_moved: bool = False
    time_phase: str = "early"


def time_phase(relative_time: float) -> str:
    """Bucket ``relative_time`` into early / mid / late (no raw number leaks)."""
    if relative_time >= LATE_THRESHOLD:
        return "late"
    if relative_time >= 0.4:
        return "mid"
    return "early"


def classify_stance(
    *,
    is_opening: bool,
    is_accept: bool,
    relative_time: float,
    we_moved: bool,
    they_moved: bool,
    their_offer_acceptable: bool,
) -> Stance:
    """Map private negotiation signals to a stance.

    The numeric inputs are used only here, to *select* a posture; none of them
    are ever forwarded to the model or the human.
    """
    if is_accept:
        return Stance.ACCEPT
    if is_opening:
        return Stance.OPENING
    if relative_time >= LATE_THRESHOLD:
        # Near the deadline: close a deal that clears our walk-away, otherwise
        # signal a polite limit. (Acceptability is computed privately upstream.)
        return Stance.CLOSING if their_offer_acceptable else Stance.WALKING
    if we_moved and they_moved:
        return Stance.RECIPROCATING
    if we_moved and not they_moved:
        return Stance.PRESSING
    return Stance.HOLDING


# ---------------------------------------------------------------------------
# Deterministic template bank (the always-available fallback).
# Every line is privacy-safe by construction: no digits, no "utility"/
# "reserved"/"weight", no named issues, no stated priorities.
# ---------------------------------------------------------------------------

_TEMPLATES: dict[Stance, list[str]] = {
    Stance.OPENING: [
        "Glad to get started. Here is where I would like to begin, and I am hopeful we can find something good for us both.",
        "Thanks for sitting down with me. Let me open with this and see where we can meet.",
        "Looking forward to working this out together. Here is my opening thought.",
    ],
    Stance.HOLDING: [
        "I would like to hold here for now, but I am staying at the table and listening.",
        "I am not quite there yet; I want to get this right. Let us keep talking.",
        "Let me stand on this for the moment. I am keen to understand what is important to you.",
    ],
    Stance.RECIPROCATING: [
        "I appreciate you moving, and I am moving too. It feels like we are heading somewhere good.",
        "Thanks for the give and take. Here is me meeting you partway; nice momentum.",
        "We are both budging and I like it. Let us keep this going.",
    ],
    Stance.PRESSING: [
        "I have shifted my position here. Some movement from your side would help us close the gap.",
        "I have come your way; could you meet me a little? I think we are close.",
        "I am trying to bridge this, and a step from you would help us land it.",
    ],
    Stance.CLOSING: [
        "I think this is fair for both of us, and I am ready to shake on it.",
        "This feels like a good place to land. Let us lock it in.",
        "I am happy with where we have arrived; let us finish here.",
    ],
    Stance.WALKING: [
        "I have gone about as far as I comfortably can. If we cannot close, I understand, no hard feelings.",
        "This is close to my edge, honestly. I would love to agree, but only if it works for me too.",
        "I am not sure I can stretch further. Let us see if there is a way, but I may have to pass.",
    ],
    Stance.ACCEPT: [
        "This works for me, and I am happy to accept. Thanks for working with me.",
        "Agreed. I think we both did well here.",
        "I accept, and I am glad we found common ground.",
    ],
}


def template_message(stance: Stance) -> str:
    """A deterministic, privacy-safe line for ``stance`` (never raises)."""
    try:
        return random.choice(_TEMPLATES[stance])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# LLM rendering (best-effort, sanitized). The model is fixed by the league to
# qwen3:4b-instruct; we only ever ask it to phrase a stance we already chose.
# ---------------------------------------------------------------------------

try:  # keep import light and optional
    from negmas_llm.common import DEFAULT_MODELS as _DEFAULT_MODELS

    _OLLAMA_MODEL = _DEFAULT_MODELS.get("ollama", "qwen3:4b-instruct")
except Exception:  # pragma: no cover - negmas_llm always present in the league env
    _OLLAMA_MODEL = "qwen3:4b-instruct"

_MODEL = f"ollama/{_OLLAMA_MODEL}"
_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
_TIMEOUT = float(os.environ.get("EQUINOX_LLM_TIMEOUT", "3"))
_MAX_TOKENS = 64

_SYSTEM_PROMPT = (
    "You speak for a negotiator chatting with a human counterpart. Reply with "
    "ONE short, natural, friendly chat line that fits the situation. Never "
    "reveal numbers, prices, percentages, how much you value anything, your "
    "walk-away point, or which terms matter most to you. Never insult or "
    "criticise the other person. Output only the line, nothing else."
)

_STANCE_BRIEF: dict[Stance, str] = {
    Stance.OPENING: "It is the very start. Greet warmly and signal you are open to a good deal for both sides.",
    Stance.HOLDING: "Politely hold your position for now, framed as principled rather than stubborn; stay friendly and curious about their needs.",
    Stance.RECIPROCATING: "Both sides are giving ground. Sound warm and encouraged about the progress.",
    Stance.PRESSING: "You moved toward them but they did not move. Warmly and gently invite them to meet you, with no blame.",
    Stance.CLOSING: "A deal that works for you is within reach. Warmly suggest finishing and shaking hands.",
    Stance.WALKING: "No deal that works for you has appeared yet. Politely signal you are near the edge of what you can do, without any rancour.",
    Stance.ACCEPT: "You are accepting their offer. Warmly confirm and thank them.",
}


def _build_user_prompt(stance: Stance, dynamics: Dynamics) -> str:
    parts = [f"Situation: {_STANCE_BRIEF[stance]}"]
    if dynamics.time_phase == "late":
        parts.append("Time is almost up.")
    parts.append("Write one short line (at most 25 words). Output only the line.")
    return " ".join(parts)


def _llm_complete(user_prompt: str) -> str | None:
    """Best-effort single completion from the fixed local model.

    Returns ``None`` on any problem (litellm missing, Ollama down, timeout,
    empty output). Never raises.
    """
    try:
        import litellm
    except Exception:
        return None
    try:
        response = litellm.completion(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            api_base=_API_BASE,
            temperature=0.7,
            max_tokens=_MAX_TOKENS,
            timeout=_TIMEOUT,
        )
        text = response.choices[0].message.content or ""
        return text.strip() or None
    except Exception:
        return None


_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_LABEL_RE = re.compile(r"^(?:line|message|text|response|output)\s*[:\-]\s*", re.IGNORECASE)
#: Any digit, currency, percent, or an identifying preference token => reject.
_BANNED_RE = re.compile(r"\d|%|\$|€|£|\b(?:utilit|reserv|weight|valuation)\w*", re.IGNORECASE)
_MAX_WORDS = 35
_MAX_CHARS = 220


def _sanitize(text: str | None) -> str | None:
    """Return a privacy-safe single line, or ``None`` to force a fallback.

    This is the guarantee that does not depend on the 4B model obeying the
    prompt: any number, currency, percent, or utility/reserved/weight token
    rejects the whole output, as does anything over-long.
    """
    if not text:
        return None
    text = _THINK_RE.sub(" ", text)
    line = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped:
            line = stripped
            break
    line = _LABEL_RE.sub("", line).strip().strip("\"'`").strip()
    if not line:
        return None
    if _BANNED_RE.search(line):
        return None
    if len(line) > _MAX_CHARS or len(line.split()) > _MAX_WORDS:
        return None
    return line


def try_llm_message(
    stance: Stance,
    dynamics: Dynamics,
    *,
    llm: Callable[[str], str | None] | None = None,
) -> str | None:
    """Render ``stance`` via the model, sanitized. ``None`` on any failure.

    ``llm`` is injectable for tests; by default the module-level ``_llm_complete``
    is resolved at call time (so it can be monkeypatched).
    """
    caller = llm if llm is not None else _llm_complete
    try:
        raw = caller(_build_user_prompt(stance, dynamics))
    except Exception:
        return None
    return _sanitize(raw)


def compose_message(stance: Stance, dynamics: Dynamics, *, allow_llm: bool) -> str:
    """LLM-rendered line when ``allow_llm`` and it succeeds, else a template."""
    if allow_llm:
        rendered = try_llm_message(stance, dynamics)
        if rendered is not None:
            return rendered
    return template_message(stance)
