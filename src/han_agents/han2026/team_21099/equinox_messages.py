"""Template-based natural language messages for Equinox (no LLM)."""

from __future__ import annotations

import random
from typing import Any

from negmas.outcomes import Outcome

__all__ = ["acceptance_text", "proposal_text", "rejection_text"]

ACCEPTANCE_MESSAGES = [
    "This works for me. I accept your offer.",
    "Thank you — I am happy to accept these terms.",
    "Agreed. Let's finalize this deal.",
    "I accept. This is a fair outcome for both of us.",
]

REJECTION_STARTERS = [
    "I appreciate your offer, but",
    "Thank you for the proposal, however",
    "I have reviewed your offer and",
]


def acceptance_text() -> str:
    return random.choice(ACCEPTANCE_MESSAGES)


def rejection_text() -> str:
    return f"{random.choice(REJECTION_STARTERS)} I need to make a counter-offer."


def _issue_name(issues: list[Any] | None, index: int) -> str:
    if issues and index < len(issues):
        issue = issues[index]
        return str(getattr(issue, "name", issue) or f"issue_{index + 1}")
    return f"issue_{index + 1}"


def proposal_text(
    proposal: Outcome,
    opponent_offer: Outcome | None,
    issues: list[Any] | None = None,
) -> str:
    """Generate a short, deterministic counter-offer message."""
    if opponent_offer is None:
        return "Here is my opening offer."

    changes: list[str] = []
    for idx, (mine, theirs) in enumerate(zip(proposal, opponent_offer)):
        if mine == theirs:
            continue
        issue = _issue_name(issues, idx)
        changes.append(f"{issue}: {theirs} -> {mine}")

    if not changes:
        return "I am proposing the same terms again."

    joined = "; ".join(changes[:3])
    return f"{random.choice(REJECTION_STARTERS)} I propose {joined}."
