"""Registry of finalist agents per year.

Mirrors the API of `anl_agents.agents.get_agents(...)` so callers
familiar with the ANL package can swap it in without re-learning the
contract. Filter by year and by `finalists_only` / `winners_only`.

Each entry is keyed by year and decorated with metadata bits used by
the filter flags. Maintained by hand or by
`scmlweb/python/set_han_finalists.py`.
"""
from __future__ import annotations

from typing import Literal, overload

from negmas.helpers import get_class, get_full_type_name

__all__ = ["get_agents", "FAILING_AGENTS"]

# (full-dotted-class-name) -> reason. Currently empty; populated as
# we discover finalists that need extra deps not yet declared.
FAILING_AGENTS: dict[str, str] = {}

# Per-year registry. Keys are integers; values are dicts of:
#   class_path   -> fully-qualified dotted class name
#   metadata     -> {"finalist": bool, "winner": bool, "team_id": str}
# Updated programmatically by set_han_finalists.py; safe to edit by hand.
_REGISTRY: dict[int, list[dict]] = {
    2026: [
        # populated by scmlweb/python/set_han_finalists.py
    ],
}


@overload
def get_agents(
    year: int,
    *,
    finalists_only: bool = False,
    winners_only: bool = False,
    as_class: Literal[False] = False,
) -> tuple[str, ...]: ...


@overload
def get_agents(
    year: int,
    *,
    finalists_only: bool = False,
    winners_only: bool = False,
    as_class: Literal[True],
) -> tuple[type, ...]: ...


def get_agents(
    year: int,
    *,
    finalists_only: bool = False,
    winners_only: bool = False,
    as_class: bool = False,
) -> tuple:
    """Return the agents registered for a given competition year.

    Args:
        year: The competition year (e.g. 2026).
        finalists_only: Only return entries flagged as finalists.
        winners_only:   Only return entries flagged as winners.
        as_class: Return the class objects instead of dotted-path strings.

    Returns:
        Tuple of dotted-path strings (or tuple of class objects when
        as_class=True). Empty tuple when the year has no entries.
    """
    entries = _REGISTRY.get(int(year), [])
    out: list = []
    for e in entries:
        if winners_only and not e.get("metadata", {}).get("winner"):
            continue
        if finalists_only and not e.get("metadata", {}).get("finalist"):
            continue
        path = e["class_path"]
        if as_class:
            out.append(get_class(path))
        else:
            out.append(path)
    return tuple(out)
