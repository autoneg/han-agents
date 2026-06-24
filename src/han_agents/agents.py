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
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    skip_failing_agents: bool = False,
    as_class: Literal[False] = False,
) -> tuple[str, ...]: ...


@overload
def get_agents(
    year: int,
    *,
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    skip_failing_agents: bool = False,
    as_class: Literal[True],
) -> tuple[type, ...]: ...


def get_agents(
    year: int,
    *,
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    skip_failing_agents: bool = False,
    as_class: bool = False,
) -> tuple:
    """Return the agents registered for a given competition year.

    The participant set, code, and metadata flags are generated into
    ``han_agents/registry.py`` by scmlweb/python/update_agents_repo.py
    (the single source of truth). HAN has no track. A qualified agent is
    any non-disqualified participant; finalists/winners are populated
    once announced (set_finalists.py / set_winners.py).

    Args:
        year: The competition year (e.g. 2026).
        qualified_only: Drop disqualified entries.
        finalists_only: Only return entries flagged as finalists.
        winners_only:   Only return entries flagged as winners.
        skip_failing_agents: Drop agents marked as failing their tests
            (listed in FAILING_AGENTS). Absence from FAILING_AGENTS is the
            default "passing" state.
        as_class: Return the class objects instead of dotted-path strings.

    Returns:
        Tuple of dotted-path strings (or tuple of class objects when
        as_class=True). Empty tuple when the year has no entries.
    """
    from han_agents.registry import get_participants

    paths = get_participants(
        int(year),
        None,
        qualified_only=qualified_only,
        finalists_only=finalists_only,
        winners_only=winners_only,
    )
    if skip_failing_agents:
        paths = tuple(p for p in paths if p not in FAILING_AGENTS.keys())
    if as_class:
        return tuple(get_class(p) for p in paths)
    return tuple(paths)
