"""Smoke test: import each registered finalist and verify it resolves
to a class. Run with `pytest -q`."""
from negmas.helpers import get_class

from han_agents import get_agents


def test_registry_imports_2026():
    paths = get_agents(2026)
    # When the registry is empty (pre-finalist-selection) we just
    # confirm the call returns an empty tuple without raising.
    if not paths:
        return
    for p in paths:
        cls = get_class(p)
        assert isinstance(cls, type), f"{p} did not resolve to a class"


def test_finalists_subset_of_all():
    all_2026 = set(get_agents(2026))
    finalists = set(get_agents(2026, finalists_only=True))
    assert finalists <= all_2026
