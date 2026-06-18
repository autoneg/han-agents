"""Finalist agents from the HAN league of ANAC.

Top-level re-exports the convenience accessor only. Each year's
finalists are reachable through `han_agents.han<year>.<team_id>` and
listed in the central registry in `agents.py`.
"""
from .agents import get_agents

__all__ = ["get_agents"]
