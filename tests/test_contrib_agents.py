"""Smoke tests for han_agents.contrib -- provider-agnostic agent variants.

These are not part of the han2026 registry (see han_agents/contrib/__init__.py),
so they aren't covered by test_agents2026.py's registry-driven parametrization.
Exercised the same way: a tiny scenario against a fixed opponent, using the
``ollama_backend`` fixture so the LLM call path runs deterministically.
"""

from negmas.helpers import get_class
from negmas.tournaments.neg import cartesian_tournament
from negmas_llm.nonllm import BoulwareWithTextNegotiator
from pytest import mark

from tests.test_agents2026 import N_STEPS, _tiny_scenario

CONTRIB_AGENTS = (
    "han_agents.contrib.agent96.GeneralizedAgent96",
    "han_agents.contrib.civic_compass.GeneralizedCivicCompassHANNegotiator",
)


@mark.parametrize("path", CONTRIB_AGENTS, ids=[p.split(".")[-1] for p in CONTRIB_AGENTS])
def test_can_run_contrib_agent(path, ollama_backend):
    cartesian_tournament(
        competitors=[get_class(path), BoulwareWithTextNegotiator],
        scenarios=[_tiny_scenario()],
        n_steps=N_STEPS,
        n_repetitions=1,
        rotate_ufuns=False,
        njobs=-1,
        verbosity=0,
        path=None,
    )
