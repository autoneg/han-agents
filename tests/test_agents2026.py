"""Run tests for the HAN 2026 participants.

HAN agents negotiate over text-enabled scenarios via negmas'
``cartesian_tournament`` (the same runner scmlweb uses). To keep the
suite small and fast despite the large field, every agent runs against
a single builtin opponent on one tiny generated scenario with very few
steps and no ufun rotation.

LLM-backed agents (``uses_llm``) make one model call per round and are
an order of magnitude slower, so they are skipped unless an Ollama
backend is reachable AND ``HANAGENTS_RUN_LLM`` is enabled. Their import
is still covered by the generated ``test_han_2026.py`` smoke test.
"""
import os

import pytest
from pytest import mark

from negmas.helpers import get_class
from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences import AffineFun, IdentityFun
from negmas.preferences import LinearAdditiveUtilityFunction as LAU
from negmas.tournaments.neg import cartesian_tournament
from negmas_llm.nonllm import BoulwareWithTextNegotiator

from han_agents import get_agents, registry

N_STEPS = 5


def _entries():
    return registry._REGISTRY.get("2026|", [])


def _split():
    llm, nonllm = [], []
    for e in _entries():
        (llm if e["metadata"].get("uses_llm") else nonllm).append(e["class_path"])
    return nonllm, llm


NONLLM_2026, LLM_2026 = _split()


def _run_llm_enabled() -> bool:
    if os.environ.get("HANAGENTS_RUN_LLM", "").lower() not in ("1", "true", "yes"):
        return False
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _tiny_scenario() -> Scenario:
    os_ = make_os([make_issue(5, "price"), make_issue(4, "qty")])
    u1 = LAU(
        values={"price": IdentityFun(), "qty": AffineFun(-1, bias=4)},
        outcome_space=os_,
        reserved_value=0.0,
    )
    u2 = LAU(
        values={"price": AffineFun(-1, bias=5), "qty": IdentityFun()},
        outcome_space=os_,
        reserved_value=0.0,
    )
    return Scenario(outcome_space=os_, ufuns=(u1.normalize(), u2.normalize()))


def _run_one(path: str):
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


def test_get_agents_2026_counts():
    assert len(get_agents(2026)) == 22
    assert len(get_agents(2026, qualified_only=True)) == 22
    assert len(get_agents(2026, finalists_only=True)) == 0
    assert len(get_agents(2026, winners_only=True)) == 0


@mark.parametrize(
    "path", NONLLM_2026, ids=[p.split(".")[2] for p in NONLLM_2026]
)
def test_can_run_nonllm_2026(path):
    _run_one(path)


@pytest.mark.skipif(
    not _run_llm_enabled(),
    reason="LLM agents: set HANAGENTS_RUN_LLM=1 with a reachable Ollama backend",
)
@mark.parametrize("path", LLM_2026, ids=[p.split(".")[2] for p in LLM_2026])
def test_can_run_llm_2026(path):
    _run_one(path)
