"""Generated registry of participants per (year, track).

WRITTEN by scmlweb/python/update_agents_repo.py and friends. Edits
made by hand will be overwritten on the next run -- use
set_finalists.py / set_winners.py to flip the per-entry metadata
flags.

`get_participants(year, track=None, qualified_only=False,
finalists_only=False, winners_only=False)` returns the participants (or
a filtered subset) for a given year / track. `track` is required for
SCML and unused for ANL/HAN. `qualified_only` drops disqualified
entries (a qualified agent is any non-disqualified participant).
"""
from __future__ import annotations

import json
from typing import Optional


# (year, track-or-None) -> list of {class_path, metadata}.
# Stored as JSON (loaded at import) so the booleans/None serialise correctly —
# a raw Python-literal paste would emit JSON `false`/`true`/`null` and break.
_REGISTRY: dict = json.loads(r"""
{
    "2026|": [
        {
            "class_path": "han_agents.han2026.team_21049.negotiatorx.negotiator.NegotiatorX",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21049",
                "name": "NegotiatorX",
                "description": "Non-LLM behavioral, Pareto-aware negotiator for HAN 2026 with template-based natural-language messages."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21058.mi_agent.MiAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": false,
                "disqualified": true,
                "uses_llm": true,
                "has_report": false,
                "has_description": false,
                "team_id": "21058",
                "name": "MiAgent",
                "description": "LLM meta-negotiator wrapping a time-based (Boulware/Conceder/Linear) core with an Ollama model that generates brief persuasive messages for each offer."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21086.semruk.SemrukNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21086",
                "name": "Semruk",
                "description": "Hybrid Boulware/BOA negotiator on scale-invariant normalized utility, with Smith-frequency opponent modeling and template-based (non-LLM) natural-language messages."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21099.equinox.Equinox",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "21099",
                "name": "Equinox",
                "description": "Reinforcement-learning (SAC) bilateral negotiator adapted from an SCML winner, with Bayesian opponent modeling and template-based (non-LLM) messages."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21113.nexus.NEXUSNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21113",
                "name": "NEXUSNegotiator",
                "description": "Negotiator combining frequency, hard-headed, and Bayesian opponent models (with optional GNash) for the stochastic alternating-offers protocol."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21125.hannari_hamaguri_han.HannariHamaguriHAN",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21125",
                "name": "HannariHamaguriHAN",
                "description": "Hybrid HAN negotiator: a deterministic BOA strategy with frequency opponent models and fast, human-like templated text."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21146.t2agent.T2Agent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "21146",
                "name": "T2Agent",
                "description": "Risk-adaptive negotiator that sets utility targets from heuristic breakdown-risk, picks near-Pareto offers via frequency-based opponent modeling, and uses an LLM only to phrase messages."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21181.agora.AgoraNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "21181",
                "name": "AgoraAINegotiator",
                "description": "Boulware-based hybrid agent with per-issue frequency opponent modeling, a keyword-based text belief state, and an Ollama LLM writing persuasive messages."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21399.group8.Group8Negotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21399",
                "name": "Group8",
                "description": "Behavior-adaptive negotiator that tunes its concession strategy to the observed opponent's behavior."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21400.sun.Sun",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21400",
                "name": "Sun",
                "description": "BOA-based negotiator (behavior-safe baseline) with Smith-frequency opponent modeling and a rescue branch for opponents stuck on rigid minimum-bundle offers."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21405.lastoffer.LastOfferNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "21405",
                "name": "LastOffer",
                "description": "Hybrid negotiator combining a utility-based decision model, a time-dependent aspiration function, and an adaptive opponent model with endgame robustness."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21480.gunner_agent.GunnerAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "21480",
                "name": "Gunner_Agent",
                "description": "LLM meta-negotiator built on the Shochan (ANL 2024 winner) aspiration and Pareto-propose core, adding a two-phase curve, reservation-value floor, adaptive opponent model, and an LLM message layer."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21627.neonegotiator.NeoNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "21627",
                "name": "NeoNegotiator",
                "description": "LLM meta-negotiator applying six progressive tactics: issue logrolling, sentiment-adaptive tone, door-in-the-face anchoring, BATNA signaling, deadline delay, and micro-step concessions."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21656.nekotiator.Nekotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "21656",
                "name": "Nekotiator",
                "description": "LLM-based negotiator that queries an Ollama model to infer opponent issue weights and targets, then concedes from a decaying aspiration level while offering win-win counter-proposals."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21687.hagent_agent.HAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": false,
                "disqualified": true,
                "uses_llm": false,
                "has_report": false,
                "has_description": false,
                "team_id": "21687",
                "name": "hagent",
                "description": "Belief-based contextual-bandit negotiator for HAN 2026."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21709.submissionhan.adaptive_bargain.AdaptiveBargainNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "21709",
                "name": "AdaptiveBargainNegotiator",
                "description": "BOA-architecture bargaining agent combining time-dependent concession, frequency-based opponent modeling, and dynamic acceptance, with an LLM generating natural-language messages for its rule-based decisions."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21723.han_agent.entry.HanOmegaNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21723",
                "name": "CodexAgentHan",
                "description": "Non-LLM bilateral negotiator using a time-dependent Boulware aspiration policy, frequency-based opponent modeling, cycle-aware exploration, and template-generated natural-language messages."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21736.hi.han.Han",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21736",
                "name": "HiHan",
                "description": "Adaptive multi-phase negotiator that moves through exploration, Boulware concession, acceleration, and deadline phases with AC-Next acceptance."
            }
        },
        {
            "class_path": "han_agents.han2026.team_21787.hybridpisa.HybridPisaNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "21787",
                "name": "HybridPisaNegotiator",
                "description": "Hybrid time- and behavior-based concession negotiator with online Bayesian opponent modeling and a local LLM that writes natural-language messages for already-chosen bids."
            }
        },
        {
            "class_path": "han_agents.han2026.team_22147.agent96.Agent96",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "22147",
                "name": "Agent96",
                "description": "Hybrid human-negotiation agent with a deterministic, utility-safe Boulware core and frequency modeling of the human's offers; the LLM only polishes human-facing text."
            }
        },
        {
            "class_path": "han_agents.han2026.team_22262.snowy_day_v2.SnowyDayV2",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "22262",
                "name": "SnowyDayAgent",
                "description": "Self-contained, no-LLM Snowy-style negotiator whose concession behavior is driven by a configurable utility curve (CurveConfig)."
            }
        },
        {
            "class_path": "han_agents.han2026.team_22270.civic_compass.CivicCompassHANNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": false,
                "team_id": "22270",
                "name": "CivicHAN",
                "description": "Non-LLM HAN negotiator using standard NegMAS decision logic and deterministic Markdown text to communicate with human partners."
            }
        },
        {
            "class_path": "han_agents.han2026.team_22280.aegis_agent_r168.AegisR168FinalNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": true,
                "has_report": true,
                "has_description": true,
                "team_id": "22280",
                "name": "AegisAgentR168",
                "description": "BOA-style negotiator (R16.8) adding an exception-safe wrapper, very-late acceptance pickup, and anti-loop rescue offers, with the LLM used only for messages."
            }
        },
        {
            "class_path": "han_agents.han2026.team_22286.closer.CloserNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "22286",
                "name": "Closerv23",
                "description": "Hard-anchoring hybrid negotiator for human-agent negotiation in HAN 2026."
            }
        }
    ]
}
""")


def get_participants(
    year: int,
    track: Optional[str] = None,
    *,
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
) -> tuple[str, ...]:
    """Return the dotted Python paths of registered participants."""
    # _REGISTRY keys are the JSON-serialised "year|track" strings (see
    # rewrite_registry); build the same form rather than a tuple.
    key = f"{int(year)}|{track.lower() if track else ''}"
    entries = _REGISTRY.get(key, [])
    out = []
    for e in entries:
        meta = e.get("metadata", {})
        if qualified_only and meta.get("disqualified"):
            continue
        if finalists_only and not meta.get("finalist"):
            continue
        if winners_only and not meta.get("winner"):
            continue
        out.append(e["class_path"])
    return tuple(out)
