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
                "name": "NegotiatorX"
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
                "name": "Semruk"
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
                "name": "Equinox"
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
                "has_description": false,
                "team_id": "21113",
                "name": "NEXUSNegotiator"
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
                "name": "HannariHamaguriHAN"
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
                "name": "T2Agent"
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
                "name": "AgoraAINegotiator"
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
                "name": "Group8"
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
                "name": "Sun"
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
                "name": "LastOffer"
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
                "name": "Gunner_Agent"
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
                "name": "NeoNegotiator"
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
                "name": "Nekotiator"
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
                "name": "AdaptiveBargainNegotiator"
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
                "name": "CodexAgentHan"
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
                "name": "HiHan"
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
                "name": "HybridPisaNegotiator"
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
                "name": "Agent96"
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
                "name": "SnowyDayAgent"
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
                "name": "CivicHAN"
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
                "name": "AegisAgentR168"
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
                "name": "Closerv23"
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
