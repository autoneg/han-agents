# diag.py — zero-cost diagnostics for NEXUSNegotiator
# Activated only when NEXUS_DIAG=1 in the environment.
# Safe to ship in competition submissions — all code-paths are no-ops when
# ENABLED is False, so there is no performance impact in tournaments.

from __future__ import annotations

import json
import os

ENABLED: bool = os.environ.get("NEXUS_DIAG", "0") == "1"
_DIAG_PATH: str = os.environ.get("NEXUS_DIAG_PATH", "nexus_diag.jsonl")


class NegotiationDiag:
    """Accumulates per-round state for one negotiation session."""

    def __init__(self) -> None:
        self.scenario: str = "default"
        self.rv: float = 0.0
        self.n_steps_mech: int | None = None
        self.time_limit_mech: float | None = None
        self.n_outcomes: int | None = None
        self.pareto_built_round: int | None = None
        self.concession_steps: int = 0
        self.accept_reason: str | None = None
        self.opp_rv_est: float | None = None
        self.tft_mode: bool = False

        self._rounds: list[dict] = []
        self._offers_sent: list[float] = []

    def round_snapshot(
        self,
        t: float,
        achievable: float,
        concession_target: float,
        recv_util: float | None,
    ) -> None:
        self._rounds.append({
            "t": round(t, 4),
            "achievable": round(achievable, 4),
            "target": round(concession_target, 4),
            "recv": round(recv_util, 4) if recv_util is not None else None,
        })

    def offered(self, util: float) -> None:
        self._offers_sent.append(round(util, 4))

    def finalize(
        self,
        agreement,
        final_util: float,
        end_t: float,
        n_rounds: int,
    ) -> dict:
        return {
            "scenario": self.scenario,
            "rv": self.rv,
            "n_steps_mech": self.n_steps_mech,
            "time_limit_mech": self.time_limit_mech,
            "n_outcomes": self.n_outcomes,
            "n_rounds": n_rounds,
            "end_t": round(end_t, 4),
            "agreement": agreement is not None,
            "final_util": round(final_util, 4),
            "accept_reason": self.accept_reason,
            "pareto_built_round": self.pareto_built_round,
            "concession_steps": self.concession_steps,
            "opp_rv_est": (
                round(self.opp_rv_est, 4) if self.opp_rv_est is not None else None
            ),
            "tft_mode": self.tft_mode,
            "rounds": self._rounds,
            "offers_sent": self._offers_sent,
        }


def log_negotiation(record: dict) -> None:
    """Append one JSON record to the diagnostics file (line-atomic)."""
    try:
        with open(_DIAG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
