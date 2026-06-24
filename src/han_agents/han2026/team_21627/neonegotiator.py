"""
HAN 2026 Negotiating Agent — NeoNegotiator

Extends LLMMetaNegotiator with six progressive strategies:
  1. Issue Logrolling         — concede on issues opponent values most
  2. Sentiment Analysis       — classify opponent emotion from text; adapt LLM tone
  3. Door-in-the-Face         — anchor with max-utility opening; frame first concession
  4. BATNA Signaling          — hint at walk-away constraints when opponent is stubborn late
  5. Deadline Delay           — block early acceptance; force agreement into panic zone
  6. Micro-Step Concessions   — tiny utility drops per round; build up concession count

Bidding Strategies (change ACTIVE_STRATEGY to switch)
------------------------------------------------------
 1. BOULWARE        — Slow-then-fast concession via NegMAS BoulwareTBNegotiator.
 2. CONCEDER        — Concedes quickly early; flattens before the deadline.
 3. LINEAR          — Uniform linear decline from max to reserved utility.
 4. TDT             — Time-Dependent Tactic; generalises all polynomial curves via β.
 5. TIT_FOR_TAT     — Mirrors opponent's last absolute concession.
 6. RELATIVE_TFT    — Mirrors opponent's relative concession rate.
 7. NICE_TFT        — TfT that opens slightly below max to signal goodwill.
 8. DEADLINE_DRIVEN — Holds max utility until a late switch-point, then dives.
 9. RANDOM_WALK     — Adds Gaussian noise around a linear baseline.
10. BAYESIAN        — Frequency-model-based optimal offer search.
11. KDE             — Kernel Density Estimation of opponent acceptance zone.
12. TRADE_OFF       — Linear scalarisation shifting from self-interested to fair.
13. PARETO          — Samples the Pareto frontier and navigates along it.
"""
from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import Any

from negmas.common import Outcome
from negmas.gb import BoulwareTBNegotiator
from negmas.gb.common import ExtendedResponseType, ResponseType
from negmas.outcomes import ExtendedOutcome
from negmas.sao import SAOState
from negmas_llm.meta import LLMMetaNegotiator

try:
    from scipy.stats import gaussian_kde as _gaussian_kde
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# Bidding Strategies
# ═════════════════════════════════════════════════════════════════════════════

class StrategyType(str, Enum):
    BOULWARE        = "boulware"
    CONCEDER        = "conceder"
    LINEAR          = "linear"
    TDT             = "tdt"
    TIT_FOR_TAT     = "tit_for_tat"
    RELATIVE_TFT    = "relative_tft"
    NICE_TFT        = "nice_tft"
    DEADLINE_DRIVEN = "deadline_driven"
    RANDOM_WALK     = "random_walk"
    BAYESIAN        = "bayesian"
    KDE             = "kde"
    TRADE_OFF       = "trade_off"
    PARETO          = "pareto"


# ── Change this variable to switch the active bidding strategy ────────────────
ACTIVE_STRATEGY: StrategyType = StrategyType.BOULWARE


class BiddingStrategy(ABC):
    """Controls which outcome NeoNegotiator offers each round.

    Acceptance decisions remain with BoulwareTBNegotiator's respond() logic.
    """

    def on_negotiation_start(self, negotiator: "NeoNegotiator") -> None:
        """Reset per-session state when a new negotiation begins."""

    def on_opponent_offer(self, offer: Outcome, negotiator: "NeoNegotiator") -> None:
        """Update internal model when the opponent makes an offer."""

    @abstractmethod
    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        """Return the next outcome to offer, or None to pass."""


# ── Shared helpers ────────────────────────────────────────────────────────────

def _bs_utility_range(negotiator: "NeoNegotiator") -> tuple[float, float]:
    if negotiator.ufun is None:
        return 0.0, 1.0
    reserved = (
        float(negotiator.ufun.reserved_value)
        if negotiator.ufun.reserved_value is not None
        else 0.0
    )
    best = negotiator.ufun.best()
    max_u = float(negotiator.ufun(best)) if best is not None else 1.0
    return reserved, max_u


def _bs_find_outcome(
    target_u: float,
    negotiator: "NeoNegotiator",
    max_cardinality: int = 500,
) -> Outcome | None:
    """Return the outcome whose utility is closest to target_u."""
    if negotiator.ufun is None or negotiator.nmi is None:
        return None
    os = negotiator.nmi.outcome_space
    if os is None:
        return None
    best_outcome: Outcome | None = None
    best_diff = float("inf")
    for outcome in os.enumerate_or_sample(max_cardinality=max_cardinality):
        if outcome is None:
            continue
        diff = abs(float(negotiator.ufun(outcome)) - target_u)
        if diff < best_diff:
            best_diff = diff
            best_outcome = outcome
    return best_outcome


# ── 1. Boulware ───────────────────────────────────────────────────────────────

class BoulwareStrategy(BiddingStrategy):
    """Polynomial approximation: stays near max utility, concedes sharply near deadline.

    NeoNegotiator uses BoulwareTBNegotiator directly for BOULWARE (exact NegMAS
    curve).  This class is a self-contained polynomial stand-in for standalone use.
    Formula: u(t) = min_u + (max_u - min_u) * (1 - t^β)  with β >> 1.
    """

    def __init__(self, exponent: float = 5.0) -> None:
        self.exponent = exponent

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        t = state.relative_time
        min_u, max_u = _bs_utility_range(negotiator)
        return _bs_find_outcome(min_u + (max_u - min_u) * (1.0 - t ** self.exponent), negotiator)


# ── 2. Conceder ───────────────────────────────────────────────────────────────

class ConcederStrategy(BiddingStrategy):
    """Concedes quickly early, flattening near the target before the deadline.

    Formula: u(t) = min_u + (max_u - min_u) * (1 - t^β)  with 0 < β < 1.
    """

    def __init__(self, exponent: float = 0.2) -> None:
        self.exponent = exponent

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        t = state.relative_time
        min_u, max_u = _bs_utility_range(negotiator)
        return _bs_find_outcome(min_u + (max_u - min_u) * (1.0 - t ** self.exponent), negotiator)


# ── 3. Linear Conceder ────────────────────────────────────────────────────────

class LinearStrategy(BiddingStrategy):
    """Decreases offered utility uniformly from max_u to min_u over time."""

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        t = state.relative_time
        min_u, max_u = _bs_utility_range(negotiator)
        return _bs_find_outcome(max_u - (max_u - min_u) * t, negotiator)


# ── 4. TDT — Time-Dependent Tactic ───────────────────────────────────────────

class TDTStrategy(BiddingStrategy):
    """Generalised time-dependent tactic parameterised by β.

    β > 1  →  Boulware-like (concedes late)
    β = 1  →  Linear
    β < 1  →  Conceder-like (concedes early)
    """

    def __init__(self, exponent: float = 2.0) -> None:
        self.exponent = exponent

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        t = state.relative_time
        min_u, max_u = _bs_utility_range(negotiator)
        return _bs_find_outcome(min_u + (max_u - min_u) * (1.0 - t ** self.exponent), negotiator)


# ── 5. Tit for Tat ────────────────────────────────────────────────────────────

class TitForTatStrategy(BiddingStrategy):
    """Mirrors the opponent's last absolute concession.

    Each time the opponent improves their offer (from our utility perspective)
    we concede by the same absolute amount.  Starts at max utility.
    """

    def __init__(self) -> None:
        self._prev_opp_util: float | None = None
        self._target_util: float | None = None

    def on_negotiation_start(self, negotiator: "NeoNegotiator") -> None:
        self._prev_opp_util = None
        self._target_util = None

    def on_opponent_offer(self, offer: Outcome, negotiator: "NeoNegotiator") -> None:
        if negotiator.ufun is None:
            return
        cur = float(negotiator.ufun(offer))
        if self._prev_opp_util is not None and self._target_util is not None:
            concession = cur - self._prev_opp_util
            if concession > 0:
                min_u, _ = _bs_utility_range(negotiator)
                self._target_util = max(self._target_util - concession, min_u)
        self._prev_opp_util = cur

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        if self._target_util is None:
            _, max_u = _bs_utility_range(negotiator)
            self._target_util = max_u
        return _bs_find_outcome(self._target_util, negotiator)


# ── 6. Relative Tit for Tat ───────────────────────────────────────────────────

class RelativeTitForTatStrategy(BiddingStrategy):
    """Mirrors the opponent's relative concession rate.

    When the opponent concedes x% of the remaining gap, we concede the same
    x% of our own remaining gap above the reserved value.
    """

    def __init__(self) -> None:
        self._prev_opp_util: float | None = None
        self._target_util: float | None = None

    def on_negotiation_start(self, negotiator: "NeoNegotiator") -> None:
        self._prev_opp_util = None
        self._target_util = None

    def on_opponent_offer(self, offer: Outcome, negotiator: "NeoNegotiator") -> None:
        if negotiator.ufun is None:
            return
        _, max_u = _bs_utility_range(negotiator)
        cur = float(negotiator.ufun(offer))
        if self._prev_opp_util is not None and self._target_util is not None:
            gap = max_u - self._prev_opp_util
            if gap > 1e-6:
                rel = (cur - self._prev_opp_util) / gap
                if rel > 0:
                    min_u, _ = _bs_utility_range(negotiator)
                    self._target_util = max(
                        self._target_util - rel * (self._target_util - min_u), min_u
                    )
        self._prev_opp_util = cur

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        if self._target_util is None:
            _, max_u = _bs_utility_range(negotiator)
            self._target_util = max_u
        return _bs_find_outcome(self._target_util, negotiator)


# ── 7. Nice Tit for Tat ───────────────────────────────────────────────────────

class NiceTitForTatStrategy(BiddingStrategy):
    """Cooperative TfT: opens slightly below max to signal goodwill, then mirrors."""

    def __init__(self, initial_discount: float = 0.05) -> None:
        self.initial_discount = initial_discount
        self._prev_opp_util: float | None = None
        self._target_util: float | None = None

    def on_negotiation_start(self, negotiator: "NeoNegotiator") -> None:
        self._prev_opp_util = None
        self._target_util = None

    def on_opponent_offer(self, offer: Outcome, negotiator: "NeoNegotiator") -> None:
        if negotiator.ufun is None:
            return
        cur = float(negotiator.ufun(offer))
        if self._prev_opp_util is not None and self._target_util is not None:
            concession = cur - self._prev_opp_util
            if concession > 0:
                min_u, _ = _bs_utility_range(negotiator)
                self._target_util = max(self._target_util - concession, min_u)
        self._prev_opp_util = cur

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        if self._target_util is None:
            min_u, max_u = _bs_utility_range(negotiator)
            self._target_util = max_u - self.initial_discount * (max_u - min_u)
        return _bs_find_outcome(self._target_util, negotiator)


# ── 8. Deadline-Driven Tactic (DDT) ──────────────────────────────────────────

class DeadlineDrivenStrategy(BiddingStrategy):
    """Holds near max utility until a late switch-point, then concedes sharply."""

    def __init__(self, switch_time: float = 0.8, exponent: float = 2.0) -> None:
        self.switch_time = switch_time
        self.exponent = exponent

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        t = state.relative_time
        min_u, max_u = _bs_utility_range(negotiator)
        if t < self.switch_time:
            target = max_u
        else:
            span = max(1.0 - self.switch_time, 1e-6)
            normalized = (t - self.switch_time) / span
            target = max(max_u - (max_u - min_u) * (normalized ** self.exponent), min_u)
        return _bs_find_outcome(target, negotiator)


# ── 9. Random Walk ────────────────────────────────────────────────────────────

class RandomWalkStrategy(BiddingStrategy):
    """Adds Gaussian noise around a linear concession baseline."""

    def __init__(self, noise_std: float = 0.04, seed: int | None = None) -> None:
        self.noise_std = noise_std
        self._rng = random.Random(seed)
        self._current_util: float | None = None

    def on_negotiation_start(self, negotiator: "NeoNegotiator") -> None:
        self._current_util = None

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        t = state.relative_time
        min_u, max_u = _bs_utility_range(negotiator)
        if self._current_util is None:
            self._current_util = max_u
        linear_target = max_u - (max_u - min_u) * t
        drift = (linear_target - self._current_util) * 0.3
        noise = self._rng.gauss(0.0, self.noise_std * (max_u - min_u))
        self._current_util = max(min_u, min(max_u, self._current_util + drift + noise))
        return _bs_find_outcome(self._current_util, negotiator)


# ── 10. Bayesian Learning ─────────────────────────────────────────────────────

class BayesianStrategy(BiddingStrategy):
    """Maximises our utility among outcomes the opponent is likely to accept.

    Uses NeoNegotiator's frequency model as a Bayesian prior on acceptance.
    The minimum aspiration utility decreases over time to ensure a deal.
    """

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        if negotiator.ufun is None or negotiator.nmi is None:
            return None
        t = state.relative_time
        min_u, max_u = _bs_utility_range(negotiator)
        aspiration = min_u + (max_u - min_u) * (1.0 - t ** 5)
        os = negotiator.nmi.outcome_space
        if os is None:
            return None
        best: Outcome | None = None
        best_score = -1.0
        for outcome in os.enumerate_or_sample(max_cardinality=500):
            if outcome is None:
                continue
            my_u = float(negotiator.ufun(outcome))
            if my_u < aspiration:
                continue
            score = my_u * (0.5 + negotiator._estimate_opp_util(outcome))
            if score > best_score:
                best_score = score
                best = outcome
        return best if best is not None else _bs_find_outcome(aspiration, negotiator)


# ── 11. KDE Strategy ──────────────────────────────────────────────────────────

class KDEStrategy(BiddingStrategy):
    """Kernel Density Estimation of the opponent's acceptance zone.

    Builds a KDE over the opponent's estimated utility values for their past
    offers.  Scores candidates by our_utility × (1 + kde_density).
    Requires scipy; falls back to frequency-weighted scoring if unavailable.
    """

    def __init__(self) -> None:
        self._opp_estimated_utils: list[float] = []

    def on_negotiation_start(self, negotiator: "NeoNegotiator") -> None:
        self._opp_estimated_utils = []

    def on_opponent_offer(self, offer: Outcome, negotiator: "NeoNegotiator") -> None:
        est = negotiator._estimate_opp_util(offer)
        if est > 0.0:
            self._opp_estimated_utils.append(est)

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        if negotiator.ufun is None or negotiator.nmi is None:
            return None
        t = state.relative_time
        min_u, max_u = _bs_utility_range(negotiator)
        aspiration = min_u + (max_u - min_u) * (1.0 - t ** 5)
        os = negotiator.nmi.outcome_space
        if os is None:
            return None
        kde = None
        if _SCIPY_AVAILABLE and len(self._opp_estimated_utils) >= 5:
            try:
                kde = _gaussian_kde(self._opp_estimated_utils, bw_method="silverman")
            except Exception:
                pass
        best: Outcome | None = None
        best_score = -1.0
        for outcome in os.enumerate_or_sample(max_cardinality=500):
            if outcome is None:
                continue
            my_u = float(negotiator.ufun(outcome))
            if my_u < aspiration:
                continue
            opp_u = negotiator._estimate_opp_util(outcome)
            if kde is not None:
                try:
                    density = float(kde.evaluate([opp_u])[0])
                except Exception:
                    density = opp_u
            else:
                density = opp_u
            score = my_u * (1.0 + density)
            if score > best_score:
                best_score = score
                best = outcome
        return best if best is not None else _bs_find_outcome(aspiration, negotiator)


# ── 12. Trade-Off Strategy ────────────────────────────────────────────────────

class TradeOffStrategy(BiddingStrategy):
    """Linear scalarisation shifting from self-interested to fair over time.

    Score = α(t) × my_util + (1 - α(t)) × opp_util_estimate
    α decreases from initial_weight to final_weight as the deadline approaches.
    """

    def __init__(self, initial_weight: float = 0.9, final_weight: float = 0.5) -> None:
        self.initial_weight = initial_weight
        self.final_weight = final_weight

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        if negotiator.ufun is None or negotiator.nmi is None:
            return None
        t = state.relative_time
        alpha = self.initial_weight + (self.final_weight - self.initial_weight) * t
        min_u, _ = _bs_utility_range(negotiator)
        os = negotiator.nmi.outcome_space
        if os is None:
            return None
        best: Outcome | None = None
        best_score = -1.0
        for outcome in os.enumerate_or_sample(max_cardinality=500):
            if outcome is None:
                continue
            my_u = float(negotiator.ufun(outcome))
            if my_u < min_u:
                continue
            score = alpha * my_u + (1.0 - alpha) * negotiator._estimate_opp_util(outcome)
            if score > best_score:
                best_score = score
                best = outcome
        return best


# ── 13. Pareto-Frontier Search ────────────────────────────────────────────────

class ParetoFrontierStrategy(BiddingStrategy):
    """Approximates the Pareto frontier and navigates along it over time.

    Samples up to sample_size outcomes, identifies the Pareto-optimal subset,
    then selects the Pareto outcome closest to our current utility aspiration.
    Aspiration decreases linearly from max to reserved utility.
    """

    def __init__(self, sample_size: int = 500) -> None:
        self.sample_size = sample_size

    def propose(self, state: SAOState, negotiator: "NeoNegotiator") -> Outcome | None:
        if negotiator.ufun is None or negotiator.nmi is None:
            return None
        t = state.relative_time
        min_u, max_u = _bs_utility_range(negotiator)
        aspiration = min_u + (max_u - min_u) * max(0.0, 1.0 - t)
        os = negotiator.nmi.outcome_space
        if os is None:
            return None
        scored: list[tuple[float, float, Outcome]] = []
        for outcome in os.enumerate_or_sample(max_cardinality=self.sample_size):
            if outcome is None:
                continue
            scored.append((
                float(negotiator.ufun(outcome)),
                negotiator._estimate_opp_util(outcome),
                outcome,
            ))
        if not scored:
            return None
        pareto = [
            (mu, ou, oc) for i, (mu, ou, oc) in enumerate(scored)
            if not any(
                j != i and mj >= mu and oj >= ou and (mj > mu or oj > ou)
                for j, (mj, oj, _) in enumerate(scored)
            )
        ] or scored
        best: Outcome | None = None
        best_diff = float("inf")
        for my_u, _, outcome in pareto:
            diff = abs(my_u - aspiration)
            if diff < best_diff:
                best_diff = diff
                best = outcome
        return best


# ── Factory ───────────────────────────────────────────────────────────────────

_STRATEGY_MAP: dict[StrategyType, type[BiddingStrategy]] = {
    StrategyType.BOULWARE:        BoulwareStrategy,
    StrategyType.CONCEDER:        ConcederStrategy,
    StrategyType.LINEAR:          LinearStrategy,
    StrategyType.TDT:             TDTStrategy,
    StrategyType.TIT_FOR_TAT:     TitForTatStrategy,
    StrategyType.RELATIVE_TFT:    RelativeTitForTatStrategy,
    StrategyType.NICE_TFT:        NiceTitForTatStrategy,
    StrategyType.DEADLINE_DRIVEN: DeadlineDrivenStrategy,
    StrategyType.RANDOM_WALK:     RandomWalkStrategy,
    StrategyType.BAYESIAN:        BayesianStrategy,
    StrategyType.KDE:             KDEStrategy,
    StrategyType.TRADE_OFF:       TradeOffStrategy,
    StrategyType.PARETO:          ParetoFrontierStrategy,
}


def _create_strategy(strategy_type: StrategyType) -> BiddingStrategy:
    cls = _STRATEGY_MAP.get(strategy_type)
    if cls is None:
        raise ValueError(f"Unknown strategy type: {strategy_type!r}")
    return cls()


# ── Counter-strategy table ─────────────────────────────────────────────────────
# Keyed by (character.upper(), emotion.upper(), detected_StrategyType).
# When the opponent sends no text, character defaults to "Compromising" and
# emotion defaults to "Neutral" — which maps to the COMPROMISING/NEUTRAL rows.
_ST = StrategyType   # brevity alias used only in this block

COUNTER_TABLE: dict[tuple[str, str, StrategyType], StrategyType] = {
    # ── COMPETING / FRUSTRATED ────────────────────────────────────────────────
    ("COMPETING","FRUSTRATED",_ST.BOULWARE):        _ST.DEADLINE_DRIVEN,
    ("COMPETING","FRUSTRATED",_ST.CONCEDER):        _ST.BOULWARE,
    ("COMPETING","FRUSTRATED",_ST.LINEAR):          _ST.BOULWARE,
    ("COMPETING","FRUSTRATED",_ST.TDT):             _ST.BOULWARE,
    ("COMPETING","FRUSTRATED",_ST.TIT_FOR_TAT):     _ST.BOULWARE,
    ("COMPETING","FRUSTRATED",_ST.RELATIVE_TFT):    _ST.BOULWARE,
    ("COMPETING","FRUSTRATED",_ST.NICE_TFT):        _ST.BOULWARE,
    ("COMPETING","FRUSTRATED",_ST.DEADLINE_DRIVEN): _ST.DEADLINE_DRIVEN,
    ("COMPETING","FRUSTRATED",_ST.RANDOM_WALK):     _ST.KDE,
    ("COMPETING","FRUSTRATED",_ST.BAYESIAN):        _ST.RANDOM_WALK,
    ("COMPETING","FRUSTRATED",_ST.KDE):             _ST.RANDOM_WALK,
    ("COMPETING","FRUSTRATED",_ST.TRADE_OFF):       _ST.BOULWARE,
    ("COMPETING","FRUSTRATED",_ST.PARETO):          _ST.BOULWARE,
    # ── COMPETING / HAPPY ─────────────────────────────────────────────────────
    ("COMPETING","HAPPY",_ST.BOULWARE):        _ST.NICE_TFT,
    ("COMPETING","HAPPY",_ST.CONCEDER):        _ST.BOULWARE,
    ("COMPETING","HAPPY",_ST.LINEAR):          _ST.BOULWARE,
    ("COMPETING","HAPPY",_ST.TDT):             _ST.NICE_TFT,
    ("COMPETING","HAPPY",_ST.TIT_FOR_TAT):     _ST.NICE_TFT,
    ("COMPETING","HAPPY",_ST.RELATIVE_TFT):    _ST.NICE_TFT,
    ("COMPETING","HAPPY",_ST.NICE_TFT):        _ST.TRADE_OFF,
    ("COMPETING","HAPPY",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("COMPETING","HAPPY",_ST.RANDOM_WALK):     _ST.KDE,
    ("COMPETING","HAPPY",_ST.BAYESIAN):        _ST.RANDOM_WALK,
    ("COMPETING","HAPPY",_ST.KDE):             _ST.RANDOM_WALK,
    ("COMPETING","HAPPY",_ST.TRADE_OFF):       _ST.PARETO,
    ("COMPETING","HAPPY",_ST.PARETO):          _ST.BAYESIAN,
    # ── COMPETING / ANXIOUS ───────────────────────────────────────────────────
    ("COMPETING","ANXIOUS",_ST.BOULWARE):        _ST.DEADLINE_DRIVEN,
    ("COMPETING","ANXIOUS",_ST.CONCEDER):        _ST.BOULWARE,
    ("COMPETING","ANXIOUS",_ST.LINEAR):          _ST.BOULWARE,
    ("COMPETING","ANXIOUS",_ST.TDT):             _ST.DEADLINE_DRIVEN,
    ("COMPETING","ANXIOUS",_ST.TIT_FOR_TAT):     _ST.BOULWARE,
    ("COMPETING","ANXIOUS",_ST.RELATIVE_TFT):    _ST.BOULWARE,
    ("COMPETING","ANXIOUS",_ST.NICE_TFT):        _ST.DEADLINE_DRIVEN,
    ("COMPETING","ANXIOUS",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("COMPETING","ANXIOUS",_ST.RANDOM_WALK):     _ST.KDE,
    ("COMPETING","ANXIOUS",_ST.BAYESIAN):        _ST.RANDOM_WALK,
    ("COMPETING","ANXIOUS",_ST.KDE):             _ST.RANDOM_WALK,
    ("COMPETING","ANXIOUS",_ST.TRADE_OFF):       _ST.DEADLINE_DRIVEN,
    ("COMPETING","ANXIOUS",_ST.PARETO):          _ST.DEADLINE_DRIVEN,
    # ── COMPETING / NEUTRAL ───────────────────────────────────────────────────
    ("COMPETING","NEUTRAL",_ST.BOULWARE):        _ST.DEADLINE_DRIVEN,
    ("COMPETING","NEUTRAL",_ST.CONCEDER):        _ST.BOULWARE,
    ("COMPETING","NEUTRAL",_ST.LINEAR):          _ST.BOULWARE,
    ("COMPETING","NEUTRAL",_ST.TDT):             _ST.BAYESIAN,
    ("COMPETING","NEUTRAL",_ST.TIT_FOR_TAT):     _ST.NICE_TFT,
    ("COMPETING","NEUTRAL",_ST.RELATIVE_TFT):    _ST.BOULWARE,
    ("COMPETING","NEUTRAL",_ST.NICE_TFT):        _ST.BOULWARE,
    ("COMPETING","NEUTRAL",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("COMPETING","NEUTRAL",_ST.RANDOM_WALK):     _ST.KDE,
    ("COMPETING","NEUTRAL",_ST.BAYESIAN):        _ST.RANDOM_WALK,
    ("COMPETING","NEUTRAL",_ST.KDE):             _ST.RANDOM_WALK,
    ("COMPETING","NEUTRAL",_ST.TRADE_OFF):       _ST.PARETO,
    ("COMPETING","NEUTRAL",_ST.PARETO):          _ST.BAYESIAN,
    # ── COLLABORATING / FRUSTRATED ────────────────────────────────────────────
    ("COLLABORATING","FRUSTRATED",_ST.BOULWARE):        _ST.NICE_TFT,
    ("COLLABORATING","FRUSTRATED",_ST.CONCEDER):        _ST.TRADE_OFF,
    ("COLLABORATING","FRUSTRATED",_ST.LINEAR):          _ST.NICE_TFT,
    ("COLLABORATING","FRUSTRATED",_ST.TDT):             _ST.NICE_TFT,
    ("COLLABORATING","FRUSTRATED",_ST.TIT_FOR_TAT):     _ST.NICE_TFT,
    ("COLLABORATING","FRUSTRATED",_ST.RELATIVE_TFT):    _ST.NICE_TFT,
    ("COLLABORATING","FRUSTRATED",_ST.NICE_TFT):        _ST.TRADE_OFF,
    ("COLLABORATING","FRUSTRATED",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("COLLABORATING","FRUSTRATED",_ST.RANDOM_WALK):     _ST.KDE,
    ("COLLABORATING","FRUSTRATED",_ST.BAYESIAN):        _ST.TRADE_OFF,
    ("COLLABORATING","FRUSTRATED",_ST.KDE):             _ST.TRADE_OFF,
    ("COLLABORATING","FRUSTRATED",_ST.TRADE_OFF):       _ST.PARETO,
    ("COLLABORATING","FRUSTRATED",_ST.PARETO):          _ST.BAYESIAN,
    # ── COLLABORATING / HAPPY ─────────────────────────────────────────────────
    ("COLLABORATING","HAPPY",_ST.BOULWARE):        _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.CONCEDER):        _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.LINEAR):          _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.TDT):             _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.TIT_FOR_TAT):     _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.RELATIVE_TFT):    _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.NICE_TFT):        _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.DEADLINE_DRIVEN): _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.RANDOM_WALK):     _ST.KDE,
    ("COLLABORATING","HAPPY",_ST.BAYESIAN):        _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.KDE):             _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.TRADE_OFF):       _ST.PARETO,
    ("COLLABORATING","HAPPY",_ST.PARETO):          _ST.PARETO,
    # ── COLLABORATING / ANXIOUS ───────────────────────────────────────────────
    ("COLLABORATING","ANXIOUS",_ST.BOULWARE):        _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.CONCEDER):        _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.LINEAR):          _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.TDT):             _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.TIT_FOR_TAT):     _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.RELATIVE_TFT):    _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.NICE_TFT):        _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.RANDOM_WALK):     _ST.KDE,
    ("COLLABORATING","ANXIOUS",_ST.BAYESIAN):        _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.KDE):             _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.TRADE_OFF):       _ST.LINEAR,
    ("COLLABORATING","ANXIOUS",_ST.PARETO):          _ST.LINEAR,
    # ── COLLABORATING / NEUTRAL ───────────────────────────────────────────────
    ("COLLABORATING","NEUTRAL",_ST.BOULWARE):        _ST.BAYESIAN,
    ("COLLABORATING","NEUTRAL",_ST.CONCEDER):        _ST.PARETO,
    ("COLLABORATING","NEUTRAL",_ST.LINEAR):          _ST.PARETO,
    ("COLLABORATING","NEUTRAL",_ST.TDT):             _ST.BAYESIAN,
    ("COLLABORATING","NEUTRAL",_ST.TIT_FOR_TAT):     _ST.NICE_TFT,
    ("COLLABORATING","NEUTRAL",_ST.RELATIVE_TFT):    _ST.PARETO,
    ("COLLABORATING","NEUTRAL",_ST.NICE_TFT):        _ST.PARETO,
    ("COLLABORATING","NEUTRAL",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("COLLABORATING","NEUTRAL",_ST.RANDOM_WALK):     _ST.KDE,
    ("COLLABORATING","NEUTRAL",_ST.BAYESIAN):        _ST.RANDOM_WALK,
    ("COLLABORATING","NEUTRAL",_ST.KDE):             _ST.TRADE_OFF,
    ("COLLABORATING","NEUTRAL",_ST.TRADE_OFF):       _ST.PARETO,
    ("COLLABORATING","NEUTRAL",_ST.PARETO):          _ST.BAYESIAN,
    # ── COMPROMISING / FRUSTRATED ─────────────────────────────────────────────
    ("COMPROMISING","FRUSTRATED",_ST.BOULWARE):        _ST.TIT_FOR_TAT,
    ("COMPROMISING","FRUSTRATED",_ST.CONCEDER):        _ST.BOULWARE,
    ("COMPROMISING","FRUSTRATED",_ST.LINEAR):          _ST.BOULWARE,
    ("COMPROMISING","FRUSTRATED",_ST.TDT):             _ST.TIT_FOR_TAT,
    ("COMPROMISING","FRUSTRATED",_ST.TIT_FOR_TAT):     _ST.NICE_TFT,
    ("COMPROMISING","FRUSTRATED",_ST.RELATIVE_TFT):    _ST.TIT_FOR_TAT,
    ("COMPROMISING","FRUSTRATED",_ST.NICE_TFT):        _ST.TIT_FOR_TAT,
    ("COMPROMISING","FRUSTRATED",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("COMPROMISING","FRUSTRATED",_ST.RANDOM_WALK):     _ST.KDE,
    ("COMPROMISING","FRUSTRATED",_ST.BAYESIAN):        _ST.RANDOM_WALK,
    ("COMPROMISING","FRUSTRATED",_ST.KDE):             _ST.RANDOM_WALK,
    ("COMPROMISING","FRUSTRATED",_ST.TRADE_OFF):       _ST.PARETO,
    ("COMPROMISING","FRUSTRATED",_ST.PARETO):          _ST.BAYESIAN,
    # ── COMPROMISING / HAPPY ──────────────────────────────────────────────────
    ("COMPROMISING","HAPPY",_ST.BOULWARE):        _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.CONCEDER):        _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.LINEAR):          _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.TDT):             _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.TIT_FOR_TAT):     _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.RELATIVE_TFT):    _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.NICE_TFT):        _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.DEADLINE_DRIVEN): _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.RANDOM_WALK):     _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.BAYESIAN):        _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.KDE):             _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.TRADE_OFF):       _ST.BOULWARE,
    ("COMPROMISING","HAPPY",_ST.PARETO):          _ST.BOULWARE,
    # ── COMPROMISING / ANXIOUS ────────────────────────────────────────────────
    ("COMPROMISING","ANXIOUS",_ST.BOULWARE):        _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.CONCEDER):        _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.LINEAR):          _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.TDT):             _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.TIT_FOR_TAT):     _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.RELATIVE_TFT):    _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.NICE_TFT):        _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.RANDOM_WALK):     _ST.KDE,
    ("COMPROMISING","ANXIOUS",_ST.BAYESIAN):        _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.KDE):             _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.TRADE_OFF):       _ST.LINEAR,
    ("COMPROMISING","ANXIOUS",_ST.PARETO):          _ST.LINEAR,
    # ── COMPROMISING / NEUTRAL (default — used when opponent sends no text) ───
    ("COMPROMISING","NEUTRAL",_ST.BOULWARE):        _ST.DEADLINE_DRIVEN,
    ("COMPROMISING","NEUTRAL",_ST.CONCEDER):        _ST.BOULWARE,
    ("COMPROMISING","NEUTRAL",_ST.LINEAR):          _ST.BOULWARE,
    ("COMPROMISING","NEUTRAL",_ST.TDT):             _ST.BAYESIAN,
    ("COMPROMISING","NEUTRAL",_ST.TIT_FOR_TAT):     _ST.TDT,
    ("COMPROMISING","NEUTRAL",_ST.RELATIVE_TFT):    _ST.BOULWARE,
    ("COMPROMISING","NEUTRAL",_ST.NICE_TFT):        _ST.BOULWARE,
    ("COMPROMISING","NEUTRAL",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("COMPROMISING","NEUTRAL",_ST.RANDOM_WALK):     _ST.KDE,
    ("COMPROMISING","NEUTRAL",_ST.BAYESIAN):        _ST.KDE,
    ("COMPROMISING","NEUTRAL",_ST.KDE):             _ST.RANDOM_WALK,
    ("COMPROMISING","NEUTRAL",_ST.TRADE_OFF):       _ST.PARETO,
    ("COMPROMISING","NEUTRAL",_ST.PARETO):          _ST.BAYESIAN,
    # ── AVOIDING / FRUSTRATED ─────────────────────────────────────────────────
    ("AVOIDING","FRUSTRATED",_ST.BOULWARE):        _ST.CONCEDER,
    ("AVOIDING","FRUSTRATED",_ST.CONCEDER):        _ST.CONCEDER,
    ("AVOIDING","FRUSTRATED",_ST.LINEAR):          _ST.CONCEDER,
    ("AVOIDING","FRUSTRATED",_ST.TDT):             _ST.CONCEDER,
    ("AVOIDING","FRUSTRATED",_ST.TIT_FOR_TAT):     _ST.NICE_TFT,
    ("AVOIDING","FRUSTRATED",_ST.RELATIVE_TFT):    _ST.CONCEDER,
    ("AVOIDING","FRUSTRATED",_ST.NICE_TFT):        _ST.TRADE_OFF,
    ("AVOIDING","FRUSTRATED",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("AVOIDING","FRUSTRATED",_ST.RANDOM_WALK):     _ST.CONCEDER,
    ("AVOIDING","FRUSTRATED",_ST.BAYESIAN):        _ST.CONCEDER,
    ("AVOIDING","FRUSTRATED",_ST.KDE):             _ST.CONCEDER,
    ("AVOIDING","FRUSTRATED",_ST.TRADE_OFF):       _ST.CONCEDER,
    ("AVOIDING","FRUSTRATED",_ST.PARETO):          _ST.TRADE_OFF,
    # ── AVOIDING / HAPPY ──────────────────────────────────────────────────────
    ("AVOIDING","HAPPY",_ST.BOULWARE):        _ST.TRADE_OFF,
    ("AVOIDING","HAPPY",_ST.CONCEDER):        _ST.TRADE_OFF,
    ("AVOIDING","HAPPY",_ST.LINEAR):          _ST.NICE_TFT,
    ("AVOIDING","HAPPY",_ST.TDT):             _ST.TRADE_OFF,
    ("AVOIDING","HAPPY",_ST.TIT_FOR_TAT):     _ST.NICE_TFT,
    ("AVOIDING","HAPPY",_ST.RELATIVE_TFT):    _ST.NICE_TFT,
    ("AVOIDING","HAPPY",_ST.NICE_TFT):        _ST.TRADE_OFF,
    ("AVOIDING","HAPPY",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("AVOIDING","HAPPY",_ST.RANDOM_WALK):     _ST.KDE,
    ("AVOIDING","HAPPY",_ST.BAYESIAN):        _ST.TRADE_OFF,
    ("AVOIDING","HAPPY",_ST.KDE):             _ST.TRADE_OFF,
    ("AVOIDING","HAPPY",_ST.TRADE_OFF):       _ST.PARETO,
    ("AVOIDING","HAPPY",_ST.PARETO):          _ST.TRADE_OFF,
    # ── AVOIDING / ANXIOUS ────────────────────────────────────────────────────
    ("AVOIDING","ANXIOUS",_ST.BOULWARE):        _ST.CONCEDER,
    ("AVOIDING","ANXIOUS",_ST.CONCEDER):        _ST.LINEAR,
    ("AVOIDING","ANXIOUS",_ST.LINEAR):          _ST.LINEAR,
    ("AVOIDING","ANXIOUS",_ST.TDT):             _ST.LINEAR,
    ("AVOIDING","ANXIOUS",_ST.TIT_FOR_TAT):     _ST.CONCEDER,
    ("AVOIDING","ANXIOUS",_ST.RELATIVE_TFT):    _ST.CONCEDER,
    ("AVOIDING","ANXIOUS",_ST.NICE_TFT):        _ST.LINEAR,
    ("AVOIDING","ANXIOUS",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("AVOIDING","ANXIOUS",_ST.RANDOM_WALK):     _ST.CONCEDER,
    ("AVOIDING","ANXIOUS",_ST.BAYESIAN):        _ST.CONCEDER,
    ("AVOIDING","ANXIOUS",_ST.KDE):             _ST.CONCEDER,
    ("AVOIDING","ANXIOUS",_ST.TRADE_OFF):       _ST.LINEAR,
    ("AVOIDING","ANXIOUS",_ST.PARETO):          _ST.CONCEDER,
    # ── AVOIDING / NEUTRAL ────────────────────────────────────────────────────
    ("AVOIDING","NEUTRAL",_ST.BOULWARE):        _ST.DEADLINE_DRIVEN,
    ("AVOIDING","NEUTRAL",_ST.CONCEDER):        _ST.DEADLINE_DRIVEN,
    ("AVOIDING","NEUTRAL",_ST.LINEAR):          _ST.DEADLINE_DRIVEN,
    ("AVOIDING","NEUTRAL",_ST.TDT):             _ST.DEADLINE_DRIVEN,
    ("AVOIDING","NEUTRAL",_ST.TIT_FOR_TAT):     _ST.DEADLINE_DRIVEN,
    ("AVOIDING","NEUTRAL",_ST.RELATIVE_TFT):    _ST.DEADLINE_DRIVEN,
    ("AVOIDING","NEUTRAL",_ST.NICE_TFT):        _ST.TRADE_OFF,
    ("AVOIDING","NEUTRAL",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("AVOIDING","NEUTRAL",_ST.RANDOM_WALK):     _ST.KDE,
    ("AVOIDING","NEUTRAL",_ST.BAYESIAN):        _ST.DEADLINE_DRIVEN,
    ("AVOIDING","NEUTRAL",_ST.KDE):             _ST.DEADLINE_DRIVEN,
    ("AVOIDING","NEUTRAL",_ST.TRADE_OFF):       _ST.TRADE_OFF,
    ("AVOIDING","NEUTRAL",_ST.PARETO):          _ST.DEADLINE_DRIVEN,
    # ── ACCOMMODATING / FRUSTRATED ────────────────────────────────────────────
    ("ACCOMMODATING","FRUSTRATED",_ST.BOULWARE):        _ST.NICE_TFT,
    ("ACCOMMODATING","FRUSTRATED",_ST.CONCEDER):        _ST.NICE_TFT,
    ("ACCOMMODATING","FRUSTRATED",_ST.LINEAR):          _ST.NICE_TFT,
    ("ACCOMMODATING","FRUSTRATED",_ST.TDT):             _ST.NICE_TFT,
    ("ACCOMMODATING","FRUSTRATED",_ST.TIT_FOR_TAT):     _ST.NICE_TFT,
    ("ACCOMMODATING","FRUSTRATED",_ST.RELATIVE_TFT):    _ST.NICE_TFT,
    ("ACCOMMODATING","FRUSTRATED",_ST.NICE_TFT):        _ST.TRADE_OFF,
    ("ACCOMMODATING","FRUSTRATED",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("ACCOMMODATING","FRUSTRATED",_ST.RANDOM_WALK):     _ST.KDE,
    ("ACCOMMODATING","FRUSTRATED",_ST.BAYESIAN):        _ST.NICE_TFT,
    ("ACCOMMODATING","FRUSTRATED",_ST.KDE):             _ST.NICE_TFT,
    ("ACCOMMODATING","FRUSTRATED",_ST.TRADE_OFF):       _ST.TRADE_OFF,
    ("ACCOMMODATING","FRUSTRATED",_ST.PARETO):          _ST.NICE_TFT,
    # ── ACCOMMODATING / HAPPY ─────────────────────────────────────────────────
    ("ACCOMMODATING","HAPPY",_ST.BOULWARE):        _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.CONCEDER):        _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.LINEAR):          _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.TDT):             _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.TIT_FOR_TAT):     _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.RELATIVE_TFT):    _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.NICE_TFT):        _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.DEADLINE_DRIVEN): _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.RANDOM_WALK):     _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.BAYESIAN):        _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.KDE):             _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.TRADE_OFF):       _ST.BOULWARE,
    ("ACCOMMODATING","HAPPY",_ST.PARETO):          _ST.BOULWARE,
    # ── ACCOMMODATING / ANXIOUS ───────────────────────────────────────────────
    ("ACCOMMODATING","ANXIOUS",_ST.BOULWARE):        _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.CONCEDER):        _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.LINEAR):          _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.TDT):             _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.TIT_FOR_TAT):     _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.RELATIVE_TFT):    _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.NICE_TFT):        _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.DEADLINE_DRIVEN): _ST.LINEAR,
    ("ACCOMMODATING","ANXIOUS",_ST.RANDOM_WALK):     _ST.KDE,
    ("ACCOMMODATING","ANXIOUS",_ST.BAYESIAN):        _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.KDE):             _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.TRADE_OFF):       _ST.BOULWARE,
    ("ACCOMMODATING","ANXIOUS",_ST.PARETO):          _ST.BOULWARE,
    # ── ACCOMMODATING / NEUTRAL ───────────────────────────────────────────────
    ("ACCOMMODATING","NEUTRAL",_ST.BOULWARE):        _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.CONCEDER):        _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.LINEAR):          _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.TDT):             _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.TIT_FOR_TAT):     _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.RELATIVE_TFT):    _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.NICE_TFT):        _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.DEADLINE_DRIVEN): _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.RANDOM_WALK):     _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.BAYESIAN):        _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.KDE):             _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.TRADE_OFF):       _ST.BOULWARE,
    ("ACCOMMODATING","NEUTRAL",_ST.PARETO):          _ST.TRADE_OFF,
}
del _ST


def _counter_for(character: str, emotion: str, detected: StrategyType) -> StrategyType:
    """Look up counter strategy from the character/emotion/detected triple.

    Falls back to the COMPROMISING/NEUTRAL row (the no-text default) if the
    exact (character, emotion) combination is not in the table.
    """
    key = (character.upper(), emotion.upper(), detected)
    if key in COUNTER_TABLE:
        return COUNTER_TABLE[key]
    fallback = ("COMPROMISING", "NEUTRAL", detected)
    return COUNTER_TABLE.get(fallback, StrategyType.BOULWARE)


# ═════════════════════════════════════════════════════════════════════════════
# NeoNegotiator
# ═════════════════════════════════════════════════════════════════════════════

try:
    from negmas_llm.common import DEFAULT_MODELS
    DEFAULT_OLLAMA_MODEL = DEFAULT_MODELS["ollama"]
except ImportError:
    DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"

_EMOTIONS = ("Frustrated", "Happy", "Anxious", "Neutral")

# Thomas-Kilmann conflict styles
_CHARACTERS = ("Competing", "Collaborating", "Compromising", "Avoiding", "Accommodating")

# Maps (Character, Emotion) → short tone instruction injected into LLM prompt
_TONE_MATRIX: dict[tuple[str, str], str] = {
    # Competing — high assertiveness, low cooperativeness
    ("Competing", "Frustrated"):    "Stay firm; acknowledge friction without yielding; reframe as mutual risk.",
    ("Competing", "Happy"):         "Match their confidence but hold your position; don't let their mood push you to over-concede.",
    ("Competing", "Anxious"):       "Signal you have no deadline pressure; let the tension work for you.",
    ("Competing", "Neutral"):       "Fact-based and firm; counter their position without emotion.",
    # Collaborating — high assertiveness, high cooperativeness
    ("Collaborating", "Frustrated"): "Acknowledge their concern; redirect to the shared goal; propose a small trade.",
    ("Collaborating", "Happy"):      "Build on the positive momentum; propose a value-creating trade to close.",
    ("Collaborating", "Anxious"):    "Reassure progress is real; suggest a specific path to agreement.",
    ("Collaborating", "Neutral"):    "Engage openly; propose trades that benefit both sides.",
    # Compromising — moderate assertiveness, moderate cooperativeness
    ("Compromising", "Frustrated"):  "Show movement; a visible step toward middle ground rebuilds trust.",
    ("Compromising", "Happy"):       "Positive; reinforce that you are close to a fair deal.",
    ("Compromising", "Anxious"):     "Keep steady pace; remind them of the progress already made.",
    ("Compromising", "Neutral"):     "Direct; propose a fair midpoint and explain the logic.",
    # Avoiding — low assertiveness, low cooperativeness
    ("Avoiding", "Frustrated"):      "Be concise and low-pressure; give them space to re-engage.",
    ("Avoiding", "Happy"):           "Gently maintain momentum; a light touch to keep them at the table.",
    ("Avoiding", "Anxious"):         "Simple and clear; remove complexity to reduce their hesitation.",
    ("Avoiding", "Neutral"):         "Short and direct; make it easy for them to respond.",
    # Accommodating — low assertiveness, high cooperativeness
    ("Accommodating", "Frustrated"):  "Express genuine appreciation for their flexibility; hold your remaining position firmly.",
    ("Accommodating", "Happy"):       "Warm but decisive; don't let their goodwill tempt you to over-concede.",
    ("Accommodating", "Anxious"):     "Reassure them; this deal is good for both sides.",
    ("Accommodating", "Neutral"):     "Acknowledge their cooperation; maintain your current position.",
}


class _FrequencyUFun:
    """Callable wrapper exposing NeoNegotiator's frequency model as opponent_ufun.

    The scoring system reads negotiator.opponent_ufun and compares it against the
    opponent's true ufun via Kendall correlation. Wrapping _estimate_opp_util here
    gives NeoNegotiator deception/modeling credit without changing offer logic.
    """

    reserved_value: float = 0.0

    def __init__(self, parent: "NeoNegotiator") -> None:
        self._parent = parent

    def __call__(self, outcome: Outcome | None) -> float:
        if outcome is None:
            return 0.0
        return self._parent._estimate_opp_util(outcome)


class NeoNegotiator(LLMMetaNegotiator):
    """LLMMetaNegotiator subclass with six toggleable negotiation strategies."""

    def __init__(
        self,
        *,
        provider: str = "ollama",
        model: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.5,
        max_tokens: int = 512,
        # ── Bidding strategy selector ─────────────────────────────────────────
        bidding_strategy: StrategyType = ACTIVE_STRATEGY,
        # ── Auto-counter: interrogate then switch to the optimal counter ──────
        auto_counter: bool = True,
        interrogate_rounds: int = 15,
        probe_round: int = 8,
        probe_fraction: float = 0.05,
        re_analyze_after: int = 15,    # counter-phase observations before first re-check
        re_analyze_interval: int = 15, # subsequent re-check interval (observations)
        re_analyze_window: int = 12,   # rolling window size for re-classification
        # ── Strategy toggles ──────────────────────────────────────────────────
        use_logrolling: bool = True,
        use_sentiment: bool = False,
        use_door_in_face: bool = False,
        use_batna: bool = False,
        use_deadline_delay: bool = False,
        use_micro_steps: bool = False,
        # ── Skip LLM text generation for fast tournament testing ──────────────
        skip_llm: bool = False,
        # ── Show per-round debug panels in terminal (single-run only) ─────────
        debug_mode: bool = False,
        # ── Strategy hyper-parameters ─────────────────────────────────────────
        micro_step_size: float = 0.05,
        delay_switch_time: float = 0.65,
        batna_trigger_time: float = 0.60,
        batna_max_signals: int = 2,
        **kwargs: Any,
    ) -> None:
        base = BoulwareTBNegotiator()
        merged_llm_kwargs = {"options": {"think": False}}
        if "llm_kwargs" in kwargs and kwargs["llm_kwargs"]:
            merged_llm_kwargs.update(kwargs.pop("llm_kwargs"))
        super().__init__(
            base_negotiator=base,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            llm_kwargs=merged_llm_kwargs,
            **kwargs,
        )
        # Bidding strategy
        self._initial_bidding_strategy_type: StrategyType = bidding_strategy
        self.bidding_strategy_type: StrategyType = bidding_strategy
        self._bidding_strategy: BiddingStrategy = _create_strategy(bidding_strategy)

        # Auto-counter
        self.auto_counter = auto_counter
        self.interrogate_rounds = interrogate_rounds
        self.probe_round = probe_round
        self.probe_fraction = probe_fraction
        self.re_analyze_after = re_analyze_after
        self.re_analyze_interval = re_analyze_interval
        self.re_analyze_window = re_analyze_window

        # Strategy flags
        self.use_logrolling = use_logrolling
        self.use_sentiment = use_sentiment
        self.use_door_in_face = use_door_in_face
        self.use_batna = use_batna
        self.use_deadline_delay = use_deadline_delay
        self.use_micro_steps = use_micro_steps
        self.skip_llm = skip_llm
        self.debug_mode = debug_mode

        # Hyper-parameters
        self.micro_step_size = micro_step_size
        self.delay_switch_time = delay_switch_time
        self.batna_trigger_time = batna_trigger_time
        self.batna_max_signals = batna_max_signals

        # Per-session state (reset in on_negotiation_start)
        self._phase: str = "interrogate" if auto_counter else "counter"
        self._detected_strategy: StrategyType | None = None
        self._probe_sent: bool = False
        self._probe_opp_util_before: float | None = None
        self._probe_mirrored: bool | None = None
        self._my_proposal_utils: list[float] = []
        self._counter_obs_count: int = 0
        self._next_reanalysis_at: int = re_analyze_after
        self._reanalysis_log: list[tuple[int, StrategyType, StrategyType, bool]] = []
        self._pending_redetection: StrategyType | None = None  # must match twice before switching
        self._proposal_count: int = 0
        self._last_proposal_util: float | None = None
        self._micro_start_util: float | None = None   # utility when micro-step activated
        self._concession_count: int = 0
        self._first_concession_done: bool = False
        self._batna_signals_sent: int = 0
        self._opponent_emotion: str = "Neutral"
        self._opponent_character: str = "Compromising"
        # Opponent offer tracking for frequency model
        self._opp_value_counts: dict[int, dict] = defaultdict(dict)
        self._my_util_of_opp_offers: list[float] = []
        # Last offers from each side (for debug display)
        self._last_my_offer: Outcome | None = None
        self._last_opp_offer: Outcome | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._phase = "interrogate" if self.auto_counter else "counter"
        self._detected_strategy = None
        self._probe_sent = False
        self._probe_opp_util_before = None
        self._probe_mirrored = None
        self._my_proposal_utils = []
        self._counter_obs_count = 0
        self._next_reanalysis_at = self.re_analyze_after
        self._reanalysis_log = []
        self._pending_redetection = None
        # Reset to the original strategy each new negotiation (counter may have changed it)
        self.bidding_strategy_type = self._initial_bidding_strategy_type
        self._bidding_strategy = _create_strategy(self.bidding_strategy_type)
        self._proposal_count = 0
        self._last_proposal_util = None
        self._micro_start_util = None
        self._concession_count = 0
        self._first_concession_done = False
        self._batna_signals_sent = 0
        self._opponent_emotion = "Neutral"
        self._opponent_character = "Compromising"
        self._opp_value_counts = defaultdict(dict)
        self._my_util_of_opp_offers = []
        self._last_my_offer = None
        self._last_opp_offer = None
        self.opponent_ufun = _FrequencyUFun(self)
        self._bidding_strategy.on_negotiation_start(self)

    # ─────────────────────────────────────────────────────────────────────────
    # Propose
    # ─────────────────────────────────────────────────────────────────────────

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        # Auto-counter: trigger phase transition once enough opponent offers observed
        if (
            self.auto_counter
            and self._phase == "interrogate"
            and len(self._my_util_of_opp_offers) >= self.interrogate_rounds
        ):
            self._transition_to_counter()

        # Rolling re-analysis: re-classify on full accumulated history
        if (
            self.auto_counter
            and self._phase == "counter"
            and self._counter_obs_count >= self._next_reanalysis_at
        ):
            self._reanalyze_opponent()

        # Get base bid from the active bidding strategy
        use_boulware = (
            self.bidding_strategy_type == StrategyType.BOULWARE
            or (self.auto_counter and self._phase == "interrogate")
        )
        if use_boulware:
            # BOULWARE uses BoulwareTBNegotiator directly so all Boulware-specific
            # overlays (door-in-face, deadline-delay, micro-steps) remain active.
            raw = self.base_negotiator.propose(state, dest=dest)
            if raw is None:
                return None
            outcome = raw.outcome if isinstance(raw, ExtendedOutcome) else raw
            if outcome is None:
                return None

            # Strategy 3: Door-in-the-Face — first proposal is always max utility
            if self.use_door_in_face and self._proposal_count == 0:
                best = self.ufun.best() if self.ufun else None
                if best is not None:
                    outcome = best

            # Strategy 5: Deadline Delay — freeze proposals at max during delay phase
            if self.use_deadline_delay and state.relative_time < self.delay_switch_time:
                best = self.ufun.best() if self.ufun else None
                if best is not None:
                    outcome = best

            # Strategy 6: Micro-Step Concessions — smooth linear decline after round 10
            if self.use_micro_steps and self._proposal_count >= 10:
                outcome = self._apply_micro_step(state, outcome)

            # Active probe: override with a deliberate concession step to test TfT mirroring
            if (
                self.auto_counter
                and self._phase == "interrogate"
                and not self._probe_sent
                and self._proposal_count == self.probe_round
                and self._last_proposal_util is not None
                and len(self._my_util_of_opp_offers) >= 3
                and self.ufun is not None
            ):
                min_u, max_u = _bs_utility_range(self)
                probe_step = self.probe_fraction * (max_u - min_u)
                probe_target = max(min_u, self._last_proposal_util - probe_step)
                probe_outcome = _bs_find_outcome(probe_target, self)
                if probe_outcome is not None:
                    self._probe_opp_util_before = self._my_util_of_opp_offers[-1]
                    self._probe_sent = True
                    outcome = probe_outcome
        else:
            # All other strategies compute the outcome directly via their own curve.
            # Door-in-face / deadline-delay / micro-steps are Boulware-specific and
            # are intentionally skipped here.
            outcome = self._bidding_strategy.propose(state, self)
            if outcome is None:
                return None

        # Issue Logrolling — swap to a better-for-opponent outcome at the same utility
        # level.  Applies regardless of the active bidding strategy.
        if self.use_logrolling:
            outcome = self._apply_logrolling(outcome)

        # Track proposal utility and concession count
        if self.ufun and outcome is not None:
            u = float(self.ufun(outcome))
            if self._last_proposal_util is not None and u < self._last_proposal_util - 0.001:
                self._concession_count += 1
                if not self._first_concession_done:
                    self._first_concession_done = True
            self._last_proposal_util = u
            if self.auto_counter and self._phase == "interrogate":
                self._my_proposal_utils.append(u)
        self._proposal_count += 1

        self._last_my_offer = outcome
        received_text = self._extract_received_text(state)
        text = self._generate_text(state, "propose", outcome, received_text)
        return ExtendedOutcome(outcome=outcome, data={"text": text})

    # ─────────────────────────────────────────────────────────────────────────
    # Respond
    # ─────────────────────────────────────────────────────────────────────────

    def respond(
        self, state: SAOState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        # Track opponent offer
        if state.current_offer is not None:
            self._track_opp_offer(state.current_offer)

        received_text = self._extract_received_text(state)

        # Strategy 2: Sentiment — classify opponent emotion from text
        if self.use_sentiment and received_text:
            self._update_emotion(received_text, state)

        base_resp = self.base_negotiator.respond(state, source=source)
        resp_type = base_resp.response if isinstance(base_resp, ExtendedResponseType) else base_resp
        base_data = (base_resp.data or {}) if isinstance(base_resp, ExtendedResponseType) else {}

        # Strategy 5: Deadline Delay — always block acceptance before switch time so
        # the opponent keeps conceding and panics near the deadline.
        if (
            self.use_deadline_delay
            and resp_type == ResponseType.ACCEPT_OFFER
            and state.relative_time < self.delay_switch_time
        ):
            text = self._generate_text(state, "reject", state.current_offer, received_text)
            return ExtendedResponseType(
                response=ResponseType.REJECT_OFFER,
                data={"text": text},
            )

        # Only generate LLM text for accept/end — reject text is generated in propose()
        if resp_type == ResponseType.ACCEPT_OFFER:
            text = self._generate_text(state, "accept", state.current_offer, received_text)
            return ExtendedResponseType(response=resp_type, data={**base_data, "text": text})
        elif resp_type == ResponseType.END_NEGOTIATION:
            text = self._generate_text(state, "end", state.current_offer, received_text)
            return ExtendedResponseType(response=resp_type, data={**base_data, "text": text})
        else:
            return base_resp

    # ─────────────────────────────────────────────────────────────────────────
    # LLM text generation overrides
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_text(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        if self.skip_llm:
            return self._fallback_text(action)
        try:
            system_prompt = self._build_system_prompt()
            user_message = self._build_user_message(state, action, outcome, received_text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            raw = self._call_llm(messages, state)
        except Exception:
            return self._fallback_text(action)

        # Parse response — LLM may return a list or string for "debug"
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
                debug = data.get("debug", "")
                if isinstance(debug, list):
                    debug = "\n".join(f"• {item.lstrip('•- ').strip()}" for item in debug if item)
                if debug and self.debug_mode:
                    self._print_debug(state, action, debug, outcome)
                return str(data.get("message", data.get("text", raw.strip())))
            except (json.JSONDecodeError, Exception):
                pass
        return raw.strip()

    def _print_debug(self, state: SAOState, action: str, debug: str, outcome: Outcome | None = None) -> None:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        n_steps = (self.nmi.n_steps if self.nmi and self.nmi.n_steps else None) or "?"
        title = f"[bold yellow]Round {state.step}/{n_steps}[/bold yellow] [dim]({action})[/dim]"

        # Compute real utilities directly — don't rely on LLM to report them
        util_lines: list[str] = []
        ref_outcome = outcome or self._last_my_offer or self._last_opp_offer
        if ref_outcome is not None:
            my_real = float(self.ufun(ref_outcome)) if self.ufun else None
            opp_real: float | None = None
            # Try to reach opponent's real ufun via the mechanism
            try:
                mech = getattr(self.nmi, "_mechanism", None)
                if mech is not None:
                    for neg in mech.negotiators:
                        if neg is not self and getattr(neg, "ufun", None) is not None:
                            opp_real = float(neg.ufun(ref_outcome))
                            break
            except Exception:
                pass
            if my_real is not None:
                util_lines.append(f"[bold]My utility:[/bold]       {my_real:.4f} (real)")
            opp_label = f"{opp_real:.4f} (real)" if opp_real is not None else f"{self._estimate_opp_util(ref_outcome):.4f} (estimated)"
            util_lines.append(f"[bold]Opp utility:[/bold]      {opp_label}")
            if self._last_my_offer is not None:
                util_lines.append(f"[bold]My last offer:[/bold]    {self._last_my_offer}")
            if self._last_opp_offer is not None:
                util_lines.append(f"[bold]Opp last offer:[/bold]   {self._last_opp_offer}")

        separator = "[dim]─────────────────────────────────[/dim]"
        body = "\n".join(util_lines) + "\n" + separator + "\n" + debug if util_lines else debug
        console.print(Panel(body, title=title, border_style="dim cyan", expand=False))

    def _build_system_prompt(self) -> str:
        return (
            "You are a skilled negotiator. For each round produce ONLY a JSON object with exactly two fields:\n"
            "  \"message\": your 1-3 sentence negotiation text the opponent will see — persuasive, professional, varied.\n"
            "  \"debug\": 3-5 bullet points (one per line, each starting with •) for the developer, covering: "
            "opponent character & emotion, opponent's top priority issue, "
            "and your tactical read of the situation.\n"
            "Never start consecutive messages the same way. "
            "Do NOT use filler phrases like 'I understand your concern'."
        )

    def _build_user_message(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        parts: list[str] = []
        n_steps = (self.nmi.n_steps if self.nmi and self.nmi.n_steps else None) or 100
        parts.append(f"Round {state.step}/{n_steps} | Time: {state.relative_time:.0%} | Action: {action.upper()}")

        if received_text:
            parts.append(f'Opponent said: "{received_text}"')

        # Inject sentiment/character context
        if self.use_sentiment:
            parts.append(f"Opponent state → Emotion: {self._opponent_emotion}, Character: {self._opponent_character}")
            tone = _TONE_MATRIX.get((self._opponent_character, self._opponent_emotion), "")
            if tone:
                parts.append(f"Tone instruction: {tone}")

        # Door-in-the-Face framing
        if self.use_door_in_face:
            if self._proposal_count == 1 and action == "propose":
                parts.append("Context: This is your opening bid. Frame it as principled and justified, not extreme.")
            elif self._proposal_count == 2 and self._first_concession_done and action == "propose":
                parts.append("Context: You are making your first concession. Frame this as a deliberate, generous gesture.")

        # Micro-step concession count framing
        if self.use_micro_steps and self._concession_count > 1 and action == "propose":
            parts.append(f"Context: You have now moved {self._concession_count} times toward the opponent. Mention this count to highlight your flexibility.")

        # BATNA signal injection
        if self.use_batna and action == "propose" and self._should_signal_batna(state):
            parts.append("Context: Subtly hint (one sentence only) that you have constraints or alternatives limiting your flexibility.")
            self._batna_signals_sent += 1

        if outcome:
            if action == "reject":
                parts.append(f"Opponent's offer (that you are rejecting): {outcome}")
            else:
                parts.append(f"Your offer: {outcome}")

        # Last offers from each side (for debug display)
        if self._last_my_offer is not None:
            parts.append(f"Your last offer: {self._last_my_offer}")
        if self._last_opp_offer is not None:
            parts.append(f"Opponent's last offer: {self._last_opp_offer}")

        # Utility snapshot for debug output
        if outcome is not None and self.ufun is not None:
            my_u = float(self.ufun(outcome))
            opp_u = self._estimate_opp_util(outcome)
            parts.append(f"Utility snapshot → My utility: {my_u:.3f}, Opponent estimated utility: {opp_u:.3f}")

        # Opponent's top-priority issue (most frequently offered value per issue)
        if self._opp_value_counts:
            issues = self.nmi.issues if self.nmi else []
            ranked = []
            for i, counts in self._opp_value_counts.items():
                if counts:
                    total = sum(counts.values())
                    spread = max(counts.values()) / total if total else 0
                    issue_name = issues[i].name if i < len(issues) else f"Issue{i}"
                    ranked.append((spread, issue_name))
            ranked.sort(reverse=True)
            if ranked:
                top = ", ".join(name for _, name in ranked[:3])
                parts.append(f"Opponent's top priorities (by offer frequency): {top}")

        return "\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_logrolling(self, outcome: Outcome) -> Outcome:
        """Find an outcome at the same utility level that is better for the opponent."""
        if self.ufun is None or self.nmi is None:
            return outcome

        my_util = float(self.ufun(outcome))
        tolerance = 0.025
        best = outcome
        best_opp = self._estimate_opp_util(outcome)

        os = self.nmi.outcome_space
        if os is None:
            return outcome

        for candidate in os.enumerate_or_sample(max_cardinality=300):
            if candidate is None:
                continue
            cand_u = float(self.ufun(candidate))
            if abs(cand_u - my_util) <= tolerance:
                opp_u = self._estimate_opp_util(candidate)
                if opp_u > best_opp:
                    best_opp = opp_u
                    best = candidate
        return best

    def _apply_micro_step(self, state: SAOState, outcome: Outcome) -> Outcome:
        """Replace Boulware's uneven drops with a smooth linear decline.

        Boulware holds high then drops sharply near the deadline.  This method
        instead targets a steady per-step decrement from whatever utility we
        held at round 10 down toward reserved value, so every concession is
        equally small.  The opponent perceives many visible moves; we slow the
        actual utility given up.
        """
        if self.ufun is None or self.nmi is None:
            return outcome

        reserved = float(self.ufun.reserved_value) if self.ufun.reserved_value is not None else 0.0
        n_steps = (self.nmi.n_steps if self.nmi and self.nmi.n_steps else None) or 100

        # Capture starting utility once, at the step micro-steps become active
        current_u = float(self.ufun(outcome))
        if self._micro_start_util is None:
            self._micro_start_util = current_u

        # Linear target: decline from start_util to reserved over remaining rounds
        steps_so_far = self._proposal_count - 10          # steps since activation
        total_steps   = max(n_steps - 10, 1)
        target_u = self._micro_start_util - (self._micro_start_util - reserved) * (steps_so_far / total_steps)
        target_u = max(target_u, reserved)

        # Find the outcome whose utility is closest to target_u.
        # If Boulware is ABOVE target, we concede down to the linear slope.
        # If Boulware is BELOW target, we hold at target (never over-concede).
        if abs(current_u - target_u) < 0.005:
            return outcome  # Already at target

        os = self.nmi.outcome_space
        if os is None:
            return outcome

        best = outcome
        best_diff = abs(current_u - target_u)
        for candidate in os.enumerate_or_sample(max_cardinality=300):
            if candidate is None:
                continue
            cu = float(self.ufun(candidate))
            if cu < reserved:
                continue
            diff = abs(cu - target_u)
            if diff < best_diff:
                best_diff = diff
                best = candidate
        return best

    def _track_opp_offer(self, offer: Outcome) -> None:
        """Record opponent's offer into our manual frequency model and track character."""
        self._last_opp_offer = offer
        for i, val in enumerate(offer):
            counts = self._opp_value_counts[i]
            counts[val] = counts.get(val, 0) + 1

        if self.ufun:
            cur_util = float(self.ufun(offer))
            self._my_util_of_opp_offers.append(cur_util)
            if self._phase == "counter":
                self._counter_obs_count += 1
            self._update_character()

            # Check if the probe was mirrored: opponent's response must exceed their pre-probe
            # baseline rate by a meaningful margin (avoids false positives from natural Boulware drift).
            if self._probe_sent and self._probe_mirrored is None and self._probe_opp_util_before is not None:
                delta = cur_util - self._probe_opp_util_before
                # Baseline = average per-round change before the probe
                pre_probe = self._my_util_of_opp_offers[:-1]  # exclude probe response
                if len(pre_probe) >= 3:
                    baseline_deltas = [pre_probe[i+1]-pre_probe[i] for i in range(len(pre_probe)-1)]
                    baseline_rate = sum(baseline_deltas) / len(baseline_deltas)
                else:
                    baseline_rate = 0.0
                # Mirror confirmed if response is clearly above baseline.
                # Scale the minimum threshold to match the ufun range (Island = 0–100).
                _ufun_scale = 100.0 if any(v > 1.5 for v in self._my_util_of_opp_offers) else 1.0
                self._probe_mirrored = delta >= max(baseline_rate * 3, 0.01 * _ufun_scale)

        self._bidding_strategy.on_opponent_offer(offer, self)

    def _estimate_opp_util(self, outcome: Outcome) -> float:
        """Estimate opponent's utility for an outcome using value frequencies."""
        if not self._opp_value_counts or outcome is None:
            return 0.0
        n = len(outcome)
        if n == 0:
            return 0.0
        total = 0.0
        for i, val in enumerate(outcome):
            counts = self._opp_value_counts.get(i, {})
            if not counts:
                total += 0.5
                continue
            max_c = max(counts.values())
            total += (counts.get(val, 0) / max_c) if max_c > 0 else 0.5
        return total / n

    def _update_character(self) -> None:
        """Classify opponent using Thomas-Kilmann styles from their offer history.

        Two observable axes:
          - concession_rate (avg delta of my utility in their offers):
              positive = they're giving me more over time  (low assertiveness)
              negative = offers getting worse for me        (high assertiveness)
          - cooperation_level (current absolute utility they give me):
              high = they care about my outcome             (high cooperativeness)
              low  = they don't                             (low cooperativeness)
        """
        hist = self._my_util_of_opp_offers
        if len(hist) < 4:
            return
        recent = hist[-5:]
        deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        concession_rate = sum(deltas) / len(deltas)
        cooperation_level = recent[-1]
        variance = sum((d - concession_rate) ** 2 for d in deltas) / len(deltas)

        if concession_rate < -0.01:
            # Offers getting worse for me → assertive, self-focused
            self._opponent_character = "Competing"
        elif concession_rate > 0.04:
            # Conceding fast toward me → low assertiveness for themselves
            self._opponent_character = "Accommodating"
        elif abs(concession_rate) < 0.005 and variance < 0.001:
            # Almost no movement and very consistent → disengaged
            self._opponent_character = "Avoiding"
        elif concession_rate > 0.01 and cooperation_level > 0.4:
            # Steady concessions AND already giving me decent utility → win-win seeker
            self._opponent_character = "Collaborating"
        else:
            # Moderate, steady movement toward middle ground
            self._opponent_character = "Compromising"

    def _update_emotion(self, text: str, state: SAOState) -> None:
        """Use LLM to classify opponent's emotional state from their message."""
        if self.skip_llm:
            self._emotion_from_keywords(text)
            return
        try:
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "Classify the emotion in this negotiation message as exactly one of: "
                        "Frustrated, Happy, Anxious, Neutral. Reply with only that single word."
                    ),
                },
                {"role": "user", "content": text},
            ]
            raw = self._call_llm(msgs, state).strip()
            word = raw.split()[0] if raw.split() else "Neutral"
            for e in _EMOTIONS:
                if e.lower() in word.lower():
                    self._opponent_emotion = e
                    return
            self._opponent_emotion = "Neutral"
        except Exception:
            self._emotion_from_keywords(text)

    def _emotion_from_keywords(self, text: str) -> None:
        """Keyword fallback for emotion detection when LLM is unavailable."""
        t = text.lower()
        if any(w in t for w in ("frustrat", "disappoint", "annoyed", "unfair", "cannot")):
            self._opponent_emotion = "Frustrated"
        elif any(w in t for w in ("deadline", "time", "running out", "urgent", "soon")):
            self._opponent_emotion = "Anxious"
        elif any(w in t for w in ("great", "excellent", "happy", "pleased", "good")):
            self._opponent_emotion = "Happy"
        else:
            self._opponent_emotion = "Neutral"

    def _should_signal_batna(self, state: SAOState) -> bool:
        # Signal walk-away constraints against low-cooperativeness styles
        # (Competing pushes hard for themselves; Avoiding is disengaged — both need a nudge)
        return (
            state.relative_time >= self.batna_trigger_time
            and self._batna_signals_sent < self.batna_max_signals
            and self._opponent_character in ("Competing", "Avoiding")
        )

    def _fallback_text(self, action: str) -> str:
        if action == "accept":
            return "I'm pleased to accept this offer. This works well for both of us."
        if action == "end":
            return "I believe we've reached an impasse. Thank you for your time."
        if self._concession_count > 0:
            return f"I've made {self._concession_count} concessions so far and I'm proposing this counteroffer in good faith."
        return "Thank you for your offer. Here is my counteroffer."

    # ─────────────────────────────────────────────────────────────────────────
    # Auto-counter: interrogation & classification
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_opponent(self, vals: list[float] | None = None) -> StrategyType:
        """Classify opponent strategy from a sequence of observed offer utilities.

        Signals used (all on normalised 0-1 / 30-step scale):
          norm_avg_rate   — mean per-step concession
          norm_accel      — second-half minus first-half rate (half-split)
          variance        — spread of per-step deltas
          rate_first/last — per-step rate in first/last third (period breakdown)
          delta_autocorr  — lag-1 autocorrelation of deltas (negative = oscillating)
          monotone_frac   — fraction of steps where opponent improved toward us
        """
        use_probe = vals is None   # probe result only meaningful for full history
        vals = vals if vals is not None else self._my_util_of_opp_offers
        if len(vals) < 4:
            return StrategyType.BOULWARE

        # Normalise utility scale: Island scenario uses 0–100; others use 0–1.
        # All thresholds below are calibrated for 0–1 scale.
        if max(vals) > 1.5:
            vals = [v / 100.0 for v in vals]

        deltas = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        n = len(deltas)
        avg_rate = sum(deltas) / n
        variance = sum((d - avg_rate) ** 2 for d in deltas) / n

        # Half-split acceleration
        mid = max(1, n // 2)
        first_half_rate = sum(deltas[:mid]) / mid
        second_half_rate = sum(deltas[mid:]) / max(1, n - mid)
        acceleration = second_half_rate - first_half_rate

        # Period breakdown into thirds
        s1, s2 = max(1, n // 3), max(1, 2 * n // 3)
        t1, t2, t3 = deltas[:s1], deltas[s1:s2], deltas[s2:]
        rate_first = sum(t1) / len(t1)
        rate_mid   = sum(t2) / len(t2) if t2 else rate_first
        rate_last  = sum(t3) / len(t3) if t3 else rate_mid

        initial_level  = vals[0]
        monotone_frac  = sum(1 for d in deltas if d >= 0) / n

        # Lag-1 autocorrelation of deltas (negative = oscillating/search)
        ss = sum((d - avg_rate) ** 2 for d in deltas)
        if ss > 1e-12 and n >= 3:
            delta_autocorr = sum(
                (deltas[i] - avg_rate) * (deltas[i + 1] - avg_rate)
                for i in range(n - 1)
            ) / ss
        else:
            delta_autocorr = 0.0

        # Normalise rates to a canonical 30-step game
        n_steps = (self.nmi.n_steps if self.nmi and self.nmi.n_steps else 100) or 100
        scale = 30.0 / n_steps
        norm_avg_rate  = avg_rate  / scale
        norm_accel     = acceleration / scale
        norm_rate_last = rate_last / scale

        # ── DEADLINE_DRIVEN: flat early, sharp dive in last third ─────────────
        # Rate spikes dramatically only in the final third; earlier thirds near zero.
        if (norm_rate_last > 0.025
                and abs(rate_first) < abs(rate_last) * 0.25
                and rate_last > 3.0 * max(abs(rate_first), abs(rate_mid), 1e-9)):
            return StrategyType.DEADLINE_DRIVEN

        # ── TDT: consistently accelerating concession (power curve) ──────────
        # Last third clearly faster than first third, and overall avg is positive.
        if (norm_avg_rate > 0.004
                and norm_rate_last > 0.010
                and rate_last > 2.0 * max(abs(rate_first), 1e-9)
                and rate_last > rate_mid * 1.2):
            return StrategyType.TDT

        # ── High variance ─────────────────────────────────────────────────────
        if variance > 0.0015:
            # Front-loaded deceleration → CONCEDER
            if norm_accel < -0.015 and norm_avg_rate > 0.015:
                return StrategyType.CONCEDER
            # Oscillating (negative autocorr) with no net drift → RANDOM_WALK
            if delta_autocorr < -0.20 and abs(norm_avg_rate) < 0.008:
                return StrategyType.RANDOM_WALK
            # Oscillating but drifting upward → deliberate BAYESIAN search
            if delta_autocorr < -0.10 and monotone_frac >= 0.50 and norm_avg_rate > 0.003:
                return StrategyType.BAYESIAN
            # Noisy but consistently improving → BAYESIAN/KDE search
            if monotone_frac >= 0.65 and norm_avg_rate > 0.002:
                return StrategyType.BAYESIAN
            return StrategyType.RANDOM_WALK

        # ── Systematic fast concession ────────────────────────────────────────
        if norm_avg_rate > 0.025 and norm_accel < -0.010:
            return StrategyType.CONCEDER

        if norm_avg_rate > 0.020 and abs(norm_accel) < 0.015:
            return StrategyType.LINEAR

        if norm_avg_rate > 0.015 and initial_level > 0.35:
            return StrategyType.NICE_TFT

        # ── Moderate rate ─────────────────────────────────────────────────────
        if 0.008 < norm_avg_rate <= 0.020 and abs(norm_accel) < 0.004:
            return StrategyType.TRADE_OFF

        if norm_avg_rate > 0.003 and norm_accel > 0.003:
            return StrategyType.TDT

        if norm_avg_rate > 0.006 and norm_accel < -0.003:
            return StrategyType.TIT_FOR_TAT

        # Boulware: tiny positive rate + tiny positive acceleration
        if norm_avg_rate > 0.0001 and norm_accel > 0.0001:
            return StrategyType.BOULWARE

        # ── Near-zero movement (Boulware / DEADLINE_DRIVEN / frozen TfT) ──────
        if use_probe and self._probe_mirrored is True:
            return StrategyType.NICE_TFT if initial_level > 0.35 else StrategyType.TIT_FOR_TAT
        if use_probe and self._probe_mirrored is False:
            return StrategyType.BOULWARE

        if initial_level > 0.40:
            return StrategyType.NICE_TFT
        return StrategyType.RELATIVE_TFT

    def _transition_to_counter(self) -> None:
        """Classify opponent using full interrogation history, look up counter, and switch."""
        detected = self._classify_opponent()
        self._detected_strategy = detected
        counter = _counter_for(self._opponent_character, self._opponent_emotion, detected)
        self.bidding_strategy_type = counter
        self._bidding_strategy = _create_strategy(counter)
        self._bidding_strategy.on_negotiation_start(self)
        self._phase = "counter"
        self._counter_obs_count = 0
        self._next_reanalysis_at = self.re_analyze_after

    def _reanalyze_opponent(self) -> None:
        """Re-classify on full accumulated history; switch counter only on two consecutive matches.

        Why full history (not a window): a mid-game slice of any monotone strategy (Boulware,
        Linear, etc.) looks like a different strategy. The whole curve shape is what the
        classifier was calibrated against.

        Why require two consecutive matches: TfT opponents mirror our counter strategy, making
        the accumulated data noisy. A single re-analysis may flip due to this contamination;
        requiring two consecutive identical predictions filters out transient oscillations.
        """
        if len(self._my_util_of_opp_offers) < 4:
            return
        new_detected = self._classify_opponent()  # full history
        self._next_reanalysis_at = self._counter_obs_count + self.re_analyze_interval

        if new_detected == self._detected_strategy:
            # Prediction stable — reset pending
            self._pending_redetection = None
            self._reanalysis_log.append((self._counter_obs_count, self._detected_strategy, new_detected, False))
            return

        if new_detected == self._pending_redetection:
            # Second consecutive match for new_detected → commit the switch
            changed = True
            self._reanalysis_log.append((self._counter_obs_count, self._detected_strategy, new_detected, True))
            self._detected_strategy = new_detected
            self._pending_redetection = None
            counter = _counter_for(self._opponent_character, self._opponent_emotion, new_detected)
            self.bidding_strategy_type = counter
            self._bidding_strategy = _create_strategy(counter)
            self._bidding_strategy.on_negotiation_start(self)
        else:
            # First time seeing new_detected — park it as pending, wait for confirmation
            self._pending_redetection = new_detected
            self._reanalysis_log.append((self._counter_obs_count, self._detected_strategy, new_detected, False))
