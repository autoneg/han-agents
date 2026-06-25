"""hagent_agent.py — Belief-based contextual-bandit negotiator for HAN 2026.

Architecture:
  * Bayesian belief over 6 opponent types (conceder/hardliner/erratic/peer/fair/wall),
    updated each turn from observed offer trajectory features.
  * Contextual Thompson Sampling bandit (class-level, persists within process)
    selects a concession-curve arm based on type-weighted Beta samples.
  * Belief-weighted parameter blending adapts floor, opp-weight, and endgame threshold.
  * AC_combi acceptance policy (Baarslag 2014): AC_next + MAX_W endgame window,
    empirically +12-18% over fixed-threshold policies.
  * Urgency-adjusted effective time compresses the timeline when opponents are slow
    (critical for LLM opponents that consume 50-350s per step).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

# Private RNG so Thompson Sampling doesn't consume the global random state
# (which would make benchmark scenario generation non-reproducible).
_rng = random.Random()

from dataclasses import dataclass

from negmas.sao.common import ResponseType, SAOResponse, SAOState
from negmas.sao.negotiators.base import SAOCallNegotiator


# ---------------------------------------------------------------------------
# OutcomeAnalyzer (inlined from sunny_day_v2 for standalone submission)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutcomeInfo:
    outcome: Any
    utility: float
    fraction: float


class OutcomeAnalyzer:
    """Enumerates legal outcomes and caches our utility values."""

    def __init__(
        self,
        ufun: Any,
        outcome_space: Any,
        max_outcomes: int = 6000,
        safety_margin_fraction: float = 0.01,
    ) -> None:
        self.ufun = ufun
        self.outcome_space = outcome_space
        self.max_outcomes = max_outcomes
        self.safety_margin_fraction = safety_margin_fraction
        self.utility_cache: dict[Any, float] = {}
        self.outcomes: list[OutcomeInfo] = []
        self.reserved_value = float(getattr(ufun, "reserved_value", 0.0) or 0.0)
        self.best_utility = self.reserved_value
        self.best_outcome = None
        self.safety_margin = 0.0
        self._analyze()

    def utility(self, outcome: Any | None) -> float:
        if outcome is None:
            return float("-inf")
        key = self.key(outcome)
        if key not in self.utility_cache:
            try:
                self.utility_cache[key] = float(self.ufun(outcome))
            except Exception:
                self.utility_cache[key] = float("-inf")
        return self.utility_cache[key]

    def fraction(self, outcome: Any | None) -> float:
        span = max(self.best_utility - self.reserved_value, 1e-9)
        return (self.utility(outcome) - self.reserved_value) / span

    def utility_at_fraction(self, frac: float) -> float:
        span = max(self.best_utility - self.reserved_value, 1e-9)
        return self.reserved_value + frac * span

    def safe_floor(self) -> float:
        return self.reserved_value + self.safety_margin

    def is_valid(self, outcome: Any | None) -> bool:
        if outcome is None:
            return False
        try:
            return self.outcome_space is None or self.outcome_space.is_valid(outcome)
        except Exception:
            return False

    def candidates_above(self, min_utility: float) -> list[OutcomeInfo]:
        return [info for info in self.outcomes if info.utility >= min_utility]

    def best_safe(self) -> OutcomeInfo | None:
        return self.outcomes[0] if self.outcomes else None

    def key(self, outcome: Any) -> Any:
        try:
            hash(outcome)
            return outcome
        except TypeError:
            return repr(outcome)

    def _analyze(self) -> None:
        raw = self._collect_outcomes()
        try:
            best = self.ufun.best()
        except Exception:
            best = None
        if best is not None and self.is_valid(best) and self.key(best) not in {
            self.key(o) for o in raw
        }:
            raw.append(best)
        if not raw and best is not None and self.is_valid(best):
            raw = [best]

        utilities = [self.utility(o) for o in raw]
        self.best_utility = max(utilities, default=self.reserved_value)
        self.best_outcome = raw[utilities.index(self.best_utility)] if utilities else None
        self.safety_margin = self.safety_margin_fraction * max(
            self.best_utility - self.reserved_value, 1e-9
        )
        infos = [
            OutcomeInfo(o, self.utility(o), self.fraction(o))
            for o in raw
            if self.utility(o) >= self.safe_floor()
        ]
        if not infos:
            infos = [OutcomeInfo(o, self.utility(o), self.fraction(o)) for o in raw]
        self.outcomes = sorted(infos, key=lambda i: i.utility, reverse=True)

    def _collect_outcomes(self) -> list[Any]:
        if self.outcome_space is None:
            return []
        try:
            sampled = list(
                self.outcome_space.enumerate_or_sample(max_cardinality=self.max_outcomes)
            )
        except Exception:
            return []
        unique: list[Any] = []
        seen: set[Any] = set()
        for o in sampled:
            if not self.is_valid(o):
                continue
            k = self.key(o)
            if k in seen:
                continue
            seen.add(k)
            unique.append(o)
        return unique

# ---------------------------------------------------------------------------
# Opponent type taxonomy and per-type strategy parameters
# ---------------------------------------------------------------------------

TYPES: list[str] = ["conceder", "hardliner", "erratic", "peer", "fair", "wall"]

# Each type has three parameters that define how we play against it:
#   final_floor  — lowest fraction we'd ever propose/accept (Boulware landing zone)
#   opp_weight   — how much to favour the opponent's likely-preferred issues when scoring
#   endgame_eps  — minimum fraction we'll accept in the last few steps
TYPE_PARAMS: dict[str, dict[str, float]] = {
    "conceder":  {"final_floor": 0.65, "opp_weight": 0.15, "endgame_eps": 0.02},
    "hardliner": {"final_floor": 0.30, "opp_weight": 0.36, "endgame_eps": 0.04},
    "erratic":   {"final_floor": 0.32, "opp_weight": 0.22, "endgame_eps": 0.05},
    "peer":      {"final_floor": 0.40, "opp_weight": 0.30, "endgame_eps": 0.02},
    "fair":      {"final_floor": 0.38, "opp_weight": 0.35, "endgame_eps": 0.02},
    "wall":      {"final_floor": 0.22, "opp_weight": 0.46, "endgame_eps": 0.02},
}

# ---------------------------------------------------------------------------
# Bandit arms — modifiers layered on top of belief-blended base params
# ---------------------------------------------------------------------------

N_TYPES = len(TYPES)

ARMS: list[dict[str, Any]] = [
    # name          boulware exponent   floor shift   opp scoring scale
    {"name": "conservative", "boulware_e": 0.12, "floor_bonus": +0.08, "opp_scale": 0.75},
    {"name": "balanced",     "boulware_e": 0.20, "floor_bonus":  0.00, "opp_scale": 1.00},
    {"name": "aggressive",   "boulware_e": 0.28, "floor_bonus": -0.08, "opp_scale": 1.20},
    {"name": "opp_focused",  "boulware_e": 0.20, "floor_bonus": -0.04, "opp_scale": 1.40},
]
N_ARMS = len(ARMS)


# ---------------------------------------------------------------------------
# Contextual Thompson Sampling bandit
# ---------------------------------------------------------------------------

class ContextualBandit:
    """Beta(α,β)-per-(arm, type) bandit.

    Class-level attributes survive across negotiations in the same process,
    allowing the agent to accumulate evidence across the tournament.
    """

    # Priors: α=β=2 → mean 0.5, moderate uncertainty, no arm bias.
    _alpha: list[list[float]] = [[2.0] * N_TYPES for _ in range(N_ARMS)]
    _beta:  list[list[float]] = [[2.0] * N_TYPES for _ in range(N_ARMS)]

    @classmethod
    def select_arm(cls, beliefs: dict[str, float]) -> int:
        """Draw a type-weighted Q-value for each arm via Thompson sampling."""
        arm_values: list[float] = []
        for arm_i in range(N_ARMS):
            q = 0.0
            for t_i, tp in enumerate(TYPES):
                w = beliefs.get(tp, 0.0)
                if w < 1e-9:
                    continue
                a = max(cls._alpha[arm_i][t_i], 0.1)
                b = max(cls._beta[arm_i][t_i], 0.1)
                q += w * _rng.betavariate(a, b)
            arm_values.append(q)
        return arm_values.index(max(arm_values))

    @classmethod
    def update(cls, arm_i: int, beliefs: dict[str, float], reward: float) -> None:
        """Fractional Bayesian update: credit is split across types by belief weight."""
        for t_i, tp in enumerate(TYPES):
            w = beliefs.get(tp, 0.0)
            if w < 1e-9:
                continue
            if reward >= 0.5:
                cls._alpha[arm_i][t_i] += w
            else:
                cls._beta[arm_i][t_i] += w


# ---------------------------------------------------------------------------
# Bayesian belief over opponent types
# ---------------------------------------------------------------------------

class BeliefState:
    """Dirichlet pseudocounts updated from offer-trajectory features."""

    def __init__(self) -> None:
        self._counts: dict[str, float] = {t: 1.0 for t in TYPES}

    def probabilities(self) -> dict[str, float]:
        total = sum(self._counts.values())
        return {t: c / total for t, c in self._counts.items()}

    def dominant_type(self) -> str:
        return max(self._counts, key=self._counts.__getitem__)

    def blended_param(self, key: str) -> float:
        """Belief-weighted average of TYPE_PARAMS[type][key]."""
        probs = self.probabilities()
        return sum(probs[tp] * TYPE_PARAMS[tp][key] for tp in TYPES)

    def update(self, offer_fracs: list[float]) -> None:
        """Update counts from observed fraction trajectory."""
        if len(offer_fracs) < 2:
            return
        diffs = [b - a for a, b in zip(offer_fracs[:-1], offer_fracs[1:])]
        n = len(diffs)
        cr   = sum(diffs) / n                              # avg concession per step
        mono = sum(1 for d in diffs if d >  0.005) / n    # fraction of improving steps
        osc  = sum(1 for d in diffs if d < -0.02)  / n    # fraction of reversal steps
        best = max(offer_fracs)

        lk = self._likelihoods(cr, mono, osc, best)
        for tp in TYPES:
            self._counts[tp] *= max(1e-9, lk[tp])

        # Rescale to prevent float underflow while preserving ratios
        total = sum(self._counts.values())
        scale = N_TYPES / total
        for tp in TYPES:
            self._counts[tp] = max(1e-9, self._counts[tp] * scale)

    @staticmethod
    def _sig(x: float, center: float, sharpness: float = 12.0) -> float:
        return 1.0 / (1.0 + math.exp(-sharpness * (x - center)))

    def _likelihoods(
        self, cr: float, mono: float, osc: float, best: float
    ) -> dict[str, float]:
        s = self._sig
        return {
            "conceder":  s(cr, 0.04)    * s(mono, 0.50),
            "hardliner": s(-cr, -0.005) * s(1.0 - osc, 0.70) * s(-best, -0.50),
            "erratic":   s(osc, 0.25),
            "peer":      s(cr, 0.01)    * (1.0 - s(cr, 0.08)) * s(mono, 0.30),
            "fair":      s(cr, 0.015)   * (1.0 - s(cr, 0.10)) * s(mono, 0.40),
            "wall":      s(-cr, -0.001) * s(-best, -0.35),
        }


# ---------------------------------------------------------------------------
# Lightweight frequency-based opponent model
# ---------------------------------------------------------------------------

class OpponentModel:
    DECAY = 0.85

    def __init__(self) -> None:
        self.value_counts: dict[int, dict[Any, float]] = defaultdict(lambda: defaultdict(float))
        self.offer_fracs: list[float] = []

    def update(self, offer: Any, frac: float) -> None:
        if not isinstance(offer, tuple):
            return
        self.offer_fracs.append(frac)
        for counts in self.value_counts.values():
            for v in list(counts):
                counts[v] *= self.DECAY
        for idx, val in enumerate(offer):
            self.value_counts[idx][val] += 1.0

    def opp_fit(self, outcome: Any) -> float:
        """Score how well this outcome matches what the opponent has been offering."""
        if not isinstance(outcome, tuple) or not self.value_counts:
            return 0.0
        scores: list[float] = []
        for i, v in enumerate(outcome):
            counts = self.value_counts.get(i, {})
            if not counts:
                continue
            top = max(counts.values())
            scores.append(counts.get(v, 0.0) / top if top > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def recent_best(self, window: int) -> float:
        tail = self.offer_fracs[-max(1, window):]
        return max(tail) if tail else 0.0

    def still_conceding(self) -> bool:
        """True if opponent has been clearly improving offers to us in recent steps."""
        if len(self.offer_fracs) < 4:
            return False
        return max(self.offer_fracs[-3:]) - max(self.offer_fracs[:-3]) > 0.03


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class HAgent(SAOCallNegotiator):
    """
    Belief-based ensemble negotiator for HAN 2026.

    Key properties vs SnowyDayAgent:
      - Maintains a full probability *distribution* over opponent types (not a single label)
      - Strategy parameters are belief-weighted blends, not hard-coded per type
      - AC_combi acceptance (proven +12-18% over fixed thresholds)
      - Urgency-adjusted effective time (critical against slow LLM opponents)
      - Cross-negotiation bandit learning via class-level Beta distributions
    """

    def __init__(
        self,
        max_outcomes: int = 6_000,
        safety_margin_fraction: float = 0.01,
        instant_accept_frac: float = 0.95,
        endgame_time: float = 0.92,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._max_outcomes = max_outcomes
        self._safety_margin = safety_margin_fraction
        self._instant_accept = instant_accept_frac
        self._endgame_time = endgame_time

        self._analyzer: OutcomeAnalyzer | None = None
        self._belief = BeliefState()
        self._opp_model = OpponentModel()
        self._last_proposal: Any | None = None
        self._wall_times: list[float] = []
        self._arm: int | None = None

    # ------------------------------------------------------------------
    # Primary entry point — always returns a valid SAOResponse
    # ------------------------------------------------------------------

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        try:
            return self._respond(state)
        except Exception:
            try:
                analyzer = self._setup()
                best = analyzer.best_safe()
                if best is not None:
                    return SAOResponse(ResponseType.REJECT_OFFER, best.outcome)
            except Exception:
                pass
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

    # ------------------------------------------------------------------
    # Core negotiation logic
    # ------------------------------------------------------------------

    def _respond(self, state: SAOState) -> SAOResponse:
        assert self.ufun is not None
        analyzer = self._setup()

        elapsed = float(getattr(state, "time", 0.0) or 0.0)
        self._wall_times.append(elapsed)
        t = self._eff_time(state)

        # Observe and model opponent's offer
        offer = state.current_offer
        valid_opp_offer = (
            offer is not None
            and analyzer.is_valid(offer)
            and not self._offer_is_ours(state)
        )
        if valid_opp_offer:
            frac = analyzer.fraction(offer)
            self._opp_model.update(offer, frac)
            self._belief.update(self._opp_model.offer_fracs)

        # Select bandit arm exactly once per negotiation
        if self._arm is None:
            self._arm = ContextualBandit.select_arm(self._belief.probabilities())
        arm = ARMS[self._arm]

        # Plan the counter-proposal
        proposal = self._plan_offer(t, arm, state, analyzer)

        # Acceptance check (AC_combi)
        if valid_opp_offer and self._should_accept(offer, t, arm, proposal, state, analyzer):
            reward = analyzer.fraction(offer)
            ContextualBandit.update(self._arm, self._belief.probabilities(), reward)
            return SAOResponse(
                ResponseType.ACCEPT_OFFER,
                offer,
                data={"text": self._accept_text()},
            )

        # No safe offer available — end rather than propose below reserve
        if proposal is None:
            arm_i = self._arm if self._arm is not None else 0
            ContextualBandit.update(arm_i, self._belief.probabilities(), 0.0)
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        self._last_proposal = proposal
        text = self._counter_text(t)
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            proposal,
            data={"text": text} if text else None,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> OutcomeAnalyzer:
        if self._analyzer is None:
            assert self.ufun is not None
            space = (
                self.nmi.outcome_space
                if self.nmi is not None and self.nmi.outcome_space is not None
                else self.ufun.outcome_space
            )
            self._analyzer = OutcomeAnalyzer(
                self.ufun,
                space,
                max_outcomes=self._max_outcomes,
                safety_margin_fraction=self._safety_margin,
            )
        return self._analyzer

    # ------------------------------------------------------------------
    # Time — urgency-adjusted for slow LLM opponents
    # ------------------------------------------------------------------

    def _eff_time(self, state: SAOState) -> float:
        t = float(getattr(state, "relative_time", 0.0) or 0.0)
        return min(1.0, max(t, self._urgency(state)))

    def _urgency(self, state: SAOState) -> float:
        tl = self.nmi.time_limit if self.nmi is not None else None
        if not tl or tl == float("inf"):
            return 0.0
        elapsed = float(getattr(state, "time", 0.0) or 0.0)
        remaining = max(0.0, float(tl) - elapsed)
        if len(self._wall_times) < 3:
            return 0.0
        recent = self._wall_times[-4:]
        deltas = [b - a for a, b in zip(recent[:-1], recent[1:]) if b > a]
        if not deltas:
            return 0.0
        per_round = max(sum(deltas) / len(deltas), 1e-3)
        turns_left = remaining / per_round
        return min(1.0, max(0.0, 1.0 - (turns_left - 1.0) / 8.0))

    def _is_last_chance(self, state: SAOState) -> bool:
        n_steps = self.nmi.n_steps if self.nmi is not None else None
        step = int(getattr(state, "step", 0) or 0)
        if n_steps is not None and int(n_steps) - step <= 2:
            return True
        tl = self.nmi.time_limit if self.nmi is not None else None
        if not tl or tl == float("inf"):
            return False
        elapsed = float(getattr(state, "time", 0.0) or 0.0)
        remaining = max(0.0, float(tl) - elapsed)
        if len(self._wall_times) < 3:
            return False
        recent = self._wall_times[-4:]
        deltas = [b - a for a, b in zip(recent[:-1], recent[1:]) if b > a]
        if not deltas:
            return False
        per_round = max(sum(deltas) / len(deltas), 1e-3)
        return remaining / per_round <= 1.5

    # ------------------------------------------------------------------
    # Offer planning — Boulware curve + belief-blended floor + opp_fit scoring
    # ------------------------------------------------------------------

    def _threshold(self, t: float, arm: dict[str, Any]) -> float:
        """Boulware concession curve in fraction space."""
        floor = self._belief.blended_param("final_floor") + arm["floor_bonus"]
        floor = min(max(floor, 0.02), 0.92)
        e = arm["boulware_e"]
        g = floor + (1.0 - floor) * (1.0 - t ** (1.0 / e))
        return min(max(g, floor), 1.0)

    def _plan_offer(
        self, t: float, arm: dict[str, Any], state: SAOState, analyzer: OutcomeAnalyzer
    ) -> Any | None:
        safe = [info for info in analyzer.outcomes if info.utility >= analyzer.safe_floor()]
        if not safe:
            return None

        # Opening offer: anchor near top but use opp_fit to break ties
        if not self._opp_model.offer_fracs:
            top_band = [info for info in safe if info.fraction >= 0.92]
            if not top_band:
                top_band = safe[:5]
            return max(
                top_band[:200],
                key=lambda info: 0.85 * info.fraction + 0.15 * self._opp_model.opp_fit(info.outcome),
            ).outcome

        g = self._threshold(t, arm)

        # Last chance: lower threshold to endgame_eps so ZoA outcomes near our reserve
        # are reachable — critical against wall opponents who need 0.80+ of their utility.
        if self._is_last_chance(state):
            eps = self._belief.blended_param("endgame_eps")
            g = min(g, max(eps, 0.02))

        candidates = [info for info in safe if info.fraction >= g]
        if not candidates:
            candidates = safe[:1]

        opp_w = min(self._belief.blended_param("opp_weight") * arm["opp_scale"], 0.80)
        own_w = 1.0 - opp_w

        # Last-chance with tough opponent: prioritise opp_fit to find the zone of agreement.
        # Without this, own_w * 1.0 (our best) can outscore opp_w * 1.0 (their best)
        # and we keep proposing our absolute best even though they'll never accept it.
        if self._is_last_chance(state) and opp_w >= 0.38:
            # 0.60/0.40 split: opp_fit dominates enough to find the ZoA
            # but 0.40 own-fraction weight steers toward Pareto-efficient deals
            # rather than giving the opponent their absolute best unconditionally.
            scored = sorted(
                candidates[:500],
                key=lambda info: self._opp_model.opp_fit(info.outcome) * 0.60
                + info.fraction * 0.40,
                reverse=True,
            )
        else:
            scored = sorted(
                candidates[:500],
                key=lambda info: own_w * info.fraction + opp_w * self._opp_model.opp_fit(info.outcome),
                reverse=True,
            )

        best = scored[0].outcome
        # Avoid exact repeat of previous proposal (stirs up the conversation)
        if best == self._last_proposal and len(scored) > 1:
            return scored[1].outcome
        return best

    # ------------------------------------------------------------------
    # Acceptance — AC_combi (Baarslag 2014)
    # ------------------------------------------------------------------

    def _should_accept(
        self,
        offer: Any,
        t: float,
        arm: dict[str, Any],
        proposal: Any | None,
        state: SAOState,
        analyzer: OutcomeAnalyzer,
    ) -> bool:
        r_raw = float(self.ufun.reserved_value or 0.0)
        if float(self.ufun(offer)) <= r_raw:
            return False

        frac = analyzer.fraction(offer)

        # Fast path: near-best offer
        if frac >= self._instant_accept:
            return True

        # Conceder hold: if they're still improving, keep waiting
        if t < 0.88 and frac < 0.90 and self._opp_model.still_conceding():
            return False

        # AC_next: accept if offer is at least as good as what we'd propose
        if proposal is not None:
            plan_frac = analyzer.fraction(proposal)
            if frac >= plan_frac - 1e-9:
                return True

        eps = self._belief.blended_param("endgame_eps")

        # MAX_W endgame window: accept if offer >= recent best they showed
        if t >= self._endgame_time:
            w = max(3, round(len(self._opp_model.offer_fracs) * (1.0 - t)))
            max_w = self._opp_model.recent_best(w)
            if frac >= max(eps, max_w - 1e-9):
                return True

        # Last-chance: accept any positive-advantage deal rather than walk away
        if self._is_last_chance(state) and frac >= eps:
            return True

        # Deadline rescue: if opponent has been very tough (best offer < 40% of our range)
        # and we're in the endgame, accept anything marginally positive rather than
        # walk away with 0 — mirrors SnowyDay's deadline_rescue mechanism.
        if t >= 0.80 and self._opp_model.offer_fracs:
            best_recv = max(self._opp_model.offer_fracs)
            if best_recv < 0.40 and frac > 0.01 and self._is_last_chance(state):
                return True

        return False

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _offer_is_ours(self, state: SAOState) -> bool:
        proposer = getattr(state, "current_proposer", None)
        if proposer is None:
            return False
        own = {str(getattr(self, "id", "") or ""), str(getattr(self, "name", "") or "")}
        own.discard("")
        return bool(own) and str(proposer) in own

    def _accept_text(self) -> str:
        return random.choice([
            "Deal — I'm satisfied with this outcome.",
            "Agreed. Good negotiating.",
            "Works for me — let's close it.",
            "I'll take it. Pleasure doing business.",
        ])

    def _counter_text(self, t: float) -> str | None:
        if t >= 0.92:
            return "Time is nearly up — this is my best offer."
        if not self._opp_model.offer_fracs:
            return "Here's my opening. I'm open to hearing what matters most to you."
        dominant = self._belief.dominant_type()
        probs = self._belief.probabilities()
        if t >= 0.70:
            return "Getting close to the deadline — I believe this works for both of us."
        if probs[dominant] > 0.55:
            if dominant == "hardliner":
                return "I've adjusted toward what I believe matters to you. Can we find a deal?"
            if dominant == "erratic":
                return "I'll stay consistent here so you know exactly where I stand."
        return None  # deliberate silence beats hollow filler
