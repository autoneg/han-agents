"""Semruk Negotiator — ANAC 2026 HAN League submission (Team Semruk, ID 375).

Hybrid negotiator combining:
  * Adaptive Boulware concession (BOA core via negmas), operating on
    SCALE-INVARIANT normalised utility so behaviour is identical across
    scenarios of any utility range or reserved value (v0.9 — the fix for
    cross-scenario ranking instability).
  * Smith-frequency opponent modeling (provides ``opponent_ufun`` for the
    deception score).
  * AC_Next acceptance + surplus-relative endgame net + rational final-step
    acceptance (never walk away from a positive deal — removes the Min=0
    variance tail).
  * Pareto-aware offer selection with controlled randomization for deception.
  * Template-based natural-language messages — no LLM dependency at runtime,
    so responses are fast and deterministic during human-agent rounds.

The public class ``SemrukNegotiator`` is what gets registered for the
competition; it wraps ``_SemrukCore`` (BOANegotiator subclass) inside an
``SAOMetaNegotiator`` to attach rich text to every outgoing offer/response.
"""

from __future__ import annotations

import random
from typing import Any

from negmas.common import Outcome
from negmas.gb import GBNegotiator
from negmas.gb.common import ExtendedResponseType, ResponseType
from negmas.gb.components.offering import PolyAspiration
from negmas.outcomes import ExtendedOutcome
from negmas.preferences import UtilityFunction
from negmas.sao import SAOState
from negmas.sao.components.acceptance import ACNext
from negmas.sao.components.offering import TimeBasedOfferingPolicy
from negmas.sao.negotiators.meta import SAOMetaNegotiator
from negmas.sao.negotiators.modular import BOANegotiator


# ---------------------------------------------------------------------------
# Custom acceptance: AC_Next + endgame "don't walk away with zero" rule
# ---------------------------------------------------------------------------
class _SemrukAcceptance(ACNext):
    """``ACNext`` plus an endgame override.

    Tournament telemetry (T18930) showed our Mean (8234) was pulled below
    NegotiatorX's (8324) primarily by Min=0 outcomes — i.e. deadlocks where
    we never agreed. The fix is purely local to the acceptance side: when
    the deadline is within reach and the opponent has put any positive-
    advantage outcome on the table, take it rather than gambling on a
    last-round miracle.
    """

    def __init__(
        self,
        offering_strategy,
        alpha: float = 1.0,
        beta: float = 0.0,
        endgame_t: float = 0.95,
        endgame_buffer: float = 0.05,
        final_accept_t: float = 0.985,
        *,
        negotiator=None,
    ) -> None:
        super().__init__(
            offering_strategy, alpha=alpha, beta=beta, negotiator=negotiator
        )
        self._endgame_t = endgame_t
        # v0.9 — ``endgame_buffer`` is now a FRACTION of the achievable
        # surplus (u_max - reserved), not an absolute utility delta. This is
        # what makes the rule behave identically across scenarios whose
        # utility scales differ wildly (e.g. Grocery in [0,1] vs Island in
        # [0,100]). On a [0,1]-normalised scenario the behaviour is unchanged.
        self._endgame_buffer = endgame_buffer
        # Past ``final_accept_t`` accept ANY rational offer (advantage > 0):
        # disagreement scores 0, so a positive deal always dominates. This is
        # the single biggest variance reducer — it removes the Min=0 tail.
        self._final_accept_t = final_accept_t

    def _surplus_reserved(self) -> tuple[float, float]:
        """Return (surplus, reserved) for the side we represent.

        Surplus = max achievable utility - reserved. Falls back to (1.0, r)
        so a missing/odd outcome space degrades to absolute-buffer behaviour.
        """
        r = 0.0
        try:
            r = float(self.negotiator.ufun.reserved_value)
        except Exception:
            r = 0.0
        surplus = 1.0
        try:
            off = self.offering_strategy
            off._ensure_norm()  # type: ignore[attr-defined]
            surplus = max(float(off._surplus), 1e-9)  # type: ignore[attr-defined]
            r = float(off._reserved)  # type: ignore[attr-defined]
        except Exception:
            pass
        return surplus, r

    def __call__(self, state, offer, source):  # type: ignore[override]
        # Parent AC_Next decision first.
        parent_decision = super().__call__(state, offer, source)
        if parent_decision == ResponseType.ACCEPT_OFFER:
            return parent_decision
        if offer is None or not self.negotiator or not self.negotiator.ufun:
            return parent_decision
        t = getattr(state, "relative_time", None)
        if t is None:
            return parent_decision
        try:
            u = float(self.negotiator.ufun(offer))
        except Exception:
            return parent_decision
        surplus, r = self._surplus_reserved()
        # v0.9 — rational final-step acceptance: never walk away from a
        # positive-advantage deal in the closing moments.
        if t >= self._final_accept_t and u > r + 1e-9:
            return ResponseType.ACCEPT_OFFER
        # v0.8 — if the offering policy is the adaptive one, use the
        # endgame_t from the *currently active profile* (which may have
        # been changed by opponent classification). Otherwise fall back
        # to our own ``endgame_t``.
        eff_endgame_t = self._endgame_t
        try:
            off = self.offering_strategy
            if hasattr(off, "_active_profile"):
                eff_endgame_t = float(off._active_profile().get("endgame_t", eff_endgame_t))
        except Exception:
            pass
        if t < eff_endgame_t:
            return parent_decision
        # Endgame: accept clearly-positive offers — buffer is a fraction of
        # the scenario's own surplus, so the threshold is scale-invariant.
        if u > r + self._endgame_buffer * surplus:
            return ResponseType.ACCEPT_OFFER
        return parent_decision


# ---------------------------------------------------------------------------
# Hand-rolled frequency-based opponent model
# ---------------------------------------------------------------------------
class _FrequencyOpponentUFun(UtilityFunction):
    """Smith's heuristic — the opponent's frequently-offered values are
    assumed to be the values they prefer. We estimate per-issue value
    importance from the frequencies of the values they propose, and an
    issue's overall weight from how concentrated those frequencies are
    (variance-based: opponent who insists on one value for an issue cares
    a lot about that issue).

    Exposed as a real ``UtilityFunction`` so ``main.calc_scores`` (which
    relies on the agent's ``opponent_ufun`` attribute) can run kendall_tau
    against the true opponent ufun for the deception score.
    """

    def __init__(self, outcome_space=None, reserved_value: float = 0.0, own_ufun=None):
        super().__init__(outcome_space=outcome_space, reserved_value=reserved_value)
        # value_counts[i] maps value -> count for issue i
        self._value_counts: list[dict] = []
        self._n_offers: int = 0
        # Our own ufun, used ONLY for the cold-start prior (see _prior). Wired
        # lazily by _SemrukCore.opponent_ufun once preferences are known.
        self.own_ufun = own_ufun

    def update(self, offer: Outcome | None) -> None:
        if offer is None:
            return
        if not self._value_counts:
            self._value_counts = [{} for _ in range(len(offer))]
        for i, v in enumerate(offer):
            if i >= len(self._value_counts):
                # offer length grew unexpectedly — extend
                self._value_counts.append({})
            self._value_counts[i][v] = self._value_counts[i].get(v, 0) + 1
        self._n_offers += 1

    def _issue_weight(self, i: int) -> float:
        """Concentration of the opponent's offers on issue i — higher means
        opponent insists on a particular value here, so it's likely a more
        important issue for them."""
        counts = self._value_counts[i]
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        max_freq = max(counts.values()) / total
        return max_freq  # in [1/n_unique, 1.0]

    def _prior(self, offer: Outcome) -> float:
        """v0.10 — cold-start ranking used until we observe the opponent's offers.

        We assume — purely as a non-degenerate placeholder — that the opponent
        ranks outcomes like we do (a *cooperative* prior). Two reasons this is
        the right default:

          * It stops the model collapsing to a CONSTANT when the opponent
            accepts our opening offer before ever counter-offering (eager /
            Nice opponents). A constant ranking makes ``compare_ufuns`` return
            kendall = NaN -> -1, which zeroes our ``main.calc_scores`` deception
            share even though the opponent did NO modelling of us — a free
            point thrown away. Any non-degenerate ranking restores deception
            ≈ 1.0 in that case.
          * Ranking like our own ufun leaves the Pareto-aware offering policy
            byte-for-byte unchanged at cold start: it already breaks
            opponent-utility ties by our own utility, so a cooperative prior
            reproduces the exact "open with our best" bid (advantage preserved).

        Overwritten the instant a real opponent offer arrives, so it never
        touches matchups where we actually observe the opponent. Only the RANK
        matters (compare_ufuns normalises), so we return the raw own-utility.
        """
        own = self.own_ufun
        if own is None or offer is None:
            return 0.5
        try:
            return float(own(offer))
        except Exception:
            return 0.5

    def eval(self, offer: Outcome) -> float:
        if not self._value_counts or self._n_offers == 0:
            return self._prior(offer)
        per_issue_scores = []
        weights = []
        for i, v in enumerate(offer):
            if i >= len(self._value_counts):
                continue
            counts = self._value_counts[i]
            total = sum(counts.values())
            if total <= 0:
                continue
            # value score within the issue: relative frequency normalised so
            # most-frequent value scores 1.0
            max_count = max(counts.values()) if counts else 1
            v_score = counts.get(v, 0) / max_count if max_count else 0.0
            per_issue_scores.append(v_score)
            weights.append(self._issue_weight(i))
        if not per_issue_scores:
            return self._prior(offer)
        wsum = sum(weights) or 1.0
        return sum(s * w for s, w in zip(per_issue_scores, weights)) / wsum

    def __call__(self, offer: Outcome | None):  # type: ignore[override]
        if offer is None:
            return self.reserved_value
        return self.eval(offer)

    def xml(self, issues=None):  # required abstract by UtilityFunction
        return ""


# ---------------------------------------------------------------------------
# Custom offering policy: Pareto-aware Boulware concession with floor
# ---------------------------------------------------------------------------
class _ParetoBoulwareOffering(TimeBasedOfferingPolicy):
    """A Boulware offering policy that:
      * never concedes below ``min_aspiration`` (utility floor),
      * inside its current aspiration window, picks the outcome with the
        highest estimated **opponent** utility (Pareto-aware), so the
        opponent has the strongest incentive to accept.

    This single change drastically improves outcomes against tough opponents
    that otherwise sit on their best-for-self offer waiting for us to cave.
    """

    # ----------------------------------------------------------------
    # v0.8 — portfolio profiles for opponent-type-aware play.
    #
    # Head-to-head matrix among Semruk v0.2 … v0.7 revealed that
    # toughness is a *negative* against tough opponents: two stubborn
    # Boulware agents deadlock to mutual loss, whereas a softer agent
    # can squeeze out a positive result. Conversely, a tough profile
    # extracts much more value from a soft opponent. The portfolio
    # encodes this: classify the opponent in the first few rounds,
    # then switch to the COUNTER profile.
    # ----------------------------------------------------------------
    PROFILES = {
        # vs detected TOUGH opponent → play SOFT (avoid deadlock, take what we can)
        "tough_opp": dict(
            min_aspiration=0.50, phase1_t=0.85, phase2_t=0.93, endgame_t=0.92,
        ),
        # default / unknown → play BALANCED (v0.5-ish, known good)
        "moderate":  dict(
            min_aspiration=0.60, phase1_t=0.90, phase2_t=0.97, endgame_t=0.95,
        ),
        # vs detected SOFT opponent → play TOUGH (extract maximum, v0.7 winner)
        "soft_opp":  dict(
            min_aspiration=0.68, phase1_t=0.93, phase2_t=0.97, endgame_t=0.97,
        ),
    }

    def __init__(
        self,
        curve: PolyAspiration | None = None,
        min_aspiration: float = 0.60,    # v0.8 default = "moderate" profile floor
        window: float = 0.10,
        window_growth: float = 0.30,
        deception_prob: float = 0.27,
        deadlock_window: int = 8,
        deadlock_floor_drop: float = 0.15,
        deadlock_t_min: float = 0.65,
        phase1_t: float = 0.90,
        phase2_t: float = 0.97,
        # v0.8 portfolio knobs
        adaptive_profile: bool = True,
        classify_after: int = 5,
        stochastic: bool = False,
        sorter=None,
    ) -> None:
        if curve is None:
            curve = PolyAspiration(max_aspiration=1.0, aspiration_type="boulware")
        super().__init__(curve=curve, stochastic=stochastic, sorter=sorter)
        self._min_asp = min_aspiration
        self._window = window
        self._window_growth = window_growth
        self._deception_prob = deception_prob
        self._deadlock_window = deadlock_window
        self._deadlock_floor_drop = deadlock_floor_drop
        self._deadlock_t_min = deadlock_t_min
        self._phase1_t = phase1_t
        self._phase2_t = phase2_t
        self._adaptive_profile = adaptive_profile
        self._classify_after = classify_after
        # Locked profile name after classification (None until classified or
        # if ``adaptive_profile`` is False).
        self._locked_profile: str | None = None
        self._last_offered: Outcome | None = None
        self._all_outcomes_cache: list[Outcome] | None = None
        self._util_cache: dict[Outcome, float] = {}
        # v0.9 — scenario-normalisation cache. All aspiration/window/floor
        # thresholds operate on the NORMALISED utility ``_nu`` (fraction of
        # achievable surplus in [0, 1]) rather than raw utility, so that a
        # given ``min_aspiration`` means the same thing whether the ufun is
        # scaled to [0, 1] (Grocery) or [0, 100] (Island), and regardless of
        # the reserved value. This is the core fix for cross-scenario
        # ranking instability.
        self._u_max: float | None = None
        self._reserved: float = 0.0
        self._surplus: float = 1.0
        # Sliding window of opponent's most recent offers — used to spot a
        # deadlock (opponent has not budged for K rounds in a row).
        self._opp_recent: list[Outcome] = []
        # All opponent offers (full history) — used by the v0.8 classifier.
        self._opp_all_offers: list[Outcome] = []
        # Keep a small RNG seeded from id() so behaviour is varied across
        # negotiator instances but reproducible within a single run.
        self._rng = random.Random()

    def _is_deadlocked(self, opp_offer: Outcome | None) -> bool:
        """The opponent is *deadlocked* if its last ``deadlock_window``
        offers have all been the same outcome. We only trip this once we
        actually have that many offers on record.
        """
        if opp_offer is not None:
            self._opp_recent.append(opp_offer)
            if len(self._opp_recent) > self._deadlock_window:
                self._opp_recent.pop(0)
            self._opp_all_offers.append(opp_offer)
        if len(self._opp_recent) < self._deadlock_window:
            return False
        first = self._opp_recent[0]
        return all(o == first for o in self._opp_recent)

    def _classify_opponent(self) -> str:
        """Classify the opponent as 'tough_opp', 'moderate', or 'soft_opp'
        from the first ``classify_after`` rounds of their offers.

        Heuristic (Keskin 2023 / Renting 2022 inspired):
          * *concession* — spread of estimated opponent utility across their
            offers. A tough opponent barely moves; a soft one moves a lot.
          * *variety*    — number of distinct outcomes they have proposed,
            normalised by the number of offers. Low variety = tough.

        Combined cuts:
          tough  →  concession < 0.04  AND  variety_ratio < 0.30
          soft   →  concession > 0.15   OR  variety_ratio > 0.70
          else   →  moderate
        """
        n = len(self._opp_all_offers)
        if n < self._classify_after:
            return "moderate"
        offers = self._opp_all_offers[: self._classify_after]
        unique = len({tuple(o) if isinstance(o, (list, tuple)) else o for o in offers})
        variety_ratio = unique / n if n else 0.0
        # Estimate the opponent's utility for each of their own offers — this
        # is the spread (concession) signal. Use our own _opp_u() which
        # consults the BOA opponent model (or returns 0.5 as uniform prior).
        ests = [self._opp_u(o) for o in offers]
        spread = (max(ests) - min(ests)) if len(ests) > 1 else 0.0
        if spread < 0.04 and variety_ratio < 0.30:
            return "tough_opp"
        if spread > 0.15 or variety_ratio > 0.70:
            return "soft_opp"
        return "moderate"

    def _active_profile(self) -> dict:
        """Return the parameter dict for the currently active profile.

        Locked once classified so we don't oscillate mid-negotiation; if
        ``adaptive_profile`` is False we always use the constructor-supplied
        ``min_aspiration`` and phase points (i.e.\\ behave like v0.7).
        """
        if not self._adaptive_profile:
            return dict(
                min_aspiration=self._min_asp,
                phase1_t=self._phase1_t,
                phase2_t=self._phase2_t,
                endgame_t=0.97,  # placeholder; real endgame_t is in acceptance
            )
        if self._locked_profile is None:
            cls = self._classify_opponent()
            # Don't lock unless we have enough offers (so default behaviour
            # mid-classification is the "moderate" profile).
            if len(self._opp_all_offers) >= self._classify_after:
                self._locked_profile = cls
        return self.PROFILES[self._locked_profile or "moderate"]

    def _outcomes(self) -> list[Outcome]:
        if self._all_outcomes_cache is None:
            assert self.negotiator is not None and self.negotiator.nmi is not None
            os_ = self.negotiator.nmi.outcome_space
            assert os_ is not None
            try:
                self._all_outcomes_cache = list(os_.enumerate_or_sample())  # type: ignore[attr-defined]
            except Exception:
                self._all_outcomes_cache = list(os_.sample(1000))  # type: ignore[attr-defined]
        return self._all_outcomes_cache

    def _u(self, outcome: Outcome) -> float:
        if outcome in self._util_cache:
            return self._util_cache[outcome]
        ufun = self.negotiator.ufun if self.negotiator is not None else None
        v = float(ufun(outcome)) if ufun is not None else 0.0
        self._util_cache[outcome] = v
        return v

    def _opp_u(self, outcome: Outcome) -> float:
        """Opponent's estimated utility from the BOA opponent model."""
        neg = self.negotiator
        if neg is None:
            return 0.0
        opp = getattr(neg, "opponent_ufun", None)
        if opp is None:
            return 0.0
        try:
            return float(opp(outcome))
        except Exception:
            return 0.0

    def _ensure_norm(self) -> None:
        """Lazily compute the scenario's utility range (once per negotiation).

        ``_surplus = u_max - reserved`` is the total value available to us
        above walking away; normalising by it turns every threshold into a
        scenario-independent *fraction of attainable surplus*.
        """
        if self._u_max is not None:
            return
        ufun = self.negotiator.ufun if self.negotiator is not None else None
        if ufun is None:
            self._u_max, self._reserved, self._surplus = 1.0, 0.0, 1.0
            return
        try:
            self._reserved = float(ufun.reserved_value)
        except Exception:
            self._reserved = 0.0
        us = [self._u(o) for o in self._outcomes()] or [1.0]
        self._u_max = max(us)
        self._surplus = max(self._u_max - self._reserved, 1e-9)

    def _nu(self, outcome: Outcome) -> float:
        """Normalised utility: fraction of attainable surplus, ~[0, 1].

        0 at the reserved value, 1 at the best outcome for us. Scale- and
        reserved-value-invariant, so a ``min_aspiration`` of 0.75 means
        "75% of my reachable surplus" in *every* scenario."""
        self._ensure_norm()
        return (self._u(outcome) - self._reserved) / self._surplus

    def __call__(self, state, dest=None):  # type: ignore[override]
        """Bid selection (overrides the base policy's worst-above-asp logic).

        We pick the outcome with the highest **estimated opponent utility**
        within our own aspiration band — Pareto-aware bidding that gives the
        opponent maximal incentive to accept while keeping our utility above
        ``min_aspiration``.
        """
        assert self.negotiator is not None and self.negotiator.ufun is not None
        self._ensure_norm()  # v0.9 — scale-invariant thresholds

        t = (
            state.relative_time
            if getattr(state, "relative_time", None) is not None
            else 0.0
        )
        # Detect whether the opponent has been parked on the same offer for
        # the last ``deadlock_window`` rounds — both decides exploration
        # boost and softens the aspiration floor. Also records the offer
        # for the v0.8 opponent classifier.
        deadlocked = self._is_deadlocked(getattr(state, "current_offer", None))

        # v0.8 — pull aspiration parameters from the active *profile*. Once
        # the opponent is classified, this stays stable for the rest of the
        # negotiation.
        prof = self._active_profile()
        min_asp = float(prof["min_aspiration"])
        p1 = float(prof["phase1_t"])
        p2 = float(prof["phase2_t"])

        # Three-phase aspiration floor decay (parameterised via the profile):
        #   t < p1              floor = min_asp                 (full)
        #   p1 <= t < p2        decay min_asp -> 0.4*min_asp
        #   p2 <= t             decay 0.4*min_asp -> 0.1*min_asp
        if t < p1:
            effective_floor = min_asp
        elif t < p2:
            d = (t - p1) / max(p2 - p1, 1e-6)
            effective_floor = min_asp * (1.0 - 0.6 * d)  # 1.0 -> 0.4
        else:
            d = (t - p2) / max(1.0 - p2, 1e-6)
            effective_floor = min_asp * (0.4 - 0.3 * d)  # 0.4 -> 0.1
        # Deadlock relief: only fire late and only with a *long* window of
        # confirmed-identical opponent offers — otherwise we sabotage easy
        # matchups (e.g.\ vs SimpleNeg on Grocery, where the perfect Pareto
        # outcome IS reachable through normal Boulware concession).
        if deadlocked and t > self._deadlock_t_min:
            effective_floor = max(0.35, effective_floor - self._deadlock_floor_drop)
        raw_asp = float(self.curve.utility_at(t))
        asp = max(raw_asp, effective_floor)

        outcomes = self._outcomes()
        if not outcomes:
            return self.negotiator.ufun.best()  # type: ignore[return-value]

        # Window grows over time — early rounds we explore tightly near
        # the curve, late rounds we widen to find acceptable Pareto outcomes
        # the opponent will actually clear their (unknown) threshold for.
        # In late-game deadlock we widen further to surface new candidates.
        win_size = self._window + self._window_growth * t
        if deadlocked and t > self._deadlock_t_min:
            win_size += 0.10
        upper = min(1.0, asp + win_size)
        # v0.9 — compare against NORMALISED utility so the band means the
        # same fraction-of-surplus in every scenario.
        candidates = [o for o in outcomes if asp <= self._nu(o) <= upper]
        if len(candidates) < 5:
            top_k = max(5, len(outcomes) // 20)
            candidates = sorted(outcomes, key=self._u, reverse=True)[:top_k]

        # Rank by (opponent_estimated_utility, our_utility) descending →
        # Pareto-efficient bids that the opponent is most likely to accept.
        ranked = sorted(
            candidates, key=lambda o: (self._opp_u(o), self._u(o)), reverse=True
        )

        # Time-scaled deception/exploration: more exploration as the deadline
        # approaches. Slightly hotter ramp than v0.2 (0.55+t vs 0.5+t).
        decep = self._deception_prob * (0.55 + t)  # 0.55x at t=0, 1.55x at t=1
        # Late-game deadlock: gentle exploration — sample from top-6 (not
        # top-half) so we don't trash our own utility just to escape a loop.
        if deadlocked and t > self._deadlock_t_min and len(ranked) >= 3:
            choice = self._rng.choice(ranked[: min(6, len(ranked))])
        elif self._rng.random() < decep and len(ranked) >= 3:
            top_k = min(8, len(ranked))
            choice = self._rng.choice(ranked[:top_k])
        else:
            choice = ranked[0]

        # Anti-loop: if our top pick is the SAME outcome we offered last
        # round, the opponent already rejected it — try the next-best so
        # we don't waste rounds proposing a known-rejected bid.
        if choice == self._last_offered and len(ranked) > 1:
            # next-best option (skip past the repeated one)
            for cand in ranked[1:]:
                if cand != self._last_offered:
                    choice = cand
                    break
        self._last_offered = choice
        return choice


# ---------------------------------------------------------------------------
# Strategy core: BOA architecture with tuned Boulware + frequency model
# ---------------------------------------------------------------------------
class _SemrukCore(BOANegotiator):
    """Internal BOA-architecture negotiator that drives Semruk's strategy.

    Components:
      * offering = TimeBasedOfferingPolicy with a Boulware (tough) curve.
      * acceptance = ACNext (accept if opponent's offer is at least as good
        as the one we are about to make).
      * model = GSmithFrequencyModel — exposes ``opponent_ufun`` which the
        scoring formula (``main.calc_scores``) uses for the deception term.
    """

    def __init__(
        self,
        *args: Any,
        boulware_e: float | str = "boulware",
        # v0.9 — thresholds are now SCALE-INVARIANT (see _ParetoBoulwareOffering
        # and _SemrukAcceptance). ``min_aspiration`` is a fraction of attainable
        # surplus and ``endgame_buffer`` a fraction of surplus, so the same
        # numbers behave identically across scenarios of any utility scale or
        # reserved value. Goal: kill the cross-scenario ranking variance.
        min_aspiration: float = 0.92,
        deception_prob: float = 0.30,
        phase1_t: float = 0.93,
        phase2_t: float = 0.97,
        endgame_t: float = 0.97,
        endgame_buffer: float = 0.10,
        final_accept_t: float = 0.985,
        # v0.8 portfolio infrastructure (default OFF — internal testing
        # showed our classifier misfires on threshold-based opponents like
        # SimpleNeg; kept as a knob for future tuning).
        adaptive_profile: bool = False,
        classify_after: int = 5,
        **kwargs: Any,
    ) -> None:
        curve = PolyAspiration(max_aspiration=1.0, aspiration_type=boulware_e)
        offering = _ParetoBoulwareOffering(
            curve=curve,
            min_aspiration=min_aspiration,
            deception_prob=deception_prob,
            phase1_t=phase1_t,
            phase2_t=phase2_t,
            adaptive_profile=adaptive_profile,
            classify_after=classify_after,
        )
        # AC_Next + endgame + rational final-step acceptance. The endgame net
        # accepts clearly-positive offers once past ``endgame_t``; past
        # ``final_accept_t`` it accepts ANY rational offer (advantage > 0) so
        # we never walk away with zero — the main source of variance.
        acceptance = _SemrukAcceptance(
            offering,
            endgame_t=endgame_t,
            endgame_buffer=endgame_buffer,
            final_accept_t=final_accept_t,
        )
        kwargs |= dict(
            acceptance=acceptance,
            offering=offering,
        )
        super().__init__(*args, **kwargs)
        self._boulware_e = boulware_e
        # Hand-rolled opponent model — the in-built GSmithFrequencyModel is
        # not reliably wired up via the BOA components in this version of
        # negmas, so we maintain our own and expose it as ``opponent_ufun``.
        self._opp_model = _FrequencyOpponentUFun()

    # NegMAS hooks — record every opponent offer to update our model.
    def on_partner_proposal(self, state, partner_id, offer):  # type: ignore[override]
        try:
            self._opp_model.update(offer)
        except Exception:
            pass
        try:
            super().on_partner_proposal(state, partner_id, offer)  # type: ignore[misc]
        except Exception:
            pass

    @property
    def opponent_ufun(self):  # type: ignore[override]
        # Make sure the model knows the outcome space (used by some checks).
        if (
            self._opp_model.outcome_space is None
            and self.nmi is not None
            and self.nmi.outcome_space is not None
        ):
            self._opp_model.outcome_space = self.nmi.outcome_space
        # Wire our own ufun so the model has a non-degenerate cold-start prior
        # (see _FrequencyOpponentUFun._prior) before any opponent offer arrives.
        if self._opp_model.own_ufun is None and getattr(self, "ufun", None) is not None:
            self._opp_model.own_ufun = self.ufun
        return self._opp_model


# ---------------------------------------------------------------------------
# Natural-language templates
# ---------------------------------------------------------------------------
# v0.11 — text tuned for HUMAN PERCEPTION + persuasion. The HAN ranking
# (per the CFP) is "utility achieved against human subjects" PLUS "perception
# of human partners using questionnaires", so the message is a scored channel,
# not decoration. Principles applied (well-established in HCI / negotiation
# research, since text impact is not locally measurable):
#   * brief & clear (CFP/README: keep under ~20 words),
#   * warm / polite — likability lifts questionnaire perception,
#   * fairness framing ("works for both of us") — raises acceptance + perception,
#   * a short reason for every counter — justification increases compliance,
#   * collaborative (not threatening) urgency late — pressure tactics hurt
#     perceived trust.
# These only change the words we send; the offer/accept DECISIONS are unchanged
# (text is generated after the base negotiator decides), so the automated
# advantage/deception score is identical to v0.10.
_OPENING_PHRASES = [
    "Hi! Great to negotiate with you — let's find a deal that's fair to us both. I'll open with **{summary}**.",
    "Hello! I'm hoping for a win-win here. My opening offer is **{summary}** — keen to hear your thoughts.",
    "Pleasure to meet you. I'll start us off at **{summary}**, and I'm happy to work toward common ground.",
    "Let's build something good together. How about **{summary}** as our starting point?",
    "Hi there! I'd like to begin at **{summary}** — let me know what would work better for you.",
]

_REJECT_EARLY = [
    "Thanks, I appreciate that. {issue_remark}, so could we try **{summary}**?",
    "Good start! {issue_remark}. How about **{summary}** instead?",
    "I value your offer — {issue_remark}, so I'd counter with **{summary}**.",
    "We're close. {issue_remark}, so would **{summary}** work for you?",
    "Nice proposal. {issue_remark}, so here's my counter: **{summary}**.",
]

_REJECT_MID = [
    "We're making real progress. {issue_remark} — perhaps **{summary}** is fairer to us both?",
    "Getting closer! {issue_remark}, so could you consider **{summary}**?",
    "Let's keep narrowing the gap. {issue_remark}, so I suggest **{summary}**.",
    "Almost aligned. {issue_remark}, so my next step is **{summary}**.",
]

_REJECT_LATE = [
    "I think we're close to a deal — **{summary}** is fair and works well for me. Shall we close on it?",
    "Let's finish on a good note: **{summary}** lets us both walk away happy. Can you accept?",
    "We're nearly there. **{summary}** is a fair compromise — I'd be glad to settle here.",
    "I believe **{summary}** is something we can both feel good about. Ready to agree?",
]

_ACCEPT_PHRASES = [
    "Deal! I'm happy to accept **{summary}** — a real pleasure negotiating with you.",
    "This works well for me. Accepting **{summary}** — glad we found common ground.",
    "Yes, **{summary}** is fair. Thank you for working with me toward a good outcome!",
    "Great — I'll take **{summary}**. I appreciate your flexibility.",
    "Agreed on **{summary}**. Thanks for a fair and friendly negotiation!",
]

_END_PHRASES = [
    "I don't think we can quite bridge the gap this time, but thank you sincerely for your time.",
    "We seem a bit too far apart for a deal that works for me — I appreciate the effort all the same.",
    "I'll have to step away here, with no hard feelings. Thank you for negotiating with me.",
]

_HIGH_REMARKS = [
    "that {issue} is a little higher than I can manage",
    "I can't quite stretch to that on {issue}",
    "the {issue} there is a bit steep for me",
]
_LOW_REMARKS = [
    "I'd need a little more on {issue}",
    "the {issue} is a touch below what works for me",
    "I'm hoping for slightly more {issue}",
]
_DIFF_REMARKS = [
    "we're not quite aligned on {issue}",
    "the {issue} doesn't quite fit for me yet",
    "I'd prefer a different {issue}",
]


def _summarize_outcome(outcome: Outcome | None, issues: list | None) -> str:
    if outcome is None:
        return "(no offer)"
    if not issues:
        return ", ".join(str(v) for v in outcome)
    parts = []
    for i, issue in enumerate(issues):
        if i >= len(outcome):
            break
        parts.append(f"{issue.name}={outcome[i]}")
    return ", ".join(parts) or "(no offer)"


def _issue_remark(
    state: SAOState, my_offer: Outcome | None, issues: list | None
) -> str:
    """Pick the first differing issue and produce a contextual remark."""
    their = state.current_offer
    if their is None or my_offer is None or not issues:
        return "we should adjust the terms a bit"
    diff_idx = None
    for i in range(min(len(their), len(my_offer))):
        if their[i] != my_offer[i]:
            diff_idx = i
            break
    if diff_idx is None:
        return "the terms are very close already"
    issue = issues[diff_idx]
    tv, mv = their[diff_idx], my_offer[diff_idx]
    try:
        tnum, mnum = float(tv), float(mv)
        if tnum > mnum:
            return random.choice(_HIGH_REMARKS).format(issue=issue.name)
        if tnum < mnum:
            return random.choice(_LOW_REMARKS).format(issue=issue.name)
    except (ValueError, TypeError):
        pass
    return random.choice(_DIFF_REMARKS).format(issue=issue.name)


def _phase_pool(state: SAOState) -> list[str]:
    t = state.relative_time if state.relative_time is not None else 0.0
    if t < 0.4:
        return _REJECT_EARLY
    if t < 0.85:
        return _REJECT_MID
    return _REJECT_LATE


# ---------------------------------------------------------------------------
# Public negotiator
# ---------------------------------------------------------------------------
class SemrukNegotiator(SAOMetaNegotiator):
    """ANAC 2026 HAN League agent for team **Semruk** (ID 375).

    Wraps a BOA-architecture core with a template-based text adapter so every
    outgoing message carries persuasive, context-aware natural language —
    important because HAN evaluators include human perception (questionnaires)
    in addition to raw utility.
    """

    def __init__(
        self,
        base_negotiator: GBNegotiator | None = None,
        boulware_e: float | str = "boulware",
        # v0.9 — scale-invariant thresholds (fractions of attainable surplus);
        # see _ParetoBoulwareOffering. ``min_aspiration`` is tuned on a
        # variance-aware round-robin across diverse scenarios (real + synthetic,
        # two seeds, vs Semruk self-play AND non-LLM baselines): 0.92 gave the
        # best Worst-case and lowest cross-scenario Std in every test — the
        # consistency goal. It is *safe* to be this tough only because the
        # rational acceptance net (``final_accept_t``) removes the deadlock-zero
        # tail. Tough offerer, reasonable acceptor.
        min_aspiration: float = 0.92,
        deception_prob: float = 0.30,
        phase1_t: float = 0.93,
        phase2_t: float = 0.97,
        endgame_t: float = 0.97,
        endgame_buffer: float = 0.10,
        final_accept_t: float = 0.985,
        adaptive_profile: bool = False,  # off by default — see semruk.py
        classify_after: int = 5,
        **kwargs: Any,
    ) -> None:
        if base_negotiator is None:
            base_negotiator = _SemrukCore(
                boulware_e=boulware_e,
                min_aspiration=min_aspiration,
                deception_prob=deception_prob,
                phase1_t=phase1_t,
                phase2_t=phase2_t,
                endgame_t=endgame_t,
                endgame_buffer=endgame_buffer,
                final_accept_t=final_accept_t,
                adaptive_profile=adaptive_profile,
                classify_after=classify_after,
            )
        super().__init__(
            negotiators=[base_negotiator],  # type: ignore[arg-type]
            negotiator_names=["semruk_core"],
            share_ufun=True,
            share_nmi=True,
            **kwargs,
        )
        self._deception_prob = deception_prob

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def base_negotiator(self) -> GBNegotiator:
        return self._negotiators[0]  # type: ignore[return-value]

    @property
    def opponent_ufun(self):
        """Forward opponent_ufun from the BOA core (used by deception scoring)."""
        base = self.base_negotiator
        opp = getattr(base, "opponent_ufun", None)
        if opp is not None:
            return opp
        return super().opponent_ufun  # fallback to default behaviour

    def _issues(self) -> list | None:
        if self.nmi is None:
            return None
        os_ = self.nmi.outcome_space
        if os_ is None:
            return None
        return getattr(os_, "issues", None)

    def _update_opp_model(self, state: SAOState) -> None:
        """Feed the opponent's most recent bid into the core's frequency
        model. Called from both ``propose`` and ``respond`` since negmas
        does not reliably trigger ``on_partner_proposal`` for adapted
        negotiators in this version.
        """
        offer = getattr(state, "current_offer", None)
        if offer is None:
            return
        core = self.base_negotiator
        model = getattr(core, "_opp_model", None)
        if model is not None:
            try:
                model.update(offer)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Propose / respond — delegate decision to core, attach text
    # ------------------------------------------------------------------
    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        # Update the opponent model with whatever they last offered. Even
        # though we are about to propose, ``state.current_offer`` reflects
        # their most recent move (None on the very first round).
        self._update_opp_model(state)
        base_proposal = self.base_negotiator.propose(state, dest=dest)
        if base_proposal is None:
            return None
        if isinstance(base_proposal, ExtendedOutcome):
            outcome = base_proposal.outcome
            base_data = dict(base_proposal.data or {})
        else:
            outcome = base_proposal
            base_data = {}
        if outcome is None:
            return None

        text = self._generate_text(state, "propose", outcome)
        base_data["text"] = text
        return ExtendedOutcome(outcome=outcome, data=base_data)

    def respond(
        self, state: SAOState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:  # type: ignore[override]
        # ``state.current_offer`` is exactly the opponent's bid we are about
        # to respond to. Update the model first.
        self._update_opp_model(state)
        base_response = self.base_negotiator.respond(state, source=source)
        if isinstance(base_response, ExtendedResponseType):
            response_type = base_response.response
            base_data = dict(base_response.data or {})
        else:
            response_type = base_response
            base_data = {}

        if response_type == ResponseType.ACCEPT_OFFER:
            base_data["text"] = self._generate_text(
                state, "accept", state.current_offer
            )
            return ExtendedResponseType(response=response_type, data=base_data)
        if response_type == ResponseType.END_NEGOTIATION:
            base_data["text"] = self._generate_text(state, "end", state.current_offer)
            return ExtendedResponseType(response=response_type, data=base_data)
        # Rejection text gets attached when the counter-offer is proposed,
        # so just forward the bare rejection.
        return base_response

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def _generate_text(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None,
    ) -> str:
        issues = self._issues()
        summary = _summarize_outcome(outcome, issues)

        if action == "accept":
            return random.choice(_ACCEPT_PHRASES).format(summary=summary)
        if action == "end":
            return random.choice(_END_PHRASES)
        if action == "propose":
            if state.current_offer is None:
                return random.choice(_OPENING_PHRASES).format(summary=summary)
            remark = _issue_remark(state, outcome, issues)
            template = random.choice(_phase_pool(state))
            return template.format(summary=summary, issue_remark=remark)
        return ""
