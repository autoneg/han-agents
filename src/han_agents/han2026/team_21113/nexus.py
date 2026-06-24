# nexus.py — NEXUSNegotiator main agent
# NEXUS = Neural Emotion and eXtended Understanding System (v4)
# HAN 2026 (ANAC at IJCAI 2026) competition agent — Team 387
#
# Architecture:
#   1. HybridOpponentModel  — 3-layer ensemble opponent preference model
#   2. ParetoAnalyser       — Pareto frontier + Nash bargaining point
#   3. LexiconPE            — manipulation-filtered emotion coefficient
#   4. BehaviourTracker     — DANS classifier + TFT concession ratio + PA
#   5. MiCRO-style concession index — utility-target walk using inverter built-ins
#   6. _estimate_achievable_utility — 5-signal blend replacing fixed hold
#   7. 13-condition ACcombi — graduated acceptance strategy
#   8. PresortingInverseUtilityFunction — O(1) candidate lookup

from __future__ import annotations

import math
from typing import Optional

from negmas import Outcome, ResponseType, SAOResponse, SAOState
from negmas.sao import SAONegotiator
from negmas.preferences import PresortingInverseUtilityFunction

from .opponent_model import HybridOpponentModel
from .pareto import ParetoAnalyser
from .sentiment import LexiconPE
from .behaviour import BehaviourTracker
from .diag import NegotiationDiag, log_negotiation, ENABLED as DIAG_ENABLED

# ── Module-level constants ────────────────────────────────────────────────────

BETA_ACCEPT = 0.15   # emotion-based acceptance relaxation
WINDOW_N    = 5

# ── Text templates (human-facing) ─────────────────────────────────────────────

TEXTS: dict[tuple[str, str], str] = {
    ("accept", "empathetic"): "I really appreciate your flexibility. I'm happy to accept.",
    ("accept", "balanced"):   "This works for me. I accept.",
    ("accept", "assertive"):  "Agreed. This is fair. Let's finalise.",
    ("reject", "empathetic"): "I can see we both want a fair deal. Here's an adjusted offer.",
    ("reject", "balanced"):   "Getting closer — here's a revised proposal.",
    ("reject", "assertive"):  "I need better terms. Here is my revised offer.",
    ("reject", "elicit"):     "I want to find something that works for us both. What matters most to you here?",
    ("reject", "tft"):        "I've matched your movement. Here's my counter-offer.",
}


class NEXUSNegotiator(SAONegotiator):
    """
    NEXUS v4: Neural Emotion and eXtended Understanding System.

    A HAN 2026 competition agent built for the Stochastic Alternating Offers
    protocol.  Extends SAONegotiator with:
    - Hybrid 3-layer opponent preference estimation (frequency + hard-headed + Bayesian)
    - Optional GNash 4th model (NegMAS built-in, attached if available)
    - Pareto frontier tracking and Nash bargaining point targeting
    - Lexicon-based emotion coefficient with Bayesian manipulation filter
    - DANS + TFT concession tracking
    - MiCRO-style concession: utility-target walk using inverter built-ins
    - _estimate_achievable_utility: 5-signal blend (Pareto/Nash/empirical/behaviour/RVFitter)
    - 13-condition acceptance strategy
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Subsystems (reinitialised on each negotiation)
        self._opp_model: Optional[HybridOpponentModel] = None
        self._pareto: Optional[ParetoAnalyser] = None
        self._sentiment: LexiconPE = LexiconPE(window=WINDOW_N)
        self._behaviour: BehaviourTracker = BehaviourTracker()
        self._inverter: Optional[PresortingInverseUtilityFunction] = None

        # Per-negotiation state (reset in on_negotiation_start)
        self._round: int = 0
        self._scenario: str = "default"
        self._reservation: float = 0.0
        self._ufun_max: float = 1.0
        self._ufun_min: float = 0.0
        self._all_outcomes: Optional[list] = None

        self._last_my_util: float = 1.0
        self._last_opp_util_est: float = 0.5
        self._my_util_history: list[float] = []
        self._opp_util_history: list[float] = []
        self._bid_history: list[float] = []

        self._best_recv_util: float = 0.0
        self._best_recv_offer: Optional[Outcome] = None
        self._recent_recv: list[float] = []

        # Concession state (MiCRO-style utility-target walk)
        self._achievable: float = 1.0
        self._concession_target_util: float = 1.0
        self._concession_ceiling: float = 1.0
        self._last_concession_t: float = 0.0
        self._in_hold_phase: bool = True
        self._concession_rate: float = 0.0

        # Pareto/Nash cache
        self._pareto_my_max: float = 1.0
        self._pareto_my_min: float = 0.0
        self._nash_util: float = 0.5
        self._pareto_built: bool = False

        self._pe: float = 0.0
        self._pa: float = 0.5
        self._tft_mode: bool = False

        # Step-granularity tracking (Change 2)
        self._dt: float = 0.03          # estimated time per step
        self._prev_t: float = 0.0
        self._n_steps_known: bool = False

        self._diag: NegotiationDiag = NegotiationDiag()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_preferences_changed(self, changes=None) -> None:
        super().on_preferences_changed(changes)
        if self.ufun is None:
            return
        try:
            self._reservation = float(self.ufun.reserved_value or 0.0)
            self._ufun_min, self._ufun_max = self.ufun.minmax()
            self._all_outcomes = None
            self._inverter = PresortingInverseUtilityFunction(self.ufun)
            self._inverter.init()
            self._opp_model = HybridOpponentModel(self.ufun)
            self._pareto = ParetoAnalyser(self.ufun, self._opp_model)
        except Exception:
            pass
        try:
            self._scenario = self._detect_scenario()
        except Exception:
            self._scenario = "default"
        self._try_attach_gnash()

    def on_negotiation_start(self, state: SAOState) -> None:
        """Reset ALL per-negotiation counters at the start of each negotiation."""
        super().on_negotiation_start(state)

        self._round = 0
        self._last_my_util = 1.0
        self._last_opp_util_est = 0.5
        self._my_util_history = []
        self._opp_util_history = []
        self._bid_history = []

        self._best_recv_util = 0.0
        self._best_recv_offer = None
        self._recent_recv = []

        self._achievable = 1.0
        self._concession_target_util = 1.0
        self._concession_ceiling = 1.0
        self._last_concession_t = 0.0
        self._in_hold_phase = True
        self._concession_rate = 0.0

        self._pareto_my_max = 1.0
        self._pareto_my_min = 0.0
        self._nash_util = 0.5
        self._pareto_built = False

        self._pe = 0.0
        self._pa = 0.5
        self._tft_mode = False

        self._dt = 0.03
        self._prev_t = 0.0
        self._n_steps_known = False

        self._sentiment = LexiconPE(window=WINDOW_N)
        self._behaviour = BehaviourTracker()

        try:
            self._scenario = self._detect_scenario()
        except Exception:
            self._scenario = "default"

        if self.ufun is not None and self._opp_model is not None:
            try:
                self._opp_model.reinit(self.ufun)
                self._pareto = ParetoAnalyser(self.ufun, self._opp_model)
            except Exception:
                pass
        self._try_attach_gnash()

        # ── diagnostics ──────────────────────────────────────────────
        self._diag = NegotiationDiag()
        if DIAG_ENABLED:
            try:
                self._diag.scenario = self._scenario
                self._diag.rv = self._reservation
                if self.nmi is not None:
                    self._diag.n_steps_mech = getattr(self.nmi, "n_steps", None)
                    self._diag.time_limit_mech = getattr(self.nmi, "time_limit", None)
                    try:
                        self._diag.n_outcomes = self.nmi.outcome_space.cardinality
                    except Exception:
                        pass
            except Exception:
                pass

    def _try_attach_gnash(self) -> None:
        """Attempt to attach the NegMAS GNash opponent model (optional)."""
        if self._opp_model is None:
            return
        try:
            from negmas.sao import GNashFrequencyModel  # type: ignore
            gnash = GNashFrequencyModel()
            gnash.negotiator = self
            self._opp_model._gnash = gnash
        except Exception:
            pass

    # ── Main per-round call ───────────────────────────────────────────────────

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        self._round += 1

        t = state.relative_time
        if t is None or not math.isfinite(t):
            t = min(1.0, self._round / 100.0)
        t = max(0.0, min(1.0, t))

        # Adaptively estimate step size from observed time deltas
        if not self._n_steps_known and self.nmi is not None:
            n_steps = getattr(self.nmi, "n_steps", None)
            if n_steps is not None and n_steps > 0:
                self._dt = 1.0 / n_steps
                self._n_steps_known = True
        if not self._n_steps_known and self._round >= 2:
            measured = t - self._prev_t
            if measured > 0:
                self._dt = 0.8 * self._dt + 0.2 * measured
        self._prev_t = t

        opp_offer = state.current_offer
        opp_text  = self._extract_text(state)

        my_util = (
            float(self.ufun(opp_offer))
            if opp_offer is not None and self.ufun is not None
            else 0.0
        )

        # ── Process opponent offer ─────────────────────────────────────────
        if opp_offer is not None:
            if self._opp_model is not None:
                self._opp_model.update(opp_offer, t)

            if my_util > self._best_recv_util:
                self._best_recv_util = my_util
                self._best_recv_offer = opp_offer

            self._recent_recv.append(my_util)
            if len(self._recent_recv) > 6:
                self._recent_recv.pop(0)

            est_opp = (
                self._opp_model.get_estimated_utility(opp_offer)
                if self._opp_model is not None
                else 0.5
            )
            self._opp_util_history.append(est_opp)
            self._my_util_history.append(my_util)

            agent_change   = my_util - self._last_my_util
            opp_change     = est_opp - self._last_opp_util_est
            opp_concession = max(0.0, self._last_opp_util_est - est_opp)
            self._behaviour.record_opponent(agent_change, opp_change, opp_concession)
            self._last_my_util = my_util
            self._last_opp_util_est = est_opp

            self._update_concession_rate()

            if not self._pareto_built and self._round >= 8:
                self._build_pareto()
            elif self._pareto_built and self._round % 15 == 0:
                self._build_pareto()

        # ── Process text ───────────────────────────────────────────────────
        if opp_text:
            self._pe = self._sentiment.update(opp_text, my_util, t)

        self._pa = self._behaviour.pa

        # ── TFT mode detection ─────────────────────────────────────────────
        if (not self._tft_mode
                and t >= 0.25
                and len(self._behaviour.opponent_moves) >= 6
                and self._behaviour.agent_total_concession >= 0.03
                and self._behaviour.tft_ratio > 1.2):
            self._tft_mode = True

        # ── Compute achievable estimate (drives acceptance threshold) ───────
        self._achievable = self._estimate_achievable_utility(t)
        tu = self._achievable

        if DIAG_ENABLED:
            self._diag.round_snapshot(
                t,
                self._achievable,
                self._concession_target_util,
                my_util if opp_offer is not None else None,
            )

        # ── Plan our counter-offer FIRST (needed for ACnext acceptance) ────
        planned_offer = self._select_offer(t)
        planned_u = 1.0
        if planned_offer is not None and self.ufun is not None:
            try:
                planned_u = float(self.ufun(planned_offer))
            except Exception:
                planned_u = 1.0

        # ── Acceptance check ───────────────────────────────────────────────
        if opp_offer is not None and self._should_accept(
                opp_offer, my_util, tu, t, planned_u):
            text = self._make_text("accept", tu, t, my_util)
            return SAOResponse(ResponseType.ACCEPT_OFFER, opp_offer, {"text": text})

        # ── First move (we go first — no opponent offer yet) ───────────────
        if opp_offer is None:
            my_offer = planned_offer
            if my_offer is None:
                my_offer = self._emergency_offer()
            if my_offer is None:
                return SAOResponse(ResponseType.END_NEGOTIATION, None)
            text = self._make_text("reject", tu, t, 0.0)
            self._track_agent_move(my_offer, is_first=True)
            if DIAG_ENABLED and self.ufun is not None:
                try:
                    self._diag.offered(float(self.ufun(my_offer)))
                except Exception:
                    pass
            return SAOResponse(ResponseType.REJECT_OFFER, my_offer, {"text": text})

        # ── Counter-offer ──────────────────────────────────────────────────
        my_offer = planned_offer
        if my_offer is None:
            my_offer = self._emergency_offer()
        if my_offer is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # Hardliner deal saver: opponent never offered anything meaningfully
        # above our reservation and the deadline is here. Offer the outcome
        # THEY value most among outcomes still rational for us — maximises
        # the chance a hardheaded opponent accepts instead of a 0-0 walkaway.
        if (self._steps_left(t) <= 2
                and self._best_recv_util <= self._reservation + 0.05
                and self._opp_model is not None):
            try:
                cands = self._get_candidates(
                    self._reservation + 0.02,
                    min(1.0, self._reservation + 0.30),
                )
                if cands:
                    my_offer = max(
                        cands, key=self._opp_model.get_estimated_utility
                    )
            except Exception:
                pass

        # Last-turns deal saver: re-propose our best received offer
        # when only 1-2 steps remain and the opponent won't close without movement
        if self._steps_left(t) <= 2 and self._best_recv_offer is not None:
            try:
                best_u = float(self.ufun(self._best_recv_offer)) if self.ufun else 0.0
                if (best_u > self._reservation + 0.02
                        and best_u > float(self.ufun(my_offer))):
                    my_offer = self._best_recv_offer
            except Exception:
                pass

        self._track_agent_move(my_offer)
        text = self._make_text("reject", tu, t, my_util)
        if DIAG_ENABLED and self.ufun is not None:
            try:
                self._diag.offered(float(self.ufun(my_offer)))
            except Exception:
                pass
        return SAOResponse(ResponseType.REJECT_OFFER, my_offer, {"text": text})

    # ── Agent-move tracking ───────────────────────────────────────────────────

    def _track_agent_move(self, my_offer: Outcome, is_first: bool = False) -> None:
        try:
            my_util_out = float(self.ufun(my_offer)) if self.ufun else 0.0
            prev_bid = self._bid_history[-1] if self._bid_history else 1.0
            my_change = my_util_out - prev_bid
            agent_concession = max(0.0, prev_bid - my_util_out)
            est_opp_our = (
                self._opp_model.get_estimated_utility(my_offer)
                if self._opp_model is not None
                else 0.5
            )
            opp_change = est_opp_our - self._last_opp_util_est
            self._behaviour.record_agent(my_change, opp_change, agent_concession)
            self._bid_history.append(my_util_out)
        except Exception:
            pass

    # ── Achievable utility estimation ─────────────────────────────────────────

    def _estimate_achievable_utility(self, t: float) -> float:
        """
        5-signal blend estimating the highest utility realistically achievable.

        Signals:
          1. Pareto ceiling  — structural max on the Pareto frontier
          2. Nash anchor     — cooperative equilibrium target
          3. Empirical ceil  — best_recv_util + buffer (opponent willing to give this)
          4. Behaviour mult  — up if opponent conceding, down if stuck
          5. RVFitter signal — up if opponent rv is very low (more room to concede)

        Blend: early → structural (Pareto/Nash); late → empirical.
        """
        rv = self._reservation

        # Signal 1: Pareto ceiling
        pareto_hi = self._pareto_my_max if self._pareto_built else self._ufun_max

        # Signal 2: Nash anchor
        nash_u = self._nash_util

        # Signal 3: Empirical ceiling
        if self._best_recv_util > rv + 0.02:
            empirical = min(pareto_hi, self._best_recv_util + 0.08)
        else:
            empirical = pareto_hi  # no data → be optimistic

        # Signal 4: Behaviour multiplier
        if self._behaviour.is_opponent_conceding(window=5):
            rate_mult = 1.06
        elif self._behaviour.is_opponent_stuck(window=6):
            rate_mult = 0.92
        else:
            rate_mult = 1.00

        # Signal 5: RVFitter — opponent has low rv → still has room → hold higher
        if (self._opp_model is not None
                and self._opp_model._rv_fitted
                and self._opp_model.estimated_opponent_rv < 0.15):
            rate_mult = min(rate_mult * 1.05, 1.15)

        # Time-weighted blend
        if not self._pareto_built or t < 0.20:
            base = pareto_hi * 0.88
        elif t < 0.50:
            alpha = (t - 0.20) / 0.30
            structural = max(pareto_hi * 0.88, nash_u)
            base = (1.0 - alpha) * structural + alpha * empirical
        else:
            alpha = min(1.0, (t - 0.50) / 0.35)
            floor = max(self._best_recv_util + 0.02, rv + 0.08)
            # Patient opponents (Boulware-style) reveal almost nothing before
            # the deadline, so best_recv is not a ceiling estimate mid-game.
            # Keep the Nash anchor as a floor while the opponent still has
            # time to concede.
            if self._pareto_built and t < 0.90:
                floor = max(floor, min(self._nash_util, pareto_hi) * 0.90)
            base = (1.0 - alpha) * empirical + alpha * floor

        achievable = base * rate_mult

        # FIX 4: Cap achievable by empirical evidence once data exists.
        # Prevents pareto_hi=1.0 (before opponent model converges) from
        # keeping achievable at 0.88 when the real ZOA ceiling is much lower.
        # FIX 5: time-gated (was round>=5, i.e. t=0.05 on 100-step runs) —
        # capping on early offers treated the opponent's OPENING position as
        # the ZOA ceiling and collapsed our whole concession band onto it.
        if t >= 0.60 and self._best_recv_util > rv + 0.02:
            cap = empirical * 1.15
            if self._pareto_built and t < 0.90:
                cap = max(cap, min(self._nash_util, pareto_hi) * 0.90)
            achievable = min(achievable, cap)

        return max(rv + 0.04, min(pareto_hi, achievable))

    def _update_concession_rate(self) -> None:
        """Track how fast the opponent concedes (measured as my utility of their offers)."""
        hist = self._my_util_history
        if len(hist) < 2:
            self._concession_rate = 0.0
            return
        window = hist[-WINDOW_N:]
        total = 0.0
        for i in range(1, len(window)):
            total += max(0.0, window[i] - window[i - 1])
        self._concession_rate = total / max(1, len(window) - 1)

    # ── Concession index / offer selection ────────────────────────────────────

    def _get_target_outcome(self, t: float) -> Optional[Outcome]:
        """
        MiCRO-style utility-target walk.

        Hold phase (t < 0.65): offer near ufun_max (best rational outcome).
        Concession phase (t ≥ 0.65): closed-form Boulware descent from
        _achievable down to floor (best_recv + 0.02 or rv + 0.04).
        Step-count-invariant — behavior identical at n_steps=30 and n_steps=300.
        """
        if self._inverter is None or self.ufun is None:
            return None
        rv = self._reservation

        # ── HOLD PHASE ────────────────────────────────────────────────────
        if t < 0.65:
            # Track achievable live so concession phase starts correctly.
            self._concession_target_util = self._achievable
            self._last_concession_t = t
            self._in_hold_phase = True
            # Open at the highest possible utility (best rational outcome).
            # _achievable is used as the FLOOR for the concession phase,
            # not as the opening anchor. This matches MiCRO's approach of
            # starting at rank 0 and only conceding when necessary.
            # worst_in_range(lo, 1.0) returns the LOWEST utility outcome
            # in the band [lo, 1.0] — which is the most Pareto-cooperative
            # offer at our current utility level.
            # By setting lo = rv + 0.01 we offer our best outcome first,
            # then only lower if the concession phase forces it.
            hi = self._ufun_max if self._ufun_max > 0 else 1.0
            target = self._worst_in_range((max(rv + 0.01, hi - 0.02), hi))
            if target is None:
                target = self._worst_in_range((max(rv + 0.01, self._achievable), 1.0))
            return target or self._worst_in_range((rv, 1.0)) or self._emergency_offer()

        # ── FIRST ENTRY into concession phase ─────────────────────────────
        # FIX 5 (anti-cliff): descend FROM the level we actually held at
        # (our last bid, near ufun_max), NOT from _achievable. _achievable
        # collapses toward best_recv+0.10 against patient opponents
        # (Boulware et al.), so using it as the ceiling made the very first
        # concession-phase offer plunge to the opponent's current position —
        # which they immediately accepted (observed: deals at t=0.66 with
        # our utility 0.37-0.47 vs opponent 0.85+).
        if self._in_hold_phase:
            self._in_hold_phase = False
            last_bid = self._bid_history[-1] if self._bid_history else self._ufun_max
            self._concession_ceiling = max(self._achievable, last_bid)
            self._concession_target_util = self._concession_ceiling

        # ── CONCESSION PHASE — closed-form Boulware descent ──────────────
        # prog=0 at t=0.65 (concession start), prog=1 at t=1.0
        # curve is Boulware-shaped: slow early, faster near deadline
        prog = max(0.0, min(1.0, (t - 0.65) / 0.35))
        curve = (1.0 - prog) ** 1.5
        ceiling = self._concession_ceiling
        floor = max(self._best_recv_util + 0.02, rv + 0.04)
        if floor >= ceiling:
            floor = max(rv + 0.02, ceiling - 0.01)
        raw_target = floor + (ceiling - floor) * curve
        new_target = max(rv + 0.02, min(ceiling, raw_target))
        if DIAG_ENABLED and new_target < self._concession_target_util - 0.005:
            self._diag.concession_steps += 1
        self._concession_target_util = new_target

        lo = max(rv, self._concession_target_util - 0.02)
        target = self._worst_in_range((lo, 1.0))
        return target or self._worst_in_range((rv, 1.0)) or self._emergency_offer()

    def _steps_left(self, t: float) -> float:
        """Estimated steps remaining — step-count-invariant."""
        return max(0.0, (1.0 - t) / max(self._dt, 1e-6))

    def _worst_in_range(self, rng: tuple[float, float]) -> Optional[Outcome]:
        """Return the worst (lowest-utility) outcome in the given utility band."""
        try:
            result = self._inverter.worst_in(rng, normalized=False)
            if result is not None:
                return result
        except Exception:
            pass
        # Fallback: use some() and pick minimum
        try:
            candidates = self._inverter.some(rng=rng, normalized=False)
            if candidates:
                return min(candidates, key=lambda o: float(self.ufun(o)))
        except Exception:
            pass
        return None

    def _select_offer(self, t: float) -> Optional[Outcome]:
        """
        Get the concession-index target, then pick the Nash-optimal outcome
        among candidates in a ±0.03 utility band around that target.
        """
        if self.ufun is None:
            return None

        target = self._get_target_outcome(t)
        if target is None:
            return self._emergency_offer()

        try:
            target_u = float(self.ufun(target))
        except Exception:
            return target

        rv = self._reservation
        lo = max(rv, target_u - 0.03)
        hi = min(1.0, target_u + 0.03)

        candidates = self._get_candidates(lo, hi)
        if not candidates:
            return target

        prev_opp = (
            self._opp_util_history[-2] if len(self._opp_util_history) >= 2
            else self._opp_util_history[-1] if self._opp_util_history
            else 0.5
        )

        def nash_score(o: Outcome) -> float:
            try:
                u   = float(self.ufun(o))
                opp = (
                    self._opp_model.get_estimated_utility(o)
                    if self._opp_model is not None
                    else 0.5
                )
                score = (u - rv) * opp

                if self._pareto_built and self._pareto is not None:
                    if self._pareto.distance_to_pareto(o) < 0.05:
                        score += 0.02

                if self._tft_mode and self._opp_util_history:
                    opp_conc = prev_opp - self._opp_util_history[-1]
                    prev_bid = self._bid_history[-1] if self._bid_history else 1.0
                    tft_target = prev_bid - opp_conc
                    score += max(0.0, 0.03 - abs(u - tft_target) * 0.3)

                if t > 0.7 and self._pareto_built and self._pareto is not None:
                    nash_o = self._pareto.get_nash_outcome()
                    if nash_o is not None and o == nash_o:
                        score += 0.05

                return score
            except Exception:
                return 0.0

        try:
            return max(candidates, key=nash_score)
        except Exception:
            return candidates[0] if candidates else target

    def _get_candidates(self, lo: float, hi: float) -> list[Outcome]:
        if self._inverter is not None:
            try:
                result = self._inverter.some(rng=(lo, hi), normalized=False)
                if result:
                    return list(result)
            except Exception:
                pass
        return self._manual_candidates(lo, hi)

    def _manual_candidates(self, lo: float, hi: float) -> list[Outcome]:
        if self.ufun is None:
            return []
        try:
            if self._all_outcomes is None:
                self._all_outcomes = list(
                    self.nmi.outcome_space.enumerate_or_sample(
                        levels=10, max_cardinality=10_000
                    )
                )
            return [o for o in self._all_outcomes if lo <= float(self.ufun(o)) <= hi]
        except Exception:
            return []

    def _emergency_offer(self) -> Optional[Outcome]:
        if self.ufun is None:
            return None
        try:
            return self.ufun.best()
        except Exception:
            pass
        try:
            if self._all_outcomes is None:
                self._all_outcomes = list(
                    self.nmi.outcome_space.enumerate_or_sample(
                        levels=10, max_cardinality=10_000
                    )
                )
            rv = self._reservation
            valid = [o for o in self._all_outcomes if float(self.ufun(o)) >= rv]
            return max(valid, key=lambda o: float(self.ufun(o))) if valid else None
        except Exception:
            return None

    # ── Acceptance strategy ───────────────────────────────────────────────────

    def _accept(self, reason: str) -> bool:
        if DIAG_ENABLED:
            self._diag.accept_reason = reason
        return True

    def _should_accept(
        self, offer: Outcome, my_util: float, tu: float, t: float,
        planned_u: float = 1.0,
    ) -> bool:
        rv = self._reservation
        sl = self._steps_left(t)

        # C1a: ACnext — offer is at least as good as our own planned next bid.
        # (Replaces the old "meets achievable" primary condition, which got
        # exploited: achievable collapses toward the opponent's revealed
        # position against patient opponents, so we accepted near THEIR terms.)
        if my_util >= planned_u - 0.005 and my_util > rv:
            return self._accept("C1a_acnext")

        # C1b: meets achievable estimate — only once mid-game (achievable is
        # Nash-floored until t=0.9, so this cannot fire on lowball offers)
        if t >= 0.60 and my_util >= tu:
            return self._accept("C1b_meets_achievable")

        # C2: opponent actively conceding AND offer is close to target
        if (t >= 0.70
                and self._behaviour.is_opponent_conceding(window=4)
                and my_util >= tu - 0.05
                and my_util >= self._concession_target_util - 0.05
                and my_util > rv + 0.10):
            return self._accept("C2_dans_concession")

        # C3: very few steps left — best offer seen, don't let it slip
        if sl <= 4 and my_util >= self._best_recv_util - 0.02 and my_util > rv + 0.05:
            return self._accept("C3_best_near_deadline")

        # C4: opponent has settled (plateau) — late-game only with quality guard
        # FIX: was round>=20 which fired at t≈0.07 on 300-step negotiations,
        # accepting opening offers of patient opponents (mean util 0.49)
        # FIX 5: t>=0.90 (was 0.70) — Boulware-style opponents plateau
        # mid-game by design and concede only near the deadline; accepting
        # their mid-game plateau hands them the whole surplus.
        if t >= 0.90 and len(self._recent_recv) >= 5:
            spread = max(self._recent_recv[-5:]) - min(self._recent_recv[-5:])
            if (spread <= 0.03
                    and my_util >= self._best_recv_util * 0.90
                    and my_util > rv + 0.05):
                return self._accept("C4_opponent_settled")

        # C5: offer is ≥95% of achievable AND best seen — time-gated (not round-gated)
        # FIX: was round>=15 which fired at t≈0.15 on 100-step negotiations
        # FIX 5: t>=0.85 (was 0.75) — give patient opponents time to concede
        if (t >= 0.85
                and my_util >= self._achievable * 0.95
                and my_util >= self._best_recv_util * 0.92
                and my_util > rv + 0.03):
            return self._accept("C5_95pct_achievable")

        # C6: opponent stuck AFTER we have moved — their plateau is their final offer
        if (self._behaviour.is_opponent_stuck(window=6)
                and t > 0.80
                and self._behaviour.agent_total_concession >= 0.03
                and my_util > rv + 0.10
                and my_util >= self._best_recv_util * 0.95):
            return self._accept("C6_opponent_stuck")

        # C7: offer is the Nash bargaining outcome
        if self._pareto_built and t > 0.50 and self._pareto is not None:
            nash_o = self._pareto.get_nash_outcome()
            if nash_o is not None and offer == nash_o and my_util > rv + 0.05:
                return self._accept("C7_nash_point")

        # C8: TFT — only after we have moved, late-game, near target
        if (self._tft_mode
                and self._behaviour.tft_ratio > 1.5
                and self._behaviour.agent_total_concession >= 0.05
                and t >= 0.70
                and my_util >= tu - 0.05
                and my_util > rv + 0.10):
            return self._accept("C8_tft")

        # C9–C12: deadline pressure using steps_left (step-count-invariant)
        # FIX: t>0.97 was unreachable at n_steps=30 (max t = 30/31 ≈ 0.968)
        if sl <= 1 and my_util > rv:
            return self._accept("C9_deadline_1step")
        if sl <= 2 and my_util > rv + 0.03:
            return self._accept("C10_deadline_2step")
        if sl <= 4 and my_util > rv + 0.08:
            return self._accept("C11_deadline_4step")
        if sl <= 8 and my_util > rv + 0.15:
            return self._accept("C12_deadline_8step")

        # C13: emotional signal — deteriorating negotiation, accept near target
        pe = self._pe
        if (pe < -0.6
                and self._sentiment.trajectory_slope < -0.05
                and my_util >= tu - BETA_ACCEPT * abs(pe)):
            return self._accept("C13_emotion")

        return False

    # ── Text generation ───────────────────────────────────────────────────────

    def _make_text(
        self, response_type: str, tu: float, t: float, opp_util: float
    ) -> str:
        pe = self._pe
        if pe < -0.3:
            tone = "empathetic"
        elif pe > 0.3:
            tone = "assertive"
        else:
            tone = "balanced"

        if self._round <= 2 and response_type == "reject":
            return TEXTS.get(
                ("reject", "elicit"), TEXTS.get(("reject", "balanced"), "Here is my offer.")
            )

        if response_type == "reject" and self._behaviour.is_opponent_conceding(window=3):
            base = TEXTS.get(("reject", tone), "Here is my offer.")
            return "I appreciate you moving toward a better outcome. " + base

        if self._tft_mode and response_type == "reject":
            return TEXTS.get(("reject", "tft"), TEXTS.get(("reject", "balanced"), "Here is my offer."))

        return TEXTS.get((response_type, tone), "Here is my offer.")

    def on_negotiation_end(self, state: SAOState) -> None:
        super().on_negotiation_end(state)
        if not DIAG_ENABLED:
            return
        try:
            agreement = getattr(state, "agreement", None)
            final_util = (
                float(self.ufun(agreement))
                if agreement is not None and self.ufun is not None
                else 0.0
            )
            end_t = state.relative_time if state.relative_time is not None else 1.0
            self._diag.opp_rv_est = (
                self._opp_model.estimated_opponent_rv
                if self._opp_model is not None and self._opp_model._rv_fitted
                else None
            )
            self._diag.tft_mode = self._tft_mode
            log_negotiation(
                self._diag.finalize(agreement, final_util, end_t, self._round)
            )
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_pareto(self) -> None:
        if self._pareto is None or self.ufun is None:
            return
        try:
            if self._all_outcomes is None:
                self._all_outcomes = list(
                    self.nmi.outcome_space.enumerate_or_sample(
                        levels=10, max_cardinality=10_000
                    )
                )
            self._pareto.build(self._all_outcomes, self._reservation)
            self._pareto_built = self._pareto.is_built
            if DIAG_ENABLED and self._pareto_built and self._diag.pareto_built_round is None:
                self._diag.pareto_built_round = self._round
            if self._pareto_built:
                lo, hi = self._pareto.frontier_my_range()
                self._pareto_my_min = lo
                self._pareto_my_max = hi
                nash_pt = self._pareto.get_nash_point()
                if nash_pt is not None:
                    self._nash_util = nash_pt[0]
        except Exception:
            self._pareto_built = False

    def _detect_scenario(self) -> str:
        try:
            if self.nmi and self.nmi.outcome_space:
                os_name = getattr(self.nmi.outcome_space, "name", "") or ""
                for name in ("Grocery", "Island", "Trade"):
                    if name.lower() in os_name.lower():
                        return name
        except Exception:
            pass
        return "default"

    def _extract_text(self, state: SAOState) -> str:
        try:
            v = getattr(state, "text", None)
            if v:
                return str(v)
        except Exception:
            pass
        for data_attr in ("current_data", "data"):
            try:
                d = getattr(state, data_attr, None)
                if isinstance(d, dict):
                    v = d.get("text")
                    if v:
                        return str(v)
            except Exception:
                pass
        return ""
