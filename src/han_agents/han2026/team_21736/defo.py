from __future__ import annotations

import random

from negmas import ResponseType, SAOResponse, SAOState
from negmas.outcomes import Outcome
from negmas.sao import SAONegotiator


# ---------------------------------------------------------------------------
# HanNegotiator
# ---------------------------------------------------------------------------

class Han(SAONegotiator):
    """
    Adaptive multi-phase negotiator for HAN 2026.

    Phase 0 – Exploration  (t < explore_end):
        Propose near-best outcome; gather opponent frequency data.

    Phase 1 – Boulware    (explore_end ≤ t < boulware_end):
        Slow, Boulware-style concession curve (exponent-driven).

    Phase 2 – Accelerate  (boulware_end ≤ t < deadline_start):
        Faster concessions; ACNext acceptance becomes more permissive.

    Phase 3 – Deadline    (deadline_start ≤ t < 1.0):
        Accept anything above reservation + margin; send urgency messages.
    """

    # ------------------------------------------------------------------
    # Message banks – class-level so they survive any import path
    # ------------------------------------------------------------------
    _OPENING = [
        "Thank you for joining this negotiation. "
        "I'm looking forward to reaching a mutually beneficial agreement.",
        "I appreciate the opportunity to negotiate with you today. "
        "Let's work together to find a deal we both like.",
        "Hello! I'm committed to finding a fair outcome for both of us. "
        "Let me share my initial position.",
    ]
    _CONCESSION_MSG = [
        "I've reconsidered and am moving toward a more balanced position — "
        "I hope you'll meet me halfway.",
        "I'm making a concession here to demonstrate good faith. "
        "Let's work towards an agreement.",
        "I'm adjusting my proposal to be more reasonable. "
        "I hope this shows my commitment to reaching a deal.",
    ]
    _HARD_MSG = [
        "This is an important issue for me, so I cannot move much further on this point.",
        "I need to maintain my position here — this is close to my walk-away point.",
        "My needs require me to hold firm on this. I hope you understand.",
    ]
    _ACCEPT_MSG = [
        "I'm happy to accept this offer — it works for both of us!",
        "This looks good to me. Let's go ahead and agree on this.",
        "I can work with this offer. Accepted!",
    ]
    _COUNTER_MSG = [
        "Here's a counter-offer I believe is fair given the circumstances.",
        "I'd like to propose this alternative that I think is more balanced.",
        "Let me suggest a slightly different arrangement that may work better for us both.",
    ]
    _ENDGAME_MSG = [
        "We're running out of time — I urge you to consider this proposal carefully.",
        "Time is short; this may be our last chance to reach an agreement.",
        "As our deadline approaches, I strongly encourage you to accept this offer.",
    ]
    _BREAKDOWN_MSG = [
        "Unfortunately I cannot accept offers below my minimum requirements. "
        "I must end the negotiation.",
        "I regret that we were unable to find common ground within the available time.",
    ]

    @staticmethod
    def _pick(lst):
        return random.choice(lst)

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        *args,
        boulware_exponent: float = 4.0,
        acceptance_margin: float = 0.02,
        ac_next_delta: float = 0.01,
        boulware_end: float = 0.75,
        explore_end: float = 0.15,
        deadline_start: float = 0.90,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.boulware_exponent = boulware_exponent
        self.acceptance_margin = acceptance_margin
        self.ac_next_delta = ac_next_delta
        self.boulware_end = boulware_end
        self.explore_end = explore_end
        self.deadline_start = deadline_start

        self._step: int = 0
        self._opponent_offers: list = []
        self._my_offers: list = []
        self._sorted_outcomes: list | None = None
        self._first_turn: bool = True

    # ---------------------------------------------------------- NegMAS hooks
    def on_preferences_changed(self, changes=None):
        self._sorted_outcomes = None
        self._first_turn = True

    # ------------------------------------------------------- cached outcomes
    def _get_sorted_outcomes(self) -> list:
        if self._sorted_outcomes is None:
            if self.ufun is None or self.nmi is None:
                return []
            try:
                outcomes = list(
                    self.nmi.outcome_space.enumerate_or_sample(max_cardinality=10_000)
                )
            except Exception:
                outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
            scored = []
            for o in outcomes:
                try:
                    scored.append((float(self.ufun(o)), o))
                except Exception:
                    pass
            scored.sort(key=lambda x: -x[0])
            self._sorted_outcomes = scored
        return self._sorted_outcomes

    # ------------------------------------------------------- utility helpers
    def _reservation(self) -> float:
        if self.ufun is None:
            return 0.0
        try:
            rv = self.ufun.reserved_value
            return float(rv) if rv is not None else 0.0
        except Exception:
            return 0.0

    def _utility(self, outcome) -> float:
        if outcome is None or self.ufun is None:
            return 0.0
        try:
            return float(self.ufun(outcome))
        except Exception:
            return 0.0

    # --------------------------------------------------- aspiration function
    def _aspiration(self, t: float) -> float:
        """
        aspiration(t) = rv + (1 - rv) * (1 - t^e)
        At t=0 → 1.0 ; at t=1 → rv
        """
        rv = self._reservation()
        return rv + (1.0 - rv) * (1.0 - t ** self.boulware_exponent)

    # -------------------------------------------- opponent modelling helpers
    def _estimate_opponent_best_offer(self):
        if not self._opponent_offers:
            return None
        rv = self._reservation()
        freq: dict = {}
        for o in self._opponent_offers:
            try:
                key = tuple(o) if o is not None else None
                if key is not None:
                    freq[key] = freq.get(key, 0) + 1
            except Exception:
                pass

        best_util = -1.0
        best_outcome = None
        for key in freq:
            for off in self._opponent_offers:
                try:
                    if tuple(off) == key:
                        u = self._utility(off)
                        if u >= rv and u > best_util:
                            best_util = u
                            best_outcome = off
                        break
                except Exception:
                    pass
        return best_outcome

    # --------------------------------------------- offer generation helpers
    def _outcome_at_utility(self, target_util: float):
        sorted_oc = self._get_sorted_outcomes()
        if not sorted_oc:
            return None
        rv = self._reservation()
        for u, o in sorted_oc:
            if u >= target_util:
                return o
        for u, o in sorted_oc:
            if u >= rv:
                return o
        return sorted_oc[0][1]

    def _build_counter_offer(self, t: float):
        target = self._aspiration(t)
        base = self._outcome_at_utility(target)

        opp_best = self._estimate_opponent_best_offer()
        if opp_best is not None:
            opp_u = self._utility(opp_best)
            rv = self._reservation()
            if opp_u >= rv + self.acceptance_margin and opp_u >= target - 0.05:
                return opp_best

        return base

    # --------------------------------------------------- text message helpers
    def _generate_text(self, t: float, response_type: str, offer) -> str:
        if response_type == "accept":
            return self._pick(self._ACCEPT_MSG)

        if self._first_turn:
            self._first_turn = False
            return self._pick(self._OPENING)

        if t >= self.deadline_start:
            msg = self._pick(self._ENDGAME_MSG)
            if offer is not None:
                u = self._utility(offer)
                msg += f"\n\n*This offer gives me a utility of **{u:.0%}** of my maximum.*"
            return msg

        if self._my_offers:
            last_u = self._utility(self._my_offers[-1])
            curr_u = self._utility(offer) if offer is not None else last_u
            if last_u - curr_u > 0.05:
                return self._pick(self._CONCESSION_MSG)
            return self._pick(self._HARD_MSG)

        return self._pick(self._COUNTER_MSG)

    # --------------------------------------------------------- main __call__
    def __call__(self, state: SAOState, **kwargs) -> SAOResponse:
        """
        Main negotiation callback.

        Accepts **kwargs and discards unknown keys (e.g. `dest`) for
        compatibility with servers that pass extra keyword arguments.
        """
        # --- guard: no ufun / nmi yet → end gracefully
        if self.ufun is None or self.nmi is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer
        t = float(state.relative_time)

        if offer is not None:
            self._opponent_offers.append(offer)

        self._step += 1
        rv = self._reservation()
        asp = self._aspiration(t)

        # ---- Acceptance decision ----------------------------------------
        if offer is not None:
            u_offer = self._utility(offer)

            # (a) Meets aspiration
            if u_offer >= asp:
                return SAOResponse(
                    ResponseType.ACCEPT_OFFER,
                    offer,
                    self._pick(self._ACCEPT_MSG),
                )

            # (b) ACNext: offer ≥ what we'd counter with (- small delta)
            next_offer = self._build_counter_offer(t)
            u_next = self._utility(next_offer) if next_offer is not None else 0.0
            if u_offer >= u_next - self.ac_next_delta:
                return SAOResponse(
                    ResponseType.ACCEPT_OFFER,
                    offer,
                    self._pick(self._ACCEPT_MSG),
                )

            # (c) Deadline pressure
            if t >= self.deadline_start and u_offer >= rv + self.acceptance_margin:
                return SAOResponse(
                    ResponseType.ACCEPT_OFFER,
                    offer,
                    self._pick(self._ACCEPT_MSG) + " We're running short on time!",
                )

            # (d) Hard walk-away at final ticks
            if t >= 0.98 and u_offer < rv:
                return SAOResponse(
                    ResponseType.END_NEGOTIATION,
                    None,
                    self._pick(self._BREAKDOWN_MSG),
                )

        # ---- Build counter-offer ----------------------------------------
        counter = self._build_counter_offer(t)
        if counter is None:
            return SAOResponse(
                ResponseType.END_NEGOTIATION,
                None,
                self._pick(self._BREAKDOWN_MSG),
            )

        if self._utility(counter) < rv:
            counter = self._outcome_at_utility(rv + self.acceptance_margin)
        if counter is None:
            return SAOResponse(
                ResponseType.END_NEGOTIATION,
                None,
                self._pick(self._BREAKDOWN_MSG),
            )

        self._my_offers.append(counter)
        text = self._generate_text(t, "reject", counter)

        return SAOResponse(ResponseType.REJECT_OFFER, counter, text)
