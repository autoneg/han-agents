"""HAN 2026 agent: a hard-anchoring hybrid negotiator for human-agent negotiation.

Architecture
------------
* A deterministic base negotiator (``HardlineBase``) owns every *decision*:
  what to offer and what to accept.  It anchors at our best outcome and
  concedes on a steep (Boulware-style) curve toward a floor that stays well
  above the reservation value, so we never give the deal away.
* ``CloserNegotiator`` wraps that base with the competition LLM
  (``qwen3:4b-instruct``) which generates *only* the natural-language text that
  accompanies each move.  The text is the persuasion channel against a human:
  anchor, justify, build rapport, apply gentle time pressure, and frame our
  small concessions as generous.

Decisions are never delegated to the LLM, so the agent is robust to LLM
latency / parsing failures: even if text generation fails the base still makes
rational offers and acceptances.
"""

from __future__ import annotations

from negmas.common import Outcome
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.sao import SAONegotiator, SAOState, ResponseType
from negmas_llm.meta import LLMMetaNegotiator

try:
    from negmas_llm.common import DEFAULT_MODELS

    DEFAULT_OLLAMA_MODEL = DEFAULT_MODELS["ollama"]
except Exception:  # pragma: no cover - defensive
    DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"


# =============================================================================
# Persuasion prompt (the only thing the LLM controls)
# =============================================================================

ANCHOR_PROMPT = """
You write the short chat message that accompanies a negotiation move that has
ALREADY been decided by a separate strategy engine. You never change the offer
or the accept/reject decision; you only craft persuasive words around it.

You are negotiating with a HUMAN. Your job is to make the human comfortable
accepting a deal that is excellent for us, and to keep them at the table.

Rules for every message:
1. Be warm, confident and likeable. Build a little rapport in one short phrase.
2. ANCHOR. When making an offer, present it as the natural, fair starting point
   and sound certain about it. Never apologise for asking for a lot.
3. JUSTIFY with a concrete, reasonable-sounding reason tied to the items being
   divided (e.g. "this split reflects what each of us needs most"). People say
   yes when they hear a reason, even a simple one.
4. Frame any move toward the other party as a meaningful goodwill gesture
   ("I can stretch to this for you"), never as weakness.
5. Apply gentle time pressure as the negotiation progresses ("let's lock this
   in while it's on the table") without being aggressive.
6. NEVER reveal which items matter most to us or disclose our numbers. Keep our
   true priorities private.
7. When accepting, be gracious and make the human feel they got a great deal.
8. Keep it to ONE or TWO short sentences. Sound human, not robotic.

You will be given a TACTIC line tuned to the current phase of the talk. Follow
it. The three phases are:
* OPENING - project warm confidence and plant the anchor; make our ask feel
  normal and reasonable. Do not concede or hedge.
* MIDDLE - justify and reframe. If we just moved toward them, sell it as real
  generosity and ask them to meet us. If we held, stay warm but immovable and
  give a fresh reason.
* CLOSING - create urgency and scarcity. The clock is the enemy; frame agreeing
  now as the smart move before the deal slips away.

If you are told what the human seems to care about, reassure them that need is
respected in our split WITHOUT revealing our own priorities or numbers.

Examples of the tone and quality to aim for (do NOT copy verbatim; adapt to the
actual items and what the human said):
* OPENING anchor: "Great to meet you! Here's where I think we should start - it's
  a clean, sensible split and I'd love to get this wrapped up quickly."
* MIDDLE, holding firm: "I hear you, and I really do want this to work - but this
  split is already fair given how the pieces line up. Let's shake on it."
* MIDDLE, after a move: "Okay - I just freed up something that matters to me so
  you come out ahead here. That's as far as I can stretch; meet me on it?"
* CLOSING urgency: "We're nearly out of time and this is a genuinely good deal
  for you - let's lock it in now rather than risk both of us walking away."
* ACCEPT: "Done - you drove a hard bargain and I'm happy with this. Pleasure
  working with you!"

Respond with ONLY a JSON object: {"text": "your message"}
"""

# Collaborative / fair tone. Live HAN signal (round #18999) showed that pure
# hard-anchoring caps out below the median: humans walk away from greedy agents
# and a no-deal scores only the reservation value. This tone reframes the agent
# as a fair, reciprocal partner who wants a deal both sides feel good about,
# which closes far more negotiations with real humans.
FAIR_PROMPT = """
You write the short chat message that accompanies a negotiation move that has
ALREADY been decided by a separate strategy engine. You never change the offer
or the accept/reject decision; you only craft the words around it.

You are negotiating with a HUMAN. Your job is to come across as fair, friendly
and trustworthy so the human happily says yes and you reach a deal. A deal that
is good for both of you beats no deal at all.

Rules for every message:
1. Be warm, genuine and collaborative. Treat them as a partner, not an opponent.
2. Frame every offer as a balanced, reasonable split that respects what they
   need - emphasise fairness and mutual benefit ("this works well for both of us").
3. Acknowledge their point of view; show you have listened.
4. Invoke reciprocity: when you give them something, note it warmly and invite
   them to meet you so you can close.
5. Gently encourage closing - reaching agreement is a shared win - without
   pressure or threats.
6. NEVER disclose your own priorities or numbers.
7. When accepting, be warm and appreciative.
8. Keep it to ONE or TWO short, natural sentences. Sound human, not robotic.

You will be given a TACTIC line for the phase of the talk:
* OPENING - friendly, set a fair and reasonable tone.
* MIDDLE - reinforce fairness; if we moved toward them, warmly point it out and
  invite them to meet us so we can wrap up.
* CLOSING - warmly encourage sealing the deal now so neither of you leaves empty
  handed.

If told what the human seems to care about, reassure them that need is respected
WITHOUT revealing your own priorities.

Persuasion levers to weave in naturally (use what fits; never force all at once):
* LABEL it fair: explicitly call the split "fair", "even" or "the standard
  way to split this" - people accept what is named fair.
* REASON: always attach a short "because ..." - a concrete reason sharply
  raises agreement, even a simple one.
* RECIPROCITY: when you concede anything, name it ("I moved on X for you") and
  ask them to meet you in return.
* RELATIONSHIP: be likeable and cooperative; people concede to those they like.
* GENTLE LOSS-AVERSION at the close: remind them a deal now beats both of you
  leaving with nothing - frame walking away as the real loss.
* CONSISTENCY: if they agreed earlier that something is fair, hold them to it.

Examples (do NOT copy verbatim; adapt to the actual items and their words):
* OPENING: "Hi! Here's a split I think is genuinely fair for both of us - I'd
  love to find something we're both happy with."
* MIDDLE, after a move: "I shifted this your way because I want this to feel fair
  to you - can we meet here and wrap it up?"
* CLOSING: "We're close and this is a good deal for both of us - let's lock it in
  so we both walk away happy."
* ACCEPT: "Sounds good to me - thank you, this feels fair and I'm glad we got there!"

Respond with ONLY a JSON object: {"text": "your message"}
"""

# =============================================================================
# Strategy archetypes. Live human play (not local bots) decides which wins, so
# we run different archetypes across our agent slots and converge on the winner.
# Selected by the ARCHETYPE constant below (set per packaged submission).
#   floor_frac : utility floor as a fraction of u_max (extraction vs deal-closing)
#   exp        : concession-curve exponent (lower = concede sooner / smoother)
#   accept_fair: satisficing acceptance - take any offer >= this * u_max at any
#                time (closes good-enough deals fast); None disables it
#   tone       : which persuasion prompt to use
# =============================================================================
ARCHETYPES = {
    # most accommodating; held in reserve to push deeper if "cooperative" wins
    "generous": dict(floor_frac=0.30, exp=1.0, accept_fair=0.45, tone="fair"),
    "cooperative": dict(floor_frac=0.40, exp=1.3, accept_fair=0.55, tone="fair"),
    "balanced": dict(floor_frac=0.52, exp=2.0, accept_fair=0.66, tone="fair"),
    "hardline": dict(floor_frac=0.62, exp=3.0, accept_fair=None, tone="anchor"),
    # reads the human within the negotiation: hold & extract against soft
    # opponents, concede & close against tough ones. The bet for #1, since it
    # earns both deal-rate and deal-quality from a single agent.
    "adaptive": dict(
        floor_frac=0.50, exp=2.2, accept_fair=0.70, tone="fair", adaptive=True,
        stall_window=0.20,
    ),
    # challenger: the #1 adaptive engine + a research-backed PHANTOM-ANCHOR
    # opening (frame our high opening as already a concession -> just as
    # effective as a raw anchor but seen as less manipulative, so humans reject
    # it less). Tests in a spare slot whether better opening framing extends the
    # lead, without touching the live #1 agent.
    "adaptive_phantom": dict(
        floor_frac=0.50, exp=2.2, accept_fair=0.70, tone="fair", adaptive=True,
        stall_window=0.20, phantom=True,
    ),
    # CONSISTENCY build: same adaptive engine + persuasion, but a LOWER floor so
    # we also score well against greedy/proud humans (fairness_floor>0.5). Live
    # we oscillated #1<->#9 because floor=0.50 was too greedy for the full human
    # spectrum - a few low-advantage rounds tanked the mean. On the full panel
    # (incl greedy humans) floor=0.40/accept=0.55 lifts mean advantage 16.6->19.2
    # AND raises the performance floor => less variance => consistent ranking.
    "adaptive_robust": dict(
        floor_frac=0.40, exp=2.2, accept_fair=0.55, tone="fair", adaptive=True,
        stall_window=0.20, phantom=True,
    ),
    # adaptive_robust + smart LLM-budget allocation (spend persuasion calls on
    # opening / close / human-objection moments, not blind first-N).
    "adaptive_smart": dict(
        floor_frac=0.40, exp=2.2, accept_fair=0.55, tone="fair", adaptive=True,
        stall_window=0.20, phantom=True, smart_budget=True,
    ),
    # SYNTHESIS bet: moderate-firm floor (0.47, between #5's 0.40 and firm 0.55)
    # + max smart persuasion + all validated persuasion features. Aims to extract
    # more than the floor-0.40 #5 line without the no-deal risk of 0.55.
    "adaptive_prime": dict(
        floor_frac=0.47, exp=2.4, accept_fair=0.58, tone="fair", adaptive=True,
        stall_window=0.20, phantom=True, smart_budget=True,
    ),
    # HIGH-UPSIDE bet (mirrors live #2 "LastOfferLLM"): hold a FIRM floor (0.55)
    # and use heavy smart persuasion to get humans to accept the firm offer,
    # rather than conceding to fair. If persuasion sustains the high floor we
    # extract far more per deal -> the jump toward #1. Anchor tone (assertive).
    "persuasive_firm": dict(
        floor_frac=0.55, exp=2.6, accept_fair=0.62, tone="anchor", adaptive=True,
        stall_window=0.20, phantom=True, smart_budget=True,
    ),
    # OPPONENT-FOLLOWING FLOOR: floor tracks the human's revealed generosity
    # (opp_best) between floor_frac (min, for greedy humans) and floor_cap (max,
    # for generous ones). Closes the two big leaks a fixed floor can't: holds
    # high vs easygoing humans (extract) and drops low vs greedy ones (close).
    # SIMPLICITY hypothesis: a deliberately minimal, predictable agent - linear
    # concession to a fair floor, accept anything reasonable, clean short text,
    # NO adaptive/plateau/phantom/follow machinery. Grounded in the live board:
    # the simple NoBrainNegotiator baseline (~#5) outranks our complex #12, so
    # our complexity may be misfiring on real humans. Robust + zero gimmicks.
    "simple": dict(floor_frac=0.45, exp=1.0, accept_fair=0.50, tone="fair"),
    # max deal-closing: even lower floor + quicker accept, to catch rounds heavy
    # with greedy/impatient humans (diversity across our 3 slots vs variance).
    "adaptive_dealmax": dict(
        floor_frac=0.35, exp=2.0, accept_fair=0.52, tone="fair", adaptive=True,
        stall_window=0.20, phantom=True,
    ),
    "adaptive_follow": dict(
        floor_frac=0.30, floor_cap=0.70, floor_follow=True, exp=2.2,
        accept_fair=0.55, tone="fair", adaptive=True, stall_window=0.20,
        phantom=True,
    ),
}
# Default archetype for this submission (overridden per packaged zip).
ARCHETYPE = "adaptive_smart"

# Text channel:
#   "llm"     = call the competition LLM every move (RISK: ~100-200 calls/neg
#               timed out the 30-min validation for the adaptive agent).
#   "terse"   = instant deterministic messages, ZERO LLM calls (timeout-proof
#               but throws away persuasion).
#   "bounded" = LLM for the first MAX_LLM_CALLS moves (the high-impact opening
#               + early persuasion), terse thereafter. Keeps persuasion (live
#               round #19013: the LLM balanced agent reached #4) while capping
#               total LLM time so even the adaptive agent can't time out -
#               strictly fewer calls than the LLM build that already passed.
# Default "bounded": persuasion upside, timeout-safe.
TEXT_MODE = "bounded"
MAX_LLM_CALLS = 15

# Short, fair, human-sounding deterministic lines (used in terse mode and as
# LLM fallbacks). Indexed by action; propose lines vary by phase to avoid
# sounding robotic across a long negotiation.
TERSE_TEXT = {
    "propose": [
        "Here's a split I think is fair for both of us - happy to start here.",
        "This feels like a balanced deal - works well for both sides.",
        "I think this is fair to both of us; I'd love to wrap it up here.",
        "We're close - this split is fair and I'm keen to lock it in.",
    ],
    "accept": "Sounds fair to me - glad we got there. Thank you!",
    "reject": "Not quite there for me yet - let's keep working toward a fair deal.",
    "end": "Thanks for the discussion - I hope we can make a deal another time.",
}

# Deterministic fallbacks used if the LLM is slow, errors, or returns nothing.
# The agent must never throw or stall inside a negotiation (the harness
# penalises exceptions and long response times), so text generation always
# degrades gracefully to one of these.
FALLBACK_TEXT = {
    "propose": "Here's a split I think is fair for both of us — happy to lock it in now.",
    "accept": "Great — I'm glad we found common ground. Pleasure doing business with you!",
    "reject": "I appreciate that, but I can't quite get there yet — let's keep working toward a deal.",
    "end": "Thanks for the discussion — I hope we can find a deal another time.",
}


# =============================================================================
# Deterministic decision engine
# =============================================================================


class HardlineBase(SAONegotiator):
    """Hard-anchoring, slow-conceding base with a floor above the reserved value.

    * Offers start at our maximum-utility outcome and concede along
      ``u(t) = u_max - (u_max - floor) * t**exp`` where ``floor`` is the larger
      of the reservation value and ``floor_frac * u_max``.
    * Accepts an incoming offer when its utility for us is at least the utility
      we would propose next (an ``ACNext``-style rule), so we never reject
      something better than our own upcoming counter-offer.
    * Never walks away: timing out yields the reservation value anyway, and
      staying in keeps the chance of a late concession from the human.
    """

    def __init__(
        self,
        *args,
        exp: float = 3.0,
        floor_frac: float = 0.62,
        accept_slack: float = 1e-6,
        endgame_time: float = 0.92,
        endgame_frac: float = 0.45,
        lastditch_time: float = 0.97,
        lastditch_margin: float = 1e-6,
        accept_fair: float | None = None,
        adaptive: bool = False,
        stall_window: float = 0.25,
        adapt_base: float = 0.45,
        adapt_span: float = 1.15,
        stall_mult: float = 0.55,
        floor_follow: bool = False,
        floor_cap: float = 0.70,
        open_frac: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._exp = exp
        self._floor_frac = floor_frac
        self._accept_fair = accept_fair
        self._accept_slack = accept_slack
        self._endgame_time = endgame_time
        self._endgame_frac = endgame_frac
        self._lastditch_time = lastditch_time
        self._lastditch_margin = lastditch_margin
        self._adaptive = adaptive
        self._stall_window = stall_window
        self._adapt_base = adapt_base
        self._adapt_span = adapt_span
        self._stall_mult = stall_mult
        self._floor_follow = floor_follow
        self._floor_cap = floor_cap
        self._open_frac = open_frac
        self._u_min: float = 0.0
        # opponent concession tracking (for adaptive mode): utility-to-us of
        # the opponent's first and best offers seen so far, plus the time of the
        # most recent improvement (to detect a stall = they are at their limit).
        self._opp_first: float | None = None
        self._opp_best: float | None = None
        self._opp_best_time: float = 0.0
        self._sorted: list[tuple[float, Outcome]] | None = None
        self._u_max: float = 1.0
        # Optional estimate of the opponent's utility function, injected by the
        # wrapping agent. Used to concede along the Pareto frontier: when we
        # have utility room, give the opponent the outcome they like most.
        self._opp_ufun = None

    def _ensure(self) -> None:
        if self._sorted is not None or self.ufun is None or self.nmi is None:
            return
        try:
            os_ = self.nmi.outcome_space
            outcomes = list(os_.enumerate_or_sample(max_cardinality=20000))
            u = self.ufun
            scored = sorted(((float(u(o)), o) for o in outcomes), key=lambda x: -x[0])
            self._sorted = scored
            self._u_max = scored[0][0] if scored else 1.0
            self._u_min = scored[-1][0] if scored else 0.0
        except Exception:
            # Leave self._sorted as None; callers fall back to ufun.best().
            self._sorted = None

    def _floor(self) -> float:
        rv = float(self.ufun.reserved_value) if self.ufun else 0.0
        frac = self._floor_frac
        # Opponent-following floor: aim our floor at what the human has revealed
        # they will give us (their best offer). Generous human -> hold a high
        # floor and extract; stingy/greedy human -> drop the floor and close.
        # One adaptive floor handles BOTH extremes a fixed floor cannot.
        if self._adaptive and self._floor_follow:
            span = self._u_max - self._u_min
            if self._opp_best is not None and span > 1e-9:
                obn = (self._opp_best - self._u_min) / span
                frac = min(self._floor_cap, max(self._floor_frac, obn))
            else:
                frac = max(self._floor_frac, 0.50)  # neutral until evidence
        return max(rv, frac * self._u_max)

    def _concession_ratio(self) -> float:
        """How much the opponent has conceded toward us so far, in [0, 1].

        0 = opponent has not improved on its first offer (tough / stuck);
        1 = opponent has already conceded across our whole bargaining range.
        """
        if self._opp_first is None or self._opp_best is None:
            return 0.5  # neutral until we have evidence
        span = self._u_max - self._floor()
        if span <= 1e-9:
            return 0.5
        return min(1.0, max(0.0, (self._opp_best - self._opp_first) / span))

    def _effective_exp(self, t: float) -> float:
        """Concession-curve exponent, modulated by opponent behaviour in
        adaptive mode. A conceding opponent -> larger exponent (hold firm and
        extract, they are coming to us). A stuck/tough opponent -> smaller
        exponent (concede sooner to secure a deal instead of timing out).
        If the opponent has stalled (no improvement for a while) we treat them
        as at-their-limit and accelerate to close before they walk away.
        """
        if not self._adaptive:
            return self._exp
        r = self._concession_ratio()
        # r=0 -> close fast; r=1 -> hold firm (extract). Coefficients tunable.
        e = self._exp * (self._adapt_base + self._adapt_span * r)
        if self._opp_first is not None and (t - self._opp_best_time) > self._stall_window:
            e *= self._stall_mult  # they have plateaued -> stop holding, close
        return e

    def _target(self, t: float) -> float:
        t = min(max(t, 0.0), 1.0)
        lo = self._floor()
        tgt = self._u_max - (self._u_max - lo) * (t ** self._effective_exp(t))
        # Opening calibration: cap the opening ask so we never demand the entire
        # pie (an absurd anchor triggers human ultimatum-rejection / early
        # disengagement). Never caps below the floor.
        if self._open_frac < 1.0:
            tgt = min(tgt, max(lo, self._open_frac * self._u_max))
        return tgt

    def _offer_for(self, t: float) -> Outcome | None:
        """Offer along the Pareto frontier: among outcomes that still meet our
        utility target for time t, propose the one the opponent likes most.

        Early (t->0) the target is near u_max, so the candidate set is tiny and
        we anchor near our best. As the target descends the set widens and we
        give the opponent their most-preferred outcome among options that are
        still good enough for us — maximising acceptance at no cost to us. If we
        have no opponent estimate yet we fall back to the most generous outcome
        by our own utility (the lowest-utility candidate above target).
        """
        self._ensure()
        if not self._sorted:
            # Fallback anchor: our best outcome if available.
            try:
                return self.ufun.best() if self.ufun else None
            except Exception:
                return None
        tgt = self._target(t)
        candidates = [o for u, o in self._sorted if u >= tgt - self._accept_slack]
        if not candidates:
            candidates = [self._sorted[0][1]]
        opp = self._opp_ufun
        if opp is not None and len(candidates) > 1:
            try:
                return max(candidates, key=lambda o: float(opp(o)))
            except Exception:
                pass
        # No opponent model: most generous by our own utility.
        return candidates[-1]

    def propose(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        return self._offer_for(state.relative_time)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return ResponseType.REJECT_OFFER
        try:
            self._ensure()
            their_util = float(self.ufun(offer))
            # Track opponent concession (drives adaptive concession curve).
            if self._opp_first is None:
                self._opp_first = their_util
            if self._opp_best is None or their_util > self._opp_best + 1e-9:
                self._opp_best = their_util
                self._opp_best_time = state.relative_time  # fresh improvement
            # Satisficing acceptance: take any genuinely good-enough deal right
            # away rather than holding out. Humans reward an agent that says yes
            # to a fair offer; holding out for the maximum makes them walk.
            if (
                self._accept_fair is not None
                and their_util >= self._accept_fair * self._u_max
            ):
                return ResponseType.ACCEPT_OFFER
            # Accept if at least as good as what we would offer next round.
            n_steps = self.nmi.n_steps if self.nmi else None
            dt = (1.0 / n_steps) if n_steps else 0.02
            next_target = self._target(min(1.0, state.relative_time + dt))
            if their_util >= next_target - self._accept_slack:
                return ResponseType.ACCEPT_OFFER
            # Endgame capture: near the deadline never walk away with nothing.
            # Accept any offer comfortably above the reservation value rather
            # than time out into a no-deal (which yields only the reserved value).
            if state.relative_time >= self._endgame_time:
                rv = float(self.ufun.reserved_value)
                capture_bar = rv + self._endgame_frac * (self._floor() - rv)
                if their_util >= capture_bar:
                    return ResponseType.ACCEPT_OFFER
                # Last-ditch: in the final moments, any deal strictly better than
                # the reservation value beats timing out into it. This only ever
                # fires when no acceptable deal has formed, so it cannot lower a
                # human negotiation (those close early and high) - it only lifts
                # the no-deal tail against brick-wall opponents.
                if (
                    state.relative_time >= self._lastditch_time
                    and their_util > rv + self._lastditch_margin
                ):
                    return ResponseType.ACCEPT_OFFER
        except Exception:
            pass
        return ResponseType.REJECT_OFFER


# =============================================================================
# The submitted agent
# =============================================================================


class CloserNegotiator(LLMMetaNegotiator):
    """Deterministic decision engine + LLM persuasion text. Behaviour is set by
    the module-level ARCHETYPE (cooperative / balanced / hardline), letting us
    run different strategies across our HAN agent slots and keep the one humans
    reward. The submitted agent for HAN."""

    def __init__(
        self,
        *,
        provider: str = "ollama",
        model: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 120,
        archetype: str | None = None,
        text_mode: str | None = None,
        **kwargs,
    ) -> None:
        cfg = ARCHETYPES.get(archetype or ARCHETYPE, ARCHETYPES["balanced"])
        self._tone = cfg["tone"]
        self._phantom = cfg.get("phantom", False)
        self._text_mode = text_mode or TEXT_MODE
        self._llm_calls = 0  # bounded-mode counter
        self._smart_budget = cfg.get("smart_budget", False)
        base = HardlineBase(
            exp=cfg["exp"],
            floor_frac=cfg["floor_frac"],
            accept_fair=cfg["accept_fair"],
            adaptive=cfg.get("adaptive", False),
            stall_window=cfg.get("stall_window", 0.25),
            floor_follow=cfg.get("floor_follow", False),
            floor_cap=cfg.get("floor_cap", 0.70),
        )
        # Bound LLM latency hard so a stalled call can't blow the per-negotiation
        # time budget (the harness penalises slow / timed-out moves). Normal
        # qwen3:4b calls take ~3s, so a 12s ceiling with one retry leaves ample
        # headroom while capping the worst case far below the old 50s; on any
        # failure we fall back instantly to a deterministic persuasive message.
        llm_kwargs = kwargs.pop("llm_kwargs", {})
        llm_kwargs.setdefault("timeout", 12.0)
        llm_kwargs.setdefault("num_retries", 1)
        super().__init__(
            base_negotiator=base,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=FAIR_PROMPT if self._tone == "fair" else ANCHOR_PROMPT,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )
        # value-frequency counts of the opponent's offers, per issue index
        self._opp_freq: dict[int, dict] = {}
        # the previous outcome we proposed, to detect when we move toward them
        self._last_propose: Outcome | None = None

    # ------------------------------------------------------------------
    # Opponent modelling (frequency heuristic).
    #
    # The HAN score awards an extra point for concealment that is split by
    # opponent-modelling accuracy, and an agent with *no* model forfeits it
    # entirely. A human opponent has no model, so maintaining even a rough
    # frequency estimate hands us the full extra point. Everything here is
    # guarded so a modelling error can never disrupt the negotiation.
    # ------------------------------------------------------------------
    def _observe(self, offer: Outcome | None) -> None:
        try:
            if offer is None or self.nmi is None:
                return
            for idx, value in enumerate(offer):
                bucket = self._opp_freq.setdefault(idx, {})
                bucket[value] = bucket.get(value, 0) + 1
            issues = list(self.nmi.outcome_space.issues)
            values = []
            for idx, issue in enumerate(issues):
                counts = self._opp_freq.get(idx, {})
                total = sum(counts.values()) or 1
                freq = {v: c / total for v, c in counts.items()}
                values.append((lambda v, d=freq: d.get(v, 0.0)))
            estimate = LinearAdditiveUtilityFunction(tuple(values), issues=issues)
            self.private_info["opponent_ufun"] = estimate
            # Share the estimate with the base so it can concede toward the
            # opponent along the Pareto frontier.
            self.base_negotiator._opp_ufun = estimate
        except Exception:
            pass

    def propose(self, state: SAOState, dest: str | None = None):
        self._observe(state.current_offer)
        result = super().propose(state, dest=dest)
        # Record the bare outcome we proposed so the next message can tell
        # whether we moved toward the human (goodwill) or held firm.
        try:
            out = getattr(result, "outcome", result)
            if out is not None:
                self._last_propose = out
        except Exception:
            pass
        return result

    def respond(self, state: SAOState, source: str | None = None):
        self._observe(state.current_offer)
        return super().respond(state, source=source)

    # ------------------------------------------------------------------
    # Persuasion helpers (text only; never affect decisions).
    # ------------------------------------------------------------------
    def _phase_tactic(self, state: SAOState, outcome: Outcome | None) -> str:
        t = state.relative_time
        moved = self._moved_toward_human(outcome)
        if t < 0.2 and self._phantom:
            # phantom anchor: frame the opening as already a concession to them
            return (
                "TACTIC (OPENING, phantom anchor): Warmly frame this as already a "
                "concession in their favour - e.g. 'I was hoping for a bit more, "
                "but here's a split I think is fair to you.' Makes the ask feel "
                "generous, not greedy, so they're far more likely to engage."
            )
        if self._tone == "fair":
            if t < 0.2:
                return (
                    "TACTIC (OPENING): Be friendly and set a fair, reasonable tone; "
                    "present this as a balanced split good for both of you."
                )
            if t < 0.85:
                if moved:
                    return (
                        "TACTIC (MIDDLE): We just moved toward them. Warmly point "
                        "out that you shifted to make it fair and invite them to "
                        "meet you so you can wrap up."
                    )
                return (
                    "TACTIC (MIDDLE): Reinforce that this split is genuinely fair "
                    "for both sides and that you'd love to find common ground."
                )
            return (
                "TACTIC (CLOSING): You're close. Use a reciprocal commitment ask - "
                "'if I make this work, can we shake on it now?' - and a warm, "
                "gracious tone so they feel good about saying yes (reduces backout)."
            )
        # anchor tone
        if t < 0.2:
            return (
                "TACTIC (OPENING): Project warm confidence and plant the anchor. "
                "Make this ask sound completely normal and fair. Do not hedge."
            )
        if t < 0.85:
            if moved:
                return (
                    "TACTIC (MIDDLE): We just moved toward them. Sell this as real, "
                    "costly generosity and warmly ask them to meet us halfway."
                )
            return (
                "TACTIC (MIDDLE): Hold firm but stay warm. Give a fresh concrete "
                "reason this split is fair; do not signal any willingness to move."
            )
        return (
            "TACTIC (CLOSING): Time is almost up. Create urgency and scarcity - "
            "frame agreeing right now as the smart move before the deal slips away."
        )

    def _moved_toward_human(self, outcome: Outcome | None) -> bool:
        try:
            opp = self.base_negotiator._opp_ufun
            if opp is None or outcome is None or self._last_propose is None:
                return False
            return float(opp(outcome)) > float(opp(self._last_propose)) + 1e-9
        except Exception:
            return False

    def _opp_priority_hint(self) -> str | None:
        """A short, human-readable note on what the human seems to want, from
        the values they have repeatedly requested in their own offers."""
        try:
            if not self._opp_freq or self.nmi is None:
                return None
            issues = list(self.nmi.outcome_space.issues)
            hints = []
            for idx, issue in enumerate(issues):
                counts = self._opp_freq.get(idx, {})
                if not counts:
                    continue
                total = sum(counts.values()) or 1
                value, n = max(counts.items(), key=lambda kv: kv[1])
                if n / total >= 0.6:  # they consistently push one value here
                    hints.append(str(issue.name))
            if not hints:
                return None
            return ", ".join(hints[:2])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Richer context for the persuasion LLM.
    #
    # By default the LLM only sees a raw outcome tuple (e.g. "(4, 4, 4, 0)"),
    # which produces vague text. We map each value to its named issue so the
    # model can write concrete, persuasive lines that reference the actual
    # items ("you keep the bananas, I take the oranges").
    # ------------------------------------------------------------------
    def _named(self, outcome: Outcome | None) -> str:
        try:
            if outcome is None or self.nmi is None:
                return "none"
            issues = list(self.nmi.outcome_space.issues)
            return ", ".join(
                f"{issue.name}={outcome[i]}" for i, issue in enumerate(issues)
            )
        except Exception:
            return str(outcome)

    def _generate_text(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Text generation that can never throw, stall, or return empty.

        In "terse" mode we skip the LLM entirely and return an instant
        deterministic line - this removes the per-move LLM latency that blew the
        validation timeout live. In "llm" mode we call the model and fall back to
        the same deterministic lines on any failure.
        """
        if self._text_mode == "terse":
            return self._terse_text(state, action)
        # bounded mode: spend the LLM on the first MAX_LLM_CALLS moves (opening
        # anchor + early persuasion), terse afterwards - BUT always reserve a
        # call for the close ("accept"/"end"), since a gracious, reciprocal
        # closing message is what seals the deal and reduces human backout. The
        # close happens at most once, so this adds <=1 call (still timeout-safe).
        if (
            self._text_mode == "bounded"
            and self._llm_calls >= MAX_LLM_CALLS
            and action not in ("accept", "end")
        ):
            return self._terse_text(state, action)
        # Smart budget: spend scarce LLM calls only where persuasion lands -
        # the opening, the close, or a move where the human just spoke
        # (objected/countered, so a tailored reply matters). Filler mid-moves
        # with no human input go terse, saving the budget for moments that
        # count. Same call cap => same timeout safety, better-aimed persuasion.
        if self._smart_budget and self._text_mode == "bounded":
            t = getattr(state, "relative_time", 0.0)
            human_spoke = bool(received_text and received_text.strip())
            high_value = (
                action in ("accept", "end") or t < 0.15 or t > 0.85 or human_spoke
            )
            if not high_value:
                return self._terse_text(state, action)
        try:
            text = super()._generate_text(state, action, outcome, received_text)
            if text and text.strip():
                self._llm_calls += 1
                return text.strip()
        except Exception:
            pass
        return self._terse_text(state, action)

    def _terse_text(self, state: SAOState, action: str) -> str:
        entry = TERSE_TEXT.get(action, TERSE_TEXT["propose"])
        if isinstance(entry, list):
            # vary the propose line by phase so it doesn't read robotically
            t = getattr(state, "relative_time", 0.0)
            # phantom-anchor opening line (terse): frame opening as a concession
            if action == "propose" and self._phantom and t < 0.25:
                msg = ("I was aiming a bit higher, but here's a split I think is "
                       "fair to you - happy to start here.")
                return msg
            idx = min(len(entry) - 1, int(t * len(entry)))
            msg = entry[idx]
            # v10: make it concrete/human-aware at zero latency by referencing
            # what the opponent model says the human cares about. Reassures them
            # their priorities are respected without revealing ours.
            try:
                hint = self._opp_priority_hint()
                if hint:
                    msg += f" I've kept what matters to you ({hint}) in mind."
            except Exception:
                pass
            return msg
        return entry

    def _build_user_message(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        parts = [
            f"Negotiation progress: {state.relative_time:.0%} of the time has "
            f"elapsed (round {state.step})."
        ]
        if received_text:
            parts.append(f'The human just said: "{received_text}"')
            parts.append(
                "Acknowledge their point warmly in a few words before steering "
                "toward our position."
            )
        their = self._named(state.current_offer)
        if action == "propose":
            parts.append(f"Our offer (already decided): {self._named(outcome)}.")
            parts.append(self._phase_tactic(state, outcome))
            hint = self._opp_priority_hint()
            if hint:
                parts.append(
                    f"The human seems to care about: {hint}. Reassure them that "
                    "need is respected, without revealing our own priorities."
                )
            parts.append(
                "Write a confident, friendly message presenting this as a fair, "
                "natural split. Reference the actual items. Give one concrete "
                "reason it makes sense, and nudge them to agree now."
            )
            parts.append(
                "Sell it from THEIR side: point out a specific item in THIS offer "
                "where they come out well, so it feels generous - but only claim "
                "what the offer above actually gives them (they can see the "
                "numbers, so never overstate)."
            )
        elif action == "accept":
            parts.append(f"We are ACCEPTING the human's offer: {their}.")
            parts.append(
                "Write a warm, gracious one-liner that makes them feel they got "
                "a great deal."
            )
        elif action == "reject":
            parts.append(f"The human's current offer is: {their}.")
            parts.append(
                "Write a warm but firm message that gently declines and keeps "
                "them engaged, without conceding."
            )
        elif action == "end":
            parts.append("Write a brief, friendly closing message.")
        parts.append('Reply with ONLY: {"text": "your one or two sentence message"}')
        return "\n".join(parts)
