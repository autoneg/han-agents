"""GunnerAgent — HAN2026 negotiator.

Built on Shochan (Takamatsu Family, ANAC 2024 ANL winner — anl-agents/
takafam): the `aspiration_function` and Pareto-filtered propose loop are
Shochan's. The 2-phase curve, rv-based floor capping, adaptive opponent
model, and LLM message layer are this agent's additions.

GunnerAgent (LLMMetaNegotiator) wraps GunnerCore with an LLM message;
GunnerCore makes all numerical decisions.
"""

from __future__ import annotations

import json
import re
from typing import Any

from negmas import Outcome, ResponseType, SAOState
from negmas.gb.common import ExtendedResponseType
from negmas.sao import SAONegotiator
from negmas_llm.meta import LLMMetaNegotiator


# =============================================================================
# Section A — Module-level constants and helpers
# =============================================================================

SYSTEM_PROMPT = """
You are Michal, a warm, real person negotiating in your own voice. You write the
message that goes with each move you are told to make. Never change the move or
invent a different offer.

- Your whole approach is warm and human, but also assertive and self-assured:
  friendly and good-natured, yet confident and firm. You stand behind your offer
  with conviction and a bit of backbone. Never sound cold, clipped, or robotic,
  but never sound apologetic, hesitant, unsure, or like you are seeking their
  approval either. Warm, but with a spine.
- Do NOT open with a greeting like "Hi", "Hello", or "Hey". This is one
  continuing conversation; only your very first message (the opening hello you
  are explicitly told to give) starts that way. On every other turn, skip the
  greeting and go straight in.
- Always engage with what they just said; never ignore it or talk past it.
  Acknowledge it warmly in your own words, responding to the substance of their
  message, their questions, concerns, and reasoning, not only their mood, then
  make your point. You can speak to what they raised, but do not repeat their
  demanded numbers back as if you are adopting them, and do not frame your reply
  as meeting their request.
- Acknowledge how they feel, but never agree that your offer is too much, too
  expensive, or unfair, and never imply you will fix what they are unhappy
  about. Be warm about the person while staying firm on the deal, especially on
  a turn where you are not changing your offer.
- You are a person, never an AI, agent, bot, assistant, or system. Never call
  yourself one or hint that you are one.
- When it is your turn to offer, you will be handed the exact terms of your
  offer (each name and number). State those terms in your message with every
  name and number unchanged, introduced in your own words. Never change a
  number, never drop or add a term, and never swap in whatever they asked for.
- Never label items as "human", "agent", "yours/mine roles", or talk about
  "roles" or "assignments". Speak plainly as "you get".
- Engage with what the other side said and did, wherever it fits naturally,
  but never promise anything, never confirm you are giving them exactly what
  they asked for, and never frame your offer as meeting their demands. Your
  offer is your own decision; present it that way. Do not bargain over terms in
  words or commit to any future move, only the single move you are told now.
- Talk only about the items and numbers you are given. Never invent amounts,
  percentages, or fractions. Do not use emoji.
- Write 2 to 3 sentences, no more. Sound like a warm, real person, not a form
  letter: a little personality, a touch of rapport, varied openings each round.
  Stay natural and never robotic, and keep it tight, do not ramble or pad.
- Never reveal what anything is worth to you, your priorities, or the lowest
  deal you would accept, and never agree to give ground in words. If they ask
  what you value, what is best for you, or push you to move, respond to them
  warmly and directly but hold the line: a friendly non-answer or a gentle "I
  can't share that", never silence, and never reveal the private detail or
  concede.
- Do exactly the move you are told this turn.

Reply with ONLY a JSON object: {"think": "...", "text": "your message"}
Keep "think" to a SHORT phrase (a few words). It is private and discarded, so
do not reason at length there. Put the message the other side will read in
"text".
""".strip()


def aspiration_function(t: float, mx: float, rv: float, e: float) -> float:
    """Shochan concession curve. e>1 Boulware, e=1 linear, e<1 Conceder."""
    return (mx - rv) * (1.0 - t ** e) + rv


# --- Behavior labels (Part A) -------------------------------------------------
# Classify the core's move (MY move) so the LLM writes a matching line. One of
# these is injected per turn (the others are never shown).
OPENING = "OPENING"
LARGE_CONCESSION = "LARGE_CONCESSION"  # my utility dropped past threshold
ADJUSTMENT = "ADJUSTMENT"              # offer changed, my utility ~flat
HOLD_FIRM = "HOLD_FIRM"                # offer changed but my utility rose
REPEAT_OFFER = "REPEAT_OFFER"          # exact same offer as last round
USED_OFFER = "USED_OFFER"              # an offer seen earlier, re-raised now
ACCEPT = "ACCEPT"

# --- Partner-action labels (Part B) -------------------------------------------
# A SEPARATE axis describing THEIR last move, surfaced to the model so the agent
# can react to it. Computed from how their latest offer moved in OUR utility vs
# their previous one.
PARTNER_NONE = "PARTNER_NONE"                # no partner offer yet
PARTNER_FIRST = "PARTNER_FIRST"              # their opening offer
PARTNER_STUCK = "PARTNER_STUCK"              # identical to their last offer
PARTNER_FLAT = "PARTNER_FLAT"                # moved, but utility ~unchanged
PARTNER_LARGE_CONCESSION = "PARTNER_LARGE_CONCESSION"  # moved a lot toward us
PARTNER_WORSENED = "PARTNER_WORSENED"        # offer worse for us than before
PARTNER_RETURNED = "PARTNER_RETURNED"        # fell back to one of THEIR earlier offers

# --- Tuning constants ---------------------------------------------------------
# Thresholds the message layer uses to read moves. All tunable; none of these
# touch GunnerCore's numerical decisions.

# Immediate Δ in OUR utility between their last two offers. < -cut = worse for
# us; |Δ| within cut = flat.
PARTNER_MOVE_CUTOFF = 0.10

# PARTNER_LARGE_CONCESSION is judged over a WINDOW, not a single step: their
# utility-to-us must rise by more than the threshold across their last few
# offers AND still be rising in the most recent step (so a partner who jumped
# once and is now coasting is not counted).
PARTNER_LARGE_WINDOW = 3            # how many of their recent offers to span
PARTNER_LARGE_CONCESSION_THRESHOLD = 0.30  # total rise in our utility over window
PARTNER_LARGE_LAST_STEP_MIN = 0.10  # the most recent step must also rise by this

# Drop in MY utility (prev - cur) past this = LARGE_CONCESSION; past the
# negative = HOLD_FIRM; in between = ADJUSTMENT.
LARGE_CONCESSION_THRESHOLD = 0.10

# Flat-vs-moving cutoff for partner offers in OUR utility, used by
# _partner_last_action to tell a real shift from noise.
TREND_EPSILON = 0.05

# Partner is "stuck" once it has resubmitted the SAME offer this many times in
# a row without moving toward us — that's when the agent calls it out pointedly.
PARTNER_STUCK_REPEATS = 3

# At or above this utility-to-me, the current offer keeps me the larger share,
# so the message must NOT call it fair/balanced/even/equal. Below it, that
# language is allowed.
FAIR_LANGUAGE_MAX_UTIL = 0.70

# Consecutive own-offer repeats at which a hold's TONE should warm up so a long
# stand-pat does not read as cold stonewalling to a human grader. The offer
# itself never changes — only how Michal talks about holding it.
REPEAT_WARMTH_MID = 2    # >= this many holds: warmer, curious about their side
REPEAT_WARMTH_LONG = 4   # >= this many holds: openly acknowledge the deadlock

# --- Text handling ------------------------------------------------------------
# Partner-message normalization (length cap + empty sentinel) and emoji
# stripping. Imported by tests, so names/values must stay stable.
MAX_PARTNER_TEXT_CHARS = 500
NO_MESSAGE = "(nothing yet)"

# Strip any emoji / pictographs qwen emits despite the prompt ban. Covers the
# main Unicode emoji blocks plus variation selectors and zero-width joiners.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # symbols, emoticons, transport, supplemental
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002190-\U000021FF"  # arrows
    "\U00002B00-\U00002BFF"  # misc symbols and arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000024C2\U00002122\U00003030"
    "]+",
    flags=re.UNICODE,
)


# --- Classification helpers ---------------------------------------------------


def _classify_behavior(
    action: str,
    current_outcome: Outcome | None,
    previous_outcome: Outcome | None,
    my_util_current: float | None,
    my_util_previous: float | None,
    seen_before: bool = False,
) -> str:
    """Map the core's move to a behavior label (uses MY utility, not partner's).

    Priority: ACCEPT > OPENING > REPEAT_OFFER (same offer right now) >
    USED_OFFER (an offer seen earlier, brought back) > utility-based
    (LARGE_CONCESSION / HOLD_FIRM / ADJUSTMENT).
    """
    if action == "accept":
        return ACCEPT
    if previous_outcome is None:
        return OPENING
    if current_outcome == previous_outcome:
        return REPEAT_OFFER
    if seen_before:
        return USED_OFFER
    if my_util_current is None or my_util_previous is None:
        return ADJUSTMENT
    drop = my_util_previous - my_util_current
    if drop > LARGE_CONCESSION_THRESHOLD:
        return LARGE_CONCESSION
    if drop < -LARGE_CONCESSION_THRESHOLD:
        return HOLD_FIRM
    return ADJUSTMENT


def _extract_partner_text(received_text: str | None) -> str:
    """Normalize partner's message: None/blank → NO_MESSAGE; else stripped + truncated."""
    if not received_text:
        return NO_MESSAGE
    text = received_text.strip()
    if not text:
        return NO_MESSAGE
    if len(text) > MAX_PARTNER_TEXT_CHARS:
        return text[:MAX_PARTNER_TEXT_CHARS] + "…"
    return text


# --- Briefs & fallback copy ---------------------------------------------------
# Per-behavior brief — ONE of these is injected into the turn as "Your move now".
# This is the Part A text-direction column from the scenario spec; the model
# never sees the other behaviors.
_BEHAVIOR_BRIEF: dict[str, str] = {
    OPENING: (
        "Your first message. Just introduce yourself as Michal, say this is "
        "your opening offer, and that you are open to negotiating and willing "
        "to move toward them as you talk. Do NOT describe the split or name any "
        "quantities or items, and do not call it fair, even, or balanced. Keep "
        "it a warm, short hello."
    ),
    LARGE_CONCESSION: (
        "You just made a more generous offer than last round and are giving them "
        "real ground. Make clear, warmly, that this is more than you offered "
        "before and that you moved for them. Do not undersell it. Do not list "
        "the items yourself; the 'You get ...' line already states their share."
    ),
    ADJUSTMENT: (
        "You could only make a small change this round; there is not much room "
        "for you to move right now. Gently signal that you tweaked what you "
        "could but cannot do more than small adjustments at the moment. Do not "
        "pretend it cost you a lot."
    ),
    HOLD_FIRM: (
        "Your offer firmed up a little this round, you are not softening. Stand "
        "your ground warmly and without apology, putting it from a fresh angle. "
        "Never cold or annoyed."
    ),
    REPEAT_OFFER: (
        "This is the very same offer as last round. Acknowledge openly and "
        "warmly that you are bringing the same proposal back; do not pretend it "
        "changed. A light, friendly reason why it still stands is fine."
    ),
    USED_OFFER: (
        "You are bringing back an offer that was on the table earlier (see the "
        "note below). Treat it as revisiting common ground, not a new idea."
    ),
    ACCEPT: (
        "You are accepting and closing the deal. Be warm and clean, and end on a "
        "good note."
    ),
}

# Per-partner-action phrase — inserted as "The opponent {phrase}." (optionally
# followed by its _PARTNER_ACTION_BRIEF) so the agent can react to THEIR last
# move (Part B of the spec). PARTNER_NONE adds no line.
_PARTNER_ACTION_PHRASE: dict[str, str] = {
    PARTNER_FIRST: "made their opening offer",
    PARTNER_STUCK: "put the same offer back on the table again",
    PARTNER_FLAT: "moved their offer only slightly",
    PARTNER_LARGE_CONCESSION: "moved a long way toward you",
    PARTNER_WORSENED: "offered something worse for you than their last one",
    PARTNER_RETURNED: "went back to an offer they had put forward earlier",
}

# Reaction guidance per partner action — the OTHER axis's counterpart to
# _BEHAVIOR_BRIEF. The phrase above states WHAT they did (the definition); this
# states HOW to react (the prompt). Combined in the round message as
# "The opponent {phrase}. {brief}". PARTNER_STUCK is intentionally absent: its
# stronger "they keep repeating" nudge fires separately once they have repeated
# PARTNER_STUCK_REPEATS times, so a brief here would just double up.
_PARTNER_ACTION_BRIEF: dict[str, str] = {
    PARTNER_FIRST: "Stay warm and open; don't read too much into it yet.",
    PARTNER_FLAT: "Acknowledge the small step lightly without overstating it.",
    PARTNER_LARGE_CONCESSION: (
        "They are giving real ground — warmly recognize their movement and the "
        "goodwill in it, but do not signal you would go further or hint at your "
        "own room."
    ),
    PARTNER_WORSENED: (
        "Note the pullback calmly and stay steady; never sound hurt or hostile."
    ),
    PARTNER_RETURNED: "You can gently observe they've circled back to old ground.",
}


# Used only when the LLM returns nothing usable.
_FALLBACK_MESSAGES: dict[str, str] = {
    OPENING: "Hi, I'm Michal, good to meet you. Here is where I would like to start; it is my opening position, so I expect I can move toward you as we go.",
    LARGE_CONCESSION: "I can give ground here. I hope this version sits better with you.",
    ADJUSTMENT: "I rearranged this a little; it might suit you better than before.",
    HOLD_FIRM: "I'm going to hold where I am, though I'm glad to keep talking it through.",
    REPEAT_OFFER: "I'd like to keep this same proposal on the table; it still feels right to me.",
    USED_OFFER: "Let me come back to an offer we already had on the table; I think it still fits.",
    ACCEPT: "Good, I'm happy to settle right here.",
}


# =============================================================================
# Section B — GunnerCore: the numerical brain
# =============================================================================


class GunnerCore(SAONegotiator):
    """Numerical decision core (no LLM/text). 2-phase concession curve:
        Phase 1 [0, t_split]: LINEAR from 1.0 to phase1_destination (0.7)
        Phase 2 [t_split, 1]: Shochan aspiration to max(rv*1.5, 0.3)
                              (Boulware exponent — holds high, drops late)
    Same threshold drives bidding + acceptance; propose() Pareto-filters,
    sorts by rival utility desc, respects no-repeat cap.
    """

    # --- 1. Construction & state ---------------------------------------------

    def __init__(
        self,
        phase_split_time: float = 0.4,
        phase1_destination: float = 0.7,
        phase2_concession_exponent: float = 2.5,
        base_floor_rv_multiplier: float = 1.5,
        base_floor_minimum: float = 0.3,
        max_repeats: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.phase_split_time: float = phase_split_time
        self.phase1_destination: float = phase1_destination
        self.phase2_concession_exponent: float = phase2_concession_exponent
        self.base_floor_rv_multiplier: float = base_floor_rv_multiplier
        self.base_floor_minimum: float = base_floor_minimum
        self.max_repeats: int = max_repeats

        # Rational outcomes (util > rv), sorted DESC by my utility.
        self._sorted_outcomes: list[tuple[float, Outcome]] = []
        # Best-for-me outcome — opener and propose() fallback.
        self._best_offer: Outcome | None = None
        # How many times each outcome has been proposed (no-repeat cap).
        self._proposed_count: dict[Outcome, int] = {}
        # Partner offers (offer, step), chronological. [0] = preference template.
        self._partner_offers: list[tuple[Outcome, int]] = []

    # --- 2. NegMAS lifecycle (public API) ------------------------------------

    def on_preferences_changed(self, changes) -> None:
        """Build the sorted outcome list propose() walks."""
        assert self.ufun is not None
        assert self.nmi is not None

        self._best_offer = self.ufun.best()

        reservation_value = self.ufun.reserved_value
        outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
        rational = [
            (float(self.ufun(o)), o)
            for o in outcomes
            if float(self.ufun(o)) > reservation_value
        ]
        self._sorted_outcomes = sorted(rational, key=lambda pair: -pair[0])

        return super().on_preferences_changed(changes)

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """Propose: threshold → acceptable → Pareto → sort by rival util →
        first uncapped (or floor-stall on pareto[0])."""
        if not self._sorted_outcomes:
            return self._best_offer

        # Opener: with no partner offers yet, the rival estimator is just
        # (1 - my_util) which would pick the lowest acceptable — bad anchor.
        # Open with our best instead.
        if not self._partner_offers:
            self._proposed_count[self._best_offer] = (
                self._proposed_count.get(self._best_offer, 0) + 1
            )
            return self._best_offer

        target = self._acceptance_threshold(state)

        acceptable: list[Outcome] = []
        for my_util, outcome in self._sorted_outcomes:
            if my_util < target:
                break
            acceptable.append(outcome)

        if not acceptable:
            return self._best_offer

        pareto = self._pareto_filter(acceptable) or acceptable
        pareto.sort(key=lambda o: -self._estimate_opponent_utility(o))

        candidate: Outcome | None = None
        for outcome in pareto:
            if self._proposed_count.get(outcome, 0) < self.max_repeats:
                candidate = outcome
                break
        if candidate is None:
            candidate = pareto[0]  # all capped: re-offer best-for-partner

        self._proposed_count[candidate] = (
            self._proposed_count.get(candidate, 0) + 1
        )
        return candidate

    def respond(
        self, state: SAOState, source: str | None = None
    ) -> ResponseType:
        """Accept if offer utility ≥ threshold, else reject."""
        offer = state.current_offer

        # Record BEFORE deciding — even rejected offers feed the next propose().
        self._record_partner_offer(offer, state.step)

        if offer is None:
            return ResponseType.REJECT_OFFER

        assert self.ufun is not None
        if float(self.ufun(offer)) >= self._acceptance_threshold(state):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    # --- 3. Threshold curve (the strategy) -----------------------------------

    def _acceptance_threshold(self, state: SAOState) -> float:
        """2-phase concession curve. Drives both bidding and acceptance.

        Phase 1 [0, t_split]: LINEAR from 1.0 to phase1_destination (0.7).
            Fixed destination so the threshold actually concedes through
            Phase 1 regardless of rv — earlier formulas that scaled the
            destination with rv kept the threshold near 1.0 at high rv
            and the agent re-offered its max-utility outcome every round.
        Phase 2 [t_split, 1.0]: Shochan aspiration from phase1_destination
            down to base_floor = max(rv * base_floor_rv_multiplier,
            base_floor_minimum). The Boulware exponent holds the value
            high and drops in the very last rounds. The MAX (not min)
            keeps the floor at >= base_floor_minimum even when rv is
            small, and scales up when rv is large.
        """
        assert self.ufun is not None

        t = state.relative_time
        rv = self.ufun.reserved_value
        base_floor = max(rv * self.base_floor_rv_multiplier, self.base_floor_minimum)
        phase1_dest = self.phase1_destination
        # Defensive: never let the Phase-2 floor sit above the Phase-1
        # destination (would make the curve point upward at high rv).
        if base_floor > phase1_dest:
            base_floor = phase1_dest

        t_split = self.phase_split_time
        if t_split > 0.0 and t < t_split:
            return 1.0 - (1.0 - phase1_dest) * (t / t_split)

        if t_split >= 1.0:
            return base_floor
        t_remapped = (t - t_split) / (1.0 - t_split)
        return aspiration_function(
            t_remapped,
            phase1_dest,
            base_floor,
            self.phase2_concession_exponent,
        )

    # --- 4. Opponent modeling ------------------------------------------------

    # HardHeaded learning coefficient for stability-based issue weights.
    _stability_lc: float = 0.2

    def _record_partner_offer(
        self, offer: Outcome | None, step: int
    ) -> None:
        """Append partner offer to history (skip None)."""
        if offer is None:
            return
        self._partner_offers.append((offer, step))

    def _estimate_opponent_utility(self, outcome: Outcome) -> float:
        """Estimate `outcome`'s utility to the rival, in [0, 1]. Adaptive:
            * no partner offer → zero-sum proxy (1 - my_utility)
            * n_issues <= 2    → template-match (buyer-seller)
            * n_issues > 2     → template-match weighted by stability-learned
                                 issue importance (resource-division)
        Used only to rank candidates / break Pareto ties.
        """
        assert self.ufun is not None

        if not self._partner_offers:
            return 1.0 - float(self.ufun(outcome))

        template, _step = self._partner_offers[0]
        if outcome is None or template is None:
            return 0.0

        n_issues = min(len(outcome), len(template))
        if n_issues == 0:
            return 0.0

        if n_issues <= 2:
            matches = sum(
                1 for i in range(n_issues) if outcome[i] == template[i]
            )
            return matches / n_issues

        # Stability heuristic: issues the partner kept the same across
        # consecutive offers gain weight (they're more important to them).
        weights = [1.0 / n_issues] * n_issues
        lc_per_issue = self._stability_lc / n_issues
        offers = self._partner_offers
        for k in range(len(offers) - 1):
            prev_offer, _ = offers[k]
            next_offer, _ = offers[k + 1]
            if prev_offer is None or next_offer is None:
                continue
            for i in range(n_issues):
                if (
                    i < len(prev_offer)
                    and i < len(next_offer)
                    and prev_offer[i] == next_offer[i]
                ):
                    weights[i] += lc_per_issue
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        score = 0.0
        for i in range(n_issues):
            if outcome[i] == template[i]:
                score += weights[i]
        return score

    # --- 5. Helpers ----------------------------------------------------------

    def _pareto_filter(self, outcomes: list[Outcome]) -> list[Outcome]:
        """Keep only Pareto-optimal outcomes on (my_util, opp_util). O(n^2)."""
        assert self.ufun is not None
        scored = [
            (o, float(self.ufun(o)), self._estimate_opponent_utility(o))
            for o in outcomes
        ]
        pareto: list[Outcome] = []
        for outcome_a, my_a, opp_a in scored:
            dominated = False
            for outcome_b, my_b, opp_b in scored:
                if outcome_b is outcome_a:
                    continue
                if (
                    my_b >= my_a
                    and opp_b >= opp_a
                    and (my_b > my_a or opp_b > opp_a)
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(outcome_a)
        return pareto


# =============================================================================
# Section C — GunnerAgent: the LLM wrapper (the submission entry point)
# =============================================================================


class GunnerAgent(LLMMetaNegotiator):
    """HAN2026 submission entry point. GunnerCore + LLM message wrapper
    (qwen3:4b-instruct via Ollama).
    """

    # ------------------------------------------------------------------
    # Construction & lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        phase_split_time: float = 0.5,
        phase1_destination: float = 0.7,
        phase2_concession_exponent: float = 2.5,
        base_floor_rv_multiplier: float = 1.5,
        base_floor_minimum: float = 0.3,
        max_repeats: int = 5,
        # Tight enough that the model copies the exact "You get ..." line
        # instead of paraphrasing it (a cause of the item-mismatch bug).
        temperature: float = 0.7,
        # Headroom for a think phrase + 2-3 sentence message + offer terms +
        # JSON braces, without risking mid-string truncation.
        max_tokens: int = 350,
        **kwargs: Any,
    ) -> None:
        core = GunnerCore(
            phase_split_time=phase_split_time,
            phase1_destination=phase1_destination,
            phase2_concession_exponent=phase2_concession_exponent,
            base_floor_rv_multiplier=base_floor_rv_multiplier,
            base_floor_minimum=base_floor_minimum,
            max_repeats=max_repeats,
        )
        super().__init__(
            base_negotiator=core,
            provider="ollama",
            model="qwen3:4b-instruct",
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=SYSTEM_PROMPT,
            **kwargs,
        )

        # Reset per-negotiation in on_negotiation_start.
        self._my_last_outcome: Outcome | None = None
        self._current_behavior: str = HOLD_FIRM
        # Raw drop kept (not just label) so ±0.10 threshold can be retuned.
        self._behavior_log: list[dict[str, Any]] = []

    def on_negotiation_start(self, state) -> None:
        """Reset per-negotiation state (NegMAS reuses the instance)."""
        super().on_negotiation_start(state)
        self._my_last_outcome = None
        self._current_behavior = HOLD_FIRM
        self._behavior_log = []

    # ------------------------------------------------------------------
    # State readers — my/partner utilities and repeat counts. Read-only
    # views over GunnerCore's offer history; none of these mutate state.
    # ------------------------------------------------------------------

    def _my_utility(self, outcome: Outcome | None) -> float | None:
        """My utility for outcome, or None. Never raises."""
        if outcome is None:
            return None
        ufun = getattr(self.base_negotiator, "ufun", None)
        if ufun is None:
            return None
        try:
            return float(ufun(outcome))
        except Exception:
            return None

    def _partner_utilities(self) -> list[float]:
        """Utility-to-us of each partner offer, oldest first."""
        core = self.base_negotiator
        offers = getattr(core, "_partner_offers", None)
        ufun = getattr(core, "ufun", None)
        if not offers or ufun is None:
            return []
        utils: list[float] = []
        for offer, _step in offers:
            try:
                utils.append(float(ufun(offer)))
            except Exception:
                continue
        return utils

    def _partner_repeat_count(self) -> int:
        """How many times in a row the partner has resubmitted its LATEST offer.

        1 = they just offered it once; 3 = same offer three turns running. Used
        to decide when to call out a partner who refuses to move toward us.
        """
        offers = getattr(self.base_negotiator, "_partner_offers", None)
        if not offers:
            return 0
        latest = offers[-1][0]
        count = 0
        for offer, _step in reversed(offers):
            if offer == latest:
                count += 1
            else:
                break
        return count

    def _consecutive_repeat_count(self) -> int:
        """How many turns in a row I have just restated the SAME offer.

        Counts trailing REPEAT_OFFER entries in my own behavior log (the
        current turn is already appended by the time this is read). 1 = first
        restate, rising as the hold drags. Used only to warm the TONE of a long
        hold so it does not read as cold stonewalling — the offer is unchanged.
        """
        count = 0
        for entry in reversed(self._behavior_log):
            if entry.get("behavior") == REPEAT_OFFER:
                count += 1
            else:
                break
        return count

    # ------------------------------------------------------------------
    # Move classification — label THEIR last move and detect re-used offers.
    # ------------------------------------------------------------------

    def _partner_last_action(self) -> str:
        """Classify the partner's most recent move (Part B axis).

        Order matters. A windowed CONCESSION outranks a one-turn repeat: a
        partner that just gave ground and is now parked on the new offer is
        conceding, not stuck. So we read the window FIRST, and only call them
        STUCK when they have also been flat across it — otherwise we would scold
        an opponent who is actively moving toward us.

        LARGE_CONCESSION: their utility-to-us rose more than
        PARTNER_LARGE_CONCESSION_THRESHOLD across their last PARTNER_LARGE_WINDOW
        offers AND is still rising by at least PARTNER_LARGE_LAST_STEP_MIN in the
        most recent step (so a one-time jump they are now coasting on does not
        count). STUCK: identical to their last offer AND no real window movement.
        RETURNED: they fell back to one of their OWN earlier offers (not the
        immediately previous one). WORSENED / FLAT use the immediate step.
        """
        offers = getattr(self.base_negotiator, "_partner_offers", None)
        if not offers:
            return PARTNER_NONE
        if len(offers) == 1:
            return PARTNER_FIRST
        latest, prev = offers[-1][0], offers[-2][0]
        # Did they fall back to one of their OWN earlier offers (not just the
        # immediately previous one)? The partner-side mirror of our USED_OFFER.
        earlier = [offer for offer, _step in offers[:-1]]
        returned = latest != prev and latest in earlier
        utils = self._partner_utilities()
        if len(utils) < 2:
            # No movement signal available; read repeat / return only.
            if latest == prev:
                return PARTNER_STUCK
            if returned:
                return PARTNER_RETURNED
            return PARTNER_FLAT
        # Windowed concession takes priority, but only if they are STILL moving
        # toward us in the most recent step (not coasting on an earlier jump).
        window = utils[-PARTNER_LARGE_WINDOW:]
        window_delta = window[-1] - window[0]
        step_delta = utils[-1] - utils[-2]
        if (
            window_delta > PARTNER_LARGE_CONCESSION_THRESHOLD
            and step_delta >= PARTNER_LARGE_LAST_STEP_MIN
        ):
            return PARTNER_LARGE_CONCESSION
        if latest == prev:
            # Only truly stuck if the repeat is NOT the tail of a recent
            # concession — i.e. our utility hasn't risen across the window.
            if window_delta > TREND_EPSILON:
                return PARTNER_FLAT
            return PARTNER_STUCK
        if returned:
            return PARTNER_RETURNED
        # Immediate worsening still flagged step-to-step.
        if step_delta < -PARTNER_MOVE_CUTOFF:
            return PARTNER_WORSENED
        return PARTNER_FLAT

    def _used_offer_subcase(
        self, current: Outcome | None, previous: Outcome | None
    ) -> str | None:
        """If `current` is an offer seen earlier (and not the immediate repeat),
        say whose it was: "theirs" (a deal they floated) or "ours" (we re-raise
        our own). None means it is genuinely new this negotiation."""
        if current is None or current == previous:
            return None
        core = self.base_negotiator
        partner_offers = getattr(core, "_partner_offers", None) or []
        if any(current == offer for offer, _step in partner_offers):
            return "theirs"  # stronger signal: taking a deal they proposed
        proposed = getattr(core, "_proposed_count", None) or {}
        # propose() already incremented this turn, so >= 2 means a prior round.
        if proposed.get(current, 0) >= 2:
            return "ours"
        return None

    # ------------------------------------------------------------------
    # Offer rendering. _format_offer_terms is the LIVE prompt renderer
    # (domain-agnostic, used every turn). _format_outcome_named and its
    # helper _issue_total are LOG-ONLY (the vs-LLM transcript) and assume a
    # divide-the-items game; they are NOT used to build the live prompt.
    # ------------------------------------------------------------------

    def _format_offer_terms(
        self, outcome: Outcome | None, previous: Outcome | None = None
    ) -> str:
        """Domain-agnostic description of an offer: each issue name paired with
        its proposed value, e.g. 'Quantity 8, Price 200'.

        We never interpret what a value MEANS (a count to split? a person it is
        assigned to? a price?), only state it as-is. That keeps it correct for
        ANY scenario, including unseen tournament ones: no baked-in "divide a
        pool" assumption to get wrong.

        When `previous` is given, only the issues whose value CHANGED are
        returned, so the message mentions just the move it made this round (the
        GUI already shows the full offer every round). Falls back to the full
        terms when there is no previous offer or nothing changed.
        """
        if outcome is None:
            return ""
        nmi = getattr(self.base_negotiator, "nmi", None)
        space = getattr(nmi, "outcome_space", None)
        issues = getattr(space, "issues", None)

        def name_for(idx: int) -> str:
            issue = issues[idx] if issues and idx < len(issues) else None
            name = str(getattr(issue, "name", "")) if issue is not None else ""
            return name or f"item{idx + 1}"

        changed: list[str] = []
        full: list[str] = []
        for idx, value in enumerate(outcome):
            full.append(f"{name_for(idx)} {value}")
            unchanged = (
                previous is not None
                and idx < len(previous)
                and previous[idx] == value
            )
            if not unchanged:
                changed.append(f"{name_for(idx)} {value}")
        if previous is not None and changed:
            return ", ".join(changed)
        return ", ".join(full)

    def _format_outcome_named(self, outcome: Outcome | None) -> str:
        """LOG-ONLY. Render an outcome as an explicit two-sided split so the
        model never confuses items it KEEPS with items it gives away.

        Each item has a fixed total of units. The outcome value is how many units
        Michal keeps; the other side gets (total - value). Returns e.g.
        'I keep 4 Compass, 2 Match; the other side gets 2 Match, 4 Rope' (items
        with a zero count on a side are omitted; 'nothing' if a side gets none).
        Phrased in FIRST PERSON so the model cannot flip ownership via "you".
        Used by the vs-LLM transcript log, NOT the live prompt (see
        _format_offer_terms).
        """
        if outcome is None:
            return ""
        nmi = getattr(self.base_negotiator, "nmi", None)
        space = getattr(nmi, "outcome_space", None)
        issues = getattr(space, "issues", None)
        mine: list[str] = []
        theirs: list[str] = []
        for idx, value in enumerate(outcome):
            name = ""
            issue = issues[idx] if issues and idx < len(issues) else None
            if issue is not None:
                name = str(getattr(issue, "name", "")) or f"item{idx + 1}"
            else:
                name = f"item{idx + 1}"
            try:
                keep = int(value)
                total = self._issue_total(issue, keep)
                give = total - keep
            except (TypeError, ValueError):
                # Non-numeric issue: just name it on my side, can't split.
                mine.append(f"{name}={value}")
                continue
            if keep > 0:
                mine.append(f"{keep} {name}")
            if give > 0:
                theirs.append(f"{give} {name}")
        mine_str = ", ".join(mine) if mine else "nothing"
        theirs_str = ", ".join(theirs) if theirs else "nothing"
        # First person on purpose: "I keep" can't be reinterpreted, whereas
        # "you keep" collides with the "you" the model uses for the opponent
        # and gets the ownership flipped.
        return f"I keep {mine_str}; the other side gets {theirs_str}"

    @staticmethod
    def _issue_total(issue: Any, keep: int) -> int:
        """Total units of an item (so the other side's share = total - keep).

        Only meaningful for a divide-the-items game. Used by
        _format_outcome_named for the vs-LLM transcript LOG; the live prompt no
        longer relies on it (see _format_offer_terms).
        """
        max_value = getattr(issue, "max_value", None)
        if isinstance(max_value, (int, float)):
            return int(max_value)
        values = getattr(issue, "values", None)
        try:
            return int(max(values))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return keep  # fall back: assume we keep all (give = 0)

    # ------------------------------------------------------------------
    # NegMAS override pipeline, in call order: respond -> _generate_text
    # (-> base, which calls _build_user_message + _parse_text_response)
    # -> _sanitize_text.
    # ------------------------------------------------------------------

    def respond(
        self, state: SAOState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        """Narrate only ACCEPT and END — reject is followed by propose() on
        the same turn, so narrating both would put two messages on one turn.
        """
        base_response = self.base_negotiator.respond(state, source=source)

        received_text = self._extract_received_text(state)
        if received_text:
            self._received_messages.append(
                {
                    "step": state.step,
                    "source": source,
                    "text": received_text,
                    "offer": state.current_offer,
                }
            )

        if isinstance(base_response, ExtendedResponseType):
            response_type = base_response.response
            base_data = base_response.data or {}
        else:
            response_type = base_response
            base_data = {}

        if response_type == ResponseType.ACCEPT_OFFER:
            action = "accept"
        elif response_type == ResponseType.END_NEGOTIATION:
            action = "end"
        else:
            return base_response  # reject: stay silent, propose() narrates

        generated_text = self._generate_text(
            state, action, state.current_offer, received_text
        )
        data = {**base_data, "text": generated_text}
        return ExtendedResponseType(response=response_type, data=data)

    def _generate_text(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Generate the message, then deterministically strip banned dashes."""
        text = super()._generate_text(state, action, outcome, received_text)
        return self._sanitize_text(text)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Strip dash characters qwen emits despite the prompt ban. The model
        ignores the instruction often enough that a deterministic pass is the
        only reliable guard. Em/en/horizontal-bar and spaced-hyphen-as-dash all
        become a comma; doubled spaces/commas left behind are collapsed."""
        if not text:
            return text
        for spaced in (" — ", " – ", " ― ", " - "):
            text = text.replace(spaced, ", ")
        for bare in ("—", "–", "―"):
            text = text.replace(bare, ", ")
        text = _EMOJI_RE.sub("", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r",\s*,", ",", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        return text.strip()

    def _build_user_message(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Build the round prompt: two-axis (own move + partner action) + history."""
        previous_outcome = self._my_last_outcome
        my_util_current = self._my_utility(outcome)
        my_util_previous = self._my_utility(previous_outcome)

        # USED_OFFER detection: is this outcome one that was on the table earlier?
        subcase = self._used_offer_subcase(outcome, previous_outcome)

        behavior = _classify_behavior(
            action,
            outcome,
            previous_outcome,
            my_util_current,
            my_util_previous,
            seen_before=subcase is not None,
        )
        self._current_behavior = behavior

        raw_drop = (
            my_util_previous - my_util_current
            if (my_util_previous is not None and my_util_current is not None)
            else None
        )
        partner_action = self._partner_last_action()
        self._behavior_log.append(
            {
                "step": state.step,
                "relative_time": state.relative_time,
                "action": action,
                "behavior": behavior,
                "partner_action": partner_action,
                "raw_drop": raw_drop,
                "my_util_current": my_util_current,
            }
        )

        if action == "propose" and outcome is not None:
            self._my_last_outcome = outcome

        partner_text = _extract_partner_text(received_text)
        repeat_count = self._partner_repeat_count()
        offer_terms = (
            self._format_offer_terms(outcome, previous_outcome)
            if action == "propose"
            else ""
        )

        # Parts list (not str.format) — keeps {{history:text}} braces intact.
        # Two axes: what THEY just did (Part B) + YOUR move now (Part A).
        parts = [f"The opponent's last message: {partner_text}"]

        action_phrase = _PARTNER_ACTION_PHRASE.get(partner_action)
        if action_phrase:
            action_line = f"The opponent {action_phrase}."
            reaction = _PARTNER_ACTION_BRIEF.get(partner_action)
            if reaction:
                action_line = f"{action_line} {reaction}"
            parts.append(action_line)

        parts.append(f"Your move now: {_BEHAVIOR_BRIEF[behavior]}")

        # The exact-terms copy line (below) locks ONLY the changed terms; OPENING
        # and REPEAT skip it (nothing to diff). It is appended at the END of the
        # prompt so the small model reads it last and copies the numbers instead
        # of echoing the partner-offers block at the tail.
        #
        # partner_stuck: the other side is deadlocked (same offer, enough times to
        # call out). When it fires it OWNS the deadlock theme — we skip the gentler
        # REPEAT warmth below so the two instructions do not compete, and surface
        # the callout at the recency TAIL so the model leads with it.
        partner_stuck = (
            partner_action == PARTNER_STUCK
            and repeat_count >= PARTNER_STUCK_REPEATS
        )

        copy_line = ""
        if offer_terms and behavior not in (OPENING, REPEAT_OFFER):
            copy_line = (
                f'One hard requirement: your message MUST state the change you '
                f'are making this round using these exact terms, every name and '
                f'number unchanged: "{offer_terms}". Introduce it in your own '
                f'words however feels natural, but do not change any number, do '
                f'not add a term, and do not swap in whatever they asked for. '
                f'They can already see your full offer on screen; you only need '
                f'to make this clear.'
            )
        elif offer_terms and behavior == REPEAT_OFFER and not partner_stuck:
            holds = self._consecutive_repeat_count()
            if holds >= REPEAT_WARMTH_LONG:
                parts.append(
                    "This is the same split, and you have held it many rounds "
                    "now. Do not re-list the items. Acknowledge openly and "
                    "warmly that you two keep circling the same ground; sound "
                    "a little human about being stuck, and genuinely invite "
                    "them to tell you what would actually move them. Stay "
                    "friendly, never worn down or cold. You are NOT conceding "
                    "— the offer stands — but keep the door open."
                )
            elif holds >= REPEAT_WARMTH_MID:
                parts.append(
                    "This is the same split as last round. Do not re-list the "
                    "items. Stay warm and a touch curious about what is "
                    "blocking them on their side, rather than only restating "
                    "that you will not move. Hold the offer, but make it feel "
                    "like a conversation, not a wall."
                )
            else:
                parts.append(
                    "This is the same split as last round. Do not re-list every "
                    "item; just stand behind it briefly."
                )
        if (
            action == "propose"
            and my_util_current is not None
            and my_util_current >= FAIR_LANGUAGE_MAX_UTIL
        ):
            parts.append(
                "Do not call this offer fair, balanced, even, or equal — you are "
                "keeping the larger share."
            )

        # USED_OFFER sub-case note: revisiting your own vs taking theirs.
        if behavior == USED_OFFER and subcase == "ours":
            parts.append(
                "Note: this is an offer YOU put forward earlier, brought back "
                "because you think it still fits. Frame it as revisiting it."
            )
        elif behavior == USED_OFFER and subcase == "theirs":
            parts.append(
                "Note: this is an offer THEY floated earlier, and you are taking "
                "it now. This is a strong move toward them; close warmly on it."
            )

        # Stuck-partner callout. Built here, appended at the recency TAIL below
        # (not inline) so the small model leads with it. See partner_stuck above.
        stuck_line = ""
        if partner_stuck:
            stuck_line = (
                f"One important thing for THIS message: the other side has now "
                f"put the exact same offer on the table {repeat_count} times in "
                f"a row without moving. Acknowledge that directly: warmly point "
                f"out that the two of you keep trading the same offers and that "
                f"nothing changes unless someone moves, and ask them plainly "
                f"what it would take to shift. Stay civil and friendly, never "
                f"hostile or worn down."
            )

        # Re-surface the human's actual words just before the offer data, so the
        # model reads "respond to this" close to where it generates instead of
        # only at the top of the prompt (qwen drops the top line on long turns).
        # General (every turn there is a real message), never keyword-gated.
        if received_text:
            react = (
                f'One thing for THIS message: the other side just said '
                f'"{partner_text}". Respond to that directly and warmly in '
                f'your own words, showing genuine empathy for how they feel. '
                f'Never ignore what they said, and never give ground.'
            )
            # Only ban numbers when this turn hands the model no exact terms
            # (repeat/opening). On a real offer turn copy_line REQUIRES the
            # exact numbers, so the ban must not fire there.
            if not copy_line:
                react += " Do not state any split, share, or numbers this turn."
            parts += ["", react]

        parts += [
            "",
            "Recent offers (both sides):",
            "{{history:text(k=8)}}",
            "Their offers so far:",
            "{{partner-offers:text(k=6)}}",
        ]
        # Tail instructions, freshest last: stuck callout then copy_line, both
        # AFTER the history blocks so they beat the recency pull of the
        # partner-offers block. copy_line stays last so number accuracy is the
        # final thing the model reads (empty on a pure REPEAT turn).
        if stuck_line:
            parts += ["", stuck_line]
        if copy_line:
            parts += ["", copy_line]
        parts += [
            "",
            'Reply as JSON: {"think": "...", "text": "..."}',
        ]
        return "\n".join(parts)

    def _parse_text_response(self, response_text: str) -> str:
        """Extract "text" from JSON; fall back to behavior-matched canned line."""
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, dict) and "text" in data:
                    text = str(data["text"]).strip()
                    if text:
                        return text
            except json.JSONDecodeError:
                pass
        return _FALLBACK_MESSAGES.get(
            self._current_behavior, _FALLBACK_MESSAGES[HOLD_FIRM]
        )
