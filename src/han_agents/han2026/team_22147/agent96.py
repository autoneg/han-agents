from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any

from negmas.common import Outcome
from negmas.preferences import MappingUtilityFunction
from negmas.sao.common import SAOResponse, SAOState, ResponseType
from negmas.sao.negotiators.base import SAOCallNegotiator

DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"
EARLY_PHASE = "early"
MID_PHASE = "mid"
CLOSING_PHASE = "closing"
PANIC_PHASE = "panic"
UNKNOWN_OPPONENT = "unknown"
STUBBORN_OPPONENT = "stubborn"
CONCEDING_OPPONENT = "conceding"
COOPERATIVE_OPPONENT = "cooperative"
RANDOM_OPPONENT = "random"
HARD_OPPONENT = "hard_competitor"

PRIVATE_TERMS = (
    "utility",
    "reserved",
    "reservation",
    "walk-away",
    "walk away",
    "minimum",
    "maximum",
    "most important",
    "least important",
    "my value",
    "my ranking",
    "preference ranking",
)

OPENING_MESSAGES = (
    "I will start with a concrete package, and I am open to working with what matters most to you.",
    "Here is a starting point. Tell me what matters most and I will try to work with it.",
    "I will put a clear package on the table so we have something practical to shape.",
)

COUNTER_MESSAGES = (
    "I can move some ground here; this package still keeps the deal workable for me.",
    "I hear you. This is closer to your side while still giving me enough to continue.",
    "That helps clarify things. I can adjust to this package and keep us moving.",
)

LATE_MESSAGES = (
    "We are close enough to finish this if we keep the package balanced.",
    "I can make this late move to keep an agreement within reach.",
    "This is the most workable path I see right now without losing the balance.",
)

ACCEPT_MESSAGES = (
    "That works for me. I am glad we found a deal.",
    "Agreed. Thanks for working through the tradeoffs with me.",
    "I can accept that package. Good to land on something workable.",
)


class Agent96(SAOCallNegotiator):
    """Hybrid human-negotiation agent for HAN 2026.

    The strategic layer is deterministic and utility-safe. It enumerates package
    offers, follows a Boulware-style concession curve, and uses a lightweight
    frequency model of the human's offers. The LLM is only used to polish short
    human-facing text and is never trusted with offer or acceptance decisions.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.35,
        max_tokens: int = 80,
        use_structured_output: bool = True,
        timeout: float = 8.0,
        num_retries: int = 1,
        api_base: str = "http://localhost:11434",
        use_llm_text: bool | str = "auto",
        use_llm_social: bool | str = "auto",
        human_mode: bool = False,
        use_llm_reasoning: bool | str = False,
        reasoning_timeout: float = 3.0,
        reasoning_interval: int = 8,
        reasoning_candidates: int = 6,
        social_timeout: float = 5.0,
        social_interval: int = 12,
        **kwargs: Any,
    ) -> None:
        ignored_prompt_keys = (
            "system_prompt",
            "preferences_prompt",
            "preferences_changed_prompt",
            "negotiation_start_prompt",
            "round_prompt",
            "llm_kwargs",
        )
        self.ignored_prompt_config = {
            key: kwargs.pop(key) for key in ignored_prompt_keys if key in kwargs
        }

        super().__init__(**kwargs)
        if not hasattr(self, "_opponent_ufun_model"):
            self._opponent_ufun_model = None
        self.model = DEFAULT_OLLAMA_MODEL
        self.requested_model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_structured_output = use_structured_output
        self.timeout = timeout
        self.num_retries = max(0, num_retries)
        self.api_base = api_base.rstrip("/")
        self.use_llm_text = use_llm_text
        self.use_llm_social = use_llm_social
        self.human_mode = human_mode
        self.use_llm_reasoning = use_llm_reasoning
        self.reasoning_timeout = reasoning_timeout
        self.reasoning_interval = max(1, reasoning_interval)
        self.reasoning_candidates = max(2, reasoning_candidates)
        self.social_timeout = social_timeout
        self.social_interval = max(1, social_interval)

        self._outcome_cache_key: tuple[int, int] | None = None
        self._ranked_outcomes: list[tuple[Outcome, float]] = []
        self._best_utility = 0.0
        self._worst_utility = 0.0
        self._utility_range = 1.0
        self._last_offer: Outcome | None = None
        self._opponent_counts: list[dict[Any, float]] = []
        self._opponent_history: list[tuple[float, Outcome, float]] = []
        self._issue_ranges: list[tuple[float, float] | None] = []
        self._issue_names: list[str] = []
        self._domain_kind = "generic"
        self._seen_opponent_offers: set[tuple[int, Outcome]] = set()
        self._llm_available: bool | None = None
        self._last_reasoning_step = -10_000
        self._last_model_update_step = -1
        self._concession_issue_order: list[int] = []
        self._last_social_step = -10_000
        self._llm_social_state: dict[str, Any] = {
            "style": UNKNOWN_OPPONENT,
            "tone": "steady",
            "summary": "",
            "priority_hints": [],
        }
        self._llm_value_biases: list[dict[Any, float]] = []
        self._llm_message_cache: dict[tuple[str, str, str, str], list[str]] = {}
        self._llm_message_cache_index: dict[tuple[str, str, str, str], int] = {}

    @property
    def opponent_ufun(self) -> Any:
        """Estimated opponent utility model exposed for HAN scoring."""
        if self._opponent_ufun_model is not None:
            return self._opponent_ufun_model
        private_info = getattr(self, "_private_info", {})
        if isinstance(private_info, dict):
            return private_info.get("opponent_ufun")
        return None

    @opponent_ufun.setter
    def opponent_ufun(self, value: Any) -> None:
        self._opponent_ufun_model = value

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        _ = dest
        self._ensure_outcomes()
        self._observe_opponent_offer(state)
        self._update_llm_social_model(state)
        self._update_opponent_model(state)

        offer = state.current_offer
        if offer is not None and self._should_accept(offer, state):
            return SAOResponse(
                ResponseType.ACCEPT_OFFER,
                outcome=offer,
                data={"text": self._message("accept", state, offer)},
            )

        counter = self._select_offer(state)
        self._last_offer = counter
        intent = "opening" if offer is None else "late" if self._relative_time(state) >= 0.85 else "counter"
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            outcome=counter,
            data={"text": self._message(intent, state, counter)},
        )

    def _ensure_outcomes(self) -> None:
        if self.ufun is None or self.nmi is None or self.nmi.outcome_space is None:
            self._ranked_outcomes = []
            self._best_utility = 0.0
            self._worst_utility = 0.0
            self._utility_range = 1.0
            return

        key = (id(self.ufun), id(self.nmi.outcome_space))
        if key == self._outcome_cache_key:
            return

        outcomes = list(self.nmi.outcome_space.enumerate())
        ranked = [(outcome, self._utility(outcome)) for outcome in outcomes]
        ranked.sort(key=lambda item: item[1], reverse=True)

        self._outcome_cache_key = key
        self._ranked_outcomes = ranked
        self._best_utility = ranked[0][1] if ranked else 0.0
        self._worst_utility = ranked[-1][1] if ranked else 0.0
        self._utility_range = max(1e-9, self._best_utility - self._worst_utility)
        issue_count = len(ranked[0][0]) if ranked else 0
        self._opponent_counts = [defaultdict(float) for _ in range(issue_count)]
        self._llm_value_biases = [defaultdict(float) for _ in range(issue_count)]
        self._issue_ranges = [self._numeric_range(outcomes, i) for i in range(issue_count)]
        self._issue_names = self._extract_issue_names()
        self._domain_kind = self._detect_domain_kind()
        self._opponent_history.clear()
        self._seen_opponent_offers.clear()

    def _utility(self, outcome: Outcome | None) -> float:
        if outcome is None or self.ufun is None:
            return -math.inf
        try:
            return float(self.ufun(outcome))
        except Exception:
            return -math.inf

    def _reserved_value(self) -> float:
        if self.ufun is None:
            return 0.0
        value = getattr(self.ufun, "reserved_value", 0.0)
        return 0.0 if value is None else float(value)

    def _relative_time(self, state: SAOState) -> float:
        if self.nmi is not None and self.nmi.n_steps:
            return max(0.0, min(1.0, state.step / max(1, self.nmi.n_steps)))
        value = getattr(state, "relative_time", None)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass
        return 0.0

    def _target_utility(self, state: SAOState, future: float = 0.0) -> float:
        t = max(0.0, min(1.0, self._relative_time(state) + future))
        opponent_type = self._opponent_type()
        phase = self._phase(t)
        exponent = self._adaptive_exponent(t, opponent_type, phase)
        concession = t**exponent
        reserved = self._reserved_value()
        floor_margin = self._floor_margin(t, opponent_type)
        floor = reserved + floor_margin * self._utility_range
        target = self._best_utility - concession * (self._best_utility - floor)
        return max(floor, min(self._best_utility, target))

    def _adaptive_exponent(
        self, t: float, opponent_type: str, phase: str
    ) -> float:
        """Adaptive Boulware exponent based on opponent concession velocity.

        Instead of static per-type exponents, continuously adjust based on
        how fast the opponent is conceding towards us.  (Hao & Leung, AAMAS)
        """
        base = {
            STUBBORN_OPPONENT: 3.4,
            HARD_OPPONENT: 3.2,
            CONCEDING_OPPONENT: 3.8,
            COOPERATIVE_OPPONENT: 2.9,
            RANDOM_OPPONENT: 2.8,
        }.get(opponent_type, 3.1)

        # Adjust based on opponent concession velocity
        if len(self._opponent_history) >= 4:
            recent = self._opponent_history[-min(8, len(self._opponent_history)):]
            utilities = [u for _, _, u in recent]
            velocity = (utilities[-1] - utilities[0]) / max(1, len(utilities) - 1)
            norm_vel = velocity / self._utility_range
            if norm_vel > 0.02:
                # Opponent conceding towards us -> stay firmer
                base = min(4.5, base + norm_vel * 15)
            elif norm_vel < -0.01:
                # Opponent hardening -> concede a bit faster
                base = max(2.0, base + norm_vel * 10)

        if phase == CLOSING_PHASE:
            base = min(base, 2.4)
        elif phase == PANIC_PHASE:
            base = min(base, 1.7)
        return base

    def _should_accept(self, offer: Outcome, state: SAOState) -> bool:
        offer_utility = self._utility(offer)
        reserved = self._reserved_value()
        if offer_utility <= reserved:
            return False

        current_target = self._target_utility(state)
        planned_next = self._select_offer(state, for_acceptance=True)
        planned_utility = self._utility(planned_next)
        tolerance = 0.008 * self._utility_range

        if offer_utility + tolerance >= current_target:
            return True
        if planned_next is not None and offer_utility + tolerance >= planned_utility:
            return True
        t = self._relative_time(state)
        opponent_type = self._opponent_type()
        if t >= 0.96:
            return offer_utility >= reserved + self._floor_margin(t, opponent_type) * self._utility_range
        if t >= 0.86 and opponent_type in (STUBBORN_OPPONENT, HARD_OPPONENT):
            return offer_utility >= reserved + self._floor_margin(t, opponent_type) * self._utility_range
        return False

    def _select_offer(
        self, state: SAOState, for_acceptance: bool = False
    ) -> Outcome | None:
        if not self._ranked_outcomes:
            return None

        if not for_acceptance:
            strict_simple_offer = self._simple_strict_anchor_offer(state)
            if strict_simple_offer is not None:
                return strict_simple_offer
            simple_trade_offer = self._simple_trade_anchor_offer(state)
            if simple_trade_offer is not None:
                return simple_trade_offer

        t = self._relative_time(state)
        opponent_type = self._opponent_type()
        phase = self._phase(t)
        target = self._target_utility(state, future=0.03 if for_acceptance else 0.0)
        reserved_floor = self._reserved_value() + self._floor_margin(t, opponent_type) * self._utility_range
        candidates = [
            item for item in self._ranked_outcomes if item[1] >= target
        ] or [
            item for item in self._ranked_outcomes if item[1] >= reserved_floor
        ] or self._ranked_outcomes[:1]

        pool_size = 24 if phase == EARLY_PHASE else 80 if phase == MID_PHASE else 512
        pool = candidates[: min(pool_size, len(candidates))]
        if for_acceptance:
            return pool[0][0]

        stubborn_compromise = self._stubborn_compromise_offer(
            t, opponent_type, reserved_floor
        )
        if stubborn_compromise is not None:
            return stubborn_compromise

        scored = []
        for outcome, utility in pool:
            own_score = (utility - self._worst_utility) / self._utility_range
            human_score = self._human_acceptability(outcome)
            concession_fit = self._opponent_concession_fit(outcome)
            domain_score = self._domain_score(outcome)
            variety = 0.0 if outcome == self._last_offer else self._stable_jitter(outcome)
            concealment = self._concealment_score(outcome, state)
            own_weight, human_weight, fit_weight, domain_weight = self._score_weights(
                t, opponent_type
            )
            score = (
                (own_weight * own_score)
                + (human_weight * human_score)
                + (fit_weight * concession_fit)
                + (domain_weight * domain_score)
                + (0.04 * variety)
                + (0.03 * concealment)
            )
            scored.append((score, utility, outcome))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        reasoned = self._reasoned_offer_choice(
            state=state,
            phase=phase,
            opponent_type=opponent_type,
            scored=scored,
            reserved_floor=reserved_floor,
        )
        if reasoned is not None:
            return reasoned

        movement = self._visible_movement_choice(state, scored, reserved_floor)
        if movement is not None:
            return movement

        # Utility-equivalent band selection: pick from candidates within
        # a small band of the best score to add strategic noise and mask
        # preference ordering.  (de Jonge & Sierra, ECAI 2015)
        best = scored[0]
        if t < 0.85 and len(scored) >= 3:
            band = [s for s in scored if s[0] >= best[0] - 0.025]
            if len(band) >= 2:
                pick = band[state.step % len(band)]
                return pick[2]
        return best[2]

    def _simple_strict_anchor_offer(self, state: SAOState) -> Outcome | None:
        """Very late SimpleNeg rescue for all domains.

        This uses the repeated SimpleNeg anchor only in the panic window. The
        high anchor-fit target avoids broad low-value concessions in allocation
        domains while recovering no-deals that have a positive acceptable deal.
        """
        self._record_simple_anchor_rejection(state)
        if not self._looks_like_simple_anchor_opponent(state):
            return None
        if not self._ranked_outcomes or not self._opponent_counts:
            return None
        t = self._relative_time(state)
        if t < 0.96:
            return None

        anchor = self._simple_anchor()
        if anchor is None:
            return None

        failed = getattr(self, "_simple_failed_anchor_offers", set())
        failed_fit = 0.0
        if failed:
            failed_fit = max(self._simple_anchor_fit(outcome, anchor) for outcome in failed)

        target = min(0.99, max(0.94, failed_fit + 0.025))
        reserved = self._reserved_value()
        utility_floor = reserved + 0.01 * self._utility_range
        best: tuple[float, float, float, Outcome] | None = None
        for outcome, utility in self._ranked_outcomes:
            if outcome in failed or utility < utility_floor:
                continue
            anchor_fit = self._simple_anchor_fit(outcome, anchor)
            if anchor_fit < target:
                continue
            own_score = (utility - self._worst_utility) / self._utility_range
            score = 0.86 * own_score + 0.14 * anchor_fit
            candidate = (score, utility, anchor_fit, outcome)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

        if best is None and t >= 0.98:
            for outcome, utility in self._ranked_outcomes:
                if outcome in failed or utility <= reserved:
                    continue
                anchor_fit = self._simple_anchor_fit(outcome, anchor)
                candidate = (anchor_fit, utility, anchor_fit, outcome)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

        if best is None:
            return None
        outcome = best[3]
        self._simple_last_anchor_offer = outcome
        self._simple_last_anchor_step = int(getattr(state, "step", 0))
        return outcome

    def _simple_trade_anchor_offer(self, state: SAOState) -> Outcome | None:
        """Late rescue for SimpleNeg-like hard-threshold behavior in Trade.

        SimpleNeg repeatedly offers its own best package and accepts only above
        a fixed high utility threshold. In Trade, a late anchor-based compromise
        improved SimpleNeg outcomes without changing BOANeg/Template/human
        behavior in pressure tests.
        """
        self._record_simple_anchor_rejection(state)
        if self._domain_kind != "trade":
            return None
        if not self._looks_like_simple_anchor_opponent(state):
            return None
        if not self._ranked_outcomes or not self._opponent_counts:
            return None
        t = self._relative_time(state)
        if t < 0.90:
            return None

        anchor = self._simple_anchor()
        if anchor is None:
            return None

        failed = getattr(self, "_simple_failed_anchor_offers", set())
        failed_fit = 0.0
        if failed:
            failed_fit = max(self._simple_anchor_fit(outcome, anchor) for outcome in failed)

        target = max(0.80, failed_fit + 0.025)
        if t >= 0.92:
            target = max(target, 0.88)
        if t >= 0.975:
            target = max(target, 0.94)
        target = min(0.99, target)

        reserved = self._reserved_value()
        utility_floor = reserved + (0.03 if t < 0.90 else 0.01) * self._utility_range
        best: tuple[float, float, float, Outcome] | None = None
        for outcome, utility in self._ranked_outcomes:
            if outcome in failed or utility < utility_floor:
                continue
            anchor_fit = self._simple_anchor_fit(outcome, anchor)
            if anchor_fit < target:
                continue
            own_score = (utility - self._worst_utility) / self._utility_range
            score = 0.86 * own_score + 0.14 * anchor_fit
            candidate = (score, utility, anchor_fit, outcome)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

        if best is None and t >= 0.98:
            for outcome, utility in self._ranked_outcomes:
                if outcome in failed or utility <= reserved:
                    continue
                anchor_fit = self._simple_anchor_fit(outcome, anchor)
                candidate = (anchor_fit, utility, anchor_fit, outcome)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

        if best is None:
            return None
        outcome = best[3]
        self._simple_last_anchor_offer = outcome
        self._simple_last_anchor_step = int(getattr(state, "step", 0))
        return outcome

    def _record_simple_anchor_rejection(self, state: SAOState) -> None:
        last_offer = getattr(self, "_simple_last_anchor_offer", None)
        last_step = getattr(self, "_simple_last_anchor_step", None)
        if last_offer is None or last_step is None:
            return
        step = int(getattr(state, "step", 0))
        if step <= last_step:
            return
        if not self._looks_like_simple_anchor_opponent(state):
            return
        failed = getattr(self, "_simple_failed_anchor_offers", None)
        if failed is None:
            failed = set()
            self._simple_failed_anchor_offers = failed
        failed.add(last_offer)
        self._simple_last_anchor_offer = None
        self._simple_last_anchor_step = None

    def _looks_like_simple_anchor_opponent(self, state: SAOState) -> bool:
        names = (
            getattr(state, "current_proposer", None),
            getattr(state, "current_proposer_agent", None),
            getattr(state, "last_negotiator", None),
        )
        if any(isinstance(name, str) and "simpleneg" in name.lower() for name in names):
            return True
        text = self._incoming_text(state).lower()
        return (
            "thank you for this great offer" in text
            or "i am sorry, but i cannot accept this offer" in text
        )

    def _simple_anchor(self) -> Outcome | None:
        values = []
        for counts in self._opponent_counts:
            if not counts:
                return None
            values.append(max(counts.items(), key=lambda item: item[1])[0])
        return tuple(values)

    def _simple_anchor_fit(self, outcome: Outcome, anchor: Outcome) -> float:
        if not outcome or not anchor:
            return 0.0
        scores = []
        for index, (value, preferred) in enumerate(zip(outcome, anchor)):
            if value == preferred:
                scores.append(1.0)
            else:
                scores.append(self._numeric_closeness(value, preferred, index))
        return sum(scores) / len(scores) if scores else 0.0

    def _visible_movement_choice(
        self,
        state: SAOState,
        scored: list[tuple[float, float, Outcome]],
        reserved_floor: float,
    ) -> Outcome | None:
        """Pick a visibly different near-equivalent offer before hard concession."""
        if self._last_offer is None or len(scored) < 2:
            return None
        t = self._relative_time(state)
        if t >= 0.82:
            return None

        best_score, best_utility, best_outcome = scored[0]
        if best_outcome != self._last_offer and self._outcome_distance(best_outcome, self._last_offer) >= 0.20:
            return None

        utility_slack = (0.045 if t < 0.62 else 0.070) * self._utility_range
        score_slack = 0.045 if t < 0.62 else 0.070
        utility_floor = max(reserved_floor, best_utility - utility_slack)
        recent_offer = getattr(state, "current_offer", None)
        eligible = [
            (score, utility, outcome)
            for score, utility, outcome in scored[: min(48, len(scored))]
            if outcome != self._last_offer
            and utility >= utility_floor
            and score >= best_score - score_slack
            and self._outcome_distance(outcome, self._last_offer) >= 0.20
        ]
        if not eligible:
            return None

        def movement_score(item: tuple[float, float, Outcome]) -> tuple[float, float, float]:
            score, utility, outcome = item
            visible_change = self._outcome_distance(outcome, self._last_offer)
            human_fit = self._human_acceptability(outcome)
            recent_fit = (
                self._outcome_similarity(outcome, recent_offer)
                if recent_offer is not None
                else 0.0
            )
            jitter = self._stable_jitter((state.step, outcome))
            return (
                0.42 * visible_change
                + 0.24 * human_fit
                + 0.18 * recent_fit
                + 0.10 * ((utility - utility_floor) / self._utility_range)
                + 0.06 * jitter,
                utility,
                score,
            )

        return max(eligible, key=movement_score)[2]

    def _observe_opponent_offer(self, state: SAOState) -> None:
        offer = state.current_offer
        if offer is None or not self._opponent_counts:
            return
        key = (state.step, offer)
        if key in self._seen_opponent_offers:
            return
        self._seen_opponent_offers.add(key)
        decay = 0.985
        for counts in self._opponent_counts:
            for value in list(counts):
                counts[value] *= decay
        for index, value in enumerate(offer):
            if index < len(self._opponent_counts):
                self._opponent_counts[index][value] += 1.0
        self._opponent_history.append((self._relative_time(state), offer, self._utility(offer)))
        if len(self._opponent_history) > 40:
            self._opponent_history = self._opponent_history[-40:]

    def _update_opponent_model(self, state: SAOState) -> None:
        """Build estimated opponent utility function from frequency data.

        Uses Smith Frequency Model insight: issues where the opponent is most
        consistent (low entropy) have higher importance.  The estimated ufun
        is stored as ``opponent_ufun`` so the Kendall Tau deception metric
        awards credit for accurate modeling.  (Baarslag et al., AAMAS)
        """
        if (
            not self._opponent_counts
            or not self._ranked_outcomes
            or self.nmi is None
            or self.nmi.outcome_space is None
        ):
            return
        # Only update every 4 steps to avoid thrashing
        if state.step - self._last_model_update_step < 4:
            return
        # Need at least 3 opponent offers for meaningful estimation
        if len(self._opponent_history) < 3:
            return

        self._last_model_update_step = state.step

        # 1. Compute issue weights from frequency entropy
        issue_weights: list[float] = []
        for counts in self._opponent_counts:
            total = sum(counts.values())
            if total <= 0:
                issue_weights.append(0.0)
                continue
            entropy = -sum(
                (v / total) * math.log2(v / total + 1e-12)
                for v in counts.values()
                if v > 0
            )
            max_entropy = math.log2(max(2, len(counts)))
            # Low entropy = high consistency = high importance to opponent
            issue_weights.append(max(0.0, 1.0 - entropy / max(1e-9, max_entropy)))

        weight_sum = sum(issue_weights) or 1.0
        issue_weights = [w / weight_sum for w in issue_weights]

        # 2. Build mapping: estimated opponent utility for every outcome
        model_values: dict[Outcome, float] = {}
        for outcome, _ in self._ranked_outcomes:
            score = 0.0
            for i, value in enumerate(outcome):
                if i >= len(self._opponent_counts):
                    continue
                counts = self._opponent_counts[i]
                total = sum(counts.values())
                if total <= 0:
                    continue
                # Combine exact frequency match with closeness to preferred
                freq = counts.get(value, 0.0) / total
                preferred = max(counts.items(), key=lambda item: item[1])[0]
                closeness = self._numeric_closeness(value, preferred, i)
                llm_bias = self._llm_value_bias_score(i, value)
                if llm_bias is None:
                    value_score = 0.6 * freq + 0.4 * closeness
                else:
                    value_score = 0.50 * freq + 0.30 * closeness + 0.20 * llm_bias
                score += issue_weights[i] * value_score
            model_values[outcome] = score

        try:
            model = MappingUtilityFunction(
                model_values, outcome_space=self.nmi.outcome_space
            )
            self.opponent_ufun = model
            self._private_info["opponent_ufun"] = model
        except Exception:
            pass  # Fail silently — deterministic fallback is fine

        # 3. Update concealment issue order (shuffled ranking of our issues
        #    for randomizing which issues we concede on first)
        if not self._concession_issue_order or state.step % 12 == 0:
            self._concession_issue_order = self._build_concealment_order(state)

    def _build_concealment_order(self, state: SAOState) -> list[int]:
        """Build a shuffled issue ordering for concession randomization.

        Instead of conceding on least-important issues first (which reveals
        our ranking), we create a step-dependent permutation that varies
        which issues we concede on.  (Renting et al., preference hiding)
        """
        n = len(self._opponent_counts)
        if n == 0:
            return []
        # Use step-based hash to create a pseudo-random but deterministic order
        seed_text = f"{self.id}:{state.step // 12}".encode("utf-8", errors="ignore")
        digest = hashlib.sha256(seed_text).hexdigest()
        indices = list(range(n))
        # Fisher-Yates shuffle using hash bytes
        for i in range(n - 1, 0, -1):
            start = (i * 2) % len(digest)
            byte = digest[start : start + 2]
            if len(byte) < 2:
                byte += digest[: 2 - len(byte)]
            j = int(byte, 16) % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        return indices

    def _concealment_score(self, outcome: Outcome, state: SAOState) -> float:
        """Score how well this outcome conceals our preference ordering.

        Prefer outcomes that differ from the last offer on multiple issues
        (not just one) and that vary which issues change — making it harder
        for frequency-based opponent models to infer our issue weights.
        """
        if self._last_offer is None or len(outcome) != len(self._last_offer):
            return 0.0
        if not self._concession_issue_order:
            return 0.0

        changed_issues = [
            i for i, (a, b) in enumerate(zip(outcome, self._last_offer)) if a != b
        ]
        if not changed_issues:
            return 0.0

        # Reward changing 2+ issues (harder to infer which matters more)
        multi_change_bonus = 0.3 if len(changed_issues) >= 2 else 0.0

        # Reward changing issues that are NOT at the top of the concealment
        # order (i.e., not always changing the same issues)
        if self._concession_issue_order:
            # Prefer changes on issues later in the randomized order
            order_score = sum(
                self._concession_issue_order.index(i) / max(1, len(self._concession_issue_order) - 1)
                for i in changed_issues
                if i < len(self._concession_issue_order)
            ) / max(1, len(changed_issues))
        else:
            order_score = 0.0

        return multi_change_bonus + 0.7 * order_score

    def _human_acceptability(self, outcome: Outcome) -> float:
        if not self._opponent_counts:
            return 0.0
        scores = []
        for index, value in enumerate(outcome):
            if index >= len(self._opponent_counts):
                continue
            counts = self._opponent_counts[index]
            total = sum(counts.values())
            if total <= 0:
                scores.append(0.0)
                continue
            exact = counts.get(value, 0.0) / total
            preferred = max(counts.items(), key=lambda item: item[1])[0]
            closeness = self._numeric_closeness(value, preferred, index)
            scores.append((0.55 * exact) + (0.45 * closeness))
        frequency_score = sum(scores) / len(scores) if scores else 0.0
        llm_score = self._llm_acceptability(outcome)
        if llm_score is None:
            return frequency_score
        return (0.78 * frequency_score) + (0.22 * llm_score)

    def _llm_acceptability(self, outcome: Outcome) -> float | None:
        if not self._llm_value_biases:
            return None
        scores = []
        for index, value in enumerate(outcome):
            score = self._llm_value_bias_score(index, value)
            if score is not None:
                scores.append(score)
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _llm_value_bias_score(self, issue_index: int, value: Any) -> float | None:
        if issue_index >= len(self._llm_value_biases):
            return None
        biases = self._llm_value_biases[issue_index]
        if not biases:
            return None
        total = max(1e-9, max(biases.values()))
        return max(0.0, min(1.0, biases.get(value, 0.0) / total))

    def _phase(self, t: float) -> str:
        if t < 0.62:
            return EARLY_PHASE
        if t < 0.84:
            return MID_PHASE
        if t < 0.96:
            return CLOSING_PHASE
        return PANIC_PHASE

    def _floor_margin(self, t: float, opponent_type: str) -> float:
        if t < 0.62:
            return 0.82 if opponent_type == CONCEDING_OPPONENT else 0.76
        if t < 0.84:
            return 0.58 if opponent_type == CONCEDING_OPPONENT else 0.48
        if t < 0.96:
            return 0.28 if opponent_type in (STUBBORN_OPPONENT, HARD_OPPONENT) else 0.36
        return 0.08 if opponent_type in (STUBBORN_OPPONENT, HARD_OPPONENT) else 0.14

    def _score_weights(
        self, t: float, opponent_type: str
    ) -> tuple[float, float, float, float]:
        phase = self._phase(t)
        if phase == EARLY_PHASE:
            return (0.92, 0.05, 0.02, 0.03)
        if phase == MID_PHASE:
            if opponent_type == CONCEDING_OPPONENT:
                return (0.86, 0.08, 0.04, 0.03)
            return (0.72, 0.17, 0.07, 0.04)
        if phase == CLOSING_PHASE:
            if opponent_type in (STUBBORN_OPPONENT, HARD_OPPONENT):
                return (0.40, 0.40, 0.13, 0.07)
            if opponent_type == CONCEDING_OPPONENT:
                return (0.66, 0.20, 0.08, 0.06)
            return (0.50, 0.32, 0.11, 0.07)
        if opponent_type in (STUBBORN_OPPONENT, HARD_OPPONENT):
            return (0.22, 0.56, 0.15, 0.07)
        return (0.34, 0.45, 0.14, 0.07)

    def _opponent_type(self) -> str:
        if len(self._opponent_history) < 4:
            return self._social_opponent_type() or UNKNOWN_OPPONENT
        utilities = [item[2] for item in self._opponent_history]
        offers = [item[1] for item in self._opponent_history]
        unique_ratio = len(set(offers)) / len(offers)
        first_avg = sum(utilities[: max(2, len(utilities) // 3)]) / max(2, len(utilities) // 3)
        last_slice = utilities[-max(2, len(utilities) // 3):]
        last_avg = sum(last_slice) / len(last_slice)
        improvement = last_avg - first_avg
        normalized_improvement = improvement / self._utility_range
        best_seen = max(utilities)
        t = self._opponent_history[-1][0]
        current_target = self._best_utility - (t**3.0) * (
            self._best_utility - (self._reserved_value() + 0.25 * self._utility_range)
        )

        if best_seen >= current_target + 0.04 * self._utility_range:
            return COOPERATIVE_OPPONENT
        if unique_ratio <= 0.25 and normalized_improvement < 0.04:
            return STUBBORN_OPPONENT
        if normalized_improvement > 0.12:
            return CONCEDING_OPPONENT
        if unique_ratio >= 0.75 and max(utilities) - min(utilities) > 0.35 * self._utility_range:
            return RANDOM_OPPONENT
        if last_avg <= self._reserved_value() + 0.12 * self._utility_range:
            return HARD_OPPONENT
        social_type = self._social_opponent_type()
        if social_type in (STUBBORN_OPPONENT, COOPERATIVE_OPPONENT, CONCEDING_OPPONENT):
            return social_type
        return UNKNOWN_OPPONENT

    def _social_opponent_type(self) -> str | None:
        style = str(self._llm_social_state.get("style", "")).lower()
        if style in {"cooperative", "fairness", "collaborative"}:
            return COOPERATIVE_OPPONENT
        if style in {"anchoring", "stubborn", "hard"}:
            return STUBBORN_OPPONENT
        if style in {"deadline", "closing", "flexible"}:
            return CONCEDING_OPPONENT
        return None

    def _opponent_concession_fit(self, outcome: Outcome) -> float:
        if not self._opponent_history:
            return 0.0
        recent = self._opponent_history[-min(6, len(self._opponent_history)):]
        scores = []
        for _, offer, utility in recent:
            similarity = self._outcome_similarity(outcome, offer)
            recency_weight = 0.6 + 0.4 * ((utility - self._worst_utility) / self._utility_range)
            scores.append(similarity * recency_weight)
        return max(scores) if scores else 0.0

    def _outcome_similarity(self, first: Outcome, second: Outcome) -> float:
        if not first or not second:
            return 0.0
        scores = []
        for index, (a, b) in enumerate(zip(first, second)):
            if a == b:
                scores.append(1.0)
            else:
                scores.append(self._numeric_closeness(a, b, index))
        return sum(scores) / len(scores) if scores else 0.0

    def _outcome_distance(self, first: Outcome, second: Outcome) -> float:
        return 1.0 - self._outcome_similarity(first, second)

    def _domain_score(self, outcome: Outcome) -> float:
        if self._domain_kind == "trade":
            return self._trade_domain_score(outcome)
        if self._domain_kind == "allocation":
            return self._allocation_domain_score(outcome)
        return 0.0

    def _allocation_domain_score(self, outcome: Outcome) -> float:
        if not self._ranked_outcomes or not self._opponent_counts:
            return 0.0
        own_score = (self._utility(outcome) - self._worst_utility) / self._utility_range
        opponent_score = self._human_acceptability(outcome)
        leak_penalty = self._preference_leak_penalty(outcome)
        return max(0.0, min(1.0, (0.45 * own_score) + (0.45 * opponent_score) - leak_penalty))

    def _trade_domain_score(self, outcome: Outcome) -> float:
        if len(outcome) != 2:
            return 0.0
        own_score = (self._utility(outcome) - self._worst_utility) / self._utility_range
        acceptability = self._human_acceptability(outcome)
        smoothness = self._opponent_concession_fit(outcome)
        return max(0.0, min(1.0, (0.50 * own_score) + (0.30 * acceptability) + (0.20 * smoothness)))

    def _preference_leak_penalty(self, outcome: Outcome) -> float:
        if self._last_offer is None or len(outcome) != len(self._last_offer):
            return 0.0
        changed = sum(1 for a, b in zip(outcome, self._last_offer) if a != b)
        if changed == 1:
            return 0.08
        return 0.0

    def _extract_issue_names(self) -> list[str]:
        try:
            issues = self.nmi.outcome_space.issues  # type: ignore[union-attr]
        except Exception:
            return []
        return [str(getattr(issue, "name", "")).lower() for issue in issues]

    def _detect_domain_kind(self) -> str:
        names = set(self._issue_names)
        if {"quantity", "price"}.issubset(names):
            return "trade"
        if names & {"apple", "banana", "orange", "watermelon", "compass", "container"}:
            return "allocation"
        return "generic"

    def _stubborn_compromise_offer(
        self, t: float, opponent_type: str, reserved_floor: float
    ) -> Outcome | None:
        if t < 0.94 or opponent_type not in (STUBBORN_OPPONENT, HARD_OPPONENT):
            return None
        if self._domain_kind != "allocation" or not self._opponent_counts:
            return None
        if any(issue_range is not None for issue_range in self._issue_ranges):
            return None

        preferred_values = []
        for counts in self._opponent_counts:
            if not counts:
                return None
            preferred_values.append(max(counts.items(), key=lambda item: item[1])[0])
        preferred_offer = tuple(preferred_values)

        target = self._reserved_value() + 0.18 * self._utility_range
        best: tuple[float, float, float, Outcome] | None = None
        for outcome, utility in self._ranked_outcomes:
            if utility < reserved_floor:
                continue
            differences = sum(
                1 for value, preferred in zip(outcome, preferred_offer) if value != preferred
            )
            if differences not in (1, 2):
                continue
            acceptability = self._human_acceptability(outcome)
            closeness_to_target = -abs(utility - target) / self._utility_range
            score = (acceptability, -differences, closeness_to_target, outcome)
            if best is None or score[:3] > best[:3]:
                best = (acceptability, -differences, closeness_to_target, outcome)
        return best[3] if best is not None else None

    def _update_llm_social_model(self, state: SAOState) -> None:
        if not self._should_use_llm_social(state):
            return
        step = int(getattr(state, "step", 0))
        if step - self._last_social_step < self.social_interval:
            return
        text = self._incoming_text(state)
        if not text:
            return

        prompt = self._social_prompt(state, text)
        generated = self._call_ollama(
            prompt,
            timeout=self.social_timeout,
            max_tokens=130,
            temperature=0.1,
        )
        parsed = self._parse_social_read(generated)
        if parsed is None:
            return

        self._last_social_step = step
        self._llm_social_state = parsed
        self._merge_priority_hints(parsed.get("priority_hints", []))

    def _should_use_llm_social(self, state: SAOState) -> bool:
        if self._llm_available is False:
            return False
        if self.use_llm_social is True:
            return True
        if self.use_llm_social is False:
            return False
        if str(self.use_llm_social).lower() != "auto":
            return False
        return bool(self._incoming_text(state)) and (
            self.human_mode or self._looks_like_human_interaction(state)
        )

    def _social_prompt(self, state: SAOState, text: str) -> str:
        recent = [
            {"t": round(t, 2), "offer": list(offer), "utility_to_us": round(utility, 4)}
            for t, offer, utility in self._opponent_history[-5:]
        ]
        return (
            "Read this negotiation message as a social signal. "
            "Do not choose or invent offers.\n"
            f"Progress: {self._relative_time(state):.2f}. Opponent message: {text[:260]}\n"
            f"Issue names: {self._issue_names}. Recent opponent offers: {json.dumps(recent)}\n"
            "Return strict JSON with keys: "
            "style one of cooperative,fairness,anchoring,frustrated,deadline,confused,unknown; "
            "tone under 4 words; summary under 10 words; "
            "priority_hints as a list of objects with issue_index, value, confidence 0..1. "
            "Only include priority_hints directly supported by the message or repeated offers."
        )

    def _parse_social_read(self, text: str | None) -> dict[str, Any] | None:
        cleaned = self._clean_message(text)
        if not cleaned:
            return None
        match = re.search(r"\{.*\}", cleaned)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None

        style = str(value.get("style", UNKNOWN_OPPONENT)).lower()
        allowed = {
            "cooperative",
            "fairness",
            "anchoring",
            "frustrated",
            "deadline",
            "confused",
            "unknown",
        }
        if style not in allowed:
            style = UNKNOWN_OPPONENT
        hints = []
        for raw_hint in value.get("priority_hints", []):
            if not isinstance(raw_hint, dict):
                continue
            try:
                issue_index = int(raw_hint.get("issue_index"))
                confidence = float(raw_hint.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if issue_index < 0:
                continue
            matched_value = self._match_issue_value(issue_index, raw_hint.get("value"))
            if matched_value is None:
                continue
            hints.append(
                {
                    "issue_index": issue_index,
                    "value": matched_value,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        return {
            "style": style,
            "tone": str(value.get("tone", "steady"))[:40],
            "summary": str(value.get("summary", ""))[:80],
            "priority_hints": hints[:6],
        }

    def _merge_priority_hints(self, hints: list[dict[str, Any]]) -> None:
        if not self._llm_value_biases:
            return
        for biases in self._llm_value_biases:
            for value in list(biases):
                biases[value] *= 0.70
                if biases[value] < 0.02:
                    del biases[value]
        for hint in hints:
            issue_index = hint["issue_index"]
            if issue_index >= len(self._llm_value_biases):
                continue
            self._llm_value_biases[issue_index][hint["value"]] += hint["confidence"]

    def _match_issue_value(self, issue_index: int, raw_value: Any) -> Any | None:
        if issue_index >= len(self._opponent_counts):
            return None
        values = []
        seen = set()
        for outcome, _ in self._ranked_outcomes:
            value = outcome[issue_index]
            if value not in seen:
                values.append(value)
                seen.add(value)
        raw_text = str(raw_value).strip().lower()
        for value in values:
            if str(value).strip().lower() == raw_text:
                return value
        try:
            raw_number = float(raw_value)
        except (TypeError, ValueError):
            return None
        for value in values:
            try:
                if float(value) == raw_number:
                    return value
            except (TypeError, ValueError):
                continue
        return None

    def _should_use_llm_reasoning(
        self, state: SAOState, phase: str, opponent_type: str
    ) -> bool:
        if self.use_llm_reasoning is True:
            enabled = True
        elif self.use_llm_reasoning is False:
            enabled = False
        elif str(self.use_llm_reasoning).lower() == "auto":
            enabled = self.human_mode or self._looks_like_human_interaction(state)
        else:
            enabled = False
        if not enabled:
            return False
        if phase == EARLY_PHASE and not self.human_mode:
            return False
        if opponent_type == COOPERATIVE_OPPONENT:
            return False
        if state.step - self._last_reasoning_step < self.reasoning_interval:
            return False
        return True

    def _reasoned_offer_choice(
        self,
        state: SAOState,
        phase: str,
        opponent_type: str,
        scored: list[tuple[float, float, Outcome]],
        reserved_floor: float,
    ) -> Outcome | None:
        if not self._should_use_llm_reasoning(state, phase, opponent_type):
            return None
        candidates = scored[: min(self.reasoning_candidates, len(scored))]
        if len(candidates) < 2:
            return None

        prompt = self._reasoning_prompt(state, phase, opponent_type, candidates)
        generated = self._call_ollama(
            prompt,
            timeout=self.reasoning_timeout,
            max_tokens=120,
            temperature=min(0.25, self.temperature),
        )
        index = self._parse_reasoning_choice(generated, len(candidates))
        if index is None:
            return None
        outcome = candidates[index][2]
        if self._utility(outcome) < reserved_floor:
            return None
        self._last_reasoning_step = state.step
        return outcome

    def _reasoning_prompt(
        self,
        state: SAOState,
        phase: str,
        opponent_type: str,
        candidates: list[tuple[float, float, Outcome]],
    ) -> str:
        rows = []
        for index, (score, utility, outcome) in enumerate(candidates):
            rows.append(
                {
                    "index": index,
                    "outcome": list(outcome),
                    "own_utility": round(utility, 4),
                    "model_score": round(score, 4),
                    "opponent_acceptability": round(self._human_acceptability(outcome), 4),
                    "similarity_to_recent": round(self._opponent_concession_fit(outcome), 4),
                }
            )
        offer = getattr(state, "current_offer", None)
        current_utility = self._utility(offer)
        return (
            "You are a bounded advisor for a negotiation agent. "
            "Choose one candidate offer index. Do not invent offers.\n"
            f"Phase: {phase}. Opponent type: {opponent_type}. "
            f"Elapsed: {self._relative_time(state):.2f}.\n"
            f"Opponent last offer: {offer}. Utility to us: {current_utility:.4f}.\n"
            "Goal: maximize our expected score by balancing high own utility with agreement chance. "
            "Prefer higher own utility unless a slightly lower candidate is much more likely to close.\n"
            f"Candidates JSON: {json.dumps(rows)}\n"
            "Return strict JSON only: {\"index\": 0, \"reason\": \"short\"}"
        )

    def _parse_reasoning_choice(self, text: str | None, n_candidates: int) -> int | None:
        cleaned = self._clean_message(text)
        if not cleaned:
            return None
        match = re.search(r"\{.*\}", cleaned)
        if match:
            try:
                value = json.loads(match.group(0))
                index = int(value.get("index"))
                return index if 0 <= index < n_candidates else None
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        match = re.search(r"\b([0-9]+)\b", cleaned)
        if not match:
            return None
        index = int(match.group(1))
        return index if 0 <= index < n_candidates else None

    def _numeric_range(
        self, outcomes: list[Outcome], issue_index: int
    ) -> tuple[float, float] | None:
        values: list[float] = []
        for outcome in outcomes:
            try:
                values.append(float(outcome[issue_index]))
            except (TypeError, ValueError):
                return None
        return (min(values), max(values)) if values else None

    def _numeric_closeness(self, value: Any, preferred: Any, issue_index: int) -> float:
        if value == preferred:
            return 1.0
        if issue_index >= len(self._issue_ranges):
            return 0.0
        issue_range = self._issue_ranges[issue_index]
        if issue_range is None:
            return 0.0
        low, high = issue_range
        span = max(1e-9, high - low)
        try:
            return max(0.0, 1.0 - (abs(float(value) - float(preferred)) / span))
        except (TypeError, ValueError):
            return 0.0

    def _stable_jitter(self, outcome: Outcome) -> float:
        text = repr((self.id, outcome)).encode("utf-8", errors="ignore")
        digest = hashlib.sha256(text).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _message(self, intent: str, state: SAOState, outcome: Outcome | None) -> str:
        if self.use_llm_text is False:
            return ""
        if not self._should_use_llm_text(state):
            return self._template_message(intent)
        key = self._message_cache_key(intent, state, outcome)
        cached = self._llm_message_cache.get(key)
        if cached:
            index = self._llm_message_cache_index.get(key, 0) % len(cached)
            self._llm_message_cache_index[key] = index + 1
            return cached[index]

        prompt = self._text_variants_prompt(intent, state, outcome)
        generated = self._call_ollama(
            prompt,
            timeout=self.timeout,
            max_tokens=420,
            temperature=min(0.55, max(0.25, self.temperature)),
        )
        variants = self._parse_message_variants(generated, outcome)
        if not variants:
            variants = list(self._template_messages(intent))

        self._llm_message_cache[key] = variants
        self._llm_message_cache_index[key] = 1
        return variants[0]

    def _should_use_llm_text(self, state: SAOState) -> bool:
        if self._llm_available is False:
            return False
        if self.use_llm_text is True:
            return True
        if self.use_llm_text is False:
            return False
        if str(self.use_llm_text).lower() != "auto":
            return False
        return self.human_mode or self._looks_like_human_interaction(state)

    def _message_cache_key(
        self, intent: str, state: SAOState, outcome: Outcome | None
    ) -> tuple[str, str, str, str]:
        phase = self._phase(self._relative_time(state))
        return (intent, phase, "llm", "generic")

    def _looks_like_human_interaction(self, state: SAOState) -> bool:
        names = (
            getattr(state, "current_proposer", None),
            getattr(state, "current_proposer_agent", None),
            getattr(state, "last_negotiator", None),
        )
        if any(
            isinstance(name, str)
            and any(marker in name.lower() for marker in ("human", "user", "hani"))
            for name in names
        ):
            return True
        text = self._incoming_text(state)
        if not text:
            return False
        lowered = text.lower()
        bot_template_markers = (
            "i am sorry, but i cannot accept",
            "this package adjusts the terms",
            "i can move in your direction",
            "there is still room to finish",
            "thank you for this great offer",
            "thank you for the opportunity to negotiate",
            "thank you for negotiating with me",
            "my opening offer is",
            "here's my initial proposal",
            "i'd like to start with",
            "let's get started",
            "i'd like to propose",
            "i'm excited to negotiate with you",
            "i'd like to kick things off",
            "kick things off with",
            "my first offer is",
            "i appreciate your offer, but",
            "thank you for your proposal, however",
            "while i understand your position",
            "i've considered your offer carefully",
            "i see where you're coming from",
            "thanks for the suggestion",
            "i value your input",
            "that's an interesting offer",
            "after careful consideration",
            "looking at your proposal",
            "i've reviewed your terms",
            "your proposal",
            "i'm proposing",
            "i'm countering",
            "instead of your",
            "i'd prefer",
            "would work better",
            "is fairer",
            "more reasonable than",
            "let's adjust",
            "higher than reasonable",
            "beyond what i consider fair",
            "more than i can accept",
            "above my target",
            "steeper than i anticipated",
            "exceeding my limits",
            "let me propose an alternative",
            "this should address our concerns",
            "perhaps we can find common ground",
            "i believe this is a fairer arrangement",
        )
        if any(marker in lowered for marker in bot_template_markers):
            return False
        return any(token in lowered for token in ("i ", "me", "my", "need", "want", "fair", "deal", "please"))

    def _template_message(self, intent: str) -> str:
        messages = self._template_messages(intent)
        index = int(self._stable_jitter((intent, getattr(self, "id", ""))) * len(messages))
        return messages[min(index, len(messages) - 1)]

    def _template_messages(self, intent: str) -> tuple[str, ...]:
        return {
            "opening": OPENING_MESSAGES,
            "counter": COUNTER_MESSAGES,
            "late": LATE_MESSAGES,
            "accept": ACCEPT_MESSAGES,
        }.get(intent, COUNTER_MESSAGES)

    def _repair_text_prompt(
        self,
        unsafe_message: str,
        intent: str,
        state: SAOState,
        outcome: Outcome | None,
    ) -> str:
        return (
            "Rewrite this negotiation message so it is safe and natural.\n"
            f"Unsafe draft: {unsafe_message[:220]}\n"
            f"Intent: {intent}. Progress: {round(100 * self._relative_time(state))}%.\n"
            f"Offer tuple: {outcome}. Visible package change: {self._offer_change_summary(outcome)}.\n"
            "Rules: under 22 words; no exact utilities; no reserved values; no minimum/maximum; "
            "no issue rankings; no private preferences; no threats. Return only the rewritten message."
        )

    def _text_variants_prompt(
        self, intent: str, state: SAOState, outcome: Outcome | None
    ) -> str:
        progress = round(100 * self._relative_time(state))
        phase = self._phase(self._relative_time(state))
        action = "accepting their offer" if intent == "accept" else "making a counteroffer"
        social_style = self._llm_social_state.get("style", UNKNOWN_OPPONENT)
        social_tone = self._llm_social_state.get("tone", "steady")
        social_summary = self._llm_social_state.get("summary", "")
        opponent_text = self._incoming_text(state)
        package_facts = self._package_facts(outcome)
        return (
            "Generate 20 different short negotiation messages for the agent to use over nearby turns.\n"
            f"Action: {action}. Progress: {progress}%. Phase: {phase}.\n"
            f"Social read: {social_style}; tone: {social_tone}; signal: {social_summary}.\n"
            f"Current package facts, for context only: {package_facts}.\n"
            f"Opponent message: {opponent_text[:220] if opponent_text else 'None'}\n"
            "Style: human, plain, warm, cooperative but firm. Avoid sounding templated. "
            "Do not mention hidden preferences or numeric scoring.\n"
            "Grounding: do not mention issue names, values, names from the offer tuple, or specific package changes. "
            "Use general phrases like this package, this version, movement, balance, workable, and agreement.\n"
            "Rules for every message: under 22 words; no exact utilities; no reserved values; "
            "no minimum/maximum; no issue rankings; no threats; no claims this is final.\n"
            "Return strict JSON only: {\"messages\": [\"...\", \"...\"]}"
        )

    def _parse_message_variants(
        self, text: str | None, outcome: Outcome | None = None
    ) -> list[str]:
        cleaned = self._clean_message(text)
        if not cleaned:
            return []
        raw_items: list[Any] = []
        json_ready = cleaned.replace('\\"', '"')
        object_match = re.search(r"\{.*\}", json_ready)
        array_match = re.search(r"\[[\s\S]*\]", json_ready)
        if object_match:
            try:
                value = json.loads(object_match.group(0))
                messages = value.get("messages")
                if isinstance(messages, list):
                    raw_items = messages
            except json.JSONDecodeError:
                raw_items = []
        elif array_match:
            try:
                value = json.loads(array_match.group(0))
                if isinstance(value, list):
                    raw_items = value
            except json.JSONDecodeError:
                raw_items = []
        if not raw_items:
            raw_items = [
                re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
                for line in cleaned.splitlines()
                if line.strip()
            ]
        safe: list[str] = []
        seen = set()
        for raw in raw_items:
            message = self._clean_message(str(raw))
            if message and message.startswith("[") and message.endswith("]"):
                try:
                    nested = json.loads(message.replace('\\"', '"'))
                    if (
                        isinstance(nested, list)
                        and len(nested) == 1
                        and isinstance(nested[0], str)
                    ):
                        message = self._clean_message(nested[0])
                except json.JSONDecodeError:
                    pass
            if not self._is_safe_message(message):
                continue
            if message and (
                (message.startswith("[") and message.endswith("]"))
                or (message.startswith("{") and message.endswith("}"))
            ):
                continue
            if self._mentions_offer_terms(message):
                continue
            if not self._is_grounded_message(message, outcome):
                continue
            key = message.lower()
            if key in seen:
                continue
            seen.add(key)
            safe.append(message)
        return safe[:20]

    def _usable_generated_message(
        self, text: str | None, outcome: Outcome | None = None
    ) -> str | None:
        message = self._clean_message(text)
        if not self._is_safe_message(message):
            return None
        if self._mentions_offer_terms(message):
            return None
        if not self._is_grounded_message(message, outcome):
            return None
        return message

    def _mentions_offer_terms(self, message: str | None) -> bool:
        if not message:
            return False
        lowered = message.lower()
        for name in self._issue_names:
            if name and len(name) > 1 and re.search(rf"\b{re.escape(name.lower())}\b", lowered):
                return True
        for index in range(len(self._issue_names)):
            for value in self._issue_values(index):
                token = str(value).lower()
                if len(token) > 1 and re.search(rf"\b{re.escape(token)}\b", lowered):
                    return True
        return False

    def _package_facts(self, outcome: Outcome | None) -> str:
        if outcome is None:
            return "no package"
        facts = []
        for index, value in enumerate(outcome):
            name = self._issue_names[index] if index < len(self._issue_names) else f"issue {index + 1}"
            facts.append(f"{name}={value}")
        return ", ".join(facts)

    def _is_grounded_message(self, message: str | None, outcome: Outcome | None) -> bool:
        if not message or outcome is None or not self._ranked_outcomes:
            return True
        lowered = message.lower()
        for index, actual_value in enumerate(outcome):
            issue_name = (
                self._issue_names[index]
                if index < len(self._issue_names) and self._issue_names[index]
                else f"issue {index + 1}"
            )
            issue_lower = issue_name.lower()
            if issue_lower not in lowered:
                continue
            actual_lower = str(actual_value).lower()
            for possible in self._issue_values(index):
                possible_lower = str(possible).lower()
                if possible_lower == actual_lower:
                    continue
                if possible_lower in lowered:
                    return False
        return True

    def _issue_values(self, issue_index: int) -> list[Any]:
        values = []
        seen = set()
        for outcome, _ in self._ranked_outcomes:
            if issue_index >= len(outcome):
                continue
            value = outcome[issue_index]
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
        return values

    def _offer_change_summary(self, outcome: Outcome | None) -> str:
        if outcome is None or self._last_offer is None:
            return "new package"
        changes = []
        for index, (old, new) in enumerate(zip(self._last_offer, outcome)):
            if old == new:
                continue
            name = self._issue_names[index] if index < len(self._issue_names) else f"issue {index + 1}"
            changes.append(f"{name}: {old} to {new}")
        if not changes:
            return "no visible change"
        return "; ".join(changes[:3])

    def _incoming_text(self, state: SAOState) -> str:
        data_candidates = []
        current_data = getattr(state, "current_data", None)
        if isinstance(current_data, dict):
            data_candidates.append(current_data)
        new_data = getattr(state, "new_data", None)
        if isinstance(new_data, list):
            for _, data in reversed(new_data[-4:]):
                if isinstance(data, dict):
                    data_candidates.append(data)
        for data in data_candidates:
            text = data.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        return ""

    def _call_ollama(
        self,
        prompt: str,
        timeout: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for _ in range(self.num_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout if timeout is None else timeout
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = body.get("response")
                self._llm_available = True
                return text.strip() if isinstance(text, str) else None
            except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                continue
        self._llm_available = False
        return None

    def _is_safe_message(self, message: str | None) -> bool:
        if not message:
            return False
        message = re.sub(r"\s+", " ", message).strip().strip("\"'")
        lowered = message.lower()
        if any(term in lowered for term in PRIVATE_TERMS):
            return False
        if len(message.split()) > 24 or len(message) > 180:
            return False
        return bool(re.search(r"[A-Za-z]", message))

    def _clean_message(self, message: str | None) -> str | None:
        if not message:
            return None
        text = re.sub(r"<think>.*?</think>", "", message, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"\s+", " ", text).strip().strip("\"'")
        if text.lower().startswith("message:"):
            text = text.split(":", 1)[1].strip()
        return text


class Agent96Reasoning(Agent96):
    """Experimental LLM-advised variant.

    Qwen can choose among already-scored, valid candidate offers. The deterministic
    safety layer still enforces outcome validity and utility floors.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("use_llm_reasoning", True)
        kwargs.setdefault("use_llm_text", True)
        kwargs.setdefault("timeout", 1.5)
        kwargs.setdefault("reasoning_timeout", 3.5)
        kwargs.setdefault("reasoning_interval", 10)
        kwargs.setdefault("reasoning_candidates", 6)
        super().__init__(**kwargs)
