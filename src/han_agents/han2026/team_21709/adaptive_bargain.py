import contextlib
import io
from typing import Any

from negmas.common import Outcome
from negmas.gb.components.genius.models import GSmithFrequencyModel
from negmas.gb.common import ResponseType
from negmas.sao import SAOState
from negmas.sao.components.acceptance import ACNext
from negmas.sao.components.offering import TimeBasedOfferingPolicy
from negmas.sao.negotiators.modular import BOANegotiator
from negmas_llm.meta import LLMMetaNegotiator

try:
    from negmas_llm.common import DEFAULT_MODELS
except ImportError:
    DEFAULT_MODELS = dict(ollama="qwen3:4b-instruct")


DEFAULT_OLLAMA_MODEL = DEFAULT_MODELS["ollama"]


SYSTEM_PROMPT = """
You are assisting a negotiation agent in the HAN 2026 league.

The underlying agent already chooses offers and accept/reject decisions using:
- a time-based offering strategy,
- a frequency-based opponent model,
- and an ACNext-style acceptance strategy.

Your job is to write concise, natural, persuasive text that explains the
agent's strategic move to a human partner. Do not change the formal offer or
decision. Keep the tone professional and cooperative while protecting the
agent's utility.

Write at most 35 words. Do not restate tuple-valued offers, internal utility
numbers, thresholds, model names, or implementation details. Avoid repetitive
thanks.

Respond with only a JSON object:
{
  "text": "message to send to the other negotiator"
}
"""


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _reserved_value(ufun) -> float:
    reserved = getattr(ufun, "reserved_value", None)
    return float(reserved) if reserved is not None else float("-inf")


class AdaptiveOfferingPolicy(TimeBasedOfferingPolicy):
    """Time-based offering with opponent-aware candidate selection.

    The base time curve still defines the minimum utility we target at each
    round. Among outcomes that satisfy that target, we prefer outcomes that the
    frequency opponent model estimates as easier for the partner to accept.
    Late in the negotiation, we can reuse the opponent's best previous offer if
    it is good enough for us because that offer is known to be feasible for the
    partner.
    """

    def __init__(
        self,
        opponent_model: GSmithFrequencyModel,
        opponent_weight: float = 0.65,
        opponent_weight_start: float = 0.20,
        concession_sensitivity: float = 0.18,
        hardline_target_discount: float = 0.02,
        candidate_limit: int = 80,
        utility_slack: float = 0.04,
        target_window: float | None = None,
        reuse_best_after: float = 0.94,
        reuse_best_slack: float = 0.04,
        prediction_discount: float = 0.60,
        max_prediction_improvement: float = 0.18,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.opponent_model = opponent_model
        self.opponent_weight = opponent_weight
        self.opponent_weight_start = opponent_weight_start
        self.concession_sensitivity = concession_sensitivity
        self.hardline_target_discount = hardline_target_discount
        self.candidate_limit = candidate_limit
        self.utility_slack = utility_slack
        self.target_window = target_window
        self.reuse_best_after = reuse_best_after
        self.reuse_best_slack = reuse_best_slack
        self.prediction_discount = prediction_discount
        self.max_prediction_improvement = max_prediction_improvement
        self._opponent_offer_utilities: list[float] = []
        self._opponent_offer_times: list[float] = []
        self._opponent_offers: list[Outcome] = []

    def on_preferences_changed(self, changes):
        self._opponent_offer_utilities.clear()
        self._opponent_offer_times.clear()
        self._opponent_offers.clear()
        return super().on_preferences_changed(changes)

    def on_partner_proposal(self, state, partner_id: str, offer) -> None:
        if offer is None or not self.negotiator or not self.negotiator.ufun:
            return
        self._opponent_offer_utilities.append(float(self.negotiator.ufun(offer)))
        self._opponent_offer_times.append(float(state.relative_time))
        self._opponent_offers.append(offer)

    def _utility_bounds(self) -> tuple[float, float]:
        assert self.negotiator is not None
        assert self.negotiator.ufun is not None
        if self.sorter is not None:
            mn, mx = self.sorter.minmax()
        else:
            worst, best = self.negotiator.ufun.extreme_outcomes()
            mn = float(self.negotiator.ufun(worst))
            mx = float(self.negotiator.ufun(best))
        reserved = _reserved_value(self.negotiator.ufun)
        return max(mn, reserved), mx

    def opponent_concession_rate(self) -> float:
        """Estimate how quickly the partner is improving offers for us."""
        utilities = self._opponent_offer_utilities
        if len(utilities) < 2:
            return 0.0
        _, mx = self._utility_bounds()
        first, last = utilities[0], utilities[-1]
        room = max(1e-9, mx - first)
        return _clamp((last - first) / room, -1.0, 1.0)

    def opponent_recent_trend(self, window: int = 4) -> float:
        """Slope of the opponent's recent improvements in our utility units."""
        utilities = self._opponent_offer_utilities
        times = self._opponent_offer_times
        if len(utilities) < 2 or len(times) < 2:
            return 0.0
        start = max(0, len(utilities) - window)
        dt = times[-1] - times[start]
        if dt <= 1e-9:
            dt = float(len(utilities) - start)
        return (utilities[-1] - utilities[start]) / max(1e-9, dt)

    def opponent_hardness(self, state) -> float:
        """A soft hardliner score in [0, 1] from concession and recent trend."""
        if len(self._opponent_offer_utilities) < 3:
            return 0.0
        concession = self.opponent_concession_rate()
        trend = self.opponent_recent_trend()
        low_concession = _clamp((0.08 - concession) / 0.08, 0.0, 1.0)
        flat_trend = _clamp((0.015 - trend) / 0.04, 0.0, 1.0)
        time_pressure = float(state.relative_time)
        return _clamp(
            0.50 * low_concession + 0.35 * flat_trend + 0.15 * time_pressure,
            0.0,
            1.0,
        )

    def best_opponent_offer(self) -> tuple[Outcome | None, float]:
        """Return the opponent proposal that gave us the highest utility."""
        if not self._opponent_offers or not self._opponent_offer_utilities:
            return None, float("-inf")
        best_index = max(
            range(len(self._opponent_offer_utilities)),
            key=self._opponent_offer_utilities.__getitem__,
        )
        return self._opponent_offers[best_index], self._opponent_offer_utilities[
            best_index
        ]

    def predicted_future_opponent_utility(self, state) -> float | None:
        """Predict the best utility we can reasonably expect from future offers.

        This is intentionally conservative. It uses the opponent's recent and
        overall concession slopes and caps projected improvement to avoid
        waiting forever for an overly optimistic future offer.
        """
        utilities = self._opponent_offer_utilities
        times = self._opponent_offer_times
        if not utilities:
            return None
        _, mx = self._utility_bounds()
        best = max(utilities)
        last = utilities[-1]
        if len(utilities) < 2 or len(times) < 2:
            return _clamp(best, float("-inf"), mx)

        total_dt = times[-1] - times[0]
        if total_dt <= 1e-9:
            total_dt = float(len(utilities) - 1)
        total_slope = (utilities[-1] - utilities[0]) / max(1e-9, total_dt)
        recent_slope = self.opponent_recent_trend()
        slope = 0.65 * recent_slope + 0.35 * total_slope
        if slope <= 0.0:
            return _clamp(best, float("-inf"), mx)

        remaining = max(0.0, 1.0 - float(state.relative_time))
        hardness = self.opponent_hardness(state)
        discount = self.prediction_discount * (1.0 - 0.45 * hardness)
        projected = last + slope * remaining * discount
        improvement_cap = self.max_prediction_improvement * remaining
        return _clamp(max(best, projected), best, min(mx, best + improvement_cap))

    def _opponent_weight(self, state) -> float:
        if not self._opponent_offer_utilities:
            return self.opponent_weight_start
        t = float(state.relative_time)
        weight = self.opponent_weight + 0.15 * t
        weight += 0.04 * self.opponent_hardness(state) * (t**1.2)
        return _clamp(weight, 0.0, 0.85)

    def _was_opponent_offer(self, outcome: Outcome) -> bool:
        return any(outcome == previous for previous in self._opponent_offers)

    def _reusable_best_offer(self, state, target: float) -> Outcome | None:
        t = float(state.relative_time)
        if t < self.reuse_best_after:
            return None
        best_offer, best_utility = self.best_opponent_offer()
        if best_offer is None:
            return None
        assert self.negotiator is not None
        assert self.negotiator.ufun is not None
        reserved = _reserved_value(self.negotiator.ufun)
        progress = (t - self.reuse_best_after) / max(1e-9, 1.0 - self.reuse_best_after)
        allowed_slack = self.utility_slack + self.reuse_best_slack * (
            _clamp(progress, 0.0, 1.0) ** 0.8
        )
        if best_utility >= max(reserved, target - allowed_slack):
            return best_offer
        return None

    def _target_utility(self, state) -> float:
        assert self.negotiator.ufun is not None
        assert self.sorter is not None
        mn, mx = self._utility_bounds()
        utility_range = max(1e-9, mx - mn)

        t = float(state.relative_time)
        target_norm = self.curve.utility_at(t)
        concession = self.opponent_concession_rate()
        hardness = self.opponent_hardness(state)

        if concession > 0.20 and t < 0.85:
            # If the partner is already moving toward us, avoid conceding too fast.
            target_norm += self.concession_sensitivity * (concession - 0.20) * (1 - t)
        elif concession < 0.05 and t > 0.35:
            # If the partner is rigid, make modest room as deadline pressure grows.
            target_norm -= self.concession_sensitivity * (0.05 - concession) * t

        if hardness > 0.50 and t > 0.45:
            target_norm -= self.hardline_target_discount * hardness * (
                (t - 0.45) / 0.55
            )

        target_norm = _clamp(target_norm, 0.0, 1.0)
        return mn + target_norm * utility_range

    def __call__(self, state, dest: str | None = None):
        assert self.negotiator.ufun is not None
        if self.sorter is None:
            return super().__call__(state, dest=dest)

        mn, mx = self._utility_bounds()
        target = self._target_utility(state)
        reusable = self._reusable_best_offer(state, target)
        if reusable is not None:
            return reusable

        lower = max(mn, target - self.utility_slack)
        upper = (
            mx if self.target_window is None else min(mx, target + self.target_window)
        )
        candidates = self.sorter.some((lower, upper), normalized=False)
        if not candidates:
            candidates = self.sorter.some((lower, mx), normalized=False)
        if not candidates:
            return super().__call__(state, dest=dest)
        if len(candidates) > self.candidate_limit:
            step = max(1, len(candidates) // self.candidate_limit)
            candidates = candidates[::step][: self.candidate_limit]

        utility_range = max(1e-9, mx - mn)
        time_pressure = float(state.relative_time)
        opponent_weight = self._opponent_weight(state)
        own_weight = 1.0 - opponent_weight

        def score(outcome) -> float:
            own_utility = float(self.negotiator.ufun(outcome))
            own = (own_utility - mn) / utility_range
            partner = _clamp(float(self.opponent_model.eval(outcome)), 0.0, 1.0)
            target_fit = 1.0 - _clamp(
                abs(own_utility - target) / utility_range, 0.0, 1.0
            )
            repeats_opponent_offer = (
                time_pressure > 0.55 and self._was_opponent_offer(outcome)
            )
            repeat_bonus = 0.04 * time_pressure if repeats_opponent_offer else 0.0
            return (
                own_weight * own
                + opponent_weight * partner
                + 0.04 * target_fit
                + repeat_bonus
            )

        return max(candidates, key=score)


class AdaptiveAcceptancePolicy(ACNext):
    """ACNext with deadline-aware and prediction-aware utility thresholds."""

    def __init__(
        self,
        offering_strategy: AdaptiveOfferingPolicy,
        start_threshold: float = 0.92,
        end_threshold: float = 0.64,
        hard_partner_discount: float = 0.08,
        next_offer_margin: float = 0.12,
        prediction_start_time: float = 0.82,
        future_margin: float = 0.04,
        best_offer_accept_after: float = 0.92,
        best_offer_margin: float = 0.02,
        final_viable_discount: float = 0.03,
        **kwargs: Any,
    ) -> None:
        super().__init__(offering_strategy=offering_strategy, **kwargs)
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.hard_partner_discount = hard_partner_discount
        self.next_offer_margin = next_offer_margin
        self.prediction_start_time = prediction_start_time
        self.future_margin = future_margin
        self.best_offer_accept_after = best_offer_accept_after
        self.best_offer_margin = best_offer_margin
        self.final_viable_discount = final_viable_discount

    @property
    def adaptive_offering(self) -> AdaptiveOfferingPolicy:
        return self.offering_strategy

    def _utility_bounds(self) -> tuple[float, float]:
        assert self.negotiator is not None
        assert self.negotiator.ufun is not None
        offering = self.adaptive_offering
        if offering.sorter is not None:
            mn, mx = offering.sorter.minmax()
        else:
            worst, best = self.negotiator.ufun.extreme_outcomes()
            mn = float(self.negotiator.ufun(worst))
            mx = float(self.negotiator.ufun(best))
        reserved = _reserved_value(self.negotiator.ufun)
        return max(mn, reserved), mx

    def _dynamic_threshold(self, state) -> float:
        assert self.negotiator.ufun is not None
        offering = self.adaptive_offering
        mn, mx = self._utility_bounds()
        utility_range = max(1e-9, mx - mn)

        t = float(state.relative_time)
        threshold_norm = self.start_threshold - (
            self.start_threshold - self.end_threshold
        ) * (t**1.6)

        concession = offering.opponent_concession_rate()
        if concession < 0.05 and t > 0.50:
            threshold_norm -= self.hard_partner_discount * t
        elif concession > 0.20 and t < 0.80:
            threshold_norm += 0.04 * (1 - t)

        threshold_norm = _clamp(threshold_norm, 0.0, 1.0)
        return mn + threshold_norm * utility_range

    def _minimum_viable_utility(self, state) -> float:
        mn, mx = self._utility_bounds()
        utility_range = max(1e-9, mx - mn)
        t = float(state.relative_time)
        threshold_norm = self.end_threshold + 0.10 * (1.0 - t)
        hardness = self.adaptive_offering.opponent_hardness(state)
        if t > 0.75:
            threshold_norm -= 0.04 * hardness * ((t - 0.75) / 0.25)
        threshold_norm = _clamp(threshold_norm, 0.0, 1.0)
        return mn + threshold_norm * utility_range

    def __call__(self, state, offer, source):
        if not self.negotiator or not self.negotiator.ufun or offer is None:
            return ResponseType.REJECT_OFFER

        offer_utility = float(self.negotiator.ufun(offer))
        reserved = _reserved_value(self.negotiator.ufun)
        if offer_utility < reserved:
            return ResponseType.REJECT_OFFER

        next_offer = self.offering_strategy(state)
        next_utility = (
            float(self.negotiator.ufun(next_offer)) if next_offer is not None else 1.0
        )
        time_pressure = float(state.relative_time)
        threshold = self._dynamic_threshold(state)
        margin = self.next_offer_margin * (time_pressure**1.4)
        required = max(reserved, min(threshold, next_utility - margin))

        if offer_utility >= required:
            return ResponseType.ACCEPT_OFFER

        minimum_viable = self._minimum_viable_utility(state)
        predicted = self.adaptive_offering.predicted_future_opponent_utility(state)
        if predicted is not None and time_pressure >= self.prediction_start_time:
            waiting_margin = self.future_margin * (time_pressure**1.5)
            if (
                offer_utility >= max(reserved, minimum_viable)
                and offer_utility >= predicted - waiting_margin
            ):
                return ResponseType.ACCEPT_OFFER

        _, best_utility = self.adaptive_offering.best_opponent_offer()
        if (
            best_utility > float("-inf")
            and time_pressure >= self.best_offer_accept_after
        ):
            progress = (time_pressure - self.best_offer_accept_after) / max(
                1e-9, 1.0 - self.best_offer_accept_after
            )
            final_floor = minimum_viable - self.final_viable_discount * _clamp(
                progress, 0.0, 1.0
            )
            if (
                offer_utility >= max(reserved, final_floor)
                and offer_utility >= best_utility - self.best_offer_margin
            ):
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


class AdaptiveBargainNegotiator(LLMMetaNegotiator):
    """HAN submission negotiator using BOA strategy plus LLM text.

    The BOA base negotiator provides the formal negotiation strategy:
    - TimeBasedOfferingPolicy for proposals
    - GSmithFrequencyModel for opponent modeling
    - ACNext for acceptance

    Text follows the official tutorial's LLMMetaNegotiator adapter pattern. If
    the LLM backend is unavailable or too slow, the agent falls back to short
    cooperative text instead of crashing the negotiation.
    """

    def __init__(
        self,
        base_negotiator: BOANegotiator | None = None,
        provider: str = "ollama",
        model: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.4,
        max_tokens: int = 128,
        use_structured_output: bool = True,
        timeout: float = 60.0,
        num_retries: int = 0,
        num_ctx: int = 2048,
        keep_alive: str | None = "5m",
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AdaptiveBargainNegotiator.

        Args:
            base_negotiator: Optional BOA negotiator to wrap.
            provider/model/temperature/max_tokens/use_structured_output/timeout/
                num_retries/llm_kwargs: LLM settings passed to LLMMetaNegotiator.
            **kwargs: Extra arguments passed to LLMMetaNegotiator.
        """
        if base_negotiator is None:
            opponent_model = GSmithFrequencyModel()
            offering = AdaptiveOfferingPolicy(opponent_model=opponent_model)
            base_negotiator = BOANegotiator(
                offering=offering,
                acceptance=AdaptiveAcceptancePolicy(offering),
                model=opponent_model,
            )

        llm_kwargs = dict(llm_kwargs or {})
        llm_kwargs.setdefault("timeout", timeout)
        llm_kwargs.setdefault("num_retries", num_retries)
        llm_kwargs.setdefault("num_ctx", num_ctx)
        llm_kwargs.setdefault("think", False)
        if keep_alive is not None:
            llm_kwargs.setdefault("keep_alive", keep_alive)
        if use_structured_output:
            llm_kwargs.setdefault("format", "json")
        system_prompt = kwargs.pop("system_prompt", SYSTEM_PROMPT)

        super().__init__(
            base_negotiator=base_negotiator,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )
        self._llm_unavailable = False

    @property
    def base_negotiator(self) -> BOANegotiator:
        return self._negotiators[0]  # type: ignore[return-value]

    def _generate_text(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        if self._llm_unavailable:
            return self._fallback_text(state, action, outcome)
        try:
            if self.verbose:
                return super()._generate_text(state, action, outcome, received_text)
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(io.StringIO()):
                    return super()._generate_text(
                        state, action, outcome, received_text
                    )
        except Exception:
            self._llm_unavailable = True
            return self._fallback_text(state, action, outcome)

    def _fallback_text(
        self, state: SAOState, action: str, outcome: Outcome | None = None
    ) -> str:
        if action == "accept":
            return (
                "Thank you. This offer is workable for me, and I am happy "
                "to accept it."
            )
        if action == "end":
            return (
                "Thank you for the discussion. I do not think we can improve "
                "this enough to continue productively."
            )
        if action == "reject":
            return (
                "Thank you for explaining your position. I cannot accept that "
                "offer yet, but I am still looking for a fair agreement."
            )
        if outcome is not None and state.current_offer is None:
            return (
                "Thank you for negotiating with me. I would like to start with "
                "this proposal and hear your thoughts."
            )
        return (
            "Thank you for the offer. I cannot accept it as it stands, so I am "
            "suggesting this alternative as a more workable agreement."
        )
