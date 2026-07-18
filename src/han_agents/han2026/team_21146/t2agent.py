"""Risk-adaptive negotiator that sets utility targets from heuristic breakdown-risk, picks near-Pareto offers via frequency-based opponent modeling, and uses an LLM only to phrase messages."""

from __future__ import annotations
import contextlib
import io
import json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeAlias
import litellm
from negmas.common import Outcome
from negmas.sao.common import ResponseType, SAOResponse, SAOState
from negmas.sao.negotiators.base import SAOCallNegotiator

OfferKey: TypeAlias = tuple[str, ...]
Mode: TypeAlias = Literal["exploit", "balanced", "agreement", "salvage"]
UtilityFn: TypeAlias = Callable[[Outcome], float]


class OpponentProfile(str, Enum):
    CONCEDER = "conceder"
    BOULWARE = "boulware"
    COOPERATIVE_HUMAN = "cooperative_human"
    VOLATILE = "volatile"
    HARDLINER = "hardliner"


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum((max(0.0, float(value)) for value in weights.values()))
    if total <= 0.0:
        equal = 1.0 / max(1, len(weights))
        return {key: equal for key in weights}
    return {key: max(0.0, float(value)) / total for key, value in weights.items()}


def offer_key(offer: Outcome | None) -> OfferKey:
    if offer is None:
        return ()
    try:
        return tuple((str(value) for value in offer))
    except TypeError:
        return (str(offer),)


@dataclass
class EvaluatedOffer:
    offer: Outcome
    my_u: float
    my_norm: float
    base_opp_u: float
    opp_u: float
    low_cost_concession_score: float = 0.0


class LLMResponseTextPolicy:
    def __init__(
        self,
        model: str = "ollama/qwen3:4b-instruct",
        temperature: float = 0.4,
        max_tokens: int = 48,
        request_timeout: float | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout

    def generate(self, *, context: dict[str, Any], fallback_text: str) -> str:
        prompt = json.dumps(context, ensure_ascii=False)
        try:
            completion_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Write one short negotiation message.\nUse only the exact issue values in selected_offer.\nDo not invent, round, discount, or change any numeric value.\nMention one concession briefly if changed_points is non-empty.\nSound cooperative but firm.\nDo not mention utilities, risk, scores, or internal strategy.\nReturn only the message, max 25 words.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.request_timeout is not None:
                completion_kwargs["request_timeout"] = self.request_timeout
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                response = litellm.completion(**completion_kwargs)
            raw = (response.choices[0].message.content or "").strip()
            text = self._clean_message(raw)
            if not text:
                raise ValueError("empty response text")
            if self._has_unsupported_number(text, context):
                raise ValueError("response text changed selected offer values")
            return text
        except Exception:
            return fallback_text

    @staticmethod
    def _clean_message(raw: str) -> str:
        text = raw.strip().strip('"').strip("'")
        text = re.sub("^message\\s*:\\s*", "", text, flags=re.IGNORECASE).strip()
        text = " ".join(text.split())
        words = text.split()
        if len(words) > 28:
            text = " ".join(words[:28]).rstrip(" ,;:")
            if not text.endswith((".", "!", "?")):
                text += "."
        return text

    @classmethod
    def _has_unsupported_number(cls, text: str, context: dict[str, Any]) -> bool:
        numbers = re.findall("\\d+(?:\\.\\d+)?", text)
        if not numbers:
            return False
        allowed: set[str] = set()
        selected_offer = context.get("selected_offer")
        if isinstance(selected_offer, dict):
            for value in selected_offer.values():
                allowed.update(cls._numeric_forms(str(value)))
        for point in context.get("changed_points", []) or []:
            allowed.update(cls._numeric_forms(str(point)))
        return any(
            (not cls._numeric_forms(number).intersection(allowed) for number in numbers)
        )

    @staticmethod
    def _numeric_forms(text: str) -> set[str]:
        forms: set[str] = set()
        for number in re.findall("\\d+(?:\\.\\d+)?", text):
            forms.add(number)
            try:
                value = float(number)
            except ValueError:
                continue
            forms.add(str(value))
            if value.is_integer():
                forms.add(str(int(value)))
        return forms


class OpponentUtilityEstimator:
    def __init__(self) -> None:
        self.smoothing = 1.0
        self.decay = 0.9
        self.neutral_utility = 0.5
        self.issue_stability_bonus = 0.05
        self.min_history_for_issue_weights = 2
        self.anchor_first_weight = 0.5
        self.anchor_last_weight = 0.5
        self.weight_schedule: tuple[tuple[int | None, dict[str, float]], ...] = (
            (
                0,
                {"history_score": 0.0, "anchor_score": 0.0, "inverse_prior_score": 1.0},
            ),
            (
                2,
                {
                    "history_score": 0.35,
                    "anchor_score": 0.45,
                    "inverse_prior_score": 0.2,
                },
            ),
            (
                4,
                {
                    "history_score": 0.55,
                    "anchor_score": 0.35,
                    "inverse_prior_score": 0.1,
                },
            ),
            (
                None,
                {
                    "history_score": 0.7,
                    "anchor_score": 0.25,
                    "inverse_prior_score": 0.05,
                },
            ),
        )
        self.issue_values: dict[int, set[str]] = defaultdict(set)
        self.offer_history: list[tuple[str, ...]] = []
        self.outcome_space: Any | None = None
        self._weighted_counts_cache: dict[int, dict[str, float]] | None = None

    @property
    def total_updates(self) -> int:
        return len(self.offer_history)

    @property
    def first_offer(self) -> OfferKey | None:
        return self.offer_history[0] if self.offer_history else None

    @property
    def last_offer(self) -> OfferKey | None:
        return self.offer_history[-1] if self.offer_history else None

    def set_outcome_space(self, outcome_space: Any) -> None:
        self.outcome_space = outcome_space
        for issue_index, issue in enumerate(getattr(outcome_space, "issues", ())):
            for value in getattr(issue, "values", ()):
                self.issue_values[issue_index].add(str(value))

    def update(self, offer: Outcome | None) -> None:
        if offer is None:
            return
        normalized = tuple((str(value) for value in offer))
        self.offer_history.append(normalized)
        self._weighted_counts_cache = None
        for issue_index, value in enumerate(normalized):
            self.issue_values[issue_index].add(value)

    def __call__(self, offer: Outcome | None) -> float:
        return self.estimate(offer)

    def estimate(
        self, offer: Outcome | None, my_utility_fn: UtilityFn | None = None
    ) -> float:
        if offer is None:
            return self.neutral_utility
        weights = self.mixed_weights()
        scores = {
            "history_score": self.history_score(offer),
            "anchor_score": self.anchor_score(offer),
            "inverse_prior_score": self.inverse_prior_score(offer, my_utility_fn),
        }
        return clip01(sum((weights[name] * scores[name] for name in weights)))

    def mixed_weights(self) -> dict[str, float]:
        n = self.total_updates
        for max_updates, weights in self.weight_schedule:
            if max_updates is None or n <= max_updates:
                return normalize_weights(weights)
        return normalize_weights(self.weight_schedule[-1][1])

    def history_score(self, offer: Outcome | None) -> float:
        if offer is None:
            return self.neutral_utility
        issue_weights = self.issue_weights()
        if not issue_weights:
            return self.neutral_utility
        counts = self.weighted_counts()
        score = 0.0
        for issue_index, value in enumerate(offer):
            score += issue_weights.get(issue_index, 0.0) * self.value_score(
                counts, issue_index, str(value)
            )
        return clip01(score)

    def weighted_counts(self) -> dict[int, dict[str, float]]:
        if self._weighted_counts_cache is not None:
            return self._weighted_counts_cache
        counts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for age, offer in enumerate(reversed(self.offer_history)):
            weight = self.decay**age
            for issue_index, value in enumerate(offer):
                counts[issue_index][str(value)] += weight
        self._weighted_counts_cache = counts
        return counts

    def value_score(
        self, counts_by_issue: dict[int, dict[str, float]], issue_index: int, value: str
    ) -> float:
        if not self.offer_history:
            return self.neutral_utility
        counts = counts_by_issue.get(issue_index, {})
        if not counts:
            return self.neutral_utility
        max_count = max(counts.values(), default=0.0)
        return clip01(
            (counts.get(value, 0.0) + self.smoothing) / (max_count + self.smoothing)
        )

    def issue_weights(self) -> dict[int, float]:
        issue_indices = sorted(self.issue_values)
        if not issue_indices:
            return {}
        if len(self.offer_history) < self.min_history_for_issue_weights:
            return {
                issue_index: 1.0 / len(issue_indices) for issue_index in issue_indices
            }
        comparable_turns = max(1, len(self.offer_history) - 1)
        raw = {}
        for issue_index in issue_indices:
            changes = 0
            for previous, current in zip(self.offer_history, self.offer_history[1:]):
                if issue_index >= len(previous) or issue_index >= len(current):
                    continue
                if previous[issue_index] != current[issue_index]:
                    changes += 1
            raw[issue_index] = (
                1.0 - changes / comparable_turns + self.issue_stability_bonus
            )
        total = sum(raw.values()) or 1.0
        return {issue_index: value / total for issue_index, value in raw.items()}

    def anchor_score(self, offer: Outcome | None) -> float:
        first = self.offer_similarity(offer, self.first_offer)
        last = self.offer_similarity(offer, self.last_offer)
        return clip01(self.anchor_first_weight * first + self.anchor_last_weight * last)

    def offer_similarity(
        self, offer: Outcome | None, reference_offer: OfferKey | None
    ) -> float:
        if offer is None or reference_offer is None:
            return self.neutral_utility
        offer_values = tuple((str(value) for value in offer))
        if not offer_values:
            return self.neutral_utility
        matches = sum(
            (
                1
                for issue_index, value in enumerate(offer_values)
                if issue_index < len(reference_offer)
                and value == reference_offer[issue_index]
            )
        )
        return clip01(matches / len(offer_values))

    def inverse_prior_score(
        self, offer: Outcome | None, my_utility_fn: UtilityFn | None = None
    ) -> float:
        if offer is None or my_utility_fn is None:
            return self.neutral_utility
        try:
            return clip01(1.0 - float(my_utility_fn(offer)))
        except Exception:
            return self.neutral_utility


class OfferSelectionPolicy:
    def __init__(self) -> None:
        self.pareto_epsilon = 0.03
        self.relaxed_target_drop = 0.05
        self.novelty_weight = 0.15
        self.balanced_opponent_weight = 0.4
        self.balanced_target_weight = 0.3

    @staticmethod
    def near_pareto_filter(
        candidates: list[EvaluatedOffer], epsilon: float
    ) -> list[EvaluatedOffer]:
        frontier = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if other is candidate:
                    continue
                no_worse = (
                    other.my_norm >= candidate.my_norm - epsilon
                    and other.opp_u >= candidate.opp_u - epsilon
                )
                clearly_better = (
                    other.my_norm > candidate.my_norm + epsilon
                    or other.opp_u > candidate.opp_u + epsilon
                )
                if no_worse and clearly_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)
        return frontier

    @staticmethod
    def novelty_score(
        candidate: EvaluatedOffer, recent_offer_keys: set[OfferKey]
    ) -> float:
        if not recent_offer_keys:
            return 1.0
        key = offer_key(candidate.offer)
        if not key:
            return 0.0
        distances = []
        for recent_key in recent_offer_keys:
            width = max(len(key), len(recent_key), 1)
            different = sum(
                (
                    1
                    for index in range(width)
                    if (key[index] if index < len(key) else None)
                    != (recent_key[index] if index < len(recent_key) else None)
                )
            )
            distances.append(different / width)
        return clip01(min(distances, default=1.0))

    def select_at_or_above_target(
        self,
        all_candidates: list[EvaluatedOffer],
        target_my_norm: float,
        reserved_value: float,
        safety_margin: float,
        recent_offer_keys: set[OfferKey] | None = None,
    ) -> EvaluatedOffer | None:
        floor = clip01(reserved_value + safety_margin)
        target = clip01(max(target_my_norm, floor))
        recent_offer_keys = recent_offer_keys or set()
        pareto_source = [
            candidate
            for candidate in all_candidates
            if candidate.my_norm > reserved_value
        ]
        pareto_candidates = self.near_pareto_filter(pareto_source, self.pareto_epsilon)

        def without_exact_recent(
            candidates: list[EvaluatedOffer],
        ) -> list[EvaluatedOffer]:
            if not recent_offer_keys:
                return candidates
            return [
                candidate
                for candidate in candidates
                if offer_key(candidate.offer) not in recent_offer_keys
            ]

        def target_closeness(candidate: EvaluatedOffer, local_target: float) -> float:
            return clip01(1.0 - abs(candidate.my_norm - local_target))

        def select_near_target(
            candidates: list[EvaluatedOffer], local_target: float
        ) -> EvaluatedOffer:
            return max(
                candidates,
                key=lambda candidate: (
                    target_closeness(candidate, local_target),
                    candidate.opp_u,
                    self.novelty_score(candidate, recent_offer_keys),
                ),
            )

        def select_balanced(
            candidates: list[EvaluatedOffer], local_target: float
        ) -> EvaluatedOffer:
            return max(
                candidates,
                key=lambda candidate: (
                    self.balanced_opponent_weight * candidate.opp_u
                    + self.balanced_target_weight
                    * target_closeness(candidate, local_target)
                    + self.novelty_weight
                    * self.novelty_score(candidate, recent_offer_keys)
                ),
            )

        relaxed_target = clip01(max(floor, target - self.relaxed_target_drop))
        above_reserved = [
            candidate
            for candidate in all_candidates
            if candidate.my_norm > reserved_value
        ]
        tiers: tuple[tuple[list[EvaluatedOffer], float, bool], ...] = (
            ([c for c in pareto_candidates if c.my_norm >= target], target, True),
            (
                [c for c in pareto_candidates if c.my_norm >= relaxed_target],
                relaxed_target,
                True,
            ),
            (pareto_candidates, relaxed_target, False),
            ([c for c in all_candidates if c.my_norm >= floor], relaxed_target, False),
            (above_reserved, relaxed_target, False),
        )
        for candidates, local_target, near_target in tiers:
            candidates = without_exact_recent(candidates)
            if candidates:
                selector = select_near_target if near_target else select_balanced
                return selector(candidates, local_target)
        return (
            select_balanced(above_reserved, relaxed_target) if above_reserved else None
        )


class T2Agent(SAOCallNegotiator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.enable_llm_response = bool(kwargs.pop("enable_llm_response", True))
        self.safety_margin = 0.05
        self.risk_factor = 1.0
        self.default_risk_time_power = 1.0
        self.profile_time_powers: dict[OpponentProfile, float] = {
            OpponentProfile.CONCEDER: 1.208,
            OpponentProfile.BOULWARE: 1.0825,
            OpponentProfile.COOPERATIVE_HUMAN: 1.0,
            OpponentProfile.VOLATILE: 0.909,
            OpponentProfile.HARDLINER: 0.844,
        }
        self.recent_offer_avoidance_window = 3
        self.offer_gap_adjustment_weight = 0.1
        self.accept_next_epsilon = 0.01
        self.max_target_increase_per_turn = 0.02
        self.min_history_for_profile = 3
        self.profile_window_size = 4
        self.volatility_variance_threshold = 0.01
        self.concession_delta_threshold = 0.04
        self.hardliner_delta_threshold = -0.01
        self.boulware_time_threshold = 0.65
        self.stable_delta_threshold = 0.02
        self.concession_window_size = 3
        self.concession_stall_threshold = 0.02
        self.concession_adjustment_slope = 5.0
        self.concession_risk_increase_limit = 0.25
        self.concession_risk_decrease_limit = 0.15
        self.mode_exploit_max_risk = 0.3
        self.mode_balanced_max_risk = 0.6
        self.mode_agreement_max_risk = 0.85
        self.end_negotiation_risk_threshold = 0.95
        self.accept_response_text = "This offer works for me. I accept it."
        self.end_negotiation_text = (
            "It seems we are unable to reach an acceptable agreement."
        )
        self.exception_fallback_text = (
            "I need to make a counteroffer that better fits my constraints."
        )
        self.fallback_response_texts: dict[Mode, str] = {
            "exploit": "This condition is important for me, so I would like to keep it for now.",
            "balanced": "I adjusted some parts of the offer while keeping my key constraints, so I hope this is easier for both sides.",
            "agreement": "It seems we may have difficulty reaching agreement, so I adjusted the offer to make it more acceptable.",
            "salvage": "To avoid ending without agreement, I made a stronger adjustment while still staying within my minimum requirements.",
        }
        self.response_tones: dict[Mode, str] = {
            "exploit": "firm but cooperative",
            "balanced": "balanced and constructive",
            "agreement": "cooperative and agreement-seeking",
            "salvage": "urgent but respectful",
        }
        super().__init__(*args, **kwargs)
        self.offer_selection: OfferSelectionPolicy = OfferSelectionPolicy()
        self.response_policy: LLMResponseTextPolicy = LLMResponseTextPolicy()
        self._init_runtime_state()
        if self.nmi and getattr(self.nmi, "outcome_space", None) is not None:
            self.opponent_estimator.set_outcome_space(self.nmi.outcome_space)
            self._publish_opponent_model()

    def on_negotiation_start(self, state: SAOState) -> None:
        try:
            super().on_negotiation_start(state)
        except AttributeError:
            pass
        self._reset_runtime()
        self._ensure_runtime_ready()

    def on_negotiation_end(self, state: SAOState) -> None:
        try:
            super().on_negotiation_end(state)
        except AttributeError:
            pass

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        try:
            self._ensure_runtime_ready()
            current_offer: Outcome | None = state.current_offer
            received_text = self._extract_received_text(state)
            if current_offer is not None:
                self._observe_opponent_offer(state, current_offer)
                self._append_history(
                    actor="opponent",
                    state=state,
                    offer=current_offer,
                    message=received_text,
                )
            risk = self._assess_risk(state=state)
            mode = self._choose_mode(risk)
            accept_threshold = self._compute_risk_adjusted_accept_threshold(risk)
            current_my_u = (
                self._utility(current_offer) if current_offer is not None else None
            )
            _, _, reserved = self._utility_bounds()
            proposal_target_norm = self._compute_proposal_target_norm(risk)
            selected = self.offer_selection.select_at_or_above_target(
                all_candidates=self._evaluate_all_offers(),
                target_my_norm=proposal_target_norm,
                reserved_value=self._reserved_norm(),
                safety_margin=self.safety_margin,
                recent_offer_keys=self._recent_offer_keys(),
            )
            acceptable_offer = (
                current_offer is not None
                and current_my_u is not None
                and (current_my_u > reserved)
            )
            accept_by_threshold = acceptable_offer and current_my_u >= accept_threshold
            accept_by_next_offer = (
                acceptable_offer
                and selected is not None
                and (current_my_u + self.accept_next_epsilon >= selected.my_u)
            )
            if accept_by_threshold or accept_by_next_offer:
                return self._make_response(
                    ResponseType.ACCEPT_OFFER, current_offer, self.accept_response_text
                )
            if selected is None:
                fallback = self._fallback_offer_above_reserved()
                if fallback is None or self._should_end_negotiation(risk):
                    return self._make_response(
                        ResponseType.END_NEGOTIATION, None, self.end_negotiation_text
                    )
                return self._counter_response(state, fallback, mode, received_text)
            return self._counter_response(state, selected.offer, mode, received_text)
        except Exception:
            return self._make_response(
                ResponseType.REJECT_OFFER,
                self._fallback_offer(),
                self.exception_fallback_text,
            )

    def _init_runtime_state(self) -> None:
        self.opponent_estimator: OpponentUtilityEstimator = OpponentUtilityEstimator()
        self._outcomes_cache: list[Outcome] | None = None
        self._utility_cache: dict[str, float] | None = None
        self._candidate_cache: list[EvaluatedOffer] | None = None
        self._last_seen_offer_key: tuple[int, str] | None = None
        self._last_my_offer: Outcome | None = None
        self._last_effective_risk_time_power: float = self.default_risk_time_power
        self._self_offer_history: list[OfferKey] = []
        self._last_combined_risk: float | None = None
        self._last_proposal_target_norm: float | None = None
        self._opponent_offer_my_norm_history: list[float] = []
        self._opponent_offer_estimated_opp_u_history: list[float] = []
        self._negotiation_history: list[dict[str, Any]] = []
        self._publish_opponent_model()

    def _reset_runtime(self) -> None:
        self._init_runtime_state()
        self._publish_opponent_model()

    @staticmethod
    def _make_response(
        response_type: ResponseType, outcome: Outcome | None, text: str
    ) -> SAOResponse:
        return SAOResponse(response_type, outcome=outcome, data={"text": text})

    def _counter_response(
        self, state: SAOState, offer: Outcome, mode: Mode, received_text: str | None
    ) -> SAOResponse:
        self._last_my_offer = offer
        self._self_offer_history.append(offer_key(offer))
        text = self._response_text(
            mode=mode, selected_offer=offer, received_text=received_text
        )
        self._append_history(actor="self", state=state, offer=offer, message=text)
        return self._make_response(ResponseType.REJECT_OFFER, offer, text)

    def _ensure_runtime_ready(self) -> None:
        if self.nmi and getattr(self.nmi, "outcome_space", None) is not None:
            self.opponent_estimator.set_outcome_space(self.nmi.outcome_space)
            self._publish_opponent_model()

    def _publish_opponent_model(self) -> None:
        if hasattr(self, "private_info") and self.private_info is not None:
            self.private_info["opponent_ufun"] = self.opponent_estimator

    def _observe_opponent_offer(self, state: SAOState, offer: Outcome) -> None:
        key = (state.step, str(offer))
        if key == self._last_seen_offer_key:
            return
        self._last_seen_offer_key = key
        self.opponent_estimator.update(offer)
        self._publish_opponent_model()
        self._opponent_offer_my_norm_history.append(
            self._normalize_utility(self._utility(offer))
        )
        self._opponent_offer_estimated_opp_u_history.append(
            self._estimate_opponent_utility(offer)
        )
        self._candidate_cache = None

    def _get_outcomes(self) -> list[Outcome]:
        if self._outcomes_cache is None:
            self._outcomes_cache = list(self.nmi.outcome_space.enumerate())
        return self._outcomes_cache

    def _utility_bounds(self) -> tuple[float, float, float]:
        reserved = float(self.ufun.reserved_value) if self.ufun else 0.0
        if self._utility_cache is None:
            values = []
            for outcome in self._get_outcomes():
                try:
                    values.append(float(self.ufun(outcome)))
                except Exception:
                    pass
            self._utility_cache = {
                "min": min(values) if values else reserved,
                "max": max(values) if values else max(1.0, reserved),
                "reserved": reserved,
            }
        return (
            float(self._utility_cache["min"]),
            float(self._utility_cache["max"]),
            float(self._utility_cache["reserved"]),
        )

    def _utility(self, outcome: Outcome | None) -> float:
        if outcome is None or self.ufun is None:
            return float(self.ufun.reserved_value) if self.ufun else 0.0
        try:
            return float(self.ufun(outcome))
        except Exception:
            return float(self.ufun.reserved_value)

    def _normalize_utility(self, utility: float) -> float:
        min_u, max_u, _ = self._utility_bounds()
        return clip01((float(utility) - min_u) / max(max_u - min_u, 1e-09))

    def _denormalize_utility(self, normalized: float) -> float:
        min_u, max_u, _ = self._utility_bounds()
        return min_u + normalized * (max_u - min_u)

    def _reserved_norm(self) -> float:
        _, _, reserved = self._utility_bounds()
        return self._normalize_utility(reserved)

    def _estimate_opponent_utility(self, outcome: Outcome | None) -> float:
        return self.opponent_estimator.estimate(
            outcome,
            my_utility_fn=lambda offer: self._normalize_utility(self._utility(offer)),
        )

    def _assess_risk(self, state: SAOState) -> float:
        opponent_profile = self._heuristic_opponent_profile()
        self._last_effective_risk_time_power = self.profile_time_powers.get(
            opponent_profile, self.default_risk_time_power
        )
        self._last_combined_risk = self._estimate_breakdown_risk(state)
        return self._last_combined_risk

    def _heuristic_opponent_profile(self) -> OpponentProfile:
        history = self._opponent_offer_my_norm_history
        if len(history) < self.min_history_for_profile:
            return OpponentProfile.COOPERATIVE_HUMAN
        window = history[-self.profile_window_size :]
        deltas = [b - a for a, b in zip(window, window[1:])]
        avg_delta = sum(deltas) / len(deltas)
        variance = sum(((delta - avg_delta) ** 2 for delta in deltas)) / len(deltas)
        relative_time = 0.0
        if self._negotiation_history:
            relative_time = clip01(
                float(self._negotiation_history[-1].get("relative_time") or 0.0)
            )
        has_unstable_concession_size = variance > self.volatility_variance_threshold
        if has_unstable_concession_size:
            return OpponentProfile.VOLATILE
        if avg_delta > self.concession_delta_threshold:
            return OpponentProfile.CONCEDER
        if avg_delta < self.hardliner_delta_threshold:
            return OpponentProfile.HARDLINER
        if (
            relative_time < self.boulware_time_threshold
            and abs(avg_delta) < self.stable_delta_threshold
        ):
            return OpponentProfile.BOULWARE
        return OpponentProfile.COOPERATIVE_HUMAN

    def _append_history(
        self, *, actor: str, state: SAOState, offer: Outcome | None, message: str | None
    ) -> None:
        utility = self._utility(offer) if offer is not None else None
        entry: dict[str, Any] = {
            "actor": actor,
            "step": state.step,
            "relative_time": state.relative_time,
            "offer": str(offer) if offer is not None else None,
            "message": message,
            "my_utility": utility,
        }
        self._negotiation_history.append(entry)

    def _compute_risk_adjusted_accept_threshold(self, risk: float) -> float:
        reserved = self._reserved_norm()
        threshold_norm = max(reserved, 1.0 - self.risk_factor * risk)
        return self._denormalize_utility(clip01(threshold_norm))

    def _compute_proposal_target_norm(self, risk: float) -> float:
        reserved = self._reserved_norm()
        raw_target = clip01(
            max(reserved + self.safety_margin, 1.0 - self.risk_factor * clip01(risk))
        )
        target = raw_target
        if self._last_proposal_target_norm is not None:
            max_allowed = clip01(
                self._last_proposal_target_norm + self.max_target_increase_per_turn
            )
            target = min(raw_target, max_allowed)
        target = clip01(max(reserved + self.safety_margin, target))
        self._last_proposal_target_norm = target
        return target

    def _estimate_breakdown_risk(self, state: SAOState) -> float:
        opponent_offer = state.current_offer
        relative_time = state.relative_time
        base_time_risk = clip01(relative_time**self._last_effective_risk_time_power)
        offer_gap = 0.0
        if opponent_offer is not None and self._last_my_offer is not None:
            offer_gap = abs(
                self._estimate_opponent_utility(opponent_offer)
                - self._estimate_opponent_utility(self._last_my_offer)
            )
        offer_gap_adjustment = self.offer_gap_adjustment_weight * clip01(offer_gap)
        concession_adjustment = 0.0
        if len(self._opponent_offer_my_norm_history) >= self.concession_window_size:
            window_size = min(
                self.concession_window_size,
                len(self._opponent_offer_my_norm_history) - 1,
            )
            latest = self._opponent_offer_my_norm_history[-1]
            previous = self._opponent_offer_my_norm_history[-1 - window_size : -1]
            previous_avg = sum(previous) / len(previous)
            concession_delta = latest - previous_avg
            effective_delta = concession_delta - self.concession_stall_threshold
            concession_adjustment = max(
                -self.concession_risk_decrease_limit,
                min(
                    self.concession_risk_increase_limit,
                    -self.concession_adjustment_slope * effective_delta,
                ),
            )
        return clip01(base_time_risk + concession_adjustment + offer_gap_adjustment)

    def _choose_mode(self, risk: float) -> Mode:
        if risk < self.mode_exploit_max_risk:
            return "exploit"
        if risk < self.mode_balanced_max_risk:
            return "balanced"
        if risk < self.mode_agreement_max_risk:
            return "agreement"
        return "salvage"

    def _evaluate_all_offers(self) -> list[EvaluatedOffer]:
        if self._candidate_cache is not None:
            return self._candidate_cache
        evaluated: list[EvaluatedOffer] = []
        for outcome in self._get_outcomes():
            my_u = self._utility(outcome)
            my_norm = self._normalize_utility(my_u)
            base_opp_u = self._estimate_opponent_utility(outcome)
            evaluated.append(
                EvaluatedOffer(
                    offer=outcome,
                    my_u=my_u,
                    my_norm=my_norm,
                    base_opp_u=base_opp_u,
                    opp_u=base_opp_u,
                    low_cost_concession_score=0.0,
                )
            )
        self._candidate_cache = evaluated
        return evaluated

    def _fallback_offer_above_reserved(self) -> Outcome | None:
        _, _, reserved = self._utility_bounds()
        candidates = [
            outcome
            for outcome in self._get_outcomes()
            if self._utility(outcome) > reserved
        ]
        return max(candidates, key=self._utility, default=None)

    def _should_end_negotiation(self, risk: float) -> bool:
        if clip01(risk) < self.end_negotiation_risk_threshold:
            return False
        return self._fallback_offer_above_reserved() is None

    def _fallback_offer(self) -> Outcome | None:
        try:
            best = self.ufun.best()
            if best is not None:
                return best
        except Exception:
            pass
        outcomes = self._get_outcomes()
        return outcomes[0] if outcomes else None

    def _recent_offer_keys(self) -> set[OfferKey]:
        return set(self._self_offer_history[-self.recent_offer_avoidance_window :])

    def _extract_received_text(self, state: SAOState) -> str | None:
        if state.current_data and state.current_data.get("text"):
            return str(state.current_data["text"])
        for _, item in reversed(state.new_data):
            if item and item.get("text"):
                return str(item["text"])
        return None

    def _response_text(
        self, *, mode: Mode, selected_offer: Outcome | None, received_text: str | None
    ) -> str:
        fallback_text = self._fallback_response_text(mode)
        if not self.enable_llm_response:
            return fallback_text
        context = {
            "tone": self._response_tone(mode),
            "selected_offer": self._offer_as_issue_dict(selected_offer),
            "opponent_likely_priority": self._likely_opponent_priority_issue(),
            "concession_made": self._main_concession_issue(selected_offer),
            "changed_points": self._changed_points(selected_offer),
            "latest_opponent_message": (received_text or "")[:180],
        }
        return self.response_policy.generate(
            context=context, fallback_text=fallback_text
        )

    def _fallback_response_text(self, mode: Mode) -> str:
        return self.fallback_response_texts[mode]

    def _response_tone(self, mode: Mode) -> str:
        return self.response_tones[mode]

    def _issue_context(self) -> list[dict[str, str | int]]:
        outcome_space = getattr(self.nmi, "outcome_space", None) if self.nmi else None
        issues = getattr(outcome_space, "issues", ()) if outcome_space else ()
        return [
            {
                "index": issue_index,
                "issue": str(getattr(issue, "name", f"issue_{issue_index}")),
            }
            for issue_index, issue in enumerate(issues)
        ]

    def _offer_as_issue_dict(self, offer: Outcome | None) -> dict[str, str]:
        if offer is None:
            return {}
        result = {}
        for item in self._issue_context():
            issue_index = int(item["index"])
            try:
                result[str(item["issue"])] = str(offer[issue_index])
            except Exception:
                continue
        return result

    def _likely_opponent_priority_issue(self) -> str | None:
        importance = self.opponent_estimator.issue_weights()
        if not importance:
            return None
        issue_index = max(importance, key=importance.get)
        for item in self._issue_context():
            if int(item["index"]) == issue_index:
                return str(item["issue"])
        return f"issue_{issue_index}"

    def _self_issue_importance(self) -> dict[int, float]:
        outcomes = self._get_outcomes()
        issue_count = len(getattr(self.nmi.outcome_space, "issues", ()) or ())
        if not outcomes or issue_count <= 0:
            return {}
        ranges: dict[int, float] = {}
        for issue_index in range(issue_count):
            by_value: dict[str, list[float]] = defaultdict(list)
            for outcome in outcomes:
                try:
                    value = str(outcome[issue_index])
                except Exception:
                    continue
                by_value[value].append(self._normalize_utility(self._utility(outcome)))
            means = [
                sum(values) / len(values) for values in by_value.values() if values
            ]
            ranges[issue_index] = max(means) - min(means) if means else 0.0
        return normalize_weights(ranges)

    def _main_concession_issue(self, selected_offer: Outcome | None) -> str | None:
        if selected_offer is None or self._last_my_offer is None:
            return self._likely_opponent_priority_issue()
        self_importance = self._self_issue_importance()
        changed: list[tuple[float, str]] = []
        for item in self._issue_context():
            issue_index = int(item["index"])
            try:
                if str(selected_offer[issue_index]) == str(
                    self._last_my_offer[issue_index]
                ):
                    continue
            except Exception:
                continue
            changed.append((self_importance.get(issue_index, 0.5), str(item["issue"])))
        if not changed:
            return self._likely_opponent_priority_issue()
        return min(changed, key=lambda item: item[0])[1]

    def _changed_points(self, selected_offer: Outcome | None) -> list[str]:
        if selected_offer is None or self._last_my_offer is None:
            priority = self._likely_opponent_priority_issue()
            return (
                [f"I considered your likely priority on {priority}."]
                if priority
                else []
            )
        points: list[str] = []
        for item in self._issue_context():
            issue_index = int(item["index"])
            try:
                old_value = str(self._last_my_offer[issue_index])
                new_value = str(selected_offer[issue_index])
            except Exception:
                continue
            if old_value != new_value:
                points.append(f"{item['issue']}: {old_value} -> {new_value}")
        return points[:3]
