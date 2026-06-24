from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, median, pstdev
from time import perf_counter
from typing import Any

from negmas.common import Outcome
from negmas.preferences import MappingUtilityFunction
from negmas.sao.common import SAOResponse, SAOState, ResponseType
from negmas.sao.negotiators.base import SAOCallNegotiator


@dataclass(frozen=True)
class OutcomeStats:
    outcome: Outcome
    raw_utility: float
    utility: float
    values: tuple[Any, ...]
    normalized_values: tuple[float, ...]
    bucket: int


@dataclass(frozen=True)
class AcceptanceDecision:
    accept: bool
    reason: str
    floor: float
    offer_utility: float
    next_utility: float
    target: float
    opponent_view: float
    next_opponent_view: float


@dataclass(frozen=True)
class EndDecision:
    end: bool
    reason: str
    floor: float
    offer_utility: float | None
    next_utility: float
    next_acceptance_fit: float
    agreement_risk: float


@dataclass(frozen=True)
class NextOfferPrediction:
    created_turn: int
    relative_time: float
    expected_self_utility: float
    lower_self_utility: float
    upper_self_utility: float
    repeat_probability: float
    concession_probability: float
    confidence: float
    last_offer: Outcome | None
    last_self_utility: float | None


@dataclass(frozen=True)
class JitterProfile:
    enabled: bool
    seed: int | None
    values: dict[str, float]


@dataclass(frozen=True)
class TextSignals:
    text_count: int
    urgency: float = 0.0
    flexibility: float = 0.0
    fairness: float = 0.0
    firmness: float = 0.0
    issue_mentions: tuple[str, ...] = ()
    value_mentions: int = 0

    @property
    def has_signal(self) -> bool:
        return bool(
            self.text_count
            or self.urgency
            or self.flexibility
            or self.fairness
            or self.firmness
            or self.issue_mentions
            or self.value_mentions
        )

    def trace(self) -> dict[str, Any]:
        return {
            "text_count": self.text_count,
            "urgency": round(self.urgency, 6),
            "flexibility": round(self.flexibility, 6),
            "fairness": round(self.fairness, 6),
            "firmness": round(self.firmness, 6),
            "issue_mentions": list(self.issue_mentions),
            "value_mentions": self.value_mentions,
        }


class OpponentTextModel:
    """Lightweight non-LLM parser for human text attached to offers."""

    URGENCY_WORDS = {
        "urgent",
        "quick",
        "quickly",
        "soon",
        "deadline",
        "final",
        "last",
        "time",
        "hurry",
    }
    FLEXIBILITY_WORDS = {
        "flexible",
        "compromise",
        "adjust",
        "move",
        "open",
        "willing",
        "trade",
        "workable",
        "esnek",
        "taviz",
    }
    FAIRNESS_WORDS = {
        "fair",
        "balanced",
        "mutual",
        "reasonable",
        "equal",
        "together",
        "ortak",
        "adil",
        "denge",
    }
    FIRMNESS_WORDS = {
        "cannot",
        "can't",
        "need",
        "must",
        "firm",
        "unacceptable",
        "no",
        "not",
        "olmaz",
        "zor",
        "gerek",
    }

    def __init__(self) -> None:
        self.issue_names: list[str] = []
        self.history: list[TextSignals] = []

    def configure(self, issue_names: list[str]) -> None:
        self.issue_names = list(issue_names)

    def observe(self, texts: list[str]) -> TextSignals:
        signals = self._parse(texts)
        if signals.has_signal:
            self.history.append(signals)
            if len(self.history) > 20:
                self.history = self.history[-20:]
        return signals

    def adjust_behavior(self, behavior: str, signals: TextSignals) -> str:
        if not signals.has_signal:
            return behavior
        if signals.firmness >= 0.65 and signals.flexibility < 0.35:
            return "hardliner"
        if signals.flexibility >= 0.65 and behavior == "hardliner":
            return "reciprocal"
        if signals.fairness >= 0.65 and behavior in {"erratic", "hardliner"}:
            return "fair-seeking"
        return behavior

    def _parse(self, texts: list[str]) -> TextSignals:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if not cleaned:
            return TextSignals(text_count=0)
        joined = " ".join(cleaned).lower()
        words = set(re.findall(r"[^\W\d_]+", joined, flags=re.UNICODE))

        def score(keywords: set[str]) -> float:
            hits = sum(1 for keyword in keywords if keyword in words or keyword in joined)
            return min(1.0, hits / 2.0)

        issue_mentions = []
        for name in self.issue_names:
            if name and name.lower() in joined:
                issue_mentions.append(name)
        values = re.findall(r"(?<![a-zA-Z])[-+]?\d+(?:\.\d+)?(?![a-zA-Z])", joined)
        return TextSignals(
            text_count=len(cleaned),
            urgency=score(self.URGENCY_WORDS),
            flexibility=score(self.FLEXIBILITY_WORDS),
            fairness=score(self.FAIRNESS_WORDS),
            firmness=score(self.FIRMNESS_WORDS),
            issue_mentions=tuple(issue_mentions[:4]),
            value_mentions=len(values),
        )


class OpponentModel:
    """Recency-weighted frequency model with numeric distance smoothing."""

    def __init__(
        self,
        issue_count: int,
        decay: float = 0.92,
        issue_weight_mode: str = "legacy",
        slope_mode: str = "legacy",
        uncertainty_mode: str = "off",
        bayes_prior: float = 0.65,
    ) -> None:
        self.issue_count = issue_count
        self.decay = decay
        self.issue_weight_mode = issue_weight_mode
        self.slope_mode = slope_mode
        self.uncertainty_mode = uncertainty_mode
        self.bayes_prior = max(0.0, bayes_prior)
        self.counts: list[defaultdict[Any, float]] = [
            defaultdict(float) for _ in range(issue_count)
        ]
        self.history: list[Outcome] = []
        self.self_utility_history: list[float] = []
        self.estimated_utility_history: list[float] = []
        self.first_seen: dict[Outcome, int] = {}
        self.rejected_self_offers: list[Outcome] = []
        self.rejected_self_utilities: list[float] = []

    @property
    def confidence(self) -> float:
        return min(1.0, len(self.history) / 8.0)

    def observe(self, offer: Outcome, self_utility: float) -> None:
        for counts in self.counts:
            for key in list(counts):
                counts[key] *= self.decay
                if counts[key] < 0.001:
                    del counts[key]

        for index, value in enumerate(offer):
            self.counts[index][value] += 1.0

        self.history.append(offer)
        self.self_utility_history.append(self_utility)
        self.estimated_utility_history.append(self.estimate(offer))
        self.first_seen.setdefault(offer, len(self.history) - 1)

    def observe_rejection(self, offer: Outcome, self_utility: float) -> None:
        self.rejected_self_offers.append(offer)
        self.rejected_self_utilities.append(self_utility)
        if len(self.rejected_self_offers) > 32:
            self.rejected_self_offers = self.rejected_self_offers[-32:]
            self.rejected_self_utilities = self.rejected_self_utilities[-32:]

    def estimate(self, outcome: Outcome) -> float:
        weights = self.issue_weights()
        scores = [
            self.value_score(index, value) * weights[index]
            for index, value in enumerate(outcome)
        ]
        return min(1.0, max(0.0, sum(scores)))

    def recent_estimate(self, outcome: Outcome, window: int = 8) -> float:
        recent = self.history[-max(1, window) :]
        if not recent:
            return self.estimate(outcome)
        counts: list[defaultdict[Any, float]] = [
            defaultdict(float) for _ in range(self.issue_count)
        ]
        for offer in recent:
            for index, value in enumerate(offer):
                counts[index][value] += 1.0
        weights = self._issue_weights_from_counts(counts)
        scores = [
            self._value_score_from_counts(counts[index], value) * weights[index]
            for index, value in enumerate(outcome)
        ]
        return min(1.0, max(0.0, sum(scores)))

    def value_score(self, index: int, value: Any) -> float:
        return self._value_score_from_counts(self.counts[index], value)

    def _value_score_from_counts(
        self, counts: defaultdict[Any, float], value: Any
    ) -> float:
        total = sum(counts.values())
        if total <= 0.0:
            return 0.5

        exact = counts[value] / total
        distance = 0.0
        if isinstance(value, (int, float)):
            distance = sum(
                count / (1.0 + abs(float(value) - float(observed)))
                for observed, count in counts.items()
                if isinstance(observed, (int, float))
            ) / total
        score = max(exact, distance)
        if self.uncertainty_mode != "bayes_light":
            return score

        support = max(1, len(counts) + (0 if value in counts else 1))
        prior_mass = self.bayes_prior * support
        smoothed_exact = (counts[value] + self.bayes_prior) / max(
            0.0001, total + prior_mass
        )
        score = max(score, smoothed_exact)
        certainty = self._issue_certainty_from_counts(counts)
        return max(0.0, min(1.0, 0.5 + (score - 0.5) * certainty))

    def issue_weights(self) -> list[float]:
        return self._issue_weights_from_counts(self.counts)

    def _issue_weights_from_counts(
        self, counts_by_issue: list[defaultdict[Any, float]]
    ) -> list[float]:
        if self.issue_count <= 0:
            return []
        if self.issue_weight_mode == "legacy" and self.uncertainty_mode != "bayes_light":
            return self._legacy_issue_weights_from_counts(counts_by_issue)

        uniform = 1.0 / self.issue_count
        raw = []
        for index, counts in enumerate(counts_by_issue):
            total = sum(counts.values())
            if total <= 0.0:
                raw.append(1.0)
                continue
            concentration = max(counts.values()) / total
            probabilities = [count / total for count in counts.values() if count > 0.0]
            if len(probabilities) <= 1:
                entropy = 0.0
            else:
                entropy = -sum(p * math.log(p) for p in probabilities) / math.log(
                    len(probabilities)
                )
            stability = 1.0 - max(0.0, min(1.0, entropy))
            movement = self._recent_issue_movement(index)
            raw.append(0.55 + 0.45 * concentration + 0.30 * stability - 0.18 * movement)

        total = sum(raw)
        if total <= 0.0:
            return [uniform] * self.issue_count
        weights = [value / total for value in raw]
        lower = 0.45 * uniform
        upper = 2.20 * uniform
        bounded = [max(lower, min(upper, weight)) for weight in weights]
        bounded_total = sum(bounded)
        if bounded_total <= 0.0:
            return [uniform] * self.issue_count
        return [value / bounded_total for value in bounded]

    def outcome_uncertainty(self, outcome: Outcome | None = None) -> float:
        if self.issue_count <= 0:
            return 1.0
        certainties = [
            self._issue_certainty_from_counts(counts)
            for counts in self.counts
        ]
        certainty = sum(certainties) / max(1, len(certainties))
        confidence = self.confidence
        return max(0.0, min(1.0, 1.0 - 0.50 * certainty - 0.50 * confidence))

    def _issue_certainty_from_counts(self, counts: defaultdict[Any, float]) -> float:
        total = sum(counts.values())
        if total <= 0.0:
            return 0.0
        support = max(1, len(counts))
        prior = self.bayes_prior * max(1, support)
        return max(0.0, min(1.0, total / (total + prior + 2.0)))

    def _legacy_issue_weights_from_counts(
        self, counts_by_issue: list[defaultdict[Any, float]]
    ) -> list[float]:
        raw = []
        for counts in counts_by_issue:
            total = sum(counts.values())
            if total <= 0.0:
                raw.append(1.0)
                continue
            concentration = max(counts.values()) / total
            raw.append(0.35 + concentration)

        total = sum(raw)
        if total <= 0.0:
            return [1.0 / self.issue_count] * self.issue_count
        return [value / total for value in raw]

    def _recent_issue_movement(self, index: int, window: int = 8) -> float:
        recent = self.history[-max(2, window) :]
        if len(recent) < 2 or index >= self.issue_count:
            return 0.0
        changes = sum(
            1
            for previous, current in zip(recent, recent[1:])
            if previous[index] != current[index]
        )
        return changes / max(1, len(recent) - 1)

    def concession_slope(self) -> float:
        if len(self.self_utility_history) < 4:
            return 0.0
        if self.slope_mode == "legacy":
            first = sum(self.self_utility_history[:2]) / 2.0
            last = sum(self.self_utility_history[-2:]) / 2.0
            return last - first
        if self.slope_mode == "block":
            return self._block_concession_slope()
        if self.slope_mode == "robust":
            return self._robust_concession_slope()
        return self._regression_concession_slope()

    def _block_concession_slope(self) -> float:
        values = self.self_utility_history
        if len(values) < 4:
            return 0.0
        section = max(2, min(5, len(values) // 3))
        first = mean(values[:section])
        last = mean(values[-section:])
        return last - first

    def _regression_concession_slope(self, window: int = 10) -> float:
        values = self.self_utility_history[-max(4, window) :]
        if len(values) < 4:
            return 0.0
        xs = list(range(len(values)))
        x_mean = mean(xs)
        y_mean = mean(values)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator <= 0.0:
            return 0.0
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denominator
        return slope * (len(values) - 1)

    def _robust_concession_slope(self, window: int = 12) -> float:
        values = self.self_utility_history[-max(5, window) :]
        if len(values) < 5:
            return self._regression_concession_slope(window=6)

        pairwise_slopes = [
            (values[j] - values[i]) / (j - i)
            for i in range(len(values))
            for j in range(i + 1, len(values))
            if j > i
        ]
        if not pairwise_slopes:
            return 0.0
        theil_sen_total = median(pairwise_slopes) * (len(values) - 1)

        section = max(2, len(values) // 3)
        first_block = mean(values[:section])
        last_block = mean(values[-section:])
        block_total = last_block - first_block

        diffs = [current - previous for previous, current in zip(values, values[1:])]
        positive = sum(1 for diff in diffs if diff > 0.005)
        negative = sum(1 for diff in diffs if diff < -0.005)
        directional_support = abs(positive - negative) / max(1, positive + negative)
        shrink = 0.60 + 0.40 * directional_support

        blended = 0.65 * theil_sen_total + 0.35 * block_total
        return max(-0.35, min(0.45, blended * shrink))

    def concession_slope_method(self) -> str:
        if self.slope_mode == "legacy":
            return "legacy_first_last"
        if self.slope_mode == "block":
            if len(self.self_utility_history) < 4:
                return "block_insufficient_history"
            return "block_first_last"
        if self.slope_mode == "robust":
            if len(self.self_utility_history) < 5:
                return "robust_insufficient_history"
            return "robust_theil_sen"
        if len(self.self_utility_history) < 4:
            return "insufficient_history"
        return "linear_regression"

    def export_estimate(
        self,
        outcome: Outcome,
        temporal_weight: float = 0.55,
        issue_temporal_weight: float = 0.25,
        frequency_weight: float = 0.20,
        temporal_floor: float = 0.30,
        temporal_power: float = 0.70,
    ) -> float:
        if len(self.first_seen) < 3:
            return self.estimate(outcome)

        distinct = [offer for offer, _ in sorted(self.first_seen.items(), key=lambda item: item[1])]
        span = max(1, len(distinct) - 1)
        temporal_scores = []
        temporal_floor = max(0.0, min(1.0, temporal_floor))
        temporal_power = max(0.05, temporal_power)
        for order, offer in enumerate(distinct):
            progress = order / span
            offered_rank = temporal_floor + (1.0 - temporal_floor) * (
                1.0 - progress
            ) ** temporal_power
            temporal_scores.append(offered_rank * self._outcome_similarity(outcome, offer))
        temporal = max(temporal_scores) if temporal_scores else self.estimate(outcome)
        frequency = self.estimate(outcome)
        issue_temporal = self._issue_temporal_estimate(outcome, distinct, span)
        total = temporal_weight + issue_temporal_weight + frequency_weight
        if total <= 0.0:
            temporal_weight, issue_temporal_weight, frequency_weight = 0.55, 0.25, 0.20
            total = 1.0
        return max(
            0.0,
            min(
                1.0,
                (
                    temporal_weight * temporal
                    + issue_temporal_weight * issue_temporal
                    + frequency_weight * frequency
                )
                / total,
            ),
        )

    def constraint_acceptance_estimate(self, outcome: Outcome) -> float:
        frequency = self.estimate(outcome)
        positive = self._positive_anchor_similarity(outcome)
        rejection = self._rejection_similarity(outcome)
        exploration_bonus = 0.0
        if self.rejected_self_offers:
            best_rejected_utility = max(self.rejected_self_utilities)
            if best_rejected_utility < 0.60:
                exploration_bonus = 0.06
        estimate = 0.45 * frequency + 0.45 * positive - 0.35 * rejection
        return max(0.0, min(1.0, estimate + exploration_bonus))

    def _positive_anchor_similarity(self, outcome: Outcome) -> float:
        if not self.history:
            return 0.5
        scores = []
        span = max(1, len(self.history) - 1)
        counts = Counter(self.history[-8:])
        for index, offer in enumerate(self.history[-8:]):
            recency = 0.50 + 0.50 * (
                (span - min(span, len(self.history) - 1 - index)) / span
            )
            repetition = 0.15 * counts[offer]
            scores.append(
                (recency + repetition) * self._outcome_similarity(outcome, offer)
            )
        return max(scores) if scores else 0.5

    def _rejection_similarity(self, outcome: Outcome) -> float:
        if not self.rejected_self_offers:
            return 0.0
        recent = self.rejected_self_offers[-8:]
        scores = [
            self._outcome_similarity(outcome, rejected)
            for rejected in recent
        ]
        return max(scores) if scores else 0.0

    def _issue_temporal_estimate(self, outcome: Outcome, distinct: list[Outcome], span: int) -> float:
        if not outcome:
            return 0.0
        per_issue: list[defaultdict[Any, float]] = [
            defaultdict(float) for _ in range(len(outcome))
        ]
        for order, offer in enumerate(distinct):
            progress = order / span
            rank = 0.30 + 0.70 * (1.0 - progress) ** 0.70
            for index, value in enumerate(offer):
                per_issue[index][value] = max(per_issue[index][value], rank)

        scores = []
        for index, value in enumerate(outcome):
            exact = per_issue[index].get(value)
            if exact is not None:
                scores.append(exact)
                continue
            numeric_scores = [
                score / (1.0 + abs(float(value) - float(observed)))
                for observed, score in per_issue[index].items()
                if isinstance(value, (int, float)) and isinstance(observed, (int, float))
            ]
            scores.append(max(numeric_scores) if numeric_scores else 0.35)
        return sum(scores) / len(scores)

    def _outcome_similarity(self, first: Outcome, second: Outcome) -> float:
        if not first:
            return 0.0
        scores = []
        for left, right in zip(first, second):
            if left == right:
                scores.append(1.0)
            elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                scores.append(1.0 / (1.0 + abs(float(left) - float(right))))
            else:
                scores.append(0.0)
        return sum(scores) / len(first)


class BehaviorModel:
    def __init__(self, mode: str = "simple") -> None:
        self.mode = mode

    def classify(self, opponent: OpponentModel) -> str:
        if self.mode == "archetype":
            return self._classify_archetype(opponent)
        return self._classify_simple(opponent)

    def _classify_simple(self, opponent: OpponentModel) -> str:
        offers = opponent.history
        if len(offers) < 3:
            return "fair-seeking"

        unique_ratio = len(set(offers[-6:])) / min(len(offers), 6)
        slope = opponent.concession_slope()
        recent_utilities = opponent.self_utility_history[-5:]
        spread = max(recent_utilities) - min(recent_utilities) if recent_utilities else 0.0

        if unique_ratio <= 0.35 and slope < 0.04:
            return "hardliner"
        if spread > 0.35:
            return "erratic"
        if slope > 0.15:
            return "conceder"
        if unique_ratio > 0.65 and slope > 0.04:
            return "reciprocal"
        return "fair-seeking"

    def _classify_archetype(self, opponent: OpponentModel) -> str:
        offers = opponent.history
        utilities = opponent.self_utility_history
        if len(offers) < 3:
            return "fair-seeking"

        recent_offers = offers[-8:]
        recent_utilities = utilities[-8:]
        short_utilities = utilities[-4:]
        unique_ratio = len(set(recent_offers)) / len(recent_offers)
        max_repeat = max(Counter(recent_offers).values())
        repeated_share = max_repeat / len(recent_offers)
        slope = opponent.concession_slope()
        short_slope = short_utilities[-1] - short_utilities[0]
        volatility = pstdev(recent_utilities) if len(recent_utilities) > 1 else 0.0
        deltas = [
            after - before
            for before, after in zip(recent_utilities, recent_utilities[1:])
        ]
        improving = sum(delta > 0.025 for delta in deltas)
        worsening = sum(delta < -0.025 for delta in deltas)
        recent_mean = mean(recent_utilities)

        if volatility > 0.22 and worsening >= 2:
            return "erratic"
        if repeated_share >= 0.50 and slope < 0.06 and short_slope < 0.04:
            return "hardliner"
        if unique_ratio <= 0.40 and recent_mean < 0.70 and slope < 0.08:
            return "hardliner"
        if slope > 0.16 or (short_slope > 0.10 and improving >= 2):
            return "conceder"
        if improving >= 2 and worsening <= 1 and unique_ratio >= 0.55:
            return "reciprocal"
        if volatility > 0.24:
            return "erratic"
        return "fair-seeking"


class BiddingPolicy:
    def __init__(
        self,
        early_target: float = 0.94,
        late_target: float = 0.45,
        reservation_margin: float = 0.03,
        anchor_end: float = 0.18,
        explore_end: float = 0.48,
        close_start: float = 0.82,
        pareto_filter: str = "off",
        pareto_epsilon: float = 0.015,
        late_target_mode: str = "fixed",
        close_opponent_scale: float = 1.0,
        close_nash_scale: float = 1.0,
        close_similarity_scale: float = 1.0,
        repeat_penalty_scale: float = 1.0,
        risk_close_mode: str = "yield",
        risk_close_yield: float = 0.05,
        risk_close_hold: float = 0.04,
        deal_mode: str = "off",
        deal_start: float = 0.55,
        deal_budget: float = 0.020,
        deal_fit_scale: float = 0.18,
        repeater_rescue_mode: str = "threshold",
        repeater_rescue_time: float = 0.70,
        repeater_rescue_floor: float = 0.10,
        repeater_rescue_fit: float = 0.90,
        allocation_rescue_time: float = 0.84,
        allocation_rescue_similarity: float = 0.60,
        allocation_rescue_model_floor: float = 0.55,
        allocation_rescue_repeat_penalty: float = 0.30,
        numeric_rescue_mode: str = "off",
        numeric_rescue_time: float = 0.84,
        numeric_rescue_similarity: float = 0.50,
        numeric_rescue_floor: float = 0.30,
        numeric_rescue_model_floor: float = 0.35,
        numeric_rescue_repeat_penalty: float = 0.20,
        target_curve_mode: str = "power",
        sigmoid_midpoint: float = 0.68,
        sigmoid_steepness: float = 9.0,
        adaptive_floor_shift: float = 0.08,
        rank_aware_denial: str = "off",
        denial_time: float = 0.84,
        denial_own_ceiling: float = 0.24,
        denial_opponent_floor: float = 0.82,
        denial_similarity_floor: float = 0.78,
        denial_keep_floor: float = 0.50,
        selection_mode: str = "annealed",
        anneal_temperature: float = 0.06,
        anneal_cooling: float = 1.35,
        anneal_top_k: int = 8,
        opponent_horizon_mode: str = "decayed",
        short_horizon: int = 8,
        short_horizon_weight: float = 0.35,
        constraint_probe_mode: str = "aggressive",
        constraint_probe_time: float = 0.74,
        constraint_probe_floor: float = 0.10,
        constraint_probe_floor_mode: str = "target_budget",
        constraint_probe_budget_start: float = 0.06,
        constraint_probe_budget_end: float = 0.18,
        constraint_probe_relax_budget: float = 0.02,
        constraint_probe_min_fit: float = 0.68,
        constraint_probe_repeat: float = 0.92,
        constraint_probe_utility_weight: float = 1.70,
        constraint_probe_fit_weight: float = 0.58,
        constraint_probe_similarity_weight: float = 0.20,
        constraint_probe_rejection_penalty: float = 0.35,
        prediction_policy: str = "off",
        prediction_min_confidence: float = 0.65,
        prediction_low_surprise: float = 0.18,
        prediction_hold_shift: float = 0.04,
        prediction_probe_time_shift: float = 0.08,
        prediction_probe_repeat: float = 0.85,
        prediction_probe_concession: float = 0.25,
        logrolling_mode: str = "off",
        logrolling_weight: float = 0.045,
    ) -> None:
        self.early_target = early_target
        self.late_target = late_target
        self.base_late_target = late_target
        self.reservation_margin = reservation_margin
        self.anchor_end = anchor_end
        self.explore_end = max(anchor_end + 0.01, explore_end)
        self.close_start = max(self.explore_end + 0.01, close_start)
        self.pareto_filter = pareto_filter
        self.pareto_epsilon = pareto_epsilon
        self.late_target_mode = late_target_mode
        self.close_opponent_scale = close_opponent_scale
        self.close_nash_scale = close_nash_scale
        self.close_similarity_scale = close_similarity_scale
        self.repeat_penalty_scale = repeat_penalty_scale
        self.risk_close_mode = risk_close_mode
        self.risk_close_yield = risk_close_yield
        self.risk_close_hold = risk_close_hold
        self.deal_mode = deal_mode
        self.deal_start = deal_start
        self.deal_budget = deal_budget
        self.deal_fit_scale = deal_fit_scale
        self.repeater_rescue_mode = repeater_rescue_mode
        self.repeater_rescue_time = repeater_rescue_time
        self.repeater_rescue_floor = repeater_rescue_floor
        self.repeater_rescue_fit = repeater_rescue_fit
        self.allocation_rescue_time = allocation_rescue_time
        self.allocation_rescue_similarity = allocation_rescue_similarity
        self.allocation_rescue_model_floor = allocation_rescue_model_floor
        self.allocation_rescue_repeat_penalty = allocation_rescue_repeat_penalty
        self.numeric_rescue_mode = numeric_rescue_mode
        self.numeric_rescue_time = numeric_rescue_time
        self.numeric_rescue_similarity = numeric_rescue_similarity
        self.numeric_rescue_floor = numeric_rescue_floor
        self.numeric_rescue_model_floor = numeric_rescue_model_floor
        self.numeric_rescue_repeat_penalty = numeric_rescue_repeat_penalty
        self.target_curve_mode = target_curve_mode
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness
        self.adaptive_floor_shift = adaptive_floor_shift
        self.rank_aware_denial = rank_aware_denial
        self.denial_time = denial_time
        self.denial_own_ceiling = denial_own_ceiling
        self.denial_opponent_floor = denial_opponent_floor
        self.denial_similarity_floor = denial_similarity_floor
        self.denial_keep_floor = denial_keep_floor
        self.selection_mode = selection_mode
        self.anneal_temperature = anneal_temperature
        self.anneal_cooling = anneal_cooling
        self.anneal_top_k = anneal_top_k
        self.opponent_horizon_mode = opponent_horizon_mode
        self.short_horizon = short_horizon
        self.short_horizon_weight = short_horizon_weight
        self.constraint_probe_mode = constraint_probe_mode
        self.constraint_probe_time = constraint_probe_time
        self.constraint_probe_floor = constraint_probe_floor
        self.constraint_probe_floor_mode = constraint_probe_floor_mode
        self.constraint_probe_budget_start = constraint_probe_budget_start
        self.constraint_probe_budget_end = constraint_probe_budget_end
        self.constraint_probe_relax_budget = constraint_probe_relax_budget
        self.constraint_probe_min_fit = constraint_probe_min_fit
        self.constraint_probe_repeat = constraint_probe_repeat
        self.constraint_probe_utility_weight = constraint_probe_utility_weight
        self.constraint_probe_fit_weight = constraint_probe_fit_weight
        self.constraint_probe_similarity_weight = constraint_probe_similarity_weight
        self.constraint_probe_rejection_penalty = constraint_probe_rejection_penalty
        self.prediction_policy = prediction_policy
        self.prediction_min_confidence = prediction_min_confidence
        self.prediction_low_surprise = prediction_low_surprise
        self.prediction_hold_shift = prediction_hold_shift
        self.prediction_probe_time_shift = prediction_probe_time_shift
        self.prediction_probe_repeat = prediction_probe_repeat
        self.prediction_probe_concession = prediction_probe_concession
        self.logrolling_mode = logrolling_mode
        self.logrolling_weight = logrolling_weight
        self.sent_offers: defaultdict[Outcome, int] = defaultdict(int)

    def configure_catalog(self, catalog: list[OutcomeStats]) -> None:
        if self.late_target_mode == "utility_shape":
            self.late_target = self._utility_shape_late_target(catalog)
        else:
            self.late_target = self.base_late_target

    def choose(
        self,
        catalog: list[OutcomeStats],
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        reserved_value: float,
        turn_index: int = 0,
        prediction: NextOfferPrediction | None = None,
        prediction_review: dict[str, Any] | None = None,
        policy_profile: str = "default",
    ) -> OutcomeStats:
        phase = self.phase(relative_time)
        target = self.effective_target(
            relative_time,
            behavior,
            opponent,
            prediction,
            prediction_review,
            policy_profile,
        )
        floor = reserved_value + self.reservation_margin
        candidate_floor = self._candidate_floor(target, floor, relative_time, phase)
        profile_floor = self._policy_profile_candidate_floor(
            policy_profile, floor, relative_time
        )
        if profile_floor is not None:
            candidate_floor = max(candidate_floor, profile_floor)
        candidates = [item for item in catalog if item.utility >= candidate_floor]
        rescue = self._repeater_rescue_active(opponent, behavior, relative_time, phase)
        allocation_rescue = self._allocation_rescue_active(opponent, relative_time)
        numeric_rescue = self._numeric_rescue_active(opponent, relative_time)
        constraint_probe = self._constraint_probe_active(
            opponent, relative_time, prediction, prediction_review
        )
        if numeric_rescue:
            numeric_candidates = self._numeric_rescue_candidates(
                catalog, opponent, floor
            )
            if numeric_candidates:
                candidates = numeric_candidates
                rescue = False
                allocation_rescue = False
                constraint_probe = False
            else:
                numeric_rescue = False
        if constraint_probe:
            probe_candidates = self._constraint_probe_candidates(
                catalog, opponent, floor, target, relative_time
            )
            if probe_candidates:
                candidates = probe_candidates
                rescue = False
                allocation_rescue = False
                numeric_rescue = False
            else:
                constraint_probe = False
        if allocation_rescue:
            rescue_candidates = self._allocation_rescue_candidates(
                catalog, opponent, floor
            )
            if rescue_candidates:
                if self._rank_denial_active(opponent, relative_time) and (
                    self._should_deny_allocation(rescue_candidates, opponent)
                ):
                    candidate_floor = max(floor, self.denial_keep_floor)
                    denial_candidates = [
                        item for item in catalog if item.utility >= candidate_floor
                    ]
                    if denial_candidates:
                        candidates = denial_candidates
                        allocation_rescue = False
                        rescue = False
                        numeric_rescue = False
                else:
                    candidates = rescue_candidates
        elif rescue and self.repeater_rescue_mode == "threshold":
            rescue_floor = max(floor, self.repeater_rescue_floor)
            rescue_candidates = [
                item
                for item in catalog
                if item.utility >= rescue_floor
                and opponent.estimate(item.outcome) >= self.repeater_rescue_fit
            ]
            if not rescue_candidates and self.repeater_rescue_fit > 0.82:
                compromise_floor = max(floor, 0.20)
                rescue_candidates = [
                    item
                    for item in catalog
                    if item.utility >= compromise_floor
                    and opponent.estimate(item.outcome) >= 0.82
                ]
            if rescue_candidates:
                candidates = rescue_candidates
        elif rescue:
            rescue_floor = max(floor, self.repeater_rescue_floor)
            rescue_candidates = [item for item in catalog if item.utility >= rescue_floor]
            if rescue_candidates:
                candidates = rescue_candidates

        denial_floor = self._rank_denial_floor(catalog, opponent, relative_time, floor)
        if denial_floor is not None:
            denial_candidates = [
                item for item in catalog if item.utility >= denial_floor
            ]
            if denial_candidates:
                candidates = denial_candidates
                rescue = False
                allocation_rescue = False

        if not candidates:
            candidates = [item for item in catalog if item.utility >= floor]
        if not candidates:
            candidates = catalog
        candidates = self._apply_pareto_filter(candidates, opponent, phase)

        return self._select_candidate(
            candidates,
            opponent,
            behavior,
            relative_time,
            phase,
            rescue,
            allocation_rescue,
            constraint_probe,
            numeric_rescue,
            turn_index,
            policy_profile,
        )

    def phase(self, relative_time: float) -> str:
        if relative_time < self.anchor_end:
            return "anchor"
        if relative_time < self.explore_end:
            return "explore"
        if relative_time < self.close_start:
            return "bridge"
        return "close"

    def target(
        self,
        relative_time: float,
        behavior: str,
        opponent: OpponentModel | None = None,
    ) -> float:
        time = max(0.0, min(1.0, relative_time))
        behavior_shift = {
            "hardliner": -0.07,
            "conceder": 0.04,
            "reciprocal": -0.02,
            "fair-seeking": 0.0,
            "erratic": -0.04,
        }.get(behavior, 0.0)

        if self.target_curve_mode == "sigmoid":
            target = self._sigmoid_target(time, self.late_target, self.sigmoid_midpoint)
        elif self.target_curve_mode == "adaptive_sigmoid":
            late_target, midpoint = self._adaptive_sigmoid_params(
                time, behavior, opponent
            )
            target = self._sigmoid_target(time, late_target, midpoint)
        else:
            concession = time**2.35
            target = (
                self.early_target
                - (self.early_target - self.late_target) * concession
            )
        return max(0.18, min(0.98, target + behavior_shift))

    def effective_target(
        self,
        relative_time: float,
        behavior: str,
        opponent: OpponentModel,
        prediction: NextOfferPrediction | None = None,
        prediction_review: dict[str, Any] | None = None,
        policy_profile: str = "default",
    ) -> float:
        target = self.target(relative_time, behavior, opponent)
        target = self._apply_policy_profile_target(target, policy_profile, relative_time)
        if relative_time < 0.82 or self.risk_close_mode == "off":
            return self._apply_prediction_target_adjustment(
                target, relative_time, prediction, prediction_review
            )

        risk = self._close_agreement_risk(opponent, behavior)
        close_progress = (relative_time - 0.82) / 0.18
        adjustment = -self.risk_close_yield * risk * close_progress
        if self.risk_close_mode == "hold_yield":
            adjustment += self.risk_close_hold * (1.0 - risk) * close_progress
        adjusted = max(0.18, min(0.98, target + adjustment))
        return self._apply_prediction_target_adjustment(
            adjusted, relative_time, prediction, prediction_review
        )

    def _apply_policy_profile_target(
        self, target: float, policy_profile: str, relative_time: float
    ) -> float:
        if policy_profile == "default":
            return target
        if policy_profile == "agreement_rescue":
            close_progress = max(0.0, min(1.0, (relative_time - 0.74) / 0.26))
            return max(0.18, min(0.98, target - 0.11 * close_progress))
        if policy_profile == "soft_leakage_guard":
            shift = 0.015
        elif policy_profile == "leakage_guard":
            shift = 0.025
        elif policy_profile == "self_floor":
            shift = 0.040
        elif policy_profile == "no_low_accept":
            shift = 0.055
        else:
            return target
        close_progress = max(0.0, min(1.0, (relative_time - 0.78) / 0.22))
        return max(0.18, min(0.98, target + shift * close_progress))

    def _policy_profile_candidate_floor(
        self, policy_profile: str, floor: float, relative_time: float
    ) -> float | None:
        if policy_profile == "default":
            return None
        if policy_profile == "agreement_rescue":
            return None
        close_progress = max(0.0, min(1.0, (relative_time - 0.78) / 0.22))
        if policy_profile == "soft_leakage_guard":
            return max(floor, 0.42 + 0.04 * close_progress)
        if policy_profile == "leakage_guard":
            return max(floor, 0.46 + 0.04 * close_progress)
        if policy_profile == "self_floor":
            return max(floor, 0.50 + 0.03 * close_progress)
        if policy_profile == "no_low_accept":
            return max(floor, 0.54 + 0.03 * close_progress)
        return None

    def _policy_profile_score_adjustment(
        self,
        policy_profile: str,
        self_utility: float,
        opponent_utility: float,
        similarity: float,
        relative_time: float,
        phase: str,
    ) -> float:
        if policy_profile == "default" or phase != "close":
            return 0.0
        pressure = max(0.0, min(1.0, (relative_time - 0.80) / 0.20))
        leakage_shape = max(0.0, opponent_utility - self_utility)
        if policy_profile == "agreement_rescue":
            return pressure * (
                0.14 * opponent_utility
                + 0.10 * similarity
                + 0.04 * leakage_shape
                - 0.03 * self_utility
            )
        if policy_profile == "soft_leakage_guard":
            return pressure * (0.05 * self_utility - 0.03 * leakage_shape)
        if policy_profile == "leakage_guard":
            return pressure * (0.10 * self_utility - 0.06 * leakage_shape)
        if policy_profile == "self_floor":
            return pressure * (0.16 * self_utility - 0.09 * leakage_shape)
        if policy_profile == "no_low_accept":
            return pressure * (0.22 * self_utility - 0.12 * leakage_shape - 0.03 * similarity)
        return 0.0

    def score(
        self,
        item: OutcomeStats,
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        phase: str,
        rescue: bool = False,
        allocation_rescue: bool = False,
        constraint_probe: bool = False,
        numeric_rescue: bool = False,
        policy_profile: str = "default",
    ) -> tuple[float, ...]:
        opponent_utility = self._opponent_utility(
            opponent, item.outcome, behavior, relative_time
        )
        nash_proxy = item.utility * max(0.05, opponent_utility)
        welfare_proxy = item.utility + opponent_utility
        similarity = self._recent_similarity(item.outcome, opponent)
        repeat_penalty = (
            0.08 * self.repeat_penalty_scale * self.sent_offers[item.outcome]
        )
        confidence = opponent.confidence

        if allocation_rescue:
            allocation_score = (
                similarity
                + 0.20 * item.utility
                + 0.05 * opponent_utility
                - self.allocation_rescue_repeat_penalty
                * self.sent_offers[item.outcome]
            )
            return (
                allocation_score,
                nash_proxy,
                opponent_utility,
                similarity,
                item.utility,
            )

        if numeric_rescue:
            numeric_score = (
                1.40 * item.utility
                + 0.30 * similarity
                + 0.15 * opponent_utility
                + 0.10 * nash_proxy
                - self.numeric_rescue_repeat_penalty
                * self.sent_offers[item.outcome]
            )
            return (
                numeric_score,
                item.utility,
                similarity,
                opponent_utility,
                nash_proxy,
            )

        if constraint_probe:
            acceptance_fit = opponent.constraint_acceptance_estimate(item.outcome)
            rejection_similarity = opponent._rejection_similarity(item.outcome)
            constraint_score = (
                self.constraint_probe_utility_weight * item.utility
                + self.constraint_probe_fit_weight * acceptance_fit
                + self.constraint_probe_similarity_weight * similarity
                + 0.10 * nash_proxy
                - self.constraint_probe_rejection_penalty * rejection_similarity
                - 0.05 * self.sent_offers[item.outcome]
            )
            return (
                constraint_score,
                nash_proxy,
                acceptance_fit,
                similarity,
                item.utility,
            )

        if rescue and self.repeater_rescue_mode == "threshold":
            threshold_score = (
                item.utility
                + 0.30 * opponent_utility
                + 0.10 * similarity
                - 0.02 * self.sent_offers[item.outcome]
            )
            return (
                threshold_score,
                nash_proxy,
                opponent_utility,
                similarity,
                item.utility,
            )

        if rescue:
            rescue_score = (
                3.00 * similarity
                + 1.25 * opponent_utility
                + 0.35 * nash_proxy
                - 0.30 * item.utility
                - 1.00 * self.sent_offers[item.outcome]
            )
            return (rescue_score, similarity, opponent_utility, nash_proxy, item.utility)

        if phase == "anchor":
            opponent_weight = 0.15 * confidence
            nash_weight = 0.05
            similarity_weight = 0.00
        elif phase == "explore":
            opponent_weight = 0.25 + 0.20 * confidence
            nash_weight = 0.12
            similarity_weight = 0.08 * confidence
        elif phase == "bridge":
            opponent_weight = 0.45 + 0.30 * confidence
            nash_weight = 0.25
            similarity_weight = 0.28 * confidence
        else:
            opponent_weight = 0.85 + 0.35 * confidence
            nash_weight = 0.45
            similarity_weight = 0.55 * confidence
            opponent_weight *= self.close_opponent_scale
            nash_weight *= self.close_nash_scale
            similarity_weight *= self.close_similarity_scale

        if behavior == "hardliner" and relative_time < 0.75:
            opponent_weight *= 0.75
            repeat_penalty *= 0.35
        elif behavior == "reciprocal":
            opponent_weight *= 1.15
        elif behavior == "erratic":
            nash_weight *= 1.20

        deal_bonus = 0.0
        if self._deal_active(relative_time, phase):
            progress = self._deal_progress(relative_time)
            fit = 0.55 * opponent_utility + 0.30 * similarity + 0.15 * nash_proxy
            deal_bonus = self.deal_fit_scale * progress * fit

        profile_bonus = self._policy_profile_score_adjustment(
            policy_profile,
            item.utility,
            opponent_utility,
            similarity,
            relative_time,
            phase,
        )
        logrolling_bonus = self._logrolling_bonus(
            item,
            opponent,
            phase,
            confidence,
        )

        score = (
            item.utility
            + opponent_weight * opponent_utility
            + nash_weight * nash_proxy
            + 0.08 * welfare_proxy
            + similarity_weight * similarity
            + deal_bonus
            + profile_bonus
            + logrolling_bonus
            - repeat_penalty
        )
        return (score, nash_proxy, opponent_utility, similarity, item.utility)

    def _logrolling_bonus(
        self,
        item: OutcomeStats,
        opponent: OpponentModel,
        phase: str,
        confidence: float,
    ) -> float:
        if self.logrolling_mode == "off" or phase not in {"bridge", "close"}:
            return 0.0
        if confidence < 0.35 or not opponent.history:
            return 0.0

        latest = opponent.history[-1]
        weights = opponent.issue_weights()
        if not weights:
            return 0.0
        important_match = sum(
            weight
            for index, weight in enumerate(weights)
            if index < len(item.outcome)
            and index < len(latest)
            and item.outcome[index] == latest[index]
        )
        if important_match <= 0.0:
            return 0.0

        changed_count = sum(
            1
            for mine, theirs in zip(item.outcome, latest)
            if mine != theirs
        )
        if changed_count <= 0:
            return 0.0
        balanced_trade = min(1.0, important_match) * min(1.0, item.utility)
        return self.logrolling_weight * confidence * balanced_trade

    def _opponent_utility(
        self,
        opponent: OpponentModel,
        outcome: Outcome,
        behavior: str,
        relative_time: float,
    ) -> float:
        decayed = opponent.estimate(outcome)
        mode = self.opponent_horizon_mode
        if mode == "decayed":
            return decayed

        recent = opponent.recent_estimate(outcome, self.short_horizon)
        if mode == "recent":
            return recent

        weight = max(0.0, min(1.0, self.short_horizon_weight))
        if mode == "adaptive":
            if behavior == "hardliner":
                weight = min(0.75, weight + 0.20)
            elif behavior == "conceder":
                weight = min(0.65, weight + 0.10)
            elif behavior == "erratic":
                weight = max(0.15, weight - 0.15)
            if relative_time >= 0.82:
                weight = min(0.80, weight + 0.10)
        return (1.0 - weight) * decayed + weight * recent

    def _select_candidate(
        self,
        candidates: list[OutcomeStats],
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        phase: str,
        rescue: bool,
        allocation_rescue: bool,
        constraint_probe: bool,
        numeric_rescue: bool,
        turn_index: int,
        policy_profile: str = "default",
    ) -> OutcomeStats:
        scored = sorted(
            (
                (
                    self.score(
                        item,
                        opponent,
                        behavior,
                        relative_time,
                        phase,
                        rescue,
                        allocation_rescue,
                        constraint_probe,
                        numeric_rescue,
                        policy_profile,
                    ),
                    item,
                )
                for item in candidates
            ),
            key=lambda row: row[0],
            reverse=True,
        )
        if not scored:
            raise ValueError("BiddingPolicy received an empty candidate list")

        if (
            self.selection_mode != "annealed"
            or rescue
            or allocation_rescue
            or constraint_probe
            or numeric_rescue
            or relative_time >= 0.96
        ):
            return scored[0][1]

        temperature = self._annealed_temperature(relative_time, opponent)
        if temperature <= 0.001:
            return scored[0][1]

        top_count = max(1, min(self.anneal_top_k, len(scored)))
        top = scored[:top_count]
        best_score = top[0][0][0]
        weights = [
            math.exp(max(-40.0, min(0.0, (score[0] - best_score) / temperature)))
            for score, _ in top
        ]
        total = sum(weights)
        if total <= 0.0:
            return top[0][1]

        draw = self._deterministic_draw(turn_index, len(opponent.history), len(candidates))
        threshold = draw * total
        cumulative = 0.0
        for weight, (_, item) in zip(weights, top):
            cumulative += weight
            if cumulative >= threshold:
                return item
        return top[-1][1]

    def _sigmoid_target(
        self, time: float, late_target: float, midpoint: float
    ) -> float:
        steepness = max(0.1, self.sigmoid_steepness)
        start = 1.0 / (1.0 + math.exp(-steepness * (0.0 - midpoint)))
        end = 1.0 / (1.0 + math.exp(-steepness * (1.0 - midpoint)))
        current = 1.0 / (1.0 + math.exp(-steepness * (time - midpoint)))
        progress = (current - start) / max(0.0001, end - start)
        progress = max(0.0, min(1.0, progress))
        return self.early_target - (self.early_target - late_target) * progress

    def _adaptive_sigmoid_params(
        self,
        time: float,
        behavior: str,
        opponent: OpponentModel | None,
    ) -> tuple[float, float]:
        late_target = self.late_target
        midpoint = self.sigmoid_midpoint
        if opponent is None or len(opponent.history) < 3:
            return late_target, midpoint

        repetition = self._recent_repetition(opponent)
        slope = opponent.concession_slope()
        if behavior == "conceder" or slope > 0.12:
            midpoint += 0.06
            late_target += 0.04
        if behavior == "hardliner" or repetition >= 0.72:
            midpoint -= 0.05
            late_target -= 0.04
        if behavior == "erratic":
            midpoint -= 0.03
            late_target -= 0.02
        if time >= 0.75:
            risk = self._close_agreement_risk(opponent, behavior)
            close_progress = (time - 0.75) / 0.25
            late_target -= self.adaptive_floor_shift * risk * close_progress

        late_target = max(0.18, min(0.78, late_target))
        midpoint = max(0.52, min(0.82, midpoint))
        return late_target, midpoint

    def _annealed_temperature(
        self, relative_time: float, opponent: OpponentModel
    ) -> float:
        time_left = max(0.0, 1.0 - relative_time)
        confidence_discount = 1.0 - 0.45 * opponent.confidence
        return (
            self.anneal_temperature
            * (time_left**max(0.1, self.anneal_cooling))
            * max(0.20, confidence_discount)
        )

    def _deterministic_draw(
        self, turn_index: int, opponent_offer_count: int, candidate_count: int
    ) -> float:
        seed = (
            (turn_index + 1) * 1_103_515_245
            + (opponent_offer_count + 7) * 12_345
            + (candidate_count + 17) * 2_654_435_761
        ) & 0xFFFFFFFF
        seed ^= seed >> 16
        seed = (seed * 2_246_822_519 + 3_266_489_917) & 0xFFFFFFFF
        return seed / 0x1_0000_0000

    def _recent_repetition(self, opponent: OpponentModel) -> float:
        if not opponent.history:
            return 0.0
        recent = opponent.history[-6:]
        return max(Counter(recent).values()) / len(recent)

    def _recent_similarity(self, outcome: Outcome, opponent: OpponentModel) -> float:
        if not opponent.history:
            return 0.0

        recent = opponent.history[-3:]
        scores = []
        for offer in recent:
            matches = sum(1 for mine, theirs in zip(outcome, offer) if mine == theirs)
            scores.append(matches / len(outcome) if outcome else 0.0)
        return max(scores) if scores else 0.0

    def _apply_pareto_filter(
        self, candidates: list[OutcomeStats], opponent: OpponentModel, phase: str
    ) -> list[OutcomeStats]:
        if self.pareto_filter == "off":
            return candidates
        if self.pareto_filter == "close" and phase != "close":
            return candidates
        if self.pareto_filter == "bridge_close" and phase not in {"bridge", "close"}:
            return candidates
        if len(candidates) < 3 or opponent.confidence < 0.35:
            return candidates

        epsilon = self.pareto_epsilon
        scored = sorted(
            ((item.utility, opponent.estimate(item.outcome), item) for item in candidates),
            key=lambda row: (row[0], row[1]),
            reverse=True,
        )
        frontier = []
        best_opponent_utility = -1.0
        for _, opponent_utility, item in scored:
            if opponent_utility > best_opponent_utility + epsilon:
                frontier.append(item)
                best_opponent_utility = opponent_utility

        return frontier or candidates

    def _candidate_floor(
        self, target: float, floor: float, relative_time: float, phase: str
    ) -> float:
        if not self._deal_active(relative_time, phase):
            return max(target, floor)
        progress = self._deal_progress(relative_time)
        budget = self.deal_budget * progress
        return max(target - budget, floor)

    def _deal_active(self, relative_time: float, phase: str) -> bool:
        if self.deal_mode == "off":
            return False
        if relative_time < self.deal_start:
            return False
        if self.deal_mode == "close_budget":
            return phase == "close"
        return self.deal_mode == "budget"

    def _deal_progress(self, relative_time: float) -> float:
        if relative_time <= self.deal_start:
            return 0.0
        span = max(0.01, 1.0 - self.deal_start)
        return max(0.0, min(1.0, (relative_time - self.deal_start) / span))

    def _repeater_rescue_active(
        self,
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        phase: str,
    ) -> bool:
        if self.repeater_rescue_mode == "off":
            return False
        if phase != "close" or relative_time < self.repeater_rescue_time:
            return False
        if len(opponent.history) < 6:
            return False
        recent = opponent.history[-6:]
        repeated_share = max(Counter(recent).values()) / len(recent)
        threshold = 0.90 if self.repeater_rescue_mode == "threshold" else 0.80
        return repeated_share >= threshold

    def _allocation_rescue_active(
        self, opponent: OpponentModel, relative_time: float
    ) -> bool:
        if self.repeater_rescue_mode != "threshold":
            return False
        if relative_time < self.allocation_rescue_time:
            return False
        if len(opponent.history) < 6:
            return False
        repeated = self._repeated_offer(opponent)
        if repeated is None or len(repeated) < 6:
            return False
        if any(isinstance(value, (int, float)) for value in repeated):
            return False
        recent = opponent.history[-6:]
        repeated_share = recent.count(repeated) / len(recent)
        return repeated_share >= 0.90

    def _allocation_rescue_candidates(
        self,
        catalog: list[OutcomeStats],
        opponent: OpponentModel,
        floor: float,
    ) -> list[OutcomeStats]:
        repeated = self._repeated_offer(opponent)
        if repeated is None:
            return []
        minimum = max(floor, self.repeater_rescue_floor)
        candidates = []
        for item in catalog:
            if item.outcome == repeated or item.utility < minimum:
                continue
            similarity = self._outcome_similarity(item.outcome, repeated)
            if similarity < self.allocation_rescue_similarity:
                continue
            if opponent.estimate(item.outcome) < self.allocation_rescue_model_floor:
                continue
            candidates.append(item)
        return candidates

    def _numeric_rescue_active(
        self, opponent: OpponentModel, relative_time: float
    ) -> bool:
        if self.numeric_rescue_mode == "off":
            return False
        if relative_time < self.numeric_rescue_time:
            return False
        if len(opponent.history) < 6:
            return False
        repeated = self._repeated_offer(opponent)
        if repeated is None or not repeated:
            return False
        if not any(isinstance(value, (int, float)) for value in repeated):
            return False
        recent = opponent.history[-6:]
        repeated_share = recent.count(repeated) / len(recent)
        return repeated_share >= 0.90

    def _numeric_rescue_candidates(
        self,
        catalog: list[OutcomeStats],
        opponent: OpponentModel,
        floor: float,
    ) -> list[OutcomeStats]:
        repeated = self._repeated_offer(opponent)
        if repeated is None:
            return []
        minimum = max(floor, self.numeric_rescue_floor)
        numeric_indices = {
            index
            for index, value in enumerate(repeated)
            if isinstance(value, (int, float))
        }
        if not numeric_indices:
            return []

        candidates = []
        for item in catalog:
            if item.outcome == repeated or item.utility < minimum:
                continue
            changed = [
                index
                for index, (left, right) in enumerate(zip(item.outcome, repeated))
                if left != right
            ]
            if self.numeric_rescue_mode == "one_issue":
                if len(changed) != 1 or changed[0] not in numeric_indices:
                    continue
            elif not changed or not any(index in numeric_indices for index in changed):
                continue

            similarity = self._outcome_similarity(item.outcome, repeated)
            if similarity < self.numeric_rescue_similarity:
                continue
            if opponent.estimate(item.outcome) < self.numeric_rescue_model_floor:
                continue
            candidates.append(item)
        return candidates

    def _constraint_probe_active(
        self,
        opponent: OpponentModel,
        relative_time: float,
        prediction: NextOfferPrediction | None = None,
        prediction_review: dict[str, Any] | None = None,
    ) -> bool:
        if self.constraint_probe_mode == "off":
            return False
        start_time = self._prediction_probe_start_time(prediction, prediction_review)
        if relative_time < start_time:
            return False
        if len(opponent.history) < 4:
            return False
        if self._recent_repetition(opponent) < self.constraint_probe_repeat:
            return False
        return self._repeated_offer(opponent) is not None

    def _prediction_target_adjustment(
        self,
        relative_time: float,
        prediction: NextOfferPrediction | None,
        prediction_review: dict[str, Any] | None,
    ) -> float:
        if self.prediction_policy not in {"hold", "hybrid"}:
            return 0.0
        if not self._prediction_reliable(prediction, prediction_review):
            return 0.0
        assert prediction is not None
        if prediction.last_self_utility is None:
            return 0.0
        expected_gain = prediction.expected_self_utility - prediction.last_self_utility
        if expected_gain < 0.025:
            return 0.0
        if prediction.concession_probability < 0.55:
            return 0.0
        if relative_time >= 0.90:
            return 0.0
        time_scale = max(0.15, 1.0 - relative_time)
        confidence_scale = max(0.0, min(1.0, prediction.confidence))
        return self.prediction_hold_shift * time_scale * confidence_scale

    def _apply_prediction_target_adjustment(
        self,
        target: float,
        relative_time: float,
        prediction: NextOfferPrediction | None,
        prediction_review: dict[str, Any] | None,
    ) -> float:
        adjustment = self._prediction_target_adjustment(
            relative_time, prediction, prediction_review
        )
        return max(0.18, min(0.98, target + adjustment))

    def _prediction_probe_start_time(
        self,
        prediction: NextOfferPrediction | None,
        prediction_review: dict[str, Any] | None,
    ) -> float:
        start_time = self.constraint_probe_time
        if self.prediction_policy not in {"probe", "hybrid"}:
            return start_time
        if not self._prediction_reliable(prediction, prediction_review):
            return start_time
        assert prediction is not None
        if prediction.repeat_probability < self.prediction_probe_repeat:
            return start_time
        if prediction.concession_probability > self.prediction_probe_concession:
            return start_time
        return max(0.50, start_time - self.prediction_probe_time_shift)

    def _prediction_reliable(
        self,
        prediction: NextOfferPrediction | None,
        prediction_review: dict[str, Any] | None,
    ) -> bool:
        if prediction is None:
            return False
        if prediction.confidence < self.prediction_min_confidence:
            return False
        if not prediction_review or not prediction_review.get("observed"):
            return True
        surprise = prediction_review.get("surprise")
        mean_error = prediction_review.get("mean_absolute_error")
        if isinstance(surprise, (int, float)) and surprise <= self.prediction_low_surprise:
            return True
        return isinstance(mean_error, (int, float)) and mean_error <= 0.08

    def _constraint_probe_candidates(
        self,
        catalog: list[OutcomeStats],
        opponent: OpponentModel,
        floor: float,
        target: float,
        relative_time: float,
    ) -> list[OutcomeStats]:
        repeated = self._repeated_offer(opponent)
        if repeated is None:
            return []
        minimum = self._constraint_probe_utility_floor(
            floor, target, relative_time, relaxed=False
        )
        min_fit = self._constraint_probe_fit_floor(relative_time)
        rejected = set(opponent.rejected_self_offers[-12:])
        candidates = []
        for item in catalog:
            if item.utility < minimum:
                continue
            if item.outcome == repeated or item.outcome in rejected:
                continue
            fit = opponent.constraint_acceptance_estimate(item.outcome)
            if fit < min_fit:
                continue
            candidates.append(item)
        if candidates:
            return candidates

        relaxed_floor = self._constraint_probe_utility_floor(
            floor, target, relative_time, relaxed=True
        )
        return [
            item
            for item in catalog
            if item.utility >= relaxed_floor
            and item.outcome != repeated
            and item.outcome not in rejected
        ]

    def _constraint_probe_utility_floor(
        self,
        floor: float,
        target: float,
        relative_time: float,
        relaxed: bool,
    ) -> float:
        if self.constraint_probe_floor_mode != "target_budget":
            if relaxed:
                return max(floor, min(0.20, self.constraint_probe_floor))
            return max(floor, self.constraint_probe_floor)

        progress = max(
            0.0,
            min(
                1.0,
                (relative_time - self.constraint_probe_time)
                / max(0.01, 1.0 - self.constraint_probe_time),
            ),
        )
        budget = self.constraint_probe_budget_start + (
            self.constraint_probe_budget_end - self.constraint_probe_budget_start
        ) * progress
        if relaxed:
            budget += self.constraint_probe_relax_budget
        return max(floor, self.constraint_probe_floor, target - budget)

    def _constraint_probe_fit_floor(self, relative_time: float) -> float:
        if self.constraint_probe_mode == "aggressive":
            return max(0.35, self.constraint_probe_min_fit - 0.12)
        if self.constraint_probe_mode == "late":
            progress = max(0.0, min(1.0, (relative_time - self.constraint_probe_time) / 0.40))
            return max(0.35, self.constraint_probe_min_fit - 0.18 * progress)
        return self.constraint_probe_min_fit

    def _rank_denial_active(
        self, opponent: OpponentModel, relative_time: float
    ) -> bool:
        if self.rank_aware_denial == "off":
            return False
        if relative_time < self.denial_time:
            return False
        if len(opponent.history) < 6:
            return False
        repeated = self._repeated_offer(opponent)
        if repeated is None:
            return False
        return self._recent_repetition(opponent) >= 0.90

    def _rank_denial_floor(
        self,
        catalog: list[OutcomeStats],
        opponent: OpponentModel,
        relative_time: float,
        floor: float,
    ) -> float | None:
        if not self._rank_denial_active(opponent, relative_time):
            return None
        repeated = self._repeated_offer(opponent)
        if repeated is None:
            return None
        repeated_stats = next(
            (item for item in catalog if item.outcome == repeated),
            None,
        )
        if repeated_stats is None:
            return None
        if repeated_stats.utility > self.denial_own_ceiling:
            return None
        if opponent.estimate(repeated) < self.denial_opponent_floor:
            return None
        return max(floor, self.denial_keep_floor)

    def _should_deny_allocation(
        self, candidates: list[OutcomeStats], opponent: OpponentModel
    ) -> bool:
        repeated = self._repeated_offer(opponent)
        if repeated is None:
            return False
        best = max(
            candidates,
            key=lambda item: (
                self._outcome_similarity(item.outcome, repeated),
                opponent.estimate(item.outcome),
                item.utility,
            ),
        )
        similarity = self._outcome_similarity(best.outcome, repeated)
        opponent_view = opponent.estimate(best.outcome)
        return (
            best.utility <= self.denial_own_ceiling
            and opponent_view >= self.denial_opponent_floor
            and similarity >= self.denial_similarity_floor
        )

    def _repeated_offer(self, opponent: OpponentModel) -> Outcome | None:
        if not opponent.history:
            return None
        return Counter(opponent.history[-8:]).most_common(1)[0][0]

    def _outcome_similarity(self, first: Outcome, second: Outcome) -> float:
        if not first:
            return 0.0
        matches = sum(1 for left, right in zip(first, second) if left == right)
        return matches / len(first)

    def _close_agreement_risk(self, opponent: OpponentModel, behavior: str) -> float:
        offer_count = len(opponent.history)
        if offer_count < 3:
            return 0.35

        recent = opponent.history[-6:]
        unique_ratio = len(set(recent)) / len(recent)
        repetition = 1.0 - unique_ratio
        slope = opponent.concession_slope()
        concession_shortfall = 1.0 - min(1.0, max(0.0, slope) / 0.16)
        rejection_pressure = min(1.0, sum(self.sent_offers.values()) / 8.0)
        behavior_risk = {
            "hardliner": 0.85,
            "erratic": 0.65,
            "fair-seeking": 0.45,
            "reciprocal": 0.35,
            "conceder": 0.20,
        }.get(behavior, 0.45)

        risk = (
            0.30 * rejection_pressure
            + 0.25 * concession_shortfall
            + 0.20 * repetition
            + 0.25 * behavior_risk
        )
        return max(0.0, min(1.0, risk))

    def _utility_shape_late_target(self, catalog: list[OutcomeStats]) -> float:
        if not catalog:
            return self.base_late_target

        utilities = [item.utility for item in catalog]
        count = len(utilities)
        top80_density = sum(value >= 0.80 for value in utilities) / count
        near45_density = sum(value >= 0.45 for value in utilities) / count
        utility_std = pstdev(utilities) if count > 1 else 0.0
        utility_mean = mean(utilities)

        if count <= 100 and (near45_density < 0.35 or utility_std > 0.25):
            return 0.38
        if count >= 500 and top80_density < 0.08:
            return 0.52
        if top80_density < 0.06 and near45_density < 0.45:
            return 0.38
        if utility_mean < 0.40 and near45_density < 0.40:
            return 0.38
        return self.base_late_target

    def remember(self, outcome: Outcome) -> None:
        self.sent_offers[outcome] += 1


class AcceptancePolicy:
    def __init__(
        self,
        next_gap: float = 0.015,
        target_bonus: float = 0.015,
        late_time: float = 0.90,
        late_slack: float = 0.02,
        hardliner_time: float = 0.70,
        hardliner_minimum: float = 0.46,
        model_time: float = 0.72,
        model_minimum: float = 0.66,
        enable_hardliner_acceptance: bool = True,
        enable_model_acceptance: bool = True,
        prediction_accept_mode: str = "off",
        prediction_accept_time: float = 0.78,
        prediction_accept_slack: float = 0.04,
        prediction_accept_min_confidence: float = 0.65,
        prediction_accept_repeat: float = 0.85,
        prediction_accept_concession: float = 0.25,
        leakage_denial_mode: str = "off",
        leakage_denial_time: float = 0.80,
        leakage_denial_own_ceiling: float = 0.33,
        leakage_denial_opponent_floor: float = 0.82,
        leakage_denial_repeat: float = 0.90,
    ) -> None:
        self.next_gap = next_gap
        self.target_bonus = target_bonus
        self.late_time = late_time
        self.late_slack = late_slack
        self.hardliner_time = hardliner_time
        self.hardliner_minimum = hardliner_minimum
        self.model_time = model_time
        self.model_minimum = model_minimum
        self.enable_hardliner_acceptance = enable_hardliner_acceptance
        self.enable_model_acceptance = enable_model_acceptance
        self.prediction_accept_mode = prediction_accept_mode
        self.prediction_accept_time = prediction_accept_time
        self.prediction_accept_slack = prediction_accept_slack
        self.prediction_accept_min_confidence = prediction_accept_min_confidence
        self.prediction_accept_repeat = prediction_accept_repeat
        self.prediction_accept_concession = prediction_accept_concession
        self.leakage_denial_mode = leakage_denial_mode
        self.leakage_denial_time = leakage_denial_time
        self.leakage_denial_own_ceiling = leakage_denial_own_ceiling
        self.leakage_denial_opponent_floor = leakage_denial_opponent_floor
        self.leakage_denial_repeat = leakage_denial_repeat

    def should_accept(
        self,
        offer_stats: OutcomeStats,
        next_offer_stats: OutcomeStats,
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        reserved_value: float,
        target: float,
        prediction: NextOfferPrediction | None = None,
        prediction_review: dict[str, Any] | None = None,
        policy_profile: str = "default",
    ) -> bool:
        return self.explain(
            offer_stats,
            next_offer_stats,
            opponent,
            behavior,
            relative_time,
            reserved_value,
            target,
            prediction,
            prediction_review,
            policy_profile,
        ).accept

    def explain(
        self,
        offer_stats: OutcomeStats,
        next_offer_stats: OutcomeStats,
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        reserved_value: float,
        target: float,
        prediction: NextOfferPrediction | None = None,
        prediction_review: dict[str, Any] | None = None,
        policy_profile: str = "default",
    ) -> AcceptanceDecision:
        floor = reserved_value + 0.02
        utility = offer_stats.utility
        next_utility = next_offer_stats.utility
        opponent_view = opponent.estimate(offer_stats.outcome)
        next_opponent_view = opponent.estimate(next_offer_stats.outcome)
        prediction_slack = self._prediction_accept_slack(
            utility, relative_time, prediction, prediction_review
        )
        effective_next_gap = max(0.0, self.next_gap + prediction_slack)
        effective_late_slack = max(0.0, self.late_slack + prediction_slack)
        effective_late_time = self.late_time
        effective_model_minimum = self.model_minimum
        effective_hardliner_minimum = self.hardliner_minimum
        if policy_profile == "agreement_rescue":
            effective_next_gap = max(effective_next_gap, 0.030)
            effective_late_slack = max(effective_late_slack, 0.065)
            effective_late_time = min(effective_late_time, 0.82)
            effective_model_minimum = min(effective_model_minimum, 0.52)
            effective_hardliner_minimum = min(effective_hardliner_minimum, 0.34)

        if utility < floor:
            return AcceptanceDecision(
                False,
                "below_reservation_floor",
                floor,
                utility,
                next_utility,
                target,
                opponent_view,
                next_opponent_view,
            )
        if self._leakage_denial_active(
            offer_stats,
            opponent,
            behavior,
            relative_time,
            opponent_view,
            policy_profile,
        ):
            return AcceptanceDecision(
                False,
                "leakage_denial",
                floor,
                utility,
                next_utility,
                target,
                opponent_view,
                next_opponent_view,
            )
        if utility >= next_utility - effective_next_gap:
            reason = "beats_or_matches_next_counter"
            if (
                prediction_slack > 0.0
                and utility < next_utility - self.next_gap
            ):
                reason = "prediction_stall_matches_next"
            return AcceptanceDecision(
                True,
                reason,
                floor,
                utility,
                next_utility,
                target,
                opponent_view,
                next_opponent_view,
            )
        if utility >= target + self.target_bonus:
            return AcceptanceDecision(
                True,
                "above_dynamic_aspiration",
                floor,
                utility,
                next_utility,
                target,
                opponent_view,
                next_opponent_view,
            )
        if relative_time > effective_late_time and utility >= max(
            floor, target - effective_late_slack
        ):
            reason = "late_good_enough"
            if (
                prediction_slack > 0.0
                and utility < max(floor, target - self.late_slack)
            ):
                reason = "prediction_stall_late_good_enough"
            return AcceptanceDecision(
                True,
                reason,
                floor,
                utility,
                next_utility,
                target,
                opponent_view,
                next_opponent_view,
            )
        if (
            self.enable_hardliner_acceptance
            and
            behavior == "hardliner"
            and policy_profile != "no_low_accept"
            and relative_time > self.hardliner_time
            and utility >= max(floor, effective_hardliner_minimum)
        ):
            return AcceptanceDecision(
                True,
                "late_hardliner_minimum",
                floor,
                utility,
                next_utility,
                target,
                opponent_view,
                next_opponent_view,
            )

        if (
            self.enable_model_acceptance
            and relative_time > self.model_time
            and utility >= effective_model_minimum
            and opponent_view >= next_opponent_view
        ):
            return AcceptanceDecision(
                True,
                "late_model_compatible",
                floor,
                utility,
                next_utility,
                target,
                opponent_view,
                next_opponent_view,
            )
        return AcceptanceDecision(
            False,
            "counter_is_better",
            floor,
            utility,
            next_utility,
            target,
            opponent_view,
            next_opponent_view,
        )

    def _leakage_denial_active(
        self,
        offer_stats: OutcomeStats,
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        opponent_view: float,
        policy_profile: str = "default",
    ) -> bool:
        if policy_profile != "default":
            return self._policy_profile_leakage_denial_active(
                offer_stats,
                opponent,
                behavior,
                relative_time,
                opponent_view,
                policy_profile,
            )
        if self.leakage_denial_mode == "off":
            return False
        if relative_time < self.leakage_denial_time:
            return False
        if behavior != "hardliner":
            return False
        if offer_stats.utility > self.leakage_denial_own_ceiling:
            return False
        if opponent_view < self.leakage_denial_opponent_floor:
            return False
        if len(opponent.history) < 6:
            return False

        if self.leakage_denial_mode == "broad":
            return True

        recent = opponent.history[-6:]
        repeated, count = Counter(recent).most_common(1)[0]
        if count / len(recent) < self.leakage_denial_repeat:
            return False
        if offer_stats.outcome != repeated:
            return False

        if self.leakage_denial_mode == "selective":
            return offer_stats.utility <= self.leakage_denial_own_ceiling
        return self.leakage_denial_mode in {"on", "aggressive"}

    def _policy_profile_leakage_denial_active(
        self,
        offer_stats: OutcomeStats,
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        opponent_view: float,
        policy_profile: str,
    ) -> bool:
        if behavior != "hardliner":
            return False
        if len(opponent.history) < 6:
            return False
        thresholds = {
            "soft_leakage_guard": (0.30, 0.84),
            "leakage_guard": (0.34, 0.78),
            "self_floor": (0.40, 0.76),
            "no_low_accept": (0.48, 0.74),
        }
        if policy_profile not in thresholds:
            return False
        utility_ceiling, opponent_floor = thresholds[policy_profile]
        if offer_stats.utility > utility_ceiling:
            return False
        if opponent_view < opponent_floor:
            return False
        if relative_time > 0.985 and offer_stats.utility >= max(0.50, utility_ceiling):
            return False
        return True

    def _prediction_accept_slack(
        self,
        current_utility: float,
        relative_time: float,
        prediction: NextOfferPrediction | None,
        prediction_review: dict[str, Any] | None,
    ) -> float:
        if self.prediction_accept_mode not in {"stall", "hybrid"}:
            return 0.0
        if prediction is None or relative_time < self.prediction_accept_time:
            return 0.0
        if prediction.confidence < self.prediction_accept_min_confidence:
            return 0.0
        if prediction.repeat_probability < self.prediction_accept_repeat:
            return 0.0
        if prediction.concession_probability > self.prediction_accept_concession:
            return 0.0
        if prediction.expected_self_utility > current_utility + 0.025:
            return 0.0
        if prediction_review and prediction_review.get("observed"):
            surprise = prediction_review.get("surprise")
            mean_error = prediction_review.get("mean_absolute_error")
            reliable = (
                isinstance(surprise, (int, float)) and surprise <= 0.22
            ) or (
                isinstance(mean_error, (int, float)) and mean_error <= 0.08
            )
            if not reliable:
                return 0.0
        return self.prediction_accept_slack


class EndPolicy:
    """Conservative walk-away policy for hopeless late-stage states."""

    def __init__(
        self,
        mode: str = "off",
        end_time: float = 0.985,
        min_history: int = 8,
        offer_slack: float = 0.005,
        safe_counter_floor: float = 0.34,
        safe_counter_margin: float = 0.04,
        safe_counter_fit: float = 0.42,
        max_slope: float = 0.015,
        min_repetition: float = 0.72,
    ) -> None:
        self.mode = mode
        self.end_time = end_time
        self.min_history = min_history
        self.offer_slack = offer_slack
        self.safe_counter_floor = safe_counter_floor
        self.safe_counter_margin = safe_counter_margin
        self.safe_counter_fit = safe_counter_fit
        self.max_slope = max_slope
        self.min_repetition = min_repetition

    def explain(
        self,
        offer_stats: OutcomeStats | None,
        next_offer: OutcomeStats,
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        reserved_value: float,
        target: float,
    ) -> EndDecision:
        floor = reserved_value + 0.02
        offer_utility = offer_stats.utility if offer_stats is not None else None
        next_fit = opponent.constraint_acceptance_estimate(next_offer.outcome)
        risk = self._agreement_risk(opponent, behavior)

        if self.mode == "off":
            return self._decision(False, "disabled", floor, offer_utility, next_offer, next_fit, risk)
        if relative_time < self.end_time:
            return self._decision(False, "too_early", floor, offer_utility, next_offer, next_fit, risk)
        if len(opponent.history) < self.min_history:
            return self._decision(False, "insufficient_history", floor, offer_utility, next_offer, next_fit, risk)
        if offer_utility is not None and offer_utility >= floor + self.offer_slack:
            return self._decision(False, "incoming_offer_above_floor", floor, offer_utility, next_offer, next_fit, risk)

        safe_counter_floor = max(
            floor + self.safe_counter_margin,
            self.safe_counter_floor,
            min(target - 0.12, 0.58),
        )
        counter_is_safe = next_offer.utility >= safe_counter_floor and next_fit >= self.safe_counter_fit
        if counter_is_safe:
            return self._decision(False, "safe_counter_available", floor, offer_utility, next_offer, next_fit, risk)
        if risk < 0.55:
            return self._decision(False, "agreement_risk_low", floor, offer_utility, next_offer, next_fit, risk)
        return self._decision(True, "late_low_offer_no_safe_counter", floor, offer_utility, next_offer, next_fit, risk)

    def _agreement_risk(self, opponent: OpponentModel, behavior: str) -> float:
        slope = opponent.concession_slope()
        repetition = self._recent_repetition(opponent)
        risk = 0.0
        if behavior in {"hardliner", "erratic"}:
            risk += 0.25
        if slope <= self.max_slope:
            risk += 0.35
        if repetition >= self.min_repetition:
            risk += 0.25
        if opponent.rejected_self_utilities and max(opponent.rejected_self_utilities[-6:]) < 0.45:
            risk += 0.15
        return max(0.0, min(1.0, risk))

    def _recent_repetition(self, opponent: OpponentModel) -> float:
        if not opponent.history:
            return 0.0
        recent = opponent.history[-6:]
        return max(Counter(recent).values()) / len(recent)

    def _decision(
        self,
        end: bool,
        reason: str,
        floor: float,
        offer_utility: float | None,
        next_offer: OutcomeStats,
        next_fit: float,
        risk: float,
    ) -> EndDecision:
        return EndDecision(
            end=end,
            reason=reason,
            floor=floor,
            offer_utility=offer_utility,
            next_utility=next_offer.utility,
            next_acceptance_fit=next_fit,
            agreement_risk=risk,
        )


class TailGuardPolicy:
    """Ultra-late lower-tail guard adapted from the ANL agent experiments."""

    def __init__(
        self,
        mode: str = "off",
        start_time: float = 0.965,
        final_time: float = 0.985,
        min_history: int = 6,
        accept_floor: float = 0.26,
        final_accept_floor: float = 0.20,
        offer_floor: float = 0.18,
        final_offer_floor: float = 0.16,
        repetition_floor: float = 0.32,
        best_seen_ceiling: float = 0.54,
        hardliner_best_seen_ceiling: float = 0.56,
        next_offer_slack: float = 0.08,
        max_slope: float = 0.06,
        fit_weight: float = 0.62,
        utility_weight: float = 0.38,
    ) -> None:
        self.mode = mode
        self.start_time = start_time
        self.final_time = final_time
        self.min_history = min_history
        self.accept_floor = accept_floor
        self.final_accept_floor = final_accept_floor
        self.offer_floor = offer_floor
        self.final_offer_floor = final_offer_floor
        self.repetition_floor = repetition_floor
        self.best_seen_ceiling = best_seen_ceiling
        self.hardliner_best_seen_ceiling = hardliner_best_seen_ceiling
        self.next_offer_slack = next_offer_slack
        self.max_slope = max_slope
        self.fit_weight = fit_weight
        self.utility_weight = utility_weight

    def active(
        self, opponent: OpponentModel, behavior: str, relative_time: float
    ) -> bool:
        if self.mode == "off":
            return False
        if relative_time < self.start_time:
            return False
        if len(opponent.history) < self.min_history:
            return False
        best_seen = max(opponent.self_utility_history or [0.0])
        repetition = self._repetition(opponent)
        slope = opponent.concession_slope()
        stalled = (
            best_seen <= self.best_seen_ceiling
            and repetition >= self.repetition_floor
            and slope <= self.max_slope
        )
        final_low_tail = (
            relative_time >= self.final_time
            and best_seen <= self.best_seen_ceiling
        )
        hardliner_tail = (
            behavior == "hardliner"
            and best_seen <= self.hardliner_best_seen_ceiling
            and repetition >= max(0.30, self.repetition_floor - 0.08)
        )
        return stalled or final_low_tail or hardliner_tail

    def choose_offer(
        self,
        catalog: list[OutcomeStats],
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        reserved_value: float,
    ) -> OutcomeStats | None:
        if not self.active(opponent, behavior, relative_time):
            return None
        floor = self._offer_floor(relative_time, reserved_value)
        repeated = Counter(opponent.history[-8:]).most_common(1)[0][0]
        candidates = [item for item in catalog if item.utility >= floor]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (
                self.fit_weight * opponent.constraint_acceptance_estimate(item.outcome)
                + self.utility_weight * item.utility
                + 0.08 * opponent.estimate(item.outcome)
                - (0.04 if item.outcome == repeated else 0.0),
                item.utility,
            ),
        )

    def should_accept(
        self,
        offer_stats: OutcomeStats,
        next_offer_stats: OutcomeStats,
        opponent: OpponentModel,
        behavior: str,
        relative_time: float,
        reserved_value: float,
    ) -> bool:
        if not self.active(opponent, behavior, relative_time):
            return False
        floor = self._accept_floor(relative_time, reserved_value)
        utility = offer_stats.utility
        if utility < floor:
            return False
        if utility + self.next_offer_slack >= next_offer_stats.utility:
            return True
        best_seen = max(opponent.self_utility_history or [utility])
        return (
            relative_time >= self.final_time
            and utility >= floor
            and (
                behavior == "hardliner"
                or self._repetition(opponent) < 0.55
                or best_seen < self.best_seen_ceiling
            )
        )

    def trace(
        self, opponent: OpponentModel, behavior: str, relative_time: float
    ) -> dict[str, Any]:
        if self.mode == "off":
            return {"mode": self.mode, "active": False}
        return {
            "mode": self.mode,
            "active": self.active(opponent, behavior, relative_time),
            "best_seen": round(max(opponent.self_utility_history or [0.0]), 6),
            "repetition": round(self._repetition(opponent), 6),
            "slope": round(opponent.concession_slope(), 6),
        }

    def _offer_floor(self, relative_time: float, reserved_value: float) -> float:
        base = self.final_offer_floor if relative_time >= self.final_time else self.offer_floor
        return max(reserved_value + 0.02, base)

    def _accept_floor(self, relative_time: float, reserved_value: float) -> float:
        base = (
            self.final_accept_floor
            if relative_time >= self.final_time
            else self.accept_floor
        )
        return max(reserved_value + 0.02, base)

    def _repetition(self, opponent: OpponentModel, window: int = 12) -> float:
        if not opponent.history:
            return 0.0
        recent = opponent.history[-max(1, window) :]
        return max(Counter(recent).values()) / len(recent)


class MessagePolicy:
    def opening(
        self,
        offer: OutcomeStats,
        phase: str,
        issue_names: list[str] | None = None,
        text_signals: TextSignals | None = None,
    ) -> str:
        summary = self._offer_text(offer.outcome, issue_names or [])
        return (
            "Let me start with a strong but workable proposal. "
            f"I am offering {summary}."
        )

    def counter(
        self,
        previous: OutcomeStats | None,
        offer: OutcomeStats,
        behavior: str,
        phase: str,
        issue_names: list[str],
        text_signals: TextSignals | None = None,
    ) -> str:
        if previous is None:
            return self.opening(offer, phase, issue_names, text_signals)

        change_text = self._change_text(previous.outcome, offer.outcome, issue_names)
        offer_text = self._offer_text(offer.outcome, issue_names)
        trade_text = self._trade_text(previous.outcome, offer.outcome)
        text_hint = self._text_hint(text_signals)
        tone = {
            "hardliner": "I need a firmer path to make this viable.",
            "conceder": "I appreciate the movement, and I can move too.",
            "reciprocal": "Your last move gives us room to trade value.",
            "fair-seeking": "This keeps us closer to a balanced agreement.",
            "erratic": "I am keeping the proposal stable and easy to evaluate.",
        }.get(behavior, "This keeps us moving toward agreement.")

        if phase == "close":
            return self._clip(
                f"We are close enough to settle. {tone} My counter is {offer_text}. "
                f"{change_text} {trade_text} {text_hint}"
            )
        return self._clip(
            f"{tone} My counter is {offer_text}. {change_text} {trade_text} {text_hint}"
        )

    def acceptance(
        self,
        offer: OutcomeStats,
        behavior: str,
        issue_names: list[str] | None = None,
    ) -> str:
        offer_text = self._offer_text(offer.outcome, issue_names or [], limit=3)
        if behavior in {"reciprocal", "conceder", "fair-seeking"}:
            return self._clip(f"I can accept this package: {offer_text}.")
        return self._clip(f"I accept this proposal: {offer_text}.")

    def end(self, reason: str) -> str:
        if reason == "late_low_offer_no_safe_counter":
            return "I cannot improve this without going below my walk-away point. I will stop here."
        return "I do not see a viable agreement path, so I will stop here."

    def _change_text(
        self, previous: Outcome, current: Outcome, issue_names: list[str]
    ) -> str:
        changes = []
        for index, (old, new) in enumerate(zip(previous, current)):
            if old == new:
                continue
            name = issue_names[index] if index < len(issue_names) else f"Issue {index + 1}"
            changes.append(f"{name}: {old} -> {new}")
            if len(changes) == 3:
                break
        remaining = sum(1 for old, new in zip(previous, current) if old != new) - len(changes)
        if not changes:
            return "I am holding this position because it remains the best bridge I see."
        suffix = f"; and {remaining} more" if remaining > 0 else ""
        return "Key adjustment: " + "; ".join(changes) + suffix + "."

    def _offer_text(
        self, outcome: Outcome, issue_names: list[str], limit: int = 4
    ) -> str:
        parts = []
        for index, value in enumerate(outcome[:limit]):
            name = issue_names[index] if index < len(issue_names) else f"Issue {index + 1}"
            parts.append(f"{name}={value}")
        if len(outcome) > limit:
            parts.append(f"{len(outcome) - limit} more terms")
        return ", ".join(parts)

    def _trade_text(self, previous: Outcome, current: Outcome) -> str:
        changes = sum(1 for old, new in zip(previous, current) if old != new)
        if changes >= 2:
            return "I am balancing the package across terms."
        return ""

    def _text_hint(self, signals: TextSignals | None) -> str:
        if signals is None or not signals.has_signal:
            return ""
        if signals.flexibility >= 0.5:
            return "I am responding to your flexibility."
        if signals.fairness >= 0.5:
            return "I am keeping the package fair."
        if signals.urgency >= 0.5:
            return "I am keeping this concise so we can close."
        return ""

    def _clip(self, text: str, limit: int = 220) -> str:
        text = " ".join(part for part in text.split() if part)
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "."


class TraceLogger:
    """Opt-in JSONL trace logger. Disabled unless NEGOTIATORX_TRACE is truthy."""

    def __init__(self) -> None:
        enabled = os.environ.get("NEGOTIATORX_TRACE", "").strip().lower()
        self.enabled = enabled in {"1", "true", "yes", "on", "debug"}
        self.top_n = self._read_int("NEGOTIATORX_TRACE_TOP", 5)
        self.path = os.environ.get("NEGOTIATORX_TRACE_FILE")
        self._stream: Any | None = None

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = {"event": event, **payload}
        line = json.dumps(record, ensure_ascii=True, sort_keys=True)
        stream = self._target_stream()
        print(line, file=stream, flush=True)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def _target_stream(self) -> Any:
        if not self.path:
            return sys.stderr
        if self._stream is None:
            self._stream = open(self.path, "a", encoding="utf-8")
        return self._stream

    def _read_int(self, name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return max(1, int(raw))
        except ValueError:
            return default


class NegotiatorX(SAOCallNegotiator):
    """Non-LLM behavioral Pareto negotiator for HAN 2026."""

    def __init__(
        self,
        early_target: float = 0.94,
        late_target: float = 0.45,
        reservation_margin: float = 0.03,
        acceptance_kwargs: dict[str, Any] | None = None,
        bidding_kwargs: dict[str, Any] | None = None,
        end_kwargs: dict[str, Any] | None = None,
        tail_guard_kwargs: dict[str, Any] | None = None,
        opponent_kwargs: dict[str, Any] | None = None,
        text_mode: str = "off",
        behavior_mode: str = "simple",
        dynamic_slope_mode: str = "dense_numeric_late",
        profile_selector_mode: str = "off",
        export_mode: str = "shape_adaptive",
        export_kwargs: dict[str, float] | None = None,
        policy_switch_mode: str = "agreement_rescue",
        policy_switch_time: float = 0.84,
        policy_switch_min_history: int = 6,
        policy_switch_min_confidence: float = 0.55,
        policy_switch_max_offer_utility: float = 0.42,
        policy_switch_min_opponent_view: float = 0.76,
        policy_switch_min_repetition: float = 0.58,
        policy_switch_max_slope: float = 0.08,
        policy_switch_min_risk: float = 0.64,
        **kwargs: Any,
    ) -> None:
        for legacy_key in (
            "model",
            "temperature",
            "max_tokens",
            "use_structured_output",
            "timeout",
            "num_retries",
            "llm_kwargs",
            "system_prompt",
            "preferences_prompt",
            "preferences_changed_prompt",
            "negotiation_start_prompt",
            "round_prompt",
        ):
            kwargs.pop(legacy_key, None)

        super().__init__(**kwargs)
        early_target = self._env_float("NEGOTIATORX_EARLY_TARGET", early_target)
        late_target = self._env_float("NEGOTIATORX_LATE_TARGET", late_target)
        behavior_mode = os.getenv("NEGOTIATORX_BEHAVIOR_MODE", behavior_mode)
        profile_selector_mode = os.getenv(
            "NEGOTIATORX_PROFILE_SELECTOR_MODE",
            profile_selector_mode,
        )
        dynamic_slope_mode = os.getenv(
            "NEGOTIATORX_DYNAMIC_SLOPE_MODE",
            dynamic_slope_mode,
        )
        export_mode = os.getenv("NEGOTIATORX_EXPORT_MODE", export_mode)
        policy_switch_mode = os.getenv(
            "NEGOTIATORX_POLICY_SWITCH_MODE", policy_switch_mode
        )
        policy_switch_time = self._env_float(
            "NEGOTIATORX_POLICY_SWITCH_TIME", policy_switch_time
        )
        policy_switch_min_confidence = self._env_float(
            "NEGOTIATORX_POLICY_SWITCH_MIN_CONFIDENCE",
            policy_switch_min_confidence,
        )
        policy_switch_max_offer_utility = self._env_float(
            "NEGOTIATORX_POLICY_SWITCH_MAX_OFFER_UTILITY",
            policy_switch_max_offer_utility,
        )
        policy_switch_min_opponent_view = self._env_float(
            "NEGOTIATORX_POLICY_SWITCH_MIN_OPPONENT_VIEW",
            policy_switch_min_opponent_view,
        )
        policy_switch_min_repetition = self._env_float(
            "NEGOTIATORX_POLICY_SWITCH_MIN_REPETITION",
            policy_switch_min_repetition,
        )
        policy_switch_max_slope = self._env_float(
            "NEGOTIATORX_POLICY_SWITCH_MAX_SLOPE",
            policy_switch_max_slope,
        )
        policy_switch_min_risk = self._env_float(
            "NEGOTIATORX_POLICY_SWITCH_MIN_RISK",
            policy_switch_min_risk,
        )
        policy_switch_min_history = self._env_int(
            "NEGOTIATORX_POLICY_SWITCH_MIN_HISTORY",
            policy_switch_min_history,
        )
        merged_bidding_kwargs = dict(bidding_kwargs or {})
        merged_bidding_kwargs.update(self._env_bidding_kwargs())
        merged_acceptance_kwargs = dict(acceptance_kwargs or {})
        merged_acceptance_kwargs.update(self._env_acceptance_kwargs())
        merged_end_kwargs = dict(end_kwargs or {})
        merged_end_kwargs.update(self._env_end_kwargs())
        merged_tail_guard_kwargs = {
            "mode": "q1_guard",
            "start_time": 0.965,
            "final_time": 0.985,
            "accept_floor": 0.26,
            "final_accept_floor": 0.20,
            "offer_floor": 0.18,
            "final_offer_floor": 0.16,
            "repetition_floor": 0.32,
            "best_seen_ceiling": 0.54,
            "hardliner_best_seen_ceiling": 0.56,
            "next_offer_slack": 0.08,
        }
        merged_tail_guard_kwargs.update(tail_guard_kwargs or {})
        merged_opponent_kwargs = dict(opponent_kwargs or {})
        merged_opponent_kwargs.update(self._env_opponent_kwargs())
        merged_export_kwargs = dict(export_kwargs or {})
        merged_export_kwargs.update(self._env_export_kwargs())
        self._jitter_profile = self._build_jitter_profile()
        if self._jitter_profile.enabled:
            early_target, late_target = self._apply_jitter_profile(
                early_target,
                late_target,
                merged_bidding_kwargs,
                merged_acceptance_kwargs,
            )
        self._catalog: list[OutcomeStats] | None = None
        self._catalog_by_outcome: dict[Outcome, OutcomeStats] = {}
        self._issue_names: list[str] = []
        self._min_utility = 0.0
        self._max_utility = 1.0
        self._numeric_issue_count = 0
        self._issue_count = 0
        self._opponent: OpponentModel | None = None
        self._base_slope_mode = str(merged_opponent_kwargs.get("slope_mode", "legacy"))
        self._dynamic_slope_mode = dynamic_slope_mode
        self._profile_selector_mode = profile_selector_mode
        self._opponent_kwargs = merged_opponent_kwargs
        self._text_mode = os.getenv("NEGOTIATORX_TEXT_MODE", text_mode)
        self._text_model = OpponentTextModel()
        self._behavior = BehaviorModel(mode=behavior_mode)
        self._bidding = BiddingPolicy(
            early_target=early_target,
            late_target=late_target,
            reservation_margin=reservation_margin,
            **merged_bidding_kwargs,
        )
        self._acceptance = AcceptancePolicy(**merged_acceptance_kwargs)
        self._end_policy = EndPolicy(**merged_end_kwargs)
        self._tail_guard = TailGuardPolicy(**merged_tail_guard_kwargs)
        self._messages = MessagePolicy()
        self._trace = TraceLogger()
        self._export_mode = export_mode
        self._export_kwargs = merged_export_kwargs
        self._policy_switch_mode = policy_switch_mode
        self._policy_switch_time = policy_switch_time
        self._policy_switch_min_history = policy_switch_min_history
        self._policy_switch_min_confidence = policy_switch_min_confidence
        self._policy_switch_max_offer_utility = policy_switch_max_offer_utility
        self._policy_switch_min_opponent_view = policy_switch_min_opponent_view
        self._policy_switch_min_repetition = policy_switch_min_repetition
        self._policy_switch_max_slope = policy_switch_max_slope
        self._policy_switch_min_risk = policy_switch_min_risk
        self._pending_prediction: NextOfferPrediction | None = None
        self._prediction_abs_errors: list[float] = []
        self._turn_index = 0
        self._last_counter: OutcomeStats | None = None
        self._catalog_build_seconds = 0.0

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _build_jitter_profile(cls) -> JitterProfile:
        if not cls._env_bool("NEGOTIATORX_JITTER", False):
            return JitterProfile(False, None, {})
        seed = cls._env_int("NEGOTIATORX_JITTER_SEED", 451551)
        scale = max(0.0, cls._env_float("NEGOTIATORX_JITTER_SCALE", 1.0))
        rng = random.Random(seed)

        def sample(name: str, default: float, low: float, high: float) -> float:
            low = default + (low - default) * scale
            high = default + (high - default) * scale
            if low > high:
                low, high = high, low
            override = os.getenv(f"NEGOTIATORX_JITTER_{name.upper()}")
            if override is not None:
                try:
                    return float(override)
                except ValueError:
                    pass
            return rng.uniform(low, high)

        values = {
            "anchor_end": sample("anchor_end", 0.18, 0.16, 0.20),
            "explore_end": sample("explore_end", 0.48, 0.46, 0.50),
            "close_start": sample("close_start", 0.82, 0.80, 0.84),
            "early_target": sample("early_target", 0.94, 0.92, 0.96),
            "late_target": sample("late_target", 0.45, 0.43, 0.48),
            "model_minimum": sample("model_minimum", 0.66, 0.64, 0.69),
            "late_time": sample("late_time", 0.90, 0.86, 0.93),
            "late_slack": sample("late_slack", 0.02, 0.01, 0.04),
            "constraint_probe_time": sample(
                "constraint_probe_time", 0.82, 0.80, 0.86
            ),
            "constraint_probe_floor": sample(
                "constraint_probe_floor", 0.10, 0.08, 0.14
            ),
            "constraint_probe_min_fit": sample(
                "constraint_probe_min_fit", 0.72, 0.68, 0.78
            ),
            "anneal_temperature": sample("anneal_temperature", 0.06, 0.03, 0.08),
            "repeat_penalty_scale": sample("repeat_penalty_scale", 1.00, 0.85, 1.15),
        }
        return JitterProfile(True, seed, values)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _apply_jitter_profile(
        self,
        early_target: float,
        late_target: float,
        bidding_kwargs: dict[str, Any],
        acceptance_kwargs: dict[str, Any],
    ) -> tuple[float, float]:
        values = self._jitter_profile.values
        if "NEGOTIATORX_EARLY_TARGET" not in os.environ:
            early_target = values["early_target"]
        if "NEGOTIATORX_LATE_TARGET" not in os.environ:
            late_target = values["late_target"]
        for key in (
            "anchor_end",
            "explore_end",
            "close_start",
            "constraint_probe_time",
            "constraint_probe_floor",
            "constraint_probe_min_fit",
            "anneal_temperature",
            "repeat_penalty_scale",
        ):
            bidding_kwargs.setdefault(key, values[key])
        for key in ("model_minimum", "late_time", "late_slack"):
            acceptance_kwargs.setdefault(key, values[key])
        return early_target, late_target

    @classmethod
    def _env_bidding_kwargs(cls) -> dict[str, Any]:
        string_keys = {
            "NEGOTIATORX_PARETO_FILTER": "pareto_filter",
            "NEGOTIATORX_LATE_TARGET_MODE": "late_target_mode",
            "NEGOTIATORX_RISK_CLOSE_MODE": "risk_close_mode",
            "NEGOTIATORX_DEAL_MODE": "deal_mode",
            "NEGOTIATORX_REPEATER_RESCUE_MODE": "repeater_rescue_mode",
            "NEGOTIATORX_NUMERIC_RESCUE_MODE": "numeric_rescue_mode",
            "NEGOTIATORX_TARGET_CURVE_MODE": "target_curve_mode",
            "NEGOTIATORX_RANK_AWARE_DENIAL": "rank_aware_denial",
            "NEGOTIATORX_SELECTION_MODE": "selection_mode",
            "NEGOTIATORX_OPPONENT_HORIZON_MODE": "opponent_horizon_mode",
            "NEGOTIATORX_CONSTRAINT_PROBE_MODE": "constraint_probe_mode",
            "NEGOTIATORX_CONSTRAINT_PROBE_FLOOR_MODE": (
                "constraint_probe_floor_mode"
            ),
            "NEGOTIATORX_PREDICTION_POLICY": "prediction_policy",
            "NEGOTIATORX_LOGROLLING_MODE": "logrolling_mode",
        }
        float_keys = {
            "NEGOTIATORX_ANCHOR_END": "anchor_end",
            "NEGOTIATORX_EXPLORE_END": "explore_end",
            "NEGOTIATORX_CLOSE_START": "close_start",
            "NEGOTIATORX_PARETO_EPSILON": "pareto_epsilon",
            "NEGOTIATORX_CLOSE_OPPONENT_SCALE": "close_opponent_scale",
            "NEGOTIATORX_CLOSE_NASH_SCALE": "close_nash_scale",
            "NEGOTIATORX_CLOSE_SIMILARITY_SCALE": "close_similarity_scale",
            "NEGOTIATORX_REPEAT_PENALTY_SCALE": "repeat_penalty_scale",
            "NEGOTIATORX_RISK_CLOSE_YIELD": "risk_close_yield",
            "NEGOTIATORX_RISK_CLOSE_HOLD": "risk_close_hold",
            "NEGOTIATORX_DEAL_START": "deal_start",
            "NEGOTIATORX_DEAL_BUDGET": "deal_budget",
            "NEGOTIATORX_DEAL_FIT_SCALE": "deal_fit_scale",
            "NEGOTIATORX_REPEATER_RESCUE_TIME": "repeater_rescue_time",
            "NEGOTIATORX_REPEATER_RESCUE_FLOOR": "repeater_rescue_floor",
            "NEGOTIATORX_REPEATER_RESCUE_FIT": "repeater_rescue_fit",
            "NEGOTIATORX_ALLOCATION_RESCUE_TIME": "allocation_rescue_time",
            "NEGOTIATORX_ALLOCATION_RESCUE_SIMILARITY": (
                "allocation_rescue_similarity"
            ),
            "NEGOTIATORX_ALLOCATION_RESCUE_MODEL_FLOOR": (
                "allocation_rescue_model_floor"
            ),
            "NEGOTIATORX_ALLOCATION_RESCUE_REPEAT_PENALTY": (
                "allocation_rescue_repeat_penalty"
            ),
            "NEGOTIATORX_NUMERIC_RESCUE_TIME": "numeric_rescue_time",
            "NEGOTIATORX_NUMERIC_RESCUE_SIMILARITY": (
                "numeric_rescue_similarity"
            ),
            "NEGOTIATORX_NUMERIC_RESCUE_FLOOR": "numeric_rescue_floor",
            "NEGOTIATORX_NUMERIC_RESCUE_MODEL_FLOOR": (
                "numeric_rescue_model_floor"
            ),
            "NEGOTIATORX_NUMERIC_RESCUE_REPEAT_PENALTY": (
                "numeric_rescue_repeat_penalty"
            ),
            "NEGOTIATORX_SIGMOID_MIDPOINT": "sigmoid_midpoint",
            "NEGOTIATORX_SIGMOID_STEEPNESS": "sigmoid_steepness",
            "NEGOTIATORX_ADAPTIVE_FLOOR_SHIFT": "adaptive_floor_shift",
            "NEGOTIATORX_DENIAL_TIME": "denial_time",
            "NEGOTIATORX_DENIAL_OWN_CEILING": "denial_own_ceiling",
            "NEGOTIATORX_DENIAL_OPPONENT_FLOOR": "denial_opponent_floor",
            "NEGOTIATORX_DENIAL_SIMILARITY_FLOOR": "denial_similarity_floor",
            "NEGOTIATORX_DENIAL_KEEP_FLOOR": "denial_keep_floor",
            "NEGOTIATORX_ANNEAL_TEMPERATURE": "anneal_temperature",
            "NEGOTIATORX_ANNEAL_COOLING": "anneal_cooling",
            "NEGOTIATORX_SHORT_HORIZON_WEIGHT": "short_horizon_weight",
            "NEGOTIATORX_CONSTRAINT_PROBE_TIME": "constraint_probe_time",
            "NEGOTIATORX_CONSTRAINT_PROBE_FLOOR": "constraint_probe_floor",
            "NEGOTIATORX_CONSTRAINT_PROBE_BUDGET_START": (
                "constraint_probe_budget_start"
            ),
            "NEGOTIATORX_CONSTRAINT_PROBE_BUDGET_END": (
                "constraint_probe_budget_end"
            ),
            "NEGOTIATORX_CONSTRAINT_PROBE_RELAX_BUDGET": (
                "constraint_probe_relax_budget"
            ),
            "NEGOTIATORX_CONSTRAINT_PROBE_MIN_FIT": "constraint_probe_min_fit",
            "NEGOTIATORX_CONSTRAINT_PROBE_REPEAT": "constraint_probe_repeat",
            "NEGOTIATORX_CONSTRAINT_PROBE_UTILITY_WEIGHT": (
                "constraint_probe_utility_weight"
            ),
            "NEGOTIATORX_CONSTRAINT_PROBE_FIT_WEIGHT": (
                "constraint_probe_fit_weight"
            ),
            "NEGOTIATORX_CONSTRAINT_PROBE_SIMILARITY_WEIGHT": (
                "constraint_probe_similarity_weight"
            ),
            "NEGOTIATORX_CONSTRAINT_PROBE_REJECTION_PENALTY": (
                "constraint_probe_rejection_penalty"
            ),
            "NEGOTIATORX_PREDICTION_MIN_CONFIDENCE": (
                "prediction_min_confidence"
            ),
            "NEGOTIATORX_PREDICTION_LOW_SURPRISE": "prediction_low_surprise",
            "NEGOTIATORX_PREDICTION_HOLD_SHIFT": "prediction_hold_shift",
            "NEGOTIATORX_PREDICTION_PROBE_TIME_SHIFT": (
                "prediction_probe_time_shift"
            ),
            "NEGOTIATORX_PREDICTION_PROBE_REPEAT": "prediction_probe_repeat",
            "NEGOTIATORX_PREDICTION_PROBE_CONCESSION": (
                "prediction_probe_concession"
            ),
            "NEGOTIATORX_LOGROLLING_WEIGHT": "logrolling_weight",
        }
        int_keys = {
            "NEGOTIATORX_ANNEAL_TOP_K": "anneal_top_k",
            "NEGOTIATORX_SHORT_HORIZON": "short_horizon",
        }

        kwargs: dict[str, Any] = {}
        for env_name, key in string_keys.items():
            raw = os.getenv(env_name)
            if raw is not None:
                kwargs[key] = raw
        for env_name, key in float_keys.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            try:
                kwargs[key] = float(raw)
            except ValueError:
                continue
        for env_name, key in int_keys.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            try:
                kwargs[key] = int(raw)
            except ValueError:
                continue
        return kwargs

    @classmethod
    def _env_acceptance_kwargs(cls) -> dict[str, Any]:
        string_keys = {
            "NEGOTIATORX_PREDICTION_ACCEPT_MODE": "prediction_accept_mode",
            "NEGOTIATORX_LEAKAGE_DENIAL_MODE": "leakage_denial_mode",
        }
        float_keys = {
            "NEGOTIATORX_NEXT_GAP": "next_gap",
            "NEGOTIATORX_TARGET_BONUS": "target_bonus",
            "NEGOTIATORX_LATE_TIME": "late_time",
            "NEGOTIATORX_HARDLINER_MINIMUM": "hardliner_minimum",
            "NEGOTIATORX_MODEL_MINIMUM": "model_minimum",
            "NEGOTIATORX_LATE_SLACK": "late_slack",
            "NEGOTIATORX_PREDICTION_ACCEPT_TIME": "prediction_accept_time",
            "NEGOTIATORX_PREDICTION_ACCEPT_SLACK": "prediction_accept_slack",
            "NEGOTIATORX_PREDICTION_ACCEPT_MIN_CONFIDENCE": (
                "prediction_accept_min_confidence"
            ),
            "NEGOTIATORX_PREDICTION_ACCEPT_REPEAT": "prediction_accept_repeat",
            "NEGOTIATORX_PREDICTION_ACCEPT_CONCESSION": (
                "prediction_accept_concession"
            ),
            "NEGOTIATORX_LEAKAGE_DENIAL_TIME": "leakage_denial_time",
            "NEGOTIATORX_LEAKAGE_DENIAL_OWN_CEILING": (
                "leakage_denial_own_ceiling"
            ),
            "NEGOTIATORX_LEAKAGE_DENIAL_OPPONENT_FLOOR": (
                "leakage_denial_opponent_floor"
            ),
            "NEGOTIATORX_LEAKAGE_DENIAL_REPEAT": "leakage_denial_repeat",
        }
        bool_keys = {
            "NEGOTIATORX_ENABLE_HARDLINER_ACCEPTANCE": (
                "enable_hardliner_acceptance"
            ),
            "NEGOTIATORX_ENABLE_MODEL_ACCEPTANCE": "enable_model_acceptance",
        }
        kwargs: dict[str, Any] = {}
        for env_name, key in string_keys.items():
            raw = os.getenv(env_name)
            if raw is not None:
                kwargs[key] = raw
        for env_name, key in float_keys.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            try:
                kwargs[key] = float(raw)
            except ValueError:
                continue
        for env_name, key in bool_keys.items():
            raw = os.getenv(env_name)
            if raw is not None:
                kwargs[key] = cls._env_bool(env_name, True)
        return kwargs

    @classmethod
    def _env_end_kwargs(cls) -> dict[str, Any]:
        string_keys = {
            "NEGOTIATORX_END_MODE": "mode",
        }
        float_keys = {
            "NEGOTIATORX_END_TIME": "end_time",
            "NEGOTIATORX_END_OFFER_SLACK": "offer_slack",
            "NEGOTIATORX_END_SAFE_COUNTER_FLOOR": "safe_counter_floor",
            "NEGOTIATORX_END_SAFE_COUNTER_MARGIN": "safe_counter_margin",
            "NEGOTIATORX_END_SAFE_COUNTER_FIT": "safe_counter_fit",
            "NEGOTIATORX_END_MAX_SLOPE": "max_slope",
            "NEGOTIATORX_END_MIN_REPETITION": "min_repetition",
        }
        int_keys = {
            "NEGOTIATORX_END_MIN_HISTORY": "min_history",
        }
        kwargs: dict[str, Any] = {}
        for env_name, key in string_keys.items():
            raw = os.getenv(env_name)
            if raw is not None:
                kwargs[key] = raw
        for env_name, key in float_keys.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            try:
                kwargs[key] = float(raw)
            except ValueError:
                continue
        for env_name, key in int_keys.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            try:
                kwargs[key] = int(raw)
            except ValueError:
                continue
        return kwargs

    @classmethod
    def _env_opponent_kwargs(cls) -> dict[str, Any]:
        string_keys = {
            "NEGOTIATORX_ISSUE_WEIGHT_MODE": "issue_weight_mode",
            "NEGOTIATORX_SLOPE_MODE": "slope_mode",
        }
        kwargs: dict[str, Any] = {}
        for env_name, key in string_keys.items():
            raw = os.getenv(env_name)
            if raw is not None:
                kwargs[key] = raw
        return kwargs

    @classmethod
    def _env_export_kwargs(cls) -> dict[str, float]:
        float_keys = {
            "NEGOTIATORX_EXPORT_TEMPORAL_WEIGHT": "temporal_weight",
            "NEGOTIATORX_EXPORT_ISSUE_TEMPORAL_WEIGHT": "issue_temporal_weight",
            "NEGOTIATORX_EXPORT_FREQUENCY_WEIGHT": "frequency_weight",
            "NEGOTIATORX_EXPORT_TEMPORAL_FLOOR": "temporal_floor",
            "NEGOTIATORX_EXPORT_TEMPORAL_POWER": "temporal_power",
        }
        kwargs: dict[str, float] = {}
        for env_name, key in float_keys.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            try:
                kwargs[key] = float(raw)
            except ValueError:
                continue
        return kwargs

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun is not None
        self._ensure_ready()
        assert self._catalog is not None
        assert self._opponent is not None
        self._turn_index += 1

        offer = state.current_offer
        relative_time = max(0.0, min(1.0, state.relative_time))
        dynamic_slope_mode = self._contextual_dynamic_slope_mode(relative_time)
        self._opponent.slope_mode = self._effective_slope_mode(
            relative_time, dynamic_slope_mode
        )
        reserved_value = self._normalize_utility(float(self.ufun.reserved_value or 0.0))

        offer_stats = self._stats_for(offer) if offer is not None else None
        if self._last_counter is not None:
            self._opponent.observe_rejection(
                self._last_counter.outcome,
                self._last_counter.utility,
            )
            self._last_counter = None
        incoming_prediction = self._pending_prediction
        self._pending_prediction = None
        observed_offers = self._observe_opponent_offers(state)
        prediction_review = self._review_prediction(
            incoming_prediction, bool(observed_offers)
        )
        text_signals = self._observe_opponent_text(state)

        behavior = self._behavior.classify(self._opponent)
        if self._text_mode != "off":
            behavior = self._text_model.adjust_behavior(behavior, text_signals)
        policy_profile = self._policy_profile(
            offer_stats,
            behavior,
            relative_time,
        )
        next_prediction = self._predict_next_opponent_offer(
            relative_time, behavior
        )
        next_offer = self._bidding.choose(
            self._catalog,
            self._opponent,
            behavior,
            relative_time,
            reserved_value,
            self._turn_index,
            next_prediction,
            prediction_review,
            policy_profile,
        )
        target = self._bidding.effective_target(
            relative_time,
            behavior,
            self._opponent,
            next_prediction,
            prediction_review,
            policy_profile,
        )
        tail_guard_offer = self._tail_guard.choose_offer(
            self._catalog,
            self._opponent,
            behavior,
            relative_time,
            reserved_value,
        )
        tail_guard_override = False
        if tail_guard_offer is not None:
            next_offer = tail_guard_offer
            tail_guard_override = True
        phase = self._bidding.phase(relative_time)
        bidding_trace = self._bidding_trace(
            behavior,
            relative_time,
            reserved_value,
            next_offer,
            next_prediction,
            prediction_review,
            policy_profile,
        )
        bidding_trace["tail_guard"] = {
            **self._tail_guard.trace(self._opponent, behavior, relative_time),
            "override": tail_guard_override,
        }

        acceptance_decision = None
        if offer_stats is not None:
            acceptance_decision = self._acceptance.explain(
                offer_stats,
                next_offer,
                self._opponent,
                behavior,
                relative_time,
                reserved_value,
                target,
                next_prediction,
                prediction_review,
                policy_profile,
            )
            if (
                not acceptance_decision.accept
                and self._tail_guard.should_accept(
                    offer_stats,
                    next_offer,
                    self._opponent,
                    behavior,
                    relative_time,
                    reserved_value,
                )
            ):
                acceptance_decision = AcceptanceDecision(
                    True,
                    "q1_tail_guard_accept",
                    self._tail_guard._accept_floor(relative_time, reserved_value),
                    offer_stats.utility,
                    next_offer.utility,
                    target,
                    self._opponent.estimate(offer_stats.outcome),
                    self._opponent.estimate(next_offer.outcome),
                )

        if acceptance_decision is not None and acceptance_decision.accept:
            message = self._messages.acceptance(
                offer_stats,
                behavior,
                self._issue_names,
            )
            self._last_counter = None
            self._emit_turn_trace(
                state=state,
                observed_offers=observed_offers,
                offer_stats=offer_stats,
                behavior=behavior,
                policy_profile=policy_profile,
                phase=phase,
                target=target,
                reserved_value=reserved_value,
                next_offer=next_offer,
                bidding_trace=bidding_trace,
                acceptance_decision=acceptance_decision,
                end_decision=None,
                prediction_review=prediction_review,
                next_prediction=None,
                action="accept",
                response_offer=offer,
                message=message,
                text_signals=text_signals,
            )
            return SAOResponse(
                ResponseType.ACCEPT_OFFER,
                offer,
                data=dict(text=message),
            )

        end_decision = self._end_policy.explain(
            offer_stats,
            next_offer,
            self._opponent,
            behavior,
            relative_time,
            reserved_value,
            target,
        )
        if end_decision.end:
            message = self._messages.end(end_decision.reason)
            self._last_counter = None
            self._pending_prediction = None
            self._emit_turn_trace(
                state=state,
                observed_offers=observed_offers,
                offer_stats=offer_stats,
                behavior=behavior,
                policy_profile=policy_profile,
                phase=phase,
                target=target,
                reserved_value=reserved_value,
                next_offer=next_offer,
                bidding_trace=bidding_trace,
                acceptance_decision=acceptance_decision,
                end_decision=end_decision,
                prediction_review=prediction_review,
                next_prediction=None,
                action="end",
                response_offer=None,
                message=message,
                text_signals=text_signals,
            )
            return SAOResponse(
                ResponseType.END_NEGOTIATION,
                None,
                data=dict(text=message),
            )

        self._bidding.remember(next_offer.outcome)
        self._last_counter = next_offer
        self._pending_prediction = next_prediction
        message = self._messages.counter(
            offer_stats,
            next_offer,
            behavior,
            phase,
            self._issue_names,
            text_signals,
        )
        self._emit_turn_trace(
            state=state,
            observed_offers=observed_offers,
            offer_stats=offer_stats,
            behavior=behavior,
            policy_profile=policy_profile,
            phase=phase,
            target=target,
            reserved_value=reserved_value,
            next_offer=next_offer,
            bidding_trace=bidding_trace,
            acceptance_decision=acceptance_decision,
            end_decision=end_decision,
            prediction_review=prediction_review,
            next_prediction=next_prediction,
            action="counter",
            response_offer=next_offer.outcome,
            message=message,
            text_signals=text_signals,
        )
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            next_offer.outcome,
            data=dict(text=message),
        )

    def _ensure_ready(self) -> None:
        if self._catalog is not None:
            return
        assert self.ufun is not None
        assert self.nmi is not None

        started = perf_counter()
        outcomes = list(self.nmi.outcome_space.enumerate())
        issue_count = len(outcomes[0]) if outcomes else 0
        self._issue_count = issue_count
        self._issue_names = self._read_issue_names(issue_count)
        self._text_model.configure(self._issue_names)
        numeric_ranges = self._numeric_ranges(outcomes, issue_count)
        self._numeric_issue_count = sum(
            1
            for index in range(issue_count)
            if all(isinstance(outcome[index], (int, float)) for outcome in outcomes)
        )

        catalog = []
        raw_utilities = [float(self.ufun(outcome)) for outcome in outcomes]
        self._min_utility = min(raw_utilities) if raw_utilities else 0.0
        self._max_utility = max(raw_utilities) if raw_utilities else 1.0

        catalog = []
        for outcome, raw_utility in zip(outcomes, raw_utilities):
            utility = self._normalize_utility(raw_utility)
            normalized = tuple(
                self._normalize_value(value, numeric_ranges[index])
                for index, value in enumerate(outcome)
            )
            item = OutcomeStats(
                outcome=outcome,
                raw_utility=raw_utility,
                utility=utility,
                values=tuple(outcome),
                normalized_values=normalized,
                bucket=int(max(0.0, min(1.0, utility)) * 20),
            )
            catalog.append(item)

        catalog.sort(key=lambda item: item.utility, reverse=True)
        self._catalog = catalog
        self._catalog_by_outcome = {item.outcome: item for item in catalog}
        opponent_kwargs = dict(self._opponent_kwargs)
        if opponent_kwargs.get("slope_mode") == "shape_adaptive":
            opponent_kwargs["slope_mode"] = self._shape_adaptive_slope_mode(
                len(outcomes),
                issue_count,
                self._numeric_issue_count,
            )
        self._opponent = OpponentModel(issue_count, **opponent_kwargs)
        self._bidding.configure_catalog(catalog)
        self._catalog_build_seconds = perf_counter() - started

    def _effective_slope_mode(
        self, relative_time: float, dynamic_slope_mode: str | None = None
    ) -> str:
        mode = dynamic_slope_mode or self._dynamic_slope_mode
        if (
            mode == "off"
            or self._base_slope_mode != "legacy"
            or self._opponent is None
        ):
            return self._opponent.slope_mode if self._opponent is not None else "legacy"
        if mode not in {"dense_numeric_late", "late_stall"}:
            return self._base_slope_mode
        if mode == "dense_numeric_late":
            if self._numeric_issue_count < 3 or relative_time < 0.80:
                return self._base_slope_mode
            if len(self._opponent.self_utility_history) < 4:
                return self._base_slope_mode
        else:
            if relative_time < 0.78:
                return self._base_slope_mode
            if len(self._opponent.self_utility_history) < 4:
                return self._base_slope_mode
            dense_numeric = self._numeric_issue_count >= 3
            low_dim_numeric = self._numeric_issue_count >= 1 and self._issue_count <= 3
            if not dense_numeric and not low_dim_numeric:
                return self._base_slope_mode
        if len(self._opponent.self_utility_history) < 4:
            return self._base_slope_mode

        recent = self._opponent.history[-6:]
        repetition = max(Counter(recent).values()) / len(recent) if recent else 0.0
        values = self._opponent.self_utility_history
        first = sum(values[:2]) / 2.0
        last = sum(values[-2:]) / 2.0
        legacy_slope = last - first
        recent_span = max(values[-6:]) - min(values[-6:])
        if mode == "dense_numeric_late":
            risky_stall = legacy_slope <= 0.04 or repetition >= 0.65
            if risky_stall and recent_span <= 0.28:
                return "regression"
            return self._base_slope_mode

        dense_numeric = self._numeric_issue_count >= 3
        low_dim_numeric = self._numeric_issue_count >= 1 and self._issue_count <= 3
        risky_stall = legacy_slope <= 0.045 or repetition >= 0.60
        span_limit = 0.28 if dense_numeric else 0.34
        if risky_stall and recent_span <= span_limit:
            return "regression" if dense_numeric else "block"
        return self._base_slope_mode

    def _contextual_dynamic_slope_mode(self, relative_time: float) -> str:
        if self._profile_selector_mode != "contextual_tail":
            return self._dynamic_slope_mode
        if self._opponent is None or len(self._opponent.history) < 4:
            return self._dynamic_slope_mode
        if relative_time < 0.78:
            return self._dynamic_slope_mode
        if self._numeric_issue_count <= 0:
            return self._dynamic_slope_mode
        best_seen = max(self._opponent.self_utility_history or [1.0])
        repetition = self._recent_repetition()
        slope = self._opponent.concession_slope()
        low_dimensional = self._issue_count <= 3
        dense_numeric = self._numeric_issue_count >= 3
        if (
            (low_dimensional or dense_numeric)
            and best_seen <= 0.60
            and (repetition >= 0.55 or slope <= 0.055)
        ):
            return "late_stall"
        return self._dynamic_slope_mode

    def _shape_adaptive_slope_mode(
        self,
        outcome_count: int,
        issue_count: int,
        numeric_issue_count: int,
    ) -> str:
        if issue_count >= 6 and numeric_issue_count == 0:
            return "legacy"
        if numeric_issue_count >= 3:
            return "regression"
        if numeric_issue_count == 2 and outcome_count > 50:
            return "block"
        return "legacy"

    def _stats_for(self, outcome: Outcome | None) -> OutcomeStats | None:
        if outcome is None:
            return None
        existing = self._catalog_by_outcome.get(outcome)
        if existing is not None:
            return existing
        assert self.ufun is not None
        raw_utility = float(self.ufun(outcome))
        utility = self._normalize_utility(raw_utility)
        return OutcomeStats(
            outcome=outcome,
            raw_utility=raw_utility,
            utility=utility,
            values=tuple(outcome),
            normalized_values=tuple(0.5 for _ in outcome),
            bucket=int(max(0.0, min(1.0, utility)) * 20),
        )

    def _is_own_current_offer(self, state: SAOState) -> bool:
        return self._is_own_proposer(state.current_proposer)

    def _is_own_proposer(self, proposer: str | None) -> bool:
        if not proposer:
            return False
        own_names = {str(self.name), str(getattr(self, "id", ""))}
        return proposer in own_names or any(
            proposer.startswith(f"{name}@") for name in own_names if name
        )

    def _observe_opponent_offers(self, state: SAOState) -> list[dict[str, Any]]:
        assert self._opponent is not None
        observed: set[tuple[str, Outcome]] = set()
        trace_items = []
        for proposer, outcome in state.new_offers or []:
            if outcome is None or self._is_own_proposer(proposer):
                continue
            key = (str(proposer), outcome)
            if key in observed:
                continue
            observed.add(key)
            stats = self._stats_for(outcome)
            if stats is not None:
                self._opponent.observe(outcome, stats.utility)
                trace_items.append(
                    {
                        "source": "new_offers",
                        "proposer": str(proposer),
                        "outcome": self._format_outcome(outcome),
                        "our_utility": round(stats.utility, 6),
                        "raw_utility": round(stats.raw_utility, 6),
                    }
                )

        if observed or state.current_offer is None or self._is_own_current_offer(state):
            return trace_items

        stats = self._stats_for(state.current_offer)
        if stats is not None:
            self._opponent.observe(state.current_offer, stats.utility)
            trace_items.append(
                {
                    "source": "current_offer_fallback",
                    "proposer": str(state.current_proposer),
                    "outcome": self._format_outcome(state.current_offer),
                    "our_utility": round(stats.utility, 6),
                    "raw_utility": round(stats.raw_utility, 6),
                }
            )
        return trace_items

    def _observe_opponent_text(self, state: SAOState) -> TextSignals:
        if self._text_mode == "off":
            return TextSignals(text_count=0)
        texts: list[str] = []
        seen: set[tuple[str, str]] = set()
        for proposer, data in getattr(state, "new_data", None) or []:
            if data is None or self._is_own_proposer(proposer):
                continue
            text = data.get("text") if isinstance(data, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            key = (str(proposer), text)
            if key in seen:
                continue
            seen.add(key)
            texts.append(text)

        current_data = getattr(state, "current_data", None)
        if (
            not texts
            and isinstance(current_data, dict)
            and not self._is_own_current_offer(state)
        ):
            text = current_data.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        return self._text_model.observe(texts)

    @property
    def opponent_ufun(self) -> MappingUtilityFunction | None:
        if self._catalog is None or self._opponent is None:
            return None
        if not self._opponent.history:
            return None
        mapping = {
            item.outcome: self._opponent.export_estimate(
                item.outcome, **self._effective_export_kwargs()
            )
            for item in self._catalog
        }
        return MappingUtilityFunction(mapping, default=0.0)

    def _effective_export_kwargs(self) -> dict[str, float]:
        kwargs = dict(self._export_kwargs)
        if self._export_mode == "shape_adaptive" and self._is_dense_numeric_domain():
            kwargs.setdefault("temporal_weight", 0.35)
            kwargs.setdefault("issue_temporal_weight", 0.20)
            kwargs.setdefault("frequency_weight", 0.45)
        return kwargs

    def _is_dense_numeric_domain(self) -> bool:
        return self._issue_count >= 3 and self._numeric_issue_count == self._issue_count

    def _predict_next_opponent_offer(
        self, relative_time: float, behavior: str
    ) -> NextOfferPrediction | None:
        assert self._opponent is not None
        offers = self._opponent.history
        utilities = self._opponent.self_utility_history
        if not offers or not utilities:
            return None

        recent_offers = offers[-6:]
        recent_utilities = utilities[-6:]
        last_offer = offers[-1]
        last_utility = utilities[-1]
        repeat_share = recent_offers.count(last_offer) / len(recent_offers)
        dominant_share = max(Counter(recent_offers).values()) / len(recent_offers)
        repeat_probability = 0.65 * repeat_share + 0.35 * dominant_share

        deltas = [
            after - before
            for before, after in zip(recent_utilities, recent_utilities[1:])
        ]
        trend = mean(deltas[-3:]) if deltas else 0.0
        concession_rate = (
            sum(delta > 0.025 for delta in deltas[-5:]) / min(len(deltas), 5)
            if deltas
            else 0.0
        )
        concession_probability = 0.60 * concession_rate + 0.40 * max(
            0.0, min(1.0, 0.5 + 4.0 * trend)
        )

        if behavior == "hardliner":
            repeat_probability = min(1.0, repeat_probability + 0.20)
            concession_probability = max(0.0, concession_probability - 0.20)
        elif behavior == "conceder":
            repeat_probability = max(0.0, repeat_probability - 0.20)
            concession_probability = min(1.0, concession_probability + 0.20)
        elif behavior == "erratic":
            repeat_probability = max(0.0, repeat_probability - 0.15)
            concession_probability = 0.50 * concession_probability + 0.25

        capped_trend = max(-0.08, min(0.12, trend))
        expected = last_utility + capped_trend * (0.65 + 0.35 * relative_time)
        if repeat_probability > 0.70:
            expected = 0.80 * last_utility + 0.20 * expected
        if behavior == "conceder":
            expected += 0.02 * (1.0 - relative_time)
        expected = max(0.0, min(1.0, expected))

        spread = pstdev(recent_utilities) if len(recent_utilities) > 1 else 0.0
        band = 0.04 + spread
        if behavior == "erratic":
            band += 0.08
        elif behavior == "hardliner":
            band *= 0.70
        lower = max(0.0, expected - band)
        upper = min(1.0, expected + band)
        confidence = min(1.0, len(offers) / 8.0)
        if behavior == "erratic":
            confidence *= 0.70

        return NextOfferPrediction(
            created_turn=self._turn_index,
            relative_time=relative_time,
            expected_self_utility=expected,
            lower_self_utility=lower,
            upper_self_utility=upper,
            repeat_probability=max(0.0, min(1.0, repeat_probability)),
            concession_probability=max(0.0, min(1.0, concession_probability)),
            confidence=confidence,
            last_offer=last_offer,
            last_self_utility=last_utility,
        )

    def _review_prediction(
        self,
        prediction: NextOfferPrediction | None,
        has_observed_offer: bool,
    ) -> dict[str, Any] | None:
        if prediction is None:
            return None
        if (
            not has_observed_offer
            or self._opponent is None
            or not self._opponent.history
        ):
            return {
                "available": True,
                "observed": False,
                "created_turn": prediction.created_turn,
            }

        actual = self._opponent.history[-1]
        actual_stats = self._stats_for(actual)
        if actual_stats is None:
            return {
                "available": True,
                "observed": False,
                "created_turn": prediction.created_turn,
            }

        actual_utility = actual_stats.utility
        utility_error = actual_utility - prediction.expected_self_utility
        abs_error = abs(utility_error)
        self._prediction_abs_errors.append(abs_error)
        if len(self._prediction_abs_errors) > 32:
            self._prediction_abs_errors = self._prediction_abs_errors[-32:]

        actual_repeated = (
            prediction.last_offer is not None and actual == prediction.last_offer
        )
        actual_concession = (
            prediction.last_self_utility is not None
            and actual_utility > prediction.last_self_utility + 0.025
        )
        band_width = max(
            0.05,
            (prediction.upper_self_utility - prediction.lower_self_utility) / 2.0,
        )
        utility_surprise = min(1.0, abs_error / band_width)
        repeat_surprise = abs(float(actual_repeated) - prediction.repeat_probability)
        concession_surprise = abs(
            float(actual_concession) - prediction.concession_probability
        )
        surprise = (
            0.55 * utility_surprise
            + 0.25 * repeat_surprise
            + 0.20 * concession_surprise
        )
        return {
            "available": True,
            "observed": True,
            "created_turn": prediction.created_turn,
            "actual_outcome": self._format_outcome(actual),
            "actual_self_utility": round(actual_utility, 6),
            "expected_self_utility": round(prediction.expected_self_utility, 6),
            "lower_self_utility": round(prediction.lower_self_utility, 6),
            "upper_self_utility": round(prediction.upper_self_utility, 6),
            "prediction_confidence": round(prediction.confidence, 6),
            "utility_error": round(utility_error, 6),
            "absolute_error": round(abs_error, 6),
            "mean_absolute_error": round(mean(self._prediction_abs_errors), 6),
            "out_of_band": (
                actual_utility < prediction.lower_self_utility
                or actual_utility > prediction.upper_self_utility
            ),
            "actual_repeated": actual_repeated,
            "repeat_probability": round(prediction.repeat_probability, 6),
            "actual_concession": actual_concession,
            "concession_probability": round(prediction.concession_probability, 6),
            "surprise": round(max(0.0, min(1.0, surprise)), 6),
        }

    def _policy_profile(
        self,
        offer_stats: OutcomeStats | None,
        behavior: str,
        relative_time: float,
    ) -> str:
        contextual_profile = self._contextual_policy_profile(
            offer_stats, behavior, relative_time
        )
        if contextual_profile != "default":
            return contextual_profile
        mode = self._policy_switch_mode
        if mode == "off" or offer_stats is None or self._opponent is None:
            return "default"
        if mode not in {
            "agreement_rescue",
            "soft_leakage_guard",
            "leakage_guard",
            "self_floor",
            "no_low_accept",
        }:
            return "default"
        if relative_time < self._policy_switch_time:
            return "default"
        if len(self._opponent.history) < self._policy_switch_min_history:
            return "default"
        if self._opponent.confidence < self._policy_switch_min_confidence:
            return "default"

        repetition = self._recent_repetition()
        slope = self._opponent.concession_slope()
        hardish = (
            behavior == "hardliner"
            or (
                repetition >= self._policy_switch_min_repetition
                and slope <= self._policy_switch_max_slope
            )
        )

        if mode == "agreement_rescue":
            risk = self._bidding._close_agreement_risk(self._opponent, behavior)
            if risk < self._policy_switch_min_risk:
                return "default"
            if not hardish and relative_time < self._policy_switch_time + 0.08:
                return "default"
            return mode

        if not hardish:
            return "default"

        opponent_view = self._opponent.estimate(offer_stats.outcome)
        if offer_stats.utility > self._policy_switch_max_offer_utility:
            return "default"
        if opponent_view < self._policy_switch_min_opponent_view:
            return "default"

        if mode == "soft_leakage_guard":
            if relative_time < self._policy_switch_time + 0.04:
                return "default"
            if offer_stats.utility > min(0.34, self._policy_switch_max_offer_utility):
                return "default"
            if opponent_view < max(0.82, self._policy_switch_min_opponent_view):
                return "default"
        return mode

    def _contextual_policy_profile(
        self,
        offer_stats: OutcomeStats | None,
        behavior: str,
        relative_time: float,
    ) -> str:
        if self._profile_selector_mode != "contextual_tail":
            return "default"
        if offer_stats is None or self._opponent is None:
            return "default"
        if relative_time < 0.88 or len(self._opponent.history) < 5:
            return "default"
        repetition = self._recent_repetition()
        slope = self._opponent.concession_slope()
        risk = self._bidding._close_agreement_risk(self._opponent, behavior)
        best_seen = max(self._opponent.self_utility_history or [offer_stats.utility])
        hardish = behavior == "hardliner" or repetition >= 0.58 or slope <= 0.06
        if risk >= 0.58 and hardish and best_seen <= 0.62:
            return "agreement_rescue"
        return "default"

    def _recent_repetition(self) -> float:
        if self._opponent is None or not self._opponent.history:
            return 0.0
        recent = self._opponent.history[-6:]
        if not recent:
            return 0.0
        return max(Counter(recent).values()) / len(recent)

    def _bidding_trace(
        self,
        behavior: str,
        relative_time: float,
        reserved_value: float,
        selected: OutcomeStats,
        prediction: NextOfferPrediction | None,
        prediction_review: dict[str, Any] | None,
        policy_profile: str = "default",
    ) -> dict[str, Any]:
        assert self._catalog is not None
        assert self._opponent is not None
        phase = self._bidding.phase(relative_time)
        target = self._bidding.effective_target(
            relative_time,
            behavior,
            self._opponent,
            prediction,
            prediction_review,
            policy_profile,
        )
        floor = reserved_value + self._bidding.reservation_margin
        candidate_floor = self._bidding._candidate_floor(
            target, floor, relative_time, phase
        )
        profile_floor = self._bidding._policy_profile_candidate_floor(
            policy_profile, floor, relative_time
        )
        if profile_floor is not None:
            candidate_floor = max(candidate_floor, profile_floor)
        candidates = [item for item in self._catalog if item.utility >= candidate_floor]
        source = "target_floor"
        if profile_floor is not None:
            source = "policy_switch_floor"
        rescue = self._bidding._repeater_rescue_active(
            self._opponent, behavior, relative_time, phase
        )
        allocation_rescue = self._bidding._allocation_rescue_active(
            self._opponent, relative_time
        )
        numeric_rescue = self._bidding._numeric_rescue_active(
            self._opponent, relative_time
        )
        constraint_probe = self._bidding._constraint_probe_active(
            self._opponent, relative_time, prediction, prediction_review
        )
        rank_denial_active = self._bidding._rank_denial_active(
            self._opponent, relative_time
        )
        allocation_denied = False
        if numeric_rescue:
            numeric_candidates = self._bidding._numeric_rescue_candidates(
                self._catalog, self._opponent, floor
            )
            if numeric_candidates:
                candidates = numeric_candidates
                candidate_floor = max(floor, self._bidding.numeric_rescue_floor)
                source = "numeric_ladder_rescue"
                rescue = False
                allocation_rescue = False
                constraint_probe = False
            else:
                numeric_rescue = False
        if constraint_probe:
            probe_candidates = self._bidding._constraint_probe_candidates(
                self._catalog, self._opponent, floor, target, relative_time
            )
            if probe_candidates:
                candidates = probe_candidates
                candidate_floor = self._bidding._constraint_probe_utility_floor(
                    floor, target, relative_time, relaxed=False
                )
                source = "constraint_probe"
                allocation_rescue = False
                rescue = False
                numeric_rescue = False
            else:
                constraint_probe = False
        if allocation_rescue:
            rescue_floor = max(floor, self._bidding.repeater_rescue_floor)
            rescue_candidates = self._bidding._allocation_rescue_candidates(
                self._catalog, self._opponent, floor
            )
            if rescue_candidates:
                if rank_denial_active and self._bidding._should_deny_allocation(
                    rescue_candidates, self._opponent
                ):
                    denial_floor = max(floor, self._bidding.denial_keep_floor)
                    denial_candidates = [
                        item for item in self._catalog if item.utility >= denial_floor
                    ]
                    if denial_candidates:
                        candidates = denial_candidates
                        candidate_floor = denial_floor
                        source = "rank_denial_floor"
                        allocation_rescue = False
                        rescue = False
                        numeric_rescue = False
                        allocation_denied = True
                else:
                    candidates = rescue_candidates
                    candidate_floor = rescue_floor
                    source = "allocation_ladder_rescue"
        elif rescue and self._bidding.repeater_rescue_mode == "threshold":
            rescue_floor = max(floor, self._bidding.repeater_rescue_floor)
            rescue_candidates = [
                item
                for item in self._catalog
                if item.utility >= rescue_floor
                and self._opponent.estimate(item.outcome)
                >= self._bidding.repeater_rescue_fit
            ]
            if not rescue_candidates and self._bidding.repeater_rescue_fit > 0.82:
                rescue_floor = max(floor, 0.20)
                rescue_candidates = [
                    item
                    for item in self._catalog
                    if item.utility >= rescue_floor
                    and self._opponent.estimate(item.outcome) >= 0.82
                ]
            if rescue_candidates:
                candidates = rescue_candidates
                candidate_floor = rescue_floor
                source = "repeater_threshold_rescue"
        elif rescue:
            rescue_floor = max(floor, self._bidding.repeater_rescue_floor)
            rescue_candidates = [
                item for item in self._catalog if item.utility >= rescue_floor
            ]
            if rescue_candidates:
                candidates = rescue_candidates
                candidate_floor = rescue_floor
                source = "repeater_similarity_rescue"

        denial_floor = self._bidding._rank_denial_floor(
            self._catalog, self._opponent, relative_time, floor
        )
        if denial_floor is not None:
            denial_candidates = [
                item for item in self._catalog if item.utility >= denial_floor
            ]
            if denial_candidates:
                candidates = denial_candidates
                candidate_floor = denial_floor
                source = "rank_denial_floor"
                allocation_rescue = False
                rescue = False
                numeric_rescue = False
                allocation_denied = True

        if not candidates:
            candidates = [item for item in self._catalog if item.utility >= floor]
            candidate_floor = floor
            source = "reservation_floor"
        if not candidates:
            candidates = self._catalog
            candidate_floor = min((item.utility for item in self._catalog), default=0.0)
            source = "full_catalog"

        before_pareto_count = len(candidates)
        candidates = self._bidding._apply_pareto_filter(candidates, self._opponent, phase)
        scored = sorted(
            (
                (
                    self._bidding.score(
                        item,
                        self._opponent,
                        behavior,
                        relative_time,
                        phase,
                        rescue,
                        allocation_rescue,
                        constraint_probe,
                        numeric_rescue,
                        policy_profile,
                    ),
                    item,
                )
                for item in candidates
            ),
            key=lambda row: row[0],
            reverse=True,
        )
        top = [
            self._candidate_trace(item, score, phase, rescue, numeric_rescue)
            for score, item in scored[: self._trace.top_n]
        ]
        selected_rank = next(
            (
                index + 1
                for index, (_, item) in enumerate(scored)
                if item.outcome == selected.outcome
            ),
            None,
        )

        return {
            "phase": phase,
            "policy_profile": policy_profile,
            "target": round(target, 6),
            "reservation_floor": round(floor, 6),
            "candidate_floor": round(candidate_floor, 6),
            "candidate_source": source,
            "candidate_count_before_pareto": before_pareto_count,
            "candidate_count_after_pareto": len(candidates),
            "pareto_filter": self._bidding.pareto_filter,
            "target_curve_mode": self._bidding.target_curve_mode,
            "selection_mode": self._bidding.selection_mode,
            "opponent_horizon_mode": self._bidding.opponent_horizon_mode,
            "short_horizon": self._bidding.short_horizon,
            "short_horizon_weight": round(self._bidding.short_horizon_weight, 6),
            "prediction_policy": self._bidding.prediction_policy,
            "logrolling_mode": self._bidding.logrolling_mode,
            "selected_logrolling_bonus": round(
                self._bidding._logrolling_bonus(
                    selected,
                    self._opponent,
                    phase,
                    self._opponent.confidence,
                ),
                6,
            ),
            "prediction_target_adjustment": round(
                self._bidding._prediction_target_adjustment(
                    relative_time, prediction, prediction_review
                ),
                6,
            ),
            "prediction_probe_start_time": round(
                self._bidding._prediction_probe_start_time(
                    prediction, prediction_review
                ),
                6,
            ),
            "rescue_active": rescue,
            "allocation_rescue_active": allocation_rescue,
            "numeric_rescue_active": numeric_rescue,
            "constraint_probe_active": constraint_probe,
            "rank_denial_active": rank_denial_active,
            "allocation_denied": allocation_denied,
            "rejected_self_offer_count": len(self._opponent.rejected_self_offers),
            "selected_rank": selected_rank,
            "top_candidates": top,
        }

    def _candidate_trace(
        self,
        item: OutcomeStats,
        score: tuple[float, ...],
        phase: str,
        rescue: bool,
        numeric_rescue: bool = False,
    ) -> dict[str, Any]:
        if numeric_rescue:
            details = {
                "score": score[0],
                "our_utility": score[1],
                "similarity": score[2],
                "estimated_opponent_utility": score[3],
                "nash_proxy": score[4],
            }
        elif rescue and self._bidding.repeater_rescue_mode != "threshold":
            details = {
                "score": score[0],
                "similarity": score[1],
                "estimated_opponent_utility": score[2],
                "nash_proxy": score[3],
                "our_utility": score[4],
            }
        else:
            details = {
                "score": score[0],
                "nash_proxy": score[1],
                "estimated_opponent_utility": score[2],
                "similarity": score[3],
                "our_utility": score[4],
            }
        result = {
            "outcome": self._format_outcome(item.outcome),
            "raw_utility": round(item.raw_utility, 6),
            "repeat_count": self._bidding.sent_offers[item.outcome],
            **{key: round(value, 6) for key, value in details.items()},
        }
        if self._opponent is not None:
            result["decayed_opponent_utility"] = round(
                self._opponent.estimate(item.outcome), 6
            )
            result["recent_opponent_utility"] = round(
                self._opponent.recent_estimate(
                    item.outcome, self._bidding.short_horizon
                ),
                6,
            )
            result["logrolling_bonus"] = round(
                self._bidding._logrolling_bonus(
                    item,
                    self._opponent,
                    phase,
                    self._opponent.confidence,
                ),
                6,
            )
        return result

    def _emit_turn_trace(
        self,
        state: SAOState,
        observed_offers: list[dict[str, Any]],
        offer_stats: OutcomeStats | None,
        behavior: str,
        policy_profile: str,
        phase: str,
        target: float,
        reserved_value: float,
        next_offer: OutcomeStats,
        bidding_trace: dict[str, Any],
        acceptance_decision: AcceptanceDecision | None,
        end_decision: EndDecision | None,
        prediction_review: dict[str, Any] | None,
        next_prediction: NextOfferPrediction | None,
        action: str,
        response_offer: Outcome | None,
        message: str,
        text_signals: TextSignals,
    ) -> None:
        if not self._trace.enabled:
            return
        assert self._opponent is not None
        self._trace.emit(
            "negotiatorx_turn",
            {
                "turn": self._turn_index,
                "step": getattr(state, "step", None),
                "relative_time": round(
                    max(0.0, min(1.0, state.relative_time)), 6
                ),
                "phase": phase,
                "current_proposer": str(getattr(state, "current_proposer", None)),
                "current_offer": self._stats_trace(offer_stats),
                "observed_offers": observed_offers,
                "incoming_text_signals": text_signals.trace(),
                "opponent_model": self._opponent_trace(),
                "jitter_profile": self._jitter_trace(),
                "behavior": behavior,
                "profile_selector_mode": self._profile_selector_mode,
                "policy_profile": policy_profile,
                "target": round(target, 6),
                "reserved_value": round(reserved_value, 6),
                "planned_counter": self._stats_trace(next_offer),
                "bidding": bidding_trace,
                "acceptance": self._acceptance_trace(acceptance_decision),
                "end_decision": self._end_trace(end_decision),
                "incoming_prediction_review": prediction_review,
                "next_opponent_prediction": self._prediction_trace(next_prediction),
                "action": action,
                "response_offer": self._format_outcome(response_offer),
                "message": message,
            },
        )

    def _stats_trace(self, stats: OutcomeStats | None) -> dict[str, Any] | None:
        if stats is None:
            return None
        return {
            "outcome": self._format_outcome(stats.outcome),
            "raw_utility": round(stats.raw_utility, 6),
            "normalized_utility": round(stats.utility, 6),
            "bucket": stats.bucket,
        }

    def _acceptance_trace(
        self, decision: AcceptanceDecision | None
    ) -> dict[str, Any] | None:
        if decision is None:
            return None
        return {
            "accept": decision.accept,
            "reason": decision.reason,
            "floor": round(decision.floor, 6),
            "offer_utility": round(decision.offer_utility, 6),
            "next_utility": round(decision.next_utility, 6),
            "target": round(decision.target, 6),
            "opponent_view": round(decision.opponent_view, 6),
            "next_opponent_view": round(decision.next_opponent_view, 6),
        }

    def _end_trace(self, decision: EndDecision | None) -> dict[str, Any] | None:
        if decision is None:
            return None
        return {
            "end": decision.end,
            "reason": decision.reason,
            "floor": round(decision.floor, 6),
            "offer_utility": (
                round(decision.offer_utility, 6)
                if decision.offer_utility is not None
                else None
            ),
            "next_utility": round(decision.next_utility, 6),
            "next_acceptance_fit": round(decision.next_acceptance_fit, 6),
            "agreement_risk": round(decision.agreement_risk, 6),
        }

    def _prediction_trace(
        self, prediction: NextOfferPrediction | None
    ) -> dict[str, Any] | None:
        if prediction is None:
            return None
        return {
            "created_turn": prediction.created_turn,
            "relative_time": round(prediction.relative_time, 6),
            "expected_self_utility": round(prediction.expected_self_utility, 6),
            "lower_self_utility": round(prediction.lower_self_utility, 6),
            "upper_self_utility": round(prediction.upper_self_utility, 6),
            "repeat_probability": round(prediction.repeat_probability, 6),
            "concession_probability": round(prediction.concession_probability, 6),
            "confidence": round(prediction.confidence, 6),
            "last_offer": self._format_outcome(prediction.last_offer),
            "last_self_utility": (
                round(prediction.last_self_utility, 6)
                if prediction.last_self_utility is not None
                else None
            ),
        }

    def _jitter_trace(self) -> dict[str, Any]:
        return {
            "enabled": self._jitter_profile.enabled,
            "seed": self._jitter_profile.seed,
            "values": {
                key: round(value, 6)
                for key, value in self._jitter_profile.values.items()
            },
        }

    def _opponent_trace(self) -> dict[str, Any]:
        assert self._opponent is not None
        recent = self._opponent.history[-3:]
        recent_rejections = self._opponent.rejected_self_offers[-3:]
        return {
            "observed_offer_count": len(self._opponent.history),
            "rejected_self_offer_count": len(self._opponent.rejected_self_offers),
            "confidence": round(self._opponent.confidence, 6),
            "uncertainty_mode": self._opponent.uncertainty_mode,
            "outcome_uncertainty": round(self._opponent.outcome_uncertainty(), 6),
            "concession_slope": round(self._opponent.concession_slope(), 6),
            "slope_method": self._opponent.concession_slope_method(),
            "issue_weights": [
                round(value, 6) for value in self._opponent.issue_weights()
            ],
            "issue_weight_mode": self._opponent.issue_weight_mode,
            "catalog_build_seconds": round(self._catalog_build_seconds, 6),
            "recent_offers": [self._format_outcome(outcome) for outcome in recent],
            "recent_our_utilities": [
                round(value, 6) for value in self._opponent.self_utility_history[-3:]
            ],
            "recent_rejected_self_offers": [
                self._format_outcome(outcome) for outcome in recent_rejections
            ],
            "recent_rejected_self_utilities": [
                round(value, 6)
                for value in self._opponent.rejected_self_utilities[-3:]
            ],
            "top_values_by_issue": self._top_values_by_issue(),
        }

    def _top_values_by_issue(self) -> list[dict[str, Any]]:
        assert self._opponent is not None
        result = []
        for index, counts in enumerate(self._opponent.counts):
            top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:3]
            issue = self._issue_names[index] if index < len(self._issue_names) else index
            result.append(
                {
                    "issue": issue,
                    "values": [
                        {"value": self._jsonable(value), "weight": round(weight, 6)}
                        for value, weight in top
                    ],
                }
            )
        return result

    def _format_outcome(self, outcome: Outcome | None) -> list[Any] | None:
        if outcome is None:
            return None
        return [self._jsonable(value) for value in outcome]

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _read_issue_names(self, issue_count: int) -> list[str]:
        assert self.nmi is not None
        issues = getattr(self.nmi.outcome_space, "issues", None) or []
        names = []
        for index in range(issue_count):
            if index < len(issues):
                names.append(str(getattr(issues[index], "name", f"Issue {index + 1}")))
            else:
                names.append(f"Issue {index + 1}")
        return names

    def _numeric_ranges(
        self, outcomes: list[Outcome], issue_count: int
    ) -> list[tuple[float, float] | None]:
        ranges = []
        for index in range(issue_count):
            values = [
                float(outcome[index])
                for outcome in outcomes
                if isinstance(outcome[index], (int, float))
            ]
            if not values:
                ranges.append(None)
                continue
            ranges.append((min(values), max(values)))
        return ranges

    def _normalize_value(
        self, value: Any, numeric_range: tuple[float, float] | None
    ) -> float:
        if numeric_range is None or not isinstance(value, (int, float)):
            return 0.5
        low, high = numeric_range
        if high <= low:
            return 0.5
        return (float(value) - low) / (high - low)

    def _normalize_utility(self, utility: float) -> float:
        span = self._max_utility - self._min_utility
        if span <= 1e-9:
            return 1.0
        return max(0.0, min(1.0, (utility - self._min_utility) / span))
