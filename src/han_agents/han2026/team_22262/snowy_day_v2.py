from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Protocol

from negmas.helpers import get_class
from negmas.inout import Scenario
from negmas.sao import SAOMechanism
from negmas.sao.common import ResponseType, SAOResponse, SAOState
from negmas.sao.negotiators.base import SAOCallNegotiator


PROJECT_ROOT = Path(__file__).resolve().parent
SCENARIO_ROOT = PROJECT_ROOT / "scenarios"
DEFAULT_CLASSIFIER_CHECKPOINT = PROJECT_ROOT / "opponent_type" / "runs" / "opponent_type_mlp.json"

FORMAL_FEATURE_NAMES: list[str] = [
    "n_offers_seen",
    "opening_advantage",
    "latest_advantage",
    "best_advantage",
    "mean_advantage",
    "concession_slope",
    "total_concession",
    "mean_offer_movement",
    "repeat_rate",
    "stubborn_issue_fraction",
    "rounds_since_improvement",
]

PERSONA_OPPONENTS: dict[str, str] = {
    "examples.persona_opponents.ConcessionaryLLMProxy": "conceder",
    "examples.persona_opponents.FairLLMProxy": "fair",
    "examples.persona_opponents.RevealingHumanLLMProxy": "revealing",
    "examples.persona_opponents.MisleadingLLMProxy": "misleading",
    "examples.persona_opponents.HardheadedLLMProxy": "hardliner",
    "examples.persona_opponents.TrapLLMProxy": "wall",
}

COARSE_PERSONA_LABELS: dict[str, str] = {
    "conceder": "conceder",
    "fair": "fairish",
    "revealing": "fairish",
    "misleading": "wallish",
    "hardliner": "hardliner",
    "wall": "wallish",
}


@dataclass(frozen=True)
class CurveConfig:
    name: str
    start_fraction: float = 1.0
    final_fraction: float = 0.35
    concession_exponent: float = 2.0
    rescue_min_time: float = 0.65
    rescue_floor_fraction: float = 0.56
    opponent_weight_scale: float = 1.0
    late_accept_fraction: float = 0.56
    persona_label: str = "unknown"

    def target_fraction(self, relative_time: float) -> float:
        t = _clamp01(relative_time)
        if t >= 0.97:
            return 0.0
        exponent = max(self.concession_exponent, 1e-6)
        target = self.final_fraction + (
            self.start_fraction - self.final_fraction
        ) * (1.0 - t**exponent)
        return _clamp(target, self.final_fraction, self.start_fraction)

    def counteroffer_fraction(
        self,
        *,
        relative_time: float,
        rescue_active: bool,
    ) -> float:
        t = _clamp01(relative_time)
        fraction = self.target_fraction(t)
        if rescue_active and t >= self.rescue_min_time:
            fraction = min(fraction, self.rescue_floor_fraction)
        if t >= 0.90:
            fraction = min(fraction, 0.40 if rescue_active else self.final_fraction)
        if t >= 0.97:
            return 0.0
        return _clamp01(fraction)


TUNED_FINE_CURVES: dict[str, CurveConfig] = {
    "conceder": CurveConfig(
        name="conceder_s1.00_f0.45_e1.00_rt0.65_rf0.56_ow1.00_la0.56",
        final_fraction=0.45,
        concession_exponent=1.0,
        rescue_min_time=0.65,
        rescue_floor_fraction=0.56,
        opponent_weight_scale=1.0,
        late_accept_fraction=0.56,
        persona_label="conceder",
    ),
    "fair": CurveConfig(
        name="fair_s1.00_f0.25_e1.50_rt0.65_rf0.56_ow1.00_la0.56",
        final_fraction=0.25,
        concession_exponent=1.5,
        rescue_min_time=0.65,
        rescue_floor_fraction=0.56,
        opponent_weight_scale=1.0,
        late_accept_fraction=0.56,
        persona_label="fair",
    ),
    "revealing": CurveConfig(
        name="revealing_s1.00_f0.55_e1.00_rt0.65_rf0.56_ow1.00_la0.56",
        final_fraction=0.55,
        concession_exponent=1.0,
        rescue_min_time=0.65,
        rescue_floor_fraction=0.56,
        opponent_weight_scale=1.0,
        late_accept_fraction=0.56,
        persona_label="revealing",
    ),
    "misleading": CurveConfig(
        name="misleading_s1.00_f0.05_e1.00_rt0.50_rf0.40_ow1.00_la0.40",
        final_fraction=0.05,
        concession_exponent=1.0,
        rescue_min_time=0.50,
        rescue_floor_fraction=0.40,
        opponent_weight_scale=1.0,
        late_accept_fraction=0.40,
        persona_label="misleading",
    ),
    "hardliner": CurveConfig(
        name="hardliner_s1.00_f0.35_e2.20_rt0.65_rf0.56_ow1.00_la0.56",
        final_fraction=0.35,
        concession_exponent=2.2,
        rescue_min_time=0.65,
        rescue_floor_fraction=0.56,
        opponent_weight_scale=1.0,
        late_accept_fraction=0.56,
        persona_label="hardliner",
    ),
    "wall": CurveConfig(
        name="wall_s1.00_f0.05_e1.00_rt0.50_rf0.40_ow1.00_la0.40",
        final_fraction=0.05,
        concession_exponent=1.0,
        rescue_min_time=0.50,
        rescue_floor_fraction=0.40,
        opponent_weight_scale=1.0,
        late_accept_fraction=0.40,
        persona_label="wall",
    ),
}

TUNED_COARSE_CURVES: dict[str, CurveConfig] = {
    "conceder": TUNED_FINE_CURVES["conceder"],
    "fairish": CurveConfig(
        name="fairish_s1.00_f0.55_e1.00_rt0.65_rf0.56_ow1.00_la0.56",
        final_fraction=0.55,
        concession_exponent=1.0,
        rescue_min_time=0.65,
        rescue_floor_fraction=0.56,
        opponent_weight_scale=1.0,
        late_accept_fraction=0.56,
        persona_label="fairish",
    ),
    "hardliner": TUNED_FINE_CURVES["hardliner"],
    "wallish": CurveConfig(
        name="wallish_s1.00_f0.05_e1.00_rt0.50_rf0.40_ow1.00_la0.40",
        final_fraction=0.05,
        concession_exponent=1.0,
        rescue_min_time=0.50,
        rescue_floor_fraction=0.40,
        opponent_weight_scale=1.0,
        late_accept_fraction=0.40,
        persona_label="wallish",
    ),
}

TEXT_CHANNELS: dict[str, dict[str, str]] = {
    "conceder": {
        "accept": "With {offer}, this works for me. I appreciate the movement.",
        "opening": "How about {offer}? I can move with you if we keep making progress.",
        "evidence": "I can move to {offer}. That reflects the direction you have been taking.",
        "rescue": "I can make {offer} work if it helps us keep an agreement within reach.",
        "deadline": "Time is short. I can close on {offer} as a workable compromise.",
    },
    "fairish": {
        "accept": "With {offer}, this feels balanced enough for me to accept.",
        "opening": "Would {offer} work as a fair starting point?",
        "evidence": "I suggest {offer}. It reflects your priorities while keeping the exchange balanced.",
        "rescue": "Could we settle around {offer}? That keeps us near fair middle ground.",
        "deadline": "Time is short. {offer} still gives us a fair way to close.",
    },
    "hardliner": {
        "accept": "With {offer}, my core needs are met. I accept.",
        "opening": "I can offer {offer}. I need to stay firm on the rest.",
        "evidence": "I can move to {offer}, but I still need to protect the core of the deal.",
        "rescue": "I can make a limited move to {offer} to keep the deal alive.",
        "deadline": "Time is short. {offer} is the firm closing point I can support.",
    },
    "wallish": {
        "accept": "I can work with {offer}. I accept.",
        "opening": "I propose {offer}. It keeps a deal possible.",
        "evidence": "Given the offers so far, {offer} looks like a practical next step.",
        "rescue": "I can move to {offer} so we do not end with nothing.",
        "deadline": "Time is short. {offer} is a practical way to avoid no agreement.",
    },
    "baseline": {
        "accept": "I can work with {offer}. I accept.",
        "opening": "How about {offer}? There is still room to reach agreement.",
        "evidence": "I suggest {offer}. It reflects the direction of our discussion.",
        "rescue": "Could we work with {offer}? I want to keep the negotiation alive.",
        "deadline": "Time is short. I can close on {offer}.",
    },
}

PERSONA_TEXT_CHANNEL: dict[str, str] = {
    "conceder": "conceder",
    "fair": "fairish",
    "fairish": "fairish",
    "revealing": "fairish",
    "hardliner": "hardliner",
    "misleading": "wallish",
    "wall": "wallish",
    "wallish": "wallish",
}


def curve_for_policy(
    policy: str,
    true_persona: str,
    *,
    classifier_label: str | None = None,
) -> CurveConfig:
    if policy == "baseline":
        return CurveConfig(name="baseline_default", persona_label="baseline")
    if policy == "oracle-fine":
        return TUNED_FINE_CURVES[true_persona]
    if policy == "oracle-coarse":
        return TUNED_COARSE_CURVES[COARSE_PERSONA_LABELS[true_persona]]
    if policy == "classifier-coarse":
        if classifier_label in TUNED_COARSE_CURVES:
            return TUNED_COARSE_CURVES[classifier_label]
        return CurveConfig(
            name="baseline_default",
            persona_label="classifier-unclassified",
        )
    raise ValueError(f"Unknown curve policy: {policy}")


@dataclass(frozen=True)
class OutcomeInfo:
    outcome: Any
    utility: float
    fraction: float


@dataclass(frozen=True)
class CandidateScore:
    opponent_fit: float
    estimated_opponent_utility: float
    own_utility: float
    rejected_penalty: float


@dataclass(frozen=True)
class CurveScore:
    persona: str
    curve: str
    n: int
    objective: float
    mean_official_adv: float
    agreement_rate: float
    zero_rate: float
    timeout_rate: float


@dataclass(frozen=True)
class ClassifierPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, float]
    prefix_turns: int


class OpponentTypeClassifier(Protocol):
    def predict(
        self,
        opponent_offers: list[Any],
        analyzer: "OutcomeAnalyzer",
    ) -> ClassifierPrediction | None:
        ...


class CheckpointOpponentTypeClassifier:
    """Self-contained loader for Tyrone's formal-behavior MLP checkpoint."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint)
        self.device = device
        self.model, self.labels, self.prefix_turns, self.feature_names = (
            self._load_checkpoint()
        )

    def predict(
        self,
        opponent_offers: list[Any],
        analyzer: "OutcomeAnalyzer",
    ) -> ClassifierPrediction | None:
        if not opponent_offers:
            return None
        features = build_opponent_type_features(
            opponent_offers,
            analyzer,
            prefix_turns=self.prefix_turns,
        )
        probs = self._predict_proba([features.get(name, 0.0) for name in self.feature_names])
        best_index = max(range(len(probs)), key=lambda index: probs[index])
        return ClassifierPrediction(
            label=self.labels[best_index],
            confidence=float(probs[best_index]),
            probabilities={
                label: round(float(prob), 6) for label, prob in zip(self.labels, probs)
            },
            features=features,
            prefix_turns=self.prefix_turns,
        )

    def _load_checkpoint(self) -> tuple[Any, list[str], int, list[str]]:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {self.checkpoint_path}. "
                "Train one with python -m opponent_type.train or pass "
                "--classifier-checkpoint."
            )
        if self.checkpoint_path.suffix.lower() == ".json":
            return self._load_json_checkpoint()
        try:
            import torch
            from torch import nn
        except Exception as exc:  # pragma: no cover - depends on local torch install
            raise RuntimeError(
                "classifier-coarse requires PyTorch to load the Tyrone checkpoint."
            ) from exc

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        labels = list(checkpoint["labels"])
        feature_names = list(checkpoint.get("feature_names", FORMAL_FEATURE_NAMES))
        input_dim = int(checkpoint.get("input_dim", len(feature_names)))
        hidden_dim = int(checkpoint.get("metadata", {}).get("hidden_dim", 64))
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, len(labels))),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, len(labels)), len(labels)),
        ).to(self.device)
        state_dict = checkpoint["state_dict"]
        if all(key.startswith("net.") for key in state_dict):
            state_dict = {key.removeprefix("net."): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return (
            model,
            labels,
            int(checkpoint.get("prefix_turns", 4)),
            feature_names,
        )

    def _load_json_checkpoint(self) -> tuple[Any, list[str], int, list[str]]:
        with open(self.checkpoint_path, encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        return (
            list(checkpoint["layers"]),
            list(checkpoint["labels"]),
            int(checkpoint.get("prefix_turns", 4)),
            list(checkpoint.get("feature_names", FORMAL_FEATURE_NAMES)),
        )

    def _predict_proba(self, features: list[float]) -> list[float]:
        if isinstance(self.model, list):
            return self._predict_proba_json(features)
        import torch

        with torch.no_grad():
            tensor = torch.tensor([features], dtype=torch.float32, device=self.device)
            probs = torch.softmax(self.model(tensor), dim=-1).squeeze(0)
        return [float(value) for value in probs.detach().cpu().tolist()]

    def _predict_proba_json(self, features: list[float]) -> list[float]:
        values = [float(value) for value in features]
        for layer in self.model:
            weights = layer["weight"]
            bias = layer.get("bias", [0.0] * len(weights))
            values = [
                sum(float(weight) * value for weight, value in zip(row, values))
                + float(offset)
                for row, offset in zip(weights, bias)
            ]
            activation = str(layer.get("activation", "")).lower()
            if activation == "relu":
                values = [max(0.0, value) for value in values]
            elif activation == "softmax":
                return _softmax(values)
        return _softmax(values)


def build_opponent_type_features(
    opponent_offers: list[Any],
    analyzer: "OutcomeAnalyzer",
    *,
    prefix_turns: int,
) -> dict[str, float]:
    k = max(int(prefix_turns), 1)
    offers = [offer for offer in opponent_offers[:k] if analyzer.is_valid(offer)]
    n = len(offers)
    if n == 0:
        return dict.fromkeys(FORMAL_FEATURE_NAMES, 0.0)

    advantages = [_clamp01(analyzer.fraction(offer)) for offer in offers]
    movements = [
        _tuple_distance(left, right) for left, right in zip(offers, offers[1:])
    ]
    values = [
        _clamp01(n / k),
        advantages[0],
        advantages[-1],
        max(advantages),
        sum(advantages) / n,
        _concession_slope(advantages),
        _clamp(advantages[-1] - advantages[0], -1.0, 1.0),
        _clamp01(sum(movements) / len(movements)) if movements else 0.0,
        _repeat_rate(offers),
        _stubborn_issue_fraction(offers),
        _rounds_since_improvement(advantages, k),
    ]
    return dict(zip(FORMAL_FEATURE_NAMES, values, strict=True))


class OutcomeAnalyzer:
    """Enumerates legal outcomes and caches our utility values."""

    def __init__(
        self,
        ufun: Any,
        outcome_space: Any,
        *,
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
        # per-issue numeric (min, max), precomputed once in _analyze so the
        # opponent model's _value_score does not rescan all outcomes per call
        self.numeric_domain: dict[int, tuple[float, float]] = {}
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

    def utility_at_fraction(self, fraction: float) -> float:
        span = max(self.best_utility - self.reserved_value, 1e-9)
        return self.reserved_value + _clamp01(fraction) * span

    def safe_floor(self) -> float:
        return self.reserved_value + self.safety_margin

    def is_valid(self, outcome: Any | None) -> bool:
        if outcome is None:
            return False
        try:
            return self.outcome_space is None or bool(self.outcome_space.is_valid(outcome))
        except Exception:
            return False

    def candidates_above(self, min_utility: float) -> list[OutcomeInfo]:
        return [info for info in self.outcomes if info.utility >= min_utility]

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
        raw_keys = {self.key(outcome) for outcome in raw}
        if best is not None and self.is_valid(best) and self.key(best) not in raw_keys:
            raw.append(best)
        if not raw and best is not None and self.is_valid(best):
            raw = [best]

        utilities = [self.utility(outcome) for outcome in raw]
        self.best_utility = max(utilities, default=self.reserved_value)
        if utilities:
            self.best_outcome = raw[utilities.index(self.best_utility)]
        self.safety_margin = self.safety_margin_fraction * max(
            self.best_utility - self.reserved_value,
            1e-9,
        )
        infos = [
            OutcomeInfo(outcome, self.utility(outcome), self.fraction(outcome))
            for outcome in raw
            if self.utility(outcome) >= self.safe_floor()
        ]
        if not infos:
            infos = [
                OutcomeInfo(outcome, self.utility(outcome), self.fraction(outcome))
                for outcome in raw
            ]
        self.outcomes = sorted(infos, key=lambda info: info.utility, reverse=True)
        self._cache_numeric_domains()

    def _cache_numeric_domains(self) -> None:
        """Precompute each issue's numeric (min, max) once.

        Previously the opponent model's ``_value_score`` rescanned every
        outcome to recover an issue's numeric range on each call; since
        ``_planned_offer`` scores up to ``max_outcomes`` candidates every turn,
        that made per-step scoring O(N^2) on large numeric domains. Caching the
        bounds here (built once, O(N)) turns the per-call cost into an O(1)
        lookup.
        """
        lo: dict[int, float] = {}
        hi: dict[int, float] = {}
        count: dict[int, int] = {}
        for info in self.outcomes:
            outcome = info.outcome
            if not isinstance(outcome, tuple):
                continue
            for issue, raw_value in enumerate(outcome):
                try:
                    numeric = float(raw_value)
                except Exception:
                    continue
                if issue not in count:
                    count[issue] = 0
                    lo[issue] = numeric
                    hi[issue] = numeric
                count[issue] += 1
                if numeric < lo[issue]:
                    lo[issue] = numeric
                elif numeric > hi[issue]:
                    hi[issue] = numeric
        # keep only issues with >=2 numeric values and a positive span, matching
        # the original guards (len(domain) < 2 or span <= 0 -> score 0.0)
        self.numeric_domain = {
            issue: (lo[issue], hi[issue])
            for issue in count
            if count[issue] >= 2 and hi[issue] > lo[issue]
        }

    def _collect_outcomes(self) -> list[Any]:
        if self.outcome_space is None:
            return []
        try:
            sampled = list(
                self.outcome_space.enumerate_or_sample(
                    max_cardinality=self.max_outcomes,
                )
            )
        except Exception:
            return []
        unique = []
        seen = set()
        for outcome in sampled:
            if not self.is_valid(outcome):
                continue
            key = self.key(outcome)
            if key in seen:
                continue
            seen.add(key)
            unique.append(outcome)
        return unique


class SnowyV2OpponentModel:
    """Formal-offer opponent model used by the self-contained curve agent."""

    def __init__(self, recency_decay: float = 0.85) -> None:
        self.recency_decay = recency_decay
        self.value_counts: dict[int, dict[Any, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.offer_history: list[Any] = []
        self.utility_history: list[float] = []
        self.rejected_counts: dict[Any, int] = defaultdict(int)
        self.best_received_offer: OutcomeInfo | None = None

    def update_offer(self, offer: Any | None, analyzer: OutcomeAnalyzer) -> None:
        if not isinstance(offer, tuple) or not analyzer.is_valid(offer):
            return
        self.offer_history.append(offer)
        for issue_scores in self.value_counts.values():
            for value in list(issue_scores):
                issue_scores[value] *= self.recency_decay
        for issue, value in enumerate(offer):
            self.value_counts[issue][value] += 1.0
        utility = analyzer.utility(offer)
        self.utility_history.append(utility)
        if (
            self.best_received_offer is None
            or utility > self.best_received_offer.utility
        ):
            self.best_received_offer = OutcomeInfo(
                offer,
                utility,
                analyzer.fraction(offer),
            )

    def update_rejected(self, offer: Any | None, key_fn: Any) -> None:
        if offer is not None:
            self.rejected_counts[key_fn(offer)] += 1

    def total_rejections(self) -> int:
        return sum(self.rejected_counts.values())

    def has_evidence(self) -> bool:
        return bool(self.offer_history)

    def concession_rate(self) -> float:
        if len(self.utility_history) < 2:
            return 0.0
        improvements = [
            max(0.0, right - left)
            for left, right in zip(self.utility_history, self.utility_history[1:])
        ]
        return sum(improvements) / len(improvements)

    def opponent_fit(self, outcome: Any | None) -> float:
        if not isinstance(outcome, tuple) or not self.value_counts:
            return 0.0
        scores = []
        for issue, value in enumerate(outcome):
            issue_scores = self.value_counts.get(issue, {})
            if not issue_scores:
                continue
            max_score = max(issue_scores.values()) or 1.0
            scores.append(float(issue_scores.get(value, 0.0)) / max_score)
        return sum(scores) / len(scores) if scores else 0.0

    def issue_weights(self) -> list[float]:
        if not self.value_counts:
            return []
        weights = []
        for issue in range(max(self.value_counts) + 1):
            scores = self.value_counts.get(issue, {})
            if not scores:
                weights.append(0.0)
                continue
            weights.append(max(scores.values()) / max(sum(scores.values()), 1e-9))
        total = sum(weights)
        if total <= 1e-9:
            return weights
        return [weight / total for weight in weights]

    def estimated_opponent_utility(
        self,
        outcome: Any | None,
        analyzer: OutcomeAnalyzer,
    ) -> float:
        if not isinstance(outcome, tuple) or not self.value_counts:
            return self.opponent_fit(outcome)
        weights = self.issue_weights()
        if not weights:
            return self.opponent_fit(outcome)
        utility = 0.0
        for issue, value in enumerate(outcome):
            if issue >= len(weights):
                continue
            utility += weights[issue] * self._value_score(issue, value, analyzer)
        return _clamp01(utility)

    def rejected_penalty(self, outcome: Any | None, key_fn: Any) -> float:
        if outcome is None:
            return 0.0
        return min(1.0, 0.35 * self.rejected_counts.get(key_fn(outcome), 0))

    def _value_score(
        self,
        issue: int,
        value: Any,
        analyzer: OutcomeAnalyzer,
    ) -> float:
        issue_scores = self.value_counts.get(issue, {})
        if not issue_scores:
            return 0.0
        if value in issue_scores:
            return issue_scores[value] / max(max(issue_scores.values()), 1e-9)
        observed_numeric = []
        try:
            numeric_value = float(value)
            for observed in issue_scores:
                observed_numeric.append(float(observed))
        except Exception:
            return 0.0
        if not observed_numeric:
            return 0.0
        # O(1) lookup of the issue's numeric (min, max), precomputed once in
        # OutcomeAnalyzer; was an O(N) rescan of all outcomes on every call.
        domain_bounds = analyzer.numeric_domain.get(issue)
        if domain_bounds is None:
            return 0.0
        lo, hi = domain_bounds
        span = hi - lo
        if span <= 0.0:
            return 0.0
        nearest = min(abs(numeric_value - observed) for observed in observed_numeric)
        return max(0.0, 1.0 - nearest / span)


class SnowyDayV2(SAOCallNegotiator):
    """Self-contained no-LLM Snowy-style agent controlled by CurveConfig."""

    def __init__(
        self,
        curve: CurveConfig | None = None,
        curve_policy: str = "classifier-coarse",
        classifier: OpponentTypeClassifier | None = None,
        classifier_checkpoint: str | Path | None = DEFAULT_CLASSIFIER_CHECKPOINT,
        max_outcomes: int = 6000,
        safety_margin_fraction: float = 0.01,
        deadline_rescue_enabled: bool = True,
        rescue_min_rejections: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.curve_policy = curve_policy
        if (
            self.curve_policy == "classifier-coarse"
            and classifier is None
            and classifier_checkpoint is not None
            and Path(classifier_checkpoint).exists()
        ):
            classifier = CheckpointOpponentTypeClassifier(classifier_checkpoint)
        self.classifier = classifier
        self.curve = curve or curve_for_policy(
            self.curve_policy,
            "unknown",
            classifier_label=None,
        )
        self.classifier_prediction: ClassifierPrediction | None = None
        self.max_outcomes = max_outcomes
        self.safety_margin_fraction = safety_margin_fraction
        self.deadline_rescue_enabled = deadline_rescue_enabled
        self.rescue_min_rejections = rescue_min_rejections
        self.analyzer: OutcomeAnalyzer | None = None
        self.opponent_model = SnowyV2OpponentModel()
        self.last_proposal: Any | None = None
        self.last_seen_offer_key: Any | None = None
        self.debug_trace: list[dict[str, Any]] = []

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun is not None
        analyzer = self._get_analyzer()
        current_offer = state.current_offer
        opponent_offer = self._actual_opponent_offer(state, current_offer, analyzer)
        self._mark_rejected_if_needed(opponent_offer, analyzer)
        if opponent_offer is not None:
            self.opponent_model.update_offer(opponent_offer, analyzer)
            self.last_seen_offer_key = analyzer.key(opponent_offer)
            self._maybe_update_curve_from_classifier(analyzer)

        planned_offer = self._planned_offer(state)
        if planned_offer is None:
            return SAOResponse(
                ResponseType.END_NEGOTIATION,
                None,
                data={"text": "I do not see a valid agreement path."},
            )

        if self._should_accept(state, current_offer, planned_offer):
            self._record(state, current_offer, planned_offer, "accepted")
            return SAOResponse(
                ResponseType.ACCEPT_OFFER,
                current_offer,
                data={"text": self._accept_text(current_offer)},
            )

        self.last_proposal = planned_offer.outcome
        self._record(state, current_offer, planned_offer, "rejected")
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            planned_offer.outcome,
            data={
                "text": self._counter_text(
                    state,
                    planned_offer.outcome,
                    current_offer,
                )
            },
        )

    def _get_analyzer(self) -> OutcomeAnalyzer:
        if self.analyzer is None:
            assert self.ufun is not None
            self.analyzer = OutcomeAnalyzer(
                self.ufun,
                self._outcome_space(),
                max_outcomes=self.max_outcomes,
                safety_margin_fraction=self.safety_margin_fraction,
            )
        return self.analyzer

    def _target_fraction(self, state: SAOState) -> float:
        return self.curve.target_fraction(self._relative_time(state))

    def _counteroffer_floor(self, state: SAOState) -> float:
        analyzer = self._get_analyzer()
        fraction = self.curve.counteroffer_fraction(
            relative_time=self._relative_time(state),
            rescue_active=self._needs_deadline_rescue(state),
        )
        if self._relative_time(state) >= 0.97:
            return analyzer.safe_floor()
        return max(analyzer.safe_floor(), analyzer.utility_at_fraction(fraction))

    def _planned_offer(self, state: SAOState) -> OutcomeInfo | None:
        analyzer = self._get_analyzer()
        candidates = analyzer.candidates_above(self._counteroffer_floor(state))
        if not candidates:
            candidates = [
                info for info in analyzer.outcomes if info.utility >= analyzer.safe_floor()
            ]
        if not candidates:
            return None
        t = self._relative_time(state)
        if not self.opponent_model.has_evidence():
            return max(
                candidates,
                key=lambda info: (
                    -self.opponent_model.rejected_penalty(info.outcome, analyzer.key),
                    info.utility,
                ),
            )
        return max(candidates, key=lambda info: self._candidate_sort_key(info, t))

    def _candidate_sort_key(self, info: OutcomeInfo, t: float) -> tuple[float, float, float]:
        analyzer = self._get_analyzer()
        score = self._candidate_score(info)
        own_fraction = analyzer.fraction(info.outcome)
        base_opponent_weight = 0.30 + 0.15 * _clamp01(t)
        if self._is_stalemated():
            base_opponent_weight = 0.45 + 0.35 * _clamp((t - 0.45) / 0.45, 0.0, 1.0)
        opponent_weight = min(
            base_opponent_weight * self.curve.opponent_weight_scale,
            0.85,
        )
        own_weight = 1.0 - min(opponent_weight, 0.78)
        combined = (
            opponent_weight * score.estimated_opponent_utility
            + 0.18 * score.opponent_fit
            + own_weight * own_fraction
            - score.rejected_penalty
        )
        return (combined, score.estimated_opponent_utility, score.own_utility)

    def _candidate_score(self, info: OutcomeInfo) -> CandidateScore:
        analyzer = self._get_analyzer()
        return CandidateScore(
            opponent_fit=self.opponent_model.opponent_fit(info.outcome),
            estimated_opponent_utility=self.opponent_model.estimated_opponent_utility(
                info.outcome,
                analyzer,
            ),
            own_utility=info.utility,
            rejected_penalty=self.opponent_model.rejected_penalty(
                info.outcome,
                analyzer.key,
            ),
        )

    def _needs_deadline_rescue(self, state: SAOState) -> bool:
        if not self.deadline_rescue_enabled:
            return False
        if self._relative_time(state) < self.curve.rescue_min_time:
            return False
        best = self.opponent_model.best_received_offer
        if best is None or len(self.opponent_model.offer_history) < 3:
            return False
        if self.opponent_model.total_rejections() < self.rescue_min_rejections:
            return False
        return best.fraction < 0.40 and self.opponent_model.concession_rate() <= 0.01

    def _is_stalemated(self) -> bool:
        if len(self.opponent_model.offer_history) < 3:
            return False
        if self.opponent_model.concession_rate() > 0.01:
            return False
        best = self.opponent_model.best_received_offer
        if best is not None and best.fraction < 0.35:
            return True
        return self.opponent_model.total_rejections() >= self.rescue_min_rejections

    def _should_accept(
        self,
        state: SAOState,
        offer: Any | None,
        planned_offer: OutcomeInfo,
    ) -> bool:
        analyzer = self._get_analyzer()
        if not analyzer.is_valid(offer):
            return False
        utility = analyzer.utility(offer)
        if utility < analyzer.safe_floor():
            return False
        t = self._relative_time(state)
        offer_fraction = analyzer.fraction(offer)
        planned_fraction = analyzer.fraction(planned_offer.outcome)
        if utility >= planned_offer.utility - 1e-9:
            return True
        if t >= 0.80 and offer_fraction >= min(
            self.curve.late_accept_fraction,
            planned_fraction + 0.04,
        ):
            return True
        if t >= 0.90 and offer_fraction >= self.curve.final_fraction:
            return True
        if t >= 0.90:
            best = self.opponent_model.best_received_offer
            if best is not None and utility >= analyzer.utility(best.outcome) - 1e-9:
                return True
        if t >= 0.97:
            return True
        return False

    def _mark_rejected_if_needed(
        self,
        opponent_offer: Any | None,
        analyzer: OutcomeAnalyzer,
    ) -> None:
        if self.last_proposal is None:
            return
        if opponent_offer is not None and analyzer.key(opponent_offer) == analyzer.key(
            self.last_proposal
        ):
            return
        self.opponent_model.update_rejected(self.last_proposal, analyzer.key)

    def _actual_opponent_offer(
        self,
        state: SAOState,
        offer: Any | None,
        analyzer: OutcomeAnalyzer,
    ) -> Any | None:
        if not analyzer.is_valid(offer):
            return None
        if self.last_proposal is not None and analyzer.key(offer) == analyzer.key(
            self.last_proposal
        ):
            return None
        proposer = getattr(state, "current_proposer", None)
        own_ids = {
            str(getattr(self, "id", "") or ""),
            str(getattr(self, "name", "") or ""),
        }
        if proposer not in (None, "") and str(proposer) in own_ids:
            return None
        return offer

    def _maybe_update_curve_from_classifier(self, analyzer: OutcomeAnalyzer) -> None:
        if self.curve_policy != "classifier-coarse" or self.classifier is None:
            return
        prediction = self.classifier.predict(self.opponent_model.offer_history, analyzer)
        if prediction is None:
            return
        self.classifier_prediction = prediction
        self.curve = curve_for_policy(
            "classifier-coarse",
            "unknown",
            classifier_label=prediction.label,
        )

    def _counter_text(
        self,
        state: SAOState,
        proposed_offer: Any | None,
        current_offer: Any | None,
    ) -> str:
        t = self._relative_time(state)
        if t >= 0.90:
            kind = "deadline"
        elif self._needs_deadline_rescue(state):
            kind = "rescue"
        elif self.opponent_model.has_evidence():
            kind = "evidence"
        else:
            kind = "opening"
        return self._type_text(
            kind,
            outcome=proposed_offer,
            reference=current_offer,
        )

    def _accept_text(self, accepted_offer: Any | None) -> str:
        return self._type_text("accept", outcome=accepted_offer)

    def _type_text(
        self,
        kind: str,
        *,
        outcome: Any | None,
        reference: Any | None = None,
    ) -> str:
        channel = PERSONA_TEXT_CHANNEL.get(self.curve.persona_label, "baseline")
        template = TEXT_CHANNELS[channel][kind]
        offer_text = self._item_context(outcome, reference) or "this offer"
        return template.format(offer=offer_text)

    def _item_context(
        self,
        outcome: Any | None,
        reference: Any | None = None,
        limit: int = 2,
    ) -> str:
        """Render terms from an offer using the formal outcome-space ordering."""
        outcome_space = self._outcome_space()
        issues = list(getattr(outcome_space, "issues", None) or [])
        if (
            not issues
            or not isinstance(outcome, (tuple, list))
            or len(outcome) != len(issues)
        ):
            return ""

        # A NegMAS tuple outcome is ordered exactly like outcome_space.issues.
        # strict=True makes a malformed outcome fail closed instead of silently
        # attaching a value to the wrong issue name.
        try:
            formal_terms = list(zip(issues, outcome, strict=True))
        except ValueError:
            return ""

        indices: list[int] = []
        if isinstance(reference, (tuple, list)) and len(reference) == len(issues):
            indices = [
                index
                for index, (proposed, current) in enumerate(
                    zip(outcome, reference, strict=True)
                )
                if proposed != current
            ]
        if not indices:
            indices = list(range(min(len(issues), limit)))

        rendered_terms: list[str] = []
        for index in indices:
            issue, value = formal_terms[index]
            raw_name = str(getattr(issue, "name", "") or "")
            name = " ".join(raw_name.split())
            if not name:
                continue
            if isinstance(value, str):
                rendered_value = " ".join(value.split())
            else:
                try:
                    rendered_value = json.dumps(
                        _jsonable(value),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError):
                    rendered_value = " ".join(str(value).split())
            if not rendered_value:
                continue
            term = f"{name} at {rendered_value}"
            if term not in rendered_terms:
                rendered_terms.append(term)
            if len(rendered_terms) >= limit:
                break
        if not rendered_terms:
            return ""
        if len(rendered_terms) == 1:
            return rendered_terms[0]
        return f"{rendered_terms[0]} and {rendered_terms[1]}"

    def _record(
        self,
        state: SAOState,
        current_offer: Any | None,
        planned_offer: OutcomeInfo,
        action: str,
    ) -> None:
        analyzer = self._get_analyzer()
        current_offer_utility = (
            analyzer.utility(current_offer) if analyzer.is_valid(current_offer) else None
        )
        entry = {
            "relative_time": self._relative_time(state),
            "curve": asdict(self.curve),
            "current_offer": _jsonable(current_offer),
            "current_offer_utility": current_offer_utility,
            "target_fraction": self._target_fraction(state),
            "counteroffer_floor": self._counteroffer_floor(state),
            "selected_target": _jsonable(planned_offer.outcome),
            "selected_target_utility": planned_offer.utility,
            "selected_target_fraction": planned_offer.fraction,
            "candidate_score": asdict(self._candidate_score(planned_offer)),
            "deadline_rescue_active": self._needs_deadline_rescue(state),
            "classifier_label": (
                self.classifier_prediction.label
                if self.classifier_prediction is not None
                else None
            ),
            "classifier_confidence": (
                self.classifier_prediction.confidence
                if self.classifier_prediction is not None
                else None
            ),
            "action": action,
        }
        self.debug_trace.append(entry)

    def _outcome_space(self) -> Any | None:
        if self.nmi is not None and self.nmi.outcome_space is not None:
            return self.nmi.outcome_space
        if self.ufun is not None:
            return self.ufun.outcome_space
        return None

    def _relative_time(self, state: SAOState) -> float:
        return _clamp01(float(getattr(state, "relative_time", 0.0) or 0.0))


class ScenarioOracle:
    """Computes official normalized utility fractions for a scenario."""

    def __init__(self, scenario: Scenario, max_outcomes: int = 20000) -> None:
        self.scenario = scenario
        self.ufuns = scenario.ufuns
        self.outcomes = self._collect_outcomes(scenario, max_outcomes)
        self.stats = [self._side_stats(0), self._side_stats(1)]

    def official_advantage(self, side: int, agreement: Any | None) -> float:
        if agreement is None:
            return 0.0
        stats = self.stats[side]
        utility = float(self.ufuns[side](agreement))
        return max(0.0, (utility - stats["reserve"]) / stats["off_span"])

    def zone(self, side: int) -> bool:
        return bool(self.stats[side]["zone"])

    def _side_stats(self, side: int) -> dict[str, Any]:
        me = self.ufuns[side]
        them = self.ufuns[1 - side]
        reserve_me = self._reserve(me)
        reserve_them = self._reserve(them)
        utilities = [float(me(outcome)) for outcome in self.outcomes]
        max_utility = max(utilities, default=reserve_me)
        feasible = [
            outcome
            for outcome in self.outcomes
            if float(me(outcome)) > reserve_me and float(them(outcome)) > reserve_them
        ]
        return {
            "reserve": reserve_me,
            "off_span": max(max_utility - reserve_me, 1e-9),
            "zone": bool(feasible),
        }

    @staticmethod
    def _reserve(ufun: Any) -> float:
        reserve = getattr(ufun, "reserved_value", None)
        return float(reserve) if reserve is not None and reserve == reserve else 0.0

    @staticmethod
    def _collect_outcomes(scenario: Scenario, max_outcomes: int) -> list[Any]:
        try:
            return list(
                scenario.outcome_space.enumerate_or_sample(
                    max_cardinality=max_outcomes,
                )
            )
        except Exception:
            return []


def generate_curve_grid(
    *,
    labels: Iterable[str],
    start_fractions: Iterable[float] = (1.0,),
    final_fractions: Iterable[float] = (0.30, 0.35, 0.40, 0.45, 0.50),
    concession_exponents: Iterable[float] = (0.8, 1.2, 1.6, 2.0, 2.6),
    rescue_min_times: Iterable[float] = (0.55, 0.65, 0.75),
    rescue_floor_fractions: Iterable[float] = (0.48, 0.56, 0.64),
    opponent_weight_scales: Iterable[float] = (0.8, 1.0, 1.2),
    late_accept_fractions: Iterable[float] = (0.50, 0.56, 0.62),
) -> list[CurveConfig]:
    curves = []
    for label in labels:
        for (
            start_fraction,
            final_fraction,
            concession_exponent,
            rescue_min_time,
            rescue_floor_fraction,
            opponent_weight_scale,
            late_accept_fraction,
        ) in product(
            start_fractions,
            final_fractions,
            concession_exponents,
            rescue_min_times,
            rescue_floor_fractions,
            opponent_weight_scales,
            late_accept_fractions,
        ):
            name = (
                f"{label}_s{start_fraction:.2f}_f{final_fraction:.2f}"
                f"_e{concession_exponent:.2f}_rt{rescue_min_time:.2f}"
                f"_rf{rescue_floor_fraction:.2f}_ow{opponent_weight_scale:.2f}"
                f"_la{late_accept_fraction:.2f}"
            )
            curves.append(
                CurveConfig(
                    name=name,
                    start_fraction=float(start_fraction),
                    final_fraction=float(final_fraction),
                    concession_exponent=float(concession_exponent),
                    rescue_min_time=float(rescue_min_time),
                    rescue_floor_fraction=float(rescue_floor_fraction),
                    opponent_weight_scale=float(opponent_weight_scale),
                    late_accept_fraction=float(late_accept_fraction),
                    persona_label=label,
                )
            )
    return curves


def score_curve_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_mean_official_adv: float | None = None,
    zero_penalty: float = 0.05,
    timeout_penalty: float = 0.10,
    regression_penalty: float = 0.25,
) -> CurveScore:
    if not rows:
        return CurveScore("", "", 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    official = [float(row.get("official_adv", 0.0) or 0.0) for row in rows]
    agreements = [int(row.get("agreement", 0) or 0) for row in rows]
    timeouts = [int(row.get("timedout", row.get("timeout", 0)) or 0) for row in rows]
    mean_official = sum(official) / len(official)
    agreement_rate = sum(agreements) / len(agreements)
    zero_rate = sum(1 for value in official if value <= 1e-12) / len(official)
    timeout_rate = sum(timeouts) / len(timeouts)
    regression = 0.0
    if baseline_mean_official_adv is not None:
        regression = max(0.0, baseline_mean_official_adv - mean_official)
    objective = (
        mean_official
        - zero_penalty * zero_rate
        - timeout_penalty * timeout_rate
        - regression_penalty * regression
    )
    return CurveScore(
        persona=str(rows[0].get("persona", rows[0].get("personality", ""))),
        curve=str(rows[0].get("curve", "")),
        n=len(rows),
        objective=round(objective, 10),
        mean_official_adv=round(mean_official, 10),
        agreement_rate=round(agreement_rate, 10),
        zero_rate=round(zero_rate, 10),
        timeout_rate=round(timeout_rate, 10),
    )


def select_best_curves(
    rows: list[dict[str, Any]],
    *,
    baselines: dict[str, float] | None = None,
    zero_penalty: float = 0.05,
    timeout_penalty: float = 0.10,
    regression_penalty: float = 0.25,
) -> dict[str, CurveScore]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        persona = str(row.get("persona", row.get("personality", "")))
        curve = str(row.get("curve", ""))
        grouped[(persona, curve)].append(row)

    best: dict[str, CurveScore] = {}
    baselines = baselines or {}
    for (persona, _curve), group in grouped.items():
        score = score_curve_rows(
            group,
            baseline_mean_official_adv=baselines.get(persona),
            zero_penalty=zero_penalty,
            timeout_penalty=timeout_penalty,
            regression_penalty=regression_penalty,
        )
        current = best.get(persona)
        if current is None or _score_key(score) > _score_key(current):
            best[persona] = score
    return best


def run_oracle_curve_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected_personas = _selected_personas(args.personas)
    curves = generate_curve_grid(
        labels=selected_personas.values(),
        start_fractions=_parse_float_list(args.start_fractions),
        final_fractions=_parse_float_list(args.final_fractions),
        concession_exponents=_parse_float_list(args.concession_exponents),
        rescue_min_times=_parse_float_list(args.rescue_min_times),
        rescue_floor_fractions=_parse_float_list(args.rescue_floor_fractions),
        opponent_weight_scales=_parse_float_list(args.opponent_weight_scales),
        late_accept_fractions=_parse_float_list(args.late_accept_fractions),
    )
    curves_by_persona: dict[str, list[CurveConfig]] = defaultdict(list)
    for curve in curves:
        curves_by_persona[curve.persona_label].append(curve)

    rows: list[dict[str, Any]] = []
    scenario_names = args.scenarios or ["Grocery", "Island", "Trade"]
    for scenario_name in scenario_names:
        scenario = Scenario.load(args.scenario_root / scenario_name, ignore_discount=True)
        if scenario is None:
            raise RuntimeError(f"Could not load scenario: {scenario_name}")
        oracle = ScenarioOracle(scenario)
        for opponent_path, persona in selected_personas.items():
            opponent_cls = get_class(opponent_path)
            opponent_name = opponent_path.split(".")[-1]
            for curve in curves_by_persona[persona]:
                for rep in range(args.repeats):
                    random.seed(args.seed + rep)
                    for side in args.sides:
                        row = _run_one_match(
                            scenario=scenario,
                            oracle=oracle,
                            scenario_name=scenario_name,
                            opponent_cls=opponent_cls,
                            opponent_name=opponent_name,
                            opponent_path=opponent_path,
                            persona=persona,
                            curve=curve,
                            side=int(side),
                            rep=rep,
                            steps=args.steps,
                            seconds=args.seconds,
                            max_outcomes=args.max_outcomes,
                        )
                        rows.append(row)
                        print(
                            f"[{persona} {scenario_name} s{side} r{rep} {curve.name}] "
                            f"agr={row['agreement']} official={row['official_adv']:+.3f} "
                            f"steps={row['steps']} {row['seconds']:.1f}s",
                            flush=True,
                        )
    return rows


def run_curve_policy_comparison(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected_personas = _selected_personas(args.personas)
    classifier = None
    if "classifier-coarse" in args.policies:
        classifier = CheckpointOpponentTypeClassifier(
            args.classifier_checkpoint,
            device=args.classifier_device,
        )
    rows: list[dict[str, Any]] = []
    scenario_names = args.scenarios or ["Grocery", "Island", "Trade"]
    for scenario_name in scenario_names:
        scenario = Scenario.load(args.scenario_root / scenario_name, ignore_discount=True)
        if scenario is None:
            raise RuntimeError(f"Could not load scenario: {scenario_name}")
        oracle = ScenarioOracle(scenario)
        for opponent_path, persona in selected_personas.items():
            opponent_cls = get_class(opponent_path)
            opponent_name = opponent_path.split(".")[-1]
            for policy in args.policies:
                curve = curve_for_policy(policy, persona)
                for rep in range(args.repeats):
                    random.seed(args.seed + rep)
                    for side in args.sides:
                        row = _run_one_match(
                            scenario=scenario,
                            oracle=oracle,
                            scenario_name=scenario_name,
                            opponent_cls=opponent_cls,
                            opponent_name=opponent_name,
                            opponent_path=opponent_path,
                            persona=persona,
                            curve=curve,
                            curve_policy=policy,
                            classifier=classifier if policy == "classifier-coarse" else None,
                            side=int(side),
                            rep=rep,
                            steps=args.steps,
                            seconds=args.seconds,
                            max_outcomes=args.max_outcomes,
                        )
                        row["policy"] = policy
                        if policy == "oracle-coarse":
                            row["classifier_label"] = COARSE_PERSONA_LABELS[persona]
                            row["classifier_correct"] = 1
                        elif policy in {"baseline", "oracle-fine"}:
                            row["classifier_label"] = persona
                            row["classifier_correct"] = ""
                        rows.append(row)
                        print(
                            f"[{policy} {persona} {scenario_name} s{side} r{rep} "
                            f"{row['curve']}] agr={row['agreement']} "
                            f"official={row['official_adv']:+.3f} "
                            f"classifier={row['classifier_label']} "
                            f"steps={row['steps']} {row['seconds']:.1f}s",
                            flush=True,
                        )
    return rows


def write_outputs(
    rows: list[dict[str, Any]],
    *,
    out_prefix: Path,
    zero_penalty: float,
    timeout_penalty: float,
    regression_penalty: float,
) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out_prefix.with_suffix(".csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    best_path = out_prefix.with_name(out_prefix.name + "_best.json")

    if rows:
        with open(rows_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_scores = _all_curve_scores(
        rows,
        zero_penalty=zero_penalty,
        timeout_penalty=timeout_penalty,
        regression_penalty=regression_penalty,
    )
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CurveScore.__dataclass_fields__))
        writer.writeheader()
        for score in summary_scores:
            writer.writerow(asdict(score))

    best = select_best_curves(
        rows,
        zero_penalty=zero_penalty,
        timeout_penalty=timeout_penalty,
        regression_penalty=regression_penalty,
    )
    with open(best_path, "w", encoding="utf-8") as handle:
        json.dump({key: asdict(value) for key, value in best.items()}, handle, indent=2)

    print(f"wrote_rows={rows_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_best={best_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-contained SnowyDay v2 oracle curve tuner.",
    )
    subparsers = parser.add_subparsers(dest="command")
    tune = subparsers.add_parser(
        "tune",
        help="Tune curve parameters using true persona labels as an oracle.",
    )
    tune.add_argument("--scenario-root", type=Path, default=SCENARIO_ROOT)
    tune.add_argument("--scenarios", nargs="*", default=["Grocery", "Island", "Trade"])
    tune.add_argument(
        "--personas",
        nargs="*",
        default=["conceder", "fair", "revealing", "misleading", "hardliner", "wall"],
        choices=["conceder", "fair", "revealing", "misleading", "hardliner", "wall"],
    )
    tune.add_argument("--sides", nargs="*", type=int, default=[0, 1])
    tune.add_argument("--repeats", type=int, default=1)
    tune.add_argument("--steps", type=int, default=40)
    tune.add_argument("--seconds", type=int, default=300)
    tune.add_argument("--seed", type=int, default=23)
    tune.add_argument("--max-outcomes", type=int, default=6000)
    tune.add_argument("--out-prefix", type=Path, default=Path("eval_runs/snowy_day_v2_oracle"))
    tune.add_argument("--start-fractions", default="1.0")
    tune.add_argument("--final-fractions", default="0.30,0.35,0.40,0.45,0.50")
    tune.add_argument("--concession-exponents", default="0.8,1.2,1.6,2.0,2.6")
    tune.add_argument("--rescue-min-times", default="0.55,0.65,0.75")
    tune.add_argument("--rescue-floor-fractions", default="0.48,0.56,0.64")
    tune.add_argument("--opponent-weight-scales", default="0.8,1.0,1.2")
    tune.add_argument("--late-accept-fractions", default="0.50,0.56,0.62")
    tune.add_argument("--zero-penalty", type=float, default=0.05)
    tune.add_argument("--timeout-penalty", type=float, default=0.10)
    tune.add_argument("--regression-penalty", type=float, default=0.25)
    compare = subparsers.add_parser(
        "compare-policies",
        help="Compare baseline, true fine persona curves, and true coarse classifier-label curves.",
    )
    compare.add_argument("--scenario-root", type=Path, default=SCENARIO_ROOT)
    compare.add_argument("--scenarios", nargs="*", default=["Grocery", "Island", "Trade"])
    compare.add_argument(
        "--personas",
        nargs="*",
        default=["conceder", "fair", "revealing", "misleading", "hardliner", "wall"],
        choices=["conceder", "fair", "revealing", "misleading", "hardliner", "wall"],
    )
    compare.add_argument(
        "--policies",
        nargs="*",
        default=["baseline", "oracle-fine", "oracle-coarse"],
        choices=["baseline", "oracle-fine", "oracle-coarse", "classifier-coarse"],
    )
    compare.add_argument("--sides", nargs="*", type=int, default=[0, 1])
    compare.add_argument("--repeats", type=int, default=1)
    compare.add_argument("--steps", type=int, default=40)
    compare.add_argument("--seconds", type=int, default=300)
    compare.add_argument("--seed", type=int, default=23)
    compare.add_argument("--max-outcomes", type=int, default=6000)
    compare.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=DEFAULT_CLASSIFIER_CHECKPOINT,
    )
    compare.add_argument("--classifier-device", default="cpu")
    compare.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("eval_runs/snowy_day_v2_policy_compare"),
    )
    compare.add_argument("--zero-penalty", type=float, default=0.05)
    compare.add_argument("--timeout-penalty", type=float, default=0.10)
    compare.add_argument("--regression-penalty", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "tune":
        rows = run_oracle_curve_sweep(args)
        write_outputs(
            rows,
            out_prefix=args.out_prefix,
            zero_penalty=args.zero_penalty,
            timeout_penalty=args.timeout_penalty,
            regression_penalty=args.regression_penalty,
        )
        return
    if args.command == "compare-policies":
        rows = run_curve_policy_comparison(args)
        write_outputs(
            rows,
            out_prefix=args.out_prefix,
            zero_penalty=args.zero_penalty,
            timeout_penalty=args.timeout_penalty,
            regression_penalty=args.regression_penalty,
        )
        return
    parser.print_help()


def _run_one_match(
    *,
    scenario: Scenario,
    oracle: ScenarioOracle,
    scenario_name: str,
    opponent_cls: Any,
    opponent_name: str,
    opponent_path: str,
    persona: str,
    curve: CurveConfig,
    curve_policy: str = "fixed",
    classifier: OpponentTypeClassifier | None = None,
    side: int,
    rep: int,
    steps: int,
    seconds: int,
    max_outcomes: int,
) -> dict[str, Any]:
    started = time.time()
    mechanism = SAOMechanism(
        n_steps=steps,
        time_limit=seconds,
        outcome_space=scenario.outcome_space,
    )
    agent = SnowyDayV2(
        curve=curve,
        curve_policy=curve_policy,
        classifier=classifier,
        max_outcomes=max_outcomes,
        id="SnowyDayV2",
        name="SnowyDayV2",
    )
    opponent = opponent_cls(id=opponent_name, name=opponent_name)
    if side == 0:
        mechanism.add(agent, ufun=scenario.ufuns[0])
        mechanism.add(opponent, ufun=scenario.ufuns[1])
    else:
        mechanism.add(opponent, ufun=scenario.ufuns[0])
        mechanism.add(agent, ufun=scenario.ufuns[1])
    mechanism.run()

    agreement = mechanism.agreement
    final_curve = agent.curve
    prediction = agent.classifier_prediction
    classifier_label = prediction.label if prediction is not None else ""
    classifier_confidence = prediction.confidence if prediction is not None else ""
    classifier_correct = (
        int(classifier_label == COARSE_PERSONA_LABELS[persona])
        if classifier_label
        else ""
    )
    official_adv = oracle.official_advantage(side, agreement)
    timedout = int(
        agreement is None
        or bool(getattr(mechanism.state, "timedout", False))
        or bool(getattr(mechanism.state, "broken", False))
    )
    return {
        "persona": persona,
        "opponent": opponent_name,
        "opponent_path": opponent_path,
        "scenario": scenario_name,
        "side": side,
        "rep": rep,
        "curve": final_curve.name,
        "initial_curve": curve.name,
        "start_fraction": final_curve.start_fraction,
        "final_fraction": final_curve.final_fraction,
        "concession_exponent": final_curve.concession_exponent,
        "rescue_min_time": final_curve.rescue_min_time,
        "rescue_floor_fraction": final_curve.rescue_floor_fraction,
        "opponent_weight_scale": final_curve.opponent_weight_scale,
        "late_accept_fraction": final_curve.late_accept_fraction,
        "classifier_label": classifier_label,
        "classifier_confidence": classifier_confidence,
        "classifier_correct": classifier_correct,
        "classifier_probabilities": (
            json.dumps(prediction.probabilities, sort_keys=True)
            if prediction is not None
            else ""
        ),
        "steps": int(getattr(mechanism, "current_step", 0) or 0),
        "seconds": round(time.time() - started, 3),
        "zone": int(oracle.zone(side)),
        "agreement": int(agreement is not None),
        "timedout": timedout,
        "official_adv": round(official_adv, 6),
        "final_outcome": json.dumps(_jsonable(agreement)),
    }


def _all_curve_scores(
    rows: list[dict[str, Any]],
    *,
    zero_penalty: float,
    timeout_penalty: float,
    regression_penalty: float,
) -> list[CurveScore]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["persona"]), str(row["curve"]))].append(row)
    scores = [
        score_curve_rows(
            group,
            zero_penalty=zero_penalty,
            timeout_penalty=timeout_penalty,
            regression_penalty=regression_penalty,
        )
        for group in grouped.values()
    ]
    return sorted(scores, key=lambda score: (score.persona, -score.objective, score.curve))


def _selected_personas(requested: Iterable[str]) -> dict[str, str]:
    requested_set = set(requested)
    return {
        opponent_path: label
        for opponent_path, label in PERSONA_OPPONENTS.items()
        if label in requested_set
    }


def _parse_float_list(raw: str | Iterable[float]) -> list[float]:
    if isinstance(raw, str):
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    return [float(item) for item in raw]


def _score_key(score: CurveScore) -> tuple[float, float, float, float]:
    return (
        score.objective,
        score.mean_official_adv,
        score.agreement_rate,
        -score.zero_rate,
    )


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _tuple_distance(left: Any, right: Any) -> float:
    if not isinstance(left, tuple) or not isinstance(right, tuple):
        return 0.0 if left == right else 1.0
    length = max(len(left), len(right), 1)
    mismatches = sum(
        1 for index in range(min(len(left), len(right))) if left[index] != right[index]
    )
    mismatches += abs(len(left) - len(right))
    return mismatches / length


def _concession_slope(advantages: list[float]) -> float:
    if len(advantages) < 2:
        return 0.0
    n = len(advantages)
    xs = [index / (n - 1) for index in range(n)]
    x_mean = sum(xs) / n
    y_mean = sum(advantages) / n
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance <= 1e-12:
        return 0.0
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, advantages))
    return _clamp(slope / variance, -1.0, 1.0)


def _repeat_rate(offers: list[Any]) -> float:
    if len(offers) < 2:
        return 0.0
    repeats = sum(1 for left, right in zip(offers, offers[1:]) if left == right)
    return _clamp01(repeats / (len(offers) - 1))


def _stubborn_issue_fraction(offers: list[Any]) -> float:
    tuple_offers = [offer for offer in offers if isinstance(offer, tuple)]
    if len(tuple_offers) < 2:
        return 0.0
    n_issues = max((len(offer) for offer in tuple_offers), default=0)
    if n_issues <= 0:
        return 0.0
    stubborn = 0
    for issue in range(n_issues):
        values = [offer[issue] for offer in tuple_offers if issue < len(offer)]
        if len(values) < 2:
            continue
        counts: dict[Any, int] = {}
        for value in values:
            counts[_hashable_key(value)] = counts.get(_hashable_key(value), 0) + 1
        if max(counts.values(), default=0) / len(values) >= 0.80:
            stubborn += 1
    return _clamp01(stubborn / n_issues)


def _rounds_since_improvement(advantages: list[float], prefix_turns: int) -> float:
    best = float("-inf")
    last_improvement = 0
    for index, advantage in enumerate(advantages):
        if advantage > best + 1e-9:
            best = advantage
            last_improvement = index
    return _clamp01((len(advantages) - 1 - last_improvement) / max(prefix_turns, 1))


def _hashable_key(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    main()
