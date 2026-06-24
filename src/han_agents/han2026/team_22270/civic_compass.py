"""
CivicCompassHANNegotiator for ANAC 2026 HAN.

The agent is a non-LLM HAN negotiator: it uses standard NegMAS decision logic
and deterministic Markdown text to communicate with human partners.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from negmas.preferences import MappingUtilityFunction
from negmas.sao.common import SAOResponse, SAOState, ResponseType
from negmas.sao.negotiators.base import SAOCallNegotiator


EPS = 1e-9


@dataclass(frozen=True)
class _Bid:
    outcome: Any
    own: float
    human: float
    features: tuple[Any, ...]
    face: float


class CivicCompassHANNegotiator(SAOCallNegotiator):
    """
    Human-aware HAN negotiator with deterministic Markdown communication.

    The strategy combines:
    - aspiration-based concession for robust utility,
    - frequency/trend opponent modeling from human offers,
    - issue-level counteroffer explanations,
    - short cooperative messages that avoid pressure or deception.
    """

    _GLOBAL_LLM_CHECKED = False
    _GLOBAL_LLM_READY = False
    _GLOBAL_LLM_HOST = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bids: list[_Bid] = []
        self._frontier: list[_Bid] = []
        self._issue_values: list[list[Any]] = []
        self._issue_own_scores: list[dict[Any, float]] = []
        self._issue_sensitivity: list[float] = []
        self._issue_scores: list[dict[Any, float]] = []
        self._issue_weights: list[float] = []
        self._text_value_boosts: list[dict[Any, float]] = []
        self._text_issue_attention: list[float] = []
        self._human_offers: list[tuple[tuple[Any, ...], float]] = []
        self._sent_counts: dict[str, int] = {}
        self._last_bid: _Bid | None = None
        self._reserved = 0.0
        self._best = 1.0
        self._worst = 0.0
        self._human_floor = 0.15
        self._estimated_human_ufun: MappingUtilityFunction | None = None
        self._last_finality = 0.0
        self._salt = "civic-compass-han"
        self._llm_failed = False
        self._llm_calls = 0
        self._llm_model = os.environ.get("HAN_LLM_MODEL", "qwen3:4b-instruct")
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self._llm_url = f"{host}/api/generate"
        self._llm_timeout = 3.5

    def on_preferences_changed(self, changes: Any) -> None:
        self._reset()
        if self.ufun is None or self.nmi is None:
            return

        self._salt = f"{self.name or self.id or 'civic'}:{getattr(self.nmi, 'id', '')}"
        self._reserved = self._safe_reserved_value()
        base: list[_Bid] = []
        for outcome in self._outcomes():
            own = self._utility(outcome)
            if not math.isfinite(own) or own <= self._reserved + EPS:
                continue
            features = self._features(outcome)
            base.append(
                _Bid(
                    outcome=outcome,
                    own=own,
                    human=0.5,
                    features=features,
                    face=self._face_score(features),
                )
            )

        if not base:
            return

        self._issue_values = self._collect_issue_values(base)
        self._build_own_issue_model(base)
        self._initialize_text_model()
        self._initialize_neutral_model()
        self._bids = [self._with_human_model(b) for b in base]
        self._best = max(b.own for b in self._bids)
        self._worst = min(b.own for b in self._bids)
        self._frontier = self._pareto_frontier(self._bids)
        self._publish_human_model()

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(
                ResponseType.END_NEGOTIATION,
                None,
                data=dict(text="I do not have enough preference information to continue safely."),
            )
        if not self._bids:
            self.on_preferences_changed(None)
        if not self._bids:
            return SAOResponse(
                ResponseType.END_NEGOTIATION,
                None,
                data=dict(text="I cannot identify a valid agreement option in this scenario."),
            )

        t = self._time(state)
        incoming_text = self._incoming_text(state)
        self._last_finality = self._finality_signal(incoming_text)
        self._observe_text(incoming_text, state.current_offer)
        self._observe(state.current_offer, t)
        planned = self._choose_bid(state)
        offer = state.current_offer

        if self._should_accept(offer, planned, t):
            self._publish_human_model()
            return SAOResponse(
                ResponseType.ACCEPT_OFFER,
                offer,
                data=dict(text=self._acceptance_text(offer, t)),
            )

        if planned is None:
            self._publish_human_model()
            return SAOResponse(
                ResponseType.END_NEGOTIATION,
                None,
                data=dict(text="I think we may be too far apart, so I will stop here."),
            )

        key = self._key(planned.outcome)
        self._sent_counts[key] = self._sent_counts.get(key, 0) + 1
        self._last_bid = planned
        self._publish_human_model()
        return SAOResponse(
            ResponseType.REJECT_OFFER,
            planned.outcome,
            data=dict(text=self._counter_text(state, planned, t)),
        )

    def estimate_opponent_ufun(self) -> MappingUtilityFunction | None:
        """Return the current estimate of the human partner's utility."""
        self._publish_human_model()
        return self._estimated_human_ufun

    def get_opponent_ufun_estimate(self) -> MappingUtilityFunction | None:
        """Compatibility alias for evaluators looking for an estimate method."""
        return self.estimate_opponent_ufun()

    @property
    def opponent_model(self) -> MappingUtilityFunction | None:
        """Compatibility property exposing the human preference estimate."""
        return self.estimate_opponent_ufun()

    @property
    def estimated_preferences(self) -> MappingUtilityFunction | None:
        """Compatibility property exposing estimated human preferences."""
        return self.estimate_opponent_ufun()

    def _reset(self) -> None:
        self._bids = []
        self._frontier = []
        self._issue_values = []
        self._issue_own_scores = []
        self._issue_sensitivity = []
        self._issue_scores = []
        self._issue_weights = []
        self._text_value_boosts = []
        self._text_issue_attention = []
        self._human_offers = []
        self._sent_counts = {}
        self._last_bid = None
        self._reserved = 0.0
        self._best = 1.0
        self._worst = 0.0
        self._human_floor = 0.15
        self._estimated_human_ufun = None
        self._last_finality = 0.0

    def _outcomes(self) -> Iterable[Any]:
        try:
            return list(self.nmi.outcome_space.enumerate_or_sample())
        except TypeError:
            return list(self.nmi.outcome_space.enumerate_or_sample(2048))

    def _observe(self, offer: Any, t: float) -> None:
        if offer is None:
            return
        features = self._features(offer)
        if not features:
            return
        self._human_offers.append((features, t))
        if len(self._human_offers) > 60:
            self._human_offers = self._human_offers[-60:]
        self._rebuild_human_model()

    def _incoming_text(self, state: SAOState) -> str:
        texts: list[str] = []
        if isinstance(getattr(state, "current_data", None), dict):
            data = state.current_data or {}
            for key in ("text", "message", "utterance", "content"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
        for _, data in getattr(state, "new_data", []) or []:
            if not isinstance(data, dict):
                continue
            for key in ("text", "message", "utterance", "content"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
        return "\n".join(texts[-3:])

    def _observe_text(self, text: str, offer: Any) -> None:
        if not text or not self._issue_values:
            return
        lowered = text.lower()
        offer_features = self._features(offer) if offer is not None else tuple()
        issue_names = self._issue_names()
        strong_words = (
            "important",
            "priority",
            "need",
            "must",
            "require",
            "care",
            "prefer",
            "want",
            "essential",
            "critical",
        )
        soft_words = ("fair", "equal", "split", "balanced", "reasonable")
        strength = 1.0
        if any(word in lowered for word in strong_words):
            strength += 0.85
        if any(word in lowered for word in soft_words):
            strength += 0.30

        touched = False
        for i, values in enumerate(self._issue_values):
            name = issue_names[i].lower() if i < len(issue_names) else f"issue {i + 1}"
            tokens = [name]
            tokens.extend(part for part in re.split(r"[^a-z0-9]+", name) if len(part) > 2)
            mentioned = any(token and token in lowered for token in tokens)
            if not mentioned:
                for value in values:
                    value_text = str(value).lower()
                    if len(value_text) > 2 and value_text in lowered:
                        mentioned = True
                        break
            if not mentioned:
                continue
            touched = True
            self._text_issue_attention[i] += strength
            direction = self._mentioned_numeric_direction(lowered)
            if direction and all(self._is_number(v) for v in values):
                numeric_values = [float(v) for v in values]
                low = min(numeric_values)
                high = max(numeric_values)
                span = max(EPS, high - low)
                for possible in values:
                    pos = (float(possible) - low) / span
                    desirability = pos if direction > 0 else 1.0 - pos
                    self._text_value_boosts[i][possible] = (
                        self._text_value_boosts[i].get(possible, 0.0)
                        + 0.90 * strength * desirability
                    )
            if i < len(offer_features):
                value = offer_features[i]
                self._text_value_boosts[i][value] = (
                    self._text_value_boosts[i].get(value, 0.0) + 1.35 * strength
                )
        if touched:
            self._rebuild_human_model()

    def _finality_signal(self, text: str) -> float:
        if not text:
            return 0.0
        lowered = text.lower()
        strong = (
            "final offer",
            "last offer",
            "best offer",
            "best i can",
            "take it or leave",
            "cannot move",
            "can't move",
            "no more",
            "not budge",
            "non-negotiable",
        )
        mild = ("deadline", "running out", "close now", "wrap this up", "only offer")
        score = 0.0
        if any(phrase in lowered for phrase in strong):
            score += 1.0
        if any(phrase in lowered for phrase in mild):
            score += 0.45
        return self._clamp(score, 0.0, 1.0)

    def _mentioned_numeric_direction(self, lowered_text: str) -> int:
        lower_phrases = (
            "too high",
            "lower",
            "less",
            "cheaper",
            "decrease",
            "reduce",
            "smaller",
        )
        higher_phrases = (
            "too low",
            "higher",
            "more",
            "increase",
            "larger",
            "raise",
            "bigger",
        )
        if any(phrase in lowered_text for phrase in lower_phrases):
            return -1
        if any(phrase in lowered_text for phrase in higher_phrases):
            return 1
        return 0

    def _rebuild_human_model(self) -> None:
        if not self._issue_values:
            return

        counts: list[dict[Any, float]] = []
        for i, values in enumerate(self._issue_values):
            counts.append(
                {
                    value: 0.18 + 0.72 * self._human_prior_score(i, value)
                    for value in values
                }
            )
        totals = [sum(c.values()) for c in counts]
        n = len(self._human_offers)
        for idx, (features, t) in enumerate(self._human_offers):
            recency = 0.75 + 0.25 * ((idx + 1) / max(1, n))
            early = 0.45 + 1.70 * ((1.0 - t) ** 1.7)
            weight = recency * early
            for i in range(min(len(features), len(counts))):
                value = features[i]
                numeric_values = [float(v) for v in counts[i] if self._is_number(v)]
                if self._is_number(value) and numeric_values:
                    low = min(numeric_values)
                    high = max(numeric_values)
                    scale = max(EPS, (high - low) / 6.5)
                    added = 0.0
                    for possible in list(counts[i]):
                        if not self._is_number(possible):
                            continue
                        delta = weight * math.exp(
                            -abs(float(possible) - float(value)) / scale
                        )
                        counts[i][possible] += delta
                        added += delta
                    totals[i] += added
                else:
                    if value not in counts[i]:
                        counts[i][value] = 0.20
                    counts[i][value] += weight
                    totals[i] += weight

        for i, boosts in enumerate(self._text_value_boosts):
            if i >= len(counts):
                continue
            for value, amount in boosts.items():
                if value not in counts[i]:
                    counts[i][value] = 0.18
                counts[i][value] += amount
                totals[i] += amount

        self._issue_scores = []
        consistency: list[float] = []
        for i, issue_counts in enumerate(counts):
            total = max(EPS, totals[i])
            scores = {value: amount / total for value, amount in issue_counts.items()}
            normalized = self._normalize_mapping(scores)
            trend, strength = self._numeric_trend_scores(i)
            if trend:
                alpha = min(0.62, 0.18 + 0.52 * strength)
                normalized = {
                    value: (1.0 - alpha) * normalized.get(value, 0.0)
                    + alpha * trend.get(value, 0.5)
                    for value in scores
                }
            self._issue_scores.append(normalized)
            consistency.append(max(scores.values()) if scores else 0.5)

        raw_weights = []
        for i, c in enumerate(consistency):
            sensitivity = self._issue_sensitivity[i] if i < len(self._issue_sensitivity) else 0.5
            attention = self._text_issue_attention[i] if i < len(self._text_issue_attention) else 0.0
            raw_weights.append(0.34 + 1.18 * c + 0.70 * sensitivity + 0.42 * attention)
        total_weight = max(EPS, sum(raw_weights))
        self._issue_weights = [w / total_weight for w in raw_weights]
        self._bids = [self._with_human_model(b) for b in self._bids]
        self._frontier = self._pareto_frontier(self._bids)
        offered_scores = [self._model_score(features) for features, _ in self._human_offers]
        if offered_scores:
            tail = sorted(offered_scores)[: max(1, len(offered_scores) // 3)]
            self._human_floor = self._clamp(0.80 * sum(tail) / len(tail), 0.05, 0.72)
        self._publish_human_model()

    def _publish_human_model(self) -> None:
        if not self._bids:
            return
        mapping = {bid.outcome: bid.human for bid in self._bids}
        self._estimated_human_ufun = MappingUtilityFunction(
            mapping=mapping,
            default=0.0,
            reserved_value=self._human_floor,
            outcome_space=getattr(self.nmi, "outcome_space", None),
            name=f"{self.name or 'CivicCompass'}HumanEstimate",
        )
        self.private_info["estimated_opponent_ufun"] = self._estimated_human_ufun
        self.private_info["opponent_model"] = self._estimated_human_ufun
        if self.private_info.get("opponent_ufun") is None:
            self.private_info["opponent_ufun"] = self._estimated_human_ufun

    def _initialize_neutral_model(self) -> None:
        n_issues = max(1, len(self._issue_values))
        raw_weights = [
            0.55 + (self._issue_sensitivity[i] if i < len(self._issue_sensitivity) else 0.0)
            for i in range(n_issues)
        ]
        total = max(EPS, sum(raw_weights))
        self._issue_weights = [w / total for w in raw_weights]
        self._issue_scores = []
        for i, values in enumerate(self._issue_values):
            self._issue_scores.append(
                self._normalize_mapping(
                    {value: self._human_prior_score(i, value) for value in values}
                )
            )

    def _initialize_text_model(self) -> None:
        self._text_value_boosts = [dict() for _ in self._issue_values]
        self._text_issue_attention = [0.0 for _ in self._issue_values]

    def _build_own_issue_model(self, bids: list[_Bid]) -> None:
        n_issues = max((len(b.features) for b in bids), default=0)
        self._issue_own_scores = []
        self._issue_sensitivity = []
        for i in range(n_issues):
            totals: dict[Any, float] = {}
            counts: dict[Any, int] = {}
            for bid in bids:
                value = bid.features[i] if i < len(bid.features) else None
                totals[value] = totals.get(value, 0.0) + bid.own
                counts[value] = counts.get(value, 0) + 1
            averages = {
                value: totals[value] / max(1, counts[value]) for value in totals
            }
            normalized = self._normalize_mapping(averages)
            self._issue_own_scores.append(normalized)
            if normalized:
                self._issue_sensitivity.append(max(normalized.values()) - min(normalized.values()))
            else:
                self._issue_sensitivity.append(0.0)

    def _human_prior_score(self, issue_index: int, value: Any) -> float:
        own_score = 0.5
        if issue_index < len(self._issue_own_scores):
            own_score = self._issue_own_scores[issue_index].get(value, 0.5)
        complement = 1.0 - own_score
        jitter = self._hash01("human-prior", issue_index, value)
        return self._clamp(0.86 * complement + 0.14 * jitter, 0.0, 1.0)

    def _choose_bid(self, state: SAOState) -> _Bid | None:
        if not self._bids:
            return None
        t = self._time(state)
        aspiration = self._aspiration(t)
        base = self._frontier or self._bids
        spread = max(EPS, self._best - self._reserved)
        own_floors = [
            aspiration,
            aspiration - 0.045 * spread,
            aspiration - 0.105 * spread,
            aspiration - 0.185 * spread,
            self._reserved + 0.16 * spread,
        ]
        human_floors = [
            self._human_floor + 0.30 * (1.0 - t),
            self._human_floor + 0.16 * (1.0 - t),
            self._human_floor + 0.06 * (1.0 - t),
            self._human_floor,
            0.0,
        ]
        for own_floor in own_floors:
            for human_floor in human_floors:
                pool = [
                    b
                    for b in base
                    if b.own >= own_floor - EPS and b.human >= human_floor - EPS
                ]
                if pool:
                    return max(pool, key=lambda b: self._bid_score(b, t, aspiration))
        return max(self._bids, key=lambda b: self._bid_score(b, t, aspiration))

    def _should_accept(self, offer: Any, planned: _Bid | None, t: float) -> bool:
        if offer is None:
            return False
        offered = self._utility(offer)
        if offered <= self._reserved + EPS:
            return False
        spread = max(EPS, self._best - self._reserved)
        minimum_fraction = 0.16 + 0.50 * ((1.0 - t) ** 1.75)
        if offered < self._reserved + minimum_fraction * spread:
            return False
        planned_utility = planned.own if planned is not None else self._aspiration(t)
        if offered + spread * (0.012 + 0.020 * t) >= planned_utility:
            return True
        if offered + spread * (0.015 + 0.035 * t * t) >= self._aspiration(t):
            return True
        if t > 0.945 and offered >= self._reserved + 0.22 * spread:
            return True
        if (
            self._last_finality > 0.0
            and t > 0.86
            and offered >= self._reserved + (0.11 - 0.04 * self._last_finality) * spread
        ):
            return True
        if t > 0.985 and offered >= self._reserved + 0.155 * spread:
            return True
        return False

    def _bid_score(self, bid: _Bid, t: float, aspiration: float) -> float:
        own_norm = self._norm(bid.own, self._reserved, self._best)
        human_norm = self._clamp(bid.human, 0.0, 1.0)
        surplus = max(0.0, bid.own - self._reserved)
        joint = math.sqrt(max(0.0, surplus * max(0.0, bid.human - self._human_floor)))
        repeat_penalty = 0.020 * self._sent_counts.get(self._key(bid.outcome), 0)
        aspiration_gap = abs(bid.own - aspiration) / max(EPS, self._best - self._reserved)
        empathy = 0.82 * ((1.0 - t) ** 1.1) + 0.24
        self_weight = 1.05 + 0.90 * (t**1.55)
        anchor_bonus = 0.035 * bid.face if t < 0.33 else 0.0
        return (
            self_weight * own_norm
            + empathy * human_norm
            + 0.20 * joint
            + 0.13 * bid.face
            + anchor_bonus
            - 0.11 * aspiration_gap
            - repeat_penalty
        )

    def _aspiration(self, t: float) -> float:
        spread = max(EPS, self._best - self._reserved)
        concession = 0.055 * t + 0.905 * (t**3.85)
        return self._reserved + spread * (1.0 - concession)

    def _acceptance_text(self, offer: Any, t: float) -> str:
        utility = self._norm(self._utility(offer), self._reserved, self._best)
        if utility > 0.82:
            return (
                "Thank you. This proposal gives me enough value to agree, "
                "and I appreciate the movement toward a workable outcome."
            )
        return (
            "I can accept this. It is not perfect for me, but it is a reasonable "
            "agreement and I would rather conclude constructively than risk no deal."
        )

    def _counter_text(self, state: SAOState, bid: _Bid, t: float) -> str:
        if state.current_offer is None:
            return self._opening_text(bid, state, t)

        differences = self._differences(state.current_offer, bid.outcome)
        phase = "early" if t < 0.35 else "middle" if t < 0.75 else "late"
        prefix = {
            "early": "Thanks for the proposal.",
            "middle": "I think there is a deal here.",
            "late": "Given the time left, I want to close on something workable.",
        }[phase]
        trade = self._trade_summary(state.current_offer, bid.outcome)
        acknowledgement = self._acknowledge_text(self._incoming_text(state))
        if differences:
            detail = "; ".join(differences[:2])
            if trade:
                message = (
                    f"{prefix}{acknowledgement} I can move to "
                    f"**{self._format_outcome(bid.outcome)}**. {trade} "
                    f"The concrete change is: {detail}. If that movement works for you, "
                    "this gives us a realistic path to close."
                )
                return self._refine_message("counter", message, state, bid, t)
            message = (
                f"{prefix}{acknowledgement} I can move to "
                f"**{self._format_outcome(bid.outcome)}**. The concrete change is: "
                f"{detail}. This is the closest package I can support while still "
                "keeping the agreement constructive."
            )
            return self._refine_message("counter", message, state, bid, t)
        message = (
            f"{prefix}{acknowledgement} I am proposing "
            f"**{self._format_outcome(bid.outcome)}** as a balanced package. "
            "It protects my key constraints while leaving room for your priorities."
        )
        return self._refine_message("counter", message, state, bid, t)

    def _opening_text(self, bid: _Bid, state: SAOState | None = None, t: float = 0.0) -> str:
        trade = self._opening_trade_hint(bid.outcome)
        message = (
            f"Hello, and thank you for negotiating with me. I will start with "
            f"**{self._format_outcome(bid.outcome)}**. {trade} "
            "If one issue is especially important to you, say so and I will trade around it."
        )
        return self._refine_message("opening", message, state, bid, t)

    def _refine_message(
        self,
        action: str,
        fallback: str,
        state: SAOState | None,
        bid: _Bid | None,
        t: float,
    ) -> str:
        if not self._can_use_llm(action):
            return fallback
        prompt = self._llm_prompt(action, fallback, state, bid, t)
        refined = self._call_ollama(prompt)
        if not refined:
            return fallback
        cleaned = self._clean_llm_message(refined)
        if not cleaned:
            return fallback
        return cleaned

    def _can_use_llm(self, action: str) -> bool:
        if self._llm_failed or self._llm_calls >= 8:
            return False
        disabled = os.environ.get("CIVIC_COMPASS_DISABLE_LLM", "").lower()
        if disabled in {"1", "true", "yes", "on"}:
            return False
        return action in {"opening", "counter"}

    def _llm_prompt(
        self,
        action: str,
        fallback: str,
        state: SAOState | None,
        bid: _Bid | None,
        t: float,
    ) -> str:
        their_offer = self._format_outcome(state.current_offer) if state and state.current_offer is not None else "none"
        our_offer = self._format_outcome(bid.outcome) if bid is not None else "none"
        partner_text = self._incoming_text(state) if state is not None else ""
        trade = (
            self._trade_summary(state.current_offer, bid.outcome)
            if state is not None and state.current_offer is not None and bid is not None
            else self._opening_trade_hint(bid.outcome) if bid is not None else ""
        )
        phase = "early" if t < 0.35 else "middle" if t < 0.75 else "late"
        return (
            "Write one concise message to a human negotiation partner.\n"
            "Rules: keep the formal offer unchanged; do not reveal numeric utilities, algorithms, or hidden preferences; "
            "do not mention being an AI model; do not threaten; use at most two short sentences; return plain text only.\n"
            "Goal: maximize our deal value while sounding fair, specific, and cooperative.\n"
            f"Phase: {phase}. Action: {action}.\n"
            f"Their last offer: {their_offer}.\n"
            f"Our formal offer: {our_offer}.\n"
            f"Partner message: {partner_text or 'none'}.\n"
            f"Tradeoff to communicate: {trade or 'ask what matters most and invite a concrete trade'}.\n"
            f"Baseline message: {fallback}\n"
            "Improved message:"
        )

    def _call_ollama(self, prompt: str) -> str | None:
        if not self._ensure_ollama_ready():
            return None
        self._llm_calls += 1
        payload = {
            "model": self._llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.35,
                "top_p": 0.85,
                "num_predict": 90,
            },
        }
        request = urllib.request.Request(
            self._llm_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._llm_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            self._llm_failed = True
            return None
        text = data.get("response")
        return text if isinstance(text, str) else None

    def _ensure_ollama_ready(self) -> bool:
        cls = type(self)
        host = self._llm_url.split("/api/generate", 1)[0]
        if cls._GLOBAL_LLM_CHECKED and cls._GLOBAL_LLM_HOST == host:
            if not cls._GLOBAL_LLM_READY:
                self._llm_failed = True
            return cls._GLOBAL_LLM_READY

        cls._GLOBAL_LLM_CHECKED = True
        cls._GLOBAL_LLM_HOST = host
        tags_url = f"{host}/api/tags"
        try:
            with urllib.request.urlopen(tags_url, timeout=0.25) as response:
                cls._GLOBAL_LLM_READY = 200 <= getattr(response, "status", 200) < 300
        except (OSError, urllib.error.URLError, TimeoutError):
            cls._GLOBAL_LLM_READY = False
            self._llm_failed = True
        return cls._GLOBAL_LLM_READY

    def _clean_llm_message(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^(message|text|improved message)\s*:\s*", "", cleaned, flags=re.I).strip()
        cleaned = cleaned.strip("\"' \n\t")
        if not cleaned:
            return ""
        forbidden = ("utility", "algorithm", "model", "hidden preference", "json", "reasoning")
        if any(word in cleaned.lower() for word in forbidden):
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        cleaned = " ".join(sentences[:2]).strip()
        if len(cleaned) > 420:
            cleaned = cleaned[:417].rstrip() + "..."
        return cleaned

    def _differences(self, their_offer: Any, my_offer: Any) -> list[str]:
        their = self._features(their_offer)
        mine = self._features(my_offer)
        issue_names = self._issue_names()
        differences: list[str] = []
        for i in range(min(len(their), len(mine))):
            if their[i] == mine[i]:
                continue
            name = issue_names[i] if i < len(issue_names) else f"issue {i + 1}"
            own_delta = self._own_value_score(i, mine[i]) - self._own_value_score(i, their[i])
            if self._is_number(their[i]) and self._is_number(mine[i]):
                direction = "higher" if float(mine[i]) > float(their[i]) else "lower"
                if own_delta < -0.04:
                    differences.append(
                        f"I move {name} {direction}, from {their[i]} to {mine[i]}, in your direction"
                    )
                else:
                    differences.append(
                        f"{name} moves {direction}, from {their[i]} to {mine[i]}"
                    )
            else:
                if own_delta < -0.04:
                    differences.append(f"I concede {name} from {their[i]} to {mine[i]}")
                else:
                    differences.append(f"{name} changes from {their[i]} to {mine[i]}")
        return differences

    def _format_outcome(self, outcome: Any) -> str:
        features = self._features(outcome)
        names = self._issue_names()
        if not features:
            return str(outcome)
        parts = []
        for i, value in enumerate(features):
            name = names[i] if i < len(names) else f"issue {i + 1}"
            parts.append(f"{name}={value}")
        if len(parts) <= 5:
            return ", ".join(parts)
        important = sorted(
            range(len(parts)),
            key=lambda i: self._issue_sensitivity[i] if i < len(self._issue_sensitivity) else 0.0,
            reverse=True,
        )[:4]
        compact = [parts[i] for i in sorted(important)]
        return ", ".join(compact) + f", plus {len(parts) - len(compact)} other issues"

    def _trade_summary(self, their_offer: Any, my_offer: Any) -> str:
        their = self._features(their_offer)
        mine = self._features(my_offer)
        names = self._issue_names()
        concessions: list[str] = []
        asks: list[str] = []
        for i in range(min(len(their), len(mine))):
            if their[i] == mine[i]:
                continue
            name = names[i] if i < len(names) else f"issue {i + 1}"
            delta = self._own_value_score(i, mine[i]) - self._own_value_score(i, their[i])
            if delta < -0.04:
                concessions.append(name)
            elif delta > 0.04:
                asks.append(name)
        if concessions and asks:
            return (
                f"I am giving ground on {self._join_names(concessions[:2])}; "
                f"in exchange I need movement on {self._join_names(asks[:2])}."
            )
        if concessions:
            return f"I am deliberately giving ground on {self._join_names(concessions[:2])}."
        if asks:
            return f"The part I still need to protect is {self._join_names(asks[:2])}."
        return ""

    def _opening_trade_hint(self, outcome: Any) -> str:
        features = self._features(outcome)
        names = self._issue_names()
        if not features:
            return "I am aiming for a serious agreement, not a deadlock."
        ranked = sorted(
            range(min(len(features), len(self._issue_values))),
            key=lambda i: self._issue_sensitivity[i] if i < len(self._issue_sensitivity) else 0.0,
            reverse=True,
        )
        asks: list[str] = []
        gives: list[str] = []
        for i in ranked:
            name = names[i] if i < len(names) else f"issue {i + 1}"
            score = self._own_value_score(i, features[i])
            if score > 0.68 and len(asks) < 2:
                asks.append(name)
            elif score < 0.38 and len(gives) < 2:
                gives.append(name)
        if asks and gives:
            return (
                f"I am prioritizing {self._join_names(asks)}, while leaving "
                f"{self._join_names(gives)} more flexible."
            )
        if asks:
            return f"My main constraint is {self._join_names(asks)}."
        return "I chose this as a firm but negotiable starting point."

    def _acknowledge_text(self, text: str) -> str:
        if not text:
            return ""
        lowered = text.lower()
        if any(word in lowered for word in ("fair", "equal", "balanced", "reasonable")):
            return " I hear the fairness concern."
        if any(word in lowered for word in ("need", "important", "priority", "must")):
            return " I hear that some issues matter more to you."
        if any(word in lowered for word in ("please", "thanks", "thank")):
            return " I appreciate the cooperative tone."
        return " I considered your message."

    def _join_names(self, names: list[str]) -> str:
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + f" and {names[-1]}"

    def _own_value_score(self, issue_index: int, value: Any) -> float:
        if issue_index < len(self._issue_own_scores):
            return self._issue_own_scores[issue_index].get(value, 0.5)
        return 0.5

    def _issue_names(self) -> list[str]:
        try:
            issues = getattr(self.nmi.outcome_space, "issues", None) or []
            names = [getattr(issue, "name", f"issue {i + 1}") for i, issue in enumerate(issues)]
            if len(names) == 1 and self._issue_values and len(self._issue_values) > 1:
                return [f"{names[0]} part {i + 1}" for i in range(len(self._issue_values))]
            return names
        except Exception:
            return []

    def _pareto_frontier(self, bids: list[_Bid]) -> list[_Bid]:
        frontier: list[_Bid] = []
        best_human = -math.inf
        for bid in sorted(bids, key=lambda b: (b.own, b.human), reverse=True):
            if bid.human > best_human + EPS:
                frontier.append(bid)
                best_human = bid.human
        return frontier or bids[:]

    def _with_human_model(self, bid: _Bid) -> _Bid:
        return _Bid(
            outcome=bid.outcome,
            own=bid.own,
            human=self._model_score(bid.features),
            features=bid.features,
            face=bid.face,
        )

    def _model_score(self, features: tuple[Any, ...]) -> float:
        if not self._issue_scores or not self._issue_weights:
            return 0.5
        score = 0.0
        for i, weight in enumerate(self._issue_weights):
            value = features[i] if i < len(features) else None
            score += weight * self._issue_scores[i].get(value, 0.25)
        similarity = 0.5
        if self._human_offers:
            total = 0.0
            weighted = 0.0
            n = len(self._human_offers)
            for idx, (other, _) in enumerate(self._human_offers[-16:]):
                recency = 0.65 + 0.35 * ((idx + 1) / min(16, n))
                issue_total = 0.0
                issue_match = 0.0
                for i in range(min(len(features), len(other), len(self._issue_weights))):
                    weight = self._issue_weights[i]
                    issue_total += weight
                    if features[i] == other[i]:
                        issue_match += weight
                    elif self._is_number(features[i]) and self._is_number(other[i]):
                        values = (
                            [
                                float(v)
                                for v in self._issue_values[i]
                                if self._is_number(v)
                            ]
                            if i < len(self._issue_values)
                            else []
                        )
                        if values:
                            span = max(EPS, max(values) - min(values))
                            distance = abs(float(features[i]) - float(other[i])) / span
                            issue_match += weight * max(0.0, 1.0 - distance)
                total += recency
                weighted += recency * (issue_match / max(EPS, issue_total))
            similarity = weighted / max(EPS, total)
        return self._clamp(0.78 * score + 0.22 * similarity, 0.0, 1.0)

    def _collect_issue_values(self, bids: list[_Bid]) -> list[list[Any]]:
        n_issues = max((len(b.features) for b in bids), default=1)
        values: list[list[Any]] = []
        for i in range(n_issues):
            seen: dict[str, Any] = {}
            for bid in bids:
                value = bid.features[i] if i < len(bid.features) else None
                seen.setdefault(repr(value), value)
            values.append(list(seen.values()))
        return values

    def _numeric_trend_scores(self, issue_index: int) -> tuple[dict[Any, float], float]:
        if issue_index >= len(self._issue_values):
            return {}, 0.0
        domain = self._issue_values[issue_index]
        if len(domain) < 2 or not all(self._is_number(v) for v in domain):
            return {}, 0.0
        observations: list[tuple[float, float]] = []
        for features, t in self._human_offers:
            if issue_index >= len(features) or not self._is_number(features[issue_index]):
                continue
            observations.append((float(features[issue_index]), 1.0 - 0.78 * (t**1.12)))
        if len(observations) < 2:
            return {}, 0.0
        xs = [x for x, _ in observations]
        ys = [y for _, y in observations]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        vx = sum((x - mx) ** 2 for x in xs)
        vy = sum((y - my) ** 2 for y in ys)
        if vx <= EPS or vy <= EPS:
            return {}, 0.0
        cov = sum((x - mx) * (y - my) for x, y in observations)
        corr = cov / math.sqrt(vx * vy)
        strength = min(1.0, abs(corr))
        if strength < 0.08:
            return {}, 0.0
        nums = [float(v) for v in domain]
        low, high = min(nums), max(nums)
        span = max(EPS, high - low)
        return {
            value: ((float(value) - low) / span if corr >= 0 else 1.0 - (float(value) - low) / span)
            for value in domain
        }, strength

    def _features(self, outcome: Any) -> tuple[Any, ...]:
        if outcome is None:
            return tuple()
        if isinstance(outcome, dict):
            names = list(getattr(self.nmi.outcome_space, "issue_names", []) or [])
            if names:
                return self._expanded_features(outcome.get(name) for name in names)
            return self._expanded_features(outcome[key] for key in sorted(outcome))
        if isinstance(outcome, tuple):
            return self._expanded_features(outcome)
        if isinstance(outcome, list):
            return self._expanded_features(outcome)
        return self._expanded_features((outcome,))

    def _expanded_features(self, values: Iterable[Any]) -> tuple[Any, ...]:
        features: list[Any] = []
        for value in values:
            if isinstance(value, str) and "_" in value:
                parts = value.split("_")
                if parts and all(self._is_number(part) for part in parts):
                    features.extend(float(part) for part in parts)
                    continue
            features.append(value)
        return tuple(features)

    def _face_score(self, features: tuple[Any, ...]) -> float:
        if not features:
            return 0.5
        weights = [0.30 + self._hash01("face-w", i) for i in range(len(features))]
        total = sum(weights)
        return sum(
            weights[i] * self._hash01("face-v", i, value)
            for i, value in enumerate(features)
        ) / max(EPS, total)

    def _time(self, state: SAOState) -> float:
        for attr in ("relative_time", "time"):
            value = getattr(state, attr, None)
            if value is not None:
                try:
                    return self._clamp(float(value), 0.0, 1.0)
                except (TypeError, ValueError):
                    pass
        step = getattr(state, "step", None)
        n_steps = getattr(self.nmi, "n_steps", None)
        if step is not None and n_steps:
            return self._clamp(float(step) / max(1.0, float(n_steps)), 0.0, 1.0)
        return 0.0

    def _utility(self, outcome: Any) -> float:
        if self.ufun is None or outcome is None:
            return -math.inf
        try:
            value = float(self.ufun(outcome))
        except Exception:
            return -math.inf
        return -math.inf if math.isnan(value) else value

    def _safe_reserved_value(self) -> float:
        try:
            value = float(getattr(self.ufun, "reserved_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(value) else value

    def _normalize_mapping(self, mapping: dict[Any, float]) -> dict[Any, float]:
        if not mapping:
            return mapping
        low = min(mapping.values())
        high = max(mapping.values())
        span = max(EPS, high - low)
        return {key: (value - low) / span for key, value in mapping.items()}

    def _norm(self, value: float, low: float, high: float) -> float:
        return self._clamp((value - low) / max(EPS, high - low), 0.0, 1.0)

    def _clamp(self, value: float, low: float, high: float) -> float:
        if low > high:
            low, high = high, low
        return max(low, min(high, value))

    def _hash01(self, *parts: Any) -> float:
        blob = "|".join([self._salt] + [repr(p) for p in parts]).encode("utf-8")
        digest = hashlib.blake2b(blob, digest_size=8).digest()
        return int.from_bytes(digest, "big") / float(2**64 - 1)

    def _is_number(self, value: Any) -> bool:
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False

    def _key(self, outcome: Any) -> str:
        return repr(outcome)
