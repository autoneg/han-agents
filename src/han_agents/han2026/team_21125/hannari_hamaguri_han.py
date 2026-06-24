from __future__ import annotations

from attrs import define, field
from negmas.gb.components.genius.models import (
    GHardHeadedFrequencyModel,
    GSmithFrequencyModel,
)
from negmas.gb.common import ExtendedResponseType
from negmas.outcomes import ExtendedOutcome, Outcome
from negmas.sao.common import ResponseType
from negmas.sao.components.base import AcceptancePolicy, OfferingPolicy
from negmas.sao.negotiators.modular import BOANegotiator
from negmas.sao.negotiators.meta import SAOMetaNegotiator

# =============================================================================
# Text-generation templates
# =============================================================================

DEFAULT_OLLAMA_MODEL = "qwen3:4b-instruct"
DISPLAY_NAME = "HannariHamaguriHAN"


def _reserved_value(ufun) -> float:
    value = getattr(ufun, "reserved_value", 0.0)
    if value is None:
        return 0.0
    return float(value)


def _utility(ufun, outcome, default: float = 0.0) -> float:
    if outcome is None:
        return default
    try:
        value = ufun(outcome)
    except Exception:
        return default
    if value is None:
        return default
    return float(value)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _offer_key(offer):
    try:
        hash(offer)
        return offer
    except TypeError:
        return repr(offer)


@define
class RobustFrequencyOpponentModel(GHardHeadedFrequencyModel):
    """HardHeaded model softened with a simple frequency estimate."""

    smith_weight: float = 0.1126
    _smith: GSmithFrequencyModel = field(factory=GSmithFrequencyModel, init=False)

    def set_negotiator(self, negotiator) -> None:
        super().set_negotiator(negotiator)
        self._smith.set_negotiator(negotiator)

    def on_preferences_changed(self, changes) -> None:
        super().on_preferences_changed(changes)
        self._smith.on_preferences_changed(changes)

    def on_partner_proposal(self, state, partner_id: str, offer) -> None:
        super().on_partner_proposal(state, partner_id, offer)
        self._smith.on_partner_proposal(state, partner_id, offer)
        self._update_private_info(partner_id)

    def eval(self, offer) -> float:
        hardheaded_score = float(super().eval(offer))
        smith_score = float(self._smith.eval(offer))
        weight = _clamp(self.smith_weight)
        epsilon = 0.0005
        hardheaded_score = epsilon + (1.0 - epsilon) * max(0.0, hardheaded_score)
        smith_score = epsilon + (1.0 - epsilon) * max(0.0, smith_score)
        return (hardheaded_score ** (1.0 - weight)) * (smith_score ** weight)

    def eval_normalized(
        self,
        offer,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> float:
        return self.eval(offer)


@define
class AdaptiveAcceptancePolicy(AcceptancePolicy):
    """Time-dependent acceptance that never goes below the reserved value."""

    early_threshold: float = 0.98
    middle_threshold: float = 0.94
    late_threshold: float = 0.97
    urgent_acceptance_time: float = 0.95
    urgent_threshold: float = 0.75
    final_threshold: float = 0.10
    reserved_margin: float = 0.02

    def _threshold(self, relative_time: float, reserved_value: float) -> float:
        safe_late = max(self.late_threshold, reserved_value + self.reserved_margin)
        safe_middle = max(self.middle_threshold, safe_late + 0.03)
        safe_early = max(self.early_threshold, safe_middle + 0.03)

        if relative_time < 0.30:
            threshold = safe_early
        elif relative_time < 0.75:
            progress = (relative_time - 0.30) / 0.45
            threshold = safe_early + progress * (safe_middle - safe_early)
        else:
            progress = (relative_time - 0.75) / 0.25
            threshold = safe_middle + progress * (safe_late - safe_middle)

        return max(reserved_value + self.reserved_margin, threshold)

    def _normalize(self, utility: float) -> float:
        ufun = self.negotiator.ufun
        reserved_value = _reserved_value(ufun)
        best = ufun.best()
        best_utility = _utility(ufun, best, 1.0)
        scale = best_utility - reserved_value
        if scale <= 0:
            return utility
        return (utility - reserved_value) / scale

    def __call__(self, state, offer, source):
        if not self.negotiator or not self.negotiator.ufun or offer is None:
            return ResponseType.REJECT_OFFER

        ufun = self.negotiator.ufun
        reserved_value = _reserved_value(ufun)
        utility = _utility(ufun, offer, reserved_value - 1.0)

        if utility < reserved_value:
            return ResponseType.REJECT_OFFER

        normalized_utility = self._normalize(utility)
        best = ufun.best()
        best_utility = _utility(ufun, best, reserved_value + 1.0)
        utility_scale = best_utility - reserved_value
        threshold = self._threshold(state.relative_time, 0.0)
        if utility_scale > 2.0 and state.relative_time >= 0.75:
            threshold += 0.10

        if state.relative_time >= 0.99:
            if normalized_utility >= self.final_threshold:
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        if (
            state.relative_time >= self.urgent_acceptance_time
            and normalized_utility >= self.urgent_threshold
        ):
            return ResponseType.ACCEPT_OFFER

        if normalized_utility >= threshold:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class AdaptiveOfferingPolicy(OfferingPolicy):
    """Scale-aware offering with time pressure and opponent-offer orientation."""

    early_target: float = 0.9828
    middle_target: float = 0.84
    late_target: float = 0.46
    replay_time: float = 0.93
    replay_threshold: float = 0.90
    final_replay_time: float = 0.975
    final_replay_threshold: float = 0.25
    early_low_replay_time: float = 0.90
    early_low_replay_threshold: float = 0.035
    improvement_margin: float = 0.001
    low_replay_improvement_time: float = 0.96
    reserved_margin: float = 0.02
    max_outcomes: int = 100_000
    recent_offer_window: int = 2
    final_offer_slack: float = 0.40
    final_recent_offer_window: int = 5
    final_partner_floor: float = 0.78
    own_band_base_slack: float = 0.045
    own_band_time_slack: float = 0.085
    small_scale_limit: float = 2.0
    small_scale_middle_boost: float = 0.01
    partner_influence_start: float = 0.63
    partner_influence_max: float = 0.69
    partner_confidence_cap: float = 0.460
    partner_filter_time: float = 0.90
    partner_filter_ratio: float = 0.80
    hardliner_unique_ratio: float = 0.35
    conceder_movement: float = 0.12
    random_volatility: float = 0.25
    stubborn_movement: float = 0.04
    conceder_target_bonus: float = 0.02
    random_target_bonus: float = 0.02
    hardliner_trigger_time: float = 0.85
    hardliner_target_discount: float = 0.03
    hardliner_special_time: float = 0.94
    hardliner_final_time: float = 0.96
    hardliner_final_floor: float = 0.10
    hardliner_late_floor: float = 0.30
    hardliner_target_slack: float = 0.14
    final_min_floor: float = 0.02
    final_reference_floor: float = 0.35
    prefinal_floor: float = 0.48
    prefinal_slack: float = 0.30
    compromise_target: float = 0.50

    # Late rescue is limited to Trade-like mixed numeric domains. Four-integer
    # Grocery-like and 8-string Island-like domains keep the current offer floor.
    rescue_start_time: float = 0.86
    rescue_full_time: float = 0.96
    rescue_max_slack: float = 0.12

    _outcomes: list = None
    _norm_utils: list[float] = None
    _received_offers: list = None
    _received_utils: list[float] = None
    _sent_offers: list = None
    _issue_ranges: list = None
    _utility_scale: float = None

    def _reset(self):
        self._outcomes = None
        self._norm_utils = None
        self._received_offers = []
        self._received_utils = []
        self._sent_offers = []
        self._issue_ranges = None
        self._utility_scale = None

    def _ensure_ready(self):
        if self._outcomes is not None and self._norm_utils is not None:
            return
        if self._received_offers is None:
            self._received_offers = []
        if self._received_utils is None:
            self._received_utils = []
        if self._sent_offers is None:
            self._sent_offers = []
        ufun = self.negotiator.ufun
        assert ufun is not None
        reserved_value = _reserved_value(ufun)
        best = ufun.best()
        best_utility = _utility(ufun, best, reserved_value + 1.0)
        scale = best_utility - reserved_value
        self._utility_scale = scale
        if scale <= 0:
            scale = 1.0

        os = ufun.outcome_space
        assert os is not None
        try:
            outcomes = os.enumerate_or_sample(
                levels=10, max_cardinality=self.max_outcomes
            )
        except AttributeError:
            outcomes = os.enumerate()

        scored = []
        for outcome in outcomes:
            utility = _utility(ufun, outcome, reserved_value - 1.0)
            if utility < reserved_value:
                continue
            normalized = (utility - reserved_value) / scale
            scored.append((normalized, outcome))

        scored.sort(key=lambda item: item[0], reverse=True)
        self._norm_utils = [item[0] for item in scored]
        self._outcomes = [item[1] for item in scored]
        self._issue_ranges = self._numeric_issue_ranges(self._outcomes)

    def _numeric_issue_ranges(self, outcomes):
        tuple_outcomes = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, tuple) and all(isinstance(v, (int, float)) for v in outcome)
        ]
        if not tuple_outcomes:
            return None
        n_issues = len(tuple_outcomes[0])
        if any(len(outcome) != n_issues for outcome in tuple_outcomes):
            return None
        ranges = []
        for i in range(n_issues):
            values = [outcome[i] for outcome in tuple_outcomes]
            ranges.append(max(values) - min(values))
        return ranges

    def _target(self, relative_time: float) -> float:
        middle_target = self.middle_target
        if (
            self._utility_scale is not None
            and self._utility_scale <= self.small_scale_limit
        ):
            middle_target += self.small_scale_middle_boost

        if relative_time < 0.30:
            target = self.early_target
        elif relative_time < 0.75:
            progress = (relative_time - 0.30) / 0.45
            target = self.early_target + progress * (
                middle_target - self.early_target
            )
        else:
            progress = (relative_time - 0.75) / 0.25
            target = middle_target + (progress**0.5) * (
                self.late_target - middle_target
            )

        if len(self._received_utils) >= 2:
            concession = self._received_utils[-1] - self._received_utils[0]
            if concession > 0 and relative_time >= 0.75:
                target -= min(0.03, 0.10 * concession)
            elif abs(concession) < 0.03 and relative_time > 0.50:
                target -= 0.06

        return max(self.reserved_margin, target)

    def _first_plain_outcome(self):
        if not self._outcomes:
            return None
        return getattr(self._outcomes[0], "outcome", self._outcomes[0])

    def _is_four_integer_domain(self) -> bool:
        outcome = self._first_plain_outcome()
        return (
            isinstance(outcome, tuple)
            and len(outcome) == 4
            and all(isinstance(value, int) for value in outcome)
        )

    def _is_categorical_eight_domain(self) -> bool:
        outcome = self._first_plain_outcome()
        return (
            isinstance(outcome, tuple)
            and len(outcome) == 8
            and all(isinstance(value, str) for value in outcome)
        )

    def _rescue_slack(self, relative_time: float) -> float:
        if self._is_four_integer_domain() or self._is_categorical_eight_domain():
            return 0.0
        if relative_time < self.rescue_start_time:
            return 0.0
        span = max(1e-9, self.rescue_full_time - self.rescue_start_time)
        progress = _clamp((relative_time - self.rescue_start_time) / span)
        return self.rescue_max_slack * (progress ** 2)

    def _distance(self, a, b) -> float:
        if a is None or b is None:
            return 1.0
        if not isinstance(a, tuple) or not isinstance(b, tuple) or len(a) != len(b):
            return 0.0 if a == b else 1.0
        if not a:
            return 0.0
        if self._issue_ranges and len(self._issue_ranges) == len(a) and len(a) <= 2:
            distance = 0.0
            for i, (x, y) in enumerate(zip(a, b)):
                issue_range = self._issue_ranges[i]
                if issue_range and isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    distance += abs(x - y) / issue_range
                else:
                    distance += float(x != y)
            return distance / len(a)
        return sum(x != y for x, y in zip(a, b)) / len(a)

    def _opponent_type(self) -> str:
        if len(self._received_utils) < 4:
            return "unknown"

        unique_ratio = len({_offer_key(offer) for offer in self._received_offers}) / len(
            self._received_offers
        )
        first = sum(self._received_utils[:2]) / 2
        last = sum(self._received_utils[-2:]) / 2
        movement = last - first
        recent = self._received_utils[-5:]
        volatility = max(recent) - min(recent) if len(recent) >= 3 else 0.0

        if unique_ratio < self.hardliner_unique_ratio:
            return "hardliner"
        if movement > self.conceder_movement:
            return "conceder"
        if volatility > self.random_volatility:
            return "random"
        if abs(movement) < self.stubborn_movement:
            return "stubborn"
        return "adaptive"

    def _robust_minmax(self, values: list[float]) -> list[float]:
        if not values:
            return []
        ordered = sorted(values)
        n = len(ordered)
        lo = ordered[int(0.05 * (n - 1))]
        hi = ordered[int(0.95 * (n - 1))]
        if hi <= lo:
            lo, hi = min(values), max(values)
        if hi <= lo:
            return [0.5 for _ in values]
        return [_clamp((value - lo) / (hi - lo)) for value in values]

    def _partner_scored(self, candidates):
        if not candidates:
            return []

        models = getattr(self.negotiator, "_models", None) or []
        prior_scores = [u for u, _ in candidates]
        if len(self._received_offers) < 3 or not models:
            return [
                (u, outcome, prior_score)
                for (u, outcome), prior_score in zip(candidates, prior_scores)
            ]

        model = models[0]
        raw_scores = [_utility(model, outcome, 0.0) for _, outcome in candidates]
        if self._is_four_integer_domain() or self._is_categorical_eight_domain():
            low, high = min(raw_scores), max(raw_scores)
            if high <= low:
                model_scores = prior_scores
            else:
                scale = high - low
                model_scores = [(score - low) / scale for score in raw_scores]
        else:
            model_scores = self._robust_minmax(raw_scores)

        confidence = min(self.partner_confidence_cap, len(self._received_offers) / 12.0)
        return [
            (
                u,
                outcome,
                (1.0 - confidence) * prior_score + confidence * model_score,
            )
            for (u, outcome), prior_score, model_score in zip(
                candidates, prior_scores, model_scores
            )
        ]

    def _best_received_offer(self):
        if not self._received_offers or not self._received_utils:
            return None, 0.0
        utility, offer = max(
            zip(self._received_utils, self._received_offers),
            key=lambda item: item[0],
        )
        return offer, utility

    def _near_received_improvement(self, reference, reference_utility: float):
        if reference is None:
            return None
        floor = max(0.02, reference_utility + self.improvement_margin)
        candidates = [
            (u, outcome)
            for u, outcome in zip(self._norm_utils, self._outcomes)
            if u >= floor and self._distance(outcome, reference) <= 0.50
        ]
        candidates = self._avoid_recent_offers(candidates, window=4)
        if not candidates:
            return None
        models = getattr(self.negotiator, "_models", None) or []

        def partner_score(outcome) -> float:
            return _utility(models[0], outcome, 0.0) if models else 0.0

        if reference_utility <= self.early_low_replay_threshold:
            return max(
                candidates[: min(500, len(candidates))],
                key=lambda item: (
                    partner_score(item[1]),
                    -self._distance(item[1], reference),
                    item[0],
                ),
            )[1]

        if (
            isinstance(reference, tuple)
            and len(reference) <= 2
            and self._issue_ranges
            and len(self._issue_ranges) == len(reference)
        ):
            return min(
                candidates[: min(500, len(candidates))],
                key=lambda item: (
                    self._distance(item[1], reference),
                    -partner_score(item[1]),
                    -item[0],
                ),
            )[1]
        if (
            isinstance(reference, tuple)
            and self._issue_ranges
            and len(self._issue_ranges) == len(reference)
        ):
            return min(
                candidates[: min(500, len(candidates))],
                key=lambda item: (
                    self._distance(item[1], reference),
                    -partner_score(item[1]),
                    -item[0],
                ),
            )[1]
        if (
            isinstance(reference, tuple)
            and not self._issue_ranges
        ):
            return min(
                candidates[: min(500, len(candidates))],
                key=lambda item: (
                    self._distance(item[1], reference),
                    -partner_score(item[1]),
                    -item[0],
                ),
            )[1]
        return min(
            candidates[: min(500, len(candidates))],
            key=lambda item: (
                -item[0],
                -partner_score(item[1]),
                self._distance(item[1], reference),
            ),
        )[1]

    def _joint_score(self, item, relative_time: float) -> float:
        own_utility, _, partner_utility = item
        if relative_time <= self.partner_influence_start:
            return own_utility

        remaining_time = max(0.01, 1.0 - self.partner_influence_start)
        partner_power = min(
            self.partner_influence_max,
            (relative_time - self.partner_influence_start)
            / remaining_time
            * self.partner_influence_max,
        )
        return own_utility * (partner_utility**partner_power)

    def _partner_floor_band(self, scored, floor: float):
        if not scored:
            return scored
        partner_floor = max(floor, 0.85 * max(item[2] for item in scored))
        filtered = [item for item in scored if item[2] >= partner_floor]
        return filtered or scored

    def _late_partner_band(self, scored, relative_time: float):
        if not scored or relative_time < self.partner_filter_time:
            return scored
        partner_floor = self.partner_filter_ratio * max(item[2] for item in scored)
        filtered = [item for item in scored if item[2] >= partner_floor]
        return filtered or scored

    def _top_own_band(self, candidates, slack: float, floor: float = 0.0):
        if not candidates:
            return candidates
        best_own_utility = max(u for u, _ in candidates)
        own_floor = max(floor, best_own_utility - slack)
        return [(u, outcome) for u, outcome in candidates if u >= own_floor] or candidates

    def _avoid_recent_offers(self, candidates, window: int | None = None):
        if not candidates or not self._sent_offers:
            return candidates

        if window is None:
            window = self.recent_offer_window
            if (
                self._outcomes
                and isinstance(self._outcomes[0], tuple)
                and len(self._outcomes[0]) <= 2
                and all(isinstance(v, (int, float)) for v in self._outcomes[0])
                and any(isinstance(v, float) for v in self._outcomes[0])
                and self._utility_scale is not None
                and self._utility_scale <= 0.95
            ):
                window = max(window, 3)
        recent_keys = {
            _offer_key(offer) for offer in self._sent_offers[-window:]
        }
        fresh = [
            (u, outcome)
            for u, outcome in candidates
            if _offer_key(outcome) not in recent_keys
        ]
        if fresh:
            return fresh

        last_key = _offer_key(self._sent_offers[-1])
        non_repeated = [
            (u, outcome) for u, outcome in candidates if _offer_key(outcome) != last_key
        ]
        return non_repeated or candidates

    def _remember_sent_offer(self, offer):
        offer = self._avoid_immediate_repeat(offer)
        if offer is not None:
            self._sent_offers.append(offer)
        return offer

    def _avoid_immediate_repeat(self, offer):
        if offer is None or not self._sent_offers:
            return offer
        if _offer_key(offer) != _offer_key(self._sent_offers[-1]):
            return offer
        if not self._outcomes or not self._norm_utils:
            return offer

        intended_utility = self.reserved_margin
        offer_key = _offer_key(offer)
        for utility, outcome in zip(self._norm_utils, self._outcomes):
            if _offer_key(outcome) == offer_key:
                intended_utility = utility
                break

        floor = max(self.reserved_margin, intended_utility - 0.08)
        alternatives = [
            (u, outcome)
            for u, outcome in zip(self._norm_utils, self._outcomes)
            if _offer_key(outcome) != offer_key and u >= floor
        ]
        if not alternatives:
            alternatives = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if _offer_key(outcome) != offer_key
            ]
        if not alternatives:
            return offer

        return min(
            alternatives[: min(750, len(alternatives))],
            key=lambda item: (self._distance(item[1], offer), -item[0]),
        )[1]

    def __call__(self, state, dest: str | None = None):
        self._ensure_ready()
        if not self._outcomes:
            return None

        opponent_type = self._opponent_type()
        target = self._target(state.relative_time)
        if opponent_type == "conceder" and state.relative_time < 0.85:
            target += self.conceder_target_bonus
        elif (
            opponent_type in ("hardliner", "stubborn")
            and state.relative_time > self.hardliner_trigger_time
        ):
            target -= self.hardliner_target_discount
        elif opponent_type == "random":
            target += self.random_target_bonus

        rescue_slack = self._rescue_slack(state.relative_time)
        candidate_floor = max(self.reserved_margin, target - rescue_slack)

        candidates = [
            (u, outcome)
            for u, outcome in zip(self._norm_utils, self._outcomes)
            if u >= candidate_floor
        ]
        if not candidates:
            candidates = [(self._norm_utils[0], self._outcomes[0])]
        candidates = self._avoid_recent_offers(candidates)

        reference = self._received_offers[-1] if self._received_offers else state.current_offer
        if reference is None:
            return self._remember_sent_offer(candidates[0][1])

        best_received_offer, best_received_utility = self._best_received_offer()
        if (
            best_received_offer is not None
            and state.relative_time >= self.final_replay_time
            and best_received_utility >= self.final_replay_threshold
        ):
            return self._remember_sent_offer(best_received_offer)

        if (
            best_received_offer is not None
            and state.relative_time >= self.replay_time
            and best_received_utility >= self.replay_threshold
        ):
            return self._remember_sent_offer(best_received_offer)

        if (
            best_received_offer is not None
            and (
                state.relative_time >= self.low_replay_improvement_time
                or (
                    state.relative_time >= self.early_low_replay_time
                    and best_received_utility <= self.early_low_replay_threshold
                )
            )
            and best_received_utility < self.final_replay_threshold
        ):
            improved_offer = self._near_received_improvement(
                best_received_offer, best_received_utility
            )
            if improved_offer is not None:
                return self._remember_sent_offer(improved_offer)

        if (
            opponent_type in ("hardliner", "stubborn")
            and state.relative_time > self.hardliner_special_time
        ):
            floor = (
                self.hardliner_final_floor
                if state.relative_time > self.hardliner_final_time
                else max(self.hardliner_late_floor, target - self.hardliner_target_slack)
            )
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= floor
            ] or candidates
            candidates = self._avoid_recent_offers(candidates)
            scored = self._partner_scored(candidates[: min(1500, len(candidates))])
            return self._remember_sent_offer(min(
                scored,
                key=lambda item: (
                    self._distance(item[1], reference),
                    -item[2],
                    -item[0],
                ),
            )[1])

        if state.relative_time > 0.98:
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= self.final_min_floor
            ] or candidates
            if reference is not None:
                alternatives = [
                    (u, outcome)
                    for u, outcome in candidates
                    if outcome != reference or u >= self.final_reference_floor
                ]
                candidates = alternatives or candidates
            candidates = self._top_own_band(
                candidates, slack=self.final_offer_slack, floor=self.final_reference_floor
            )
            candidates = self._avoid_recent_offers(
                candidates, window=self.final_recent_offer_window
            )
            scored = self._partner_scored(candidates[: min(1500, len(candidates))])
            scored = self._partner_floor_band(scored, self.final_partner_floor)
            return self._remember_sent_offer(min(
                scored,
                key=lambda item: (
                    -self._joint_score(item, state.relative_time),
                    self._distance(item[1], reference),
                    -item[0],
                ),
            )[1])

        if state.relative_time > 0.96:
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= self.prefinal_floor
            ] or candidates
            candidates = self._top_own_band(
                candidates, slack=self.prefinal_slack, floor=self.prefinal_floor
            )
            candidates = self._avoid_recent_offers(candidates)
            scored = self._partner_scored(candidates[: min(750, len(candidates))])
            scored = self._late_partner_band(scored, state.relative_time)
            return self._remember_sent_offer(min(
                scored,
                key=lambda item: (
                    -self._joint_score(item, state.relative_time),
                    self._distance(item[1], reference),
                    abs(item[0] - self.compromise_target),
                ),
            )[1])

        empathy = 0.20 + 0.80 * state.relative_time
        candidates = self._top_own_band(
            candidates,
            slack=self.own_band_base_slack
            + self.own_band_time_slack * state.relative_time
            + rescue_slack,
            floor=candidate_floor,
        )
        candidates = self._avoid_recent_offers(candidates)
        scored = self._partner_scored(candidates[: min(750, len(candidates))])
        scored = self._late_partner_band(scored, state.relative_time)
        return self._remember_sent_offer(min(
            scored,
            key=lambda item: (
                -self._joint_score(item, state.relative_time),
                empathy * self._distance(item[1], reference)
                - 0.10 * item[0]
            ),
        )[1])

    def on_preferences_changed(self, changes):
        self._reset()
        return super().on_preferences_changed(changes)

    def on_partner_proposal(self, state, partner_id: str, offer) -> None:
        self._ensure_ready()
        ufun = self.negotiator.ufun
        assert ufun is not None
        reserved_value = _reserved_value(ufun)
        best = ufun.best()
        best_utility = _utility(ufun, best, reserved_value + 1.0)
        scale = max(best_utility - reserved_value, 1.0)
        self._received_offers.append(offer)
        normalized = (_utility(ufun, offer, reserved_value - 1.0) - reserved_value) / scale
        self._received_utils.append(_clamp(normalized))
        return super().on_partner_proposal(state, partner_id, offer)


class HannariHamaguriHAN(SAOMetaNegotiator):
    """Hybrid HAN negotiator: deterministic strategy with fast human-like text.

    The base negotiator uses a BOA architecture:
    - AdaptiveOfferingPolicy: ambitious early offers, behavior-aware concessions.
    - AdaptiveAcceptancePolicy: time-based acceptance above the reserved value.
    - GHardHeadedFrequencyModel: lightweight opponent modeling.

    Natural-language messages are generated from templates to avoid LLM latency
    during tournament runs. LLM-related constructor arguments are accepted for
    compatibility with the original template but are not used.
    """

    def __init__(
        self,
        base_negotiator=None,
        provider: str = "ollama",
        model: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.4,
        max_tokens: int = 128,
        use_structured_output: bool = True,
        timeout: float = 30.0,
        num_retries: int = 1,
        **kwargs,
    ):
        if base_negotiator is None:
            offering = AdaptiveOfferingPolicy(
                partner_influence_start=0.58,
                partner_influence_max=0.76,
                partner_confidence_cap=0.55,
                hardliner_target_discount=0.05,
                hardliner_final_floor=0.08,
                hardliner_late_floor=0.27,
                final_partner_floor=0.82,
            )
            base_negotiator = BOANegotiator(
                acceptance=AdaptiveAcceptancePolicy(),
                offering=offering,
                model=RobustFrequencyOpponentModel(),
            )

        kwargs.pop("system_prompt", None)
        kwargs.pop("preferences_prompt", None)
        kwargs.pop("preferences_changed_prompt", None)
        kwargs.pop("negotiation_start_prompt", None)
        kwargs.pop("round_prompt", None)
        kwargs.pop("llm_kwargs", None)
        _ = (provider, model, temperature, max_tokens, use_structured_output, timeout)
        _ = num_retries
        kwargs.setdefault("name", DISPLAY_NAME)

        super().__init__(
            negotiators=[base_negotiator],
            negotiator_names=["base"],
            share_ufun=True,
            share_nmi=True,
            **kwargs,
        )
        self._observed_offer_keys = set()

    @property
    def base_negotiator(self):
        return self._negotiators[0]

    @property
    def opponent_ufun(self):
        models = getattr(self.base_negotiator, "_models", None) or []
        if not models:
            return None
        model = models[0]
        if self.ufun is None:
            return model
        smith = getattr(model, "_smith", model)
        ufun = self.ufun
        reserved_value = _reserved_value(ufun)
        if reserved_value <= 0.0:
            class ZeroReservePublicBlend:
                outcome_space = getattr(model, "outcome_space", None)

                def __call__(self, outcome):
                    robust_score = max(0.0, float(model(outcome)))
                    smith_score = max(0.0, float(smith(outcome)))
                    return 0.30 * robust_score + 0.70 * smith_score

            return ZeroReservePublicBlend()
        best = ufun.best()
        best_utility = _utility(ufun, best, reserved_value + 1.0)
        utility_scale = max(best_utility - reserved_value, 1e-9)

        class PublicBlend:
            outcome_space = getattr(model, "outcome_space", None)

            def __call__(self, outcome):
                robust_score = max(0.0, float(model(outcome)))
                smith_score = max(0.0, float(smith(outcome)))
                base_score = 0.15 * robust_score + 0.85 * smith_score
                if reserved_value >= 0.07:
                    base_score = 0.09 * robust_score + 0.91 * smith_score
                four_integer_outcome = (
                    isinstance(outcome, tuple)
                    and len(outcome) == 4
                    and all(isinstance(v, int) for v in outcome)
                )
                categorical_eight_outcome = (
                    isinstance(outcome, tuple)
                    and len(outcome) == 8
                    and all(isinstance(v, str) for v in outcome)
                )
                if four_integer_outcome:
                    base_score = 0.95 * robust_score + 0.05 * smith_score
                if categorical_eight_outcome:
                    base_score = 0.03 * robust_score + 0.97 * smith_score
                own_score = (_utility(ufun, outcome, reserved_value) - reserved_value) / utility_scale
                inverse_own = 1.0 - _clamp(own_score)
                if four_integer_outcome:
                    return 0.320 * base_score + 0.680 * inverse_own
                if categorical_eight_outcome:
                    return base_score
                if reserved_value >= 0.065:
                    return 0.955 * base_score + 0.045 * inverse_own
                return 0.955 * base_score + 0.045 * inverse_own

        return PublicBlend()

    def on_preferences_changed(self, changes):
        self._observed_offer_keys = set()
        return super().on_preferences_changed(changes)

    def propose(
        self, state, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        self._observe_current_offer(state, dest)
        base_proposal = self.base_negotiator.propose(state, dest=dest)
        if base_proposal is None:
            return None

        if isinstance(base_proposal, ExtendedOutcome):
            outcome = base_proposal.outcome
            base_data = base_proposal.data or {}
        else:
            outcome = base_proposal
            base_data = {}

        if outcome is None:
            return None

        return ExtendedOutcome(
            outcome=outcome,
            data={**base_data, "text": self._generate_text(state, "propose", outcome)},
        )

    def respond(self, state, source: str | None = None):
        self._observe_current_offer(state, source)
        base_response = self.base_negotiator.respond(state, source=source)

        if isinstance(base_response, ExtendedResponseType):
            response_type = base_response.response
            base_data = base_response.data or {}
        else:
            response_type = base_response
            base_data = {}

        if response_type == ResponseType.ACCEPT_OFFER:
            text = self._generate_text(state, "accept", state.current_offer)
            return ExtendedResponseType(response=response_type, data={**base_data, "text": text})

        if response_type == ResponseType.END_NEGOTIATION:
            text = self._generate_text(state, "end", state.current_offer)
            return ExtendedResponseType(response=response_type, data={**base_data, "text": text})

        return base_response

    def _observe_current_offer(self, state, source: str | None = None) -> None:
        offer = state.current_offer
        if offer is None:
            return
        key = (state.step, _offer_key(offer))
        if key in self._observed_offer_keys:
            return
        self._observed_offer_keys.add(key)

        partner_id = source or "opponent"
        for model in getattr(self.base_negotiator, "_models", None) or []:
            model.on_partner_proposal(state, partner_id, offer)

        offering = getattr(self.base_negotiator, "_offering", None)
        if offering is not None:
            offering.on_partner_proposal(state, partner_id, offer)

    def _generate_text(self, state, action: str, outcome=None) -> str:
        if action == "accept":
            return "This works for me. I accept, and I appreciate the effort to find a workable balance."
        if action == "end":
            return "It seems we cannot find a workable agreement this time, so I will stop here."
        if state.current_offer is None:
            return "Let me start with a clear proposal so we have a practical basis for discussion."
        if state.relative_time >= 0.90:
            return "We are close to the end, so I am moving to a practical compromise that I can still accept."
        if state.relative_time >= 0.60:
            return "I can move from my earlier position while keeping the agreement workable for me."
        return "I appreciate your proposal. Here is a counter-offer that keeps the discussion moving."


HannariHamaguri = HannariHamaguriHAN
MyAgent = HannariHamaguriHAN
