from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any

from negmas.common import Outcome
from negmas.gb.components.genius.models import GSmithFrequencyModel
from negmas.outcomes import ExtendedOutcome
from negmas.sao.common import ResponseType, SAOResponse, SAOState
from negmas.sao.negotiators.base import SAOCallNegotiator


@dataclass(frozen=True)
class RankedOutcome:
    outcome: Outcome
    utility: float


class _BaseHanOmegaNegotiator(SAOCallNegotiator):
    """A human-facing HAN negotiator with explicit concession control.

    The strategy combines four ideas:
    1. Pre-rank all discrete outcomes by our utility.
    2. Estimate opponent preferences from repeated values in their offers.
    3. Select counteroffers on a time-based aspiration frontier.
    4. Attach concise, context-aware text aimed at human negotiators.
    """

    def __init__(
        self,
        concession_exponent: float = 4.9,
        acceptance_slack: float = 0.05,
        late_acceptance_slack: float = 0.10,
        candidate_window: int = 72,
        min_progress: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.concession_exponent = concession_exponent
        self.acceptance_slack = acceptance_slack
        self.late_acceptance_slack = late_acceptance_slack
        self.candidate_window = candidate_window
        self.min_progress = min_progress

        self.private_info["opponent_ufun"] = GSmithFrequencyModel(negotiator=self)
        self._ranked_outcomes: list[RankedOutcome] = []
        self._rational_outcomes: list[RankedOutcome] = []
        self._issue_values: list[list[Any]] = []
        self._issue_best_value: list[Any] = []
        self._observed_offers: list[Outcome] = []
        self._observed_texts: list[str] = []
        self._value_counts: dict[int, Counter[Any]] = defaultdict(Counter)
        self._issue_change_counts: Counter[int] = Counter()
        self._last_received_offer: Outcome | None = None
        self._last_proposed_offer: Outcome | None = None
        self._recent_proposals: list[Outcome] = []
        self._proposal_history: list[Outcome] = []
        self._best_received_offer: Outcome | None = None
        self._best_received_utility: float = float("-inf")
        self._best_received_step = 0
        self._repeat_offer_count = 0
        self._domain_kind = "generic"
        self._initialized = False

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        assert self.ufun is not None
        self._ensure_initialized()
        self._observe_partner_move(state)

        offer = state.current_offer
        if offer is not None and self._should_accept(state, offer):
            return SAOResponse(
                ResponseType.ACCEPT_OFFER,
                outcome=offer,
                data={"text": self._acceptance_text(state, offer)},
            )

        counter = self._select_offer(state)
        if counter is None:
            return SAOResponse(
                ResponseType.END_NEGOTIATION,
                data={"text": "I cannot reach an agreement above my minimum acceptable outcome."},
            )

        return SAOResponse(
            ResponseType.REJECT_OFFER,
            outcome=counter,
            data={"text": self._counter_text(state, counter)},
        )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        assert self.ufun is not None
        outcome_space = self.ufun.outcome_space
        assert outcome_space is not None

        ranked: list[RankedOutcome] = []
        reserved = float(self.ufun.reserved_value)
        for outcome in outcome_space.enumerate():
            utility = float(self.ufun(outcome))
            ranked.append(RankedOutcome(outcome=outcome, utility=utility))

        ranked.sort(key=lambda item: item.utility, reverse=True)
        self._ranked_outcomes = ranked
        self._rational_outcomes = [item for item in ranked if item.utility >= reserved]
        if ranked:
            n_issues = len(ranked[0].outcome)
            self._issue_values = [
                sorted({item.outcome[i] for item in ranked}, key=str) for i in range(n_issues)
            ]
            self._issue_best_value = [self._ranked_outcomes[0].outcome[i] for i in range(n_issues)]
        self._domain_kind = self._infer_domain_kind()
        self._initialized = True

    def _infer_domain_kind(self) -> str:
        if not self.nmi or not getattr(self.nmi.outcome_space, "issues", None):
            return "generic"
        issue_names = [str(_.name).lower() for _ in self.nmi.outcome_space.issues]
        if len(issue_names) == 2 and {"price", "quantity"}.issubset(set(issue_names)):
            return "trade"
        values = {str(v).lower() for vals in self._issue_values for v in vals}
        if values.issubset({"alice", "bob"}):
            return "binary-allocation"
        if all(all(isinstance(v, int) for v in vals) for vals in self._issue_values):
            return "count-allocation"
        return "generic"

    def _observe_partner_move(self, state: SAOState) -> None:
        offer = state.current_offer
        if offer is None:
            return
        if offer == self._last_received_offer:
            self._repeat_offer_count += 1
            return
        self._repeat_offer_count = 0

        partner_id = state.current_proposer or "opponent"
        opponent_ufun = self.opponent_ufun
        assert opponent_ufun is not None
        opponent_ufun.on_partner_proposal(state, partner_id, offer)
        self._observed_offers.append(offer)

        if len(self._observed_offers) >= 2:
            prev = self._observed_offers[-2]
            for i, (old, new) in enumerate(zip(prev, offer)):
                if old != new:
                    self._issue_change_counts[i] += 1

        for i, value in enumerate(offer):
            self._value_counts[i][value] += 1

        utility = float(self.ufun(offer))
        if utility > self._best_received_utility:
            self._best_received_utility = utility
            self._best_received_offer = offer
            self._best_received_step = len(self._observed_offers)

        received_text = self._extract_received_text(state)
        if received_text:
            self._observed_texts.append(received_text)

        self._last_received_offer = offer

    def _should_accept(self, state: SAOState, offer: Outcome) -> bool:
        assert self.ufun is not None
        offer_utility = float(self.ufun(offer))
        reserved = float(self.ufun.reserved_value)
        if offer_utility < reserved:
            return False

        threshold = self._aspiration_threshold(state)
        if offer_utility >= threshold:
            return True

        next_offer = self._select_offer(state, update_state=False)
        next_utility = float(self.ufun(next_offer)) if next_offer is not None else reserved
        best = self._ranked_outcomes[0].utility if self._ranked_outcomes else reserved
        utility_span = max(0.0, best - reserved)
        slack = self.acceptance_slack + self._relative_time(state) * (
            self.late_acceptance_slack - self.acceptance_slack
        )
        slack *= utility_span
        if offer_utility + slack >= next_utility:
            return True

        if next_offer is not None:
            next_norm = self._normalize_utility(next_utility)
            offer_norm = self._normalize_utility(offer_utility)
            offer_closeness = 1.0 - self._offer_distance(offer, next_offer)
            if (
                self._relative_time(state) >= 0.72
                and offer_norm >= max(0.2, next_norm - (0.04 + 0.1 * self._relative_time(state)))
                and offer_closeness >= 0.5
            ):
                return True
            current_opp = self._estimated_opponent_utility(offer)
            next_opp = self._estimated_opponent_utility(next_offer)
            if (
                self._relative_time(state) >= 0.8
                and offer_norm >= max(0.25, next_norm - 0.08)
                and current_opp >= next_opp + 0.12
            ):
                return True

        offer_norm = self._normalize_utility(offer_utility)
        if (
            self._repeat_offer_count >= 2
            and self._relative_time(state) >= 0.85
            and offer_norm >= max(0.72, self._normalize_utility(self._best_fallback_utility()) - 0.02)
        ):
            return True

        if (
            self._best_received_offer is not None
            and self._relative_time(state) >= 0.92
            and offer_utility >= self._best_received_utility - 0.02
        ):
            return True

        if self._relative_time(state) >= 0.97 and offer_utility >= self._best_fallback_utility():
            return True

        return False

    def _select_offer(self, state: SAOState, update_state: bool = True) -> Outcome | None:
        frontier = self._candidate_frontier(state)
        if not frontier:
            frontier = self._rational_outcomes[: self.candidate_window] or self._ranked_outcomes[:1]
        if not frontier:
            return None

        t = self._relative_time(state)
        best = max(frontier, key=lambda item: self._offer_score(item, t))
        forced_compromise = self._late_satisficing_offer(state)
        if (
            forced_compromise is not None
            and (
                self._acceptance_likelihood(best.outcome) < 0.78
                or self._repeat_offer_count >= 1
                or self._hard_opponent_signal() > 0.7
            )
        ):
            best = forced_compromise
        if best.outcome == self._last_proposed_offer and len(frontier) > 1:
            alternatives = sorted(frontier, key=lambda item: self._offer_score(item, t), reverse=True)
            for candidate in alternatives:
                if candidate.outcome != self._last_proposed_offer:
                    best = candidate
                    break

        if self._partner_cycles_offers(t) and best.outcome in self._proposal_history:
            proposed = set(self._proposal_history)
            unseen = [item for item in frontier if item.outcome not in proposed]
            if unseen:
                best = max(unseen, key=lambda item: self._offer_score(item, t))

        if update_state:
            self._last_proposed_offer = best.outcome
            self._recent_proposals.append(best.outcome)
            self._recent_proposals = self._recent_proposals[-4:]
            self._proposal_history.append(best.outcome)
        return best.outcome

    def _partner_cycles_offers(self, relative_time: float) -> bool:
        if relative_time < 0.20 or len(self._observed_offers) < 6:
            return False
        unique_ratio = len(set(self._observed_offers)) / len(self._observed_offers)
        return unique_ratio <= 0.75

    def _late_satisficing_offer(self, state: SAOState) -> RankedOutcome | None:
        assert self.ufun is not None
        t = self._relative_time(state)
        if t < 0.72 or len(self._observed_offers) < 2:
            return None
        hardliner = self._hard_opponent_signal()
        reserved = float(self.ufun.reserved_value)
        best = self._ranked_outcomes[0].utility if self._ranked_outcomes else reserved
        rescue = None
        if hardliner >= 0.45:
            floor = reserved + (best - reserved) * max(0.18, 0.42 - 0.18 * max(0.0, t - 0.72))
            target = 0.86 - 0.10 * max(0.0, t - 0.72)
            candidates = [
                item
                for item in self._rational_outcomes
                if item.utility >= floor and self._acceptance_likelihood(item.outcome) >= target
            ]
            if candidates:
                candidates.sort(
                    key=lambda item: (
                        -item.utility,
                        -self._acceptance_likelihood(item.outcome),
                        -self._estimated_opponent_utility(item.outcome),
                    )
                )
                rescue = candidates[0]

        if t < 0.75 or self._best_received_offer is None:
            return rescue

        relaxed_floor = max(reserved, self._aspiration_threshold(state) - 0.05)
        repaired = self._repair_from_offer(self._best_received_offer, relaxed_floor)
        if repaired is None:
            return rescue

        repaired_ranked = RankedOutcome(outcome=repaired, utility=float(self.ufun(repaired)))
        if rescue is None:
            return repaired_ranked
        if (
            self._acceptance_likelihood(repaired_ranked.outcome)
            > self._acceptance_likelihood(rescue.outcome) + 0.05
        ):
            return repaired_ranked
        return rescue

    def _text_anchor_probe(
        self,
        state: SAOState,
        baseline: RankedOutcome | None,
    ) -> RankedOutcome | None:
        """Explore a different compromise axis after repeated textual rejections."""
        assert self.ufun is not None
        t = self._relative_time(state)
        reserved = float(self.ufun.reserved_value)
        anchor = self._last_received_offer
        if (
            t < 0.75
            or anchor is None
            or self._repeat_offer_count < 2
            or not self._observed_texts
            or self._best_received_utility >= reserved
        ):
            return baseline

        target = max(0.62, 0.69 - 0.55 * (t - 0.75))
        candidates: list[tuple[RankedOutcome, float]] = []
        for item in self._rational_outcomes:
            opponent_estimate = self._ordinal_opponent_utility(item.outcome)
            if opponent_estimate >= target:
                candidates.append((item, opponent_estimate))
        if not candidates:
            return baseline

        recent_signatures = {
            frozenset(
                index
                for index, (proposed, anchored) in enumerate(zip(proposal, anchor))
                if proposed != anchored
            )
            for proposal in self._recent_proposals
        }

        def probe_key(pair: tuple[RankedOutcome, float]) -> tuple[bool, float, float]:
            signature = frozenset(
                index
                for index, (proposed, anchored) in enumerate(
                    zip(pair[0].outcome, anchor)
                )
                if proposed != anchored
            )
            return signature not in recent_signatures, pair[0].utility, pair[1]

        candidate, candidate_opponent = max(candidates, key=probe_key)
        if baseline is None:
            return candidate
        baseline_opponent = self._ordinal_opponent_utility(baseline.outcome)
        if candidate_opponent >= baseline_opponent + 0.08:
            return candidate
        return baseline

    def _candidate_frontier(self, state: SAOState) -> list[RankedOutcome]:
        threshold = self._aspiration_threshold(state)
        candidates = [item for item in self._rational_outcomes if item.utility >= threshold]
        if state.current_offer is not None:
            candidates = self._merge_candidates(
                candidates,
                [
                    *self._nearby_candidates(state.current_offer, floor=max(float(self.ufun.reserved_value), threshold - 0.08)),  # type: ignore[arg-type]
                    *self._repair_candidates(state.current_offer, threshold),
                    *self._domain_candidates(state.current_offer, threshold),
                    *self._incremental_candidates(state.current_offer, threshold),
                ],
            )
        if candidates:
            frontier = candidates[: self.candidate_window]
        else:
            if not self._rational_outcomes:
                return []

            floor = max(float(self.ufun.reserved_value), threshold - 0.08)  # type: ignore[arg-type]
            relaxed = [item for item in self._rational_outcomes if item.utility >= floor]
            if state.current_offer is not None:
                relaxed = self._merge_candidates(
                    relaxed,
                    [
                        *self._nearby_candidates(state.current_offer, floor=floor),
                        *self._repair_candidates(state.current_offer, threshold),
                        *self._domain_candidates(state.current_offer, threshold),
                        *self._incremental_candidates(state.current_offer, threshold),
                    ],
                )
            frontier = (
                relaxed[: self.candidate_window]
                if relaxed
                else self._rational_outcomes[: self.candidate_window]
            )

        accept_floor = max(float(self.ufun.reserved_value), threshold - 0.08)
        accept_pool = [item for item in self._rational_outcomes if item.utility >= accept_floor]
        accept_pool.sort(
            key=lambda item: (
                self._acceptance_likelihood(item.outcome),
                item.utility,
            ),
            reverse=True,
        )
        frontier = self._merge_candidates(
            frontier,
            accept_pool[:8],
        )
        return frontier[: self.candidate_window]

    def _incremental_candidates(self, offer: Outcome, threshold: float) -> list[RankedOutcome]:
        assert self.ufun is not None
        variants: list[RankedOutcome] = []
        for issue_index, best_value in enumerate(self._issue_best_value):
            if issue_index >= len(offer) or offer[issue_index] == best_value:
                continue
            trial = list(offer)
            trial[issue_index] = best_value
            outcome = tuple(trial)
            utility = float(self.ufun(outcome))
            if utility < max(float(self.ufun.reserved_value), threshold - 0.08):
                continue
            variants.append(RankedOutcome(outcome=outcome, utility=utility))
        variants.sort(
            key=lambda item: (
                -self._acceptance_likelihood(item.outcome),
                -item.utility,
            )
        )
        return variants[: max(4, self.candidate_window // 16)]

    def _domain_candidates(self, offer: Outcome, threshold: float) -> list[RankedOutcome]:
        if self._domain_kind == "trade":
            return self._trade_candidates(offer, threshold)
        if self._domain_kind in {"binary-allocation", "count-allocation"}:
            return self._allocation_candidates(offer, threshold)
        return []

    def _trade_candidates(self, offer: Outcome, threshold: float) -> list[RankedOutcome]:
        if not self.nmi or not getattr(self.nmi.outcome_space, "issues", None):
            return []
        issues = [str(_.name).lower() for _ in self.nmi.outcome_space.issues]
        quantity_idx = issues.index("quantity")
        price_idx = issues.index("price")
        quantity_target = offer[quantity_idx]
        candidates = [item for item in self._rational_outcomes if item.utility >= max(float(self.ufun.reserved_value), threshold - 0.05)]
        candidates.sort(
            key=lambda item: (
                abs(float(item.outcome[quantity_idx]) - float(quantity_target)),
                self._offer_distance(item.outcome, offer),
                -item.utility,
            )
        )
        short = candidates[: max(10, self.candidate_window // 4)]
        short.sort(
            key=lambda item: (
                -self._estimated_opponent_utility(item.outcome),
                -item.utility,
                abs(float(item.outcome[price_idx]) - float(offer[price_idx])),
            )
        )
        return short[: max(6, self.candidate_window // 6)]


    def _allocation_candidates(self, offer: Outcome, threshold: float) -> list[RankedOutcome]:
        outcome = self._repair_from_offer(offer, threshold)
        if outcome is None:
            return []
        variants: list[Outcome] = [outcome]
        if outcome is not None:
            for issue_index, best_value in enumerate(self._issue_best_value):
                if issue_index >= len(outcome) or outcome[issue_index] == best_value:
                    continue
                trial = list(outcome)
                trial[issue_index] = best_value
                trial_outcome = tuple(trial)
                if float(self.ufun(trial_outcome)) >= max(float(self.ufun.reserved_value), threshold - 0.03):
                    variants.append(trial_outcome)
                if len(variants) >= 4:
                    break
        unique = []
        seen = set()
        for variant in variants:
            if variant in seen:
                continue
            seen.add(variant)
            unique.append(RankedOutcome(outcome=variant, utility=float(self.ufun(variant))))
        unique.sort(key=lambda item: (-self._estimated_opponent_utility(item.outcome), -item.utility))
        return unique

    def _merge_candidates(
        self, primary: list[RankedOutcome], extras: list[RankedOutcome]
    ) -> list[RankedOutcome]:
        seen: set[Outcome] = set()
        merged: list[RankedOutcome] = []
        for item in [*primary, *extras]:
            if item.outcome in seen:
                continue
            seen.add(item.outcome)
            merged.append(item)
        merged.sort(key=lambda item: item.utility, reverse=True)
        return merged

    def _nearby_candidates(self, offer: Outcome, floor: float) -> list[RankedOutcome]:
        nearby = [item for item in self._rational_outcomes if item.utility >= floor]
        nearby.sort(
            key=lambda item: (
                self._offer_distance(item.outcome, offer),
                -item.utility,
            )
        )
        return nearby[: max(8, self.candidate_window // 3)]

    def _repair_candidates(self, offer: Outcome, threshold: float) -> list[RankedOutcome]:
        repaired = self._repair_from_offer(offer, threshold)
        if repaired is None:
            return []
        utility = float(self.ufun(repaired))
        return [RankedOutcome(outcome=repaired, utility=utility)]

    def _repair_from_offer(self, offer: Outcome, threshold: float) -> Outcome | None:
        assert self.ufun is not None
        if not self._issue_values:
            return None
        candidate = list(offer)
        current_utility = float(self.ufun(tuple(candidate)))
        if current_utility >= threshold:
            return tuple(candidate)

        for _ in range(len(candidate) * 2):
            best_move: tuple[float, int, Any, float] | None = None
            for issue_index, current_value in enumerate(candidate):
                for alt in self._issue_values[issue_index]:
                    if alt == current_value:
                        continue
                    trial = list(candidate)
                    trial[issue_index] = alt
                    trial_outcome = tuple(trial)
                    trial_utility = float(self.ufun(trial_outcome))
                    gain = trial_utility - current_utility
                    if gain <= 0:
                        continue
                    opponent_loss = self._value_loss(issue_index, current_value, alt)
                    move_score = gain / max(0.05, opponent_loss)
                    if best_move is None or move_score > best_move[0]:
                        best_move = (move_score, issue_index, alt, trial_utility)
            if best_move is None:
                return None
            _, issue_index, alt, current_utility = best_move
            candidate[issue_index] = alt
            if current_utility >= threshold:
                return tuple(candidate)
        return tuple(candidate) if current_utility >= threshold else None

    def _value_loss(self, issue_index: int, old_value: Any, new_value: Any) -> float:
        counts = self._value_counts[issue_index]
        if not counts:
            return 1.0
        old_score = counts.get(old_value, 0)
        new_score = counts.get(new_value, 0)
        if old_score == new_score:
            return 1.0
        return max(0.1, (old_score - new_score) / max(1, max(counts.values())) + 0.5)

    def _aspiration_threshold(self, state: SAOState) -> float:
        assert self.ufun is not None
        reserved = float(self.ufun.reserved_value)
        best = self._rational_outcomes[0].utility if self._rational_outcomes else self._ranked_outcomes[0].utility
        t = self._relative_time(state)

        empathy_signal = self._text_softening_signal()
        exponent = max(1.5, self.concession_exponent - empathy_signal)
        threshold = reserved + (best - reserved) * (1.0 - math.pow(t, exponent))
        threshold = max(reserved, threshold)
        if self._best_received_offer is not None:
            threshold = max(threshold, self._best_received_utility - self.min_progress)
        threshold -= self._stagnation_discount(best - reserved)

        if t > 0.9:
            threshold = min(threshold, reserved + 0.15 * (best - reserved))

        return threshold

    def _stagnation_discount(self, span: float) -> float:
        if len(self._observed_offers) < 4 or self._best_received_offer is None:
            return 0.0
        stale_rounds = len(self._observed_offers) - self._best_received_step
        if stale_rounds <= 3:
            return 0.0
        hardliner = self._hard_opponent_signal()
        return span * min(0.06, 0.008 * (stale_rounds - 1) * (0.5 + hardliner))

    def _offer_score(self, ranked: RankedOutcome, t: float) -> float:
        own = self._normalize_utility(ranked.utility)
        opp = self._estimated_opponent_utility(ranked.outcome)
        acceptability = self._acceptance_likelihood(ranked.outcome)
        efficiency = self._concession_efficiency(own, opp)
        fairness = 1.0 - abs(own - opp)
        closeness = 1.0
        if self._last_received_offer is not None:
            closeness = 1.0 - self._offer_distance(ranked.outcome, self._last_received_offer)
        novelty = 0.025
        if ranked.outcome == self._last_proposed_offer:
            novelty = -0.04
        elif ranked.outcome in self._recent_proposals:
            novelty = -0.015
        hardliner = self._hard_opponent_signal()
        late = 1.0 if t >= 0.65 else 0.0
        pressure = 0.12 + 0.45 * t + 0.25 * max(0.0, t - 0.65) * hardliner
        own_weight = 1.2 - 0.5 * t - 0.22 * max(0.0, t - 0.65) * hardliner - 0.08 * late
        accept_weight = 0.10 + 0.28 * t + 0.22 * hardliner + 0.08 * late + 0.12 * hardliner * late
        close_weight = 0.10 + 0.18 * t + 0.16 * max(0.0, t - 0.6) * hardliner + 0.12 * hardliner * late
        return (
            own_weight * own
            + pressure * opp
            + accept_weight * acceptability
            + (0.05 + 0.18 * t) * efficiency
            + 0.2 * fairness
            + close_weight * closeness
            + novelty
        )

    def _concession_efficiency(self, own: float, opp: float) -> float:
        if self._last_received_offer is None or self._best_received_offer is None:
            return opp
        baseline_opp = self._estimated_opponent_utility(self._best_received_offer)
        baseline_own = self._normalize_utility(self._best_received_utility)
        opp_gain = max(0.0, opp - baseline_opp)
        own_loss = max(0.0, baseline_own - own)
        if opp_gain <= 0:
            return 0.0
        return max(0.0, min(1.0, opp_gain / max(0.08, own_loss + 0.05)))

    def _acceptance_likelihood(self, outcome: Outcome) -> float:
        opp = self._estimated_opponent_utility(outcome)
        if not self._observed_offers:
            return opp
        anchor = self._observed_offers[-1]
        anchor_closeness = 1.0 - self._offer_distance(outcome, anchor)
        best_anchor_closeness = 0.0
        if self._best_received_offer is not None:
            best_anchor_closeness = 1.0 - self._offer_distance(outcome, self._best_received_offer)
        repeated_bonus = min(0.2, 0.06 * self._repeat_offer_count)
        return max(
            0.0,
            min(
                1.0,
                0.55 * opp + 0.25 * anchor_closeness + 0.2 * best_anchor_closeness + repeated_bonus,
            ),
        )

    def _hard_opponent_signal(self) -> float:
        if len(self._observed_offers) < 3:
            return 0.0
        issue_changes = sum(self._issue_change_counts.values())
        average_changes = issue_changes / max(1, len(self._observed_offers) - 1)
        repeated = min(1.0, self._repeat_offer_count / 3)
        low_variation = max(0.0, 1.0 - average_changes / max(1.0, len(self._observed_offers[-1]) * 0.75))
        return max(0.0, min(1.0, 0.7 * low_variation + 0.3 * repeated))

    def _offer_distance(self, first: Outcome, second: Outcome) -> float:
        if len(first) == 0:
            return 0.0
        return sum(1.0 for a, b in zip(first, second) if a != b) / len(first)

    def _normalize_utility(self, utility: float) -> float:
        assert self.ufun is not None
        reserved = float(self.ufun.reserved_value)
        best = self._ranked_outcomes[0].utility if self._ranked_outcomes else reserved
        if best <= reserved:
            return 1.0
        return max(0.0, min(1.0, (utility - reserved) / (best - reserved)))

    def _estimated_opponent_utility(self, outcome: Outcome) -> float:
        if not self._observed_offers:
            return 0.4

        n_issues = len(outcome)
        weights: list[float] = []
        scores: list[float] = []
        offer_count = len(self._observed_offers)

        for i in range(n_issues):
            counts = self._value_counts[i]
            if not counts:
                weights.append(1.0)
                scores.append(0.5)
                continue

            max_count = max(counts.values())
            stability = max_count / offer_count
            change_penalty = self._issue_change_counts[i] / max(1, offer_count - 1)
            total = sum(counts.values())
            entropy = 0.0
            for count in counts.values():
                p = count / total
                entropy -= p * math.log(max(p, 1e-12))
            max_entropy = math.log(max(1, len(counts)))
            concentration = 1.0 if max_entropy == 0 else 1.0 - entropy / max_entropy
            weight = 0.15 + 0.55 * stability + 0.45 * concentration - 0.45 * change_penalty
            weights.append(max(0.05, weight))
            scores.append(counts[outcome[i]] / max_count)

        weight_sum = sum(weights)
        return sum(w * s for w, s in zip(weights, scores)) / weight_sum

    def _ordinal_opponent_utility(self, outcome: Outcome) -> float:
        """Frequency estimate augmented with numeric proximity to modal values."""
        if not self._observed_offers:
            return 0.4

        weights: list[float] = []
        scores: list[float] = []
        offer_count = len(self._observed_offers)
        for issue_index, value in enumerate(outcome):
            counts = self._value_counts[issue_index]
            if not counts:
                weights.append(1.0)
                scores.append(0.5)
                continue

            max_count = max(counts.values())
            stability = max_count / offer_count
            change_penalty = self._issue_change_counts[issue_index] / max(1, offer_count - 1)
            total = sum(counts.values())
            entropy = 0.0
            for count in counts.values():
                p = count / total
                entropy -= p * math.log(max(p, 1e-12))
            max_entropy = math.log(max(1, len(counts)))
            concentration = 1.0 if max_entropy == 0 else 1.0 - entropy / max_entropy
            weight = 0.15 + 0.55 * stability + 0.45 * concentration - 0.45 * change_penalty
            weights.append(max(0.05, weight))

            exact_score = counts[value] / max_count
            ordinal_score = 0.0
            values = self._issue_values[issue_index]
            try:
                low = min(float(candidate) for candidate in values)
                high = max(float(candidate) for candidate in values)
                modal = max(counts, key=counts.get)
                value_span = high - low
                if value_span > 0.0:
                    ordinal_score = 1.0 - abs(float(value) - float(modal)) / value_span
            except (TypeError, ValueError):
                pass
            scores.append(max(exact_score, max(0.0, min(1.0, ordinal_score))))

        weight_sum = sum(weights)
        return sum(weight * score for weight, score in zip(weights, scores)) / weight_sum

    def _relative_time(self, state: SAOState) -> float:
        return max(0.0, min(1.0, float(getattr(state, "relative_time", 0.0) or 0.0)))

    def _best_fallback_utility(self) -> float:
        assert self.ufun is not None
        reserved = float(self.ufun.reserved_value)
        if self._best_received_offer is None:
            return reserved
        return max(reserved, self._best_received_utility - 0.02)

    def _extract_received_text(self, state: SAOState) -> str | None:
        if state.current_data and isinstance(state.current_data, dict):
            text = state.current_data.get("text")
            if text:
                return str(text).strip()
        for _, data in reversed(state.new_data):
            if data and isinstance(data, dict):
                text = data.get("text")
                if text:
                    return str(text).strip()
        return None

    def _text_softening_signal(self) -> float:
        if not self._observed_texts:
            return 0.0
        text = self._observed_texts[-1].lower()
        signal = 0.0
        if any(token in text for token in ("fair", "middle", "reasonable", "both")):
            signal += 0.8
        if any(token in text for token in ("need", "must", "important", "urgent")):
            signal += 0.4
        if any(token in text for token in ("final", "last offer", "deadline")):
            signal += 0.6
        return signal

    def _acceptance_text(self, state: SAOState, offer: Outcome) -> str:
        highlights = self._describe_offer(offer, limit=2)
        prefix = "I can agree to this."
        if self._extract_received_text(state):
            prefix = "Thank you for working through the details. I can agree to this."
        if highlights:
            return f"{prefix} The arrangement on {highlights} is acceptable to me."
        return prefix

    def _counter_text(self, state: SAOState, counter: Outcome) -> str:
        their_offer = state.current_offer
        received_text = self._extract_received_text(state)
        if their_offer is None:
            intro = "I would like to start from a strong but workable opening proposal."
            details = self._describe_offer(counter, limit=2)
            return f"{intro} I suggest {details}."

        changed = self._describe_differences(their_offer, counter)
        empathy = self._empathy_prefix(received_text)
        closing = self._closing_phrase(state, counter)
        if changed:
            return f"{empathy}I cannot accept the current terms. My counteroffer moves {changed}. {closing}"
        return f"{empathy}I cannot accept the current terms, so I am proposing a closer alternative. {closing}"

    def _empathy_prefix(self, received_text: str | None) -> str:
        if not received_text:
            return ""
        lowered = received_text.lower()
        if any(token in lowered for token in ("fair", "reasonable", "middle")):
            return "I understand you are aiming for a fair middle ground. "
        if any(token in lowered for token in ("need", "must", "important")):
            return "I understand some points matter a lot on your side. "
        if any(token in lowered for token in ("final", "last offer", "deadline")):
            return "I understand you want to conclude this soon. "
        return ""

    def _closing_phrase(self, state: SAOState, counter: Outcome) -> str:
        t = self._relative_time(state)
        opp = self._estimated_opponent_utility(counter)
        if t > 0.9:
            return "This is close to the best compromise I can justify now."
        if opp > 0.7:
            return "I believe this should work well for both of us."
        return "This keeps the deal viable while moving closer to agreement."

    def _describe_offer(self, offer: Outcome, limit: int = 2) -> str:
        if not self.nmi or not getattr(self.nmi.outcome_space, "issues", None):
            return str(offer)
        parts: list[str] = []
        for issue, value in zip(self.nmi.outcome_space.issues[:limit], offer):
            parts.append(f"{issue.name}={value}")
        return ", ".join(parts)

    def _describe_differences(self, their_offer: Outcome, counter: Outcome) -> str:
        if not self.nmi or not getattr(self.nmi.outcome_space, "issues", None):
            return "the offer"

        changes: list[str] = []
        for issue, old, new in zip(self.nmi.outcome_space.issues, their_offer, counter):
            if old == new:
                continue
            direction = self._change_direction(old, new)
            changes.append(f"{issue.name} from {old} to {new}{direction}")
            if len(changes) == 2:
                break
        return " and ".join(changes)

    def _change_direction(self, old: Any, new: Any) -> str:
        try:
            old_num = float(old)
            new_num = float(new)
        except (TypeError, ValueError):
            return ""

        if new_num > old_num:
            return " to improve my side"
        if new_num < old_num:
            return " to keep the package balanced"
        return ""


class HanOmegaNegotiator(_BaseHanOmegaNegotiator):
    def _should_accept(self, state: SAOState, offer: Outcome) -> bool:
        if super()._should_accept(state, offer):
            return True
        assert self.ufun is not None
        offer_utility = float(self.ufun(offer))
        return (
            self._best_received_offer is not None
            and self._relative_time(state) >= 0.92
            and offer_utility >= self._best_received_utility - 0.02
        )

    def _late_satisficing_offer(self, state: SAOState) -> RankedOutcome | None:
        rescue = super()._late_satisficing_offer(state)
        if self._relative_time(state) < 0.75 or self._best_received_offer is None:
            return self._text_anchor_probe(state, rescue)
        assert self.ufun is not None
        threshold = self._aspiration_threshold(state)
        repaired = self._repair_from_offer(
            self._best_received_offer,
            max(float(self.ufun.reserved_value), threshold - 0.05),
        )
        if repaired is None:
            return self._text_anchor_probe(state, rescue)
        repaired_ranked = RankedOutcome(outcome=repaired, utility=float(self.ufun(repaired)))
        if rescue is None:
            return self._text_anchor_probe(state, repaired_ranked)
        if (
            self._acceptance_likelihood(repaired_ranked.outcome)
            > self._acceptance_likelihood(rescue.outcome) + 0.05
        ):
            return self._text_anchor_probe(state, repaired_ranked)
        return self._text_anchor_probe(state, rescue)


class MyNegotiator(HanOmegaNegotiator):
    """Backward-compatible alias for local development only."""
