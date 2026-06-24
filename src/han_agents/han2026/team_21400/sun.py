from __future__ import annotations

from typing import Any

from negmas.common import Outcome
from negmas.gb.common import ExtendedResponseType, ResponseType
from negmas.gb.components.genius.models import GSmithFrequencyModel
from negmas.outcomes import ExtendedOutcome
from negmas.sao import SAOState
from negmas.sao.components.acceptance import ACNext
from negmas.sao.components.offering import TimeBasedOfferingPolicy
from negmas.sao.negotiators.modular import BOANegotiator


class Sun(BOANegotiator):
    """
    behavior_safe baseline + grocery rigid-zero fix.

    Main idea:
    - Keep Trade behavior unchanged.
    - Keep named Island behavior unchanged.
    - Add a narrow grocery rescue branch for opponents that keep repeating
      the minimum bundle (e.g. SimpleNeg-like rigid zero offers).
    - IMPORTANT: Do not let the normal late-phase guard block those rescue offers.
    """

    # Start adapting earlier, but avoid over-conceding too soon.
    LATE_COUNTER_START = 0.50
    ISLAND_RIGID_SPECIAL_START = 0.75
    GROCERY_ZERO_SPECIAL_START = 0.40

    # Concession/acceptance safety parameters.
    MAX_DROP_PER_STEP = 0.07
    LAST_OFFER_ACCEPT_GAP = 0.04
    MAX_EXACT_SEARCH = 8000

    # Social/Nash-aware search starts after we have some opponent evidence.
    NASH_START = 0.40
    FINAL_SAFETY_START = 0.92

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Compatibility with the original LLM-based template/tests.
        # Sun is BOA-based, so these LLM-specific parameters are accepted
        # but intentionally ignored.
        self._llm_compat_params = {
            "model": kwargs.pop("model", None),
            "temperature": kwargs.pop("temperature", None),
            "max_tokens": kwargs.pop("max_tokens", None),
            "use_structured_output": kwargs.pop("use_structured_output", None),
            "system_prompt": kwargs.pop("system_prompt", None),
            "preferences_prompt": kwargs.pop("preferences_prompt", None),
            "preferences_changed_prompt": kwargs.pop("preferences_changed_prompt", None),
            "negotiation_start_prompt": kwargs.pop("negotiation_start_prompt", None),
            "round_prompt": kwargs.pop("round_prompt", None),
        }

        offering = TimeBasedOfferingPolicy()
        kwargs |= dict(
            offering=offering,
            acceptance=ACNext(offering),
            model=GSmithFrequencyModel(),
        )
        super().__init__(*args, **kwargs)
        self._last_offer: Outcome | None = None
        self._last_offer_ratio: float = 1.0
        self._candidate_cache: list[Outcome] | None = None
        self._scenario_kind_cache: str | None = None
        self._best_utility_cache: float | None = None
        self._partner_offer_history: list[Outcome] = []

    # ------------------------------------------------------------
    # Lifecycle / partner history
    # ------------------------------------------------------------
    def on_partner_proposal(self, state: SAOState, partner_id: str, offer: Outcome) -> None:  # type: ignore[override]
        try:
            super().on_partner_proposal(state, partner_id, offer)  # type: ignore[arg-type]
        except Exception:
            pass

        if offer is not None:
            try:
                self._partner_offer_history.append(tuple(offer))
                self._partner_offer_history = self._partner_offer_history[-10:]
            except Exception:
                pass

    # ------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------
    def _utility(self, outcome: Outcome | None) -> float:
        if outcome is None or self.ufun is None:
            return 0.0
        try:
            return float(self.ufun(outcome))
        except Exception:
            return 0.0

    def _best_utility(self) -> float:
        if self._best_utility_cache is not None:
            return self._best_utility_cache
        if self.ufun is None:
            self._best_utility_cache = 1.0
            return self._best_utility_cache
        try:
            best_outcome = self.ufun.best()
            best_u = float(self.ufun(best_outcome))
            self._best_utility_cache = best_u if best_u > 0 else 1.0
        except Exception:
            self._best_utility_cache = 1.0
        return self._best_utility_cache

    def _utility_ratio(self, outcome: Outcome | None) -> float:
        best_u = self._best_utility()
        if best_u <= 0:
            return 0.0
        return self._utility(outcome) / best_u

    def _estimated_opp_utility(self, outcome: Outcome) -> float:
        try:
            model = self.opponent_ufun
        except Exception:
            return 0.0
        if model is None:
            return 0.0
        try:
            return self._clamp01(float(model(outcome)))
        except Exception:
            return 0.0

    def _clamp01(self, value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _reserved_ratio(self) -> float:
        if self.ufun is None:
            return 0.0
        try:
            rv = getattr(self.ufun, "reserved_value", 0.0)
            if rv is None:
                return 0.0
            return self._clamp01(float(rv) / max(self._best_utility(), 1e-9))
        except Exception:
            return 0.0

    def _agreement_pressure(self, t: float) -> float:
        # 0 before NASH_START, 1 at the deadline. Used to move from selfish
        # bidding toward agreement-oriented bidding without becoming a conceder.
        if t <= self.NASH_START:
            return 0.0
        return self._clamp01((t - self.NASH_START) / max(1e-9, 1.0 - self.NASH_START))

    def _social_score(self, my_ratio: float, opp_u: float, t: float, dist: float = 0.0) -> float:
        my_ratio = self._clamp01(my_ratio)
        opp_u = self._clamp01(opp_u)
        dist = self._clamp01(dist)
        pressure = self._agreement_pressure(t)

        # Early: mostly self utility. Late: consider opponent model and Nash product.
        nash = (max(my_ratio, 0.0) * max(opp_u, 0.0)) ** 0.5
        self_w = 0.72 - 0.20 * pressure
        opp_w = 0.12 + 0.12 * pressure
        nash_w = 0.10 + 0.18 * pressure
        dist_w = 0.06 + 0.06 * pressure
        return self_w * my_ratio + opp_w * opp_u + nash_w * nash - dist_w * dist

    def _deadline_accept_floor(self, t: float) -> float:
        # Safety net: near the end, accept reasonable offers above reservation
        # instead of risking a zero-agreement close.
        reserved = self._reserved_ratio()
        if t < self.FINAL_SAFETY_START:
            return 1.0
        if t < 0.97:
            return max(reserved + 0.04, 0.34)
        if t < 0.99:
            return max(reserved + 0.02, 0.28)
        return max(reserved + 0.01, 0.22)

    # ------------------------------------------------------------
    # Thresholds / floors in ratio space
    # ------------------------------------------------------------
    def _accept_threshold(self, t: float) -> float:
        # Slightly less rigid than the old 0.95/0.92 early policy.
        # Still protects us from accepting weak offers too early.
        if t < 0.20:
            return 0.92
        if t < 0.40:
            return 0.84
        if t < 0.60:
            return 0.74
        if t < 0.75:
            return 0.64
        if t < 0.90:
            return 0.54
        if t < 0.97:
            return 0.42
        return 0.32

    def _offer_floor(self, t: float) -> float:
        # Controlled concession curve. This starts moving earlier than 0.60
        # while keeping a firm lower bound above reservation.
        if t < 0.20:
            base = 0.88
        elif t < 0.40:
            base = 0.80
        elif t < 0.60:
            base = 0.70
        elif t < 0.75:
            base = 0.60
        elif t < 0.90:
            base = 0.50
        elif t < 0.97:
            base = 0.40
        else:
            base = 0.30
        return max(base, self._reserved_ratio() + 0.03)

    def _generic_closeness_cap(self, t: float) -> float:
        if t < 0.70:
            return 0.45
        if t < 0.80:
            return 0.30
        if t < 0.90:
            return 0.18
        return 0.10

    # ------------------------------------------------------------
    # Outcome-space helpers
    # ------------------------------------------------------------
    def _issue_values(self, issue) -> list[Any]:
        for attr in ("all", "values"):
            try:
                vals = list(getattr(issue, attr))
                if vals:
                    return vals
            except Exception:
                pass
        return []

    def _issue_bounds(self, issue) -> tuple[float, float] | None:
        vals = self._issue_values(issue)
        if not vals:
            return None
        try:
            return float(min(vals)), float(max(vals))
        except Exception:
            return None

    def _issue_range(self, issue) -> float:
        b = self._issue_bounds(issue)
        if b is None:
            return 1.0
        return max(1.0, b[1] - b[0])

    def _enumerate_candidates(self) -> list[Outcome]:
        if self._candidate_cache is not None:
            return self._candidate_cache
        if self.nmi is None:
            self._candidate_cache = []
            return self._candidate_cache
        try:
            self._candidate_cache = [
                tuple(x)
                for x in self.nmi.outcome_space.enumerate_or_sample(
                    max_cardinality=self.MAX_EXACT_SEARCH
                )
            ]
        except Exception:
            self._candidate_cache = []
        return self._candidate_cache

    def _scenario_kind(self) -> str:
        if self._scenario_kind_cache is not None:
            return self._scenario_kind_cache
        if self.nmi is None:
            self._scenario_kind_cache = "generic"
            return self._scenario_kind_cache
        try:
            issues = list(self.nmi.outcome_space.issues)
        except Exception:
            self._scenario_kind_cache = "generic"
            return self._scenario_kind_cache
        if not issues:
            self._scenario_kind_cache = "generic"
            return self._scenario_kind_cache

        values0 = self._issue_values(issues[0])
        if values0 and any(isinstance(v, str) for v in values0):
            self._scenario_kind_cache = "island"
            return self._scenario_kind_cache

        numeric_bounds = [self._issue_bounds(i) for i in issues]
        if len(issues) == 2 and all(b is not None for b in numeric_bounds):
            ranges = [(b[1] - b[0]) if b else 0.0 for b in numeric_bounds]
            if max(ranges) >= 20:
                self._scenario_kind_cache = "trade"
                return self._scenario_kind_cache

        if len(issues) == 4 and all(b is not None for b in numeric_bounds):
            pairs = [(b[0], b[1]) for b in numeric_bounds if b is not None]
            if all(lo >= 0 and hi <= 4 for lo, hi in pairs):
                self._scenario_kind_cache = "grocery"
                return self._scenario_kind_cache

        self._scenario_kind_cache = "generic"
        return self._scenario_kind_cache

    def _normalized_distance(self, a: Outcome, b: Outcome) -> float:
        if self.nmi is None:
            return 1.0
        try:
            issues = list(self.nmi.outcome_space.issues)
        except Exception:
            return 1.0

        n = min(len(a), len(b), len(issues))
        if n == 0:
            return 1.0

        total = 0.0
        for i in range(n):
            rng = self._issue_range(issues[i])
            try:
                total += abs(float(a[i]) - float(b[i])) / rng
            except Exception:
                total += 0.0 if a[i] == b[i] else 1.0
        return total / n

    # ------------------------------------------------------------
    # Behavior detectors
    # ------------------------------------------------------------
    def _partner_is_rigid_island_extreme(self) -> bool:
        if self._scenario_kind() != "island":
            return False
        if len(self._partner_offer_history) < 4:
            return False

        tail = self._partner_offer_history[-4:]
        first = tuple(tail[0])

        if any(tuple(x) != first for x in tail[1:]):
            return False

        try:
            return len(set(first)) == 1
        except Exception:
            return False

    def _partner_is_rigid_grocery_zero(self) -> bool:
        if self._scenario_kind() != "grocery":
            return False
        if len(self._partner_offer_history) < 4 or self.nmi is None:
            return False

        tail = self._partner_offer_history[-4:]
        first = tuple(tail[0])

        if any(tuple(x) != first for x in tail[1:]):
            return False

        try:
            issues = list(self.nmi.outcome_space.issues)
            if len(first) != len(issues):
                return False

            mins = []
            for issue in issues:
                bounds = self._issue_bounds(issue)
                if bounds is None:
                    return False
                mins.append(bounds[0])

            return all(abs(float(v) - float(m)) < 1e-9 for v, m in zip(first, mins))
        except Exception:
            return False

    def _grocery_rescue_active(self, state: SAOState | None) -> bool:
        return (
            self._scenario_kind() == "grocery"
            and self._partner_is_rigid_grocery_zero()
            and float(getattr(state, "relative_time", 0.0)) >= self.GROCERY_ZERO_SPECIAL_START
        )

    # ------------------------------------------------------------
    # Guard / offer selection
    # ------------------------------------------------------------
    def _guard_offer(self, state: SAOState, outcome: Outcome) -> Outcome:
        t = float(getattr(state, "relative_time", 0.0))
        ratio = self._utility_ratio(outcome)
        floor = self._offer_floor(t)
        kind = self._scenario_kind()

        # Very important fix:
        # If grocery rescue is active, do NOT let the normal late-phase
        # guard force us back to high-utility selfish offers.
        if self._grocery_rescue_active(state):
            if t < 0.70:
                relaxed_floor = 0.35
            elif t < 0.85:
                relaxed_floor = 0.20
            elif t < 0.95:
                relaxed_floor = 0.08
            else:
                relaxed_floor = 0.02

            if ratio >= relaxed_floor:
                return outcome
            if self._last_offer is not None:
                return self._last_offer
            return outcome

        if kind in {"trade", "island"} and t >= self.LATE_COUNTER_START:
            if ratio < floor and self._last_offer is not None:
                return self._last_offer
            return outcome

        if self._last_offer is not None:
            floor = max(floor, self._last_offer_ratio - self.MAX_DROP_PER_STEP)

        if ratio < floor and self._last_offer is not None:
            return self._last_offer
        return outcome

    def _best_trade_offer(self, state: SAOState, anchor: Outcome | None) -> Outcome | None:
        if self.ufun is None or anchor is None or self.nmi is None:
            return None
        if len(anchor) != 2:
            return None

        t = float(getattr(state, "relative_time", 0.0))
        floor = self._offer_floor(t)
        progress = max(
            0.0,
            min(1.0, (t - self.LATE_COUNTER_START) / max(1e-9, 1.0 - self.LATE_COUNTER_START)),
        )

        try:
            issues = list(self.nmi.outcome_space.issues)
        except Exception:
            return None

        range0 = self._issue_range(issues[0]) if len(issues) > 0 else 1.0
        range1 = self._issue_range(issues[1]) if len(issues) > 1 else 1.0

        candidates = self._enumerate_candidates()
        if not candidates:
            return None

        good: list[tuple[float, Outcome]] = []

        for cand in candidates:
            if len(cand) != 2:
                continue

            my_ratio = self._utility_ratio(cand)
            if my_ratio < floor:
                continue

            try:
                norm_gap1 = abs(float(cand[0]) - float(anchor[0])) / range0
                norm_gap2 = abs(float(cand[1]) - float(anchor[1])) / range1
            except Exception:
                continue

            opp_u = self._estimated_opp_utility(cand)
            dist = (norm_gap1 + norm_gap2) / 2.0
            score = self._social_score(my_ratio, opp_u, t, dist=dist)
            score += 0.05 * (1.0 - dist)
            score -= 0.06 * (norm_gap1 + norm_gap2) * progress
            good.append((score, cand))

        if not good:
            return None
        good.sort(reverse=True, key=lambda x: x[0])
        return good[0][1]

    def _best_island_offer(self, state: SAOState, anchor: Outcome | None) -> Outcome | None:
        if self.ufun is None or anchor is None:
            return None

        t = float(getattr(state, "relative_time", 0.0))
        floor = max(0.20, self._offer_floor(t) - 0.15)
        progress = max(
            0.0,
            min(1.0, (t - self.LATE_COUNTER_START) / max(1e-9, 1.0 - self.LATE_COUNTER_START)),
        )

        n = len(anchor)
        if n == 0:
            return None

        candidates = self._enumerate_candidates()
        if not candidates:
            return None

        if progress < 0.25:
            target_matches = 2
        elif progress < 0.50:
            target_matches = 3
        elif progress < 0.75:
            target_matches = 4
        else:
            target_matches = 5

        good: list[tuple[float, Outcome]] = []
        fallback: list[tuple[float, Outcome]] = []

        for cand in candidates:
            if len(cand) != n:
                continue

            my_ratio = self._utility_ratio(cand)
            if my_ratio < floor:
                continue

            matches = sum(1 for x, y in zip(cand, anchor) if x == y)

            if t >= self.LATE_COUNTER_START and (matches == 0 or matches == n):
                continue

            match_ratio = matches / n
            target_bonus = 1.0 - abs(matches - target_matches) / n
            opp_u = self._estimated_opp_utility(cand)

            # Keep island conservative, but use a little opponent/Nash awareness
            # after the late counter phase starts.
            score = self._social_score(my_ratio, opp_u, t, dist=1.0 - match_ratio)
            score += 0.12 * target_bonus + 0.04 * match_ratio

            if matches >= target_matches:
                good.append((score, cand))
            else:
                fallback.append((score, cand))

        if good:
            good.sort(reverse=True, key=lambda x: x[0])
            return good[0][1]
        if fallback:
            fallback.sort(reverse=True, key=lambda x: x[0])
            return fallback[0][1]
        return None

    def _best_island_rigid_offer(self, state: SAOState, anchor: Outcome | None) -> Outcome | None:
        if self.ufun is None or anchor is None:
            return None

        t = float(getattr(state, "relative_time", 0.0))
        if t < self.ISLAND_RIGID_SPECIAL_START:
            return None

        n = len(anchor)
        if n == 0:
            return None

        candidates = self._enumerate_candidates()
        if not candidates:
            return None

        floor = max(0.15, self._offer_floor(t) - 0.22)
        progress = max(
            0.0,
            min(1.0, (t - self.ISLAND_RIGID_SPECIAL_START) / max(1e-9, 1.0 - self.ISLAND_RIGID_SPECIAL_START)),
        )

        if progress < 0.33:
            target_matches = max(4, n // 2)
        elif progress < 0.66:
            target_matches = max(5, n - 3)
        else:
            target_matches = max(6, n - 2)

        good: list[tuple[float, Outcome]] = []
        fallback: list[tuple[float, Outcome]] = []

        for cand in candidates:
            if len(cand) != n:
                continue

            my_ratio = self._utility_ratio(cand)
            if my_ratio < floor:
                continue

            matches = sum(1 for x, y in zip(cand, anchor) if x == y)
            match_ratio = matches / n
            target_bonus = 1.0 - abs(matches - target_matches) / n
            opp_u = self._estimated_opp_utility(cand)

            try:
                mixed = len(set(cand)) > 1
            except Exception:
                mixed = True

            if not mixed:
                continue

            score = self._social_score(my_ratio, opp_u, t, dist=1.0 - match_ratio)
            score += 0.14 * target_bonus + 0.06 * match_ratio

            if matches >= target_matches:
                good.append((score, cand))
            else:
                fallback.append((score, cand))

        if good:
            good.sort(reverse=True, key=lambda x: x[0])
            return good[0][1]
        if fallback:
            fallback.sort(reverse=True, key=lambda x: x[0])
            return fallback[0][1]
        return None

    def _best_grocery_rigid_offer(self, state: SAOState, anchor: Outcome | None) -> Outcome | None:
        """
        Grocery rescue for opponents that keep repeating the minimum bundle.

        Key idea:
        - prioritize closeness to the minimum bundle (what the opponent keeps asking for)
        - still keep some self-utility
        - use a time-dependent target so we move earlier and more aggressively
        """
        if self.ufun is None or self.nmi is None:
            return None
        if not self._grocery_rescue_active(state):
            return None

        t = float(getattr(state, "relative_time", 0.0))
        try:
            issues = list(self.nmi.outcome_space.issues)
        except Exception:
            return None

        candidates = self._enumerate_candidates()
        if not candidates:
            return None

        if t < 0.70:
            floor = 0.35
            target_zero = 0.55
        elif t < 0.85:
            floor = 0.20
            target_zero = 0.72
        elif t < 0.95:
            floor = 0.08
            target_zero = 0.85
        else:
            floor = 0.02
            target_zero = 0.95

        strict: list[tuple[float, Outcome]] = []
        fallback: list[tuple[float, Outcome]] = []

        for cand in candidates:
            if len(cand) != len(issues):
                continue

            my_ratio = self._utility_ratio(cand)
            if my_ratio < floor:
                continue

            opp_u = self._estimated_opp_utility(cand)

            # closeness to the repeated minimum bundle
            zero_closeness = 0.0
            for i, issue in enumerate(issues):
                bounds = self._issue_bounds(issue)
                if bounds is None:
                    continue
                lo, hi = bounds
                rng = max(1e-9, hi - lo)
                try:
                    zero_closeness += 1.0 - ((float(cand[i]) - lo) / rng)
                except Exception:
                    pass
            zero_closeness /= max(1, len(issues))

            # Prefer candidates that both approach the minimum bundle and keep some utility.
            # Estimated opponent utility is still used, but zero_closeness gets the highest weight.
            threshold_progress = min(zero_closeness / max(target_zero, 1e-9), 1.2)
            social = self._social_score(my_ratio, opp_u, t, dist=1.0 - zero_closeness)
            score = (
                0.35 * threshold_progress
                + 0.35 * social
                + 0.20 * my_ratio
                + 0.10 * zero_closeness
            )

            if zero_closeness >= target_zero:
                strict.append((score, cand))
            else:
                fallback.append((score, cand))

        if strict:
            strict.sort(reverse=True, key=lambda x: x[0])
            return strict[0][1]

        if fallback:
            fallback.sort(reverse=True, key=lambda x: x[0])
            return fallback[0][1]

        return None

    def _best_generic_offer(self, state: SAOState, anchor: Outcome | None) -> Outcome | None:
        if self.ufun is None:
            return None

        t = float(getattr(state, "relative_time", 0.0))
        floor = self._offer_floor(t)
        cap = self._generic_closeness_cap(t)
        candidates = self._enumerate_candidates()

        good: list[tuple[float, float, float, Outcome]] = []
        fallback: list[tuple[float, Outcome]] = []

        for cand in candidates:
            my_ratio = self._utility_ratio(cand)
            if my_ratio < floor:
                continue

            dist = self._normalized_distance(cand, anchor) if anchor is not None else 0.0
            opp_u = self._estimated_opp_utility(cand)

            if anchor is not None and dist <= cap:
                good.append((my_ratio, opp_u, -dist, cand))

            score = self._social_score(my_ratio, opp_u, t, dist=dist)
            fallback.append((score, cand))

        if good:
            good.sort(reverse=True)
            return good[0][3]
        if fallback:
            fallback.sort(key=lambda x: x[0], reverse=True)
            return fallback[0][1]
        return None

    def _best_deadline_offer(self, state: SAOState, anchor: Outcome | None) -> Outcome | None:
        if self.ufun is None:
            return None
        t = float(getattr(state, "relative_time", 0.0))
        if t < self.FINAL_SAFETY_START:
            return None

        floor = self._deadline_accept_floor(t)
        candidates = self._enumerate_candidates()
        if not candidates:
            return None

        best: tuple[float, Outcome] | None = None
        for cand in candidates:
            my_ratio = self._utility_ratio(cand)
            if my_ratio < floor:
                continue
            opp_u = self._estimated_opp_utility(cand)
            dist = self._normalized_distance(cand, anchor) if anchor is not None else 0.0
            score = self._social_score(my_ratio, opp_u, t, dist=dist) + 0.08 * opp_u
            if best is None or score > best[0]:
                best = (score, cand)

        return best[1] if best is not None else None

    def _base_offer(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        raw = super().propose(state, dest=dest)
        if raw is None:
            return None
        return raw.outcome if isinstance(raw, ExtendedOutcome) else raw

    def _planned_offer(self, state: SAOState, dest: str | None = None) -> Outcome | None:
        t = float(getattr(state, "relative_time", 0.0))
        anchor = getattr(state, "current_offer", None)

        if anchor is not None:
            kind = self._scenario_kind()

            if kind == "trade" and t >= self.LATE_COUNTER_START:
                special = self._best_trade_offer(state, anchor)
                if special is not None:
                    return self._guard_offer(state, special)

            if kind == "island" and t >= self.LATE_COUNTER_START:
                if self._partner_is_rigid_island_extreme():
                    rigid_special = self._best_island_rigid_offer(state, anchor)
                    if rigid_special is not None:
                        return self._guard_offer(state, rigid_special)

                special = self._best_island_offer(state, anchor)
                if special is not None:
                    return self._guard_offer(state, special)

            if kind == "grocery" and t >= self.GROCERY_ZERO_SPECIAL_START:
                special = self._best_grocery_rigid_offer(state, anchor)
                if special is not None:
                    return self._guard_offer(state, special)

            if t >= self.FINAL_SAFETY_START:
                deadline = self._best_deadline_offer(state, anchor)
                if deadline is not None:
                    return self._guard_offer(state, deadline)

            if t >= self.LATE_COUNTER_START:
                generic = self._best_generic_offer(state, anchor)
                if generic is not None:
                    return self._guard_offer(state, generic)

        base = self._base_offer(state, dest=dest)
        if base is None:
            return None
        return self._guard_offer(state, base)

    def propose(self, state: SAOState, dest: str | None = None):
        offer = self._planned_offer(state, dest=dest)
        if offer is None:
            return None

        self._last_offer = offer
        self._last_offer_ratio = self._utility_ratio(offer)

        return ExtendedOutcome(
            outcome=offer,
            data={"text": self._compose_offer_text(state, offer)},
        )

    # ------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------
    def _should_accept(self, state: SAOState, offer: Outcome | None) -> bool:
        if offer is None or self.ufun is None:
            return False

        t = float(getattr(state, "relative_time", 0.0))
        ratio = self._utility_ratio(offer)

        if ratio >= self._accept_threshold(t):
            return True

        planned = self._planned_offer(state)
        if planned is not None and ratio >= self._utility_ratio(planned):
            return True

        kind = self._scenario_kind()

        if kind == "trade" and t >= 0.75 and planned is not None:
            try:
                if abs(float(offer[1]) - float(planned[1])) <= 15 and ratio >= 0.52:
                    return True
            except Exception:
                pass

        if kind == "island" and t >= 0.80:
            try:
                n = len(offer)
                anchor = getattr(state, "current_offer", None)
                matches = sum(1 for x, y in zip(offer, anchor) if x == y) if anchor is not None else 0

                if n > 0 and matches >= max(3, n // 2) and ratio >= 0.15:
                    return True

                if self._partner_is_rigid_island_extreme():
                    mixed = len(set(offer)) > 1
                    if mixed and matches >= max(4, n // 2) and ratio >= 0.12:
                        return True
            except Exception:
                pass

        if kind == "grocery" and self._partner_is_rigid_grocery_zero():
            try:
                opp_u = self._estimated_opp_utility(offer)
                if t >= 0.90 and opp_u >= 0.70 and ratio >= 0.02:
                    return True
                if t >= 0.97 and ratio >= 0.01:
                    return True
            except Exception:
                pass

        # Near-deadline safety net. This avoids losing all value by timing out
        # when the current offer is reasonable and above reservation.
        if t >= self.FINAL_SAFETY_START and ratio >= self._deadline_accept_floor(t):
            return True

        if (
            self._last_offer is not None
            and t >= self.LATE_COUNTER_START
            and ratio >= (self._last_offer_ratio - self.LAST_OFFER_ACCEPT_GAP)
        ):
            return True

        return False

    def respond(self, state: SAOState, source: str | None = None):
        offer = getattr(state, "current_offer", None)

        if self._should_accept(state, offer):
            return ExtendedResponseType(
                response=ResponseType.ACCEPT_OFFER,
                data={"text": self._compose_accept_text(state, offer)},
            )

        base = super().respond(state, source=source)
        response_type = base.response if isinstance(base, ExtendedResponseType) else base

        if response_type == ResponseType.END_NEGOTIATION:
            return ExtendedResponseType(
                response=ResponseType.END_NEGOTIATION,
                data={"text": self._compose_end_text()},
            )
        return base

    # ------------------------------------------------------------
    # Deterministic human-facing text
    # ------------------------------------------------------------
    def _offer_str(self, outcome: Outcome | None) -> str:
        if outcome is None:
            return "this deal"
        try:
            return str(tuple(outcome))
        except Exception:
            return "this deal"

    def _phase(self, state: SAOState | None) -> str:
        t = float(getattr(state, "relative_time", 0.0)) if state is not None else 0.0
        if t < 0.33:
            return "early"
        if t < 0.75:
            return "mid"
        return "late"

    def _compose_offer_text(self, state: SAOState, outcome: Outcome) -> str:
        offer_str = self._offer_str(outcome)
        phase = self._phase(state)
        if phase == "early":
            return f"I want to start from a firm but fair position. My proposal is {offer_str}."
        if phase == "mid":
            return f"I’ve moved where I can, and I’m looking for movement from your side too. I’m proposing {offer_str}."
        return f"We’re close to the end, and I want a practical close. I can do {offer_str}."

    def _compose_accept_text(self, state: SAOState, outcome: Outcome | None) -> str:
        offer_str = self._offer_str(outcome)
        if self._phase(state) == "late":
            return f"We’re close enough now, and this works for me. I accept {offer_str}."
        return f"This works for me. I accept {offer_str}."

    def _compose_end_text(self) -> str:
        return "I do not think we are finding a balanced agreement here. Thank you for the discussion."
