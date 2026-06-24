# pareto.py — Pareto frontier + Nash bargaining point via NegMAS built-ins
# HERALD HAN 2026 agent
#
# Uses negmas.pareto_frontier (C-optimized) and negmas.nash_points instead of
# the manual O(n²) algorithm.  A MappingUtilityFunction wraps the opponent
# model's estimate so both NegMAS functions receive standard ufun objects.

from __future__ import annotations
from typing import Optional

from negmas import Outcome, pareto_frontier, nash_points
from negmas.preferences import MappingUtilityFunction


class ParetoAnalyser:
    """
    Maintains the Pareto frontier and Nash bargaining solution in the joint
    utility space (my_util × estimated_opp_util).

    Call build() after the opponent model has enough data (round ≥ 8).
    Call rebuild() whenever the opponent model's estimates change significantly.
    """

    def __init__(self, ufun, opp_model) -> None:
        self._ufun = ufun
        self._opp_model = opp_model
        # frontier entries: (outcome, my_util, est_opp_util)
        self._frontier: list[tuple[Outcome, float, float]] = []
        self._nash_outcome: Optional[Outcome] = None
        self._nash_point: Optional[tuple[float, float]] = None
        self._built: bool = False
        self._outcome_space = None   # cached for opp_proxy

    def build(self, outcomes: list[Outcome], reservation: float) -> None:
        """
        Compute Pareto frontier and Nash bargaining point using NegMAS built-ins.

        Creates a MappingUtilityFunction proxy from opp_model estimates, then
        calls pareto_frontier() and nash_points().
        """
        if not outcomes:
            return
        try:
            self._built = False
            self._frontier = []
            self._nash_outcome = None
            self._nash_point = None

            # Build opponent utility proxy
            mapping = {
                o: self._opp_model.get_estimated_utility(o) for o in outcomes
            }
            opp_proxy = MappingUtilityFunction(mapping)

            # Give opp_proxy an outcome_space so nash_points can call minmax()
            os_ = getattr(self._ufun, "outcome_space", None)
            if os_ is None and self._outcome_space is not None:
                os_ = self._outcome_space
            if os_ is not None:
                try:
                    opp_proxy.outcome_space = os_
                except Exception:
                    pass

            # Compute Pareto frontier
            pf_utils, pf_indices = pareto_frontier((self._ufun, opp_proxy), outcomes)

            if not pf_utils:
                return

            # Store frontier entries (outcome, my_util, opp_util)
            self._frontier = [
                (
                    outcomes[int(pf_indices[i])],
                    float(pf_utils[i][0]),
                    float(pf_utils[i][1]),
                )
                for i in range(len(pf_utils))
            ]
            # Sort descending by my utility
            self._frontier.sort(key=lambda x: x[1], reverse=True)

            self._built = True

            # Nash bargaining point
            self._compute_nash(pf_utils, pf_indices, outcomes, opp_proxy, reservation)

        except Exception:
            self._built = False

    def _compute_nash(
        self,
        pf_utils,
        pf_indices,
        outcomes: list[Outcome],
        opp_proxy,
        reservation: float,
    ) -> None:
        """
        Find Nash bargaining outcome.  Tries negmas.nash_points first; falls back
        to manual argmax over (my_u - rv) × opp_u on the Pareto frontier.
        """
        # Try NegMAS built-in
        try:
            np_res = nash_points((self._ufun, opp_proxy), pf_utils)
            if np_res:
                nash_utils, nash_frontier_idx = np_res[0]
                self._nash_outcome = outcomes[int(pf_indices[nash_frontier_idx])]
                self._nash_point = (float(nash_utils[0]), float(nash_utils[1]))
                return
        except Exception:
            pass

        # Manual fallback: argmax Nash product over frontier
        best_nash = -1.0
        best_item = self._frontier[0] if self._frontier else None
        opp_rv = 0.0
        for o, u, v in self._frontier:
            nash_val = max(0.0, u - reservation) * max(0.0, v - opp_rv)
            if nash_val > best_nash:
                best_nash = nash_val
                best_item = (o, u, v)
        if best_item is not None:
            self._nash_outcome = best_item[0]
            self._nash_point = (best_item[1], best_item[2])

    def rebuild(self, outcomes: list[Outcome], reservation: float) -> None:
        """Alias for build() — used when opp_model updates change estimates."""
        self.build(outcomes, reservation)

    def set_outcome_space(self, os_) -> None:
        """Store the negotiation outcome space for opp_proxy use."""
        self._outcome_space = os_

    # ── Query methods ────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        return self._built

    def get_pareto_candidates(self, my_util_min: float) -> list[Outcome]:
        """All Pareto outcomes with my utility ≥ my_util_min."""
        return [o for o, u, _ in self._frontier if u >= my_util_min]

    def get_nash_outcome(self) -> Optional[Outcome]:
        return self._nash_outcome

    def get_nash_point(self) -> Optional[tuple[float, float]]:
        return self._nash_point

    def get_social_welfare_outcome(self, my_util_min: float) -> Optional[Outcome]:
        """Pareto outcome with my_util ≥ my_util_min that maximises u + v."""
        candidates = [(o, u, v) for o, u, v in self._frontier if u >= my_util_min]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1] + x[2])[0]

    def distance_to_pareto(self, outcome: Outcome) -> float:
        """Euclidean distance from outcome's (my_u, opp_u) to nearest frontier point."""
        if not self._frontier:
            return 1.0
        try:
            my_u = float(self._ufun(outcome))
            opp_u = self._opp_model.get_estimated_utility(outcome)
            return min(
                ((u - my_u) ** 2 + (v - opp_u) ** 2) ** 0.5
                for _, u, v in self._frontier
            )
        except Exception:
            return 1.0

    def is_on_frontier(self, outcome: Outcome, tol: float = 0.05) -> bool:
        return self.distance_to_pareto(outcome) <= tol

    def frontier_my_range(self) -> tuple[float, float]:
        if not self._frontier:
            return (0.0, 1.0)
        utils = [u for _, u, _ in self._frontier]
        return (min(utils), max(utils))
