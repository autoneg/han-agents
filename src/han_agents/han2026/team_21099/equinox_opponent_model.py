"""Bayesian opponent model for Equinox.

``HindriksTykhonovBayesianOpponentModel`` is a NegMAS-compatible full-joint
discrete Bayesian opponent model. It observes the opponent's proposals and
estimates their utility function by marginalizing over a grid of concession
speeds (``beta``), issue-weight rankings, and per-issue evaluation-curve shapes.

Equinox uses it for two things:

1. Scoring the **Deception** term (it is exposed as ``opponent_ufun``), and
2. Driving the propose-time Pareto nudge (``estimated_utility`` ranks candidate
   offers by how good they are for the opponent).

The posterior is the *full joint* over (beta x weight-rank x eval-combo), which
grows as ``n_beta * n_issues! * prod(2 + n_values_i)``. That is exact and cheap
for the small domains this model is meant for (<= ~3 issues) but explodes on
larger ones, so :func:`bayesian_state_count` lets callers gate on tractability
and ``max_states`` is a hard self-limit (the model stays uninitialized above it,
and Equinox falls back to a cheap frequency model).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from attrs import define, field

try:  # pragma: no cover - import shape depends on how the package is loaded
    from negmas.gb.components.genius.base import GeniusOpponentModel
except Exception:  # pragma: no cover - fallback for older negmas layouts
    from negmas.gb.components.genius.models import GeniusOpponentModel  # type: ignore

__all__ = [
    "HindriksTykhonovBayesianOpponentModel",
    "bayesian_state_count",
]

#: Default ceiling on the posterior size. Above this the model refuses to
#: initialize (Equinox then uses a frequency model instead).
DEFAULT_MAX_STATES = 200_000


def _safe_normalize_log_probs(log_probs: np.ndarray) -> np.ndarray:
    """Normalize an array of log-probabilities so ``exp`` sums to 1.

    Falls back to a uniform distribution if the input is degenerate (all
    ``-inf`` / ``nan``), which keeps the posterior well-defined no matter what
    the likelihoods do.
    """
    if log_probs.size == 0:
        return log_probs
    flat_max = float(np.max(log_probs))
    if not math.isfinite(flat_max):
        return np.full_like(log_probs, -math.log(log_probs.size))
    total = float(np.sum(np.exp(log_probs - flat_max)))
    if not math.isfinite(total) or total <= 0.0:
        return np.full_like(log_probs, -math.log(log_probs.size))
    return log_probs - (flat_max + math.log(total))


def enumerate_outcomes(outcome_space) -> list:
    """Best-effort enumeration of an outcome space (fallback path only)."""
    if outcome_space is None:
        return []
    sampler = getattr(outcome_space, "enumerate_or_sample", None)
    if sampler is not None:
        try:
            return list(sampler(max_cardinality=100_000))
        except Exception:
            pass
    enumerator = getattr(outcome_space, "enumerate", None)
    if enumerator is not None:
        try:
            return list(enumerator())
        except Exception:
            pass
    return []


def _issue_value_counts(outcome_space) -> list[int] | None:
    """Return the number of values per issue, or ``None`` if undeterminable."""
    issues = getattr(outcome_space, "issues", None)
    if not issues:
        return None
    counts: list[int] = []
    for issue in issues:
        k = 0
        values = getattr(issue, "values", None)
        if values is not None:
            try:
                k = len(list(values))
            except TypeError:
                k = 0
        if k <= 0:
            cardinality = getattr(issue, "cardinality", None)
            try:
                if cardinality is not None and math.isfinite(float(cardinality)):
                    k = int(cardinality)
            except (TypeError, ValueError):
                k = 0
        if k <= 0:
            return None
        counts.append(k)
    return counts


def bayesian_state_count(outcome_space, n_beta: int = 5) -> int:
    """Project the posterior size for a domain without building anything.

    Returns a large sentinel when the domain shape cannot be determined, so
    callers treat "unknown" as "too big" and stay on the safe (cheap) path.
    """
    counts = _issue_value_counts(outcome_space)
    if not counts:
        return 10**9
    n_eval = 1
    for k in counts:
        n_eval *= 2 + k
        if n_eval > 10**9:
            return 10**9
    return n_beta * math.factorial(len(counts)) * n_eval


@define
class HindriksTykhonovBayesianOpponentModel(GeniusOpponentModel):
    """
    NegMAS-compatible full-joint discrete Bayesian opponent model.

    This model observes the opponent's proposals and estimates the opponent's
    utility function. It only assumes that observed bids roughly follow a
    time-dependent concession process. It does NOT assume a fixed Boulware curve;
    instead, it marginalizes over beta_grid.

    Posterior space:
        beta hypothesis × weight-rank hypothesis × eval hypothesis per issue

    For your 3-issue, 10-value domain:
        len(beta_grid) × 3! × 12^3 = 5 × 6 × 1728 = 51,840 states
    """

    sigma: float = 0.05
    min_utility: float = 0.5
    beta_grid: Sequence[float] = (0.2, 0.5, 1.0, 2.0, 3.0)
    default_utility_before_observations: float = 0.5
    #: Hard ceiling on the posterior size; above it the model stays uninitialized.
    max_states: int = DEFAULT_MAX_STATES

    _initialized: bool = field(init=False, default=False, repr=False)
    _issue_values: list[list[Any]] = field(init=False, factory=list, repr=False)
    _value_to_index: list[dict[Any, int]] = field(init=False, factory=list, repr=False)
    _n_issues: int = field(init=False, default=0, repr=False)
    _total_bids: int = field(init=False, default=0, repr=False)

    _weight_hypotheses: np.ndarray = field(init=False, factory=lambda: np.empty((0, 0)), repr=False)
    _rank_hypotheses: list[tuple[int, ...]] = field(init=False, factory=list, repr=False)
    _eval_curves: list[np.ndarray] = field(init=False, factory=list, repr=False)
    _eval_names: list[list[str]] = field(init=False, factory=list, repr=False)
    _eval_combo_array: np.ndarray = field(init=False, factory=lambda: np.empty((0, 0), dtype=int), repr=False)
    _log_posterior: np.ndarray = field(init=False, factory=lambda: np.empty(0), repr=False)

    _expected_components: list[np.ndarray] = field(init=False, factory=list, repr=False)
    _expected_weights_cache: np.ndarray | None = field(init=False, default=None, repr=False)
    _expected_eval_cache: list[np.ndarray] | None = field(init=False, default=None, repr=False)
    _offer_utility_cache: dict[tuple[Any, ...], float] = field(init=False, factory=dict, repr=False)
    _entropy_history: list[float] = field(init=False, factory=list, repr=False)
    _beta_posterior_history: list[np.ndarray] = field(init=False, factory=list, repr=False)

    # NegMAS lifecycle hooks ---------------------------------------------------

    def on_preferences_changed(self, changes) -> None:
        self._initialize_from_negotiator()

    def after_join(self, nmi) -> None:
        try:
            super().after_join(nmi)
        except Exception:
            pass
        self._initialize_from_nmi(nmi)

    def on_negotiation_start(self, state) -> None:
        try:
            super().on_negotiation_start(state)
        except Exception:
            pass
        self._initialize_from_negotiator()

    def _initialize_from_negotiator(self) -> None:
        negotiator = getattr(self, "negotiator", None)
        nmi = getattr(negotiator, "nmi", None) if negotiator is not None else None
        if nmi is not None:
            self._initialize_from_nmi(nmi)

    def _initialize_from_nmi(self, nmi) -> None:
        outcome_space = getattr(nmi, "outcome_space", None)
        if outcome_space is None:
            self._initialized = False
            return
        self.init_from_outcome_space(outcome_space)

    def init_from_outcome_space(self, outcome_space) -> None:
        issue_values = self._extract_issue_values(outcome_space)
        if not issue_values:
            self._initialized = False
            return

        # Refuse domains whose full-joint posterior would be intractable.
        n_eval = 1
        for values in issue_values:
            n_eval *= 2 + len(values)
        projected = len(self.beta_grid) * math.factorial(len(issue_values)) * n_eval
        if projected > self.max_states:
            self._initialized = False
            return

        self._issue_values = issue_values
        self._value_to_index = [
            {value: index for index, value in enumerate(values)}
            for values in self._issue_values
        ]
        self._n_issues = len(self._issue_values)
        self._total_bids = 0

        self._build_weight_hypotheses()
        self._build_evaluation_hypotheses()
        self._build_evaluation_combinations()
        self._initialize_uniform_posterior()
        self._recompute_expectations()

        self._entropy_history = [self.posterior_entropy()]
        self._beta_posterior_history = [self.beta_posterior()]
        self._initialized = True

    def _extract_issue_values(self, outcome_space) -> list[list[Any]]:
        issue_values: list[list[Any]] = []

        issues = getattr(outcome_space, "issues", None)
        if issues is not None:
            for issue in issues:
                values = None
                for attr_name in ("all", "values"):
                    attr = getattr(issue, attr_name, None)
                    if attr is None:
                        continue
                    try:
                        values = list(attr() if callable(attr) else attr)
                        break
                    except Exception:
                        pass
                if values is None:
                    try:
                        values = list(issue)
                    except Exception:
                        values = []
                issue_values.append(values)

        if issue_values and all(len(values) > 0 for values in issue_values):
            return issue_values

        outcomes = enumerate_outcomes(outcome_space)
        if not outcomes:
            return []

        n_issues = len(outcomes[0])
        inferred: list[list[Any]] = []
        for i in range(n_issues):
            try:
                inferred.append(sorted({outcome[i] for outcome in outcomes}))
            except TypeError:
                inferred.append(list(dict.fromkeys(outcome[i] for outcome in outcomes)))
        return inferred

    # Hypothesis construction --------------------------------------------------

    def _build_weight_hypotheses(self) -> None:
        n = self._n_issues
        ranks = list(itertools.permutations(range(1, n + 1)))
        weights = []
        for rank_tuple in ranks:
            weights.append([2.0 * r / (n * (n + 1)) for r in rank_tuple])
        self._rank_hypotheses = ranks
        self._weight_hypotheses = np.asarray(weights, dtype=float)

    def _build_evaluation_hypotheses(self) -> None:
        self._eval_curves = []
        self._eval_names = []

        for values in self._issue_values:
            m = len(values)
            curves = []
            names = []

            if m == 1:
                uphill = np.ones(1, dtype=float)
            else:
                uphill = np.linspace(0.0, 1.0, m)
            curves.append(uphill)
            names.append("uphill")

            curves.append(uphill[::-1].copy())
            names.append("downhill")

            for peak in range(m):
                curve = np.zeros(m, dtype=float)
                for j in range(m):
                    if j <= peak:
                        curve[j] = 1.0 if peak == 0 else j / peak
                    else:
                        curve[j] = 1.0 if peak == m - 1 else 1.0 - (j - peak) / (m - 1 - peak)
                curves.append(np.clip(curve, 0.0, 1.0))
                names.append(f"triangular_peak_{peak}")

            self._eval_curves.append(np.asarray(curves, dtype=float))
            self._eval_names.append(names)

    def _build_evaluation_combinations(self) -> None:
        ranges = [range(len(curves)) for curves in self._eval_curves]
        self._eval_combo_array = np.asarray(list(itertools.product(*ranges)), dtype=int)

    def _initialize_uniform_posterior(self) -> None:
        n_beta = len(self.beta_grid)
        n_weight = len(self._weight_hypotheses)
        n_eval_combo = len(self._eval_combo_array)
        n_states = n_beta * n_weight * n_eval_combo
        self._log_posterior = np.full(
            (n_beta, n_weight, n_eval_combo),
            -math.log(n_states),
            dtype=float,
        )

    # Bayesian update ----------------------------------------------------------

    def on_partner_proposal(self, state, partner_id: str, offer) -> None:
        if not self._initialized:
            self._initialize_from_negotiator()
        t = float(getattr(state, "relative_time", 0.0) or 0.0)
        self.on_offer(offer, t=t)

    def on_offer(self, offer, t: float = 0.0) -> None:
        if not self._initialized:
            self._initialize_from_negotiator()
        if not self._initialized or offer is None:
            return

        offer_tuple = tuple(offer)
        if len(offer_tuple) != self._n_issues:
            return
        if any(offer_tuple[i] not in self._value_to_index[i] for i in range(self._n_issues)):
            return

        tau = min(max(float(t), 0.0), 1.0)
        utility_matrix = self._utility_matrix_for_offer(offer_tuple)

        sigma2 = max(float(self.sigma) ** 2, 1e-12)
        for beta_index, beta in enumerate(self.beta_grid):
            beta = max(float(beta), 1e-12)
            target_u = 1.0 - (1.0 - self.min_utility) * (tau ** (1.0 / beta))
            log_likelihood = -((utility_matrix - target_u) ** 2) / (2.0 * sigma2)
            self._log_posterior[beta_index] += log_likelihood

        self._log_posterior = _safe_normalize_log_probs(self._log_posterior)
        self._total_bids += 1
        self._recompute_expectations()
        self._entropy_history.append(self.posterior_entropy())
        self._beta_posterior_history.append(self.beta_posterior())

    def _utility_matrix_for_offer(self, offer: tuple[Any, ...]) -> np.ndarray:
        value_indices = [self._value_to_index[i][offer[i]] for i in range(self._n_issues)]

        eval_values_by_combo = np.empty((len(self._eval_combo_array), self._n_issues), dtype=float)
        for issue_index, value_index in enumerate(value_indices):
            eval_indices = self._eval_combo_array[:, issue_index]
            eval_values_by_combo[:, issue_index] = self._eval_curves[issue_index][eval_indices, value_index]

        return self._weight_hypotheses @ eval_values_by_combo.T

    def _posterior_probs(self) -> np.ndarray:
        return np.exp(self._log_posterior)

    def _utility_posterior_probs(self) -> np.ndarray:
        return self._posterior_probs().sum(axis=0)

    def _recompute_expectations(self) -> None:
        utility_post = self._utility_posterior_probs()
        self._expected_components = []
        self._expected_eval_cache = []

        for issue_index, values in enumerate(self._issue_values):
            component = np.zeros(len(values), dtype=float)
            eval_curve = np.zeros(len(values), dtype=float)

            weighted_combo_marginal = (
                utility_post * self._weight_hypotheses[:, issue_index][:, None]
            ).sum(axis=0)
            combo_marginal = utility_post.sum(axis=0)

            for eval_index in range(len(self._eval_curves[issue_index])):
                mask = self._eval_combo_array[:, issue_index] == eval_index
                component += weighted_combo_marginal[mask].sum() * self._eval_curves[issue_index][eval_index]
                eval_curve += combo_marginal[mask].sum() * self._eval_curves[issue_index][eval_index]

            self._expected_components.append(component)
            self._expected_eval_cache.append(eval_curve)

        weight_marginal = utility_post.sum(axis=1)
        self._expected_weights_cache = weight_marginal @ self._weight_hypotheses
        self._offer_utility_cache.clear()

    # Utility API used by NegMAS offering components ---------------------------

    def estimated_utility(self, offer) -> float:
        if offer is None:
            return 0.0
        if not self._initialized:
            self._initialize_from_negotiator()
        if not self._initialized:
            return float(self.default_utility_before_observations)
        if self._total_bids == 0:
            return float(self.default_utility_before_observations)

        offer_tuple = tuple(offer)
        cached = self._offer_utility_cache.get(offer_tuple)
        if cached is not None:
            return cached

        if len(offer_tuple) != self._n_issues:
            return float(self.default_utility_before_observations)

        total = 0.0
        for issue_index, value in enumerate(offer_tuple):
            value_index = self._value_to_index[issue_index].get(value)
            if value_index is None:
                return float(self.default_utility_before_observations)
            total += self._expected_components[issue_index][value_index]

        utility = float(np.clip(total, 0.0, 1.0))
        self._offer_utility_cache[offer_tuple] = utility
        return utility

    def eval(self, offer) -> float:
        return self.estimated_utility(offer)

    def eval_normalized(self, offer, above_reserve: bool = True, expected_limits: bool = True) -> float:
        return self.estimated_utility(offer)

    def __call__(self, offer) -> float:
        return self.estimated_utility(offer)

    # Diagnostics --------------------------------------------------------------

    @property
    def n_observed_bids(self) -> int:
        return self._total_bids

    def expected_weights(self) -> np.ndarray:
        if self._expected_weights_cache is None:
            return np.array([])
        return self._expected_weights_cache.copy()

    def expected_evaluation(self, issue_index: int) -> np.ndarray:
        if self._expected_eval_cache is None:
            return np.array([])
        return self._expected_eval_cache[issue_index].copy()

    def beta_posterior(self) -> np.ndarray:
        if self._log_posterior.size == 0:
            return np.array([])
        return self._posterior_probs().sum(axis=(1, 2))

    def posterior_entropy(self) -> float:
        if self._log_posterior.size == 0:
            return float("nan")
        probs = self._posterior_probs().reshape(-1)
        positive = probs[probs > 0.0]
        return float(-np.sum(positive * np.log(positive)))

    def normalized_entropy(self) -> float:
        if self._log_posterior.size == 0:
            return float("nan")
        return self.posterior_entropy() / math.log(self._log_posterior.size)

    def posterior_summary(self) -> dict[str, Any]:
        if self._log_posterior.size == 0:
            return {}

        flat_index = int(np.argmax(self._log_posterior))
        beta_index, weight_index, eval_combo_index = np.unravel_index(
            flat_index, self._log_posterior.shape
        )
        eval_indices = self._eval_combo_array[eval_combo_index].tolist()
        return {
            "n_observed_bids": self._total_bids,
            "entropy": self.posterior_entropy(),
            "normalized_entropy": self.normalized_entropy(),
            "beta_grid": list(map(float, self.beta_grid)),
            "beta_posterior": self.beta_posterior().round(6).tolist(),
            "most_probable_beta": float(self.beta_grid[beta_index]),
            "most_probable_probability": float(np.exp(self._log_posterior[beta_index, weight_index, eval_combo_index])),
            "most_probable_rank_hypothesis": list(self._rank_hypotheses[weight_index]),
            "most_probable_weights": self._weight_hypotheses[weight_index].round(6).tolist(),
            "expected_weights": self.expected_weights().round(6).tolist(),
            "evaluation_indices": eval_indices,
            "evaluation_names": [self._eval_names[i][e] for i, e in enumerate(eval_indices)],
        }
