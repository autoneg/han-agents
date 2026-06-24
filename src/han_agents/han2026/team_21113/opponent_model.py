# opponent_model.py — Hybrid three-layer opponent preference model
# HERALD HAN 2026 agent
#
# Three internal sub-models fused via adaptive ensemble weighting:
#   1. FrequencyModel      — Laplace-smoothed value frequency counts
#   2. HardHeadedModel     — weight stable (unchanged) issues higher
#   3. BayesianLayer       — Dirichlet posterior, entropy-based issue salience
#
# No NegMAS built-ins used here — all three are pure-Python implementations
# of the same algorithms, allowing standalone use without a live negotiator.

from __future__ import annotations
import math
from typing import Optional

from negmas import Outcome


def _get_val(offer: Outcome, idx: int, name: str) -> object:
    """Extract issue value from an outcome by integer index, then by name."""
    try:
        return offer[idx]
    except (IndexError, TypeError, KeyError):
        pass
    try:
        return offer[name]
    except (KeyError, TypeError):
        pass
    return None


# ── Sub-model 1: Simple frequency model ──────────────────────────────────────

class _FrequencyModel:
    """
    Tracks how often each value is seen per issue.
    Value weights: Laplace-smoothed + γ-dampened frequency counts.
    Issue weights: uniform (this model doesn't estimate issue importance).
    """

    def __init__(self, gamma: float = 0.25) -> None:
        self._gamma = gamma
        self._issues: list = []
        self._issue_names: list[str] = []
        self._value_counts: list[dict] = []   # per issue: {val: count}
        self._value_weights: list[dict] = []  # per issue: {val: weight in [0,1]}

    def init(self, issues: list, issue_values: list[list]) -> None:
        self._issues = issues
        self._issue_names = [iss.name for iss in issues]
        self._value_counts = [{v: 0 for v in vals} for vals in issue_values]
        self._value_weights = [{v: 1.0 for v in vals} for vals in issue_values]

    def update(self, offer: Outcome) -> None:
        for i, (name, counts) in enumerate(
            zip(self._issue_names, self._value_counts)
        ):
            val = _get_val(offer, i, name)
            if val in counts:
                counts[val] += 1
                self._recompute_weights(i)

    def _recompute_weights(self, i: int) -> None:
        counts = self._value_counts[i]
        max_smoothed = max((1 + c) ** self._gamma for c in counts.values()) or 1.0
        self._value_weights[i] = {
            v: (1 + c) ** self._gamma / max_smoothed
            for v, c in counts.items()
        }

    def eval(self, offer: Outcome) -> float:
        if not self._issues:
            return 0.5
        n = len(self._issues)
        total = 0.0
        for i, name in enumerate(self._issue_names):
            val = _get_val(offer, i, name)
            w = self._value_weights[i].get(val, 0.5) if val is not None else 0.5
            total += w / n
        return total


# ── Sub-model 2: Hard-headed frequency model ──────────────────────────────────

class _HardHeadedModel:
    """
    Tracks which issues the opponent does NOT change between consecutive offers.
    Unchanged issues are assumed to be important → their weight increases.
    Learning coefficient controls how fast issue weights adapt.
    """

    def __init__(self, learning_coef: float = 0.2, value_addition: int = 1) -> None:
        self._lc = learning_coef
        self._va = value_addition
        self._issue_names: list[str] = []
        self._issue_weights: dict[int, float] = {}
        self._value_freqs: dict[int, dict] = {}   # {issue_idx: {val: count}}
        self._last_offer: Optional[Outcome] = None
        self._n: int = 0

    def init(self, issues: list, issue_values: list[list]) -> None:
        self._issue_names = [iss.name for iss in issues]
        self._n = len(issues)
        if self._n > 0:
            w = 1.0 / self._n
            self._issue_weights = {i: w for i in range(self._n)}
        self._value_freqs = {i: {} for i in range(self._n)}
        self._last_offer = None

    def update(self, offer: Outcome) -> None:
        if self._n == 0:
            return

        # Count value frequencies
        for i, name in enumerate(self._issue_names):
            val = _get_val(offer, i, name)
            if val is not None:
                self._value_freqs[i][val] = self._value_freqs[i].get(val, 0) + 1

        # Boost weight of unchanged issues (they matter to opponent)
        if self._last_offer is not None:
            unchanged = []
            for i, name in enumerate(self._issue_names):
                if _get_val(offer, i, name) == _get_val(self._last_offer, i, name):
                    unchanged.append(i)

            if unchanged:
                addition = self._lc * len(unchanged) / len(unchanged)
                for i in unchanged:
                    self._issue_weights[i] += addition

                # Normalize to sum = 1
                total = sum(self._issue_weights.values())
                if total > 0:
                    for i in self._issue_weights:
                        self._issue_weights[i] /= total

        self._last_offer = offer

    def eval(self, offer: Outcome) -> float:
        if self._n == 0:
            return 0.5
        total = 0.0
        for i, name in enumerate(self._issue_names):
            val = _get_val(offer, i, name)
            iw = self._issue_weights.get(i, 1.0 / max(1, self._n))
            freqs = self._value_freqs.get(i, {})
            vw = freqs.get(val, 0) if val is not None else 0
            max_vw = max(freqs.values(), default=1)
            norm_vw = vw / max_vw if max_vw > 0 else 1.0
            total += iw * norm_vw
        return total


# ── Sub-model 3: Bayesian Dirichlet layer ────────────────────────────────────

class _BayesianLayer:
    """
    Maintains a Dirichlet posterior over value counts per issue.
    Issue importance is estimated via entropy: low entropy = opponent cares
    (they keep choosing the same value consistently).
    """

    def __init__(self) -> None:
        self._issue_names: list[str] = []
        self._issue_values: list[list] = []
        self._dirichlet: list[dict] = []   # per issue: {val: alpha count}

    def init(self, issues: list, issue_values: list[list]) -> None:
        self._issue_names = [iss.name for iss in issues]
        self._issue_values = [list(vals) for vals in issue_values]
        # Uniform Dirichlet prior: α=1 per value
        self._dirichlet = [{v: 1 for v in vals} for vals in issue_values]

    def update(self, offer: Outcome) -> None:
        for i, name in enumerate(self._issue_names):
            val = _get_val(offer, i, name)
            if val in self._dirichlet[i]:
                self._dirichlet[i][val] += 1

    def eval(self, offer: Outcome) -> float:
        if not self._issue_names:
            return 0.5

        issue_scores = []
        saliences = []

        for i, name in enumerate(self._issue_names):
            counts = self._dirichlet[i]
            total = sum(counts.values())
            if total == 0:
                issue_scores.append(0.5)
                saliences.append(1.0)
                continue

            # Posterior probability of the observed value (Laplace smoothed)
            val = _get_val(offer, i, name)
            p_val = (counts.get(val, 0) + 1e-9) / (total + 1e-9) if val is not None else 0.5

            # Entropy of the distribution: low entropy = opponent repeats same values
            probs = [c / total for c in counts.values()]
            entropy = -sum(p * math.log(p + 1e-9) for p in probs)
            max_ent = math.log(len(counts)) if len(counts) > 1 else 1.0
            salience = 1.0 - (entropy / (max_ent + 1e-9))   # high = opponent guards issue

            issue_scores.append(p_val)
            saliences.append(max(0.0, salience))

        total_sal = sum(saliences) or 1.0
        result = sum(
            issue_scores[i] * saliences[i] / total_sal
            for i in range(len(issue_scores))
        )
        return max(0.0, min(1.0, result))


# ── Main class: HybridOpponentModel ──────────────────────────────────────────

class HybridOpponentModel:
    """
    Ensemble of three (optionally four) sub-models with data-adaptive weighting.

    Weight evolution:
    - Early (few offers): Bayesian prior trusted more (regularises with little data)
    - Late (many offers): frequency models dominate (converged on real observations)

    A fourth sub-model (_gnash) can be attached externally from NEXUSNegotiator
    if the NegMAS GNashFrequencyModel is available. It contributes a 15% blend.
    """

    def __init__(self, ufun) -> None:
        self._freq = _FrequencyModel(gamma=0.25)
        self._hh = _HardHeadedModel(learning_coef=0.2, value_addition=1)
        self._bayes = _BayesianLayer()
        self._gnash = None   # attached externally if GNashFrequencyModel is available
        self._offer_count: int = 0
        self._initialized: bool = False
        # RVFitter state
        self._offer_times: list[float] = []
        self._offer_opp_utils: list[float] = []
        self._est_rv: float = 0.0
        self._rv_fitted: bool = False
        try:
            self._init_from_ufun(ufun)
        except Exception:
            pass

    def _init_from_ufun(self, ufun) -> None:
        issues = list(ufun.issues)
        issue_values = [list(iss.all) for iss in issues]
        self._freq.init(issues, issue_values)
        self._hh.init(issues, issue_values)
        self._bayes.init(issues, issue_values)
        self._initialized = True

    def reinit(self, ufun) -> None:
        """Reinitialize with a new utility function (call on new negotiation)."""
        self._offer_count = 0
        self._initialized = False
        self._offer_times = []
        self._offer_opp_utils = []
        self._est_rv = 0.0
        self._rv_fitted = False
        try:
            self._init_from_ufun(ufun)
        except Exception:
            pass

    def update(self, offer: Outcome, t: float) -> None:
        """Process a new opponent offer."""
        if not self._initialized or offer is None:
            return
        try:
            self._freq.update(offer)
            self._hh.update(offer)
            self._bayes.update(offer)
            self._offer_count += 1

            # RVFitter: track (t, estimated_opp_util_of_their_offer)
            opp_util_est = self._bayes.eval(offer)
            self._offer_times.append(t)
            self._offer_opp_utils.append(opp_util_est)
            if len(self._offer_times) >= 10 and len(self._offer_times) % 5 == 0:
                self._fit_opponent_rv()
        except Exception:
            pass

    def _fit_opponent_rv(self) -> None:
        """
        RVFitter (ANL 2024): estimate the opponent's reservation value by fitting
        their offer history to an aspiration curve:
            u(t) = (u0 - rv)(1 - t^e) + rv
        where u0 = first offer's estimated utility (opponent's starting aspiration).

        Uses scipy.optimize.curve_fit with bounded parameters.
        Silently keeps previous estimate on any failure.
        """
        try:
            from scipy.optimize import curve_fit
            import numpy as np

            times = np.array(self._offer_times, dtype=float)
            utils = np.array(self._offer_opp_utils, dtype=float)
            if len(times) < 8:
                return

            u0 = float(utils[0]) if utils[0] > 0.0 else 0.9

            def aspiration(t, e, rv):
                return (u0 - rv) * (1.0 - np.power(np.clip(t, 0.0, 1.0), e)) + rv

            min_u = float(np.min(utils))
            if min_u <= 0.0:
                min_u = 0.0
            rv_upper = max(0.0, min_u - 1e-6)
            if rv_upper <= 0.0:
                return   # no valid range — all utilities ≈ 0

            bounds = ((0.2, 0.0), (5.0, rv_upper))
            popt, _ = curve_fit(
                aspiration, times, utils,
                p0=[1.0, rv_upper * 0.5],
                bounds=bounds,
                maxfev=200,
            )
            self._est_rv = max(0.0, min(0.5, float(popt[1])))
            self._rv_fitted = True
        except Exception:
            pass   # keep previous estimate, never crash

    @property
    def estimated_opponent_rv(self) -> float:
        return self._est_rv

    def get_estimated_utility(self, outcome: Outcome) -> float:
        """
        Ensemble estimate of the opponent's utility for a given outcome.
        Weights adapt with data volume.
        If _gnash is attached, its estimate is blended in at 15% weight.
        """
        if not self._initialized or outcome is None:
            return 0.5

        try:
            t_weight = min(1.0, self._offer_count / 20.0)
            w_freq = 0.35 + 0.15 * t_weight
            w_hh   = 0.25 + 0.10 * t_weight
            w_b    = max(0.0, 1.0 - w_freq - w_hh)

            freq_u  = self._freq.eval(outcome)
            hh_u    = self._hh.eval(outcome)
            bayes_u = self._bayes.eval(outcome)

            result = w_freq * freq_u + w_hh * hh_u + w_b * bayes_u

            # Blend GNash if available (15% weight, scaled down from the 3-model total)
            if self._gnash is not None:
                try:
                    gnash_u = float(self._gnash.eval(outcome))
                    result = 0.85 * result + 0.15 * gnash_u
                except Exception:
                    pass

            return max(0.0, min(1.0, result))
        except Exception:
            return 0.5
