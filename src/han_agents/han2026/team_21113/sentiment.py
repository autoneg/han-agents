# sentiment.py — Lexicon PE engine + Bayesian manipulation filter
# HERALD HAN 2026 agent — pure Python, zero external calls

from __future__ import annotations
import re

# ── Negotiation-domain lexicon ────────────────────────────────────────────────

_POSITIVE: dict[str, float] = {
    "accept": 0.7, "agree": 0.8, "deal": 0.6, "fair": 0.5, "reasonable": 0.5,
    "happy": 0.8, "pleased": 0.7, "glad": 0.6, "great": 0.7, "excellent": 0.9,
    "good": 0.5, "fine": 0.4, "okay": 0.3, "ok": 0.3, "flexible": 0.6,
    "generous": 0.7, "cooperative": 0.7, "appreciate": 0.6, "thank": 0.5,
    "welcome": 0.5, "understand": 0.4, "consider": 0.3, "progress": 0.5,
    "compromise": 0.6, "mutual": 0.5, "benefit": 0.5, "constructive": 0.6,
    "promising": 0.6, "positive": 0.7, "together": 0.4, "close": 0.4,
    "almost": 0.3, "nearly": 0.3, "workable": 0.5, "satisfactory": 0.6,
}

_NEGATIVE: dict[str, float] = {
    "reject": -0.7, "disagree": -0.6, "unacceptable": -0.9, "unfair": -0.8,
    "disappointed": -0.7, "frustrated": -0.8, "unhappy": -0.8, "upset": -0.7,
    "angry": -0.9, "absurd": -0.8, "ridiculous": -0.8, "unreasonable": -0.8,
    "impossible": -0.7, "terrible": -0.9, "bad": -0.5, "poor": -0.5,
    "low": -0.3, "insult": -0.8, "offensive": -0.7, "waste": -0.5,
    "cannot": -0.4, "refuse": -0.7, "walk": -0.5, "away": -0.3,
    "deadline": -0.3, "pressure": -0.4, "forced": -0.5, "stuck": -0.4,
    "outrageous": -0.9, "pathetic": -0.7,
}

_INTENSIFIERS: frozenset[str] = frozenset({
    "very", "extremely", "quite", "really", "absolutely", "completely",
    "totally", "utterly", "highly", "deeply", "so", "incredibly",
})

_NEGATORS: frozenset[str] = frozenset({
    "not", "no", "never", "don't", "cannot", "can't", "won't",
    "isn't", "aren't", "wasn't", "weren't", "doesn't", "didn't",
    "hardly", "barely", "scarcely",
})


def lexicon_score(text: str) -> float:
    """
    Pure-Python lexicon sentiment. Returns float in [-1, +1].

    Applies:
    - Intensifiers (prev token): multiply score × 1.4
    - Negators (within prev 3 tokens): flip sign × -0.8
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    score = 0.0
    count = 0
    for i, tok in enumerate(tokens):
        val = _POSITIVE.get(tok) or _NEGATIVE.get(tok)
        if val is None:
            continue

        negated = any(
            tokens[max(0, i - j)] in _NEGATORS for j in range(1, min(4, i + 1))
        )
        intensified = i > 0 and tokens[i - 1] in _INTENSIFIERS

        if intensified:
            val *= 1.4
        if negated:
            val *= -0.8

        score += max(-1.0, min(1.0, val))
        count += 1

    if count == 0:
        return 0.0
    return max(-1.0, min(1.0, score / count))


class LexiconPE:
    """
    Computes a manipulation-filtered emotion coefficient PE ∈ [-1, +1].

    Two components:
    - pe_raw:  lexicon score for this round's message
    - pe_traj: linear regression slope over last k PE values (mood trend)

    Blended as: PE = t² × pe_raw + (1−t²) × pe_traj
    Then filtered through a Bayesian manipulation detector.
    """

    def __init__(self, window: int = 5) -> None:
        self._k = window
        self._history: list[float] = []
        self._offer_history: list[float] = []
        self._p_genuine: float = 0.7

    def update(self, text: str, opp_offer_util: float, t: float) -> float:
        """
        Compute filtered PE for this round.

        Args:
            text:           Opponent's text message this round.
            opp_offer_util: HERALD's utility for the opponent's offer.
            t:              Normalised negotiation time [0, 1].

        Returns:
            PE_filtered ∈ [-1, +1]
        """
        pe_raw = lexicon_score(text)
        self._history.append(pe_raw)
        self._offer_history.append(opp_offer_util)

        # Trajectory: linear regression slope over last k PE values
        if len(self._history) >= 3:
            window = self._history[-self._k:]
            n = len(window)
            xs = list(range(n))
            mx = sum(xs) / n
            my = sum(window) / n
            num = sum((xs[i] - mx) * (window[i] - my) for i in range(n))
            den = sum((xs[i] - mx) ** 2 for i in range(n)) or 1e-9
            pe_traj = max(-1.0, min(1.0, num / den))
        else:
            pe_traj = pe_raw

        gamma = t ** 2
        pe_blended = gamma * pe_raw + (1.0 - gamma) * pe_traj

        return self._apply_filter(pe_blended, opp_offer_util)

    def _apply_filter(self, pe_blended: float, offer_util: float) -> float:
        """Bayesian manipulation filter: attenuate PE when strategic use suspected."""
        if len(self._offer_history) >= 2:
            k = min(self._k, len(self._offer_history))
            offer_trend = self._offer_history[-1] - self._offer_history[-k]
        else:
            offer_trend = 0.0

        # Negative words + improving offer = likely strategic
        if pe_blended < -0.3 and offer_trend > 0.05:
            likelihood = 0.3
        elif pe_blended < -0.3:
            likelihood = 0.8
        else:
            likelihood = 0.6

        prior = self._p_genuine
        evidence = prior * likelihood + (1.0 - prior) * (1.0 - likelihood)
        if evidence > 0.0:
            self._p_genuine = (prior * likelihood) / evidence
        self._p_genuine = max(0.1, min(0.9, self._p_genuine))

        pe_filtered = pe_blended * self._p_genuine
        return max(-1.0, min(1.0, pe_filtered))

    @property
    def trajectory_slope(self) -> float:
        """Slope of PE over the last k values (+ve = improving mood)."""
        if len(self._history) < 2:
            return 0.0
        w = self._history[-self._k:]
        n = len(w)
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(w) / n
        num = sum((xs[i] - mx) * (w[i] - my) for i in range(n))
        den = sum((xs[i] - mx) ** 2 for i in range(n)) or 1e-9
        return num / den

    @property
    def current_pe(self) -> float:
        """Most recent raw PE value, or 0.0 if no data."""
        return self._history[-1] if self._history else 0.0
