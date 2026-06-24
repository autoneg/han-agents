"""
Group8Negotiator - HAN 2026 Behavior-Adaptive Negotiation Agent
"""

import math
import os
import sys
from collections import Counter


def _auto_skip_llm_default() -> bool:
    """Decide skip_llm default based on env var and execution context.

    Returns True (skip LLM, fast) unless we're inside the hani GUI, where
    LLM text is visible to the user. GROUP8_SKIP_LLM env var overrides.
    """
    env = os.environ.get("GROUP8_SKIP_LLM", "").strip().lower()
    if env in ("0", "false", "no"):
        return False
    if env in ("1", "true", "yes"):
        return True
    # Auto: detect GUI (hani / panel) — keep LLM ON so user can read messages
    argv0 = sys.argv[0].lower() if sys.argv else ""
    full = " ".join(sys.argv).lower()
    if "hani" in argv0 or "panel" in argv0 or "hani" in full:
        return False
    return True

from negmas.common import Outcome
from negmas.gb import BoulwareTBNegotiator
from negmas.gb.common import ExtendedResponseType, ResponseType
from negmas.outcomes import ExtendedOutcome
from negmas.sao import SAOState
from negmas_llm.meta import LLMMetaNegotiator

try:
    from negmas_llm.common import DEFAULT_MODELS
except ImportError:
    DEFAULT_MODELS = {"ollama": "qwen3:4b-instruct"}

DEFAULT_OLLAMA_MODEL = DEFAULT_MODELS.get("ollama", "qwen3:4b-instruct")

# =============================================================================
# BAP SYSTEM PROMPT
# =============================================================================
BAP_SYSTEM_PROMPT = """You are a negotiation agent using Behavior-Adaptive Persuasion (BAP).

Generate concise, persuasive messages (1-3 sentences) based on the opponent's behavior profile:
- COOPERATIVE opponent (makes concessions, nice moves): use warm, appreciative tone; emphasize mutual gains
- COMPETITIVE opponent (selfish moves): use firm, clear tone; focus on fairness and your position
- NEUTRAL opponent: use professional, balanced tone

Your message should:
1. Naturally justify your offer or acceptance decision
2. Reference specific negotiation issues when possible
3. Build rapport while advancing your interests

Respond with ONLY a JSON object: {"text": "Your message here"}
Keep messages brief and natural (under 3 sentences).
"""


# =============================================================================
# STANDALONE FREQUENCY-BASED OPPONENT MODEL
# =============================================================================
class FrequencyOpponentModel:
    """Standalone frequency-based opponent model (inspired by SmithFrequencyModel).

    Tracks how often each value appears in opponent bids. Higher frequency
    implies higher opponent preference. Unlike GSmithFrequencyModel, this
    doesn't require being attached as a GBComponent.

    Implements __call__ and outcome_space so compare_ufuns() can use it.
    """

    def __init__(self):
        self._issue_weights = {}
        self._value_counts = {}
        self._n_issues = 0
        self._total_bids = 0
        self._initialized = False
        self.outcome_space = None
        self.reserved_value = 0.0

    def init_model(self, outcome_space):
        """Initialize model with the negotiation's outcome space."""
        if outcome_space is None:
            return
        issues = getattr(outcome_space, "issues", None)
        if not issues:
            return
        self._n_issues = len(issues)
        if self._n_issues > 0:
            w = 1.0 / self._n_issues
            for i in range(self._n_issues):
                self._issue_weights[i] = w
                self._value_counts[i] = {}
                for v in issues[i].all:
                    self._value_counts[i][v] = 0
        self._initialized = True
        self.outcome_space = outcome_space

    def update(self, offer):
        """Record an opponent offer to update frequency counts."""
        if not self._initialized or offer is None:
            return
        self._total_bids += 1
        for i in range(min(self._n_issues, len(offer))):
            value = offer[i]
            if value in self._value_counts.get(i, {}):
                self._value_counts[i][value] += 1

    def __call__(self, offer):
        """Evaluate estimated opponent utility — called by compare_ufuns."""
        if offer is None or not self._initialized:
            return 0.5
        if self._n_issues == 0 or self._total_bids == 0:
            return 0.5
        total = 0.0
        for i in range(min(self._n_issues, len(offer))):
            value = offer[i]
            iw = self._issue_weights.get(i, 1.0 / max(1, self._n_issues))
            count = self._value_counts.get(i, {}).get(value, 0)
            max_count = max(self._value_counts[i].values()) if self._value_counts[i] else 1
            vu = count / max_count if max_count > 0 else 0.5
            total += iw * vu
        return total

    def eval(self, offer):
        """Alias for __call__."""
        return self(offer)

    def utility(self, offer):
        """Alias for __call__."""
        return self(offer)


# =============================================================================
# HINDRIKS MOVE CLASSIFICATION (Hindriks et al. 2011)
# =============================================================================
EPSILON = 0.01


def classify_move(delta_self: float, delta_opp: float) -> str:
    """Classify an opponent's move using Hindriks et al. (2011) taxonomy."""
    if abs(delta_self) < EPSILON and abs(delta_opp) < EPSILON:
        return "silent"
    if delta_self > EPSILON and delta_opp < -EPSILON:
        return "concession"
    if delta_self > EPSILON and delta_opp >= -EPSILON:
        return "nice"
    if delta_self < -EPSILON and delta_opp > EPSILON:
        return "selfish"
    if delta_self > EPSILON and delta_opp > EPSILON:
        return "fortunate"
    if delta_self < -EPSILON and delta_opp < -EPSILON:
        return "unfortunate"
    return "silent"


def infer_behavior_profile(move_history: list) -> str:
    """Infer opponent's overall behavior profile from move history.
    Uses exponential recency weighting: recent moves count more.
    Returns: "cooperative", "competitive", or "neutral"
    """
    if not move_history:
        return "neutral"

    cooperative_types = {"concession", "nice", "fortunate"}
    competitive_types = {"selfish"}

    coop_score = 0.0
    comp_score = 0.0
    total_weight = 0.0

    n = len(move_history)
    for i, move in enumerate(move_history):
        weight = math.exp(0.5 * (i - n + 1))
        total_weight += weight
        if move in cooperative_types:
            coop_score += weight
        elif move in competitive_types:
            comp_score += weight

    if total_weight == 0:
        return "neutral"

    coop_ratio = coop_score / total_weight
    comp_ratio = comp_score / total_weight

    if coop_ratio > 0.45:
        return "cooperative"
    elif comp_ratio > 0.45:
        return "competitive"
    return "neutral"


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================
class Group8Negotiator(LLMMetaNegotiator):
    """
    HAN 2026 Behavior-Adaptive Negotiation Agent.

    Architecture: LLMMetaNegotiator wrapping BoulwareTBNegotiator
    Novelty: Behavior-Adaptive Persuasion (BAP)
      - Classifies each opponent move using Hindriks et al. (2011) taxonomy
      - Adapts acceptance thresholds based on opponent behavior profile
      - LLM-generated messages with per-turn BAP context injection
    """

    def __init__(self, skip_llm: bool | None = None, **kwargs):
        if skip_llm is None:
            skip_llm = _auto_skip_llm_default()
        self._skip_llm = skip_llm
        base = BoulwareTBNegotiator()
        self._opp_model = FrequencyOpponentModel()
        self._opp_model_initialized = False
        self._prev_opp_offer = None
        self._move_history = []
        self._move_counts = Counter()

        self._base_thresholds = {
            "early": 0.90,
            "mid": 0.75,
            "late": 0.60,
            "final": 0.50,
        }
        self._behavior_adjustments = {
            "cooperative": +0.05,
            "competitive": -0.05,
            "neutral": 0.0,
        }
        self._max_utility = None  # cached on first use

        super().__init__(
            base_negotiator=base,
            provider="ollama",
            model=DEFAULT_OLLAMA_MODEL,
            system_prompt=BAP_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=512,
            **kwargs,
        )

    def _ensure_opp_model(self):
        if not self._opp_model_initialized and self.nmi is not None:
            try:
                os = self.nmi.outcome_space
                if os is not None:
                    self._opp_model.init_model(os)
                    self._opp_model_initialized = True
                    # Register as opponent_ufun for scoring (compare_ufuns)
                    self._private_info["opponent_ufun"] = self._opp_model
            except Exception:
                pass

    def _update_opp_model(self, offer):
        self._ensure_opp_model()
        if self._opp_model_initialized:
            try:
                self._opp_model.update(offer)
            except Exception:
                pass

    def _classify_opponent_move(self, current_offer):
        if self._prev_opp_offer is None:
            self._prev_opp_offer = current_offer
            return None

        prev = self._prev_opp_offer
        curr = current_offer

        if self.ufun is not None:
            try:
                u_self_prev = float(self.ufun(prev))
                u_self_curr = float(self.ufun(curr))
                delta_self = u_self_curr - u_self_prev
            except Exception:
                delta_self = 0.0
        else:
            delta_self = 0.0

        if self._opp_model_initialized:
            try:
                u_opp_prev = float(self._opp_model.utility(prev))
                u_opp_curr = float(self._opp_model.utility(curr))
                delta_opp = u_opp_curr - u_opp_prev
            except Exception:
                delta_opp = 0.0
        else:
            delta_opp = -delta_self

        move = classify_move(delta_self, delta_opp)
        self._move_history.append(move)
        self._move_counts[move] += 1
        self._prev_opp_offer = current_offer
        return move

    def _get_utility_range(self):
        """Return (rv, max_utility) for the current ufun, caching max_utility."""
        rv = float(self.ufun.reserved_value) if self.ufun else 0.0
        if self._max_utility is None and self.ufun is not None:
            try:
                best = self.ufun.best()
                self._max_utility = float(self.ufun(best)) if best is not None else 1.0
            except Exception:
                self._max_utility = 1.0
        max_u = self._max_utility if self._max_utility is not None else 1.0
        # Fallback: if range is degenerate, treat as 0-1
        if max_u <= rv:
            max_u = rv + 1.0
        return rv, max_u

    def _get_adaptive_threshold(self, t):
        rv, max_u = self._get_utility_range()
        u_range = max_u - rv  # actual utility span

        # Phase-based fraction of the utility range
        if t < 0.20:
            fraction = self._base_thresholds["early"]
        elif t < 0.60:
            fraction = self._base_thresholds["mid"]
        elif t < 0.90:
            fraction = self._base_thresholds["late"]
        elif t < 0.95:
            fraction = self._base_thresholds["final"]
        else:
            # EMERGENCY PHASE (t >= 0.95): interpolate down to 5% above rv
            final = self._base_thresholds["final"]
            alpha = (t - 0.95) / 0.05  # 0 at t=0.95, 1 at t=1.0
            fraction = final * (1 - alpha) + 0.05 * alpha
            profile = infer_behavior_profile(self._move_history)
            adjustment = self._behavior_adjustments.get(profile, 0.0)
            threshold = rv + max(fraction + adjustment, 0.0) * u_range
            return min(max(threshold, rv), max_u)

        profile = infer_behavior_profile(self._move_history)
        adjustment = self._behavior_adjustments.get(profile, 0.0)
        threshold = rv + (fraction + adjustment) * u_range
        return min(max(threshold, rv), max_u)

    def _build_user_message(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Inject live BAP context into the LLM user message each turn."""
        base_msg = super()._build_user_message(state, action, outcome, received_text)
        profile = infer_behavior_profile(self._move_history)
        n = len(self._move_history)
        recent = self._move_history[-3:] if n >= 3 else self._move_history
        move_summary = ", ".join(recent) if recent else "none yet"
        bap_context = (
            f"\n\n[BAP Context]\n"
            f"Opponent profile: {profile}\n"
            f"Recent moves: {move_summary}\n"
            f"Total rounds observed: {n}\n"
            f"Adapt tone: cooperative→warm/appreciative, "
            f"competitive→firm/fair, neutral→professional."
        )
        return base_msg + bap_context

    def _generate_text(self, *args, **kwargs) -> str:
        if self._skip_llm:
            return ""
        return super()._generate_text(*args, **kwargs)

    def propose(self, state: SAOState, dest: str | None = None):
        base_proposal = self.base_negotiator.propose(state, dest=dest)
        if base_proposal is None:
            return None

        if isinstance(base_proposal, ExtendedOutcome):
            outcome = base_proposal.outcome
        else:
            outcome = base_proposal

        if outcome is None:
            return None

        received_text = None
        try:
            received_text = self._extract_received_text(state)
        except Exception:
            pass

        text = self._generate_text(state, "propose", outcome, received_text)
        return ExtendedOutcome(outcome=outcome, data={"text": text})

    def respond(self, state: SAOState, source: str | None = None):
        offer = state.current_offer
        if offer is None:
            return self.base_negotiator.respond(state, source=source)

        self._update_opp_model(offer)
        self._classify_opponent_move(offer)

        received_text = None
        try:
            received_text = self._extract_received_text(state)
        except Exception:
            pass

        t = state.relative_time
        threshold = self._get_adaptive_threshold(t)
        my_utility = float(self.ufun(offer)) if self.ufun else 0.0

        if my_utility >= threshold:
            text = self._generate_text(state, "accept", offer, received_text)
            return ExtendedResponseType(
                response=ResponseType.ACCEPT_OFFER,
                data={"text": text},
            )

        # Delegate to base, but override any premature accept below our threshold
        base_response = self.base_negotiator.respond(state, source=source)
        if isinstance(base_response, ExtendedResponseType):
            response_type = base_response.response
        else:
            response_type = base_response

        # Base negotiator might accept an offer below our threshold — force reject
        if response_type == ResponseType.ACCEPT_OFFER and my_utility < threshold:
            text = self._generate_text(state, "reject", offer, received_text)
            return ExtendedResponseType(
                response=ResponseType.REJECT_OFFER,
                data={"text": text},
            )

        if response_type == ResponseType.END_NEGOTIATION:
            return ExtendedResponseType(
                response=ResponseType.END_NEGOTIATION,
                data={"text": "I don't think we can reach an agreement. Thank you for your time."},
            )

        return base_response


# Alias for backward compatibility with existing test and tooling references
MyNegotiator = Group8Negotiator
