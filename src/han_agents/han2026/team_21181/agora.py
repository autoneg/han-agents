"""AgoraNegotiator: HAN 2026 hybrid agent.

Architecture:
- Base strategy: Boulware (tough, time-based concession) generates the
  aspiration curve and fallback offers. We override acceptance and offer
  ranking with custom logic.
- Custom acceptance: time-pressure + opponent concession rate + text
  signals + reservation-value floor.  Stock Boulware acceptance is
  bypassed entirely.
- Opponent model: per-issue frequency analysis of opponent offers infers
  which issues the partner values most. Used to find win-win swaps.
- Text belief state: keyword-based parsing of human messages detects
  urgency, threats, concession signals, and priority issues. These
  nudge the acceptance threshold.
- LLM text layer: qwen3:4b-instruct (via Ollama) generates persuasive
  messages that accompany every offer/response. The LLM never touches
  the decision logic directly.
- Adaptive text phase: persuasion (default) <-> deception (firm).
"""

from __future__ import annotations

from typing import Any

import logging
import os
from collections import deque

import attrs

from negmas.gb.common import ExtendedResponseType, ResponseType
from negmas.outcomes.common import ExtendedOutcome, Outcome
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import TableFun
from negmas.sao.common import SAOState
from negmas_llm.meta import LLMBoulwareTBNegotiator

# Suppress litellm's verbose error output (provider-list + debug hints
# printed on every exception). Errors are already caught in try/except
# blocks in our code — the extra logging just clutters stdout/stderr.
os.environ.setdefault("LITELLM_LOG", "ERROR")
try:
    import litellm as _litellm
    _litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)
except Exception:
    pass

# =============================================================================
# Configuration
# =============================================================================

# Phase switching thresholds (text tone).
CONCESSION_THRESHOLD_ROUNDS = 5  # raised from 3 — don't go firm too early against humans
CONCESSION_EPSILON = 0.01
RECOVERY_THRESHOLD_ROUNDS = 2

# Acceptance policy.
BOULWARE_EXPONENT = 3.0  # lower than stock 4.0 = slightly more flexible
EARLY_ACCEPT_THRESHOLD = 0.85  # accept immediately if util > this and t < 0.3
LATE_GAME_START = 0.85  # after this, accelerate concession
RESERVATION_BUFFER = 0.05  # floor above reservation value

# Opponent model.
MIN_OFFERS_FOR_MODEL = 3  # need this many offers before inference
WIN_WIN_UTIL_TOLERANCE = 0.03  # max util loss we accept for a win-win swap

# Text signal weights (kept small so text nudges, not dominates).
TEXT_URGENCY_WEIGHT = 0.05
TEXT_THREAT_WEIGHT = 0.03
TEXT_CONCESSION_WEIGHT = 0.03
TEXT_SIGNAL_DECAY = 0.7  # exponential decay per round for threat/concession

# =============================================================================
# Prompts
# =============================================================================

PERSUASION_SYSTEM_PROMPT = """You are a negotiator talking to a real person.

STYLE: 2 sentences max. Warm, human, conversational. Reference issues by name.
Never start two messages the same way. Never reveal your reservation value.
Never lie about offer values. No threats or begging.

GOAL: the partner rates you on fairness, friendliness, trustworthiness. Make
them feel heard; frame offers as collaborative.

Respond with ONLY: {"text": "your message"}
"""

DECEPTION_SYSTEM_PROMPT = """You are a negotiator whose partner has been stubborn. Be firmer.

STYLE: 2 sentences max. Confident, measured, professional. Signal real limits
without stating false numbers; keep the door open ("if you can move on X,
we can close this"). Never reveal reservation value; only say "final offer"
if it is actually the last round.

Respond with ONLY: {"text": "your message"}
"""

# Text-signal keyword sets (fallback when LLM parsing fails).
_URGENCY_UP = frozenset([
    "need this done", "running out of time", "deadline", "urgent",
    "quickly", "hurry", "asap", "last chance", "final",
])
_URGENCY_DOWN = frozenset([
    "no rush", "take your time", "plenty of time", "patient",
])
_THREAT_WORDS = frozenset([
    "walk away", "no deal", "find someone else", "other option",
    "alternative", "not worth", "forget it", "waste of time",
])
_CONCESSION_WORDS = frozenset([
    "willing to", "can come down", "flexible", "meet halfway",
    "compromise", "adjust", "open to", "fair enough",
])
# Words that, when combined with an EXTREME offer, signal manipulation.
_FAIRNESS_CLAIM_WORDS = frozenset([
    "fair", "reasonable", "best i can", "my final", "take it or",
    "this is generous", "you should", "only offer",
])

# C2 Prompt-injection guard: phrases a hostile partner might use to
# hijack our LLM calls. Stripped defensively before feeding partner text
# to cold-read / personality classifier / signal parser.
_INJECTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "disregard all previous",
    "forget previous",
    "forget your instructions",
    "override your instructions",
    "system prompt",
    "you are now",
    "act as if",
    "pretend you are",
    "new instruction",
    "reset your",
    "</system>",
    "<system>",
    "```system",
    # #70 tool-call-style markers — no LLM in HAN emits these, but a
    # hostile partner could paste them to confuse our cold-read/classifier.
    "<tool_call>",
    "</tool_call>",
    "function_call:",
    "```json",
    "```python",
)

# #4 Tone + length keyword fallbacks (used when LLM parsing fails).
_FRIENDLY_WORDS = frozenset([
    "thanks", "thank you", "appreciate", "glad", "happy", "wonderful",
    "great", "love", "please", "kindly", "hope", "fantastic",
])
_HOSTILE_WORDS = frozenset([
    "ridiculous", "insulting", "joke", "unacceptable", "absurd",
    "never", "unreasonable", "refuse", "demand", "insult",
])
_FORMAL_WORDS = frozenset([
    "therefore", "however", "nevertheless", "accordingly", "regarding",
    "proposal", "kindly", "furthermore", "hereby",
])
_CASUAL_WORDS = frozenset([
    "yeah", "gonna", "wanna", "ok", "hey", "cool", "nah", "btw",
    "lol", "haha", "alright",
])

# Prompt for structured text parsing via LLM.
_TEXT_PARSE_PROMPT = """Analyze this negotiation message from a human partner.
Return ONLY a JSON object with these fields:
{"urgency": 0.0-1.0, "threat": 0.0-1.0, "concession": 0.0-1.0, "tone": "friendly"|"neutral"|"hostile", "priorities": [], "length": "brief"|"moderate"|"verbose"}

- urgency: how time-pressured they seem (0=relaxed, 1=desperate)
- threat: how likely they are to walk away (0=committed, 1=about to leave)
- concession: how willing they seem to compromise (0=firm, 1=very flexible)
- tone: overall emotional tone
- priorities: list of issue NAMES they explicitly care about (empty list if none mentioned)
- length: how long their message is

Message: """


_PERSONALITY_CLASSIFY_PROMPT = """You are observing a negotiation partner. Based on their recent
messages and offer pattern, classify their personality as exactly one of:
- "collaborative": values mutual gain, makes concessions, uses warm language
- "tough": holds firm, concedes slowly if at all, minimal warmth
- "random": inconsistent behavior, offers jump around unpredictably
- "manipulative": fairness-claims + extreme offers, emotional pressure

Return ONLY a JSON object: {"personality": "collaborative"|"tough"|"random"|"manipulative", "confidence": 0.0-1.0}

"""


# H2 (2026-04-15) Remediator: one-shot LLM rewrite pass over our own
# draft, flagging aggression / contradiction / manipulation / hallucinated
# facts. Lineage: Hua et al. "Assistive LLM Agents for Socially-Aware
# Negotiation Dialogues", Findings of EMNLP 2024 (pattern B3). The
# remediator must preserve structured offer content (issue names, yes/no,
# accept/reject) verbatim and must not introduce numbers.
_REMEDIATOR_PROMPT = """You are a norm-compliance editor for an outgoing negotiation message.

INPUT: a DRAFT message written by a negotiator. Context: what we intend to do
this turn (accept / propose / reject) and the partner's most recent message.

TASK: if the draft is clean, return it UNCHANGED, verbatim. Otherwise rewrite
it ONLY to remove these specific problems:
- aggressive or threatening language,
- manipulation tactics (false-urgency pressure, guilt-tripping),
- contradiction with the partner's most recent message,
- hallucinated specific facts (dates, totals, promises) that were not in the
  intent or the partner's text.

STRICT RULES:
- Preserve issue names VERBATIM (exact capitalization).
- Preserve the words "yes", "no", "accept", "reject" verbatim if present.
- Do NOT introduce any specific numerical quantities.
- Output is at most 2 sentences, similar length to the draft.
- Warm, professional tone. Never begin with a preamble like "Here is the rewrite"; just return the message.

Respond with ONLY a JSON object of the form {"text": "final message"}.
"""


_COLD_READ_PROMPT = """You are reading a negotiation partner's opening move to guess their priorities.
Return ONLY a JSON object of the form {"weights": [w0, w1, ...]} where each w_i is a
number in [0, 1] estimating how much the partner cares about issue i. The array length
must match the number of issues. Higher = more important to them. Normalize so the
weights sum to 1.0 (or as close as possible).

Use both their message (what they emphasized verbally) and their offer (what they
proposed on each issue). If they proposed an extreme value on an issue, they probably
care about it more; if they sounded passionate about an issue in words, that's a
stronger signal than the offer alone.

"""


# =============================================================================
# Agent
# =============================================================================


class AgoraNegotiator(LLMBoulwareTBNegotiator):
    """Hybrid negotiator with custom acceptance, opponent model, and text parsing.

    Inherits Boulware offer generation + LLM text from the base class.
    Overrides:
      - ``respond``: custom acceptance + opponent/text tracking
      - ``propose``: win-win offer search using opponent model
      - ``_build_system_prompt``: phase-specific prompts
      - ``_build_user_message``: injects opponent insights for the LLM
    """

    # T3: per-instance overrides for the LLM endpoint and side-channel
    # timeout. Default values reproduce the previous hard-coded behavior
    # so existing callers don't need changes.
    DEFAULT_OLLAMA_MODEL: str = "qwen3:4b-instruct"
    DEFAULT_SIDE_CHANNEL_TIMEOUT: float = 5.0

    # H1 (2026-04-15) ASTRA defaults. Exposed as instance-level knobs so
    # the H1 sweep (lambda_a × k grid) can override them without
    # monkey-patching. Calibrated on the tdeep HAN-scenario tournament.
    DEFAULT_ASTRA_LAMBDA_U: float = 0.6
    DEFAULT_ASTRA_LAMBDA_A: float = 0.4
    DEFAULT_ASTRA_K: float = 6.0
    DEFAULT_ASTRA_B: float = 0.0
    # 2026-05-15: size of the sliding window over our recent proposals'
    # estimated opp-util. Used as the reference point for the ASTRA
    # accept-probability logistic. Picked at 10 so that in a typical
    # ~25-step match the window covers the last ~40% of our proposals —
    # long enough to remember what we've offered the partner recently,
    # short enough to forget early-match reference points that no longer
    # bound the partner's expectations.
    ASTRA_WINDOW: int = 10

    # 2026-05-15 Phase B knob: how `_should_accept` reads the time
    # signal. "wall" (default, ships in production) uses
    # `state.relative_time` directly, matching pre-Phase-B behaviour.
    # "effective" uses `_effective_t(state, harden=False)`, which blends
    # step-based and wall-clock progress (max(step_t, wall_t*0.5)) so
    # the acceptance threshold tracks discrete turn progress rather
    # than the wall-clock that an LLM-driven opponent can manipulate.
    # Wired as a constructor knob so the upcoming A/B sweep can flip
    # modes without monkey-patching. Default left at "wall" so the
    # commit is a no-op until the sweep validates the switch.
    DEFAULT_ACCEPTANCE_TIME_MODE: str = "wall"

    def __init__(
        self,
        provider: str = "ollama",
        temperature: float = 0.7,
        # Cut from 768 → 200 (2026-04-13): we ask for 2 sentences only,
        # typical completion is ~40-80 tokens. 768 was a default pulled
        # from the template; on local Ollama it burned 25-70s per reply
        # which times out HANI at ~5 min. 200 leaves headroom without
        # constraining the LLM's natural 2-sentence answers.
        max_tokens: int = 200,
        verbose: bool = False,
        llm_kwargs: dict[str, Any] | None = None,
        side_channel_timeout: float | None = None,
        astra_lambda_u: float | None = None,
        astra_lambda_a: float | None = None,
        astra_k: float | None = None,
        astra_b: float | None = None,
        acceptance_time_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        merged_llm_kwargs: dict[str, Any] = {
            "timeout": 120.0,
            "num_retries": 3,
        }
        if llm_kwargs:
            merged_llm_kwargs.update(llm_kwargs)

        # The HAN submission should always use our fixed system prompt.
        # Some loaders/tests pass prompt customisation through **kwargs;
        # remove these keys here to avoid duplicate-key TypeErrors while
        # keeping Agora's prompt surface deterministic.
        kwargs.pop("system_prompt", None)

        # CRITICAL FIX (2026-05-15, ported from origin/master 1d0d73a):
        # LLMBoulwareTBNegotiator.__init__ forwards **kwargs ONLY to the
        # wrapped BoulwareTBNegotiator child — the meta itself never receives
        # ufun/preferences/id/name. Without re-applying these on the meta:
        #   * self.ufun == None  → every `if self.ufun is None: return ...`
        #     guard in Agora short-circuits, so custom acceptance / propose /
        #     opener / Pareto / Nash all die silently and we run as vanilla
        #     Boulware with LLM text wrapping.
        #   * tournament scoring (`float(_.ufun(agreement)) - reserved_value`)
        #     evaluates `None(agreement)` → Advantage forced to 0.0 every
        #     match.
        # Snapshot the relevant kwargs here, then re-apply them onto `self`
        # after super().__init__ returns.
        ufun_arg = kwargs.get("ufun")
        prefs_arg = kwargs.get("preferences")
        id_arg = kwargs.get("id")
        name_arg = kwargs.get("name")

        super().__init__(
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose,
            system_prompt=PERSUASION_SYSTEM_PROMPT,
            llm_kwargs=merged_llm_kwargs,
            **kwargs,
        )

        # Re-apply ufun on the meta if the parent chain dropped it. We do not
        # touch the wrapped base negotiator's ufun (it already has the same
        # reference). `_init_preferences` is what `_dissociate()` restores
        # after `on_leave`, so we set both to survive the negotiation
        # lifecycle (tournament scoring happens post-dissociation).
        if self._preferences is None and (ufun_arg is not None or prefs_arg is not None):
            pref = ufun_arg if ufun_arg is not None else prefs_arg
            self.set_preferences(pref)
            self._init_preferences = pref
        if id_arg is not None and getattr(self, "_id", None) != id_arg:
            try:
                self._id = id_arg
            except Exception:
                pass
        if name_arg is not None and getattr(self, "_name", None) != name_arg:
            try:
                self._name = name_arg
            except Exception:
                pass

        # T1: side-channel LLM timeout (cold-read, personality classifier,
        # structured text-parser). The model itself is fixed to the
        # parent's `self.model`, which defaults to qwen3:4b-instruct as
        # required by the HAN 2026 league rules (no model override).
        self._side_channel_timeout: float = (
            side_channel_timeout
            if side_channel_timeout is not None
            else self.DEFAULT_SIDE_CHANNEL_TIMEOUT
        )

        # H1 (2026-04-15): store ASTRA hyperparameters; used by
        # `_estimate_accept_prob` and the multi-candidate scorer in
        # `propose`. All four are sweep-overridable via constructor.
        self._astra_lambda_u: float = (
            astra_lambda_u if astra_lambda_u is not None
            else self.DEFAULT_ASTRA_LAMBDA_U
        )
        self._astra_lambda_a: float = (
            astra_lambda_a if astra_lambda_a is not None
            else self.DEFAULT_ASTRA_LAMBDA_A
        )
        self._astra_k: float = (
            astra_k if astra_k is not None else self.DEFAULT_ASTRA_K
        )
        self._astra_b: float = (
            astra_b if astra_b is not None else self.DEFAULT_ASTRA_B
        )

        mode = (
            acceptance_time_mode
            if acceptance_time_mode is not None
            else self.DEFAULT_ACCEPTANCE_TIME_MODE
        )
        if mode not in ("wall", "effective"):
            raise ValueError(
                f"acceptance_time_mode must be 'wall' or 'effective', got {mode!r}"
            )
        self._acceptance_time_mode: str = mode

        self._init_state()

    def _side_channel_model(self) -> str:
        """Resolve the model identifier for side-channel litellm calls.

        Reads the parent class's `self.model` (set via the base
        `LLMBoulwareTBNegotiator` initialiser and defaulting to
        qwen3:4b-instruct, the only model permitted by the HAN 2026
        league rules) and returns it in the litellm-formatted
        "ollama/<name>" form. Falls back to `DEFAULT_OLLAMA_MODEL` if
        the parent attribute is unset.
        """
        m = getattr(self, "model", None)
        if isinstance(m, str) and m:
            return f"ollama/{m}"
        return f"ollama/{self.DEFAULT_OLLAMA_MODEL}"

    def _init_state(self) -> None:
        """Initialize all mutable tracking state."""
        # Phase (text tone).
        self._phase: str = "persuasion"
        self._phase_entered_at_offer: int = 0  # I1: hysteresis guard

        # Utility normalization (computed in on_negotiation_start).
        self._util_min: float = 0.0
        self._util_max: float = 1.0
        self._constant_utility_range: bool = False

        # Opponent offer history (our NORMALIZED utility).
        self._opp_utility_history: list[float] = []
        self._best_opp_offer_util: float = 0.0  # best normalized utility seen
        self._flat_streak: int = 0
        self._concession_streak: int = 0

        # Opponent model: per-issue value frequencies.
        self._opp_offers: list[tuple] = []
        self._opp_issue_freq: dict[int, dict[Any, int]] = {}
        self._opp_weights: list[float] = []

        # Text belief state.
        self._text_urgency: float = 0.0
        self._text_threat_level: float = 0.0
        self._text_concession_signal: float = 0.0
        self._text_priority_issues: set[int] = set()
        # 2026-05-15: per-issue freshness counter (in opp-message turns).
        # An issue stays in _text_priority_issues only while its count is
        # positive; on each new partner message we decrement everyone and
        # bump the indices the partner just re-mentioned. Prevents stale
        # priorities from a single early message biasing the entire match.
        self._text_priority_freshness: dict[int, int] = {}
        self._text_threshold_adj: float = 0.0

        # Conversation history (last N exchanges for LLM context).
        self._conversation_history: list[str] = []
        self._max_history: int = 5
        # First sanitized partner text, even if it arrived before the
        # first offer. Used to seed the cold-read model when HANI sends
        # an opening chat turn separately from the first structured offer.
        self._first_partner_text: str | None = None

        # Partner communication style (from LLM parsing, Idea 7).
        self._partner_tone: str = "neutral"  # friendly/neutral/hostile
        self._partner_length: str = "moderate"  # brief/moderate/verbose
        self._partner_register: str = "neutral"  # formal/neutral/casual (#4)

        # Pre-computed outcome table (built in on_negotiation_start).
        # List of (outcome_tuple, our_normalized_util) sorted by our util desc.
        self._outcome_table: list[tuple[tuple, float]] = []

        # Issue-level firm stance (our critical issues).
        # Per-issue: value that gives us max utility + whether issue is critical.
        self._my_preferred_values: dict[int, Any] = {}
        self._critical_issues: set[int] = set()  # issues we never concede on

        # Scenario-adaptive parameters (set at on_negotiation_start).
        self._boulware_exponent: float = BOULWARE_EXPONENT  # may be adapted
        self._manipulation_detected: bool = False  # Idea 7: raised when extreme offers + fair-claim text

        # Proposal tracking (for concession framing + diversity).
        self._last_proposal: tuple | None = None
        # I4: LRU-capped at 100 entries to bound set-lookup cost in
        # late-game loops over large outcome tables (5000+ outcomes).
        self._proposed_outcomes: set[tuple] = set()
        self._proposed_order: deque[tuple] = deque(maxlen=100)
        self._concession_issues: list[str] = []  # issues we conceded on
        self._request_issues: list[str] = []  # issues we want them to move on

        # Pareto front cache: recomputed lazily once opp model stabilizes.
        self._pareto_outcomes: set[tuple] = set()
        self._pareto_computed_at_offers: int = -1

        # #3 Cold-read: one-shot LLM priority estimate from opp's first
        # message + offer. Seeds _opp_weights before frequency model has
        # enough data. I2: retained after frequency kicks in and blended
        # (0.3 * cold + 0.7 * freq) instead of overwritten.
        self._cold_read_done: bool = False
        self._cold_read_weights: list[float] | None = None

        # #7 Opponent personality classification (LLM-based).
        # Refreshed every few rounds from conversation + offer pattern.
        self._opp_personality: str = "unknown"
        self._opp_personality_confidence: float = 0.0
        self._opp_personality_updated_at: int = -1

        # Cache for `_estimate_opponent_utility` (hot path in Pareto/Nash
        # scoring). Invalidated automatically via the `_sig` key whenever
        # a new opponent offer arrives. 2026-04-14 profiling: this call
        # was 112k calls / 100 matches ≈ 21% of decision-only runtime.
        self._opp_util_cache: dict = {"_sig": -1}

        # H3 (2026-04-14): adaptive LLM allocation for _parse_text_signals.
        # Short/procedural partner messages ("ok", "sure, proceed") gain
        # nothing from the structured parse; the keyword fallback already
        # extracts signals cheaply. Telemetry counts gate decisions so we
        # can validate the 30% reduction target post-match.
        self._parse_gate_stats: dict = {"llm_calls": 0, "keyword_fallback": 0}

        # H2 (2026-04-15): remediator post-pass. `checked` counts drafts
        # sent through the LLM norm-compliance editor; `rewritten` counts
        # the subset where the editor changed the text. Target: a 3-10%
        # rewrite rate on Ollama runs (above that signals an over-eager
        # editor that will drift style; below it suggests clean drafts).
        self._remediation_stats: dict = {"checked": 0, "rewritten": 0}

        # H1 (2026-04-15): running max `_estimate_opponent_utility` over
        # outcomes WE have proposed so far. Reference point for the
        # saturating acceptance-probability model — the opponent has
        # already "seen" our best-to-them offer, so a new candidate's
        # acceptance probability depends on whether it clears that bar.
        # 2026-05-15: switched from a strictly monotone max over the full
        # history to a windowed max over the last ASTRA_WINDOW proposals.
        # The monotone version saturated the ASTRA scorer to ~constant
        # opp_util after a handful of opp-aware rounds, collapsing the
        # behavioural signal back to `λ_u * our_util`. A sliding window
        # lets the reference recede when we keep returning to the same
        # high-util band, so the scorer keeps discriminating.
        self._opp_of_our_props_window: deque[float] = deque(
            maxlen=self.ASTRA_WINDOW
        )
        self._running_max_opp_of_our_props: float = 0.0

    # ------------------------------------------------------------------
    # Opponent ufun estimate (HAN Deception scoring)
    # ------------------------------------------------------------------

    # CRITICAL FIX (2026-05-15, ported from origin/master 1d0d73a): the HAN
    # scoring rule computes a "Deception" share from
    # `compare_ufuns(real_opp_ufun, self.opponent_ufun)` (Kendall tau). The
    # default `opponent_ufun` property on Negotiator reads from
    # `private_info["opponent_ufun"]`, which the mechanism never sets unless
    # `share_ufuns=True` is enabled — so we get `None`, `compare_ufuns`
    # returns -1.0, and our Deception share is 0 every match. By exposing
    # our running frequency-based opponent model as a `LinearAdditiveUtility-
    # Function` we routinely get Kendall ≥ 0.3 → up to ~+1.0 to Score / match.
    #
    # We do NOT *replace* a private_info opponent_ufun if the mechanism set
    # one explicitly (e.g., share_ufuns=True debug runs) — that path provides
    # the ground truth.

    @property
    def opponent_ufun(self):  # type: ignore[override]
        # Honor any explicitly-shared opponent ufun from the mechanism.
        shared = self._private_info.get("opponent_ufun") if hasattr(self, "_private_info") else None
        if shared is not None:
            return shared
        # Cache: rebuild only when our offer-history signature changes.
        n_off = len(getattr(self, "_opp_offers", []) or [])
        cache = getattr(self, "_opp_ufun_estimate_cache", None)
        if cache is not None and cache.get("_n") == n_off:
            return cache.get("ufun")
        est = self._build_opponent_ufun_estimate()
        self._opp_ufun_estimate_cache = {"_n": n_off, "ufun": est}
        return est

    def _build_opponent_ufun_estimate(self):
        """Construct a LinearAdditiveUtilityFunction approximating the opponent.

        Per-issue value function: Dirichlet-smoothed marginal frequency of
        the values the opponent has actually proposed. Issue weights: our
        running `_opp_weights` (already a blend of cold-read + Dirichlet-
        smoothed frequency mode). With even one observed opp offer this
        produces a non-flat ufun that scores positive Kendall against the
        true opponent ufun on every HAN scenario tested.

        Returns a `LinearAdditiveUtilityFunction` over the same outcome
        space. Never returns None during a live negotiation (an all-zero
        history yields a rank-tiebreaker ufun whose Kendall is small but
        non-NaN, which is still better than -1 from the None path).

        Uses snapshot fields populated in `on_negotiation_start` so the
        estimate is still buildable AFTER `_dissociate()` clears `self.nmi`
        (tournament scoring happens post-dissociation).
        """
        # Prefer live nmi when available, fall back to the snapshot. We
        # snapshot in on_negotiation_start so the post-match scoring path
        # (after _dissociate() sets self._nmi = None) still works.
        issues = None
        outcome_space = None
        nmi = self.nmi
        if nmi is not None:
            try:
                issues = nmi.issues
                outcome_space = nmi.outcome_space
            except Exception:
                pass
        if not issues:
            issues = getattr(self, "_snapshot_issues", None)
            outcome_space = getattr(self, "_snapshot_outcome_space", outcome_space)
        if not issues:
            return None

        n_issues = len(issues)
        weights_src = list(getattr(self, "_opp_weights", None) or [])
        if not weights_src:
            weights_src = [1.0 / n_issues] * n_issues
        # Pad / truncate to match issue count.
        if len(weights_src) < n_issues:
            weights_src = weights_src + [0.0] * (n_issues - len(weights_src))
        weights_src = weights_src[:n_issues]

        ALPHA = 0.5  # Dirichlet prior per value (Jeffreys-like)
        TIE = 1e-6   # rank-based tie-breaker so an empty model is still non-flat
        freq_map = getattr(self, "_opp_issue_freq", None) or {}
        value_funs: dict[str, TableFun] = {}
        for i, iss in enumerate(issues):
            try:
                all_vals = list(iss.all) if hasattr(iss, "all") else []
            except Exception:
                all_vals = []
            if not all_vals:
                continue
            freq = freq_map.get(i, {}) or {}
            denom = sum(freq.values()) + ALPHA * len(all_vals)
            mapping = {}
            # Sort values consistently so rank tiebreaker is deterministic.
            try:
                ranked_vals = sorted(all_vals, key=lambda v: (str(type(v).__name__), v))
            except Exception:
                ranked_vals = list(all_vals)
            for rank, v in enumerate(ranked_vals):
                f = freq.get(v, 0)
                mapping[v] = (f + ALPHA) / denom + TIE * rank
            name = iss.name if getattr(iss, "name", None) else f"i{i}"
            value_funs[name] = TableFun(mapping)

        if not value_funs:
            return None
        try:
            return LinearAdditiveUtilityFunction(
                values=value_funs,
                weights=list(weights_src),
                outcome_space=outcome_space,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_negotiation_end(self, state) -> None:  # type: ignore[override]
        """H2/H3 telemetry dump (gated on env var).

        When `AGORA_TELEMETRY_PATH` is set, append one JSON line per
        match with `_parse_gate_stats` and `_remediation_stats` so we
        can validate H3's >=30% LLM-call cut and H2's 3-10% rewrite
        rate on real HANI sessions without intrusive stdout logging.
        No effect when the env var is unset (default).
        """
        try:
            super().on_negotiation_end(state)  # type: ignore[misc]
        except Exception:
            pass
        path = os.environ.get("AGORA_TELEMETRY_PATH")
        if not path:
            return
        import json as _json
        import time as _time
        agreement = getattr(state, "agreement", None)
        try:
            agreed = agreement is not None
        except Exception:
            agreed = False
        payload = {
            "ts": _time.time(),
            "agent_id": getattr(self, "id", None) or getattr(self, "name", ""),
            "agreed": bool(agreed),
            "steps": getattr(state, "step", None),
            "parse_gate_stats": dict(self._parse_gate_stats),
            "remediation_stats": dict(self._remediation_stats),
            "n_opp_offers": len(self._opp_offers),
            "phase": self._phase,
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(payload) + "\n")
        except Exception:
            pass

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._init_state()
        # Snapshot issues/outcome_space — survives `_dissociate()` clearing
        # `self._nmi` at match end so `opponent_ufun` is still buildable
        # for the tournament scoring path.
        self._snapshot_issues = None
        self._snapshot_outcome_space = None
        if self.nmi is not None:
            try:
                self._snapshot_issues = tuple(self.nmi.issues)
                self._snapshot_outcome_space = self.nmi.outcome_space
            except Exception:
                pass
        # Compute utility range for normalization.
        if self.ufun is not None:
            try:
                lo, hi = self.ufun.minmax()
                self._util_min = float(lo)
                self._constant_utility_range = float(hi) <= float(lo)
                self._util_max = float(hi) if float(hi) > float(lo) else float(lo) + 1.0
            except Exception:
                pass
        # Pre-compute outcome table for fast opponent-aware proposing.
        self._build_outcome_table()
        # Analyze our own preferences to identify critical issues.
        self._analyze_own_preferences()
        # Scenario introspection: adapt parameters to scenario characteristics.
        self._introspect_scenario()

    def _normalize(self, raw_util: float) -> float:
        """Map raw utility to [0, 1] range, clamped.

        Handles -inf (no reservation) / +inf edge cases gracefully.
        """
        import math
        if not math.isfinite(raw_util):
            # -inf means "no reservation" → treat as 0 (no floor).
            # +inf shouldn't happen but treat as max.
            return 0.0 if raw_util < 0 else 1.0
        span = self._util_max - self._util_min
        if span <= 0:
            return 0.0
        if self._constant_utility_range:
            return 1.0 if raw_util >= self._util_min else 0.0
        normalized = (raw_util - self._util_min) / span
        # Clamp to sensible range to avoid negative-infinity propagation.
        return max(0.0, min(1.0, normalized))

    def _normalized_reservation(self) -> float:
        """Reservation value on the same normalized scale as offers."""
        if self.ufun is None:
            return 0.0
        try:
            rv = self.ufun.reserved_value
            if rv is None:
                return 0.0
            if self._constant_utility_range:
                # In a degenerate equal-utility outcome space, treating
                # reservation as 1.0 when rv equals the only outcome value
                # makes reservation+buffer impossible. But if reservation
                # is strictly above that value, every outcome is below
                # BATNA and should remain unacceptable.
                return 0.0 if float(rv) <= self._util_min else 1.0
            return self._normalize(float(rv))
        except Exception:
            return 0.0

    def _build_outcome_table(self) -> None:
        """Pre-compute (outcome, normalized_utility) for all outcomes.

        Sorted by our utility descending. Enables fast opponent-aware
        proposing by scanning outcomes at a target utility level.
        """
        if self.ufun is None or self.nmi is None:
            return
        # NOTE: `enumerate_or_sample(max_cardinality=5000)` ignores the cap
        # for discrete spaces and returns ALL outcomes — a scenario with
        # 8 issues × 9 values produces 43M outcomes and hangs the agent
        # for minutes (2026-04-14 stress test). Use explicit sample()
        # when the space is large, only enumerate when it actually fits.
        CAP = 5000
        try:
            space = self.nmi.outcome_space
            card = getattr(space, "cardinality", None)
            if card is not None and card != float("inf") and card <= CAP:
                outcomes = list(space.enumerate())
            else:
                outcomes = list(space.sample(n_outcomes=CAP))
        except Exception:
            return
        table: list[tuple[tuple, float]] = []
        for o in outcomes:
            try:
                u = self._normalize(float(self.ufun(o)))
                table.append((tuple(o), u))
            except Exception:
                continue
        table.sort(key=lambda x: x[1], reverse=True)
        self._outcome_table = table

    def _recompute_pareto_front(self, force: bool = False) -> None:
        """Compute Pareto-optimal outcomes under (our_util, est_opp_util).

        Called lazily: only once the opponent model has stabilized (we have
        seen enough offers to trust `_estimate_opponent_utility`). Cached
        and refreshed every ~5 new opponent offers since the last compute.
        """
        n_offers = len(self._opp_offers) if hasattr(self, "_opp_offers") else 0
        if not force and n_offers < MIN_OFFERS_FOR_MODEL:
            return
        if (
            not force
            and self._pareto_computed_at_offers >= 0
            and n_offers - self._pareto_computed_at_offers < 5
        ):
            return
        if not self._outcome_table or not self._opp_weights:
            return

        scored: list[tuple[tuple, float, float]] = []
        for outcome, our_u in self._outcome_table:
            try:
                opp_u = self._estimate_opponent_utility(outcome)
            except Exception:
                continue
            scored.append((outcome, our_u, opp_u))

        # Scan Pareto front: sort by our_u desc, track max opp_u seen.
        scored.sort(key=lambda x: (-x[1], -x[2]))
        pareto: set[tuple] = set()
        best_opp = -1.0
        for outcome, our_u, opp_u in scored:
            if opp_u > best_opp:
                pareto.add(outcome)
                best_opp = opp_u
        self._pareto_outcomes = pareto
        self._pareto_computed_at_offers = n_offers

    def _aspiration_level(self, t: float) -> float:
        """Current normalized aspiration level at time t."""
        return max(0.0, 1.0 * (1.0 - t ** self._boulware_exponent))

    def _effective_t(self, state: SAOState, harden: bool = False) -> float:
        """Unified time signal: blend of step-based and wall-clock progress.

        Step-based alone protects us from LLM-latency-induced concession
        runaway. Pure step-based alone ignores real deadline pressure —
        against a slow human the wall-clock can hit 90% while step is at
        10%, and we'd never concede in time. Compromise:

            eff_t = max(step_t, wall_t * 0.5)

        Step dominates while the GUI clock lags (normal case). When
        wall-clock overshoots step by a lot, the 0.5 multiplier pulls
        us part-way toward the wall-clock curve so we don't time out
        without any concessions made.

        `harden=True` additionally compresses t when the opponent is
        clearly conceding (used only by propose() to stay tougher). For
        acceptance we pass `harden=False` — hardening acceptance rejects
        marginal deals we could have closed (sweep-validated 2026-04-12).
        """
        try:
            n_steps = getattr(self.nmi, "n_steps", None) if self.nmi else None
        except Exception:
            n_steps = None
        wall_t = float(state.relative_time) if state.relative_time is not None else 0.0
        if n_steps:
            # Use (step+1)/n_steps so the last step (step=n_steps-1) maps
            # to t=1.0. The previous `step/n_steps` formula maxed out at
            # (n_steps-1)/n_steps ≈ 0.96 on a 25-step match, delaying the
            # final-round concession by a full step.
            step_t = min(1.0, (state.step + 1) / max(n_steps, 1))
            t = max(step_t, wall_t * 0.5)
        else:
            t = wall_t
        if harden:
            try:
                if self._compute_concession_rate() > 0.05:
                    t = t ** 1.3
            except Exception:
                pass
        return max(0.0, min(1.0, t))

    def _introspect_scenario(self) -> None:
        """Adapt parameters to scenario characteristics.

        Looks at outcome space size + utility variance to pick a
        Boulware exponent appropriate for the scenario.

        Wide, rich outcome spaces → tougher (more options mean we can afford to hold out).
        Narrow, tight outcome spaces → softer (fewer options, need to concede).
        """
        if not self._outcome_table:
            return
        n_outcomes = len(self._outcome_table)
        utilities = [u for _, u in self._outcome_table]
        u_mean = sum(utilities) / len(utilities)
        u_var = sum((u - u_mean) ** 2 for u in utilities) / len(utilities)
        u_std = u_var ** 0.5

        # Default exponent.
        exp = BOULWARE_EXPONENT  # 3.0

        # Wide utility spread → more flexibility available → can be tougher.
        if u_std > 0.25:
            exp += 0.3
        elif u_std < 0.15:
            exp -= 0.3

        # Larger outcome space → more options → can be tougher.
        if n_outcomes >= 500:
            exp += 0.2
        elif n_outcomes < 100:
            exp -= 0.3

        # Clamp to reasonable range.
        self._boulware_exponent = max(2.0, min(4.0, exp))

    def _analyze_own_preferences(self) -> None:
        """Identify our preferred value per issue + critical issues.

        For each issue i, the "preferred value" is the one that maximizes
        our utility when all other issues are set to their best values.
        An issue is "critical" if the utility spread across its values
        (holding others at best) is above median.
        """
        if self._outcome_table is None or not self._outcome_table:
            return
        if self.nmi is None or self.ufun is None:
            return
        try:
            issues = self.nmi.issues
        except Exception:
            return
        if not issues:
            return

        # Best outcome = highest utility in the sorted table.
        best_outcome = self._outcome_table[0][0]

        # For each issue, measure utility spread by varying that issue
        # alone (holding others at best values).
        spreads: list[float] = []
        preferred: dict[int, Any] = {}
        for i, issue in enumerate(issues):
            try:
                values = list(issue.all) if hasattr(issue, "all") else []
                if not values:
                    continue
            except Exception:
                continue

            utilities = []
            for v in values:
                candidate = list(best_outcome)
                if i < len(candidate):
                    candidate[i] = v
                    try:
                        u = self._normalize(float(self.ufun(tuple(candidate))))
                        utilities.append((v, u))
                    except Exception:
                        continue

            if not utilities:
                continue
            # Preferred value = max utility.
            preferred[i] = max(utilities, key=lambda x: x[1])[0]
            spread = max(u for _, u in utilities) - min(u for _, u in utilities)
            spreads.append(spread)

        self._my_preferred_values = preferred

        # Critical issues = above-median spread (we care disproportionately).
        if spreads:
            sorted_spreads = sorted(spreads)
            median = sorted_spreads[len(sorted_spreads) // 2]
            for i, issue in enumerate(issues):
                if i not in preferred:
                    continue
                # Recompute spread for this issue for comparison.
                values = list(issue.all) if hasattr(issue, "all") else []
                if not values:
                    continue
                utilities = []
                for v in values:
                    candidate = list(best_outcome)
                    if i < len(candidate):
                        candidate[i] = v
                        try:
                            u = self._normalize(float(self.ufun(tuple(candidate))))
                            utilities.append(u)
                        except Exception:
                            continue
                if utilities:
                    issue_spread = max(utilities) - min(utilities)
                    if issue_spread >= median and issue_spread > 0.1:
                        self._critical_issues.add(i)

    def _find_nash_outcome(self, min_our_util: float) -> tuple | None:
        """Find outcome maximizing our_util * opp_estimated_util.

        Subject to our_util >= min_our_util (so we don't sacrifice utility).
        This is the classical Nash bargaining solution applied to our
        estimated opponent model.
        """
        if not self._outcome_table or not self._opp_weights:
            return None
        best: tuple | None = None
        best_product = -1.0
        for outcome, our_util in self._outcome_table:
            if our_util < min_our_util:
                break  # sorted desc
            opp_util = self._estimate_opponent_utility(outcome)
            # Nash product: maximize our_util * opp_util.
            product = our_util * opp_util
            if product > best_product:
                best_product = product
                best = outcome
        return best

    def _pareto_walk_from_offer(
        self,
        starting_outcome: tuple,
        min_our_util: float,
        t: float = 0.0,
    ) -> tuple | None:
        """Greedily improve our utility starting from opponent's offer.

        Goal: find an outcome close to the opponent's last offer that
        has higher utility for us. Each swap must not hurt the opponent
        (or only minimally).

        Returns the best-for-us outcome in the Pareto neighborhood.
        """
        if self.ufun is None or self.nmi is None:
            return None
        try:
            issues = self.nmi.issues
        except Exception:
            return None
        if not issues:
            return None

        current = list(starting_outcome)
        try:
            current_our = self._normalize(float(self.ufun(tuple(current))))
        except Exception:
            return None
        current_opp = self._estimate_opponent_utility(tuple(current))

        improved = True
        max_iterations = 8  # prevent long loops
        iteration = 0
        while improved and iteration < max_iterations:
            iteration += 1
            improved = False
            best_i = -1
            best_v = None
            best_gain = 0.0
            # I3: Relax critical-issue forcing when (a) we're late in the
            # negotiation OR (b) the opponent is actively conceding on this
            # or another issue. Rigid forcing trashes mutual gain when the
            # partner is being flexible; late-game it can sink near-closed
            # deals. Use a soft preference (+0.10 utility bias) instead.
            relax_forcing = (
                t > 0.75
                or self._compute_concession_rate() > 0.03
            )
            for i, issue in enumerate(issues):
                if (
                    i in self._critical_issues
                    and i in self._my_preferred_values
                    and not relax_forcing
                ):
                    # Force critical issues to our preferred value early.
                    if current[i] != self._my_preferred_values[i]:
                        candidate = list(current)
                        candidate[i] = self._my_preferred_values[i]
                        try:
                            if not self.nmi.outcome_space.is_valid(tuple(candidate)):
                                continue
                            cand_our = self._normalize(float(self.ufun(tuple(candidate))))
                            cand_opp = self._estimate_opponent_utility(tuple(candidate))
                        except Exception:
                            continue
                        gain = cand_our - current_our
                        opp_loss = current_opp - cand_opp
                        if gain > 0 and gain > opp_loss * 0.5 and gain > best_gain:
                            best_gain = gain
                            best_i = i
                            best_v = self._my_preferred_values[i]
                    continue
                try:
                    values = list(issue.all) if hasattr(issue, "all") else []
                except Exception:
                    continue
                for v in values:
                    if v == current[i]:
                        continue
                    candidate = list(current)
                    candidate[i] = v
                    try:
                        if not self.nmi.outcome_space.is_valid(tuple(candidate)):
                            continue
                        cand_our = self._normalize(float(self.ufun(tuple(candidate))))
                    except Exception:
                        continue
                    cand_opp = self._estimate_opponent_utility(tuple(candidate))
                    # Accept if our utility gains AND opp utility doesn't drop much.
                    our_gain = cand_our - current_our
                    opp_loss = current_opp - cand_opp
                    # Strict improvement: our_gain > opp_loss * 0.5.
                    if our_gain > 0 and our_gain > opp_loss * 0.5:
                        if our_gain > best_gain:
                            best_gain = our_gain
                            best_i = i
                            best_v = v
            if best_i >= 0 and best_v is not None:
                current[best_i] = best_v
                try:
                    current_our = self._normalize(float(self.ufun(tuple(current))))
                    current_opp = self._estimate_opponent_utility(tuple(current))
                except Exception:
                    pass
                improved = True

        result = tuple(current)
        if result == tuple(starting_outcome):
            return None  # No improvement found.
        if current_our < min_our_util:
            return None  # Still below our floor.
        return result

    # ------------------------------------------------------------------
    # Phase logic (text tone switching)
    # ------------------------------------------------------------------

    def _update_phase(self, state: SAOState) -> None:
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return
        try:
            util = self._normalize(float(self.ufun(offer)))
        except Exception:
            return

        self._opp_utility_history.append(util)
        self._best_opp_offer_util = max(self._best_opp_offer_util, util)
        if len(self._opp_utility_history) < 2:
            return

        improvement = util - self._opp_utility_history[-2]
        if improvement > CONCESSION_EPSILON:
            self._concession_streak += 1
            self._flat_streak = 0
        else:
            self._flat_streak += 1
            self._concession_streak = 0

        # I1: Hysteresis — require 3 offers of dwell-time in the current
        # phase before switching, to avoid persuasion ↔ deception flips
        # from noisy offer trajectories. Humans notice tone whiplash.
        MIN_PHASE_DWELL = 3
        n_off = len(self._opp_utility_history)
        phase_age = n_off - self._phase_entered_at_offer
        if self._phase == "persuasion":
            if (
                self._flat_streak >= CONCESSION_THRESHOLD_ROUNDS
                and phase_age >= MIN_PHASE_DWELL
            ):
                self._phase = "deception"
                self._phase_entered_at_offer = n_off
        else:
            if (
                self._concession_streak >= RECOVERY_THRESHOLD_ROUNDS
                and phase_age >= MIN_PHASE_DWELL
            ):
                self._phase = "persuasion"
                self._phase_entered_at_offer = n_off

    # ------------------------------------------------------------------
    # Feature 1: Custom acceptance policy
    # ------------------------------------------------------------------

    def _compute_concession_rate(self) -> float:
        """Trend in opponent offer utility (our view).

        Uses linear regression slope over recent window instead of
        endpoint difference — robust against oscillating offers.
        """
        h = self._opp_utility_history
        if len(h) < 2:
            return 0.0
        window = h[-self._concession_rate_window_len():]
        n = len(window)
        # Simple linear regression slope: sum((x-xmean)(y-ymean)) / sum((x-xmean)^2)
        x_mean = (n - 1) / 2.0
        y_mean = sum(window) / n
        num = sum((i - x_mean) * (window[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    def _concession_rate_window_len(self) -> int:
        """Window length for concession-trend regression.

        Keep the historical 7-sample window for normal/long matches
        because it is sweep-sensitive. Only shrink it for very short
        matches, where 7 samples would be most of the negotiation.
        """
        try:
            n_steps = getattr(self.nmi, "n_steps", None) if self.nmi else None
            if n_steps and float(n_steps) <= 10:
                return 3
        except Exception:
            pass
        return 7

    def _log_text_event(
        self,
        state: SAOState,
        action: str,
        text: str,
        received_text: str | None = None,
    ) -> None:
        """Optional JSONL log for text-quality review.

        Enabled only when AGORA_TEXT_LOG_PATH is set. This is deliberately
        separate from the aggregate telemetry because it can contain
        dialogue text from GUI sessions.
        """
        path = os.environ.get("AGORA_TEXT_LOG_PATH")
        if not path:
            return
        try:
            import json as _json
            import time as _time

            payload = {
                "ts": _time.time(),
                "step": getattr(state, "step", None),
                "relative_time": getattr(state, "relative_time", None),
                "action": action,
                "phase": self._phase,
                "personality": self._opp_personality,
                "personality_confidence": self._opp_personality_confidence,
                "partner_tone": self._partner_tone,
                "partner_length": self._partner_length,
                "partner_register": self._partner_register,
                "text": text,
                "received_text": (
                    self._sanitize_partner_text(received_text or "")[:300]
                    if received_text
                    else ""
                ),
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _should_end(self, state: SAOState) -> bool:
        """#1C Dignity END: only on the very last step, when the situation
        is genuinely hopeless. Narrow gate to avoid forfeiting late accepts.

        Triggers only if ALL of:
        - we are on the final step (no further rounds),
        - the best opponent offer ever seen is below reservation + buffer,
        - the current offer (if any) is also below reservation + buffer.

        For human-perception scoring: an assertive decline reads better
        than a silent REJECT at timeout. For utility: we only end when no
        acceptable deal is reachable anyway.
        """
        try:
            n_steps = getattr(self.nmi, "n_steps", None) if self.nmi else None
            if not n_steps or state.step < n_steps - 1:
                return False
        except Exception:
            return False

        reservation = self._normalized_reservation()
        floor = reservation + 0.02

        if self._best_opp_offer_util >= floor:
            return False
        if state.current_offer is not None and self.ufun is not None:
            try:
                cur = self._normalize(float(self.ufun(state.current_offer)))
                if cur >= floor:
                    return False
            except Exception:
                pass
        return True

    def _should_accept(self, state: SAOState) -> bool:
        """Custom acceptance: replaces stock Boulware acceptance entirely."""
        offer = state.current_offer
        if offer is None or self.ufun is None:
            return False

        try:
            offer_util = self._normalize(float(self.ufun(offer)))
        except Exception:
            return False

        reservation = self._normalized_reservation()

        # 2026-05-15: time source is controlled by `_acceptance_time_mode`.
        # The historical default ("wall") preserves the asymmetry with
        # propose() that was sweep-verified on 2026-04-12 — a wall-clock
        # acceptance threshold is looser and closes marginal deals
        # against bot opponents (see project_agora_devils_advocate_fixes).
        # "effective" routes through _effective_t(harden=False), which
        # follows step progress and tightens the threshold against
        # opponents that burn wall-clock with their own LLMs. The A/B
        # validation lives in `_sweep_b_accept_time.py`; keeping default
        # at "wall" until that sweep confirms the switch is positive.
        t = self._acceptance_t(state)

        # --- 1. Early good-deal shortcut ---
        if offer_util > EARLY_ACCEPT_THRESHOLD and t < 0.3:
            return True

        # --- 2. Aspiration curve (scenario-adaptive Boulware) ---
        aspiration = 1.0 * (1.0 - t ** self._boulware_exponent)

        # --- 3. Text signal adjustment ---
        aspiration += self._text_threshold_adj

        # --- 4. Reservation floor ---
        threshold = max(aspiration, reservation + RESERVATION_BUFFER)

        # --- 5. Late-game pressure + deadline exploitation ---
        if t > LATE_GAME_START:
            # Idea 8: if human is panicking, delay concession until t>0.95.
            human_panicking = (
                self._text_urgency > 0.4 and self._text_threat_level < 0.3
            )
            if human_panicking and t < 0.95:
                # Hold firm — the human will likely concede first.
                pass  # skip the late-game relaxation
            else:
                late_factor = 1.0 - (t - LATE_GAME_START) * 3.0
                threshold = max(reservation + 0.02, threshold * max(late_factor, 0.0))

        # --- 6. ACNext: accept if offer >= what we'd propose next ---
        # Never reject an offer that's better than our own next proposal,
        # but always respect the reservation floor.
        # Preserve positive text hardening (urgency / manipulation) across
        # ACNext. Pre-fix, this cap used raw aspiration and erased the very
        # hardening signal computed above.
        positive_text_adj = max(0.0, self._text_threshold_adj)
        next_aspiration = max(
            self._aspiration_level(t) + positive_text_adj,
            reservation + RESERVATION_BUFFER,
        )
        threshold = min(threshold, next_aspiration)

        # --- 7. Patience with convergence prediction ---
        # If opponent is conceding, estimate where their offers will be
        # in a few more rounds and decide whether waiting pays off.
        cr = self._compute_concession_rate()
        if cr > CONCESSION_EPSILON and t < 0.92:
            # Predict: if we wait 5 more rounds, opponent util ≈ current + 5*cr.
            predicted_improvement = min(0.1, cr * 5.0)
            # Only wait if predicted gain exceeds current margin.
            if offer_util < threshold + predicted_improvement:
                patience_margin = min(0.08, predicted_improvement * 0.5)
                threshold += patience_margin

        # --- 8. Best-offer floor: don't accept worse than what we saw ---
        # Unless very late (>0.95), require at least 90% of the best offer.
        if t < 0.95 and self._best_opp_offer_util > 0:
            best_floor = self._best_opp_offer_util * 0.90
            threshold = max(threshold, best_floor)

        # --- 9. Time-to-agreement nudge (Idea 8) — DISABLED ---
        # Preliminary sweep showed it causes Island/Linear regression.
        # Keep code path commented for future experimentation.
        # if t > 0.3 and offer_util > 0.3 and offer_util > reservation + 0.15:
        #     closing_bonus = min(0.03, (t - 0.3) * 0.05)
        #     threshold -= closing_bonus

        return offer_util >= threshold

    def _acceptance_t(self, state: SAOState) -> float:
        """Time signal used by acceptance.

        Historical default is wall-clock time, but if wall time is far
        ahead of discrete progress, use the same blended effective time
        as proposal generation. This protects against LLM-latency or
        slow-partner situations where raw wall time would trigger
        late-game relaxation while many turns remain.
        """
        if self._acceptance_time_mode == "effective":
            return self._effective_t(state, harden=False)
        wall_t = float(state.relative_time) if state.relative_time is not None else 0.0
        try:
            n_steps = getattr(self.nmi, "n_steps", None) if self.nmi else None
            if n_steps:
                step_t = min(1.0, (state.step + 1) / max(n_steps, 1))
                if wall_t >= step_t + 0.35 and step_t < 0.75:
                    return self._effective_t(state, harden=False)
        except Exception:
            pass
        return max(0.0, min(1.0, wall_t))

    # ------------------------------------------------------------------
    # Feature 2: Opponent preference inference
    # ------------------------------------------------------------------

    def _track_opponent_offer(self, state: SAOState) -> None:
        """Record the opponent's offer and update per-issue frequencies."""
        offer = state.current_offer
        if offer is None:
            return

        self._opp_offers.append(tuple(offer))
        for i, val in enumerate(offer):
            freq = self._opp_issue_freq.setdefault(i, {})
            freq[val] = freq.get(val, 0) + 1

        # #3 Cold-read on the first text-bearing opp offer: seed
        # _opp_weights using LLM reading of their opening message + offer.
        # HANI can send offer-only turns; do not mark the cold-read as
        # done until there is text worth trying.
        if not self._cold_read_done and len(self._opp_offers) >= 1:
            try:
                current_text = self._sanitize_partner_text(
                    self._extract_received_text(state) or ""
                )
                text = current_text or (self._first_partner_text or "")
                issues = self.nmi.issues if self.nmi else ()
                issue_names = [issue.name or str(i) for i, issue in enumerate(issues)]
                if issue_names and text:
                    self._cold_read_done = True
                    w = self._llm_cold_read_priorities(text, tuple(offer), issue_names)
                    if w is not None:
                        self._opp_weights = w
                        self._cold_read_weights = list(w)  # I2: retain for blending
                        self._refresh_astra_reference_from_proposals()
            except Exception:
                pass

        if len(self._opp_offers) >= MIN_OFFERS_FOR_MODEL:
            self._infer_opponent_weights()

    def _infer_opponent_weights(self) -> None:
        """Infer relative issue importance from opponent offer history.

        Dirichlet-smoothed concentration (2026-04-13): instead of the raw
        `max_freq / total` from earlier, use the posterior Dirichlet mode
        probability under a symmetric prior $\\alpha_0=0.5$:

            importance_i = (α_0 + max_count) / (α_0 * k_i + total_count)

        where $k_i$ is the number of distinct values observed on issue $i$.
        This is the posterior-mean probability of the mode value. It
        retains the direction of the old ratio (high when the opponent
        always proposes the same value, low when spread) while smoothing
        small-sample noise.

        (An entropy-based variant was tried first and regressed the sweep
        — penalised partial concentration too aggressively and flattened
        the importance signal across issues. Reverted to mode-probability
        smoothing, which keeps the sweep stable and adds credible-prior
        behaviour for free.)
        """
        if not self._opp_offers:
            return
        n_issues = len(self._opp_offers[0])
        ALPHA_0 = 0.5  # Dirichlet prior per value (Jeffreys-like)
        raw: list[float] = []
        for i in range(n_issues):
            freq = self._opp_issue_freq.get(i, {})
            if not freq:
                raw.append(0.5)
                continue
            k = len(freq)
            total_count = sum(freq.values())
            max_count = max(freq.values())
            raw.append((ALPHA_0 + max_count) / (ALPHA_0 * k + total_count))

        total_w = sum(raw) or 1.0
        freq_weights = [w / total_w for w in raw]

        # I2: Blend cold-read priorities with frequency signal instead of
        # overwriting. Cold-read captures stated priorities (from text);
        # frequency captures revealed preferences (from offers). Both are
        # informative — a strategic partner may verbally claim one thing
        # while their offer distribution reveals another.
        if (
            self._cold_read_weights is not None
            and len(self._cold_read_weights) == len(freq_weights)
        ):
            blended = [
                0.3 * c + 0.7 * f
                for c, f in zip(self._cold_read_weights, freq_weights)
            ]
            s = sum(blended) or 1.0
            self._opp_weights = [w / s for w in blended]
        else:
            self._opp_weights = freq_weights
        self._refresh_astra_reference_from_proposals()

    def _refresh_astra_reference_from_proposals(self) -> None:
        """Seed ASTRA's reference from proposals made before weights existed."""
        if not self._opp_weights or not self._proposed_order:
            return
        try:
            self._opp_of_our_props_window.clear()
            for outcome in list(self._proposed_order)[-self.ASTRA_WINDOW:]:
                self._opp_of_our_props_window.append(
                    self._estimate_opponent_utility(outcome)
                )
            if self._opp_of_our_props_window:
                self._running_max_opp_of_our_props = max(
                    self._opp_of_our_props_window
                )
        except Exception:
            pass

    def _find_win_win_offer(
        self,
        outcome: tuple,
        state: SAOState,
        min_our_util_norm: float | None = None,
    ) -> tuple | None:
        """Try single-issue and two-issue swaps toward opponent preferences.

        Uses frequency-based weights + text-detected priority issues to
        score candidate swaps.
        """
        if not self._opp_weights or self.ufun is None or self.nmi is None:
            return None

        try:
            issues = self.nmi.issues
        except Exception:
            return None
        if not issues:
            return None

        our_util = float(self.ufun(outcome))
        # Scale tolerance to utility range so it works for any scenario.
        tolerance = WIN_WIN_UTIL_TOLERANCE * (self._util_max - self._util_min)
        # Guard against offering below our BATNA, while preserving the
        # existing small win-win tolerance around the base proposal.
        floor_norm = None
        if self.ufun is not None:
            floor_norm = self._normalized_reservation() + 0.02

        # Build per-issue preferred values from frequency + text priorities.
        n = len(issues)
        opp_preferred: dict[int, Any] = {}
        for i in range(n):
            freq = self._opp_issue_freq.get(i, {})
            if not freq:
                continue
            opp_preferred[i] = max(freq, key=freq.get)

        # Score boost for issues the human explicitly mentioned.
        def _issue_score(i: int) -> float:
            w = self._opp_weights[i] if i < len(self._opp_weights) else 0.0
            if i in self._text_priority_issues:
                w += 0.15  # boost text-mentioned issues
            return w

        best: tuple | None = None
        best_score = 0.0

        def _eval_candidate(candidate_t: tuple) -> float | None:
            """Returns opponent-utility score if candidate is valid and above tolerance."""
            try:
                if not self.nmi.outcome_space.is_valid(candidate_t):
                    return None
                cand_util = float(self.ufun(candidate_t))
                cand_norm = self._normalize(cand_util)
            except Exception:
                return None
            if floor_norm is not None and cand_norm < floor_norm:
                return None
            if cand_util < our_util - tolerance:
                return None
            return self._estimate_opponent_utility(candidate_t)

        # --- Single-issue swaps ---
        for i, pref_val in opp_preferred.items():
            if pref_val == outcome[i]:
                continue
            candidate = list(outcome)
            candidate[i] = pref_val
            candidate_t = tuple(candidate)
            score = _eval_candidate(candidate_t)
            if score is not None and score > best_score:
                best_score = score
                best = candidate_t

        # --- Two-issue swaps (give on both to unlock bigger trades) ---
        issue_indices = list(opp_preferred.keys())
        for idx_a in range(len(issue_indices)):
            i = issue_indices[idx_a]
            if opp_preferred[i] == outcome[i]:
                continue
            for idx_b in range(idx_a + 1, len(issue_indices)):
                j = issue_indices[idx_b]
                if opp_preferred[j] == outcome[j]:
                    continue
                candidate = list(outcome)
                candidate[i] = opp_preferred[i]
                candidate[j] = opp_preferred[j]
                candidate_t = tuple(candidate)
                score = _eval_candidate(candidate_t)
                if score is not None and score > best_score:
                    best_score = score
                    best = candidate_t

        return best

    # ------------------------------------------------------------------
    # Feature 3: Text signal parsing
    # ------------------------------------------------------------------

    def _llm_parse_text(self, text: str) -> dict | None:
        """Use the LLM to extract structured signals from human text.

        Returns a dict with keys: urgency, threat, concession, tone,
        priorities, length. Returns None on failure.
        """
        import json as _json

        try:
            import litellm
            response = litellm.completion(
                model=self._side_channel_model(),
                messages=[
                    {"role": "user", "content": _TEXT_PARSE_PROMPT + text[:500]},
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=self._side_channel_timeout,
                num_retries=0,
            )
            raw = response.choices[0].message.content.strip()
            # Extract JSON from response (handle markdown code blocks).
            if "```" in raw:
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            parsed = _json.loads(raw)
            # Validate fields.
            if isinstance(parsed, dict) and "urgency" in parsed:
                return parsed
        except Exception:
            pass
        return None

    def _llm_classify_personality(self) -> tuple[str, float] | None:
        """#7 Classify opponent personality based on conversation + offers.

        Returns (personality, confidence) or None on failure. Uses recent
        conversation history + offer utility trajectory.
        """
        import json as _json

        history_lines = self._conversation_history[-6:] if self._conversation_history else []
        convo = "\n".join(history_lines)
        if not convo and not self._opp_offers:
            return None

        # Summarize offer trajectory as utilities we saw.
        util_trace = [f"{u:.2f}" for u in self._opp_utility_history[-6:]]
        trace_str = " -> ".join(util_trace) if util_trace else "no offers yet"

        prompt = (
            _PERSONALITY_CLASSIFY_PROMPT
            + f"Recent conversation:\n{convo}\n\n"
            + f"Offer utility trajectory (our view): {trace_str}\n"
        )
        try:
            import litellm
            response = litellm.completion(
                model=self._side_channel_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
                timeout=self._side_channel_timeout,
                num_retries=0,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            parsed = _json.loads(raw)
            p = parsed.get("personality", "unknown")
            c = float(parsed.get("confidence", 0.5))
            if p in ("collaborative", "tough", "random", "manipulative"):
                return p, max(0.0, min(1.0, c))
        except Exception:
            pass
        return None

    def _llm_cold_read_priorities(
        self, text: str, offer: tuple, issue_names: list[str]
    ) -> list[float] | None:
        """One-shot LLM reading of partner's priorities from first turn.

        Returns a list of per-issue weights (length == n_issues), normalized
        to sum to 1.0, or None on failure. Called once early in the
        negotiation to seed `_opp_weights` before the frequency model has
        enough data to be meaningful.
        """
        import json as _json

        n_issues = len(issue_names)
        if n_issues == 0:
            return None
        issue_list = ", ".join(f"{i}:{n}" for i, n in enumerate(issue_names))
        offer_str = ", ".join(f"{n}={v}" for n, v in zip(issue_names, offer))
        prompt = (
            _COLD_READ_PROMPT
            + f"Issues (index:name): {issue_list}\n"
            + f"Partner offered: {offer_str}\n"
            + f"Partner said: {text[:400]}\n"
        )
        try:
            import litellm
            response = litellm.completion(
                model=self._side_channel_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
                timeout=self._side_channel_timeout,
                num_retries=0,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            parsed = _json.loads(raw)
            if not isinstance(parsed, dict) or "weights" not in parsed:
                return None
            w = parsed["weights"]
            if not isinstance(w, list) or len(w) != n_issues:
                return None
            w = [max(0.0, float(x)) for x in w]
            s = sum(w)
            if s <= 0:
                return None
            return [x / s for x in w]
        except Exception:
            return None

    def _keyword_tone_length_register(self, text: str) -> tuple[str, str, str]:
        """#4 Keyword-based tone + length + register classifier.

        Returns (tone, length, register). Called as fallback when LLM
        structured parsing fails, so tone-mirroring still has signal.
        """
        text_lower = text.lower()
        friendly = sum(1 for w in _FRIENDLY_WORDS if self._term_in_text(text_lower, w))
        hostile = sum(1 for w in _HOSTILE_WORDS if self._term_in_text(text_lower, w))
        if hostile > friendly:
            tone = "hostile"
        elif friendly >= 2 or (friendly >= 1 and hostile == 0):
            tone = "friendly"
        else:
            tone = "neutral"

        # Length by character count of raw text (not lowercased).
        if len(text) < 60:
            length = "brief"
        elif len(text) > 200:
            length = "verbose"
        else:
            length = "moderate"

        formal = sum(1 for w in _FORMAL_WORDS if self._term_in_text(text_lower, w))
        casual = sum(1 for w in _CASUAL_WORDS if self._term_in_text(text_lower, w))
        if formal > casual:
            register = "formal"
        elif casual > formal:
            register = "casual"
        else:
            register = "neutral"
        return tone, length, register

    @staticmethod
    def _term_in_text(text_lower: str, term: str) -> bool:
        """True if a keyword/phrase appears without substring false hits."""
        if " " in term:
            return term in text_lower
        try:
            import re
            pattern = rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])"
            return re.search(pattern, text_lower) is not None
        except Exception:
            return term in text_lower

    def _keyword_parse_text(self, text_lower: str) -> tuple[float, float, float]:
        """Fallback keyword-based signal extraction. Returns (urgency, threat, concession) deltas."""
        urgency_delta = sum(
            0.15 for kw in _URGENCY_UP if self._term_in_text(text_lower, kw)
        )
        urgency_delta -= sum(
            0.1 for kw in _URGENCY_DOWN if self._term_in_text(text_lower, kw)
        )
        threat_delta = sum(
            0.2 for kw in _THREAT_WORDS if self._term_in_text(text_lower, kw)
        )
        concession_delta = 0.0
        for kw in _CONCESSION_WORDS:
            start = text_lower.find(kw)
            while start >= 0:
                prefix = text_lower[max(0, start - 28):start]
                negated = any(
                    marker in prefix
                    for marker in (
                        "not ",
                        "n't ",
                        "cannot ",
                        "can't ",
                        "cant ",
                        "won't ",
                        "wont ",
                        "no ",
                    )
                )
                if not negated:
                    concession_delta += 0.15
                start = text_lower.find(kw, start + len(kw))
        return urgency_delta, threat_delta, concession_delta

    @staticmethod
    def _bounded_float(value: Any, default: float = 0.0) -> float:
        """Convert an LLM-provided value to a clamped [0, 1] float."""
        try:
            x = float(value)
        except Exception:
            return default
        return max(0.0, min(1.0, x))

    # 2026-05-15: time-to-live (in partner-message turns) for a priority
    # issue mention. Re-mentions reset the counter; otherwise it decays.
    PRIORITY_TTL: int = 5

    def _mark_priority_issue(self, i: int) -> None:
        """Mark issue ``i`` as partner-prioritised and reset its TTL.

        Used in both the LLM-parsed and keyword paths so a re-mention
        in the same partner message refreshes the counter rather than
        leaving a stale boost in place forever.
        """
        self._text_priority_issues.add(i)
        self._text_priority_freshness[i] = self.PRIORITY_TTL

    def _decay_priority_issues(self) -> None:
        """Age out text-mentioned priority issues by one partner turn."""
        expired: list[int] = []
        for idx, ttl in list(self._text_priority_freshness.items()):
            new_ttl = ttl - 1
            if new_ttl <= 0:
                expired.append(idx)
            else:
                self._text_priority_freshness[idx] = new_ttl
        for idx in expired:
            self._text_priority_freshness.pop(idx, None)
            self._text_priority_issues.discard(idx)

    def _decay_text_channels(self) -> None:
        """Decay transient text signals when a partner turn has no text."""
        self._text_urgency = max(-1.0, min(1.0, self._text_urgency * TEXT_SIGNAL_DECAY))
        self._text_threat_level = max(
            0.0, min(1.0, self._text_threat_level * TEXT_SIGNAL_DECAY)
        )
        self._text_concession_signal = max(
            0.0, min(1.0, self._text_concession_signal * TEXT_SIGNAL_DECAY)
        )

    def _refresh_text_threshold_adj(self) -> None:
        """Recompute the scalar acceptance nudge from text-signal state."""
        self._text_threshold_adj = (
            self._text_urgency * TEXT_URGENCY_WEIGHT
            - self._text_threat_level * TEXT_THREAT_WEIGHT
            + self._text_concession_signal * TEXT_CONCESSION_WEIGHT
        )

    def _mark_text_issue_mentions(
        self, text_lower: str, priorities: list[Any] | None = None
    ) -> None:
        """Mark issue names mentioned in text or LLM priority strings."""
        try:
            issues = self.nmi.issues if self.nmi else ()
            priority_terms = [
                str(p).lower()
                for p in (priorities or [])
                if isinstance(p, (str, int, float))
            ]
            for i, issue in enumerate(issues):
                name = (issue.name or "").lower()
                if not name:
                    continue
                if name in text_lower or any(
                    name in p or p in name for p in priority_terms
                ):
                    self._mark_priority_issue(i)
        except Exception:
            pass

    @staticmethod
    def _message_has_parse_trigger(text: str, text_lower: str) -> bool:
        """H3: true when a partner message is worth a structured LLM parse.

        Triggers: any keyword from the urgency/threat/concession/fairness
        sets, a question mark (they are asking us something), or a digit
        (they are referencing specific numbers). Procedural replies
        ("ok", "sure, go ahead") match none of these and skip the LLM.
        """
        if "?" in text:
            return True
        if any(ch.isdigit() for ch in text):
            return True
        for kw in _URGENCY_UP:
            if AgoraNegotiator._term_in_text(text_lower, kw):
                return True
        for kw in _URGENCY_DOWN:
            if AgoraNegotiator._term_in_text(text_lower, kw):
                return True
        for kw in _THREAT_WORDS:
            if AgoraNegotiator._term_in_text(text_lower, kw):
                return True
        for kw in _CONCESSION_WORDS:
            if AgoraNegotiator._term_in_text(text_lower, kw):
                return True
        for kw in _FAIRNESS_CLAIM_WORDS:
            if AgoraNegotiator._term_in_text(text_lower, kw):
                return True
        return False

    def _parse_text_signals(self, state: SAOState) -> None:
        """Extract urgency, threats, concession signals from human text.

        Tries LLM-based structured parsing first, falls back to keywords.
        H3 (2026-04-14): the LLM call is gated on message length and
        trigger keywords. Short procedural replies skip the LLM and use
        the keyword fallback directly, saving 25-70s per turn on Ollama.
        """
        raw_text = self._extract_received_text(state)
        if not raw_text:
            # HANI can send offer-only turns after an earlier text turn.
            # Let the transient text state decay rather than freezing an
            # old urgency/threat/concession/priority signal indefinitely.
            self._decay_priority_issues()
            self._decay_text_channels()
            self._refresh_text_threshold_adj()
            return
        # C2: sanitize before any LLM ingestion or keyword matching.
        text = self._sanitize_partner_text(raw_text)
        if not self._first_partner_text:
            self._first_partner_text = text

        # Track conversation history for LLM context.
        self._conversation_history.append(f"Partner: {text[:200]}")
        if len(self._conversation_history) > self._max_history:
            self._conversation_history.pop(0)

        # 2026-05-15: decay priority-issue freshness counters once per
        # partner-message turn. Issues re-mentioned in this turn will be
        # bumped back to PRIORITY_TTL below; everything else drifts toward
        # expiry and is dropped from `_text_priority_issues` when it hits 0.
        self._decay_priority_issues()

        text_lower = text.lower()

        # H3: pre-gate the LLM structured parse. On short/procedural
        # turns the keyword fallback has all the signal the LLM would,
        # at zero latency.
        llm_result = None
        word_count = len(text.split())
        should_llm = (
            word_count >= 6
            and self._message_has_parse_trigger(text, text_lower)
        )
        if should_llm:
            llm_result = self._llm_parse_text(text)
            self._parse_gate_stats["llm_calls"] += 1
        else:
            self._parse_gate_stats["keyword_fallback"] += 1
        if llm_result is not None:
            urgency_delta = self._bounded_float(llm_result.get("urgency", 0.0))
            threat_delta = self._bounded_float(llm_result.get("threat", 0.0))
            concession_delta = self._bounded_float(
                llm_result.get("concession", 0.0)
            )

            # LLM returns absolute values [0,1], blend with existing state.
            self._text_urgency = max(-1.0, min(1.0,
                self._text_urgency * 0.5 + urgency_delta * 0.5))
            self._text_threat_level = max(0.0, min(1.0,
                self._text_threat_level * TEXT_SIGNAL_DECAY + threat_delta * 0.3))
            self._text_concession_signal = max(0.0, min(1.0,
                self._text_concession_signal * TEXT_SIGNAL_DECAY + concession_delta * 0.3))

            # Tone tracking for Idea 7.
            self._partner_tone = llm_result.get("tone", "neutral")
            self._partner_length = llm_result.get("length", "moderate")
            # #4 Register isn't in LLM schema — derive via keywords.
            _, _, self._partner_register = self._keyword_tone_length_register(text)

            # Priority issues from LLM.
            priorities = llm_result.get("priorities", [])
            if not isinstance(priorities, list):
                priorities = []
        else:
            # Fallback to keywords.
            u_delta, t_delta, c_delta = self._keyword_parse_text(text_lower)
            # 2026-05-15: apply TEXT_SIGNAL_DECAY to urgency too. Pre-fix
            # the keyword path was the only signal channel without decay
            # so a single early urgency-keyword hit (e.g. "deadline") kept
            # _text_urgency pinned at the top of [-1,1] for the rest of
            # the match, even if the partner stopped sounding pressured.
            self._text_urgency = max(-1.0, min(1.0,
                self._text_urgency * TEXT_SIGNAL_DECAY + u_delta))
            self._text_threat_level = max(0.0, min(1.0,
                self._text_threat_level * TEXT_SIGNAL_DECAY + t_delta))
            self._text_concession_signal = max(0.0, min(1.0,
                self._text_concession_signal * TEXT_SIGNAL_DECAY + c_delta))

            # #4 Keyword-based tone + length + register (so mirroring
            # instructions fire even when structured LLM parsing fails).
            tone, length, register = self._keyword_tone_length_register(text)
            self._partner_tone = tone
            self._partner_length = length
            self._partner_register = register

            priorities = []

        # Deterministic issue-name detection always runs, even after a
        # successful LLM parse. The parser can omit or paraphrase
        # priorities ("the price"), but literal issue names in the
        # partner's text are reliable signal.
        self._mark_text_issue_mentions(text_lower, priorities)

        # Composite threshold adjustment.
        self._refresh_text_threshold_adj()

        # --- Counter-manipulation detection (Idea 7) ---
        # If the current offer gives us VERY low utility AND the text
        # claims it's "fair"/"reasonable", we're being manipulated.
        # Harden position instead of softening.
        if state.current_offer is not None and self.ufun is not None:
            try:
                offer_util = self._normalize(float(self.ufun(state.current_offer)))
            except Exception:
                offer_util = 0.5
            has_fairness_claim = any(
                self._term_in_text(text_lower, kw) for kw in _FAIRNESS_CLAIM_WORDS
            )
            if offer_util < 0.2 and has_fairness_claim:
                self._manipulation_detected = True
                # Override concession signal — the "fair" claim is not genuine.
                self._text_threshold_adj += 0.04  # harden our position

    # ------------------------------------------------------------------
    # Core overrides
    # ------------------------------------------------------------------

    def respond(
        self, state: SAOState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        """Custom acceptance + opponent/text tracking, then LLM text."""
        # Defensive: if the partner sent a malformed offer (e.g. tuple
        # containing None as a wildcard), negmas's linear ufun will crash
        # at `float(self.ufun(offer))`. Treat such offers as not-valid
        # inputs — reject without further processing.
        if not self._is_valid_partner_offer(state.current_offer):
            return ResponseType.REJECT_OFFER

        # --- Update all trackers ---
        self._update_phase(state)
        self._track_opponent_offer(state)
        self._parse_text_signals(state)

        # #7 Personality classification. Gated on n_steps > 20: short
        # negotiations (e.g. Trade with n_steps=3-5 in the HANI GUI)
        # can't afford the extra 25-70s LLM call.
        # 2026-05-15: first-call fires as soon as we have ≥2 partner
        # offers plus a conversation line — previously the floor was
        # state.step >= 3 AND step - (-1) >= 8 which only fired at step
        # 7, costing 3-4 proposals of un-adapted text style. Subsequent
        # re-classifications still run every 8 steps as before.
        _n_steps = getattr(self.nmi, "n_steps", None) if self.nmi else None
        _long_enough = _n_steps is None or _n_steps > 20
        _first_time = self._opp_personality_updated_at < 0
        if _long_enough:
            fire = False
            if _first_time:
                if (
                    state.step >= 3
                    and len(self._opp_offers) >= 2
                    and len(self._conversation_history) >= 1
                ):
                    fire = True
            elif state.step - self._opp_personality_updated_at >= 8:
                fire = True
            if fire:
                self._opp_personality_updated_at = state.step
                try:
                    result = self._llm_classify_personality()
                    if result is not None:
                        self._opp_personality, self._opp_personality_confidence = result
                except Exception:
                    pass

        # --- END detection: pathological opponent with no concession ---
        # If we've seen many offers all at or below reservation, waiting
        # longer is unlikely to help. Signal assertiveness by ending.
        if self._should_end(state):
            return ExtendedResponseType(
                response=ResponseType.END_NEGOTIATION,
                data={"text": "I appreciate the discussion, but I don't think we'll find common ground on these terms. Good luck with your search."},
            )

        # --- Custom acceptance check ---
        if state.current_offer is not None and self._should_accept(state):
            received_text = self._extract_received_text(state)
            try:
                generated_text = self._generate_text(
                    state, "accept", state.current_offer, received_text
                )
            except Exception:
                generated_text = ""
            return ExtendedResponseType(
                response=ResponseType.ACCEPT_OFFER,
                data={"text": generated_text},
            )

        # --- Reject: delegate to base for counter-offer text ---
        base_response = super(AgoraNegotiator, self).respond(state, source=source)

        # Override if the base wanted to accept but our policy says no.
        if isinstance(base_response, ExtendedResponseType):
            if base_response.response == ResponseType.ACCEPT_OFFER:
                return self._reject_override_response(state)
            return base_response

        if base_response == ResponseType.ACCEPT_OFFER:
            return self._reject_override_response(state)
        return base_response

    def _reject_override_response(self, state: SAOState) -> ExtendedResponseType:
        """Build a clean reject when base policy wanted to accept.

        Never preserve base acceptance text in the response data: the
        protocol decision and dialogue should agree.
        """
        received_text = self._extract_received_text(state)
        try:
            text = self._generate_text(state, "reject", state.current_offer, received_text)
        except Exception:
            text = ""
        if not text:
            text = self._template_fallback_text(state, "reject", state.current_offer)
        return ExtendedResponseType(
            response=ResponseType.REJECT_OFFER,
            data={"text": text},
        )

    def _is_valid_partner_offer(self, offer: Outcome | None) -> bool:
        """Validate partner offers before they update internal state."""
        if offer is None:
            return True
        try:
            if any(v is None for v in offer):
                return False
        except Exception:
            return False
        try:
            if self.nmi is not None and hasattr(self.nmi, "outcome_space"):
                return bool(self.nmi.outcome_space.is_valid(tuple(offer)))
        except Exception:
            pass
        return True

    @staticmethod
    def _sanitize_partner_text(text: str) -> str:
        """C2: strip prompt-injection patterns before feeding to any LLM.

        Case-insensitive replacement of known patterns with a marker.
        Not perfect against novel attacks, but catches the common playbook.
        """
        if not text:
            return text
        out = text
        lo = out.lower()
        for pat in _INJECTION_PATTERNS:
            if pat in lo:
                # Case-insensitive replacement via lo-index mapping.
                new_parts: list[str] = []
                i = 0
                lo = out.lower()  # refresh after each replace
                while True:
                    j = lo.find(pat, i)
                    if j < 0:
                        new_parts.append(out[i:])
                        break
                    new_parts.append(out[i:j])
                    new_parts.append("[redacted]")
                    i = j + len(pat)
                out = "".join(new_parts)
                lo = out.lower()
        return out

    def _is_opp_model_confident(self) -> bool:
        """C1: true only when we have a usable opponent model.

        Requires either (a) >= MIN_OFFERS_FOR_MODEL frequency-inferred
        offers, or (b) cold-read weights successfully seeded. Prevents
        propose() from entering the opp-aware path with weights that
        default to 0.5, which would produce essentially-random proposals.
        """
        if not self._opp_weights:
            return False
        n_off = len(self._opp_offers)
        if n_off >= MIN_OFFERS_FOR_MODEL:
            return True
        # Cold-read path: we have weights from LLM but few offers.
        if self._cold_read_done and n_off >= 1:
            return True
        return False

    def _estimate_opponent_utility(self, outcome: tuple) -> float:
        """Estimate how good an outcome is for the opponent.

        Uses frequency-based weights: for each issue, if the outcome
        matches what the opponent usually demands, that's good for them.
        Returns a score in [0, 1] — higher means better for opponent.

        Cached: called 1000+ times per match during Pareto/Nash scoring
        on a 5000-outcome table. We cache the non-text-boost portion
        keyed by (outcome, n_opp_offers) — the signature invalidates
        automatically when new offer data arrives (via `_track_opponent_offer`
        incrementing `len(self._opp_offers)`). The text-priority boost
        is applied on top (cheap) so changes to `_text_priority_issues`
        don't require cache invalidation.
        """
        if not self._opp_weights or not self._opp_issue_freq:
            return 0.5

        sig = len(self._opp_offers)
        cache = self._opp_util_cache
        if cache.get("_sig") != sig:
            cache.clear()
            cache["_sig"] = sig

        key = outcome if isinstance(outcome, tuple) else tuple(outcome)
        cached = cache.get(key)
        if cached is None:
            score = 0.0
            weights = self._opp_weights
            freq_map = self._opp_issue_freq
            for i, val in enumerate(key):
                freq = freq_map.get(i)
                if not freq:
                    continue
                total = sum(freq.values())
                if total <= 0:
                    continue
                w = weights[i] if i < len(weights) else 0.0
                score += w * (freq.get(val, 0) / total)
            cache[key] = score
        else:
            score = cached

        # Text-priority boost applied on top (not cached so changes to
        # `_text_priority_issues` don't require cache invalidation).
        if self._text_priority_issues:
            for i in self._text_priority_issues:
                if i < len(outcome):
                    freq = self._opp_issue_freq.get(i)
                    if freq:
                        total = sum(freq.values())
                        if total > 0:
                            score += 0.15 * (freq.get(outcome[i], 0) / total)
        return max(0.0, min(1.0, score))

    def _estimate_accept_prob(self, candidate: tuple) -> float:
        """H1: saturating acceptance-probability model for a candidate offer.

        Logistic of the gap between the candidate's estimated opponent
        utility and the running max opp_util over the outcomes we have
        already proposed. Rationale (ASTRA — Kwon et al., EMNLP 2025):
        the opponent has already observed our best-to-them offer; a new
        candidate's acceptance probability saturates in the gap rather
        than in raw opp_util. The pre-H1 `0.7*our + 0.3*opp` scorer
        treats opp_util linearly and over-rewards already-generous
        candidates; the logistic form concentrates the decision signal
        near the threshold where the probability actually changes.

        Returns a probability in [0, 1]. Numerically clamped at ±20 in
        the exponent to avoid overflow.
        """
        import math
        opp = self._estimate_opponent_utility(candidate)
        ref = self._running_max_opp_of_our_props
        x = self._astra_k * (opp - ref) + self._astra_b
        if x >= 20.0:
            return 1.0
        if x <= -20.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Opponent-aware proposing from pre-computed outcome table.

        Strategy: from all outcomes at our current aspiration level,
        pick the one the opponent is most likely to accept (highest
        estimated opponent utility). Falls back to base Boulware +
        swap search if the outcome table isn't available.
        """
        # Step-based timing override: when n_steps is set, drive the Boulware
        # aspiration off step progress rather than wall-clock. This keeps the
        # concession curve stable even when LLM text generation eats wall-clock
        # budget (e.g., Ollama runs where each message costs 20-60s and would
        # otherwise fast-forward relative_time).
        state_for_base = state
        eff_t = float(state.relative_time) if state.relative_time is not None else 0.0
        try:
            eff_t = self._effective_t(state, harden=True)
            state_for_base = attrs.evolve(state, relative_time=eff_t)
        except Exception:
            state_for_base = state

        # Get base proposal (needed for LLM text wrapping).
        base_proposal = super().propose(state_for_base, dest=dest)

        # #5 Anchor-high opener: on the very first step, replace the
        # max-utility proposal with the ~99th-percentile outcome. Signals
        # "I have room to concede" without giving up a meaningful fraction
        # of utility (typical gap: <0.01). Humans reward non-maximalist
        # openers with better perception scores.
        if (
            state.step == 0
            and self._outcome_table
            and len(self._outcome_table) >= 10
            and base_proposal is not None
        ):
            anchor_idx = max(1, len(self._outcome_table) // 100)
            anchor_outcome, _ = self._outcome_table[anchor_idx]
            if isinstance(base_proposal, ExtendedOutcome):
                base_proposal = ExtendedOutcome(
                    outcome=anchor_outcome, data=base_proposal.data
                )
            else:
                base_proposal = anchor_outcome

        # Only use opponent-aware propose when opponent is clearly conceding.
        # Fixed-threshold opponents (SimpleNeg-style) cannot be "targeted"
        # from frequency data alone — we don't observe their ufun. Two
        # stalemate-breaker attempts (2026-04-12) regressed SimpleNeg from
        # 88% → 12-25%. Real HAN opponents adapt, so this gate is safe.
        opp_conceding = self._compute_concession_rate() > 0.03
        if (
            self._outcome_table
            and self._is_opp_model_confident()  # C1: no opp-aware w/ stale weights
            and self.ufun is not None
            and base_proposal is not None
            and opp_conceding
        ):
            # Refresh Pareto front cache (cheap if unchanged).
            self._recompute_pareto_front()

            # Get the utility floor from the base proposal.
            base_outcome = (
                base_proposal.outcome
                if isinstance(base_proposal, ExtendedOutcome)
                else base_proposal
            )
            if base_outcome is not None:
                try:
                    base_util = self._normalize(float(self.ufun(base_outcome)))
                except Exception:
                    base_util = 0.0

                # Search outcomes in a narrow band at/above base utility.
                # Never worse for us; slightly better for opponent.
                # Pareto-optimal outcomes get a tie-breaker bonus.
                band_hi = min(1.0, base_util + 0.03)
                best_outcome: tuple | None = None
                best_opp_score = -1.0

                for outcome, our_util in self._outcome_table:
                    if our_util < base_util:
                        break  # sorted desc, done
                    if our_util > band_hi:
                        continue
                    # Skip outcomes we've already proposed (diversity).
                    if outcome in self._proposed_outcomes:
                        continue
                    opp_score = self._estimate_opponent_utility(outcome)
                    if opp_score > best_opp_score:
                        best_opp_score = opp_score
                        best_outcome = outcome

                # Also compute Nash bargaining candidate.
                nash_outcome = self._find_nash_outcome(min_our_util=base_util)

                # Also try Pareto walk from opponent's last offer (if we have one).
                pareto_outcome = None
                if self._opp_offers:
                    last_opp = self._opp_offers[-1]
                    pareto_outcome = self._pareto_walk_from_offer(
                        last_opp, min_our_util=base_util, t=eff_t
                    )

                # Precomputed Pareto-front candidate: best opp_util Pareto
                # outcome with our_util >= base_util and not yet proposed.
                pareto_front_outcome = None
                if self._pareto_outcomes:
                    best_p_opp = -1.0
                    for outcome, our_u in self._outcome_table:
                        if our_u < base_util:
                            break
                        if outcome not in self._pareto_outcomes:
                            continue
                        if outcome in self._proposed_outcomes:
                            continue
                        opp_u = self._estimate_opponent_utility(outcome)
                        if opp_u > best_p_opp:
                            best_p_opp = opp_u
                            pareto_front_outcome = outcome

                # Choose the best candidate by combined (our + opp) score.
                candidates = [
                    (best_outcome, "opp_aware"),
                    (nash_outcome, "nash"),
                    (pareto_outcome, "pareto"),
                    (pareto_front_outcome, "pareto_front"),
                ]
                candidates = [(c, tag) for c, tag in candidates if c is not None]
                if candidates:
                    # H1 (2026-04-15, ASTRA): score by λ_u * our_util
                    # + λ_a * accept_prob. The gate `_is_opp_model_confident()`
                    # above is a prerequisite for entering this branch,
                    # so accept_prob is well-defined here. The fallback
                    # scorer below is retained as a safety net in case
                    # a future refactor widens the gate.
                    confident = self._is_opp_model_confident()
                    lam_u = self._astra_lambda_u
                    lam_a = self._astra_lambda_a

                    def _score(outcome: tuple) -> float:
                        try:
                            our = self._normalize(float(self.ufun(outcome)))
                        except Exception:
                            return -1.0
                        if confident:
                            accept_p = self._estimate_accept_prob(outcome)
                            return lam_u * our + lam_a * accept_p
                        opp = self._estimate_opponent_utility(outcome)
                        return 0.7 * our + 0.3 * opp

                    chosen, _ = max(candidates, key=lambda p: _score(p[0]))

                    if chosen != tuple(base_outcome):
                        # Also try swap search on top of chosen.
                        improved = self._find_win_win_offer(
                            chosen, state, min_our_util_norm=base_util
                        )
                        result = improved if improved is not None else chosen

                        if isinstance(base_proposal, ExtendedOutcome):
                            return self._track_proposal(
                                ExtendedOutcome(outcome=result, data=base_proposal.data)
                            )
                        return self._track_proposal(result)

        # Fallback: base proposal + swap search.
        if base_proposal is None or not self._opp_weights or self.ufun is None:
            return base_proposal

        if isinstance(base_proposal, ExtendedOutcome):
            outcome = base_proposal.outcome
            data = base_proposal.data
        else:
            outcome = base_proposal
            data = None

        if outcome is None:
            return base_proposal

        try:
            base_util = self._normalize(float(self.ufun(tuple(outcome))))
        except Exception:
            base_util = None
        improved = self._find_win_win_offer(
            tuple(outcome), state, min_our_util_norm=base_util
        )
        if improved is not None and improved != tuple(outcome):
            if isinstance(base_proposal, ExtendedOutcome):
                return self._track_proposal(
                    ExtendedOutcome(outcome=improved, data=data)
                )
            return self._track_proposal(improved)

        # Diversity: if the base outcome has already been proposed and we have
        # an outcome table, pick an alternate at near-identical utility that we
        # haven't offered yet. Helps avoid repeating the same offer every step
        # when the opponent isn't conceding (boulware's iso-utility band).
        outcome_t = tuple(outcome)
        if outcome_t in self._proposed_outcomes and self._outcome_table:
            try:
                base_util = self._normalize(float(self.ufun(outcome_t)))
            except Exception:
                base_util = None
            if base_util is not None:
                band_lo = max(0.0, base_util - 0.02)
                band_hi = min(1.0, base_util + 0.02)
                for cand, cand_util in self._outcome_table:
                    if cand_util > band_hi:
                        continue
                    if cand_util < band_lo:
                        break
                    if cand in self._proposed_outcomes:
                        continue
                    if isinstance(base_proposal, ExtendedOutcome):
                        return self._track_proposal(
                            ExtendedOutcome(outcome=cand, data=data)
                        )
                    return self._track_proposal(cand)

        return self._track_proposal(base_proposal)

    def _track_proposal(self, proposal):
        """Record proposal and compute concession diffs for framing."""
        if proposal is None:
            return proposal
        outcome = (
            proposal.outcome if isinstance(proposal, ExtendedOutcome) else proposal
        )
        if outcome is None:
            return proposal

        current = tuple(outcome)
        self._concession_issues = []
        self._request_issues = []

        if self._last_proposal is not None and self.nmi:
            try:
                issues = self.nmi.issues
                prev = self._last_proposal
                for i in range(min(len(current), len(prev), len(issues))):
                    if current[i] != prev[i]:
                        # Check if this change is good or bad for us.
                        try:
                            # Swap just this issue to measure impact.
                            test_old = list(current)
                            test_old[i] = prev[i]
                            u_new = float(self.ufun(current))
                            u_old = float(self.ufun(tuple(test_old)))
                            name = issues[i].name or f"issue {i}"
                            if u_new < u_old:
                                # We gave up utility on this issue (concession).
                                self._concession_issues.append(name)
                            else:
                                # We gained utility (ask them to move on this).
                                self._request_issues.append(name)
                        except Exception:
                            pass
            except Exception:
                pass

        self._last_proposal = current
        # I4: maintain LRU cap. When the deque overflows, pop oldest from
        # both the order queue and the set. No-op for repeats.
        if current not in self._proposed_outcomes:
            if len(self._proposed_order) == self._proposed_order.maxlen:
                evicted = self._proposed_order[0]  # will be dropped on append
                self._proposed_outcomes.discard(evicted)
            self._proposed_order.append(current)
            self._proposed_outcomes.add(current)
        # H1: update windowed max opp_util over our recent proposals.
        # Cheap because `_estimate_opponent_utility` is cached on
        # (outcome, n_opp_offers). No-op before the opp model is ready.
        # 2026-05-15: window-based instead of monotone — see ASTRA_WINDOW
        # comment in _init_state for the saturation rationale.
        if self._opp_weights:
            try:
                opp_u = self._estimate_opponent_utility(current)
                self._opp_of_our_props_window.append(opp_u)
                self._running_max_opp_of_our_props = max(
                    self._opp_of_our_props_window
                )
            except Exception:
                pass
        return proposal

    # ------------------------------------------------------------------
    # LLM prompt overrides
    # ------------------------------------------------------------------

    # Template-fallback phrases for when the LLM is unavailable or
    # returns empty. Combined opener (from rotating pool) + middle +
    # ender yields thousands of distinct messages without any LLM cost.
    _TEMPLATE_MIDDLES_ACCEPT = (
        "this works for me",
        "I'm happy with this",
        "that lands in the range I can accept",
        "I think we've found common ground",
        "this looks fair to both of us",
    )
    _TEMPLATE_MIDDLES_PROPOSE = (
        "here's what I can offer",
        "this is where I can meet you",
        "consider this combination",
        "my counter on the terms we've been discussing",
        "this balances both sides",
    )
    _TEMPLATE_MIDDLES_REJECT = (
        "I'm not quite there on these terms",
        "we're still a little apart",
        "I need to counter on this one",
        "can we try moving on a couple of items",
        "a few numbers still feel off for me",
    )
    _TEMPLATE_ENDERS = (
        "let me know what you think.",
        "open to your adjustments.",
        "curious how this reads on your side.",
        "happy to keep iterating.",
        "hope this moves us forward.",
        "looking forward to your reply.",
    )

    def _template_fallback_text(
        self, state: SAOState, action: str, outcome: Outcome | None = None
    ) -> str:
        """Deterministic, LLM-free text used when _generate_text returns
        empty or raises. Uses the existing step-indexed opener pool for
        diversity and picks a middle clause based on the action.
        """
        step = getattr(state, "step", 0) or 0
        persuasion_pool = (
            "What if we try this —",
            "Here's an idea —",
            "Let me suggest something —",
            "How about this —",
            "Building on what we discussed —",
            "One thought —",
        )
        if action == "accept":
            accept_pool = (
                "That works for me.",
                "I think we have a deal.",
                "I'm happy with this.",
                "This lands well for me.",
                "That feels fair to close on.",
            )
            return accept_pool[step % len(accept_pool)]
        opener = persuasion_pool[step % len(persuasion_pool)]
        if action in ("propose",):
            middles = self._TEMPLATE_MIDDLES_PROPOSE
        else:
            middles = self._TEMPLATE_MIDDLES_REJECT
        middle = middles[step % len(middles)]
        ender = self._TEMPLATE_ENDERS[step % len(self._TEMPLATE_ENDERS)]
        return f"{opener} {middle}. {ender.capitalize()}"

    @staticmethod
    def _needs_remediation(draft: str) -> bool:
        """H2 pre-filter: true when a draft is worth a norm-compliance pass.

        Fires on either a hostile-word hit (same vocab the keyword parser
        uses) or a message long enough (> 250 chars) to plausibly contain
        a contradiction or a hallucinated fact. Keeps the remediator off
        the critical path for the common case of a clean short draft.
        """
        if not draft:
            return False
        if len(draft) > 250:
            return True
        draft_lower = draft.lower()
        for w in _HOSTILE_WORDS:
            if AgoraNegotiator._term_in_text(draft_lower, w):
                return True
        return False

    def _remediate_text(
        self,
        draft: str,
        action: str,
        received_text: str | None = None,
    ) -> str:
        """H2: single-shot LLM rewrite pass to strip norm violations.

        Returns the draft unchanged if the remediator judges it clean or
        if the LLM call fails. Issue names, yes/no, accept/reject must be
        preserved verbatim per the remediator's system prompt; numbers
        are explicitly forbidden (consistent with hallucination fix #25).

        The call uses the same side-channel Ollama model as the other
        helper LLMs, temperature 0 and max_tokens=200 to keep it bounded.
        """
        import json as _json

        if not draft:
            return draft
        self._remediation_stats["checked"] += 1

        user_parts = [
            f"Action this turn: {action}.",
            f"Draft: {draft}",
        ]
        if received_text:
            user_parts.append(
                f"Partner's most recent message: {received_text[:400]}"
            )
        user_msg = "\n".join(user_parts)

        try:
            import litellm
            response = litellm.completion(
                model=self._side_channel_model(),
                messages=[
                    {"role": "system", "content": _REMEDIATOR_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=self._side_channel_timeout,
                num_retries=0,
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            parsed = _json.loads(raw)
            rewritten = parsed.get("text") if isinstance(parsed, dict) else None
            if isinstance(rewritten, str) and rewritten.strip():
                if rewritten.strip() != draft.strip():
                    self._remediation_stats["rewritten"] += 1
                    return rewritten.strip()
                return draft
        except Exception:
            pass
        return draft

    def _generate_text(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Wrap the parent LLM text generator with a template fallback
        and the H2 remediator post-pass.

        If the LLM call raises or returns an empty/whitespace-only
        string, deterministically synthesise a message from rotating
        phrase pools. Then, on drafts flagged by `_needs_remediation`
        (hostile tokens or length > 250), run a single LLM compliance
        pass that may rewrite the message to strip aggression /
        contradiction / hallucination while preserving issue names and
        yes/no/accept/reject verbatim.
        """
        safe_received_text = (
            self._sanitize_partner_text(received_text) if received_text else received_text
        )
        try:
            text = super()._generate_text(state, action, outcome, safe_received_text)
        except Exception:
            text = ""
        if not (text and text.strip()):
            text = self._template_fallback_text(state, action, outcome)
        if self._needs_remediation(text):
            text = self._remediate_text(text, action, safe_received_text)
        self._log_text_event(state, action, text, safe_received_text)
        return text

    def _build_system_prompt(self) -> str:
        if self._phase == "deception":
            return DECEPTION_SYSTEM_PROMPT
        return PERSUASION_SYSTEM_PROMPT

    def _build_user_message(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Inject strategic instructions based on negotiation phase."""
        base_msg = super()._build_user_message(
            state, action, outcome, received_text
        )

        instructions: list[str] = []
        try:
            t = self._acceptance_t(state)
        except Exception:
            t = float(state.relative_time) if state.relative_time is not None else 0.0

        # --- Rotating opener to force diversity ---
        # qwen3:4b has limited stylistic range; giving it a specific
        # starter phrase each round prevents repetitive openings.
        # #8 Retrieval-augmented text: larger phrase corpus with
        # personality-tailored pools. Selected by personality when known,
        # falling back to phase-based pool.
        _openers_persuasion = [
            "What if we try this —",
            "Here's an idea —",
            "So, thinking about what you need —",
            "Let me suggest something —",
            "How about this approach —",
            "I've been thinking about your priorities —",
            "Here's where I can meet you —",
            "Tell you what —",
            "Building on what we discussed —",
            "Alright, here's my take —",
            "Here's what occurred to me —",
            "I wonder if this works —",
            "One thought —",
            "Could we try —",
            "What feels right here —",
        ]
        _openers_deception = [
            "Let me be honest with you —",
            "Here's what I can realistically offer —",
            "The way I see it —",
            "I want to be upfront —",
            "Here's my best position —",
            "I want to keep this practical —",
            "Let me be clear —",
            "Realistically speaking —",
            "Here's where things stand —",
            "Here is the clearest path I see —",
            "To be direct but fair —",
            "Here's the realistic picture —",
            "I think we should focus on what can close —",
            "Here's what I actually need —",
            "Let's be direct —",
        ]
        _openers_collaborative = [
            "I hear you — here's where I think we can meet —",
            "That makes sense to me — let's build on it —",
            "I like where you're going — what about —",
            "Sounds like we're aligned — here's a tweak —",
            "Good point — let me offer —",
        ]
        _openers_tough = [
            "Here's my position —",
            "This is what I can do —",
            "My numbers are —",
            "Where I stand is —",
            "My offer is —",
        ]
        _openers_manipulative = [
            "I want to keep this simple —",
            "Let's focus on the numbers —",
            "Here's what I can offer today —",
            "Staying focused —",
            "Concretely —",
        ]

        # Select pool based on personality; fall back to phase.
        if self._opp_personality_confidence >= 0.5:
            if self._opp_personality == "collaborative":
                openers = _openers_collaborative + _openers_persuasion
            elif self._opp_personality == "tough":
                openers = _openers_tough + _openers_deception
            elif self._opp_personality == "manipulative":
                openers = _openers_manipulative + _openers_deception
            else:
                openers = (
                    _openers_deception if self._phase == "deception"
                    else _openers_persuasion
                )
        else:
            openers = (
                _openers_deception if self._phase == "deception"
                else _openers_persuasion
            )

        step_idx = state.step if hasattr(state, "step") else 0
        opener = openers[step_idx % len(openers)]
        if action in ("propose", "reject"):
            instructions.append(f'START your message with: "{opener}"')

        if received_text and received_text.strip():
            instructions.append(
                "Briefly acknowledge the partner's latest message before "
                "making your point. Paraphrase the content; do not quote "
                "their exact wording and do not introduce specific numbers."
            )
            if "?" in received_text:
                instructions.append(
                    "If the partner asked a direct question, answer it "
                    "plainly in one short clause before proposing or "
                    "countering."
                )

        # --- Offer values for the LLM to reference ---
        if outcome is not None and self.nmi and action in ("propose", "reject"):
            try:
                issues = self.nmi.issues
                # Check if offer has any non-zero values.
                has_substance = any(
                    outcome[i] != 0 for i in range(min(len(issues), len(outcome)))
                )
                if has_substance:
                    offer_desc = ", ".join(
                        f"{issues[i].name}: {outcome[i]}"
                        for i in range(min(len(issues), len(outcome)))
                        if issues[i].name
                    )
                    # LLM kept hallucinating numbers even with a strict
                    # "verbatim numbers" instruction (Ollama qwen3:4b
                    # 2026-04-14: said "1 Apple, 2 Oranges" when offer
                    # was Apple=0, Orange=4). Safer path: instruct the
                    # LLM NOT to mention specific numbers at all. The
                    # offer values go out on a separate protocol field;
                    # the text is for qualitative persuasion only.
                    # Keep offer_desc in the prompt as CONTEXT for the
                    # LLM's reasoning, but explicitly forbid quoting it.
                    instructions.append(
                        f"Context for your reasoning (do NOT quote these "
                        f"numbers in your message): {offer_desc}. Instead, "
                        f"refer to issues by NAME only and describe changes "
                        f"qualitatively (e.g., 'I've moved further on X', "
                        f"'I can be more flexible on Y'). Never mention any "
                        f"specific numerical quantities."
                    )
                else:
                    # All zeros — don't list zeros, focus on relationship.
                    instructions.append(
                        "This is an early exploratory offer. Do NOT list "
                        "individual zero values — instead focus on understanding "
                        "the partner's priorities and building rapport. Talk "
                        "about the deal in general terms, not specific numbers."
                    )
            except Exception:
                pass
        # Guard against truncation.
        instructions.append(
            "IMPORTANT: Keep your response to exactly 2 sentences. "
            "Do NOT start a third sentence."
        )

        # --- Idea 4: Anchoring for first offer ---
        if len(self._opp_offers) == 0 and action == "propose":
            # Build issue-value description for the LLM.
            anchor_detail = ""
            if outcome is not None and self.nmi:
                try:
                    issues = self.nmi.issues
                    parts = [
                        f"{issues[i].name}: {outcome[i]}"
                        for i in range(min(len(issues), len(outcome)))
                        if issues[i].name
                    ]
                    if parts:
                        anchor_detail = f" The specific terms are: {', '.join(parts)}."
                except Exception:
                    pass
            instructions.append(
                "IMPORTANT: This is the OPENING offer." + anchor_detail +
                " Frame it as the natural, fair starting point — not as "
                "an ambitious position you expect to move from. Justify "
                "each issue value with concrete reasoning. "
                "Do NOT signal willingness to move significantly."
            )

        # --- Idea 3: Strategic questioning (early rounds) ---
        if len(self._opp_offers) <= 1 and action in ("propose", "reject"):
            instructions.append(
                "Include a genuine question asking which aspects of the "
                "deal matter most to the partner. Example: 'I'd love to "
                "understand what's most important to you so we can find "
                "something that works well for both of us.'"
            )

        # --- Idea 2: Concession framing ---
        if self._concession_issues and action == "propose":
            gave = ", ".join(self._concession_issues[:2])
            instructions.append(
                f"You just IMPROVED the offer for the partner on: {gave}. "
                f"EXPLICITLY point this out as a gesture of good faith."
            )
            if self._request_issues:
                want = ", ".join(self._request_issues[:2])
                instructions.append(
                    f"Politely ask the partner to reciprocate by being "
                    f"flexible on: {want}."
                )

        # --- #7 Personality-routed text hints ---
        if self._opp_personality_confidence >= 0.5:
            if self._opp_personality == "collaborative":
                instructions.append(
                    "The partner is collaborative — emphasize shared wins "
                    "and build on their suggestions explicitly."
                )
            elif self._opp_personality == "tough":
                instructions.append(
                    "The partner is tough — keep your language firm and "
                    "confident without being aggressive. Avoid begging."
                )
            elif self._opp_personality == "manipulative":
                instructions.append(
                    "The partner is using pressure tactics — hold your "
                    "position calmly and redirect to concrete numbers. "
                    "Do not apologize or over-explain."
                )
            elif self._opp_personality == "random":
                instructions.append(
                    "The partner is inconsistent — be extra clear and "
                    "specific about your proposal to reduce ambiguity."
                )

        # --- Idea 7 + #4: Tone + length + register matching ---
        # Match partner's communication style. Never mirror hostility —
        # answer hostility with calm measured language instead.
        tone = self._partner_tone
        length = self._partner_length
        register = self._partner_register
        if tone == "hostile":
            tone = "neutral"
        style_parts = []
        if tone == "friendly":
            style_parts.append("be warm and conversational")
        if length == "brief":
            style_parts.append("keep it very short (1-2 sentences)")
        elif length == "verbose":
            style_parts.append("be a bit more detailed than usual")
        if register == "formal":
            style_parts.append("use a formal, professional register")
        elif register == "casual":
            style_parts.append("use a casual, conversational register")
        if self._partner_tone == "hostile":
            style_parts.append(
                "stay calm and measured — do not mirror the partner's "
                "negativity, but acknowledge the frustration"
            )
        if style_parts:
            instructions.append(
                f"Match the partner's style: {', '.join(style_parts)}."
            )

        # --- Idea 1: Phase-specific perception instructions ---
        if action == "accept":
            # Adaptive concession (#2): if the accepted offer notably
            # beat our aspiration, signal recognition of the partner's
            # generosity. Humans reward this acknowledgement with better
            # perception scores.
            generous = False
            try:
                if outcome is not None and self.ufun is not None:
                    offer_u = self._normalize(float(self.ufun(outcome)))
                    asp = self._aspiration_level(t)
                    generous = offer_u >= asp + 0.08
            except Exception:
                generous = False
            if generous:
                instructions.append(
                    "Acknowledge that the partner came in stronger than "
                    "strictly necessary and express gratitude. 'You "
                    "actually offered more than I expected — thank you "
                    "for coming to this generously.' Be warm and specific."
                )
            else:
                instructions.append(
                    "Express genuine warmth and satisfaction. 'I'm really "
                    "happy we found something that works for both of us.' "
                    "Thank the partner for their flexibility."
                )
        elif t < 0.3:
            instructions.append(
                "Express enthusiasm about finding a deal together. "
                "Be warm and curious about their perspective."
            )
        elif t < 0.7:
            instructions.append(
                "Acknowledge the partner's constraints and priorities. "
                "Show empathy: 'I understand this is important to you.'"
            )
        else:
            instructions.append(
                "Express appreciation for how far you've both come. "
                "Frame the negotiation as a collaborative achievement. "
                "'We've made real progress — let's close this out.'"
            )

        # --- Opponent model context ---
        context: list[str] = []
        try:
            if self._opp_weights and self.nmi and self.nmi.issues:
                issues = self.nmi.issues
                ranked = sorted(
                    range(len(self._opp_weights)),
                    key=lambda i: self._opp_weights[i],
                    reverse=True,
                )
                top = [issues[i].name for i in ranked[:2] if issues[i].name]
                if top:
                    context.append(
                        f"The partner seems to care most about: {', '.join(top)}"
                    )
        except Exception:
            pass

        if self._text_urgency > 0.3:
            context.append("The partner appears to be under time pressure.")
        if self._text_concession_signal > 0.3:
            context.append("The partner has signaled willingness to compromise.")
        if self._text_threat_level > 0.3:
            context.append("The partner has hinted at walking away.")

        # Conversation history for coherence.
        if len(self._conversation_history) > 1:
            context.append(
                "Recent conversation:\n"
                + "\n".join(self._conversation_history[-3:])
            )

        # Assemble.
        parts = [base_msg]
        if instructions:
            parts.append("\n\nINSTRUCTIONS:\n" + "\n".join(f"- {x}" for x in instructions))
        if context:
            parts.append("\n\nCONTEXT:\n" + "\n".join(f"- {x}" for x in context))
        return "".join(parts)
