import statistics

from negmas.sao.common import SAOResponse, SAOState, ResponseType
from negmas.sao.negotiators.base import SAOCallNegotiator
from negmas.gb.components.genius.models import GScalableBayesianModel
from negmas_llm import OllamaNegotiator

_W = {
    1: [1],
    2: [0.25, 0.75],
    3: [0.11, 0.22, 0.66],
    4: [0.05, 0.15, 0.3, 0.5],
}

_BEHAVIOUR_THRESHOLD = 5

class HybridPisaNegotiator(OllamaNegotiator):
    #Heuristic bidder (Keskin2021 + Bayesian opponent model) that uses the
    #built-in OllamaNegotiator only for its LLM machinery (``_call_llm``).
    _opp_behaviour = "silent"

    def __init__(self, **kwargs):
        kwargs.setdefault("model", "qwen3:4b-instruct")
        kwargs.setdefault("api_base", None)
        super().__init__(**kwargs)

        # LLM call configuration
        self.temperature = 0.7
        self.max_tokens = 64
        self.timeout = 8.0
        self.num_retries = 0

    def on_preferences_changed(self, changes):
        # Skip the base LLMNegotiator version, which sends the preferences to
        # the LLM.
        SAOCallNegotiator.on_preferences_changed(self, changes)

    def on_negotiation_start(self, state):
        # Skip the base LLMNegotiator version, which sends the preferences to
        # the LLM.
        SAOCallNegotiator.on_negotiation_start(self, state)

        # The strategy operates in a normalized [0,1] utility space so it is
        # scale-invariant
        # the concession curve / acceptance thresholds assume [0,1].
        self._nufun = self.ufun
        if self.ufun is not None:
            try:
                self._nufun = self.ufun.normalize(normalize_reserved_values=True)
            except Exception:
                self._nufun = self.ufun

        rv = self._nufun.reserved_value if self._nufun else 0.0

        self._p0 = 1.0
        self._p1 = 0.8
        self._p2 = max(0.6, rv)
        self._p3 = 0.5
        self._rv = rv

        self._ema_alpha = 0.05
        self._param_targets = {
            "competitive": {"p1": 0.9, "p2": max(0.75, rv), "p3": 0.7},
            "cooperative": {"p1": 0.7, "p2": max(0.55, rv), "p3": 0.35},
            "silent":      {"p1": 0.8, "p2": max(0.6, rv),  "p3": 0.5},
        }

        self._my_bids = []
        self._my_bid_utilities = []
        self._opp_bids = []
        self._opp_bid_utilities = []

        self._outcomes = []
        self._outcome_utilities = []
        self._n_issues = 0

        if self._nufun and self.nmi:
            os = self.nmi.outcome_space
            if os:
                self._n_issues = len(os.issues)
                pairs = [(o, float(self._nufun(o))) for o in os.enumerate_or_sample()]
                pairs.sort(key=lambda x: x[1])
                self._outcomes = [p[0] for p in pairs]
                self._outcome_utilities = [p[1] for p in pairs]

        # One-time domain context for the LLM
        self._domain_context = self._build_domain_context()

        self._opp_model = GScalableBayesianModel(learning_rate=0.1)
        self._opp_model.set_negotiator(self)
        self._opp_behaviour = "silent"

        self._endgame_t = 0.97
        self._endgame_factor = 0.95

        self._llm_warmup_timeout = 60.0
        self._llm_fail_count = 0
        self._llm_fail_limit = 2
        # One-time warmup doubles as the reachability test
        self._llm_enabled = self._warmup_llm()

    def _warmup_llm(self):
        saved = self.timeout
        self.timeout = self._llm_warmup_timeout
        try:
            self._call_llm([{"role": "user", "content": "Reply with: ok"}])
            return True
        except Exception:
            return False
        finally:
            self.timeout = saved

    def _adapt_parameters(self):
        targets = self._param_targets.get(self._opp_behaviour)
        if not targets:
            return
        a = self._ema_alpha
        self._p1 = (1 - a) * self._p1 + a * targets["p1"]
        self._p2 = (1 - a) * self._p2 + a * targets["p2"]
        self._p3 = (1 - a) * self._p3 + a * targets["p3"]

    def _time_based(self, t):
        return (1 - t) ** 2 * self._p0 + 2 * (1 - t) * t * self._p1 + t * t * self._p2

    def _behaviour_based(self, t):
        diffs = [
            self._opp_bid_utilities[i + 1] - self._opp_bid_utilities[i]
            for i in range(len(self._opp_bid_utilities) - 1)
        ]
        if len(diffs) > len(_W):
            diffs = diffs[-len(_W):]
        if not diffs:
            return self._my_bid_utilities[-1]
        delta = sum(d * w for d, w in zip(diffs, _W[len(diffs)]))
        return self._my_bid_utilities[-1] - (self._p3 + self._p3 * t) * delta

    def _calculate_target_utility(self, t):
        target = self._time_based(t)
        if len(self._opp_bid_utilities) > 2 and self._my_bid_utilities:
            target = (1.0 - t * t) * self._behaviour_based(t) + t * t * target
        rv = self._nufun.reserved_value if self._nufun else 0.0
        return max(target, rv)

    def _get_bid_at(self, target, window=0.05):
        if not self._outcomes:
            return self._nufun.best() if self._nufun else None

        pool = []
        for outcome, u in zip(self._outcomes, self._outcome_utilities):
            if target - window <= u <= target + window:
                pool.append((outcome, u))

        if len(pool) < 2:
            pool = []
            for outcome, u in zip(self._outcomes, self._outcome_utilities):
                if target - window * 2 <= u <= target + window * 2:
                    pool.append((outcome, u))

        if not pool:
            best_outcome = None
            best_dist = float("inf")
            for outcome, u in zip(self._outcomes, self._outcome_utilities):
                dist = abs(u - target)
                if dist < best_dist:
                    best_dist = dist
                    best_outcome = outcome
            return best_outcome

        if not self._opp_bids:
            return max(pool, key=lambda p: p[1])[0]

        last_opp_bid = self._opp_bids[-1]
        best_bid = None
        best_score = -1.0
        for outcome, u in pool:
            sim = self._bid_similarity(outcome, last_opp_bid)
            opp_pref = self._opponent_value_score(outcome)
            # Our utility is primary; opponent appeal breaks ties
            score = 0.6 * u + 0.2 * sim + 0.2 * opp_pref
            if score > best_score:
                best_score = score
                best_bid = outcome
        return best_bid

    def _update_opponent_model(self, state, offer):
        try:
            self._opp_model.on_partner_proposal(state, "opponent", offer)
        except Exception:
            pass

    def _opponent_value_score(self, bid):
        if not self._opp_bids:
            return 0.0
        try:
            return float(self._opp_model(bid))
        except Exception:
            return 0.0

    def _bid_similarity(self, bid1, bid2):
        if not bid1 or not bid2:
            return 0.0
        return sum(1 for a, b in zip(bid1, bid2) if a == b) / len(bid1)

    def _detect_behaviour(self):
        if len(self._opp_bids) < _BEHAVIOUR_THRESHOLD:
            return "silent"

        similarities = [
            self._bid_similarity(self._opp_bids[i], self._opp_bids[i + 1]) * 100
            for i in range(len(self._opp_bids) - 1)
        ]
        if len(similarities) < 2:
            return "silent"

        sim_mean = statistics.mean(similarities)
        sim_std = statistics.stdev(similarities) if len(similarities) > 1 else 0.0

        point_diffs = [
            self._opp_bid_utilities[i + 1] - self._opp_bid_utilities[i]
            for i in range(len(self._opp_bid_utilities) - 1)
        ]
        if len(point_diffs) < 2:
            return "silent"
        point_std = statistics.stdev(point_diffs) if len(point_diffs) > 1 else 0.0

        # Degenerate case: opponent barely changes (std ≈ 0 means rigid/hardball)
        if sim_std < 1.0 and point_std < 0.01:
            return "silent"

        last_sim = similarities[-1]
        last_pdiff = point_diffs[-1]

        # Opponent changed bid MORE than normal (similarity dropped)
        if sim_std > 0 and last_sim < sim_mean - sim_std:
            if last_pdiff > 0 and abs(last_pdiff) > point_std:
                return "competitive"
            elif last_pdiff < 0 and abs(last_pdiff) > point_std:
                return "cooperative"
            else:
                return "cooperative"

        # Opponent changed bid LESS than normal (similarity rose)
        if sim_std > 0 and last_sim > sim_mean + sim_std:
            if 0 < point_std < abs(last_pdiff):
                return "silent"
            else:
                return "cooperative"

        # Average change
        if 0 < point_std < abs(last_pdiff):
            return "silent"
        return "cooperative"

    def _update_behaviour(self):
        self._opp_behaviour = self._detect_behaviour()
        self._adapt_parameters()

    def _select_bid_with_opponent_model(self, target):
        if self._opp_behaviour == "competitive":
            window = 0.03
        elif self._opp_behaviour == "cooperative":
            window = 0.06
        else:  # silent
            window = 0.05

        return self._get_bid_at(target, window=window)

    def _should_accept(self, offer_utility, next_bid_utility, target, t):
        if offer_utility is None:
            return False

        rv = self._nufun.reserved_value if self._nufun else 0.0
        if offer_utility < rv:
            return False

        # ACtarget
        if offer_utility >= target:
            return True

        # ACnext
        if next_bid_utility is not None and offer_utility >= next_bid_utility:
            return True

        # ACcombi endgame
        if t >= self._endgame_t and self._opp_bid_utilities:
            best_seen = max(self._opp_bid_utilities)
            if offer_utility >= best_seen * self._endgame_factor and offer_utility > rv:
                return True

        # Deep endgame safety: on the final turn, any offer above the
        # reservation value beats timing out
        if t >= 0.99 and offer_utility > rv:
            return True

        return False

    def _template_message(self, kind, bid_utility=None):
        if kind == "accept":
            return "That works for me - I accept."

        if kind == "open":
            return "Here is my opening proposal."

        # counter-offer: tone follows detected behavior + how far we've conceded
        conceded = bid_utility is not None and bid_utility < self._p0 - 0.15
        if self._opp_behaviour == "competitive":
            return "I can move a little, but I need a fair deal."
        if self._opp_behaviour == "cooperative":
            return "Appreciate the flexibility - here is a step toward you."
        if conceded:
            return "I have come down some; let's try to close this."
        return "Here is my counter-offer."

    def _format_bid(self, bid):
        if bid is None:
            return "nothing yet"
        try:
            issues = self.nmi.outcome_space.issues
            parts = [f"{iss.name} {val}" for iss, val in zip(issues, bid)]
            return ", ".join(parts)
        except Exception:
            return str(bid)

    def _load_scenario_info(self, scenario):
        if not scenario:
            return {}
        import os
        candidates = [
            os.path.join(os.path.dirname(__file__), "scenarios", scenario, "_info.yaml"),
            os.path.join("scenarios", scenario, "_info.yaml"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                try:
                    import yaml
                    with open(path, encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}
                except Exception:
                    return {}
        return {}

    def _build_domain_context(self):
        # Scenario + issue-space description for the LL
        try:
            os_ = self.nmi.outcome_space
            scenario = getattr(os_, "name", None)
            info = self._load_scenario_info(scenario)
            issue_desc = info.get("issue_description", {}) if isinstance(info, dict) else {}
            title = info.get("title") if isinstance(info, dict) else None

            parts = []
            for iss in os_.issues:
                try:
                    vals = list(iss.all)
                except Exception:
                    vals = list(getattr(iss, "values", []) or [])
                if not vals:
                    desc = "?"
                elif len(vals) <= 8:
                    desc = ", ".join(str(v) for v in vals)
                else:
                    desc = f"{vals[0]}..{vals[-1]}"
                d = issue_desc.get(iss.name)
                parts.append(f"{iss.name} ({desc}) - {d}" if d else f"{iss.name} ({desc})")
            issues_str = "; ".join(parts)

            if scenario and title:
                head = f"Scenario: {scenario} ({title})."
            elif scenario:
                head = f"Scenario: {scenario}."
            else:
                head = ""
            ctx = f"{head} Negotiating over: {issues_str}.".strip()
            return ctx.encode("ascii", "ignore").decode("ascii")
        except Exception:
            return ""

    def _llm_message(self, kind, template, bid=None, opp_offer=None):
        #Given truthful context (our offer on the table + their last offer)
        import re

        our_text = self._format_bid(bid)
        opp_text = self._format_bid(opp_offer)

        intent = {
            "open": "This is your opening offer, so sound warm and optimistic.",
            "offer": "You are countering their offer; nudge them toward your terms.",
            "accept": "You are happily accepting their offer; sound pleased and warm.",
        }.get(kind, "You are countering their offer.")

        mood = {
            "competitive": "They are playing hardball, so stay friendly but firm.",
            "cooperative": "They are being flexible, so sound appreciative and warm.",
            "silent": "They have barely moved, so gently encourage them.",
        }.get(self._opp_behaviour, "")

        prompt = (
            f"{self._domain_context} "
            "You are a real person negotiating this deal in a friendly chat. "
            "Write ONE short, natural sentence (how a human would actually talk "
            "- relaxed, warm, conversational). No preamble, no quotes, no emojis, "
            "no lists, under 25 words. You may naturally reference the deal terms. "
            f"{intent} {mood} "
            f"Your offer on the table: {our_text}. Their last offer: {opp_text}."
        )

        try:
            text = self._call_llm([{"role": "user", "content": prompt}]) or ""
        except Exception:
            self._llm_fail_count += 1
            if self._llm_fail_count >= self._llm_fail_limit:
                self._llm_enabled = False
            return None

        # Strip <think> blocks and take the first non-empty line.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = text.split("\n")[0] if text else ""

        # Normalize common Unicode punctuation to ASCII
        for uni, asc in (
            ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
            ("—", " - "), ("–", "-"), ("…", "..."),
        ):
            text = text.replace(uni, asc)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.strip().strip('"').strip()

        # The model responded, so reset the hard-failure counter.
        self._llm_fail_count = 0

        # Reject empty/over-long output.
        # Fall back to the template for this one message only.
        if not text or len(text) > 220:
            return None

        print(self._domain_context);
        print(text)
        return text

    def _message(self, kind, bid=None, bid_utility=None, opp_offer=None):
        #Return a message for this turn: LLM-generated if available and
        #responsive, otherwise the template
        template = self._template_message(kind, bid_utility)
        if getattr(self, "_llm_enabled", False):
            llm_text = self._llm_message(kind, template, bid=bid, opp_offer=opp_offer)
            if llm_text:
                return llm_text
        return template

    def __call__(self, state: SAOState, dest=None) -> SAOResponse:
        offer = state.current_offer
        t = state.relative_time

        if offer is not None and self._nufun:
            u = float(self._nufun(offer))
            self._opp_bids.append(offer)
            self._opp_bid_utilities.append(u)
            self._update_opponent_model(state, offer)

        # Detect behavior and adapt p1/p2/p3 BEFORE computing target
        if len(self._opp_bids) >= _BEHAVIOUR_THRESHOLD:
            self._update_behaviour()

        target = self._calculate_target_utility(t)

        # Decide the bid we would send next
        if len(self._opp_bids) >= _BEHAVIOUR_THRESHOLD:
            bid = self._select_bid_with_opponent_model(target)
        else:
            bid = self._get_bid_at(target)
        bid_utility = float(self._nufun(bid)) if bid is not None and self._nufun else None

        offer_utility = self._opp_bid_utilities[-1] if self._opp_bid_utilities else None
        if offer is not None and self._should_accept(offer_utility, bid_utility, target, t):
            return SAOResponse(
                ResponseType.ACCEPT_OFFER, offer,
                data=dict(text=self._message("accept", bid=offer, opp_offer=offer)),
            )

        if bid is not None and self._nufun:
            self._my_bids.append(bid)
            self._my_bid_utilities.append(bid_utility)

        kind = "open" if len(self._my_bids) <= 1 else "offer"
        return SAOResponse(
            ResponseType.REJECT_OFFER, bid,
            data=dict(text=self._message(kind, bid=bid, bid_utility=bid_utility, opp_offer=offer)),
        )
