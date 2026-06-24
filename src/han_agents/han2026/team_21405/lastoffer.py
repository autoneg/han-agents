import math
from random import sample
from negmas import PresortingInverseUtilityFunction, ResponseType
from negmas.gb import ResponseType
from negmas.preferences import PresortingInverseUtilityFunction
from negmas_llm import OllamaNegotiator
from negmas.sao import SAOResponse
from collections import Counter, deque, defaultdict
import random

class LastOfferNegotiator(OllamaNegotiator):
    """
    Our negotiation agent is a hybrid system combining a utility-based mathematical decision model,
    a time-dependent aspiration function, and an adaptive opponent model. The strategy balances self-interest
    maximization with concession dynamics, opponent behavior tracking, and endgame robustness.
    """

    def __init__(
            self,
            strategy_kwargs=None,
            temperature: float = 0.7,
            max_tokens: int = 1024,
            use_structured_output: bool = True,
            timeout: float = 120.0,
            num_retries: int = 3,
            **kwargs,
    ):
        """
        Initialize the negotiator and configure the LLM backend.
        """

        llm_kwargs = kwargs.pop("llm_kwargs", {})
        llm_kwargs.setdefault("timeout", timeout)
        llm_kwargs.setdefault("num_retries", num_retries)

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            use_structured_output=use_structured_output,
            llm_kwargs=llm_kwargs,
            system_prompt=None,
            preferences_prompt=None,
            preferences_changed_prompt=None,
            negotiation_start_prompt=None,
            round_prompt=None,
            **kwargs,
        )

        strategy_kwargs = strategy_kwargs or {}
        self.engine = None
        self.opponent_model = OpponentModel()
        self.argument_counts = Counter()

    def __call__(self, state, dest=None):
        """
        Main entry point for each negotiation step.
        """

        assert self.ufun is not None

        self._ensure_engine()
        self.engine.ufun = self.ufun

        # extract opponent message text if available
        opponent_text = getattr(state, "current_data", None)
        if isinstance(opponent_text, dict):
            opponent_text = opponent_text.get("text", "")

        # estimate opponent sentiment using LLM classification
        # self.opponent_model.current_sentiment = self.analyze_opponent_message(opponent_text)

        offer = state.current_offer
        offer_utility = self.engine.get_utility(offer) if offer is not None else float("-inf")

        my_weights = None
        if hasattr(self.ufun, 'weights'):
            my_weights = list(self.ufun.weights)

        # update opponent model and track best observed utility
        best_utility_opponent, _ = self.opponent_model.update(offer, offer_utility, my_weights)

        # update internal decision engine state
        self.engine.update(state)

        # compute accept/reject decision
        response, outcome = self.engine.decide(
            state,
            self.ufun,
            best_utility_opponent,
            opponent_sentiment=self.opponent_model.current_sentiment,
            opponent_weights=self.opponent_model.estimated_weights, # Safe weights are passed here
        )

        # if offer is rejected, generate a new proposal
        if response == ResponseType.REJECT_OFFER:
            outcome = self.engine.propose(
                state,
                best_utility_opponent,
                opponent_weights=self.opponent_model.estimated_weights, # Safe weights are passed here
                opponent_sentiment=self.opponent_model.current_sentiment
            )

        # generate natural language message (LLM only for text, not logic)
        text = self.generate_negotiation_text(
            response,
            outcome,
            state,
            opponent_text
        )

        # return SAO-compatible response
        return SAOResponse(
            response,
            outcome=outcome,
            data={"text": text},
        )

    def _ensure_engine(self):
        """Lazy initialization of decision engine."""
        if self.engine is None:
            self.engine = NegotiationDecision(self.ufun)

    def generate_negotiation_text(
            self,
            response_type,
            outcome,
            state,
            opponent_text,
    ):
        """
        Generate final negotiation message text.
        """

        # initial greeting at start of negotiation
        if state.step == 0 or getattr(state, "last_negotiator", None) is None:
            return (
                "I am the Last Offer Negotiator. "
                "I look forward to a fair and constructive negotiation. "
                "I will do my best to find an agreement that works for both sides."
            )

        # acceptance message
        if response_type == ResponseType.ACCEPT_OFFER:
            return "I accept this proposal and look forward to concluding our agreement."

        # termination message
        if response_type == ResponseType.END_NEGOTIATION:
            return "Unfortunately, I do not believe we can reach an agreement at this stage."

        # LLM-generated rejection message
        # return "No LLM Answer"
        return self.generate_reject_text(outcome, state, opponent_text)

    def generate_reject_text(self, outcome, state, opponent_text):
        """
        Generate rejection message using LLM with structured prompt.
        """

        argument = self.generate_argument(outcome=outcome, state=state)

        # This prompt generates the final natural language message when the agent rejects an offer.
        # The goal is to produce a short, human-like negotiation message
        prompt = f"""
                You are writing a short reply in an ongoing negotiation.

                [Goal]
                Create a concise, natural supporting message to my new offer.

                [CONTEXT]
                My new offer: {outcome}

                [AVAILABLE ARGUMENTS]
                {argument}

                [RULES]
                - Keep the conversation natural and grounded in the ongoing exchange.
                - Keep the tone cooperative but firm.
                - Use the available argument; do not list them explicitly, integrate naturally
                - Do NOT explicitly mention numbers, structure, or that this is a "proposal" or "offer".
                - Do not mention "arguments" or their types.
                - No mention of utilities, strategy, or internal reasoning.
                
                [STYLE]
                - Maximum 2 sentences.
                - Simple, conversational language.
                - No dashes (no —, –, or hyphenated breaks).
                - Avoid stock phrases like:
                  "I appreciate", "I'm confident", "let's move forward", "agreed before", "what’s been discussed"
                
                [OUTPUT]
                Return ONLY the final message.
                """

        response = self._call_llm(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            require_json=False,
        )

        #return f"{response}/n{argument}"
        return response

    def generate_argument(
            self,
            outcome,
            state,
    ):
        """
        Select an argument type based on scored negotiation signals
        and return a corresponding pre-generated argument sentence.
        """

        # Argument pool: predefined natural language templates grouped by strategic intent
        argument_pool = {
             "fairness": [ # fairness: balanced compromise between both parties
                "This proposal maintains a balanced distribution across the issues.",
                "The offer reflects a fair compromise for both sides.",
                "The current proposal keeps the overall arrangement equitable.",
                "This division of terms ensures neither side is disproportionately advantaged.",
                "The structure of this offer preserves a fair balance of outcomes.",
                "This arrangement treats both sides' priorities with equal consideration.",
                "The proposal remains within a mutually fair range across all issues."
            ],

            "reciprocity": [ # reciprocity: tit-for-tat adjustment
                "I have adjusted parts of the proposal and hope for movement in return.",
                "This offer reflects mutual concessions from both sides.",
                "I have already made changes to support progress.",
                "I am responding to your previous adjustments with corresponding flexibility.",
                "This proposal builds on your last move with reciprocal adaptation.",
                "I have shifted my position in response to your willingness to engage.",
                "Further progress depends on continued mutual adjustments."
            ],

            "progress": [ # progress: emphasis on "forward" movement in negotiation
                "This proposal is intended to keep the discussion moving forward.",
                "The offer is designed to help us make further progress.",
                "I hope this proposal brings us closer to an agreement.",
                "We are building step by step toward a workable solution.",
                "This move helps reduce remaining gaps in the negotiation.",
                "This adjustment supports continued forward momentum.",
                "The proposal is structured to prevent stagnation in the discussion."
            ],

            "risk_reduction": [  # risk_reduction: avoiding negotiation failure
                "The offer reduces the likelihood that we leave without an agreement.",
                "Reaching an agreement now prevents both sides from losing accumulated value.",
                "Without agreement, both sides risk walking away from already improved positions.",
                "It is better to secure a workable agreement than to risk complete breakdown at this stage.",
                "This proposal helps avoid an impasse that would hurt both sides.",
                "An agreement now is safer than prolonged uncertainty.",
                "This structure minimizes the risk of negotiation failure."
            ],

            "benefit_maximization": [ # benefit_maximization: self-optimizing change of offer, direction change
                "This proposal shifts the balance toward a more favorable outcome for my side.",
                "This configuration better aligns the terms with my priorities across the issues.",
                "This adjustment improves the overall value of the deal from my perspective.",
                "This version of the agreement increases my expected benefit while remaining workable.",
                "This is a more advantageous distribution of terms compared to earlier offers.",
                "This structure improves my payoff without breaking the negotiation balance.",
                "This version optimizes outcomes more strongly in my favor while staying within a reasonable range."
            ],

            "deadline": [ # deadline: urgency due to remaining time pressure
                "As time is running short, reaching an agreement becomes increasingly important.",
                "We are approaching the end of the negotiation and should try to finalize an agreement.",
                "Given the limited time remaining, finding a workable agreement is beneficial.",
                "The remaining time increases pressure to converge on a solution.",
                "We should avoid delaying agreement given the approaching deadline.",
                "Time constraints make further delay costly for both sides.",
                "It is important to converge before the deadline limits our options further."
            ],

            "concession": [ # concession: explicit signaling of own concessions
                "I have already moved closer to your position on several points.",
                "This proposal reflects meaningful adjustments on my side.",
                "I have made concessions to support an agreement.",
                "I have shifted my position to better accommodate your preferences.",
                "Several elements have been adjusted in your direction.",
                "I have relaxed parts of my position to reduce the gap between us.",
                "This offer includes clear concessions compared to my earlier stance."
            ],

            "similar_offer": [ # similar_offer: structurally similar to opponent's last offer
                "This offer is structurally very similar to your previous proposal.",
                "The current proposal closely follows the direction of your last offer.",
                "This builds on your previous offer with only minor adjustments.",
                "The structure remains largely aligned with your prior suggestion.",
                "This proposal keeps most elements unchanged from your earlier version.",
                "Only small modifications separate this offer from your previous one."
            ],

            "equivalent_offer": [ # equivalent_offer: equal-value to the opponents offer
                "This proposal represents a balanced and equivalent exchange of value.",
                "Both sides receive comparable outcomes under this arrangement.",
                "This offer maintains parity across the main issues.",
                "The overall value distribution is approximately equal for both parties.",
                "Neither side gains a structural advantage under this proposal.",
                "This arrangement preserves equality in expected outcomes."
            ],

            "beginning": [ # beginning: early-stage, still close to our best value
                "We are still in the early stages of the negotiation and should establish direction.",
                "At this point, it is important to define a workable starting structure.",
                "This proposal helps us establish a solid baseline for further negotiation.",
                "We are still exploring the space of possible agreements.",
                "This stage is about identifying a viable negotiation framework.",
                "We are laying the foundation for more refined offers.",
                "This helps us map out the structure of the negotiation space."
            ],

            "no_progress": [ # no_progress: stagnation / lack of convergence signal
                "We have not made meaningful progress in the last exchange.",
                "The negotiation appears to be stagnating, so adjustments are needed.",
                "We risk going in circles unless we change direction.",
                "We are repeating positions without narrowing the gap.",
                "There is no significant movement toward agreement so far.",
                "The current trajectory is not reducing our differences.",
                "We are stuck without measurable convergence between offers."
            ],

            # legacy, low-use
            "similarity": [ # similarity: structurally similar to opponent's last offer
                "This proposal remains close to the structure of your previous offer.",
                "I tried to keep this offer aligned with the priorities you expressed.",
                "This proposal builds directly on the direction of your last offer."
            ],
            "collaboration": [ # collaboration: cooperative framing
                "This proposal is intended to keep the discussion constructive.",
                "The offer is designed to help us move closer to an agreement.",
                "I hope this proposal supports further cooperation."
            ],
        }

        # Calculate values
        scores = Counter()

        utility_range = self.engine.utility_range
        my_u = self.engine.get_utility(outcome)

        # opponent utility of current offer (fallback: symmetric assumption)
        if state.current_offer is not None:
            opp_u = self.engine.get_utility(state.current_offer)
        else:
            opp_u = my_u

        # relative advantage vs opponent offer
        gap = my_u - opp_u
        gap_ratio = gap / utility_range if utility_range > 0 else 0.0

        target = self.engine.calculate_aspiration(
            state.relative_time,
            opponent_sentiment=self.opponent_model.current_sentiment,
            opponent_weights=self.opponent_model.estimated_weights,
        )

        # distance to aspiration threshold
        target_gap = my_u - target

        similarity = self.engine.calculate_similarity(
            outcome,
            state.current_offer,
            self.opponent_model.estimated_weights,
        )

        # where is my current offer
        distance_from_best = (self.engine.get_utility(self.engine._best) - my_u) / utility_range

        # how much we already conceded
        concession_ratio = (self.engine.target_alpha - self.engine.lowest_offered_utility) / utility_range


        # scoring
        # base preference fairness, always active
        scores["fairness"] += 1

        # indicates a balanced situation
        if similarity > 0.8 and abs(gap_ratio) < 0.08:
            scores["fairness"] += 2

        # the offer is structurally aligned with the opponent
        if similarity > 0.8 and gap_ratio > 0.15:
            scores["similar_offer"] += 4

        # similar value despite structural differences
        if similarity < 0.3 and abs(gap_ratio) < 0.1:
            scores["equivalent_offer"] += 5

        # direction change, which supports a benefit-maximization
        if similarity < 0.3 and gap_ratio > 0.2:
            scores["benefit_maximization"] += 5

        # close to the best possible utility indicates early-stage or exploratory
        if distance_from_best < 0.1:
            scores["beginning"] += 1

        # concession detection
        if concession_ratio > 0.2 and state.relative_time > 0.25:
            scores["concession"] += 3
        elif concession_ratio > 0.4:
            scores["concession"] += 4

        # lack of progress vs aspiration
        if target_gap > 0.1 * utility_range:
            scores["progress"] += 3

        # reciprocity detection based on last self-move
        if self.engine.last_offered_by_me is not None:
            previous_my_u = self.engine.get_utility(self.engine.last_offered_by_me)
            my_change = previous_my_u - my_u

            if my_change > 0.03 * utility_range:
                scores["reciprocity"] += 3

        # stagnation detection
        if self.engine.stagnant_rounds >= 4:
            scores["no_progress"] += 4

        # time pressure signal
        if state.relative_time > 0.85:
            scores["deadline"] += 4

        # combined risk signal (time + stagnation)
        if state.relative_time > 0.85 and self.engine.stagnant_rounds >= 4:
            scores["risk_reduction"] += 6


        # penalize overused argument types
        for arg_type in list(scores.keys()):
            usage_penalty = self.argument_counts[arg_type] * 0.05
            scores[arg_type] = max(
                0.1,
                scores[arg_type] - usage_penalty
            )

        max_score = max(scores.values())

        # select all near-optimal argument types
        top_candidates = [
            arg_type
            for arg_type, score in scores.items()
            if score >= max_score - 0.0001
        ]

        argument_type = random.choice(top_candidates)

        # update usage statistics
        self.argument_counts[argument_type] += 1

        # decay of usage memory
        for arg_type in self.argument_counts:
            self.argument_counts[arg_type] *= 0.9

        # return {"type": argument_type, "arguments": random.choice(argument_pool[argument_type])}
        return {"arguments": random.choice(argument_pool[argument_type])}

    def analyze_opponent_message(self, opponent_text: str):
        """
        Uses the LLM to detect the opponent's sentiment and willingness to walk away.
        """
        if not opponent_text or opponent_text.strip() == "":
            return "NEUTRAL"

        # This prompt classifies the emotional state of the opponent.
        # The classification is used to adapt the negotiation strategy dynamically
        prompt = f"""
        Classify sentiment.

        Text:
        {opponent_text}

        Rules:
        - Only JSON
        - No extra text

        Return JSON:
        {{"sentiment": "HAPPY|NEUTRAL|FRUSTRATED|ANGRY"}}
        """

        response = self._call_llm(
            messages=[{"role": "user", "content": prompt}],
            require_json=False
        )

        return response

class NegotiationDecision:

    def __init__(self, ufun):
        self.alpha = 0.97 # start utility: percentage of the maximum utility we could reach
        self.gamma = 4 # decay of our aspiration function
        self.ufun = ufun

        self._inv = None # the inverse utility function to get offers corresponding to a given utility
        self._min = None # the min of the utility function
        self._max = None # the max of the utility function
        self._best = None # the best offer for ourselves

        self.safety_floor = None # the utility we do not want to fall under
        self.target_alpha = None # the utility with which we want to start

        self.proposed_offers = set() # already proposed offers
        self.offer_counts = Counter() # counting the number of times we have proposed an offer
        self.opponent_history = [] # offers made by the opponent in a sequential order

        self.stagnant_rounds = 0 # how often the opponent keeps repeating its offer
        self.last_offered_by_me = None # the last offer made by ourselves

        self.lowest_offered_utility = float("inf") # the lowest utility we have offered
        self.ufun_cache = {} # cache for the utility to an offer we have already computed

    def _ensure_initialized(self):
        """
        We initialise all the relevant parameters once (inverse utility function, min, max, best offer, safety floor, target alpha).
        """

        if self._inv is not None:
            return

        self._inv = PresortingInverseUtilityFunction(self.ufun)
        self._inv.init()

        worst, self._best = self.ufun.extreme_outcomes()
        self._min = float(self.ufun(worst))
        self._max = float(self.ufun(self._best))
        self.utility_range = self._max - self._min

        reserved = float(self.ufun.reserved_value)
        self.safety_floor = max(reserved, self._min + self.utility_range * 0.4)

        self.target_alpha = self._min + self.utility_range * self.alpha

    def get_utility(self, offer):
        """
        Only calculates utility the first time we see an offer.
        """

        if offer not in self.ufun_cache:
            self.ufun_cache[offer] = float(self.ufun(offer))

        return self.ufun_cache[offer]

    def update(self, state):
        """
        Updates our opponent history and the stagnant rounds every time we get a new offer.
        """

        offer = state.current_offer

        if offer is None:
            return None

        if self.opponent_history:
            # find the best utility they have offered us so far
            best_historical_u = max(self.get_utility(o) for o in self.opponent_history)

            # if their new offer isn't pushing the boundary of what they've already offered, they are stalling
            if self.get_utility(offer) <= best_historical_u + 0.01:
                self.stagnant_rounds += 1
            else:
                self.stagnant_rounds = 0
        else:
            self.stagnant_rounds = 0

        self.opponent_history.append(offer)
        return offer

    def decide(self, state, ufun, best_utility_opponent, opponent_sentiment="NEUTRAL", opponent_weights=None):
        """
        Decides whether we should accept or reject an offer.
        """

        self._ensure_initialized()
        self.ufun = ufun

        offer = state.current_offer

        # if no offer was made: reject
        if offer is None:
            return ResponseType.REJECT_OFFER, None

        # if opponent offers something we have already proposed before: accept
        if offer in self.proposed_offers:
            return ResponseType.ACCEPT_OFFER, offer

        # calculate the utility we get from the offer and the target aspiration function at the given time
        my_u = float(self.get_utility(offer))
        target = self.calculate_aspiration(state.relative_time, best_utility_opponent=best_utility_opponent, opponent_sentiment=opponent_sentiment, opponent_weights=opponent_weights)

        # if we still have enough time we will reject an offer if the historically best offer made by the opponent compared to the recent offer is significantly better.
        if best_utility_opponent is not None:

            if state.relative_time < 0.8 and (best_utility_opponent - my_u) > 0.08 * self.utility_range:
                return ResponseType.REJECT_OFFER, None

        # Endgame safety implementation:
        # punish the opponent by rejecting his offer if:
        # he has barely made any offers until now
        # his current offer is significantly worse than the historically best offer.
        if state.relative_time > 0.95:
            if len(self.opponent_history) < 4:
                return ResponseType.REJECT_OFFER, None

            if best_utility_opponent is not None and (best_utility_opponent - my_u) > 0.15 * self.utility_range:
                return ResponseType.REJECT_OFFER, None

        # Main acceptance rule
        # accept if the utility of the current offer is not less than the aspiration function at the given time
        if my_u >= target:
            return ResponseType.ACCEPT_OFFER, offer

        # Endgame rule
        # if the opponent keeps repeating its offer at the end we accept it if
        # it is at least above our safety floor or our reservation value + 30 % of our utility range
        if state.relative_time > 0.95 and self.stagnant_rounds > 2:
            if my_u > min((self.ufun.reserved_value + 0.3*self.utility_range), self.safety_floor):
                return ResponseType.ACCEPT_OFFER, offer

        # otherwise: reject
        return ResponseType.REJECT_OFFER, None

    def calculate_aspiration(self, t: float, best_utility_opponent: float = None, opponent_sentiment: str = "NEUTRAL", opponent_weights = None) -> float:
        """
        Calculates our aspiration function.
        """

        # calculate opponent's concession ratio based on the best utility he has offered
        opp_concession_ratio = 0.0
        if best_utility_opponent is not None and best_utility_opponent != float("-inf") and self.utility_range > 0:
            opp_concession_ratio = (best_utility_opponent - self._min) / self.utility_range

        # if we are past 50% of the time limit and they haven't conceded at least 15%, freeze our concession.
        effective_t = t
        if t > 0.5 and opp_concession_ratio < 0.15:
            effective_t = 0.5

        # our base aspiration function
        base_target = self.target_alpha - (self.target_alpha - self.safety_floor) * (effective_t ** self.gamma)

        # if opponent keeps repeating the same offer we want to concede in order to avoiding not reaching an agreement.
        if self.stagnant_rounds > 2:
            penalty = min(0.02 * (self.stagnant_rounds - 2), 0.05)
            base_target -= penalty

        # we adapt our aspiration function to the sentiment of our opponent
        # if happy -> we raise our expectations
        # if frustrated -> we lower to reach an agreement.
        sentiment_penalties = {"HAPPY": 0.05, "NEUTRAL": 0.0, "FRUSTRATED": -0.05, "ANGRY": -0.1}
        penalty = sentiment_penalties.get(opponent_sentiment, 0.0)
        base_target += penalty

        # ensure to never drop below our safety floor
        return max(self.safety_floor, base_target)

    def calculate_similarity(self, offer1, offer2, weights=None):
        """
        Calculates the weighted similarity between two offers.
        """

        if offer1 is None or offer2 is None:
            return 0.0

        similarity_score = 0.0
        for i, (val1, val2) in enumerate(zip(offer1, offer2)):
            weight = weights[i] if weights else (1.0 / len(offer1))

            if val1 == val2:
                # exact match
                similarity_score += weight
            else:
                # otherwise calculate similarity if numerical value
                try:
                    v1 = float(val1)
                    v2 = float(val2)
                    max_val = max(abs(v1), abs(v2))
                    if max_val > 0:
                        sim = max(0.0, 1.0 - (abs(v1 - v2) / max_val))
                        similarity_score += (sim * weight)
                except (ValueError, TypeError):
                    pass

        return similarity_score

    def propose(self, state, best_utility_opponent, opponent_weights, opponent_sentiment="NEUTRAL"):
        """
        Calculates the next offer we want to propose.
        """

        self._ensure_initialized()

        # calculate our aspiration function and find offer which are
        # above the aspiration function and below an upper bound
        a = self.calculate_aspiration(state.relative_time, best_utility_opponent=best_utility_opponent, opponent_sentiment=opponent_sentiment, opponent_weights=opponent_weights)
        historical_cap = self.lowest_offered_utility + (0.03 * self.utility_range)
        raw_upper = a + 0.15 * self.utility_range

        # the upper bound is based on the aspiration function and our historical lowest offer
        upper_bound = min(self._max, raw_upper, historical_cap)
        valid_offers = list(self._inv.some((a - 1e-5, upper_bound + 1e-5), False))

        # randomly sample up to 100 offers to reduce computational time
        if len(valid_offers) > 100:
            offers = sample(valid_offers, 100)
        else:
            offers = valid_offers

        # only propose novel offers if there are any
        outcomes = [o for o in offers if o not in self.proposed_offers and o != self._best]

        # if there are no novel offers: propose offers already made before
        if not outcomes:
            raw_upper = a + 0.075 * self.utility_range

            # upper bound
            upper_bound = min(self._max, raw_upper, historical_cap)
            valid_outcomes = list(self._inv.some((a - 1e-5, upper_bound + 1e-5), False))

            if len(valid_outcomes) > 50:
                outcomes = sample(valid_outcomes, 50)
            else:
                outcomes = valid_outcomes

        # if there are no offers: propose the last offer made by us
        # and if there is no last offer made by us: propose our absolute best offer
        if not outcomes:
            if self.last_offered_by_me is not None:
                return self.last_offered_by_me
            else:
                return self._best

        opponent_offer = state.current_offer
        best_joint_offer = None
        max_joint_benefit = -math.inf

        # calculate which offer to propose
        # offer should be similar to the first and most recent offer made by the opponent
        for o in outcomes:
            match_score = self.calculate_similarity(o, opponent_offer, opponent_weights)
            match_first_score = self.calculate_similarity(o, self.opponent_history[0], opponent_weights) if self.opponent_history else 0.0

            # calculate as a base our own utility for this offer
            my_u = float(self.get_utility(o)) - a

            # punish if we are repeating an offer
            # punish even more if it was the last offer we proposed
            times_proposed = self.offer_counts[o]
            frequency_penalty = times_proposed * -0.05
            immediate_penalty = -0.2 if o == self.last_offered_by_me else 0.0

            # our own utility should be weighted higher at the beginning
            time = state.relative_time
            time_left = max(0.0, 1.0 - time)
            my_u_weight = 1.0 + (4.0 * time_left)
            # match to the opponent's offer should be weighted higher at the end
            match_score_weight = 1.2 + time*3

            # calculate the score our offer gets in summary.
            score = my_u_weight*my_u + match_score_weight*match_score + 0.8*match_score_weight*match_first_score + frequency_penalty + immediate_penalty

            # keep track of the best offer to propose
            if score > max_joint_benefit:
                max_joint_benefit = score
                best_joint_offer = o


        # Endgame Logic
        # if time starts running out and the opponent has made offers before:
        # start reproposing the best offers for ourselves made by the opponent
        # if they are at least as good as our safety floor and not significantly worse than the next offer we would make
        if state.relative_time > 0.8 and self.opponent_history:
            my_u = self.ufun(best_joint_offer) if best_joint_offer else self._min

            # sort the opponent offers by their utility for ourselves
            unique_opponent_history = list(set(self.opponent_history))
            sorted_opponent_history = sorted(unique_opponent_history, key = lambda o: float(self.get_utility(o)), reverse=True)

            # go through all the offers starting with the best which are meeting our criterions
            for o in sorted_opponent_history:
                o_u = self.get_utility(o)

                if o_u < self.safety_floor or o_u < my_u -0.05:
                    break

                # repropose always a new offer!
                if o not in self.proposed_offers:
                    self.last_offered_by_me = o
                    self.proposed_offers.add(self.last_offered_by_me) # Keep track of what we offered
                    self.offer_counts[self.last_offered_by_me] += 1

                    current_u = float(self.get_utility(self.last_offered_by_me))
                    self.lowest_offered_utility = min(self.lowest_offered_utility, current_u)
                    return o

        self.last_offered_by_me = best_joint_offer if best_joint_offer is not None else self._best
        self.proposed_offers.add(self.last_offered_by_me) # Keep track of what we offered
        self.offer_counts[self.last_offered_by_me] += 1

        current_u = float(self.get_utility(self.last_offered_by_me))
        self.lowest_offered_utility = min(self.lowest_offered_utility, current_u)
        return self.last_offered_by_me

class OpponentModel:
    """
    The Opponent Model is based on:
    Conflict-based Opponent Model (CBOM) presented in:
    Keskin, M. O., Buzcu, B., & Aydoğan, R. (2023).
    Conflict-based negotiation strategy for human-agent negotiation.
    Applied Intelligence, 53, 29741-29757.
    https://doi.org/10.1007/s10489-023-05001-9
    """

    def __init__(self):
        self.best_offer = None # best offer made by the opponent
        self.best_utility = float("-inf") # best utility from the opponent
        self.offer_history = [] # opponent's offer history

        self.estimated_weights = None # estimated weights for the preferences of the issues
        self.current_sentiment = "NEUTRAL" # current sentiment of the opponent

        # conflicted issue pairs
        self.IC_norms = defaultdict(int)

        # current Beliefs
        self.B_issues = set()

        # our initial weights
        self.initial_inverse_weights = None

    def update(self, offer, utility, my_weights=None):
        """
        Updates our believes when we receive a new offer.
        """

        if offer is None:
            return None, float("-inf")

        # keep track of the best offer made by the opponent and its utility
        if utility > self.best_utility:
            self.best_utility = utility
            self.best_offer = offer

        # keep track of the number of issues
        num_issues = len(offer)

        # initialise the weights as the inverse of our own weights if possible
        # else: uniform weights
        if not self.offer_history:
            if my_weights and len(my_weights) == num_issues:
                inv = [max(0.01, 1.0 - w) for w in my_weights]
                total_inv = sum(inv)
                self.initial_inverse_weights = [w / total_inv for w in inv]
            else:
                self.initial_inverse_weights = [1.0 / num_issues] * num_issues

            self.estimated_weights = list(self.initial_inverse_weights)
            self.B_values = {i: set() for i in range(num_issues)}
            self.offer_history.append(offer)
            return utility, offer

        # unique values per each offer pair
        CM = []
        for past_offer in self.offer_history:
            # unique values per each offer pair
            UV = []
            for i in range(num_issues):
                if past_offer[i] != offer[i]:
                    UV.append((i, past_offer[i], offer[i]))
            if UV:
                CM.append(UV)

        # extract all conflicts
        for c in CM:
            changed_issues = [item[0] for item in c]
            unchanged_issues = [i for i in range(num_issues) if i not in changed_issues]

            for unchanged in unchanged_issues:
                for changed in changed_issues:
                    self.IC_norms[(unchanged, changed)] += 1

        #resolve issue beliefs
        self.B_issues = set()
        for (A, B), count_AB in list(self.IC_norms.items()):
            count_BA = self.IC_norms.get((B, A), 0)

            if count_AB > count_BA:
                self.B_issues.add((A, B))
            elif count_BA > count_AB:
                self.B_issues.add((B, A))

        # calculate the number of times one issue wins over the other
        wins = [0] * num_issues
        for (A, B) in self.B_issues:
            wins[A] += 1

        # update initial scores with the wins
        fused_scores = [wins[i] + (self.initial_inverse_weights[i] * num_issues) for i in range(num_issues)]

        # normalize the scores
        total_score = sum(fused_scores)
        if total_score > 0:
            self.estimated_weights = [score / total_score for score in fused_scores]
        else:
            self.estimated_weights = list(self.initial_inverse_weights)

        self.offer_history.append(offer)

        return utility, offer