from negmas.common import Outcome
from negmas.gb import BoulwareTBNegotiator
from negmas.sao import (
    SAOState,
    ConcederTBNegotiator,
    LinearTBNegotiator
)
from negmas_llm.meta import LLMMetaNegotiator

try:
    from negmas_llm.common import DEFAULT_MODELS
except ImportError:
    DEFAULT_MODELS = dict(ollama="qwen3:4b-instruct")

DEFAULT_OLLAMA_MODEL = DEFAULT_MODELS["ollama"]

# =============================================================================
# Default Prompt (copied from negmas_llm, customize as needed)
# =============================================================================

# System prompt for text generation in LLMMetaNegotiator
SYSTEM_PROMPT = """
You are assisting a negotiator by generating persuasive text to
accompany offers.

Your role is to:
1. Generate natural, persuasive messages that explain or justify the offer based on negotiation context.
2. Be BRIEF and DIRECT - keep messages under 20 words.
3. Consider any messages received from the other party and build rapport.
4. Focus on logical reasoning to convince the opponent.

You will receive:
- The offer being made (or acceptance/rejection decision)
- Any text received from the other party in their last offer
- Context about the negotiation state (outcome-space and history)

Any deal with positive utility is better than a disagreement (0 utility). 
Even when being firm, ensure the door remains open for a deal.

Your tone must match the current negotiation strategy:
1. STUBBORN MODE: If the offer is high utility for you and has not changed much, be firm. Say things like "This is my limit" or "I cannot concede further."
2. CONCEDING MODE: If you are lowering your demands, be cooperative. Say things like "I am making a concession for a fair deal" or "Let's meet in the middle."

Guidelines:
- Be BRIEF and DIRECT - keep messages under 20 words.
- Focus on logical reasoning to convince the opponent.
- Respond with ONLY a JSON object: {"text": "Your message"}
"""


class MiAgent(LLMMetaNegotiator):
    def __init__(self, **kwargs):
        super().__init__(
            base_negotiator=LinearTBNegotiator(), 
            system_prompt=SYSTEM_PROMPT,
            provider="ollama",
            model="qwen3:4b-instruct",
            **kwargs
        )
        self.strategy_fixed = False
        self.strategy_type = None
        self.stubborn_fixed = False
        self.opponent_offer_history = [] 
           
    def on_partner_proposal(self, partner_id, offer, state):
        print(f"DEBUG >>> 曲線の中身: {vars(self.base_negotiator._offering_curve)}")

        if offer:
            my_util = float(self.ufun(offer))
            self.opponent_offer_history.append({"offer": offer, "util": my_util})

        super().on_partner_proposal(partner_id=partner_id, offer=offer, state=state)
        opponent_outcomes = [t[1] for t in self.nmi.trace if t[0] == partner_id]
        total_diff = self.calculate_total_diff(opponent_outcomes)
        current_utility = float(self.ufun(offer))
        print(f"💰 相手からの提案の価値: {current_utility:.4f}")

        if not self.strategy_fixed:
            self.base_negotiator._offering_curve.exponent = 1.5

        if state.relative_time > 0.9:
            self.base_negotiator._offering_curve.exponent = 0.5
            print("⏳時間切れ間近！合意を優先して柔軟になります。")
            return

        if not self.strategy_fixed and 0.05 < state.relative_time < 0.1:
            if len(opponent_outcomes) < 5:
                return

            if total_diff < 0.05:
                if current_utility >= 0.5:
                    print(f"🕵️判定: 相手は【最初からお人好し】！")
                    self.base_negotiator._offering_curve.exponent = 0.2
                else:
                    print(f"🕵️判定完了: 相手は【頑固者】！")
                    self.base_negotiator._offering_curve.exponent = 7.0
                    self.strategy_type = "stubborn"
            elif total_diff > 0.2:
                print(f"🕵️判定完了: 相手は【お人好し】！")
                self.base_negotiator._offering_curve.exponent = 1.5
            else:
                print(f"🕵️判定完了: 相手は【普通】。")
                self.base_negotiator._offering_curve.exponent = 1.0
            
            self.strategy_fixed = True

        if not self.stubborn_fixed and self.strategy_type == 'stubborn':
            if total_diff < 0.001 and state.relative_time > 0.5:
                self.base_negotiator._offering_curve.exponent = 3.0
                self.stubborn_fixed = True

    def propose(self, state: SAOState) -> Outcome | None:
        my_next_extended = super().propose(state)
        
        if my_next_extended is None:
            return None
            
        if hasattr(my_next_extended, "outcome"):
            my_next_offer = my_next_extended.outcome
        else:
            my_next_offer = my_next_extended
            
        my_next_util = float(self.ufun(my_next_offer))
        
        if self.opponent_offer_history:
            best_past = max(self.opponent_offer_history, key=lambda x: x["util"])
            
            if my_next_util < best_past["util"]:
                print(f"💡【新・逆転検知】自分の次の提案(util:{my_next_util:.3f})より、"
                      f"相手の過去最高案(util:{best_past['util']:.3f})を採用します！")
                
                return best_past["offer"]

        # 5. 逆転していなければ、親クラスが作ったものをそのまま返す
        return my_next_extended

    def calculate_total_diff(self, opponent_outcomes):
        if not opponent_outcomes:
            return 0.0
        
        val_start = float(self.ufun(opponent_outcomes[0]))
        val_now = float(self.ufun(opponent_outcomes[-1]))
        total_diff = abs(val_now - val_start)
        return total_diff
    