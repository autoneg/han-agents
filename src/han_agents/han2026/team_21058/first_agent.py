from negmas_llm import OllamaNegotiator

# =============================================================================
# Default Prompts (copied from negmas_llm, customize as needed)
# =============================================================================

# System prompt for MyNegotiator
SYSTEM_PROMPT = """
You are an expert negotiator participating in an automated negotiation.
Your goal is to negotiate effectively to achieve good outcomes for yourself
while finding mutually acceptable agreements when possible.

You will receive information about the negotiation setup (outcome space,
utility functions) at the start, and then be asked to make decisions for
each negotiation round.

Always respond in the exact JSON format requested. Be strategic and
rational, aiming to maximize your utility while reaching agreements.
"""

# Prompt sent when preferences are first set
PREFERENCES_PROMPT = """
# Negotiation Setup

You are about to participate in a negotiation. Here is the setup:

## Mechanism Information
The Negotiator Mechanism Interface (NMI) provides information about the negotiation:
- n_steps: Maximum number of negotiation steps (None = unlimited)
- time_limit: Maximum time in seconds (None = unlimited)
- n_outcomes: Total number of possible outcomes in the outcome space
- n_negotiators: Number of participants in this negotiation
- end_on_no_response: If true, negotiation ends when any negotiator returns None
- one_offer_per_step: If true, only one negotiator acts per step
- offering_is_accepting: If true, making an offer implies accepting it if echoed

{{nmi:text}}

## Outcome Space
The negotiation outcome space defines possible agreements:
{{outcome-space:json}}

## Your Utility Function
A utility function maps outcomes to real numbers representing preference.
- Higher values = more preferred outcomes
- reserved_value: The utility of no agreement (your walk-away point)
- You should aim to get outcomes with utility > reserved_value

{{utility-function:text}}
Your reserved value (utility of no agreement): {{reserved-value}}

## Opponent's Utility Function
{{opponent-utility-function:text}}
"""

# Prompt sent when preferences change during negotiation
PREFERENCES_CHANGED_PROMPT = """
# Preferences Changed

Your preferences have changed. Change types: {change_types}

## Mechanism Information
The Negotiator Mechanism Interface (NMI) provides information about the negotiation:
- n_steps: Maximum number of negotiation steps (None = unlimited)
- time_limit: Maximum time in seconds (None = unlimited)
- n_outcomes: Total number of possible outcomes in the outcome space
- n_negotiators: Number of participants in this negotiation
- end_on_no_response: If true, negotiation ends when any negotiator returns None
- one_offer_per_step: If true, only one negotiator acts per step
- offering_is_accepting: If true, making an offer implies accepting it if echoed

{{nmi:text}}

## Outcome Space
The negotiation outcome space defines possible agreements:
{{outcome-space:json}}

## Your Utility Function
A utility function maps outcomes to real numbers representing preference.
- Higher values = more preferred outcomes
- reserved_value: The utility of no agreement (your walk-away point)
- You should aim to get outcomes with utility > reserved_value

{{utility-function:text}}
Your reserved value (utility of no agreement): {{reserved-value}}

## Opponent's Utility Function
{{opponent-utility-function:text}}
"""

# Prompt sent when negotiation starts
NEGOTIATION_START_PROMPT = """
# Negotiation Started

The negotiation has now started. For each round, you will be asked to:
1. Analyze the current state and any offer received
2. Decide whether to ACCEPT, REJECT (with counter-offer), or END
3. Optionally provide persuasive text for the other party

Respond in this JSON format for each decision:
```json
{
    "response_type": "accept" | "reject" | "end" | "wait",
    "outcome": [value1, value2, ...] | null,
    "text": "optional persuasive message to send to your opponent",
    "reasoning": "brief explanation of your decision (not sent to opponent)"
}
```

Where:
- "accept": Accept the current offer on the table
- "reject": Reject and provide a counter-offer in "outcome"
- "end": End the negotiation without agreement
- "wait": Wait without making an offer (only if allowed by mechanism)
- "outcome": Your counter-offer as a list matching issue order, or null
- "text": A message to send to your opponent (actually delivered to them)
- "reasoning": Your internal reasoning (optional, not sent to opponent)

You may occasionally send ONLY text (null outcome) to persuade the
opponent, but this should be rare and strategic. Include an outcome usually.

Ready to begin!
"""

# Prompt sent each negotiation round
ROUND_PROMPT = """
# Round {step}

**Step**: {step} | **Time**: {relative_time:.1%} | **Running**: {running}

{offer_info}

What is your decision? Respond with JSON.
"""


class FirstAgent(OllamaNegotiator):
    """A simple LLM-based negotiator using Ollama.

    This negotiator uses a local Ollama instance to make negotiation decisions.
    It inherits all the LLM-based negotiation capabilities from OllamaNegotiator
    and can be customized through various prompts and parameters.
    """

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        use_structured_output: bool = True,
        timeout: float = 120.0,
        num_retries: int = 3,
        **kwargs,
    ):
        """Initialize the MiAgent with Ollama.

        Args:
            model: The Ollama model to use (default: "llama3.1:8b").
                Common options: "llama3.1:8b", "llama3.2", "mistral", "qwen2.5:7b"
            temperature: Sampling temperature for the LLM (default: 0.7).
                Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.3) make it more deterministic.
            max_tokens: Maximum tokens in the LLM response (default: 1024).
                Controls the length of generated responses.
            use_structured_output: If True, use structured output/JSON mode when supported (default: True).
                Guarantees valid JSON responses from the LLM.
            timeout: Timeout in seconds for LLM API calls (default: 120.0).
                Increase this if you experience timeout issues with slower models.
            num_retries: Number of retries for failed LLM API calls (default: 3).
                Helps handle transient connection issues with Ollama.
            **kwargs: Additional arguments passed to OllamaNegotiator parent class.
                May include: api_base (Ollama API URL), preferences (Preferences object),
                ufun (utility function), name (negotiator name), can_propose (bool), etc.
                Also supports prompt customization: system_prompt, preferences_prompt,
                preferences_changed_prompt, negotiation_start_prompt, round_prompt.
        """
        # Merge user-provided llm_kwargs with timeout/retry settings
        llm_kwargs = kwargs.pop("llm_kwargs", {})
        llm_kwargs.setdefault("timeout", timeout)
        llm_kwargs.setdefault("num_retries", num_retries)

        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            use_structured_output=use_structured_output,
            llm_kwargs=llm_kwargs,
            system_prompt=SYSTEM_PROMPT,
            preferences_prompt=PREFERENCES_PROMPT,
            preferences_changed_prompt=PREFERENCES_CHANGED_PROMPT,
            negotiation_start_prompt=NEGOTIATION_START_PROMPT,
            round_prompt=ROUND_PROMPT,
            **kwargs,
        )
