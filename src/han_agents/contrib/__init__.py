"""Hand-adapted variants of han2026 finalists, not part of the HAN 2026 submission set.

Unlike ``han_agents.han2026`` (generated, one-to-one with what teams
submitted), this package is hand-maintained. Each module here pairs with a
specific ``han2026`` agent whose original implementation talks to Ollama
through a hand-rolled HTTP client (bypassing ``negmas_llm``/``litellm``
entirely), so its provider could not be resolved centrally without changing
its wire format. The ``Generalized*`` classes in this package are that
provider-agnostic rewrite: same strategic/decision logic (often inherited
directly from the original), but the LLM call itself is routed through
``litellm`` so ``NEGMAS_LLM_PROVIDER``/``NEGMAS_LLM_MODEL`` (and friends)
work like they do for every other agent in this repo.

These are deliberately NOT registered in ``han_agents.registry`` — they are
not competition submissions, just usable, provider-agnostic alternatives.
"""

__all__: list[str] = []
