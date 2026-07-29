"""Provider-agnostic variant of ``han_agents.han2026.team_22147.agent96.Agent96``.

The original ``Agent96`` talks to Ollama via a hand-rolled ``urllib`` POST to
its native ``/api/generate`` endpoint (see ``Agent96._call_ollama``), so its
provider is architecturally fixed to Ollama — that HTTP call has no idea how
to talk to OpenAI, Anthropic, or any other provider. ``GeneralizedAgent96``
keeps every strategic/decision-making method unchanged (inherited as-is) and
overrides only the LLM transport, routing it through ``litellm`` so
``provider``/``model`` resolve centrally like every other agent in this repo
(explicit arg > per-type env var > global env var > built-in default of
ollama/qwen3:4b-instruct).

Behavioral note: this is NOT purely a config change. litellm's ``ollama``
provider calls Ollama's native ``/api/generate`` (single-shot prompt,
matching the original); this class instead uses ``negmas_llm``'s
``ollama_chat`` routing (``/api/chat``, proper chat templating) for the
default case, matching how every other negmas_llm-based agent in this repo
talks to Ollama. That means the *default* behavior of this class can differ
slightly from the original ``Agent96`` — same model, different request
shape, so completions are not guaranteed byte-identical. Use ``Agent96``
(han2026.team_22147.agent96) if you need the exact tuned/submitted behavior;
use this class if you need to point the agent at a non-Ollama provider.
"""

from __future__ import annotations

import litellm
from negmas_llm.common import litellm_model_string
from negmas_llm.config import resolve_llm_config

from han_agents.han2026.team_22147.agent96 import Agent96


class GeneralizedAgent96(Agent96):
    """``Agent96`` with its LLM transport routed through litellm.

    See the module docstring for what "generalized" means here and how its
    default behavior can differ from ``Agent96``'s tuned/submitted output.
    """

    def __init__(
        self,
        provider: str | None = None,  # resolved centrally
        model: str | None = None,  # resolved centrally
        api_base: str | None = None,  # resolved centrally
        **kwargs,
    ) -> None:
        super().__init__(model=model, api_base=api_base, **kwargs)
        resolved = resolve_llm_config(
            type(self).__name__,
            provider=provider,
            model=model,
            api_base=api_base,
            default_provider="ollama",
            # default_model intentionally omitted: None defers to negmas_llm's
            # per-provider fallback table, so a provider override gets that
            # provider's own default model instead of inheriting Ollama's.
        )
        self.provider = resolved.provider
        self.model = resolved.model
        if resolved.provider == "ollama":
            # Preserve Agent96's "always has a usable local endpoint" default.
            self.api_base = resolved.api_base or "http://localhost:11434"
        else:
            # Other providers resolve their own endpoint; litellm/env vars
            # (e.g. OPENAI_API_KEY) supply credentials, not api_base.
            self.api_base = resolved.api_base

    def _call_ollama(
        self,
        prompt: str,
        timeout: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        messages = [{"role": "user", "content": prompt}]
        call_kwargs = {
            "model": litellm_model_string(self.provider, self.model),
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "timeout": self.timeout if timeout is None else timeout,
        }
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        for _ in range(self.num_retries + 1):
            try:
                response = litellm.completion(**call_kwargs)
                text = response.choices[0].message.content
                self._llm_available = True
                return text.strip() if isinstance(text, str) else None
            except Exception:
                continue
        self._llm_available = False
        return None
