"""Provider-agnostic variant of
``han_agents.han2026.team_22270.civic_compass.CivicCompassHANNegotiator``.

The original talks to Ollama via a hand-rolled ``urllib`` POST to its native
``/api/generate`` endpoint (see ``CivicCompassHANNegotiator._call_ollama``),
gated by an Ollama-specific ``/api/tags`` readiness probe
(``_ensure_ollama_ready``). Neither endpoint exists on other providers, so
its provider is architecturally fixed to Ollama.
``GeneralizedCivicCompassHANNegotiator`` keeps every strategic/decision
method unchanged (inherited as-is) and overrides only the LLM transport,
routing it through ``litellm`` so ``provider``/``model`` resolve centrally
like every other agent in this repo (explicit arg > per-type env var >
global env var > built-in default of ollama/qwen3:4b-instruct).

Behavioral note: this is NOT purely a config change. litellm's ``ollama``
provider calls Ollama's native ``/api/generate`` (single-shot prompt,
matching the original); this class instead uses ``negmas_llm``'s
``ollama_chat`` routing (``/api/chat``, proper chat templating) for the
default case, matching how every other negmas_llm-based agent in this repo
talks to Ollama. That means the *default* behavior of this class can differ
slightly from the original ``CivicCompassHANNegotiator``: same model,
different request shape, so completions are not guaranteed byte-identical.
The readiness pre-check (``_ensure_ollama_ready``) is also dropped — it is
meaningless for non-Ollama providers — so an unreachable endpoint is now
discovered on the first real call instead of a cheap upfront probe. Use
``CivicCompassHANNegotiator`` (han2026.team_22270.civic_compass) if you need
the exact tuned/submitted behavior; use this class if you need to point the
agent at a non-Ollama provider.
"""

from __future__ import annotations

import litellm
from negmas_llm.common import litellm_model_string
from negmas_llm.config import resolve_llm_config

from han_agents.han2026.team_22270.civic_compass import CivicCompassHANNegotiator


class GeneralizedCivicCompassHANNegotiator(CivicCompassHANNegotiator):
    """``CivicCompassHANNegotiator`` with its LLM transport routed through litellm.

    See the module docstring for what "generalized" means here and how its
    default behavior can differ from ``CivicCompassHANNegotiator``'s
    tuned/submitted output.
    """

    def __init__(
        self,
        *args,
        provider: str | None = None,  # resolved centrally
        model: str | None = None,  # resolved centrally
        api_base: str | None = None,  # resolved centrally
        **kwargs,
    ) -> None:
        super().__init__(*args, model=model, api_base=api_base, **kwargs)
        resolved = resolve_llm_config(
            type(self).__name__,
            provider=provider,
            model=model,
            api_base=api_base,
            default_provider="ollama",
            # default_model intentionally omitted: None defers to negmas_llm's
            # per-provider fallback table, so a provider override gets that
            # provider's own default model instead of inheriting Ollama's
            # (or the parent's legacy HAN_LLM_MODEL env var, which only makes
            # sense for the Ollama case anyway).
        )
        self._llm_provider = resolved.provider
        self._llm_model = resolved.model
        if resolved.provider == "ollama":
            # Preserve the original's "always has a usable local endpoint"
            # default; other providers resolve their own endpoint (litellm
            # picks the right one from the provider name and env-var creds).
            self._llm_api_base = resolved.api_base or self._llm_url.rsplit(
                "/api/generate", 1
            )[0]
        else:
            self._llm_api_base = resolved.api_base

    def _call_ollama(self, prompt: str) -> str | None:
        if self._llm_failed:
            return None
        self._llm_calls += 1
        call_kwargs = {
            "model": litellm_model_string(self._llm_provider, self._llm_model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.35,
            "top_p": 0.85,
            "max_tokens": 90,
            "timeout": self._llm_timeout,
        }
        if self._llm_api_base:
            call_kwargs["api_base"] = self._llm_api_base
        try:
            response = litellm.completion(**call_kwargs)
            text = response.choices[0].message.content
            return text if isinstance(text, str) else None
        except Exception:
            self._llm_failed = True
            return None
