"""Equinox: reinforcement-learning bilateral negotiator for HAN 2026.

Strategy core adapted from the University of Tehran SCML 2025 SAC winner, mapped
onto NegMAS bilateral SAO negotiation with template-based (non-LLM) messages.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch

from negmas.gb.common import ExtendedResponseType, ResponseType
from negmas.gb.components.genius.models import GSmithFrequencyModel
from negmas.outcomes import ExtendedOutcome, Outcome
from negmas.preferences import BaseUtilityFunction, PresortingInverseUtilityFunction
from negmas.sao import SAOPRNegotiator, SAOState

try:
    from .equinox_messages import acceptance_text, rejection_text
    from .equinox_llm_messages import (
        Dynamics,
        Stance,
        classify_stance,
        template_message,
        time_phase,
        try_llm_message,
    )
    from .equinox_opponent_model import (
        HindriksTykhonovBayesianOpponentModel,
        bayesian_state_count,
    )
    from .equinox_policy import (
        DEFAULT_DELTA,
        DEFAULT_EXP_MAX,
        DEFAULT_EXP_MIN,
        DEFAULT_FLOOR,
        ActionBounds,
        action_to_concession_params,
        action_to_delta,
        load_policy,
        negotiation_start_observation,
        squash_action,
    )
except ImportError:
    from equinox_messages import acceptance_text, rejection_text
    from equinox_llm_messages import (
        Dynamics,
        Stance,
        classify_stance,
        template_message,
        time_phase,
        try_llm_message,
    )
    from equinox_opponent_model import (
        HindriksTykhonovBayesianOpponentModel,
        bayesian_state_count,
    )
    from equinox_policy import (
        DEFAULT_DELTA,
        DEFAULT_EXP_MAX,
        DEFAULT_EXP_MIN,
        DEFAULT_FLOOR,
        ActionBounds,
        action_to_concession_params,
        action_to_delta,
        load_policy,
        negotiation_start_observation,
        squash_action,
    )

__all__ = ["Equinox"]

# Pareto-nudge configuration.
#: Below this opponent-model confidence (in [0, 1]) the nudge is disabled and
#: Equinox falls back to its original in-band offer selection.
_NUDGE_TRUST_THRESHOLD = 0.05
#: Cap on the number of in-band candidate offers scored per proposal (bounds
#: cost on large domains; small domains stay fully enumerated).
_NUDGE_MAX_CANDIDATES = 1024
#: Use the full-joint Bayesian model only when its posterior fits under this
#: many states; larger domains fall back to the cheap frequency model.
_BAYES_MAX_STATES = 200_000

#: Minimum normalized-utility change that counts as a concession (someone moved).
_MOVE_EPS = 1e-3
#: Stances worth a (best-effort) LLM call; HOLDING recurs on the Boulware
#: plateau so it stays on cheap templates.
_SALIENT_STANCES = frozenset(
    {
        Stance.OPENING,
        Stance.RECIPROCATING,
        Stance.PRESSING,
        Stance.CLOSING,
        Stance.WALKING,
        Stance.ACCEPT,
    }
)
#: Hard cap on LLM message calls per negotiation (bounds latency).
_MAX_LLM_CALLS = 8


class Equinox(SAOPRNegotiator):
    """SAC-inspired bilateral negotiator for the HAN track."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        use_policy: bool = True,
        default_exp_min: float = DEFAULT_EXP_MIN,
        default_exp_max: float = DEFAULT_EXP_MAX,
        default_floor: float = DEFAULT_FLOOR,
        default_delta: float = DEFAULT_DELTA,
        use_llm_messages: bool = True,
        ufun: BaseUtilityFunction | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(ufun=ufun, **kwargs)
        self._use_policy = use_policy
        self._use_llm_messages = use_llm_messages
        self._policy = load_policy(model_path) if use_policy else None
        self._default_exp_min = default_exp_min
        self._default_exp_max = default_exp_max
        self._default_floor = default_floor
        self._default_delta = default_delta

        self._inverter: PresortingInverseUtilityFunction | None = None
        self._best: Outcome | None = None
        self._mx = 1.0
        self._mn = 0.0
        self._exp_min = default_exp_min
        self._exp_max = default_exp_max
        self._floor = default_floor
        # delta = how far below the concession band the opponent-aware nudge may
        # reach when it buys the opponent a much better deal (learned action[3]).
        self._delta = default_delta
        self._opponent_model = GSmithFrequencyModel()
        self._partner_bid_count = 0
        self._external_action: torch.Tensor | None = None
        self._action_bounds = ActionBounds()
        self._reset_message_state()

    def apply_concession_from_action(self, action: torch.Tensor) -> None:
        """Set concession parameters from a policy action (used during training)."""
        exp_min, exp_max, floor = action_to_concession_params(action)
        self._exp_min = max(exp_min, 1e-3)
        self._exp_max = max(exp_max, 1e-3)
        self._floor = floor
        self._delta = action_to_delta(action)
        self._external_action = action.detach().cpu()

    def on_negotiation_start(self, state: SAOState) -> None:
        super().on_negotiation_start(state)
        self._reset_session(state)

    def _reset_session(self, state: SAOState) -> None:
        _ = state
        self._inverter = None
        self._best = None
        self._mx = 1.0
        self._mn = 0.0
        self._partner_bid_count = 0
        self._reset_message_state()
        self._opponent_model = self._build_opponent_model()
        self.private_info["opponent_ufun"] = self._opponent_model
        if self._external_action is not None:
            self.apply_concession_from_action(self._external_action)
        else:
            self._sample_concession_params()

    def _build_opponent_model(self):
        """Pick an opponent model that fits the domain.

        The full-joint Bayesian model is accurate but its posterior is
        exponential in the issue count, so it is used only on small domains; we
        fall back to the cheap Smith frequency model otherwise. Either choice
        backs both the Deception score and the propose-time Pareto nudge.
        """
        outcome_space = self.nmi.outcome_space if self.nmi is not None else None
        if outcome_space is None and self.ufun is not None:
            outcome_space = getattr(self.ufun, "outcome_space", None)

        if (
            outcome_space is not None
            and bayesian_state_count(outcome_space) <= _BAYES_MAX_STATES
        ):
            model = HindriksTykhonovBayesianOpponentModel(max_states=_BAYES_MAX_STATES)
            try:
                model.set_negotiator(self)
            except Exception:
                pass
            model.init_from_outcome_space(outcome_space)
            if model._initialized:
                return model

        fallback = GSmithFrequencyModel()
        fallback.set_negotiator(self)
        fallback._initialize()
        return fallback

    def _role_flag(self) -> float:
        if self.nmi is None:
            return 1.0
        try:
            index = self.nmi.negotiator_index(self.id)
        except Exception:
            return 1.0
        return 1.0 if index in (None, 0) else 0.0

    def _sample_concession_params(self) -> None:
        if self._policy is None:
            self._exp_min = self._default_exp_min
            self._exp_max = self._default_exp_max
            self._floor = self._default_floor
            self._delta = self._default_delta
            return

        n_steps = getattr(self.nmi, "n_steps", 100) if self.nmi else 100
        obs = negotiation_start_observation(
            ufun=self.ufun,
            n_steps=n_steps or 100,
            role_flag=self._role_flag(),
        )
        with torch.no_grad():
            mu, _sigma = self._policy(obs.unsqueeze(0))
            action_low, action_high = self._action_bounds.as_tensors(device=mu.device)
            squashed = squash_action(mu, action_low, action_high).squeeze(0)
            exp_min, exp_max, floor = action_to_concession_params(squashed)
        self._exp_min = max(exp_min, 1e-3)
        self._exp_max = max(exp_max, 1e-3)
        self._floor = floor
        self._delta = action_to_delta(squashed)

    def _ensure_inverter(self) -> None:
        if self._inverter is not None or self.ufun is None:
            return
        self._inverter = PresortingInverseUtilityFunction(self.ufun, rational_only=True)
        self._inverter.init()
        self._best = self._inverter.best()
        self._mx = self._inverter.max()
        reserved = float(self.ufun(None))
        self._mn = max(self._inverter.min(), reserved)
        if self._mx == 0:
            self._mx = 1.0

    def _utility(self, offer: Outcome | None, *, normalized: bool = False) -> float:
        if offer is None or self.ufun is None:
            return 0.0
        if normalized:
            return float(self.ufun.eval_normalized(offer))
        return float(self.ufun(offer))

    def _proposal_in_range(self, u_min: float, u_max: float) -> Outcome | None:
        self._ensure_inverter()
        assert self._inverter is not None
        proposal = self._inverter.one_in((u_min, u_max), normalized=True)
        if proposal is None:
            return self._best
        return proposal

    def _opp_utility(self, outcome: Outcome | None) -> float:
        """Estimated opponent utility for an outcome (uniform across models)."""
        if outcome is None:
            return 0.0
        try:
            return float(self._opponent_model.eval_normalized(outcome))
        except Exception:
            try:
                return float(self._opponent_model(outcome))
            except Exception:
                return 0.0

    def _opp_trust(self) -> float:
        """Confidence in the opponent model, in [0, 1].

        For the Bayesian model this is ``1 - normalized posterior entropy`` once
        it has observed a bid; otherwise it ramps with the number of observed
        partner offers. This gates how aggressively the Pareto nudge acts so we
        never trade utility away on an uninformed model.
        """
        model = self._opponent_model
        entropy_fn = getattr(model, "normalized_entropy", None)
        if callable(entropy_fn) and getattr(model, "n_observed_bids", 0) > 0:
            value = entropy_fn()
            if math.isfinite(value):
                return max(0.0, min(1.0, 1.0 - value))
        return min(1.0, self._partner_bid_count / 5.0)

    def _select_proposal(self, u_min: float, u_max: float) -> Outcome | None:
        """Pick an in-band offer, nudged toward the opponent's preferences.

        The RL policy fixes the self-utility band ``[u_min, u_max]``; within it
        (optionally widened downward by the learned ``delta`` budget when the
        opponent model is confident) we choose the outcome the opponent likes
        most, breaking ties by our own utility. This is a Pareto-rational
        refinement that leaves the policy's concession schedule untouched, and
        it falls back to the original random in-band pick whenever the opponent
        model is not yet informative.
        """
        self._ensure_inverter()
        assert self._inverter is not None

        trust = self._opp_trust()
        if trust <= _NUDGE_TRUST_THRESHOLD:
            return self._proposal_in_range(u_min, u_max)

        u_lo = max(self._floor, u_min - self._delta * trust)
        u_lo = min(u_lo, u_max)
        candidates = self._inverter.some(
            (u_lo, u_max), normalized=True, n=_NUDGE_MAX_CANDIDATES
        )
        if not candidates:
            return self._proposal_in_range(u_min, u_max)

        return max(
            candidates,
            key=lambda outcome: (
                self._opp_utility(outcome),
                self._utility(outcome, normalized=True),
            ),
        )

    def _safe_fallback_offer(self) -> Outcome | None:
        """A guaranteed-valid offer for when the inverter machinery errors out."""
        if self._best is not None:
            return self._best
        try:
            return self.ufun.best() if self.ufun is not None else None
        except Exception:
            return None

    def _offer_bounds(self, relative_time: float) -> tuple[float, float]:
        t = min(max(relative_time, 0.0), 1.0)
        u_min = 1.0 - t**self._exp_min
        u_max = 1.0 - t**self._exp_max
        if u_max < u_min:
            u_min, u_max = u_max, u_min
        # The learned floor stops the curve from collapsing to 0 at the
        # deadline, preventing exploitation by hardliner opponents.
        u_min = max(u_min, self._floor)
        u_max = max(u_max, self._floor)
        return u_min, u_max

    def _reset_message_state(self) -> None:
        """Reset per-session state for stance-aware messaging."""
        self._our_offer_count = 0
        self._last_our_offer_u: float | None = None
        self._last_opp_u: float | None = None
        self._they_moved = False
        self._their_offer_acceptable = False
        self._last_llm_stance: Stance | None = None
        self._llm_healthy = True
        self._llm_calls = 0

    def _human_session(self) -> bool:
        """Only spend LLM calls when we are (heuristically) facing a human.

        HANI human sessions are wall-clock bounded (a *finite* ``time_limit``)
        and/or run without a fixed step budget; agent tournaments and training
        are step-bounded with an infinite ``time_limit`` (negmas' default when
        only ``n_steps`` is given). ``EQUINOX_FORCE_LLM_MSG`` overrides for local
        testing / finals if the heuristic ever misfires.
        """
        if not self._use_llm_messages:
            return False
        if os.environ.get("EQUINOX_FORCE_LLM_MSG"):
            return True
        nmi = self.nmi
        if nmi is None:
            return False
        time_limit = getattr(nmi, "time_limit", None)
        n_steps = getattr(nmi, "n_steps", None)
        has_wall_clock = time_limit is not None and math.isfinite(time_limit)
        return has_wall_clock or n_steps is None

    def _should_llm(self, stance: Stance) -> bool:
        """Key-moments-only gate: bounds LLM calls to ~one per stance change."""
        if not self._llm_healthy or self._llm_calls >= _MAX_LLM_CALLS:
            return False
        if stance not in _SALIENT_STANCES:
            return False
        if stance in (Stance.CLOSING, Stance.ACCEPT):
            return True
        return stance != self._last_llm_stance

    def _stance_text(self, stance: Stance, dynamics: Dynamics) -> str:
        """Chat text for a stance; LLM at key moments, template otherwise.

        Never raises and never stalls the negotiation: the first LLM failure
        trips a per-session breaker so we stop retrying a slow/absent model.
        """
        try:
            if self._human_session() and self._should_llm(stance):
                self._last_llm_stance = stance
                self._llm_calls += 1
                rendered = try_llm_message(stance, dynamics)
                if rendered is None:
                    self._llm_healthy = False
                else:
                    return rendered
            return template_message(stance)
        except Exception:
            return self._fallback_text(stance)

    def _fallback_text(self, stance: Stance) -> str:
        """Last-resort deterministic text if even templating misbehaves."""
        try:
            if stance is Stance.ACCEPT:
                return acceptance_text()
            return rejection_text()
        except Exception:
            return ""

    def _update_opponent_dynamics(self, offer: Outcome | None) -> None:
        """Track (privately) whether the opponent conceded and is acceptable."""
        if offer is None or self.ufun is None:
            return
        try:
            opp_u = self._utility(offer, normalized=True)
            self._they_moved = (
                self._last_opp_u is not None and opp_u > self._last_opp_u + _MOVE_EPS
            )
            self._last_opp_u = opp_u
            self._their_offer_acceptable = float(self.ufun(offer)) >= float(self.ufun(None))
        except Exception:
            self._they_moved = False
            self._their_offer_acceptable = False

    def _proposal_message(self, proposal: Outcome, state: SAOState) -> str:
        """Stance-aware text to accompany our counter-offer (never raises)."""
        try:
            opp_offer = state.current_offer
            is_opening = self._our_offer_count == 0 and opp_offer is None
            our_u = self._utility(proposal, normalized=True)
            we_moved = (
                self._last_our_offer_u is not None
                and our_u < self._last_our_offer_u - _MOVE_EPS
            )
            stance = classify_stance(
                is_opening=is_opening,
                is_accept=False,
                relative_time=state.relative_time,
                we_moved=we_moved,
                they_moved=self._they_moved,
                their_offer_acceptable=self._their_offer_acceptable,
            )
            dynamics = Dynamics(
                we_moved=we_moved,
                they_moved=self._they_moved,
                time_phase=time_phase(state.relative_time),
            )
            text = self._stance_text(stance, dynamics)
            self._last_our_offer_u = our_u
            self._our_offer_count += 1
            return text
        except Exception:
            return self._fallback_text(Stance.HOLDING)

    def _wrap_proposal(
        self,
        proposal: Outcome | None,
        state: SAOState,
    ) -> ExtendedOutcome | Outcome | None:
        if proposal is None:
            return None
        text = self._proposal_message(proposal, state)
        return ExtendedOutcome(outcome=proposal, data={"text": text})

    def propose(self, state: SAOState, dest: str | None = None) -> ExtendedOutcome | Outcome | None:
        _ = dest
        try:
            u_min, u_max = self._offer_bounds(state.relative_time)
            proposal = self._select_proposal(u_min, u_max)
        except Exception:
            # Never let an offering-side error (e.g. a negmas inverter edge
            # case) crash the negotiation; fall back to our best outcome.
            proposal = self._safe_fallback_offer()
        return self._wrap_proposal(proposal, state)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType | ExtendedResponseType:
        offer = state.current_offer
        if offer is not None and source and source != self.id:
            try:
                self._opponent_model.on_partner_proposal(state, source, offer)
                self._partner_bid_count += 1
                self.private_info["opponent_ufun"] = self._opponent_model
            except Exception:
                # A model-update error must not break our acceptance logic.
                pass
            self._update_opponent_dynamics(offer)

        self._ensure_inverter()
        reserved = float(self.ufun(None)) if self.ufun else 0.0
        if self._mx < reserved:
            return ResponseType.END_NEGOTIATION

        if offer is None:
            return ResponseType.REJECT_OFFER

        offer_util = self._utility(offer)
        offer_util_norm = self._utility(offer, normalized=True)
        u_reject, _ = self._offer_bounds(state.relative_time)

        if offer_util_norm < u_reject or offer_util < reserved:
            # The substantive, stance-aware line rides on the counter-offer the
            # mechanism requests next (propose); keep the reject itself quiet so
            # the human sees one message per round.
            return ResponseType.REJECT_OFFER

        dyn = Dynamics(
            we_moved=False,
            they_moved=self._they_moved,
            time_phase=time_phase(state.relative_time),
        )
        return ExtendedResponseType(
            response=ResponseType.ACCEPT_OFFER,
            data={"text": self._stance_text(Stance.ACCEPT, dyn)},
        )
