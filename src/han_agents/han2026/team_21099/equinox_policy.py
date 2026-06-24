"""Soft Actor-Critic policy network for Equinox.

Adapted from the University of Tehran SCML 2025 winner:
https://github.com/autoneg/anl-agents/blob/main/src/anl_agents/anl2025/university_of_tehran/sac.py
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Policy",
    "ActionBounds",
    "DEFAULT_ACTION_LOW",
    "DEFAULT_ACTION_HIGH",
    "DEFAULT_EXP_MIN",
    "DEFAULT_EXP_MAX",
    "DEFAULT_FLOOR",
    "DEFAULT_DELTA",
    "OBSERVATION_SIZE",
    "ACTION_SIZE",
    "action_to_concession_params",
    "action_to_delta",
    "load_policy",
    "negotiation_start_observation",
    "scenario_features",
    "squash_action",
    "squash_log_det_jacobian",
]

DEVICE = torch.device("cpu")

# Bundled checkpoint for competition submission (see README submission section).
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "saves" / "p_model.pt"

# Boulware-like defaults used when no trained checkpoint is available.
DEFAULT_EXP_MIN = math.e
DEFAULT_EXP_MAX = 1.0
DEFAULT_FLOOR = 0.0
# No band widening by default: the opponent-aware nudge only reorders offers
# inside the policy's self-utility band until a trained policy learns a delta.
DEFAULT_DELTA = 0.0

# Observation: [role_flag, n_steps, log-outcomes, n_issues, reserved value,
#               mean utility, std utility, rational fraction]
OBSERVATION_SIZE = 8
# Action: [a1, a2] -> concession exponents, [a3] -> late-game utility floor,
#         [a4] -> Pareto-nudge band-widening budget (delta, in normalized utility).
ACTION_SIZE = 4

DEFAULT_ACTION_LOW = (-1.0, 0.0, 0.0, 0.0)
DEFAULT_ACTION_HIGH = (3.0, 3.0, 0.9, 0.2)

_MAX_FEATURE_SAMPLES = 1_000


@dataclass(frozen=True)
class ActionBounds:
    low: tuple[float, ...] = DEFAULT_ACTION_LOW
    high: tuple[float, ...] = DEFAULT_ACTION_HIGH

    def as_tensors(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.tensor(self.low, device=device, dtype=torch.float32)
        high = torch.tensor(self.high, device=device, dtype=torch.float32)
        return low, high


class Policy(nn.Module):
    """Gaussian policy head used by the SCML SAC agent."""

    def __init__(self, observation_size: int = OBSERVATION_SIZE, action_size: int = ACTION_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(observation_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_size)
        self.logvar = nn.Linear(128, action_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        logvar = torch.clamp(self.logvar(x), -5, 2)
        return self.mean(x), logvar.exp()


def scenario_features(ufun: Any) -> dict[str, float]:
    """Extract scenario-level features from a utility function.

    Returns raw (unscaled) features describing the negotiation domain from the
    perspective of the owner of ``ufun``.
    """
    features = {
        "n_outcomes": 1.0,
        "n_issues": 0.0,
        "reserved_value": 0.0,
        "mean_utility": 0.5,
        "std_utility": 0.0,
        "rational_fraction": 1.0,
    }
    if ufun is None:
        return features

    reserved = getattr(ufun, "reserved_value", 0.0)
    if reserved is not None and math.isfinite(float(reserved)):
        features["reserved_value"] = float(reserved)

    outcome_space = getattr(ufun, "outcome_space", None)
    if outcome_space is None:
        return features

    issues = getattr(outcome_space, "issues", None) or []
    features["n_issues"] = float(len(issues))
    cardinality = getattr(outcome_space, "cardinality", None)
    if cardinality is not None and math.isfinite(float(cardinality)):
        features["n_outcomes"] = float(cardinality)

    try:
        outcomes = list(
            outcome_space.enumerate_or_sample(max_cardinality=_MAX_FEATURE_SAMPLES)
        )
    except Exception:
        outcomes = []
    if outcomes:
        utils = [float(ufun(outcome)) for outcome in outcomes]
        n = len(utils)
        mean = sum(utils) / n
        variance = sum((u - mean) ** 2 for u in utils) / n
        features["mean_utility"] = mean
        features["std_utility"] = math.sqrt(variance)
        features["rational_fraction"] = sum(
            1 for u in utils if u >= features["reserved_value"]
        ) / n
    return features


def negotiation_start_observation(
    *,
    ufun: Any = None,
    n_steps: int = 100,
    role_flag: float = 1.0,
) -> torch.Tensor:
    """Build the policy observation for a negotiation, given the agent's ufun.

    All features are scaled to roughly [0, 1] so the network sees a
    well-conditioned input regardless of domain size.
    """
    feats = scenario_features(ufun)
    data = [
        float(role_flag),
        min(float(n_steps), 1000.0) / 1000.0,
        math.log10(feats["n_outcomes"] + 1.0) / 4.0,
        feats["n_issues"] / 10.0,
        feats["reserved_value"],
        feats["mean_utility"],
        feats["std_utility"],
        feats["rational_fraction"],
    ]
    return torch.tensor(data, device=DEVICE, dtype=torch.float32)


def squash_action(
    raw: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    """Map unbounded Gaussian samples into a bounded action range."""
    return action_low + (torch.tanh(raw) + 1.0) / 2.0 * (action_high - action_low)


def squash_log_det_jacobian(
    raw: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    """Log-determinant of the tanh squashing Jacobian (per action dimension)."""
    scale = (action_high - action_low) / 2.0
    return torch.log(scale.clamp_min(1e-6)) + torch.log(
        (1.0 - torch.tanh(raw).pow(2)).clamp_min(1e-6)
    )


def action_to_concession_params(action: torch.Tensor) -> tuple[float, float, float]:
    """Map policy output to concession parameters used by the bidding rule.

    The exponent mapping matches the reference implementation
    (exp_min = e^(a1-a2), exp_max = e^a1). The third dimension is a utility
    floor below which Equinox never offers or accepts (normalized utility).
    """
    values = action.detach().cpu().tolist()
    action1, action2 = values[0], values[1]
    floor = values[2] if len(values) > 2 else DEFAULT_FLOOR
    if action2 < 0:
        action2 = 0.0
    floor = min(max(floor, 0.0), 0.95)
    return math.e ** (action1 - action2), math.e**action1, floor


def action_to_delta(action: torch.Tensor) -> float:
    """Map the 4th policy output to the Pareto-nudge band-widening budget.

    ``delta`` is how much normalized self-utility Equinox is willing to give up,
    below the policy's concession band, when doing so buys the opponent a much
    better deal (a Pareto-rational concession). It is clamped defensively even
    though the action bounds already constrain it; legacy 3-d actions widen by
    nothing.
    """
    values = action.detach().cpu().tolist()
    if len(values) < 4:
        return DEFAULT_DELTA
    return float(min(max(values[3], 0.0), 0.5))


def load_policy(model_path: str | Path | None = None) -> Policy | None:
    """Load a trained policy checkpoint.

    Returns ``None`` when no checkpoint file is found so callers fall back to
    the Boulware-style defaults instead of using an untrained network.
    """
    if model_path is None:
        model_path = os.environ.get("EQUINOX_MODEL_PATH")
    if model_path is None:
        # Module-relative first (competition submission layout), then CWD
        # (running the installed package from the project root).
        for candidate in (DEFAULT_CHECKPOINT, Path.cwd() / "models" / "saves" / "p_model.pt"):
            if candidate.is_file():
                model_path = candidate
                break

    if not model_path:
        return None
    path = Path(model_path)
    if not path.is_file():
        return None

    policy = Policy().to(DEVICE)
    policy.eval()
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        policy.load_state_dict(state)
    except Exception as exc:
        # A checkpoint trained against a different observation/action layout
        # (e.g. a pre-delta 3-d action head) must not brick the agent: fall
        # back to the Boulware-style default schedule instead of crashing.
        warnings.warn(
            f"Equinox: ignoring incompatible policy checkpoint at {path} "
            f"({exc}); using the default concession schedule.",
            stacklevel=2,
        )
        return None
    return policy
