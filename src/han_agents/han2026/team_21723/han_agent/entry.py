"""Non-LLM bilateral negotiator using a time-dependent Boulware aspiration policy, frequency-based opponent modeling, cycle-aware exploration, and template-generated natural-language messages."""
from ..han_omega import HanOmegaNegotiator


class HybridNegotiator(HanOmegaNegotiator):
    pass


__all__ = ["HanOmegaNegotiator", "HybridNegotiator"]
