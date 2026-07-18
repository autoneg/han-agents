"""A BOA-architecture bargaining agent combining time-dependent concession, frequency-based opponent modeling, and dynamic acceptance, with an LLM generating natural-language messages for its rule-based decisions."""

try:
    from ..adaptive_bargain import AdaptiveBargainNegotiator
except ImportError:
    from adaptive_bargain import AdaptiveBargainNegotiator

__all__ = ["AdaptiveBargainNegotiator"]
