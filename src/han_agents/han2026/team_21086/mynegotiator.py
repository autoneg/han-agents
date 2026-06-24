"""Compatibility shim — keeps the original ``mynegotiator.MyNegotiator``
import path working while the real implementation lives in ``semruk``.

The ANAC 2026 HAN League submission for team **Semruk** is implemented in
``semruk.SemrukNegotiator``. This module re-exports it under the legacy name
so existing tests, examples, and harness code that import
``mynegotiator.MyNegotiator`` continue to work unchanged.
"""

from semruk import SemrukNegotiator as MyNegotiator

__all__ = ["MyNegotiator"]
