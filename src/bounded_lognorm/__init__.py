"""
bounded_lognorm -- bounded lognormal continuous probability distributions.

This package provides the BL2 (bounded lognormal 2) distribution, a custom
scipy.stats.rv_continuous subclass using the non-piecewise geometric-mean
formulation (Eq. 3). `bl2` is a ready-made instance, as with scipy.stats.norm;
the implementation and its speed/precision dial are in bounded_lognorm._bl2.

Attribution (see ATTRIBUTION.md for the full, tiered breakdown and references):
  - Original work (2020-2024): Ted Teske -- the distributions, the mode/sigma
    reparametrization, the rv_continuous subclass and its API.
  - Optimization & dial (2026): Ted Teske with Claude (Anthropic).
  - Phase 2 math fixes and Phase 3 scipy.stats conformance (2026): Ted Teske
    with Claude (Anthropic).

Usage: BL2 is a scipy location-scale family with shapes c (relative mode) and
s (relative sigma) on [0, 1]; bl2.from_bounds(mode, sigma, lower, upper)
returns a frozen distribution in the bounds form.
"""
from ._bl2 import bounded_lognorm2_gen, bl2, PRECISION_DIAL

__version__ = "0.1.0"

__all__ = ["bounded_lognorm2_gen", "bl2", "PRECISION_DIAL", "__version__"]
