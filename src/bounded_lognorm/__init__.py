"""
bounded_lognorm -- bounded lognormal continuous probability distributions.

Two distributions, each a scipy.stats.rv_continuous subclass with a
ready-made instance, as with scipy.stats.norm:

  bl2  BL2, the geometric-mean form (Eq. 3): smooth everywhere; cdf and
       quantiles by quadrature, with a speed/precision dial.
       Implementation: bounded_lognorm._bl2.
  bl   BL, the piecewise form (Eq. 2): lognormal pieces joined at the mode;
       everything in closed form. Implementation: bounded_lognorm._bl.

They share the parameters and the bounds bridge but are DIFFERENT
distributions; do not substitute one for the other.

Usage: both are scipy location-scale families with shapes c (relative mode)
and s (relative sigma) on [0, 1]; dist.from_bounds(mode, sigma, lower, upper)
returns a frozen distribution in the bounds form.

Attribution (see ATTRIBUTION.md for the full, tiered breakdown and references):
  - Original work (2020-2024): Ted Teske -- the distributions, the mode/sigma
    reparametrization, the rv_continuous subclasses and their API.
  - Optimization & dial (2026): Ted Teske with Claude (Anthropic).
  - Phase 2 math fixes, Phase 3 scipy.stats conformance, and the Phase 5
    port of BL (2026): Ted Teske with Claude (Anthropic).
"""
from ._bl2 import bounded_lognorm2_gen, bl2, PRECISION_DIAL
from ._bl import bounded_lognorm_gen, bl

__version__ = "0.2.1"

__all__ = ["bounded_lognorm2_gen", "bl2", "bounded_lognorm_gen", "bl",
           "PRECISION_DIAL", "__version__"]

_MIGRATION = (
    "bounded_lognorm.bounded_lognorm (the notebook's instance) is not "
    "provided: its API changed, and an alias would let old calls such as "
    "bounded_lognorm.pdf(x, mode, sigma, lower, upper) run and return NaN or "
    "wrong values. Use `from bounded_lognorm import bl` and "
    "bl.from_bounds(mode, sigma, lower, upper).pdf(x); see the migration "
    "notes in docs/phase5_bl_port.md.")


def __getattr__(name):
    # ImportError, not AttributeError: `from bounded_lognorm import
    # bounded_lognorm` (as the 100m-sprints notebook does) converts an
    # AttributeError into a generic "cannot import name" and drops this
    # message, while an ImportError reaches the user as written.
    if name == "bounded_lognorm":
        raise ImportError(_MIGRATION, name=name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
