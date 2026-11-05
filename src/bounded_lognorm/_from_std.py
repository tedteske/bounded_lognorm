"""
Sigma from a standard deviation (0.4.0): the inverse of

    sigma -> dist.from_bounds(mode, sigma, lower, upper).std()

for fixed mode and bounds, shared by BL and BL2.

Neither family has a closed-form inverse (docs/v0.4_from_std.md, section
1): BL's standard deviation is closed form in sigma but transcendental
(normal cdfs and exponentials at three arguments, even when symmetric), and
BL2's has no closed form at all, since its normalizing constant has none
(Phase 1). The inverse is found by root finding instead, which is exact to
the precision of std itself.

What the root finding relies on (section 2 of the note, numerical evidence,
not a proof): for every mode, at every dial position, std increases with
sigma up to sigma = _SIGMA_CAP * (upper - lower), so the root in that range
is unique. Further out, BL2's std can turn down (mode within ~1e-3 of a
bound, sigma beyond ~1e20 * (upper - lower)), so the search stops at the
cap instead of following sigma to infinity.

Uses only bounds_to_shapes and std, so it serves both families and honours
the instance's precision. Ted Teske with Claude (Anthropic), 2026.
"""
import warnings

import numpy as np
from scipy.optimize import brentq

# The largest sigma searched, as a multiple of upper - lower (or, with an
# unbounded side, of the larger of std and the finite distances from the
# mode to a bound). Monotonicity was checked to here with a margin of eight
# decades before BL2's first decrease (docs/v0.4_from_std.md, section 2).
_SIGMA_CAP = 1e12
# Outward bracket step in ln(sigma), and the root tolerance on ln(sigma / L),
# which is the relative accuracy of sigma.
_STEP = np.log(10.0)
_XTOL = 1e-13
_RTOL = 4 * np.finfo(float).eps
# std < sigma in every case checked (std/sigma -> 1 from below as the bounds
# recede; it reached 1 + 7e-9 only at sigma = 1e-9 * (upper - lower), where
# float64 resolution already limits accuracy), so std(_LO_FACTOR * std) < std
# starts the bracket with a wide margin. The downward steps are a guard only.
_LO_FACTOR = 0.5
_MAX_DOWN = 60


def _std(dist, mode, sigma, lower, upper):
    c, s, loc, scale = dist.bounds_to_shapes(mode, sigma, lower, upper)
    return float(dist.std(c, s, loc=loc, scale=scale))


def _unattainable(dist, mode, std, lower, upper, L):
    top = _std(dist, mode, _SIGMA_CAP * L, lower, upper)
    finite = lower is not None and upper is not None
    msg = ("std = %g is not attainable for %s with mode %g on [%s, %s]: "
           "std rises with sigma, and reaches only %.6g at sigma = %.0e * %s, "
           "the end of the search."
           % (std, dist.name, mode, lower, upper, top, _SIGMA_CAP,
              "(upper - lower)" if finite else "%g" % L))
    if finite:
        msg += (" As sigma grows the distribution flattens toward the "
                "uniform on [lower, upper], whose std is (upper - lower)/"
                "sqrt(12) = %.6g, but only logarithmically slowly. A sample "
                "this spread out is closer to uniform than any bounded "
                "lognormal with a usable sigma; check the bounds, or model "
                "it as uniform." % ((upper - lower) / np.sqrt(12)))
    return ValueError(msg)


def _solve(dist, mode, std, lower, upper):
    """Scalar sigma with dist.from_bounds(mode, sigma, lower, upper).std()
    == std. Inputs are already validated."""
    if lower is not None and upper is not None:
        L = upper - lower
    else:
        L = max([std] + [abs(b - mode) for b in (lower, upper)
                         if b is not None])

    def gap(u):
        g = _std(dist, mode, L * np.exp(u), lower, upper) - std
        if not np.isfinite(g):
            raise ValueError(
                "std could not be evaluated at sigma = %g for mode %g on "
                "[%s, %s]: the parameters are beyond float64 resolution "
                "(a bound at mode -/+ 1000 sigma swamps the finite one)."
                % (L * np.exp(u), mode, lower, upper))
        return g

    u_cap = np.log(_SIGMA_CAP)
    u_lo = np.log(_LO_FACTOR * std / L)
    if u_lo >= u_cap:
        raise _unattainable(dist, mode, std, lower, upper, L)
    g_lo = gap(u_lo)
    for _ in range(_MAX_DOWN):
        if g_lo < 0:
            break
        if g_lo == 0:
            return float(L * np.exp(u_lo))
        u_lo -= _STEP
        g_lo = gap(u_lo)
    else:                               # pragma: no cover - guard only
        raise RuntimeError("could not bracket sigma from below")

    while True:
        u_hi = min(u_lo + _STEP, u_cap)
        g_hi = gap(u_hi)
        if g_hi >= 0:
            break
        if u_hi >= u_cap:
            raise _unattainable(dist, mode, std, lower, upper, L)
        u_lo = u_hi
    if g_hi == 0:
        return float(L * np.exp(u_hi))
    u = brentq(gap, u_lo, u_hi, xtol=_XTOL, rtol=_RTOL)
    return float(L * np.exp(u))


def sigma_from_std(dist, mode, std, lower=None, upper=None):
    """The work behind dist.sigma_from_std; see its docstring."""
    mode = np.asarray(mode, dtype=float)
    std = np.asarray(std, dtype=float)
    lower = None if lower is None else np.asarray(lower, dtype=float)
    upper = None if upper is None else np.asarray(upper, dtype=float)
    given = [a for a in (mode, std, lower, upper) if a is not None]
    shape = np.broadcast_shapes(*(a.shape for a in given))

    if not all(np.all(np.isfinite(a)) for a in given):
        raise ValueError(
            "mode, std, lower and upper must be finite; use None for an "
            "unbounded side")
    if not np.all(std > 0):
        raise ValueError("Requires std > 0")
    if (lower is not None and not np.all(lower < mode)) or \
            (upper is not None and not np.all(mode < upper)):
        raise ValueError("Requires (lower < mode < upper)")

    def at(a, idx):
        return None if a is None else float(np.broadcast_to(a, shape)[idx])

    out = np.empty(shape)
    with warnings.catch_warnings():
        # from_bounds' float64-resolution warnings would repeat at every
        # trial sigma; from_std warns once, for the sigma it returns.
        warnings.simplefilter('ignore', RuntimeWarning)
        for idx in np.ndindex(shape):
            out[idx] = _solve(dist, at(mode, idx), at(std, idx),
                              at(lower, idx), at(upper, idx))
    return out[()]
