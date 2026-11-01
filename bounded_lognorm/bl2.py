"""
bounded_lognorm2 (BL2) -- bounded lognormal distribution, geometric-mean form.

A scipy.stats.rv_continuous subclass. The pdf is the normalized geometric
mean of a lognormal at the lower bound and a reflected lognormal at the upper
bound (Eq. 3). Since Phase 3 it is a standard scipy location-scale family:
shapes c = (mode - lower)/(upper - lower) and s = sigma/(upper - lower) on
[0, 1], with loc = lower and scale = upper - lower. bl2.from_bounds(mode,
sigma, lower, upper) builds a frozen distribution from the bounds form.

Speed vs. precision is controlled by a single dial set at construction:
    bounded_lognorm2_gen(precision='fast'|'balanced'|'precise'|'exact')
See the class docstring for the measured tradeoffs.

-----------------------------------------------------------------------------
Attribution (full detail and references in ATTRIBUTION.md):
  Original work, 2020-2024 -- Ted Teske (sole author):
    the BL/BL2 distributions, the lognorm_params mode/sigma reparametrization,
    the rv_continuous subclass design and full public API, the entropy
    adaptation, and all example usage.
  Optimization, 2026 -- Ted Teske with Claude (Anthropic):
    cached hoisting of the normalizing constant, vectorized composite
    Gauss-Legendre quadrature over the effective support, the cached PCHIP
    cumulative grid for ppf/rvs, the precision dial, and the
    _reference_unnorm / _unnorm_pdf split. (The fused form carries
    1/sqrt(2*pi*s1 s2), exactly as the original sqrt(lognorm_pdf *
    lognorm_pdf) did; an intermediate draft had 1/(2*pi*sqrt(s1 s2)), which
    was corrected before release. normed=False output matches the original.)
  Phase 2 math fixes, 2026 -- Ted Teske with Claude (Anthropic):
    a stable, scale-free solver for the lognorm_params quartic; mode-graded
    quadrature panels; moments by quadrature instead of a grid sum; argument
    guards. See docs/phase2_math_verification.md.
  Phase 3 scipy conformance, 2026 -- Ted Teske with Claude (Anthropic):
    the (c, s, loc, scale) parametrization and the from_bounds / to_bounds
    bridge; _pdf/_cdf/_sf/_ppf/_rvs hooks with per-(c, s) grouped caching;
    precision carried through freezing; entropy by quadrature (replacing
    the truncated-normal adaptation); moment(n) for every n; logpdf in log
    form; fit via scipy's MLE with a data-based start; the 1e-40 tail
    cutoff. See docs/phase3_api_conformance.md.
  Third-party helpers retain their inline citations (Stack Overflow, CC BY-SA).
-----------------------------------------------------------------------------
"""
import warnings
from collections import namedtuple
from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import erf, erfinv
from scipy.stats import rv_continuous

try:                                    # scipy >= 1.9; used by scipy.stats.fit
    from scipy.stats._distn_infrastructure import _ShapeInfo
except ImportError:                     # pragma: no cover
    _ShapeInfo = None


# ===========================================================================
# Helper functions -- from the original implementation (T. Teske). All are
# unchanged except lognorm_params, whose quartic is now solved by a stable
# root-finder (Phase 2); its derivation and references are kept as written.
# ===========================================================================

def lognorm_pdf(x, mu_N=0, sigma_N=1):
    """
    Log-normal (aka lognormal) probability density function (pdf).

    Parameters
    ----------
    x : {float, list or ndarray of floats}
        x-value(s)
    mu_N : float
        mean of underlying normal distribution
    sigma_N : float
        standard deviation of underlying normal distribution

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    # Creates an index of positive elements in x.
    x = np.atleast_1d(x)
    i = np.nonzero(x > 0)

    # Applies lognormal pdf to positive elements. All others evaluate to 0.
    y = np.zeros(len(x))
    y[i] = 1 / (x[i] * sigma_N * np.sqrt(2 * np.pi)) \
        * np.exp(-(np.log(x[i]) - mu_N)**2 / (2 * sigma_N**2))
    return y


def lognorm_cdf(x, mu_N=0, sigma_N=1):
    """
    Log-normal cumulative density function (cdf).

    Parameters
    ----------
    x : {float, list or ndarray of floats}
        x-value(s)
    mu_N : float
        mean of underlying normal distribution
    sigma_N : float
        standard deviation of underlying normal distribution

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    # Creates an index of positive elements in x.
    x = np.atleast_1d(x)
    i = np.nonzero(x > 0)

    # Applies lognormal cdf to positive elements. All others evaluate to 0.
    y = np.zeros(len(x))
    y[i] = 0.5 + 0.5 * erf((np.log(x[i]) - mu_N) / (np.sqrt(2) * sigma_N))
    return y


def lognorm_ppf(q, mu_N=0, sigma_N=1):
    """
    Log-normal percent-point function (ppf), also known as the quantile
    function, is the inverse of the cdf.

    Parameters
    ----------
    q : {float, list or ndarray of floats}
        quantile(s)
    mu_N : float
        mean of underlying normal distribution
    sigma_N : float
        standard deviation of underlying normal distribution

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    y = np.exp(mu_N + np.sqrt(2 * sigma_N**2) * erfinv(2 * q - 1))
    return y


def norm_pdf(x, mu_N=0, sigma_N=1):
    """
    Normal probability density function (pdf).

    Parameters
    ----------
    x : {float, list or ndarray of floats}
        x-value(s)
    mu_N : float
        mean
    sigma_N : float
        standard deviation

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Normal_distribution
    """
    y = 1 / (sigma_N * np.sqrt(2 * np.pi))
    y *= np.exp(-(x - mu_N)**2 / (2 * sigma_N**2))
    return y


def norm_cdf(x, mu_N=0, sigma_N=1):
    """
    Normal cumulative density function (cdf).

    Parameters
    ----------
    x : {float, list or ndarray of floats}
        x-value(s)
    mu_N : float
        mean
    sigma_N : float
        standard deviation

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Normal_distribution
    """
    y = 0.5 * (1 + erf((x - mu_N) / (sigma_N * np.sqrt(2))))
    return y


def lognorm_params(mode, sigma):
    """
    Given user-defined mode and sigma of a lognormal distribution,
    calculates the mu and sigma of its underlying normal distribution.

    Parameters
    ----------
    mode : float
        mode of lognormal distribution
    sigma : float
        standard deviation of lognormal distribution

    Returns
    -------
    mu_N : float
    sigma_N : float

    References
    ----------
    [1] Exact form adapted from the result of the query:
        solve[g^4 - g^3 - sigma^2/m^2] for g
        at https://www.wolframalpha.com. Accessed 21 Oct. 2020.
    [2] Numerical approximation from Warren Weckesser's answer to question at
        https://stackoverflow.com/questions/41464753. Accessed 21 Oct. 2020.

    Notes
    -----
    Derivation (unchanged): with g = exp(sigma_N^2), mode = exp(mu_N -
    sigma_N^2) gives mu_N = ln(mode * g), and Var = mode^2 g^3 (g - 1), so
    SD = sigma is the quartic g^4 - g^3 - (sigma/mode)^2 = 0. It has exactly
    one positive root, and that root exceeds 1.

    Numerics (Phase 2): the closed form [1] is algebraically exact but loses
    precision in float64 -- it cancels catastrophically for sigma/mode below
    ~1e-5, returns NaN below ~1e-9 or above ~1e6, and overflows m^8 near
    mode ~ 1e30. The same quartic is therefore solved here for
    t = sigma_N^2 = ln g in the equivalent form

        3 t + log(expm1(t)) = 2 log(sigma / mode),

    whose left side is strictly increasing in t, so the root is unique. Since
    log t <= LHS <= 4t + log t, the root is bracketed analytically and found
    with brentq in tau = log t. Depends only on sigma/mode (scale-free);
    ~1e-14 relative accuracy for sigma/mode from 1e-150 to 1e150.
    """
    # Converts parameters to floats in case exponents below too big for ints.
    # https://stackoverflow.com/questions/67097840
    def GetItem(inputobject):
        while type(inputobject) is np.ndarray:
            inputobject = inputobject[0]
        return inputobject

    mode, sigma = float(GetItem(mode)), float(GetItem(sigma))
    if not (mode > 0 and sigma > 0 and np.isfinite(mode)
            and np.isfinite(sigma)):
        raise ValueError(
            "lognorm_params requires finite mode > 0 and sigma > 0 "
            "(mode is measured from the bound); got mode=%r, sigma=%r"
            % (mode, sigma))

    log_r = np.log(sigma) - np.log(mode)      # log(sigma/mode); no overflow

    def _F(tau):                              # LHS - RHS, t = exp(tau)
        t = np.exp(tau)
        return 3.0 * t + np.log(np.expm1(t)) - 2.0 * log_r

    # Bracket from log t <= F(t) + 2 log r <= 4t + log t.
    if log_r < 0:
        tau_lo, tau_hi = 2.0 * log_r - 10.0, 2.0 * log_r
    else:
        tau_lo = -10.0
        tau_hi = np.log(min(np.exp(min(2.0 * log_r, 700.0)),
                            0.5 * log_r + 1.0))
    t = np.exp(brentq(_F, tau_lo, tau_hi, xtol=1e-15, rtol=1e-15,
                      maxiter=200))

    sigma_N = np.sqrt(t)                      # = sqrt(ln g)
    mu_N = np.log(mode) + t                   # = ln(mode * g)
    return mu_N, sigma_N


# ---------------------------------------------------------------------------
# Module-level helpers.
#
# These are deliberately module-level (not methods) so that lru_cache keys on
# the four shape parameters alone and does not hold a reference to `self`.
# All arguments must be plain Python floats -- ndarrays are unhashable.
# ---------------------------------------------------------------------------

# Gauss-Legendre nodes/weights are fixed per order; compute each order once.
@lru_cache(maxsize=8)
def _gauss_legendre(n):
    return np.polynomial.legendre.leggauss(n)


def _reference_unnorm(x, mu1, s1, mu2, s2, lower, u):
    """
    Unnormalized BL2 density -- THE DEFINITION, written exactly as Eq. 3:
    the geometric mean of the left lognormal and the reflected right lognormal.

        g(x) = sqrt( f_L(a; mu1, s1) * f_L(b; mu2, s2) )

    with a = x - lower and b = u - a, and g == 0 outside (0, u).

    This is the readable reference. It is slower than _unnorm_pdf (two full
    lognormal_pdf calls plus a sqrt) but it reads like the math, so a human can
    check it by eye. _unnorm_pdf is the algebraically-fused, faster form of the
    SAME quantity; the test suite pins the two equal to ~1e-12, so the fast
    path can never silently drift from the definition. Takes the same argument
    tuple as _unnorm_pdf so the two are directly comparable.
    """
    t = np.atleast_1d(np.asarray(x, dtype=float)) - lower
    y = np.zeros_like(t)
    i = (t > 0) & (t < u)
    if np.any(i):
        a = t[i]
        b = u - a
        left = lognorm_pdf(a, mu1, s1)
        right = lognorm_pdf(b, mu2, s2)
        y[i] = np.sqrt(left * right)
    return y


def _unnorm_pdf(x, mu1, s1, mu2, s2, lower, u):
    """
    Unnormalized BL2 density -- the algebraically-fused, fast form of
    _reference_unnorm (which see for the plain-language definition). Vectorized
    over x, no recursion, and the normalizing constant is NOT recomputed here
    (see _prep). Equal to _reference_unnorm to ~machine precision; the test
    suite enforces that so this optimized form cannot drift from Eq. 3.

        g(x) = 1/sqrt(2*pi*s1*s2) * 1/sqrt(a*b)
               * exp( -(ln a - mu1)^2 / (4 s1^2)
                      -(ln b - mu2)^2 / (4 s2^2) )

    where a = x - lower and b = u - a, and g == 0 outside the support.
    """
    t = np.atleast_1d(np.asarray(x, dtype=float)) - lower
    y = np.zeros_like(t)
    i = (t > 0) & (t < u)
    if np.any(i):
        a = t[i]
        b = u - a
        y[i] = (np.exp(-(np.log(a) - mu1) ** 2 / (4 * s1 ** 2)
                       - (np.log(b) - mu2) ** 2 / (4 * s2 ** 2))
                / (np.sqrt(2 * np.pi) * np.sqrt(s1 * s2) * np.sqrt(a * b)))
    return y


def _graded_edges(a, b, mode, lower, upper, k_max=16, n_even=4):
    """
    Panel edges for composite Gauss-Legendre quadrature over [a, b].

    The two lognormal factors vary on the length scales (mode - lower) and
    (upper - mode), so edges are graded geometrically -- half a decade per
    step, down to 10**(-k_max/2) of each scale -- toward the mode and toward
    each bound. When the mode sits close to a bound, this resolves a feature
    of width ~(mode - lower) that equal panels cannot see.

    The effective support [a, b] is also split into `n_even` equal panels on
    each side of the mode. That covers the opposite regime, a peak much
    narrower than (mode - lower) (sigma -> 0, or wide bounds), where the
    geometric edges stop short of the peak's own width. Measured benefit is
    modest (~4x) and appears only below ~1e-8 relative width, where float64
    resolution already limits accuracy; it is a cheap margin, not a
    requirement, and no test depends on it.

    An edge always sits AT the mode, so no panel straddles the peak.
    """
    m, r = mode - lower, upper - mode
    q = 10.0 ** (-np.arange(0, k_max + 1) / 2.0)
    e = np.concatenate([
        [a, b, mode],
        lower + m * q, mode - m * q, mode + r * q, upper - r * q,
        np.linspace(a, mode, n_even + 1), np.linspace(mode, b, n_even + 1),
    ])
    e = np.unique(np.clip(e, a, b))
    return e




def _log_unnorm_pdf(x, mu1, s1, mu2, s2, lower, u):
    """
    log of _unnorm_pdf, computed directly rather than as log(exp(...)), so it
    stays finite wherever x is inside the support -- including the far tails
    where _unnorm_pdf itself underflows to 0. -inf outside the support. Used
    by logpdf (and hence fit) and by the entropy. The test suite pins
    exp(_log_unnorm_pdf) to _unnorm_pdf.
    """
    t = np.atleast_1d(np.asarray(x, dtype=float)) - lower
    y = np.full_like(t, -np.inf)
    i = (t > 0) & (t < u)
    if np.any(i):
        la = np.log(t[i])
        lb = np.log(u - t[i])
        y[i] = (-(la - mu1) ** 2 / (4 * s1 ** 2)
                - (lb - mu2) ** 2 / (4 * s2 ** 2)
                - 0.5 * np.log(2 * np.pi * s1 * s2) - 0.5 * (la + lb))
    return y


# Density cutoff, relative to the peak, that defines the effective support.
# Phase 3 lowered it from 1e-14 to keep sf/logcdf accurate in the tails
# (see _effective_support).
_SUPPORT_TOL = 1e-40


_Table = namedtuple('_Table', 'args a b edges cum rcum nodes weights Z')
_Table.__doc__ = """Cached per-parameter-set quadrature table (see _panel_table)."""


@lru_cache(maxsize=64)
def _panel_table(mode, sigma, lower, upper, n_quad):
    """
    Hoisted, cached setup for one parameter set.

    Computes the underlying normal parameters for both lognormal factors, the
    effective support, the graded panel edges, and the integral of the
    unnormalized density over every panel -- exactly once.

    In the original implementation Z was recomputed inside every single pdf()
    call -- including at every quadrature node of every cdf() point -- even
    though it does not depend on x. That redundancy, not the choice of
    integrator, was the dominant cost.

    Since Phase 3 the class always calls this in standard coordinates
    (lower=0, upper=1, mode=c, sigma=s), so the cache is keyed on the two
    shape parameters only: shifted or rescaled copies of one distribution
    share a single table.

    Returns a _Table with fields
      args    : (mu1, s1, mu2, s2, lower, u), the args for _unnorm_pdf
      a, b    : effective support; g is negligible outside it
      edges   : panel edges over [a, b] (see _graded_edges)
      cum     : integral of g from a to each edge; cum[-1] == Z
      rcum    : integral of g from each edge to b (summed from the right, so
                the upper tail keeps its relative accuracy); rcum[-1] == 0
      nodes, weights : every GL node and its weight * g(node), used for the
                moments and the entropy (sum(weights) == Z)
      Z       : the normalizing constant

    The arrays are shared through the cache: callers must not modify them.
    """
    u = upper - lower
    m = mode - lower
    mu1, s1 = lognorm_params(m, sigma)
    mu2, s2 = lognorm_params(u - m, sigma)
    args = (float(mu1), float(s1), float(mu2), float(s2),
            float(lower), float(u))

    a, b = _effective_support(args, mode, lower, upper)
    edges = _graded_edges(a, b, mode, lower, upper)

    xn, wn = _gauss_legendre(n_quad)
    mid = (edges[1:] + edges[:-1]) / 2.0
    half = (edges[1:] - edges[:-1]) / 2.0
    nodes = mid[:, None] + half[:, None] * xn                  # (P, n)
    weights = (half[:, None] * wn) * _unnorm_pdf(
        nodes.ravel(), *args).reshape(nodes.shape)
    panel = weights.sum(axis=1)
    cum = np.concatenate(([0.0], np.cumsum(panel)))
    rcum = np.concatenate((np.cumsum(panel[::-1])[::-1], [0.0]))
    nodes, weights = nodes.ravel(), weights.ravel()
    for arr in (edges, cum, rcum, nodes, weights):
        arr.setflags(write=False)
    return _Table(args, float(a), float(b), edges, cum, rcum, nodes, weights,
                  float(cum[-1]))


def _gl_cdf(x, mode, sigma, lower, upper, n_quad):
    """
    Unnormalized integral of g from a to each x (x already clipped to [a, b]).

    Whole panels below x come from the cached table; only the partial panel
    [edge_k, x] is integrated fresh, with one n_quad-point Gauss-Legendre rule.
    The partial piece lies inside a single graded panel, so it is at least as
    well resolved as that panel.
    """
    T = _panel_table(mode, sigma, lower, upper, n_quad)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    k = np.clip(np.searchsorted(T.edges, x, side='right') - 1,
                0, len(T.edges) - 2)
    lo = T.edges[k]
    xn, wn = _gauss_legendre(n_quad)
    mid = (x + lo) / 2.0
    half = (x - lo) / 2.0
    pts = mid[:, None] + half[:, None] * xn
    vals = _unnorm_pdf(pts.ravel(), *T.args).reshape(pts.shape)
    return T.cum[k] + (vals * wn).sum(axis=1) * half


def _gl_sf(x, mode, sigma, lower, upper, n_quad):
    """
    Unnormalized integral of g from each x to b (x already clipped to [a, b]).

    The mirror image of _gl_cdf: whole panels above x come from the
    right-summed table rcum, and only [x, edge_{k+1}] is integrated fresh.
    Unlike 1 - cdf, this keeps full relative accuracy in the upper tail.
    """
    T = _panel_table(mode, sigma, lower, upper, n_quad)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    k = np.clip(np.searchsorted(T.edges, x, side='right') - 1,
                0, len(T.edges) - 2)
    hi = T.edges[k + 1]
    xn, wn = _gauss_legendre(n_quad)
    mid = (hi + x) / 2.0
    half = (hi - x) / 2.0
    pts = mid[:, None] + half[:, None] * xn
    vals = _unnorm_pdf(pts.ravel(), *T.args).reshape(pts.shape)
    return T.rcum[k + 1] + (vals * wn).sum(axis=1) * half


def _gl_moments(mode, sigma, lower, upper, n_quad):
    """
    Mean, variance, skewness and excess kurtosis by the same graded
    Gauss-Legendre rule the cdf uses -- population moments, no grid.

    Weights are renormalized on the same nodes (so the rule integrates the
    density to exactly 1), and the mean is accumulated as an offset from the
    mode, which keeps it accurate when |mode| >> sigma.
    """
    T = _panel_table(mode, sigma, lower, upper, n_quad)
    x, w = T.nodes, T.weights
    p = w / w.sum()
    mean = mode + np.sum(p * (x - mode))
    d = x - mean
    var = np.sum(p * d ** 2)
    skew = np.sum(p * d ** 3) / var ** 1.5
    kurt = np.sum(p * d ** 4) / var ** 2 - 3.0
    return float(mean), float(var), float(skew), float(kurt)


def _gl_raw_moment(n, mode, sigma, lower, upper, n_quad):
    """
    E[X**n] for any integer n >= 0, on the same nodes as _gl_moments.

    The class calls this in standard coordinates, where 0 < X < 1, so every
    term is in [0, 1] and the sum has no cancellation for any n. (scipy then
    maps it to loc/scale by the binomial expansion.)
    """
    T = _panel_table(mode, sigma, lower, upper, n_quad)
    p = T.weights / T.weights.sum()
    return float(np.sum(p * T.nodes ** int(n)))


def _gl_entropy(mode, sigma, lower, upper, n_quad):
    """
    Differential entropy -integral(f ln f), in nats, by the same graded
    Gauss-Legendre rule.

    With f = g / Z this is  ln Z - (1/Z) * integral(g ln g).  ln g comes from
    _log_unnorm_pdf, so it is exact where g itself has underflowed; nodes
    where g == 0 contribute 0, the limit of g ln g. Z is the sum of the same
    weights, so the rule is self-consistent.

    Replaces the truncated-normal approximation used up to Phase 2.
    """
    T = _panel_table(mode, sigma, lower, upper, n_quad)
    w = T.weights
    Z = w.sum()
    lg = _log_unnorm_pdf(T.nodes, *T.args)
    with np.errstate(invalid='ignore'):
        glng = np.where(w > 0, w * lg, 0.0)
    return float(np.log(Z) - glng.sum() / Z)


def _effective_support(args, mode, lower, upper, tol=None):
    """
    Locate the sub-interval of (lower, upper) carrying essentially all the mass.

    Uses bisection outward from the mode rather than a uniform scan, so it is
    scale-free: it works even when the peak is vanishingly narrow relative to
    the support (e.g. mode=1, sigma=0.062 on (0, 100), or mode=950, sigma=1 on
    (-1000, 1000)) -- cases where a uniform scan can miss the peak entirely and
    where a fixed-order rule spread over the full support badly under-resolves
    it. The density is unimodal with its maximum at `mode`, so the level set
    g(x) = tol * g(mode) is bracketed by bisection on each side.

    tol defaults to _SUPPORT_TOL = 1e-40 (Phase 3; was 1e-14). The cutoff,
    not the quadrature, sets how deep the tails stay accurate: at 1e-14, sf
    and the lower-tail cdf lost relative accuracy below ~1e-12; at 1e-40 they
    hold ~1e-12 relative down to ~1e-32. Body accuracy, the moments and the
    timings are unchanged (a few more graded panels, built once per table).
    """
    tol = _SUPPORT_TOL if tol is None else tol
    y0 = float(_unnorm_pdf(mode, *args)[0])
    if not np.isfinite(y0) or y0 <= 0:
        return lower, upper
    thresh = tol * y0

    def _edge(lo, hi):
        # g(lo) < thresh <= g(hi); returns x where g crosses thresh
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(_unnorm_pdf(mid, *args)[0]) < thresh:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    a = _edge(lower, mode)
    b = _edge(upper, mode)
    if not (b > a):
        return lower, upper
    return float(a), float(b)


@lru_cache(maxsize=32)
def _cdf_grid(mode, sigma, lower, upper, n_quad, n_grid):
    """
    Cached CDF/PPF grid.

    Places n_grid uniform points over the effective support plus n_grid more
    spread across the graded panels (dense wherever the density changes
    fastest), evaluates the cdf there with the panel-table quadrature, and
    wraps the result in monotone (PCHIP) splines. Both cdf and ppf then come
    from one cached pass, which is what makes ppf/rvs fast.

    Returns (cdf_spline, ppf_spline, a, b) where (a, b) is the effective
    support; outside it the cdf is 0 or 1 by construction.
    """
    T = _panel_table(mode, sigma, lower, upper, n_quad)
    a, b, edges, Z = T.a, T.b, T.edges, T.Z
    if not (Z > 0 and np.isfinite(Z)):
        raise ValueError("Density integrates to zero; check parameters.")

    # Union of two point sets, ~2 * n_grid points in all:
    #  - n_grid uniform points over the support (as before) carry the body of
    #    the distribution, where ppf inversion error is set by spacing;
    #  - n_grid points shared equally among the graded panels resolve the
    #    density wherever it changes on scales the uniform grid cannot see
    #    (mode near a bound). Either set alone fails one of those regimes.
    per = max(2, int(np.ceil(n_grid / (len(edges) - 1))))
    refine = [np.linspace(lo, hi, per + 1)
              for lo, hi in zip(edges[:-1], edges[1:])]
    xs = np.unique(np.concatenate([np.linspace(a, b, n_grid)] + refine))
    F = _gl_cdf(xs, mode, sigma, lower, upper, n_quad) / Z
    F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))   # guard fp wiggle
    F[0], F[-1] = 0.0, 1.0

    cdf_spline = PchipInterpolator(xs, F, extrapolate=False)

    # PPF needs strictly increasing abscissae. Dropping merely-equal values is
    # not enough: increments of order 1e-300 survive `> 0` but produce
    # non-finite slopes. Require a real increment.
    keep = np.concatenate(([True], np.diff(F) > 1e-12))
    keep[-1] = True
    Fk, xk = F[keep], xs[keep]
    # Guard against the endpoint being merged into its neighbour.
    if Fk.size < 2 or not np.all(np.diff(Fk) > 0):
        Fk, uniq = np.unique(F, return_index=True)
        xk = xs[uniq]
    ppf_spline = PchipInterpolator(Fk, xk, extrapolate=False)
    return cdf_spline, ppf_spline, float(a), float(b)


# ---------------------------------------------------------------------------
# Array-valued shapes (Phase 3).
#
# scipy hands _pdf/_cdf/_ppf/... arrays of shape parameters, broadcast against
# x. The caches above need plain floats, so every method groups its inputs by
# unique (c, s) pair and evaluates each group through the cached scalar
# tables. The overwhelmingly common case -- one parameter set, broadcast --
# takes a fast path with no np.unique.
# ---------------------------------------------------------------------------

def _param_groups(c, s):
    """[(c, s, index)] for each distinct (c, s) pair in the flat arrays."""
    c = np.ravel(np.asarray(c, dtype=float))
    s = np.ravel(np.asarray(s, dtype=float))
    if c.size == 0:
        return []
    if np.all(c == c[0]) and np.all(s == s[0]):
        return [(float(c[0]), float(s[0]), slice(None))]
    uniq, inv = np.unique(np.stack([c, s], axis=1), axis=0,
                          return_inverse=True)
    inv = np.ravel(inv)
    return [(float(cc), float(ss), inv == k)
            for k, (cc, ss) in enumerate(uniq)]


def _by_params(fn, x, c, s):
    """Apply fn(x_group, c, s) -> array per distinct (c, s); keep x's shape."""
    x, c, s = np.broadcast_arrays(np.asarray(x, dtype=float),
                                  np.asarray(c, dtype=float),
                                  np.asarray(s, dtype=float))
    xf = np.ravel(x)
    out = np.empty(xf.shape)
    for cc, ss, idx in _param_groups(c, s):
        out[idx] = fn(xf[idx], cc, ss)
    return out.reshape(x.shape)


# ===========================================================================
# PRECISION DIAL -- the single speed <-> precision control.
#
# One ordered knob, passed once at construction:
#
#       fast  ──  balanced  ──  precise  ──  exact
#     (fastest,                          (most digits,
#      fewest digits)                     slowest)
#
# Each position resolves to concrete internal settings. This dict is the
# SINGLE SOURCE OF TRUTH -- to retune or add a position, edit here and
# nowhere else. Everything downstream reads n_quad / n_grid / ppf off it.
#
#   n_quad : Gauss-Legendre order per graded panel, for cdf and moments
#            (since Phase 2, all positions reach ~1e-13; higher = slower)
#   n_grid : points in the cached grid backing ppf/rvs (higher -> finer
#            quantile inversion)
#   ppf    : 'grid'  -> invert the cached spline (fast, same error as n_grid)
#            'solve' -> root-find cdf(x)=q with brentq (~machine precision,
#                       but one solve per quantile, so much slower)
#
# NOT on the dial, on purpose (keeps the knob honest -- one thing, not five):
#   - cdf panel edges are not a setting: they are graded geometrically toward
#     the mode and both bounds (see _graded_edges), which is what keeps the
#     cdf accurate when the mode sits near a bound.
#   - rvs ALWAYS uses the grid, at every position including 'exact': grid
#     inversion error is far below Monte-Carlo sampling noise, and a root-find
#     per sample would be needlessly slow. The dial sets cdf/ppf accuracy; it
#     does not change sampling quality in any measurable way.
#
# The dial survives freezing: scipy rebuilds the instance from
# _updated_ctor_param(), which carries `precision`.
# ===========================================================================
PRECISION_DIAL = ('fast', 'balanced', 'precise', 'exact')

_PRECISION_SETTINGS = {
    'fast':     {'n_quad':  24, 'n_grid':  1001, 'ppf': 'grid'},
    'balanced': {'n_quad':  64, 'n_grid':  4001, 'ppf': 'grid'},
    'precise':  {'n_quad':  96, 'n_grid': 20001, 'ppf': 'grid'},
    'exact':    {'n_quad': 128, 'n_grid': 20001, 'ppf': 'solve'},
}

# Below this ratio of sigma to the coordinate scale, float64 resolution of
# (x - lower) alone limits cdf accuracy to worse than ~1e-8.
_RESOLUTION_WARN = 1e-8


class bounded_lognorm2_gen(rv_continuous):
    r"""
    A bounded lognormal (BL2) continuous random variable.

    The probability density function is the normalized geometric mean of
        1. a lognormal pdf located at the lower bound, and
        2. a reflected lognormal pdf located at the upper bound.

    Parametrization (Phase 3)
    -------------------------
    BL2 is an exact location-scale family, so it follows the scipy.stats
    convention: a standard form on [0, 1] with two shape parameters, moved
    and stretched by loc and scale.

        loc   = lower
        scale = upper - lower
        c     = (mode - lower) / (upper - lower)     relative mode, 0 < c < 1
        s     = sigma / (upper - lower)              relative sigma, s > 0

    c is the same convention as scipy.stats.triang. sigma is the standard
    deviation of each (untruncated) lognormal factor, not of BL2 itself.

    Every rv_continuous method works: pdf, logpdf, cdf, sf, ppf, isf, rvs,
    stats, moment(n) for any n, entropy, median, interval, expect, fit, and
    freezing. If you think in bounds, use the bridge:

        rv = bl2.from_bounds(mode=5, sigma=10, lower=-5, upper=20)  # frozen
        rv.pdf(x); rv.cdf(x); rv.mean(); rv.entropy()
        c, s, loc, scale = bl2.fit(data)
        mode, sigma, lower, upper = bl2.to_bounds(c, s, loc, scale)

    from_bounds accepts None for an unbounded side (see its docstring).

    MIGRATION from the pre-Phase-3 API: positional calls such as
    bl2.pdf(x, mode, sigma, lower, upper) are now read as (x, c, s, loc,
    scale). With mode > 1 that is an invalid c, and scipy returns NaN rather
    than raising. Wrap old parameter sets in from_bounds().

    The speed <-> precision dial (set once, at construction)
    -------------------------------------------------------
    Pass precision='fast' | 'balanced' | 'precise' | 'exact' (default
    'balanced'). Left is fastest with the fewest correct digits; right is the
    most digits and the slowest. Pick by task: 'fast' for plotting and
    exploration, 'balanced' for ordinary work and fitting, 'precise' when you
    need ~10 good digits, 'exact' only when you need the cdf/ppf essentially to
    machine precision. Frozen distributions keep their instance's setting.

    Measured tradeoffs (worst case over the test suite; timings for 1000
    points at mode=5, sigma=10, bounds=(-5, 20); your machine will differ).
    Since the Phase 2 graded panels, cdf accuracy is at the reference
    quadrature's own floor (~1e-13) at EVERY position; the dial now mainly
    trades cdf speed and ppf accuracy.

      fast
        - cdf accuracy  ~1e-13
        - ppf accuracy  ~2e-6
        - cdf speed     ~0.4 ms / 1000 pts
      balanced  (default)
        - cdf accuracy  ~1e-13
        - ppf accuracy  ~1e-7
        - cdf speed     ~0.9 ms / 1000 pts
      precise
        - cdf accuracy  ~1e-13
        - ppf accuracy  ~4e-9 (grid built once per parameter set: ~80 ms)
        - cdf speed     ~1.3 ms / 1000 pts
      exact
        - cdf accuracy  ~1e-13
        - ppf accuracy  ~1e-13 (brentq root-find, not the grid)
        - cdf speed     ~1.9 ms / 1000 pts
        - ppf speed     ~0.3 ms PER QUANTILE -- i.e. ~300x slower than the
          other positions. Use 'exact' for one-off high-accuracy quantiles,
          not for inverting thousands at once.

    ppf accuracy above is the cdf(ppf(q)) round-trip error. rvs uses the grid
    at every position and is unaffected by the dial. Since Phase 3 these go
    through scipy's argument handling, which adds roughly 0.1-0.6 ms per call
    of 1000 points on top of the timings above.

    Caching
    -------
    Shape parameters, the normalizing constant Z and the quadrature panels
    are computed once per (c, s) and cached; loc and scale are not part of
    the key, so shifted or rescaled copies of one distribution share them.
    Array-valued shapes are grouped by distinct (c, s) pair, one cached table
    per group. Many distinct pairs in one call cost one table build each
    (~1 ms), so e.g. 10_000 different shapes in a single call are slow.
    """

    def __init__(self, precision='balanced', **kwargs):
        """
        Parameters
        ----------
        precision : {'fast', 'balanced', 'precise', 'exact'}
            Position on the speed<->precision dial (left to right). Default
            'balanced'. See PRECISION_DIAL / _PRECISION_SETTINGS above and the
            measured tradeoffs in the class docstring. Set it once here; it
            then governs cdf and ppf for the life of the instance (and of
            anything frozen from it). To mix precisions in one script, make
            more than one instance:
                bl2_fast  = bounded_lognorm2_gen(precision='fast')
                bl2_exact = bounded_lognorm2_gen(precision='exact')
        **kwargs
            Passed to rv_continuous (name, seed, ...). The support (a=0, b=1)
            and the shape names (c, s) are fixed by default.
        """
        if precision not in _PRECISION_SETTINGS:
            raise ValueError(
                "precision must be one of %s, got %r"
                % (PRECISION_DIAL, precision))
        kwargs.setdefault('a', 0.0)
        kwargs.setdefault('b', 1.0)
        kwargs.setdefault('name', 'bounded_lognorm2')
        kwargs.setdefault('shapes', 'c, s')
        super().__init__(**kwargs)
        self.precision = precision
        cfg = _PRECISION_SETTINGS[precision]
        self.n_quad = cfg['n_quad']
        self.n_grid = cfg['n_grid']
        self._ppf_method = cfg['ppf']

    def _updated_ctor_param(self):
        # scipy freezes by re-instantiating from these; without `precision`
        # every frozen distribution would silently fall back to 'balanced'.
        d = super()._updated_ctor_param()
        d['precision'] = self.precision
        return d

    # -- scipy hooks: parameter validation and description -----------------

    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, 1), (False, False)),
                _ShapeInfo("s", False, (0, np.inf), (False, False))]

    def _argcheck(self, c, s):
        """
        0 < c < 1 and s > 0, both finite. Elementwise.

        Also warns when s < 1e-8: in standard coordinates float64 resolves
        1 - x only to ~1e-16, so accuracy is limited to roughly 1e-16 / s.
        (A large |loc| relative to sigma has the same effect in x - loc;
        only from_bounds can see that, and it warns for it.)
        """
        c, s = np.broadcast_arrays(np.asarray(c, dtype=float),
                                   np.asarray(s, dtype=float))
        cond = (np.isfinite(c) & np.isfinite(s)
                & (c > 0) & (c < 1) & (s > 0))
        if np.any(cond & (s < _RESOLUTION_WARN)):
            smin = float(np.min(s[cond]))
            warnings.warn(
                "s = %.1e: sigma is below 1e-8 of (upper - lower), so float64 "
                "resolution limits cdf/ppf accuracy to roughly %.0e."
                % (smin, 2.2e-16 / smin), RuntimeWarning, stacklevel=3)
        return cond

    def _fitstart(self, data, args=None):
        """
        Starting point for fit(): bounds just outside the data, mode at the
        median, sigma at the sample standard deviation. (scipy's generic
        start sets every shape to 1, which is an invalid c.)
        """
        data = np.asarray(data, dtype=float)
        lo, hi = float(np.min(data)), float(np.max(data))
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        loc = lo - 0.05 * span
        scale = 1.1 * span
        c = float(np.clip((np.median(data) - loc) / scale, 0.05, 0.95))
        s = float(max(np.std(data) / scale, 1e-3))
        return c, s, loc, scale

    # -- per-parameter-set cached work --------------------------------------

    def _table(self, c, s):
        return _panel_table(c, s, 0.0, 1.0, self.n_quad)

    def _grid(self, c, s):
        return _cdf_grid(c, s, 0.0, 1.0, self.n_quad, self.n_grid)

    # -- scipy hooks: the distribution itself, in standard coordinates -----
    #
    # scipy calls these only for x inside (0, 1) (q inside (0, 1) for _ppf)
    # and for parameters that passed _argcheck; it handles loc/scale, the
    # support boundaries and invalid arguments itself.

    def _pdf(self, x, c, s):
        def f(x, c, s):
            T = self._table(c, s)
            return _unnorm_pdf(x, *T.args) / T.Z
        return _by_params(f, x, c, s)

    def _logpdf(self, x, c, s):
        def f(x, c, s):
            T = self._table(c, s)
            return _log_unnorm_pdf(x, *T.args) - np.log(T.Z)
        return _by_params(f, x, c, s)

    def _cdf(self, x, c, s):
        """
        Whole panels come from the cached graded-panel table; each x adds one
        vectorized Gauss-Legendre rule over its partial panel. Integration is
        confined to the effective support [a, b]: below a the mass is
        negligible, above b the cdf has saturated.
        """
        def f(x, c, s):
            T = self._table(c, s)
            y = _gl_cdf(np.clip(x, T.a, T.b), c, s, 0.0, 1.0,
                        self.n_quad) / T.Z
            y = np.where(x <= T.a, 0.0, np.where(x >= T.b, 1.0, y))
            return np.clip(y, 0.0, 1.0)
        return _by_params(f, x, c, s)

    def _sf(self, x, c, s):
        """1 - cdf, integrated from the right so upper tails keep their
        relative accuracy (1 - cdf would bottom out near 1e-16)."""
        def f(x, c, s):
            T = self._table(c, s)
            y = _gl_sf(np.clip(x, T.a, T.b), c, s, 0.0, 1.0,
                       self.n_quad) / T.Z
            y = np.where(x <= T.a, 1.0, np.where(x >= T.b, 0.0, y))
            return np.clip(y, 0.0, 1.0)
        return _by_params(f, x, c, s)

    def _ppf(self, q, c, s):
        """
        At dial positions 'fast'/'balanced'/'precise' this inverts the cached
        cumulative spline (fast; error set by n_grid). At 'exact' it instead
        root-finds cdf(x) = q with brentq, to ~machine precision, at the cost
        of one solve per quantile.
        """
        if self._ppf_method == 'solve':
            return _by_params(self._ppf_solve, q, c, s)
        return _by_params(self._ppf_grid, q, c, s)

    def _ppf_grid(self, q, c, s):
        _, ppf_spline, a, b = self._grid(c, s)
        y = ppf_spline(np.clip(q, 0.0, 1.0))
        # PCHIP returns nan outside its knots; pin those to the support.
        y = np.where(np.isnan(y), np.where(q >= 0.5, b, a), y)
        return np.clip(y, 0.0, 1.0)

    def _ppf_solve(self, q, c, s):
        T = self._table(c, s)
        n_quad = self.n_quad

        def _f(x, target):
            return float(_gl_cdf([x], c, s, 0.0, 1.0, n_quad)[0] / T.Z
                         - target)

        out = np.empty(np.shape(q))
        for i, qi in enumerate(np.ravel(q)):
            try:
                # xtol scales with s so a very narrow peak is still resolved;
                # rtol is brentq's floor (4 * machine eps).
                out.flat[i] = brentq(_f, T.a, T.b, args=(qi,),
                                     xtol=1e-14 * min(1.0, s),
                                     rtol=4 * np.finfo(float).eps,
                                     maxiter=200)
            except ValueError:
                # q within rounding of 0 or 1: no sign change on [a, b].
                out.flat[i] = T.b if qi >= 0.5 else T.a
        return out

    def _rvs(self, c, s, size=None, random_state=None):
        """
        Inverse-transform sampling. Always inverts the cached grid, regardless
        of the precision dial -- including at 'exact'. Grid inversion error is
        far below Monte-Carlo sampling noise for any realistic sample size,
        and a brentq solve per sample would be needlessly slow.
        """
        u = random_state.uniform(size=size)
        return _by_params(self._ppf_grid, u, c, s)

    # -- moments and entropy -------------------------------------------------

    def _stats(self, c, s, moments='mv'):
        """
        Population mean, variance, skewness and excess kurtosis of the
        standard form, integrated with the cdf's graded Gauss-Legendre rule
        (see _gl_moments). scipy maps them to loc/scale.
        """
        c, s = np.broadcast_arrays(np.asarray(c, dtype=float),
                                   np.asarray(s, dtype=float))
        out = [np.empty(c.size) for _ in range(4)]
        for cc, ss, idx in _param_groups(c, s):
            vals = _gl_moments(cc, ss, 0.0, 1.0, self.n_quad)
            for o, v in zip(out, vals):
                o[idx] = v
        return tuple(o.reshape(c.shape) for o in out)

    def _munp(self, n, c, s):
        """
        n-th raw moment E[Y**n] of the standard form, for ANY integer n >= 0
        (previously None for n >= 5). Y is in (0, 1), so the quadrature sum
        has no cancellation; scipy's moment() maps it to loc/scale.
        """
        c, s = np.broadcast_arrays(np.asarray(c, dtype=float),
                                   np.asarray(s, dtype=float))
        out = np.empty(c.size)
        for cc, ss, idx in _param_groups(c, s):
            out[idx] = _gl_raw_moment(n, cc, ss, 0.0, 1.0, self.n_quad)
        return out.reshape(c.shape)

    def _entropy(self, c, s):
        """
        Differential entropy of the standard form, in nats, by quadrature of
        -f ln f (see _gl_entropy). scipy adds ln(scale). This replaces the
        truncated-normal approximation used through Phase 2, which was not
        BL2's own entropy.
        """
        return _gl_entropy(float(c), float(s), 0.0, 1.0, self.n_quad)

    # -- extras beyond the rv_continuous API ---------------------------------

    def normalizing_constant(self, c, s):
        """
        Z, the integral of the unnormalized geometric-mean density. It is the
        Bhattacharyya coefficient of the two lognormal factors, so 0 < Z <= 1,
        and it is unchanged by loc and scale. pdf * Z gives the unnormalized
        density in x units (what `normed=False` returned before Phase 3).
        """
        c, s = np.broadcast_arrays(np.asarray(c, dtype=float),
                                   np.asarray(s, dtype=float))
        out = np.empty(c.size)
        for cc, ss, idx in _param_groups(c, s):
            out[idx] = self._table(cc, ss).Z
        return out.reshape(c.shape)[()]

    def density_reference(self, x, c, s, loc=0.0, scale=1.0):
        """
        The pdf written as its definition (Eq. 3), for reading and checking --
        the normalized geometric mean of the two lognormal factors.

        pdf() returns the same values via an algebraically-fused fast path;
        this returns them via the plain definition. They agree to ~machine
        precision (the test suite enforces it). Use pdf() in practice; use this
        when you want to confirm by eye what pdf() is computing, or as the
        ground-truth density in a derivation. Scalar c, s only.
        """
        c, s = float(c), float(s)
        T = self._table(c, s)
        y = (np.asarray(x, dtype=float) - loc) / scale
        return _reference_unnorm(y, *T.args).reshape(np.shape(y)) \
            / T.Z / scale

    def from_bounds(self, mode, sigma, lower=None, upper=None):
        """
        Frozen distribution from the bounds parametrization.

        Parameters
        ----------
        mode : float
            x at the point of maximum density
        sigma : float
            spread around the mode (SD of each lognormal factor), > 0
        lower, upper : float or None
            bounds of the support. None means unbounded on that side and is
            replaced by mode -/+ 1000 sigma, where BL2 is indistinguishable
            from the normal side. Infinite bounds are rejected; pass None.

        Returns
        -------
        A frozen rv with c = (mode - lower)/(upper - lower),
        s = sigma/(upper - lower), loc = lower, scale = upper - lower, at this
        instance's precision.

        Unlike the native API (which returns NaN for invalid parameters, as
        scipy does), this raises ValueError, so a bad parameter set fails
        loudly. Accepts arrays (broadcast elementwise).
        """
        c, s, loc, scale = self.bounds_to_shapes(mode, sigma, lower, upper)
        return self(c, s, loc=loc, scale=scale)

    @staticmethod
    def bounds_to_shapes(mode, sigma, lower=None, upper=None):
        """
        (mode, sigma, lower, upper) -> (c, s, loc, scale), validated.
        None bounds as in from_bounds. Useful with fit, e.g. to fix known
        bounds: bl2.fit(data, floc=loc, fscale=scale).
        """
        normal_approx_factor = 1e+3
        mode = np.asarray(mode, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        lower = (mode - normal_approx_factor * sigma if lower is None
                 else np.asarray(lower, dtype=float))
        upper = (mode + normal_approx_factor * sigma if upper is None
                 else np.asarray(upper, dtype=float))
        if not np.all(np.isfinite(mode) & np.isfinite(sigma)
                      & np.isfinite(lower) & np.isfinite(upper)):
            raise ValueError(
                "mode, sigma, lower and upper must be finite; use None for "
                "an unbounded side")
        if not np.all((lower < mode) & (mode < upper) & (sigma > 0)):
            raise ValueError("Requires (lower < mode < upper) and (sigma > 0)")
        scale = upper - lower
        c = (mode - lower) / scale
        s = sigma / scale
        # Large |bounds| relative to sigma limit the resolution of x - loc.
        # (A small s on its own is reported by _argcheck instead.)
        bound = np.maximum(np.abs(lower), np.abs(upper))
        hit = (sigma < _RESOLUTION_WARN * bound) & ~(s < _RESOLUTION_WARN)
        if np.any(hit):
            r = float(np.min((sigma / bound)[hit]))
            warnings.warn(
                "sigma is %.1e of the coordinate scale (max |bound|); "
                "float64 resolution of x - lower limits cdf/ppf accuracy to "
                "roughly %.0e. Shift the problem nearer the origin if you "
                "need more." % (r, 2.2e-16 / r), RuntimeWarning, stacklevel=3)
        return c[()], s[()], lower[()], scale[()]

    @staticmethod
    def to_bounds(c, s, loc=0.0, scale=1.0):
        """(c, s, loc, scale) -> (mode, sigma, lower, upper). Inverse of
        bounds_to_shapes; e.g. bl2.to_bounds(*bl2.fit(data))."""
        c, s, loc, scale = (np.asarray(v, dtype=float)
                            for v in (c, s, loc, scale))
        return ((loc + c * scale)[()], (s * scale)[()], loc[()],
                (loc + scale)[()])


bl2 = bounded_lognorm2_gen()
