"""
bounded_lognorm2 (BL2) -- bounded lognormal distribution, geometric-mean form.

A scipy.stats.rv_continuous subclass parameterized by mode, sigma, and
lower/upper bounds. The pdf is the normalized geometric mean of a lognormal at
the lower bound and a reflected lognormal at the upper bound (Eq. 3).

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
  Third-party helpers retain their inline citations (Stack Overflow, CC BY-SA).
-----------------------------------------------------------------------------
"""
import warnings
from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, curve_fit, minimize
from scipy.special import erf, erfinv
from scipy.stats import rv_continuous


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

    Returns
    -------
    args  : tuple   -> (mu1, s1, mu2, s2, lower, u), the args for _unnorm_pdf
    a, b  : float   -> effective support; g is negligible outside it
    edges : ndarray -> panel edges over [a, b] (see _graded_edges)
    cum   : ndarray -> integral of g from a to each edge; cum[-1] == Z
    nodes, weights : ndarray -> every GL node and its weight * g(node), used
                      for the moments (sum(weights) == Z)

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
    cum = np.concatenate(([0.0], np.cumsum(weights.sum(axis=1))))
    for arr in (edges, cum, nodes, weights):
        arr.setflags(write=False)
    return args, float(a), float(b), edges, cum, nodes.ravel(), weights.ravel()


def _prep(mode, sigma, lower, upper, n_quad):
    """Shape args, effective support and Z: (args, a, b, Z)."""
    args, a, b, _e, cum, _n, _w = _panel_table(mode, sigma, lower, upper,
                                               n_quad)
    return args, a, b, float(cum[-1])


def _gl_cdf(x, mode, sigma, lower, upper, n_quad):
    """
    Unnormalized integral of g from a to each x (x already clipped to [a, b]).

    Whole panels below x come from the cached table; only the partial panel
    [edge_k, x] is integrated fresh, with one n_quad-point Gauss-Legendre rule.
    The partial piece lies inside a single graded panel, so it is at least as
    well resolved as that panel.
    """
    args, _a, _b, edges, cum, _n, _w = _panel_table(mode, sigma, lower, upper,
                                                    n_quad)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    k = np.clip(np.searchsorted(edges, x, side='right') - 1,
                0, len(edges) - 2)
    lo = edges[k]
    xn, wn = _gauss_legendre(n_quad)
    mid = (x + lo) / 2.0
    half = (x - lo) / 2.0
    pts = mid[:, None] + half[:, None] * xn
    vals = _unnorm_pdf(pts.ravel(), *args).reshape(pts.shape)
    return cum[k] + (vals * wn).sum(axis=1) * half


def _gl_moments(mode, sigma, lower, upper, n_quad):
    """
    Mean, variance, skewness and excess kurtosis by the same graded
    Gauss-Legendre rule the cdf uses -- population moments, no grid.

    Weights are renormalized on the same nodes (so the rule integrates the
    density to exactly 1), and the mean is accumulated as an offset from the
    mode, which keeps it accurate when |mode| >> sigma.
    """
    _args, _a, _b, _e, _cum, x, w = _panel_table(mode, sigma, lower, upper,
                                                 n_quad)
    p = w / w.sum()
    mean = mode + np.sum(p * (x - mode))
    d = x - mean
    var = np.sum(p * d ** 2)
    skew = np.sum(p * d ** 3) / var ** 1.5
    kurt = np.sum(p * d ** 4) / var ** 2 - 3.0
    return float(mean), float(var), float(skew), float(kurt)


def _effective_support(args, mode, lower, upper, tol=1e-14):
    """
    Locate the sub-interval of (lower, upper) carrying essentially all the mass.

    Uses bisection outward from the mode rather than a uniform scan, so it is
    scale-free: it works even when the peak is vanishingly narrow relative to
    the support (e.g. mode=1, sigma=0.062 on (0, 100), or mode=950, sigma=1 on
    (-1000, 1000)) -- cases where a uniform scan can miss the peak entirely and
    where a fixed-order rule spread over the full support badly under-resolves
    it. The density is unimodal with its maximum at `mode`, so the level set
    g(x) = tol * g(mode) is bracketed by bisection on each side.
    """
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
    fastest),
    evaluates the cdf there with the panel-table quadrature, and wraps the
    result in monotone (PCHIP) splines. Both cdf and ppf then come from one cached pass, which is what
    makes ppf/rvs fast.

    Returns (cdf_spline, ppf_spline, a, b) where (a, b) is the effective
    support; outside it the cdf is 0 or 1 by construction.
    """
    _args, a, b, edges, cum, _n, _w = _panel_table(mode, sigma, lower, upper,
                                                   n_quad)
    Z = cum[-1]
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
# ===========================================================================
PRECISION_DIAL = ('fast', 'balanced', 'precise', 'exact')

_PRECISION_SETTINGS = {
    'fast':     {'n_quad':  24, 'n_grid':  1001, 'ppf': 'grid'},
    'balanced': {'n_quad':  64, 'n_grid':  4001, 'ppf': 'grid'},
    'precise':  {'n_quad':  96, 'n_grid': 20001, 'ppf': 'grid'},
    'exact':    {'n_quad': 128, 'n_grid': 20001, 'ppf': 'solve'},
}

class bounded_lognorm2_gen(rv_continuous):
    """
    Subclass of scipy.stats.rv_continuous. Domain-bounded lognormal continuous
    probability distribution, fully described by four parameters: mode, sigma,
    lower and upper bounds.

    The probability density function is the geometric mean of
        1. a lognormal pdf located at the lower bound, and
        2. a reflected lognormal pdf located at the upper bound,
    scaled so that its integral equals 1.

    The speed <-> precision dial (set once, at construction)
    -------------------------------------------------------
    Pass precision='fast' | 'balanced' | 'precise' | 'exact' (default
    'balanced'). Left is fastest with the fewest correct digits; right is the
    most digits and the slowest. Pick by task: 'fast' for plotting and
    exploration, 'balanced' for ordinary work and fitting, 'precise' when you
    need ~10 good digits, 'exact' only when you need the cdf/ppf essentially to
    machine precision.

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
    at every position (see rvs docstring) and is unaffected by the dial.

    Why it is fast at all (vs. the original implementation)
    ------------------------------------------------------
    * Shape parameters and the normalizing constant Z are computed once per
      parameter set and cached, not recomputed at every pdf call.
    * cdf uses a single vectorized Gauss-Legendre evaluation across all
      x-values, and integrates only over the effective support (the sub-
      interval actually carrying the mass), not the full domain.
    * ppf/rvs/median read from a cached cumulative grid instead of driving a
      fresh quadrature per point.
    """

    def __init__(self, name='bounded_lognorm2',
                 shapes='mode, sigma, lower, upper', n=1001,
                 precision='balanced'):
        """
        Parameters
        ----------
        name : string
            name of instance
        shapes : string
            shape parameters, separated by commas
        n : int
            formerly the grid resolution for stats()/entropy(); retained for
            backward compatibility but no longer used -- moments are now
            integrated with the cdf's quadrature rule (Phase 2)
        precision : {'fast', 'balanced', 'precise', 'exact'}
            Position on the speed<->precision dial (left to right). Default
            'balanced'. See PRECISION_DIAL / _PRECISION_SETTINGS above and the
            measured tradeoffs in the class docstring. Set it once here; it
            then governs cdf and ppf for the life of the instance. To mix
            precisions in one script, make more than one instance:
                bl2_fast  = bounded_lognorm2_gen(precision='fast')
                bl2_exact = bounded_lognorm2_gen(precision='exact')
        """
        super().__init__()
        self.name = name
        self.shapes = shapes
        self.n = n
        if precision not in _PRECISION_SETTINGS:
            raise ValueError(
                "precision must be one of %s, got %r"
                % (PRECISION_DIAL, precision))
        self.precision = precision
        cfg = _PRECISION_SETTINGS[precision]
        self.n_quad = cfg['n_quad']
        self.n_grid = cfg['n_grid']
        self._ppf_method = cfg['ppf']
        return None

    def _argcheck(self, mode, sigma, lower=None, upper=None):
        """
        Overrides the default _argcheck (which demands positive shapes) so
        that negative locations are allowed, while still encoding the
        distribution's real constraints: finite mode, finite sigma > 0, and
        lower < mode < upper for each bound that is given (None = unbounded
        side, see _argcheck_final). Elementwise for array arguments.
        """
        mode = np.asarray(mode, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        cond = np.isfinite(mode) & np.isfinite(sigma) & (sigma > 0)
        if lower is not None:
            lower = np.asarray(lower, dtype=float)
            cond = cond & np.isfinite(lower) & (lower < mode)
        if upper is not None:
            upper = np.asarray(upper, dtype=float)
            cond = cond & np.isfinite(upper) & (mode < upper)
        return cond

    # Below this ratio of sigma to the coordinate scale, float64 resolution of
    # (x - lower) alone limits cdf accuracy to worse than ~1e-8.
    _RESOLUTION_WARN = 1e-8

    def _argcheck_final(self, x, mode, sigma, lower, upper,
                        ignore_checks=False):
        """
        Further compatibility checks and conversions for parameters.

        Unlike the original, this returns mode/sigma/lower/upper as plain
        Python floats. That is required for the lru_cache keys below --
        ndarrays are unhashable and would raise on lookup.

        A bound of None means "unbounded on that side" and is replaced by
        mode -/+ 1000 sigma, where BL2 is indistinguishable from the normal
        side. Infinite bounds are rejected rather than silently returning NaN;
        pass None instead.
        """
        normal_approx_factor = 1e+3
        if lower is None:
            lower = mode - normal_approx_factor * sigma
        if upper is None:
            upper = mode + normal_approx_factor * sigma

        def _scalar(v):
            v = np.asarray(v, dtype=float)
            return float(v.ravel()[0])

        mode = _scalar(mode)
        sigma = _scalar(sigma)
        lower = _scalar(lower)
        upper = _scalar(upper)

        if not all(np.isfinite([mode, sigma, lower, upper])):
            raise ValueError(
                "mode, sigma, lower and upper must be finite; use None for "
                "an unbounded side")

        if not ignore_checks:
            if not ((lower < mode < upper) and (sigma > 0)):
                raise ValueError(
                    "Requires (lower < mode < upper) and (sigma > 0)")
            scale = max(abs(lower), abs(upper))
            if scale > 0 and sigma < self._RESOLUTION_WARN * scale:
                warnings.warn(
                    "sigma is %.1e of the coordinate scale (max |bound| = "
                    "%.3g); float64 resolution of x - lower limits cdf/ppf "
                    "accuracy to roughly %.0e. Shift the problem nearer the "
                    "origin if you need more."
                    % (sigma / scale, scale, 2.2e-16 * scale / sigma),
                    RuntimeWarning, stacklevel=3)

        x = np.atleast_1d(np.asarray(x, dtype=float))
        return x, mode, sigma, lower, upper

    def pdf(self, x, mode, sigma, lower=None, upper=None,
            ignore_checks=False, normed=True):
        """
        Probability density function (pdf).

        Parameters
        ----------
        x : {float, list, ndarray of floats}
            x-value(s)
        mode : float
            x at the point of maximum frequency
        sigma : float
            indicates spread of data around mode
        lower, upper : float or None
            bounds of the domain
        """
        x, mode, sigma, lower, upper = self._argcheck_final(
            x, mode, sigma, lower, upper, ignore_checks=ignore_checks)

        args, _a, _b, Z = _prep(mode, sigma, lower, upper, self.n_quad)
        y = _unnorm_pdf(x, *args)
        if normed:
            y = y / Z
        return y

    def density_reference(self, x, mode, sigma, lower=None, upper=None,
                          ignore_checks=False):
        """
        The pdf written as its definition (Eq. 3), for reading and checking --
        the normalized geometric mean of the two lognormal factors.

        pdf() returns the same values via an algebraically-fused fast path;
        this returns them via the plain definition. They agree to ~machine
        precision (the test suite enforces it). Use pdf() in practice; use this
        when you want to confirm by eye what pdf() is computing, or as the
        ground-truth density in a derivation.
        """
        x, mode, sigma, lower, upper = self._argcheck_final(
            x, mode, sigma, lower, upper, ignore_checks=ignore_checks)
        args, _a, _b, Z = _prep(mode, sigma, lower, upper, self.n_quad)
        return _reference_unnorm(x, *args) / Z

    def logpdf(self, x, mode, sigma, lower=None, upper=None):
        """Log of the pdf."""
        with np.errstate(divide='ignore'):
            return np.log(self.pdf(x, mode, sigma, lower, upper))

    def cdf(self, x, mode, sigma, lower=None, upper=None,
            ignore_checks=False, normed=True):
        """
        Cumulative distribution function (cdf).

        Whole panels come from the cached graded-panel table; each x adds one
        vectorized Gauss-Legendre rule over its partial panel.
        """
        x, mode, sigma, lower, upper = self._argcheck_final(
            x, mode, sigma, lower, upper, ignore_checks=ignore_checks)

        args, a, b, Z = _prep(mode, sigma, lower, upper, self.n_quad)

        # Integrate only over the effective support: below `a` the mass is
        # negligible, above `b` the cdf has saturated. Spreading a fixed-order
        # rule across the full (possibly vastly wider) support is what makes
        # sharply-peaked cases inaccurate.
        xc = np.clip(np.asarray(x, dtype=float), a, b)
        y = _gl_cdf(xc, mode, sigma, lower, upper, self.n_quad)
        y = np.where(np.asarray(x, dtype=float) <= a, 0.0, y)
        y = np.where(np.asarray(x, dtype=float) >= b, Z, y)
        if normed:
            y = np.clip(y / Z, 0.0, 1.0)
        return y

    def logcdf(self, x, mode, sigma, lower=None, upper=None):
        """Log of the cdf."""
        with np.errstate(divide='ignore'):
            return np.log(self.cdf(x, mode, sigma, lower, upper))

    def sf(self, x, mode, sigma, lower=None, upper=None):
        """Survival function, 1 - cdf."""
        return 1.0 - self.cdf(x, mode, sigma, lower, upper)

    def logsf(self, x, mode, sigma, lower=None, upper=None):
        """Log of the survival function."""
        with np.errstate(divide='ignore'):
            return np.log(self.sf(x, mode, sigma, lower, upper))

    def ppf(self, q, mode, sigma, lower=None, upper=None,
            ignore_checks=False):
        """
        Percent-point function (ppf), the inverse of the cdf.

        At dial positions 'fast'/'balanced'/'precise' this inverts the cached
        cumulative spline (fast; error set by n_grid). At 'exact' it instead
        root-finds cdf(x) = q with brentq, to ~machine precision, at the cost
        of one solve per quantile.
        """
        q, mode, sigma, lower, upper = self._argcheck_final(
            q, mode, sigma, lower, upper, ignore_checks=ignore_checks)
        qc = np.clip(q, 0.0, 1.0)

        if self._ppf_method == 'solve':
            args, a, b, Z = _prep(mode, sigma, lower, upper, self.n_quad)

            def _f(x, target):
                return float(_gl_cdf([x], mode, sigma, lower, upper,
                                     self.n_quad)[0] / Z - target)

            out = np.empty_like(qc)
            flat_in = qc.ravel()
            flat_out = out.ravel()
            for i, qi in enumerate(flat_in):
                if qi <= 0.0:
                    flat_out[i] = lower
                elif qi >= 1.0:
                    flat_out[i] = upper
                else:
                    # xtol scales with sigma so a very narrow peak is still
                    # resolved; rtol is brentq's floor (4 * machine eps).
                    flat_out[i] = brentq(_f, a, b, args=(qi,),
                                         xtol=1e-13 * min(1.0, sigma),
                                         rtol=4 * np.finfo(float).eps,
                                         maxiter=200)
            return out

        _, ppf_spline, a, b = _cdf_grid(mode, sigma, lower, upper,
                                        self.n_quad, self.n_grid)
        y = ppf_spline(qc)
        # PCHIP returns nan at/outside the ends; pin them to the support.
        y = np.where(qc <= 0.0, lower, y)
        y = np.where(qc >= 1.0, upper, y)
        y = np.where(np.isnan(y), np.clip(y, a, b), y)
        return np.clip(y, lower, upper)

    def isf(self, q, mode, sigma, lower=None, upper=None):
        """Inverse survival function."""
        return self.ppf(1.0 - np.asarray(q, dtype=float),
                        mode, sigma, lower, upper)

    def rvs(self, mode, sigma, lower=None, upper=None, size=1,
            random_state=None):
        """
        Generates random variate(s) by inverse-transform sampling.

        Always inverts the cached grid, regardless of the precision dial --
        including at 'exact'. Grid inversion error is far below Monte-Carlo
        sampling noise for any realistic sample size, and a brentq solve per
        sample would be needlessly slow. (The dial governs cdf/ppf accuracy,
        not sampling quality.)
        """
        q, mode, sigma, lower, upper = self._argcheck_final(
            np.asarray([], dtype=float), mode, sigma, lower, upper)
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=int(size))
        _, ppf_spline, a, b = _cdf_grid(mode, sigma, lower, upper,
                                        self.n_quad, self.n_grid)
        y = ppf_spline(np.clip(u, 0.0, 1.0))
        y = np.where(np.isnan(y), np.clip(y, a, b), y)
        return np.clip(y, lower, upper)

    def _stats(self, mode, sigma, lower, upper, moments='mv'):
        """
        Calculates population mean, variance, skewness and excess kurtosis.

        Phase 2: these are now integrated with the same graded Gauss-Legendre
        rule as the cdf (see _gl_moments), replacing the original grid sum.
        The grid sum weighted points by (upper - lower)/n instead of /(n - 1)
        and never renormalized, which biased the mean toward zero (so it was
        not shift-equivariant); it applied the *sample* skewness correction
        sqrt(n(n-1))/(n-2) to a population moment; and its fixed 1001-point
        grid aliased narrow peaks -- e.g. var(100, 1) came out 2.68, not 1.
        """
        _, mode, sigma, lower, upper = self._argcheck_final(
            0, mode, sigma, lower, upper)
        return _gl_moments(mode, sigma, lower, upper, self.n_quad)

    def stats(self, mode, sigma, lower=None, upper=None, moments='mv'):
        """
        Calculates population mean('m'), variance('v'), skew('s'),
        and/or kurtosis('k').
        """
        output = []
        mu, mu2, g1, g2 = self._stats(mode, sigma, lower, upper)
        if 'm' in moments:
            output.append(mu)
        if 'v' in moments:
            output.append(mu2)
        if 's' in moments:
            output.append(g1)
        if 'k' in moments:
            output.append(g2)
        if len(output) == 1:
            return output[0]
        return tuple(output)

    def moment(self, n, mode, sigma, lower=None, upper=None):
        """Calculates n-th order non-central moment of distribution."""
        mu, mu2, g1, g2 = self._stats(mode, sigma, lower, upper)
        mu3 = g1 * mu2 ** 1.5
        mu4 = (g2 + 3.0) * mu2 ** 2

        if n == 0:
            return 1.0
        elif n == 1:
            return mu
        elif n == 2:
            return mu2 + mu ** 2
        elif n == 3:
            return mu3 + mu * (mu ** 2 + 3 * mu2)
        elif n == 4:
            return mu4 + mu * ((mu ** 2 + 6 * mu2) * mu + 4 * mu3)
        else:
            pass

    def support(self, mode, sigma, lower=None, upper=None):
        """End-points of the distribution's support."""
        _, mode, sigma, lower, upper = self._argcheck_final(
            0, mode, sigma, lower, upper)
        return lower, upper

    def median(self, mode, sigma, lower=None, upper=None):
        """Median."""
        return self.ppf(0.5, mode, sigma, lower, upper)

    def mean(self, mode, sigma, lower=None, upper=None):
        """Mean."""
        return self.stats(mode, sigma, lower, upper, moments='m')

    def var(self, mode, sigma, lower=None, upper=None):
        """Variance."""
        return self.stats(mode, sigma, lower, upper, moments='v')

    def std(self, mode, sigma, lower=None, upper=None):
        """Standard deviation."""
        return np.sqrt(self.var(mode, sigma, lower, upper))

    def entropy(self, mode, sigma, lower=None, upper=None):
        """
        Differential entropy of the random variable, in nats.
        Adapts the formula for entropy of the truncated normal distribution.
        """
        _, mode, sigma, lower, upper = self._argcheck_final(
            0, mode, sigma, lower, upper)

        mu, var = self.stats(mode, sigma, lower, upper)
        std = np.sqrt(var)
        alpha = (lower - mu) / std
        beta = (upper - mu) / std
        Z = norm_cdf(beta) - norm_cdf(alpha)
        entropy = 0.5 * np.log(2 * np.pi * np.e * var * Z ** 2)
        entropy += (alpha * norm_pdf(alpha) - beta * norm_pdf(beta)) / (2 * Z)
        return entropy

    def nnlf(self, x, mode, sigma, lower=None, upper=None):
        """Negative loglikelihood function."""
        return -np.sum(self.logpdf(x, mode, sigma, lower, upper), axis=0)

    def fit(self, data, MLE=True, verbose=False,
            fmode=None, fsigma=None, flower=None, fupper=None,
            gmode=None, gsigma=None, glower=None, gupper=None):
        """
        Estimates bounded_lognorm parameters from user-supplied random
        variates. See original docstring; behaviour is unchanged.
        """
        data = np.asarray(data)
        sample_size = len(data)
        shapes = np.ones(4)

        fparams = np.array([fmode, fsigma, flower, fupper])
        fmask = (fparams != np.array(None))
        shapes[fmask] = fparams[fmask]

        gparams = np.array([gmode, gsigma, glower, gupper])
        minmax_cushion = 0.0001
        gdefaults = np.array([np.median(data), data.std(),
                              data.min() * (1 - minmax_cushion),
                              data.max() * (1 + minmax_cushion)])
        gmask = (gparams != np.array(None))
        gparams[~gmask] = gdefaults[~gmask]
        gparams = np.array(gparams[~fmask], dtype='float64')

        def minimize_this(args):
            shapes[~fmask] = args
            try:
                return self.nnlf(data, *shapes)
            except ValueError:
                return np.inf

        def curve_to_be_fit(x, *args):
            shapes[~fmask] = args
            try:
                if fit_cdf:
                    return self.cdf(x, *shapes, ignore_checks=True)
                return self.pdf(x, *shapes, ignore_checks=True)
            except ValueError:
                # Invalid trial parameters (e.g. mode outside the bounds,
                # sigma <= 0) now raise instead of returning NaN; steer the
                # optimizer away with a large, finite residual.
                return np.full(np.shape(x), 1e6)

        with np.errstate(invalid='ignore', divide='ignore'):

            if MLE:
                results = minimize(minimize_this, gparams,
                                   method='Nelder-Mead')
                if not results.success:
                    raise Exception(results)
                shapes[~fmask] = results.x

            else:
                try:
                    xdata, counts = np.unique(data, return_counts=True)
                    ydata = np.cumsum(counts, dtype=np.double) / sample_size
                    fit_cdf = True
                    results = curve_fit(curve_to_be_fit, xdata, ydata,
                                        p0=gparams)
                except Exception:
                    ydata, xdata = np.histogram(data, density=True,
                                                bins=sample_size)
                    xdata = np.mean([xdata[:-1], xdata[1:]], axis=0)
                    fit_cdf = False
                    results = curve_fit(curve_to_be_fit, xdata, ydata,
                                        p0=gparams)

                shapes[~fmask] = results[0]

            if verbose:
                return shapes, results
            return shapes


bl2 = bounded_lognorm2_gen()
