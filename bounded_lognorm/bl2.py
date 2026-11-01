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
    cumulative grid for ppf/rvs, the precision dial, the _reference_unnorm /
    _unnorm_pdf split, and the correction of the unnormalized density constant
    from 1/(2*pi*sqrt(s1 s2)) to 1/sqrt(2*pi*s1 s2) (cancels under
    normalization; affects only normed=False output).
  Third-party helpers retain their inline citations (Stack Overflow, CC BY-SA).
-----------------------------------------------------------------------------
"""
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, curve_fit, minimize
from scipy.special import erf, erfinv
from scipy.stats import rv_continuous


# ===========================================================================
# Helper functions -- UNCHANGED from the original implementation (T. Teske).
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

    """
    # Converts parameters to floats in case exponents below too big for ints.
    # https://stackoverflow.com/questions/67097840
    def GetItem(inputobject):
        while type(inputobject) is np.ndarray:
            inputobject = inputobject[0]
        return inputobject

    mode, sigma = float(GetItem(mode)), float(GetItem(sigma))

    # Finds the roots of the polynomial g^4 - g^3 - sigma^2/mode^2,
    # where g = e^(variance of the associated Normal distribution),
    # and uses just the real, positive root.
    try:
        # This is the exact form [1].
        a = np.cbrt(np.sqrt(81 * mode**8 * sigma**4 +
                            768 * mode**6 * sigma**6) -
                    9 * mode**4 * sigma**2)
        b = np.cbrt(2 / 3) * sigma**2
        c = np.cbrt(2 / 3)**2 * 2
        d = np.sqrt((-16 * b / a) + (c * a / mode**2) + 1)

        g1 = d / 4
        g2 = (4 * b / a) - a / (np.cbrt(18) * mode**2)
        g3 = 1 / (2 * d)

        g = g1 + np.sqrt(g2 + g3 + 0.5) / 2 + 0.25  # the real, positive root

        # Whereas the above is the real, positive root (the one we'll use),
        # negative or complex roots are commented below:
        # g =  g1 - np.sqrt(g2 + g3 + 0.5) / 2 + 0.25
        # g = -g1 + np.sqrt(g2 - g3 + 0.5) / 2 + 0.25
        # g = -g1 - np.sqrt(g2 - g3 + 0.5) / 2 + 0.25

    except:
        # And this is a slightly slower, numerical approximation of g [2].
        # It is however much more concise.
        p = np.poly1d([1, -1, 0, 0, -(sigma / mode)**2])
        r = p.roots
        g = r[(r.imag == 0) & (r.real > 0)].real

    # Plugs g into the expressions for sigma_N and mu_N.
    sigma_N = np.sqrt(np.log(g))
    mu_N = np.log(mode * g)
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


def _gl_integrate(a, b, args, n, panels=4):
    """
    Vectorized composite Gauss-Legendre integration of _unnorm_pdf from `a` to
    each element of `b`.

    Two things matter here:
    * Composite (`panels` sub-intervals) rather than a single rule, so that a
      narrow peak inside a wide interval is still resolved.
    * Fully vectorized: the density is evaluated ONCE on an (N, panels, n) node
      array, instead of running one Python-level quadrature per endpoint.
    """
    xn, wn = _gauss_legendre(n)
    b = np.atleast_1d(np.asarray(b, dtype=float))

    # Panel edges for each endpoint: shape (N, panels + 1)
    frac = np.linspace(0.0, 1.0, panels + 1)
    edges = a + (b - a)[:, None] * frac[None, :]
    lo, hi = edges[:, :-1], edges[:, 1:]              # (N, panels)

    mid = (hi + lo) / 2.0
    half = (hi - lo) / 2.0
    pts = mid[..., None] + half[..., None] * xn       # (N, panels, n)
    vals = _unnorm_pdf(pts.ravel(), *args).reshape(pts.shape)
    return (vals * wn).sum(axis=2).__mul__(half).sum(axis=1)


@lru_cache(maxsize=64)
def _prep(mode, sigma, lower, upper, n_quad):
    """
    Hoisted, cached setup for one parameter set.

    Computes the underlying normal parameters for both lognormal factors, the
    effective support, and the normalizing constant Z exactly once.

    In the original implementation Z was recomputed inside every single pdf()
    call -- including at every quadrature node of every cdf() point -- even
    though it does not depend on x. That redundancy, not the choice of
    integrator, was the dominant cost.

    Returns
    -------
    args : tuple -> (mu1, s1, mu2, s2, lower, u), the args for _unnorm_pdf
    a, b : float -> effective support; g is negligible outside it
    Z    : float -> integral of the unnormalized density over the support
    """
    u = upper - lower
    m = mode - lower
    mu1, s1 = lognorm_params(m, sigma)
    mu2, s2 = lognorm_params(u - m, sigma)
    args = (float(mu1), float(s1), float(mu2), float(s2),
            float(lower), float(u))

    a, b = _effective_support(args, mode, lower, upper)
    Z = float(_gl_integrate(a, [b], args, n_quad)[0])
    return args, float(a), float(b), Z


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
    Cached one-pass CDF/PPF grid.

    Evaluates the density on a fine grid once, integrates it cumulatively,
    normalizes, and wraps the result in monotone (PCHIP) splines. This yields
    both cdf and ppf from a single density pass, which is what makes ppf/rvs
    fast -- previously ppf ran a full quadrature at each of n grid points.

    Returns (cdf_spline, ppf_spline, a, b) where (a, b) is the effective
    support; outside it the cdf is 0 or 1 by construction.
    """
    args, a, b, _ = _prep(mode, sigma, lower, upper, n_quad)

    xs = np.linspace(a, b, n_grid)
    ys = _unnorm_pdf(xs, *args)

    F = cumulative_trapezoid(ys, xs, initial=0.0)
    total = F[-1]
    if total <= 0:
        raise ValueError("Density integrates to zero; check parameters.")
    F /= total
    F = np.maximum.accumulate(F)          # guard against fp non-monotonicity
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
#   n_quad : Gauss-Legendre order used by cdf (higher -> more accurate cdf)
#   n_grid : points in the cached grid backing ppf/rvs (higher -> finer
#            quantile inversion)
#   ppf    : 'grid'  -> invert the cached spline (fast, same error as n_grid)
#            'solve' -> root-find cdf(x)=q with brentq (~machine precision,
#                       but one solve per quantile, so much slower)
#
# NOT on the dial, on purpose (keeps the knob honest -- one thing, not five):
#   - cdf composite panels are fixed at 4. Benchmarking showed the count is
#     irrelevant once integration is confined to the effective support.
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
    points at mode=5, sigma=10, bounds=(-5, 20); your machine will differ):

      fast
        - cdf accuracy  ~5e-8
        - ppf accuracy  ~6e-6
        - cdf speed     ~2 ms / 1000 pts
      balanced  (default)
        - cdf accuracy  ~8e-12
        - ppf accuracy  ~4e-7
        - cdf speed     ~6 ms / 1000 pts
      precise
        - cdf accuracy  ~4e-13
        - ppf accuracy  ~2e-8
        - cdf speed     ~10 ms / 1000 pts
      exact
        - cdf accuracy  ~1e-13
        - ppf accuracy  ~1e-13 (brentq root-find, not the grid)
        - cdf speed     ~16 ms / 1000 pts
        - ppf speed     ~1 ms PER QUANTILE -- i.e. ~1000x slower than the
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
            grid resolution for the moment calculations in stats()/entropy();
            retained from the original for backward compatibility
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

    def _argcheck(self, *args):
        """
        Overrides default _argcheck to allow negative parameters.
        """
        cond = 1
        for arg in args:
            if arg is None:
                arg = 0
            cond = np.logical_and(cond, np.isfinite(arg))
        return cond

    def _argcheck_final(self, x, mode, sigma, lower, upper,
                        ignore_checks=False):
        """
        Further compatibility checks and conversions for parameters.

        Unlike the original, this returns mode/sigma/lower/upper as plain
        Python floats. That is required for the lru_cache keys below --
        ndarrays are unhashable and would raise on lookup.
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

        if not ignore_checks:
            if not ((lower < mode < upper) and (sigma > 0)):
                raise ValueError(
                    "Requires (lower < mode < upper) and (sigma > 0)")

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

        Uses a single vectorized Gauss-Legendre rule over all x at once.
        """
        x, mode, sigma, lower, upper = self._argcheck_final(
            x, mode, sigma, lower, upper, ignore_checks=ignore_checks)

        args, a, b, Z = _prep(mode, sigma, lower, upper, self.n_quad)

        # Integrate only over the effective support: below `a` the mass is
        # negligible, above `b` the cdf has saturated. Spreading a fixed-order
        # rule across the full (possibly vastly wider) support is what makes
        # sharply-peaked cases inaccurate.
        xc = np.clip(np.asarray(x, dtype=float), a, b)
        y = _gl_integrate(a, xc, args, self.n_quad)
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
                return float(_gl_integrate(a, [x], args, self.n_quad)[0]
                             / Z - target)

            out = np.empty_like(qc)
            flat_in = qc.ravel()
            flat_out = out.ravel()
            for i, qi in enumerate(flat_in):
                if qi <= 0.0:
                    flat_out[i] = lower
                elif qi >= 1.0:
                    flat_out[i] = upper
                else:
                    flat_out[i] = brentq(_f, a, b, args=(qi,),
                                         xtol=1e-13, rtol=1e-14, maxiter=200)
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
        Calculates population mean, variance, skew and kurtosis numerically.
        Retains the original quadrature-free (grid sum) approach.
        """
        _, mode, sigma, lower, upper = self._argcheck_final(
            0, mode, sigma, lower, upper)

        n = self.n
        xs = np.linspace(lower, upper, num=n)
        ys = self.pdf(xs, mode, sigma, lower, upper)
        ps = ys * (upper - lower) / n

        mu = np.sum(xs * ps)
        mu2 = np.sum((xs - mu) ** 2 * ps)
        s = np.sqrt(mu2)

        g1 = np.sum(((xs - mu) / s) ** 3 * ps)
        g1 *= np.sqrt(n * (n - 1)) / (n - 2)

        g2 = np.sum(((xs - mu) / s) ** 4 * ps) - 3
        return mu, mu2, g1, g2

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
            if fit_cdf:
                return self.cdf(x, *shapes, ignore_checks=True)
            return self.pdf(x, *shapes, ignore_checks=True)

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
