"""
bounded_lognorm (BL) -- bounded lognormal distribution, piecewise form.

A scipy.stats.rv_continuous subclass. Left of the mode the density is a
lognormal pdf located at the lower bound; right of it, a reflected lognormal
pdf located at the upper bound, scaled to meet the left piece at the mode;
the whole is normalized (Eq. 2). Each piece has its mode at `mode` and
standard deviation `sigma` before truncation.

BL is NOT BL2 (bounded_lognorm._bl2). They share the parameters and the
bounds bridge but are different distributions: BL2 is the normalized
geometric mean of the two lognormals, smooth everywhere; BL joins them at the
mode, where its density is only C^1. Their cdfs differ by 0.03-0.05 for
typical parameters and by up to ~0.26 with the mode near a bound.

Like BL2 since Phase 3, BL is a scipy location-scale family: shapes c =
(mode - lower)/(upper - lower) and s = sigma/(upper - lower) on [0, 1], with
loc = lower and scale = upper - lower. bl.from_bounds(mode, sigma, lower,
upper) builds a frozen distribution from the bounds form.

Everything is closed form (see docs/phase5_bl_port.md): writing t_i = a_i^2
for the log-variance of piece i (a_i from lognorm_params), the density is

    f(y) = f0 * exp(-ln(y/c)^2 / (2 a1^2))                  y <= c
    f(y) = f0 * exp(-ln((1-y)/(1-c))^2 / (2 a2^2))          y >= c

with f0 = 1 / (c a1 R(a1) + (1-c) a2 R(a2)), R the Mills ratio. The cdf,
sf, quantiles, moments and entropy follow from it through the normal cdf.

-----------------------------------------------------------------------------
Attribution (full detail and references in ATTRIBUTION.md):
  Original work, 2020-2024 -- Ted Teske (sole author):
    the BL distribution (Eq. 2), the lognorm_params mode/sigma
    reparametrization, closed-form cdf and ppf by concatenating lognormal
    pieces, moments as weighted truncated-lognormal partial moments, the
    rv_continuous subclass and its API, and all example usage.
  Phase 5 port, 2026 -- Ted Teske with Claude (Anthropic):
    the (c, s, loc, scale) parametrization and shared bounds bridge; scipy
    hooks in place of public overrides; the mode-normalized closed forms
    above; cdf and sf each summed from their own end; quantiles inverted in
    the tails and polished; moments about the mode with a series for narrow
    pieces; closed-form entropy; input order preserved. See
    docs/phase5_bl_port.md.
-----------------------------------------------------------------------------
"""
from collections import namedtuple
from functools import lru_cache
from math import comb, log

import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, log_ndtr, ndtri_exp
from scipy.stats import rv_continuous

from ._bl2 import lognorm_cdf, lognorm_params, lognorm_pdf, _by_params, \
    _param_groups
from ._bridge import _BoundsBridge, PRECISION_DIAL

_SQRT_HALF_PI = np.sqrt(np.pi / 2.0)
_SQRT2 = np.sqrt(2.0)


def _mills(x):
    """Mills ratio R(x) = Phi(-x) / phi(x), for x >= 0 (arrays fine).
    Overflow-free: R(x) ~ 1/x for large x."""
    return erfcx(np.asarray(x, dtype=float) / _SQRT2) * _SQRT_HALF_PI


def _log_mills(x):
    return np.log(erfcx(np.asarray(x, dtype=float) / _SQRT2)) \
        + np.log(_SQRT_HALF_PI)


# ---------------------------------------------------------------------------
# Per-(c, s) constants, standard form.
#
# For piece i (L_1 = c on the left, L_2 = 1 - c on the right), a_i is the
# lognormal shape that lognorm_params gives for mode L_i and SD s. In the
# coordinate v = ln(y / c) (left) or ln((1 - y)/(1 - c)) (right), v <= 0,
# each piece is f0 * exp(-v^2 / (2 a^2)), and its mass is f0 * L * a * R(a).
# ---------------------------------------------------------------------------
_Params = namedtuple('_Params', 'c s a1 a2 f0 log_f0 p1 p2')


@lru_cache(maxsize=1024)
def _params(c, s):
    a1 = float(lognorm_params(c, s)[1])
    a2 = float(lognorm_params(1.0 - c, s)[1])
    h1 = c * a1 * float(_mills(a1))
    h2 = (1.0 - c) * a2 * float(_mills(a2))
    f0 = 1.0 / (h1 + h2)
    return _Params(c, s, a1, a2, f0, -log(h1 + h2), h1 * f0, h2 * f0)


# Gauss-Legendre nodes for the short integrals next to the mode.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


def _mode_mass(v, a):
    """
    M(v) = integral from v to 0 of exp(u - u^2 / (2 a^2)) du, for v <= 0.

    Times f0 * L, this is the mass of a piece between the mode and the point
    with log-coordinate v. Far from the mode it is the closed form
    a * (R(a) - exp(v - v^2/(2a^2)) * R(a - v/a)), a difference of two normal
    tail areas whose ratio is below ~0.6 there, so nothing cancels. Within
    |v| <= min(a, 1) of the mode that difference would cancel, so the short
    integral is taken directly by 16-point Gauss-Legendre (the integrand is
    smooth and varies by at most a factor ~e^1.5 over it: exact to rounding).
    """
    v = np.asarray(v, dtype=float)
    out = np.empty(v.shape)
    near = v >= -min(a, 1.0)
    if np.any(near):
        vn = v[near][:, None]
        u = 0.5 * vn * (1.0 - _GL_X)            # nodes mapped onto [v, 0]
        out[near] = (-0.5 * vn[:, 0]) * \
            (np.exp(u - u * u / (2.0 * a * a)) @ _GL_W)
    far = ~near
    if np.any(far):
        vf = v[far]
        with np.errstate(over='ignore', invalid='ignore'):
            tail = np.exp(vf - vf * vf / (2.0 * a * a)) * _mills(a - vf / a)
        out[far] = a * (_mills(a) - np.where(np.isfinite(tail), tail, 0.0))
    return out


def _tail(v, a):
    """exp(v - v^2/(2a^2)) * R(a - v/a), v <= 0: the mass of a piece beyond
    log-coordinate v, divided by f0 * L * a. Relative-accurate however small."""
    v = np.asarray(v, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        e = np.exp(v - v * v / (2.0 * a * a))
        return np.where(e > 0, e * _mills(a - v / a), 0.0)


def _log_tail(v, a):
    v = np.asarray(v, dtype=float)
    return v - v * v / (2.0 * a * a) + _log_mills(a - v / a)


# ---------------------------------------------------------------------------
# Moments of D = 1 - exp(V) within one piece, V = ln(y/c) or ln((1-y)/(1-c)).
#
# The closed form is the truncated-lognormal partial moment
#     E[exp(jV) | piece] = R((1 + j) a) / R(a),
# so E[D^k] = sum_j C(k, j) (-1)^j R((1 + j) a) / R(a). That alternating sum
# cancels when the piece is narrow (a small): it loses about k log10(1/a)
# digits. It is therefore evaluated three ways, picking by conditioning:
#   - direct, when its terms do not cancel much (computed condition number);
#   - the same finite difference expanded in a series in a (exact
#     rearrangement, via Stirling numbers of the second kind and the
#     derivatives of R), when k a is small;
#   - otherwise (only for k >= ~8 in a band of a) adaptive quadrature of
#     the same expectation, which is smooth on [0, inf).
# ---------------------------------------------------------------------------
_DIRECT_COND_OK = 50.0       # direct sum used outright below this
_DIRECT_COND_MAX = 1000.0    # ... and beyond the series band below this (~2e-13)
_SERIES_KA_MAX = 2.0         # series used when k * a is at most this


def _d_direct(k, a, Ra):
    t = np.array([comb(k, j) * (-1.0) ** j * float(_mills((1 + j) * a))
                  for j in range(k + 1)])
    s = t.sum()
    cond = np.abs(t).sum() / abs(s) if s > 0 else np.inf
    return s / Ra, cond


def _d_series_all(kmax, a, Ra):
    """E[D^k] for k = 0..kmax by the series (valid for small k a)."""
    M = kmax + 60
    e = np.zeros(M + 1)                       # [x^m] (exp(a x) - 1)
    f = 1.0
    for m in range(1, M + 1):
        f *= a / m
        e[m] = f
    d = np.empty(M + 1)                       # R^(m)(a)
    d[0] = Ra
    d[1] = a * Ra - 1.0
    for m in range(2, M + 1):
        d[m] = a * d[m - 1] + (m - 1) * d[m - 2]
    out = np.empty(kmax + 1)
    out[0] = 1.0
    b = np.zeros(M + 1)                       # [x^m] (exp(a x) - 1)^k
    b[0] = 1.0
    for k in range(1, kmax + 1):
        b = np.convolve(b, e)[:M + 1]
        out[k] = (-1.0) ** k * float(b @ d) / Ra
    return out


def _d_quad(k, a, Ra):
    """E[D^k] as a smooth integral over w = -(a + Z) >= 0, scaled by a^k."""
    def f(w):
        return (-np.expm1(-a * w) / a) ** k * np.exp(-a * w - 0.5 * w * w)
    val = quad(f, 0.0, np.inf, epsabs=0.0, epsrel=2e-14, limit=200)[0]
    return val * a ** k / Ra


def _d_path(k, a, cond):
    """'direct', 'series' or 'quad' for E[D^k], given the direct sum's
    condition number. For k <= 4 (the stats) this never returns 'quad'."""
    if cond <= _DIRECT_COND_OK:
        return 'direct'
    if k * a <= _SERIES_KA_MAX:
        return 'series'
    return 'direct' if cond <= _DIRECT_COND_MAX else 'quad'


@lru_cache(maxsize=256)
def _d_moments(a, kmax):
    """(E[D^0], ..., E[D^kmax]) for a piece of shape a."""
    Ra = float(_mills(a))
    series = None
    out = np.empty(kmax + 1)
    out[0] = 1.0
    for k in range(1, kmax + 1):
        val, cond = _d_direct(k, a, Ra)
        path = _d_path(k, a, cond)
        if path == 'series':
            if series is None:
                series = _d_series_all(kmax, a, Ra)
            val = series[k]
        elif path == 'quad':
            val = _d_quad(k, a, Ra)
        out[k] = val
    out.flags.writeable = False
    return out


def _stats_std(c, s):
    """Mean, variance, skewness, excess kurtosis of the standard form,
    assembled from moments about the mode (no large cancellations)."""
    P = _params(c, s)
    D1 = _d_moments(P.a1, 4)
    D2 = _d_moments(P.a2, 4)
    # A[k] = E[(Y - c)^k]: left piece Y - c = -c D1, right (1 - c) D2.
    A = [P.p1 * (-c) ** k * D1[k] + P.p2 * (1.0 - c) ** k * D2[k]
         for k in range(5)]
    m1 = A[1]
    mu2 = A[2] - m1 * m1
    mu3 = A[3] - 3.0 * m1 * A[2] + 2.0 * m1 ** 3
    mu4 = A[4] - 4.0 * m1 * A[3] + 6.0 * m1 * m1 * A[2] - 3.0 * m1 ** 4
    return c + m1, mu2, mu3 / mu2 ** 1.5, mu4 / (mu2 * mu2) - 3.0


def _raw_moment_std(n, c, s):
    """E[Y^n] on [0, 1]; every term is positive."""
    P = _params(c, s)
    left = c ** n * float(_mills((1 + n) * P.a1) / _mills(P.a1))
    D2 = _d_moments(P.a2, n)
    right = sum(comb(n, k) * c ** (n - k) * (1.0 - c) ** k * D2[k]
                for k in range(n + 1))
    return P.p1 * left + P.p2 * right


def _entropy_std(c, s):
    """H = -ln f0 + (1/2) sum_i p_i E[(a_i + Z)^2 | Z <= -a_i]
         = -ln f0 + (1/2) sum_i p_i (1 + a_i^2 - a_i / R(a_i))."""
    P = _params(c, s)
    h = -P.log_f0
    for p, a in ((P.p1, P.a1), (P.p2, P.a2)):
        h += 0.5 * p * (1.0 + a * a - a / float(_mills(a)))
    return h


class bounded_lognorm_gen(_BoundsBridge, rv_continuous):
    r"""
    A bounded lognormal (BL) continuous random variable, piecewise form.

    Left of the mode the density is a lognormal pdf located at the lower
    bound; right of it, a reflected lognormal pdf located at the upper bound,
    scaled so the two pieces meet at the mode; the whole is normalized. Both
    pieces have their mode at `mode` and standard deviation `sigma` before
    truncation. The density is continuous with a zero slope at the mode
    (C^1) but its curvature jumps there unless the two sides happen to match.

    BL is a different distribution from BL2 (bounded_lognorm.bl2), the
    geometric-mean form, even with the same parameters: their cdfs differ by
    0.03-0.05 for typical cases and up to ~0.26 with the mode near a bound.
    Do not substitute one for the other.

    Parametrization
    ---------------
    A scipy location-scale family, as BL2:

        loc   = lower
        scale = upper - lower
        c     = (mode - lower) / (upper - lower)     relative mode, 0 < c < 1
        s     = sigma / (upper - lower)              relative sigma, s > 0

    Every rv_continuous method works: pdf, logpdf, cdf, logcdf, sf, logsf,
    ppf, isf, rvs, stats, moment(n) for any n, entropy, median, interval,
    expect, fit, and freezing. If you think in bounds, use the bridge:

        rv = bl.from_bounds(mode=5, sigma=10, lower=-5, upper=20)  # frozen
        rv.pdf(x); rv.cdf(x); rv.mean(); rv.entropy()
        c, s, loc, scale = bl.fit(data)
        mode, sigma, lower, upper = bl.to_bounds(c, s, loc, scale)
        mode, sigma, lower, upper = bl.fit_bounds(data, flower=0)

    MIGRATION from the notebook's bounded_lognorm: positional calls such as
    bounded_lognorm.pdf(x, mode, sigma, lower, upper) become
    bl.from_bounds(mode, sigma, lower, upper).pdf(x). Calling
    bl.pdf(x, mode, sigma, lower, upper) instead reads them as (x, c, s,
    loc, scale): with mode outside (0, 1) that returns NaN, and with mode
    inside (0, 1) it silently means something else.

    Precision
    ---------
    Every quantity has a closed form, accurate to roughly 1e-13 relative
    or better wherever float64 can represent the answer, so there is no
    speed/accuracy trade-off to make. `precision` is accepted, validated and
    kept through freezing only so that code written for BL2's dial runs
    unchanged on BL; it has no effect.

    Caching
    -------
    The per-(c, s) constants (two lognorm_params solves) are cached, keyed
    on (c, s) only, so shifted or rescaled copies share them. Array-valued
    shapes are grouped by distinct (c, s) pair.
    """

    def __init__(self, precision='balanced', **kwargs):
        """
        Parameters
        ----------
        precision : {'fast', 'balanced', 'precise', 'exact'}
            Accepted for API parity with bounded_lognorm2_gen; no effect on
            BL, whose closed forms are already at full accuracy.
        **kwargs
            Passed to rv_continuous (name, seed, ...). The support (a=0, b=1)
            and the shape names (c, s) are fixed by default.
        """
        if precision not in PRECISION_DIAL:
            raise ValueError(
                "precision must be one of %s, got %r"
                % (PRECISION_DIAL, precision))
        kwargs.setdefault('a', 0.0)
        kwargs.setdefault('b', 1.0)
        kwargs.setdefault('name', 'bounded_lognorm')
        kwargs.setdefault('shapes', 'c, s')
        super().__init__(**kwargs)
        self.precision = precision

    # -- the density and its integrals, standard coordinates ---------------
    #
    # scipy calls these only for x (or q) strictly inside (0, 1) and for
    # parameters that passed _argcheck; it handles loc/scale, the support
    # boundaries and invalid arguments itself. Each works elementwise, so
    # the output keeps the input's order and shape.

    @staticmethod
    def _logv(y, c):
        """(left mask, v): v = ln(y/c) left of the mode, ln((1-y)/(1-c))
        right of it; v <= 0 on both sides.

        Near the mode both are taken as log1p of (y - c), which is exact
        there: ln((1-y)/(1-c)) would round 1 - y first, which next to a mode
        at c ~ 1e-7 leaves v with a relative error of ~1e-6."""
        left = y <= c
        d = y - c
        with np.errstate(divide='ignore', invalid='ignore'):
            vl = np.where(d > -0.5 * c, np.log1p(d / c), np.log(y / c))
            vr = np.where(-d > -0.5 * (1.0 - c), np.log1p(-d / (1.0 - c)),
                          np.log((1.0 - y) / (1.0 - c)))
        return left, np.where(left, vl, vr)

    def _pdf(self, x, c, s):
        def f(y, c, s):
            P = _params(c, s)
            left, v = self._logv(y, c)
            a = np.where(left, P.a1, P.a2)
            return P.f0 * np.exp(-v * v / (2.0 * a * a))
        return _by_params(f, x, c, s)

    def _logpdf(self, x, c, s):
        def f(y, c, s):
            P = _params(c, s)
            left, v = self._logv(y, c)
            a = np.where(left, P.a1, P.a2)
            return P.log_f0 - v * v / (2.0 * a * a)
        return _by_params(f, x, c, s)

    def _cdf(self, x, c, s):
        """Left of the mode: the left piece's own tail (relative-accurate in
        the lower tail). Right of it: p1 plus the mass from the mode, a sum
        of positive terms (no 1 - sf)."""
        def f(y, c, s):
            P = _params(c, s)
            left, v = self._logv(y, c)
            out = np.empty(y.shape)
            out[left] = P.f0 * c * P.a1 * _tail(v[left], P.a1)
            r = ~left
            out[r] = P.p1 + P.f0 * (1.0 - c) * _mode_mass(v[r], P.a2)
            return np.clip(out, 0.0, 1.0)
        return _by_params(f, x, c, s)

    def _sf(self, x, c, s):
        """Mirror of _cdf: integrated from the right, so the upper tail keeps
        its relative accuracy."""
        def f(y, c, s):
            P = _params(c, s)
            left, v = self._logv(y, c)
            out = np.empty(y.shape)
            r = ~left
            out[r] = P.f0 * (1.0 - c) * P.a2 * _tail(v[r], P.a2)
            out[left] = P.p2 + P.f0 * c * _mode_mass(v[left], P.a1)
            return np.clip(out, 0.0, 1.0)
        return _by_params(f, x, c, s)

    def _logcdf(self, x, c, s):
        """Finite in the lower tail even where cdf underflows."""
        def f(y, c, s):
            P = _params(c, s)
            left, v = self._logv(y, c)
            out = np.empty(y.shape)
            out[left] = (P.log_f0 + log(c * P.a1)
                         + _log_tail(v[left], P.a1))
            r = ~left
            out[r] = np.log(P.p1 + P.f0 * (1.0 - c) * _mode_mass(v[r], P.a2))
            return np.minimum(out, 0.0)
        return _by_params(f, x, c, s)

    def _logsf(self, x, c, s):
        """Finite in the upper tail even where sf underflows."""
        def f(y, c, s):
            P = _params(c, s)
            left, v = self._logv(y, c)
            out = np.empty(y.shape)
            r = ~left
            out[r] = (P.log_f0 + log((1.0 - c) * P.a2)
                      + _log_tail(v[r], P.a2))
            out[left] = np.log(P.p2 + P.f0 * c * _mode_mass(v[left], P.a1))
            return np.minimum(out, 0.0)
        return _by_params(f, x, c, s)

    # -- quantiles -----------------------------------------------------------

    @staticmethod
    def _invert_piece(logq, a, p):
        """Log-coordinate v of the point where one piece's own tail (mass
        beyond it, away from the mode) equals exp(logq), logq <= log p.
        Closed form: that tail is p * Phi(beta) / Phi(-a), v = a (beta + a).
        Worked in log probability, so q may be as small as float64 allows."""
        logP = logq + log_ndtr(-a) - log(p)
        beta = ndtri_exp(np.minimum(logP, 0.0))
        return np.minimum(a * (beta + a), 0.0)

    def _quantile_closed(self, q, c, s, upper_tail):
        """Closed-form inverse, before polishing. q is the lower-tail
        probability (ppf) or the upper-tail one (isf)."""
        P = _params(c, s)
        q = np.asarray(q, dtype=float)
        # Each q is inverted on the piece whose own tail it lies in; the
        # other side is reached through the complement 1 - q.
        p_near, p_far = (P.p2, P.p1) if upper_tail else (P.p1, P.p2)
        a_near, a_far = (P.a2, P.a1) if upper_tail else (P.a1, P.a2)
        near = q <= p_near
        y = np.empty(q.shape)
        with np.errstate(divide='ignore'):
            v = self._invert_piece(np.log(q[near]), a_near, p_near)
            w = self._invert_piece(np.log1p(-q[~near]), a_far, p_far)
        # v (left piece) and w (right piece) back to y. The left piece is
        # y = c e^v, relative-accurate however small. The right piece is
        # 1 - (1 - c) e^w far out (relative-accurate in 1 - y) but
        # c - (1 - c) expm1(w) next to the mode, where rounding 1 - ... would
        # lose y - c.
        def from_left(v):
            return c * np.exp(v)

        def from_right(w):
            return np.where(w > -np.log(2.0), c - (1.0 - c) * np.expm1(w),
                            1.0 - (1.0 - c) * np.exp(w))

        if upper_tail:
            y[near] = from_right(v)
            y[~near] = from_left(w)
        else:
            y[near] = from_left(v)
            y[~near] = from_right(w)
        return np.clip(y, 0.0, 1.0)

    def _polish(self, y, q, c, s, upper_tail, steps=2):
        """Newton steps on the accurate cdf (or sf). The closed-form inverse
        goes through 1 - q on the far side of the mode, which loses relative
        accuracy when the mode piece on the near side carries little mass;
        two steps restore it. Deep in a tail (q < 1e-280) the closed form is
        already relative-accurate and is left alone."""
        for _ in range(steps):
            F = self._sf(y, c, s) if upper_tail else self._cdf(y, c, s)
            f = self._pdf(y, c, s)
            ok = (f > 0) & (q > 1e-280) & (y > 0) & (y < 1)
            step = np.where(ok, (F - q) / np.where(ok, f, 1.0), 0.0)
            y = np.clip(y + step if upper_tail else y - step, 0.0, 1.0)
        return y

    def _ppf(self, q, c, s):
        """Closed-form inverse of the cdf, piece by piece (the lower tail by
        the left piece's own tail, so relative accuracy in q holds down to
        q ~ 1e-300), then polished by Newton on the cdf."""
        def f(q, c, s):
            y = self._quantile_closed(q, c, s, upper_tail=False)
            return self._polish(y, q, c, s, upper_tail=False)
        return _by_params(f, q, c, s)

    def _isf(self, q, c, s):
        """Inverse of the sf, mirroring _ppf from the upper bound. (scipy's
        default, ppf(1 - q), cannot resolve q below ~1e-16.)"""
        def f(q, c, s):
            y = self._quantile_closed(q, c, s, upper_tail=True)
            return self._polish(y, q, c, s, upper_tail=True)
        return _by_params(f, q, c, s)

    def _rvs(self, c, s, size=None, random_state=None):
        """Inverse-transform sampling with the closed-form inverse, unpolished
        (its error, at most ~1e-9 of the scale next to the mode, is far below
        sampling noise). Draws come out in the order generated."""
        u = random_state.uniform(size=size)
        return _by_params(
            lambda q, c, s: self._quantile_closed(q, c, s, upper_tail=False),
            u, c, s)

    # -- moments and entropy -------------------------------------------------

    def _stats(self, c, s, moments='mv'):
        """Mean, variance, skewness and excess kurtosis of the standard form,
        from the truncated-lognormal partial moments of each piece taken
        about the mode (see _d_moments). scipy maps them to loc/scale."""
        c, s = np.broadcast_arrays(np.asarray(c, dtype=float),
                                   np.asarray(s, dtype=float))
        out = [np.empty(c.size) for _ in range(4)]
        for cc, ss, idx in _param_groups(c, s):
            for o, v in zip(out, _stats_std(cc, ss)):
                o[idx] = v
        return tuple(o.reshape(c.shape) for o in out)

    def _munp(self, n, c, s):
        """n-th raw moment E[Y**n] of the standard form, for any integer
        n >= 0; a sum of positive terms. scipy's moment() maps it to
        loc/scale."""
        c, s = np.broadcast_arrays(np.asarray(c, dtype=float),
                                   np.asarray(s, dtype=float))
        out = np.empty(c.size)
        for cc, ss, idx in _param_groups(c, s):
            out[idx] = _raw_moment_std(int(n), cc, ss)
        return out.reshape(c.shape)

    def _entropy(self, c, s):
        """Differential entropy of the standard form, in nats, in closed
        form; scipy adds ln(scale)."""
        c, s = np.broadcast_arrays(np.asarray(c, dtype=float),
                                   np.asarray(s, dtype=float))
        out = np.empty(c.size)
        for cc, ss, idx in _param_groups(c, s):
            out[idx] = _entropy_std(cc, ss)
        return out.reshape(c.shape)[()]

    # -- extras beyond the rv_continuous API ---------------------------------

    def density_reference(self, x, c, s, loc=0.0, scale=1.0):
        """
        The pdf written as its definition (Eq. 2), for reading and checking:
        a lognormal pdf at the lower bound left of the mode, a reflected one
        at the upper bound right of it scaled by k to meet it, divided by
        Z = F_left(mode) + k F_right(mode), using the original helper
        functions. pdf() returns the same values by the mode-normalized form;
        the test suite pins the two together. Scalar c, s only.
        """
        c, s = float(c), float(s)
        y = np.ravel((np.asarray(x, dtype=float) - loc) / scale)
        mu1, s1 = lognorm_params(c, s)
        mu2, s2 = lognorm_params(1.0 - c, s)
        k = lognorm_pdf(c, mu1, s1)[0] / lognorm_pdf(1.0 - c, mu2, s2)[0]
        Z = lognorm_cdf(c, mu1, s1)[0] + k * lognorm_cdf(1.0 - c, mu2, s2)[0]
        g = np.where(y <= c, lognorm_pdf(y, mu1, s1),
                     k * lognorm_pdf(1.0 - y, mu2, s2))
        return (g / Z / scale).reshape(np.shape(x))


bl = bounded_lognorm_gen()
