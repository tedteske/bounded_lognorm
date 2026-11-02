"""
Fitting in the bounds parametrization (mode, sigma, lower, upper).

`fit_bounds` restores what the original notebook's fit() offered and the
scipy-conformant `fit` (Phase 3) cannot express: holding any of mode, sigma,
lower and upper fixed (e.g. lower = 0 alone, or the mode alone), and fitting
by least squares on the empirical CDF as well as by maximum likelihood. It
works through a distribution's public API only (bounds_to_shapes, logpdf,
cdf), so it serves any family in this package with the same bridge.

Added in 0.2.0. Ted Teske with Claude (Anthropic), 2026.
"""
import warnings
from collections import namedtuple

import numpy as np
from scipy.optimize import least_squares, minimize

PARAMS = ('mode', 'sigma', 'lower', 'upper')

BoundsParams = namedtuple('BoundsParams', PARAMS)

# 0.2.1: a free bound is reported as collapsed onto the data when moving it
# COLLAPSE_FACTOR times closer to the edge it is held outside of (the
# sample extreme, or a fixed mode) does not worsen the objective by more
# than COLLAPSE_RTOL relative: the fit is then not an interior optimum.
COLLAPSE_FACTOR = 1e3
COLLAPSE_RTOL = 1e-9


class BoundsFit(namedtuple('BoundsFit', PARAMS)):
    """
    Result of fit_bounds. Unpacks like a tuple:

        mode, sigma, lower, upper = bl2.fit_bounds(data)

    and carries the details as attributes:

      method   : 'MLE' or 'LSQ'
      free     : names of the parameters that were fitted (not fixed)
      cov      : LSQ only, 4x4 covariance of (mode, sigma, lower, upper),
                 zero in the rows and columns of fixed parameters, NaN in
                 those of a bound that collapsed onto the data; None for
                 MLE. See fit_bounds for what it does and does not mean.
      stderr   : BoundsParams of standard errors, sqrt(diag(cov)); None
                 for MLE
      nnlf     : negative log-likelihood of the data at the fit (both
                 methods, so fits can be compared)
      success, message : from the optimizer
      optimizer : the raw scipy result, in internal unconstrained
                 coordinates (not the bounds parametrization)
    """
    # No __slots__ = (): instances need a __dict__ for the attributes above.

    def __repr__(self):
        vals = ", ".join(f"{k}={v!r}" for k, v in zip(PARAMS, self))
        return f"BoundsFit({vals}, method={self.method!r})"


def _ecdf(data, kind):
    """
    Empirical CDF at the distinct data values.

    'midpoint' (default): the middle of each jump, (N_k - n_k / 2) / n, where
        N_k counts the data <= x_k and n_k the data == x_k. For distinct data
        this is the Hazen plotting position (i - 1/2) / n. It is unbiased at
        both ends: the largest value sits at 1 - 1/(2n), not at 1.
    'step': the top of each jump, N_k / n, as the original notebook used.
        The largest value then sits at exactly 1, which pulls a fitted upper
        bound toward max(data).
    """
    x, counts = np.unique(data, return_counts=True)
    N = np.cumsum(counts, dtype=float)
    n = float(len(data))
    if kind == 'midpoint':
        return x, (N - counts / 2.0) / n
    if kind == 'step':
        return x, N / n
    raise ValueError("ecdf must be 'midpoint' or 'step'")


def fit_bounds(dist, data, method='MLE', *, fmode=None, fsigma=None,
               flower=None, fupper=None, gmode=None, gsigma=None,
               glower=None, gupper=None, ecdf='midpoint'):
    """See bounded_lognorm2_gen.fit_bounds for the full description."""
    data = np.asarray(data, dtype=float).ravel()
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("data must be a non-empty array of finite values")
    method = str(method).upper()
    if method not in ('MLE', 'LSQ'):
        raise ValueError("method must be 'MLE' or 'LSQ'")

    fixed = dict(zip(PARAMS, (fmode, fsigma, flower, fupper)))
    fixed = {k: float(v) for k, v in fixed.items() if v is not None}
    free = tuple(k for k in PARAMS if k not in fixed)
    if not free:
        raise ValueError("all four parameters are fixed; nothing to fit")
    if any(not np.isfinite(v) for v in fixed.values()):
        raise ValueError("fixed parameters must be finite")

    dmin, dmax = float(data.min()), float(data.max())
    if not dmax > dmin:
        raise ValueError("data must contain at least two distinct values")
    n_points = data.size if method == 'MLE' else np.unique(data).size
    if n_points <= len(free):
        raise ValueError(f"need more than {len(free)} "
                         f"{'data' if method == 'MLE' else 'distinct'} "
                         f"values to fit {len(free)} parameters")

    # -- validate the fixed values against the data and each other ---------
    if 'sigma' in fixed and not fixed['sigma'] > 0:
        raise ValueError("fsigma must be > 0")
    if 'lower' in fixed and not fixed['lower'] < dmin:
        raise ValueError("flower must be below every data value")
    if 'upper' in fixed and not fixed['upper'] > dmax:
        raise ValueError("fupper must be above every data value")
    if 'mode' in fixed:
        m = fixed['mode']
        if ('lower' in fixed and not m > fixed['lower']) or \
                ('upper' in fixed and not m < fixed['upper']):
            raise ValueError("fmode must lie strictly between the bounds")

    # Free bounds stay outside the data and outside a fixed mode, so every
    # trial parameter set is valid and every data point has positive density.
    lo_cap = min(dmin, fixed.get('mode', dmin))
    hi_cap = max(dmax, fixed.get('mode', dmax))
    span = dmax - dmin

    # -- the transform: unconstrained theta <-> bounds parameters ----------
    def to_bounds(theta):
        t = dict(zip(free, theta))
        p = dict(fixed)
        if 'lower' not in p:
            p['lower'] = lo_cap - np.exp(t['lower'])
        if 'upper' not in p:
            p['upper'] = hi_cap + np.exp(t['upper'])
        if 'mode' not in p:
            # Clipped so the mode never rounds onto a bound.
            w = np.clip(1.0 / (1.0 + np.exp(-t['mode'])), 1e-15, 1 - 1e-15)
            p['mode'] = p['lower'] + (p['upper'] - p['lower']) * w
        if 'sigma' not in p:
            p['sigma'] = np.exp(t['sigma'])
        return tuple(p[k] for k in PARAMS)

    def to_theta(p):
        p = dict(zip(PARAMS, p))
        t = {}
        if 'lower' in free:
            t['lower'] = np.log(lo_cap - p['lower'])
        if 'upper' in free:
            t['upper'] = np.log(p['upper'] - hi_cap)
        if 'mode' in free:
            w = (p['mode'] - p['lower']) / (p['upper'] - p['lower'])
            t['mode'] = np.log(w / (1.0 - w))
        if 'sigma' in free:
            t['sigma'] = np.log(p['sigma'])
        return np.array([t[k] for k in free])

    # -- starting point, as the original fit: median, SD, a margin outside
    # the data; then any user guesses; then clipped into the valid region --
    g = {'lower': lo_cap - 0.05 * span, 'upper': hi_cap + 0.05 * span,
         'mode': float(np.median(data)), 'sigma': float(np.std(data))}
    for k, v in zip(PARAMS, (gmode, gsigma, glower, gupper)):
        if v is not None:
            g[k] = float(v)
    g.update(fixed)
    g['lower'] = min(g['lower'], lo_cap - 1e-3 * span)
    g['upper'] = max(g['upper'], hi_cap + 1e-3 * span)
    inner = 1e-3 * (g['upper'] - g['lower'])
    g['mode'] = float(np.clip(g['mode'], g['lower'] + inner,
                              g['upper'] - inner))
    g['sigma'] = g['sigma'] if g['sigma'] > 0 else 0.1 * span
    theta0 = to_theta(tuple(g[k] for k in PARAMS))

    # Caps on the unconstrained coordinates. A model the data cannot support
    # (e.g. a fixed mode outside the data) otherwise runs sigma or a bound
    # off to 1e100+, where every evaluation is slow and the result
    # meaningless. The caps are far outside any sensible fit: sigma at
    # 1000x, and a bound's distance from the data at 10000x, the data range.
    # Ending within a factor of 10 of one (sigma above 100x, a bound more
    # than 1000x the data range from the data) is reported as a failure: the
    # data do not identify that parameter.
    cap = {'sigma': np.log(1e3 * span), 'lower': np.log(1e4 * span),
           'upper': np.log(1e4 * span), 'mode': np.inf}
    theta_cap = np.array([cap[k] for k in free])
    theta0 = np.minimum(theta0, theta_cap - 1.0)

    # The other way a bound can go unidentified: collapsing onto the data.
    # A free bound is kept strictly outside the data (and a fixed mode), but
    # the optimum can still be the edge of that region: the fit improves all
    # the way as the bound approaches the sample extreme, and the optimizer
    # stops a hair outside it. The data then do not place the bound beyond
    # the extreme, and the delta-method covariance is zero by construction
    # (d bound / d theta = exp(theta) -> 0), a spurious standard error of
    # ~0. The test is whether the fit is an interior optimum in that bound:
    # moving the bound COLLAPSE_FACTOR times closer to the edge (theta down
    # by log COLLAPSE_FACTOR, the other parameters held) must make the
    # objective worse by more than rounding. A distance threshold would not
    # do: a genuine gap can be tiny (G4b: the true lower bound 1e-6 below a
    # fixed mode at 1e-6, which MLE recovers), while a collapsed one is tiny
    # too. See docs/v0.2_tails_and_fit_bounds.md, section 2.1.
    edge = {'lower': ('min(data)' if lo_cap == dmin else 'the fixed mode'),
            'upper': ('max(data)' if hi_cap == dmax else 'the fixed mode')}

    def shapes(p):
        return dist.bounds_to_shapes(*p)

    def nnlf(p):
        c, s, loc, scale = shapes(p)
        lp = dist.logpdf(data, c, s, loc, scale)
        return float(-np.sum(lp)) if np.all(np.isfinite(lp)) else np.inf

    def penalized(fn, bad):
        # An extreme trial point can still round into an invalid set (e.g.
        # mode == lower at 1e9 offsets); score it as the worst value rather
        # than abort the optimizer. The final fit is validated below.
        def wrapped(theta):
            try:
                return fn(theta)
            except ValueError:
                return bad
        return wrapped

    x_e, F_e = _ecdf(data, ecdf)

    def residuals_b(p):
        c, s, loc, scale = shapes(p)
        return np.asarray(dist.cdf(x_e, c, s, loc, scale), float) - F_e

    with warnings.catch_warnings():
        # Trial points may pass through the resolution warning's range;
        # the final fit is checked once, outside this block.
        warnings.simplefilter('ignore', RuntimeWarning)
        if method == 'MLE':
            def objective(theta):
                theta = np.minimum(theta, theta_cap)
                v = nnlf(to_bounds(theta))
                return v if np.isfinite(v) else 1e300
            opt = minimize(penalized(objective, 1e300), theta0,
                           method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-10,
                                    'maxiter': 2000 * len(free),
                                    'maxfev': 4000 * len(free)})
            theta = np.minimum(opt.x, theta_cap)
        else:
            opt = least_squares(
                penalized(lambda th: residuals_b(to_bounds(th)),
                          np.ones(x_e.size)),
                theta0, bounds=(-np.inf, theta_cap),
                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300 * len(free))
            theta = opt.x
        at_cap = [k for k, t, c in zip(free, theta, theta_cap)
                  if np.isfinite(c) and t > c - np.log(10)]
        if method == 'MLE':
            obj = penalized(objective, 1e300)
        else:
            res = penalized(lambda th: residuals_b(to_bounds(th)), None)

            def obj(th):
                r = res(th)
                return np.inf if r is None else float(r @ r)
        f_hat = obj(theta)
        collapsed = []
        for j, k in enumerate(free):
            if k not in ('lower', 'upper'):
                continue
            th = theta.copy()
            th[j] -= np.log(COLLAPSE_FACTOR)
            f_in = obj(th)
            if np.isfinite(f_in) and \
                    f_in <= f_hat + COLLAPSE_RTOL * max(1.0, abs(f_hat)):
                collapsed.append(k)
        p_hat = tuple(float(v) for v in to_bounds(theta))
        cov = _lsq_cov(lambda th: residuals_b(to_bounds(th)), to_bounds,
                       theta, free) if method == 'LSQ' else None

    shapes(p_hat)          # raises for an invalid fit; warns at the extremes
    if cov is not None:
        # No standard error for a collapsed bound; the remaining entries
        # treat it as fixed at the data's edge (its Jacobian column is ~0).
        for k in collapsed:
            i = PARAMS.index(k)
            cov[i, :] = np.nan
            cov[:, i] = np.nan
    out = BoundsFit(*p_hat)
    out.method = method
    out.free = free
    out.cov = cov
    out.stderr = (BoundsParams(*np.sqrt(np.clip(np.diag(cov), 0, None)))
                  if cov is not None else None)
    out.nnlf = nnlf(p_hat)
    out.success = bool(opt.success) and not at_cap and not collapsed
    msgs = []
    if at_cap:
        msgs.append(f"{', '.join(at_cap)} ran away toward its cap "
                    f"(sigma above 100x, or a bound more than 1000x the "
                    f"data range from the data): the data do not identify "
                    f"it, or a fixed value is inconsistent with them")
    for k in collapsed:
        msgs.append(f"{k} collapsed onto {edge[k]}: the fit does not get "
                    f"worse as the bound moves onto it, so the data do not "
                    f"place the {k} bound beyond it; its standard error is "
                    f"undefined (NaN)")
    out.message = "; ".join(msgs) if msgs else str(opt.message)
    out.optimizer = opt
    if not out.success:
        what = ("did not converge" if not opt.success
                else "did not identify every parameter")
        warnings.warn(f"fit_bounds ({method}) {what}: {out.message}",
                      RuntimeWarning, stacklevel=3)
    return out


def _lsq_cov(residuals_t, to_bounds, theta, free):
    """
    curve_fit's covariance, in the bounds parametrization: s^2 (J^T J)^-1
    with J the Jacobian of the residuals with respect to the free bounds
    parameters and s^2 = SSR / (m - p).

    This is what scipy.optimize.curve_fit returns as pcov with its default
    absolute_sigma=False, which is what the original notebook's fit()
    exposed. It treats the ECDF residuals as independent, which they are
    not, so it understates the real sampling uncertainty; a bootstrap is the
    reliable alternative (see fit_bounds).

    The derivatives are taken in the unconstrained coordinates theta, where
    every step is a valid parameter set (a step in a bound itself can cross
    the data or a fixed mode), then carried over exactly by the chain rule:
    with b = to_bounds(theta) and H = db/dtheta, J_theta = J_b H, so
    s^2 (J_b^T J_b)^-1 = H [s^2 (J_theta^T J_theta)^-1] H^T.
    """
    idx = [PARAMS.index(k) for k in free]
    theta = np.asarray(theta, dtype=float)
    r0 = residuals_t(theta)
    p = len(free)
    J = np.empty((r0.size, p))
    H = np.empty((p, p))
    for j in range(p):
        # Central differences with a step ~ eps^(1/3) (the cdf is accurate
        # to ~1e-13) in coordinates that are logs or logits of order 1.
        h = 1e-4 * max(1.0, abs(theta[j]))
        hi, lo = theta.copy(), theta.copy()
        hi[j] += h
        lo[j] -= h
        J[:, j] = (residuals_t(hi) - residuals_t(lo)) / (2 * h)
        H[:, j] = (np.take(to_bounds(hi), idx)
                   - np.take(to_bounds(lo), idx)) / (2 * h)
    s2 = float(r0 @ r0) / (r0.size - p)
    cov_free = H @ (s2 * np.linalg.pinv(J.T @ J)) @ H.T
    cov = np.zeros((4, 4))
    cov[np.ix_(idx, idx)] = cov_free
    return cov
