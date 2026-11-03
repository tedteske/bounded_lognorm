"""
Profile-likelihood interval for a bound (0.3.0).

A bound of BL or BL2 is a support parameter: the data can only ever sit
inside it. The bootstrap fails for a bound at or near the sample extreme (a
resample never contains a value beyond the data's own), and the LSQ
covariance understates the uncertainty, so neither gives an interval there.
The profile likelihood does not need resamples to reach past the data: at
each trial value of the bound the likelihood is maximized over the other
free parameters, and the interval is the set of values whose likelihood
ratio statistic stays below a critical value.

Uses only fit_bounds (any parameters fixed, an optional truncation window)
and the distribution's public API, so it serves both families.

Calibration, what the chi-squared value does and does not guarantee here,
and the evidence: docs/phase7_bound_intervals.md.

Ted Teske with Claude (Anthropic), 2026.
"""
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

from ._fit import PARAMS, _truncation, fit_bounds

# fit_bounds caps a free bound's distance from the data at 1e4 times the
# data range; an end of the interval still open there is reported as inf.
_FAR = 1e4
# How close to the data the search goes before calling the near end the
# data's edge (only reached if the likelihood stays flat right up to it).
_NEAR = 1e-12
# Outward steps in log(distance from the data), and the root tolerance.
_STEP = np.log(2.0)
_XTOL = 1e-6
# A profile point beating the full fit by more than this (in nnlf) means
# the full fit stopped short; it is refitted from that point.
_REFIT_TOL = 1e-6


class BoundProfile:
    """
    Result of profile_bound.

      bound       : 'upper' or 'lower'
      estimate    : the maximum-likelihood value of the bound (+-inf when
                    the likelihood keeps rising away from the data)
      interval    : (lo, hi) in data units; an end the data cannot close is
                    +-inf
      level, critical, calibration : the confidence level, the critical
                    value of LR used, and how it was obtained
      fit         : the full maximum-likelihood BoundsFit
      profile     : (values, lr), every point evaluated, sorted by value
      lr_boot     : calibration='bootstrap' only, LR at the fitted bound in
                    each replicate
      success, message : False, with the reason, if a fit the interval
                    depends on did not converge or a replicate failed
    """

    def __repr__(self):
        lo, hi = self.interval
        return (f"BoundProfile({self.bound}: estimate {self.estimate!r}, "
                f"{self.level:.0%} interval ({lo!r}, {hi!r}), "
                f"{self.calibration})")


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return fn(*args, **kw)


def profile_bound(dist, data, bound='upper', level=0.95, *,
                  calibration='chi2', n_boot=200, random_state=None,
                  truncate=None, fmode=None, fsigma=None, flower=None,
                  fupper=None, gmode=None, gsigma=None, glower=None,
                  gupper=None, grid=None):
    """See _BoundsBridge.profile_bound for the full description."""
    if bound not in ('upper', 'lower'):
        raise ValueError("bound must be 'upper' or 'lower'")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    if calibration not in ('chi2', 'bootstrap'):
        raise ValueError("calibration must be 'chi2' or 'bootstrap'")
    fixed = dict(mode=fmode, sigma=fsigma, lower=flower, upper=fupper)
    if fixed[bound] is not None:
        raise ValueError(f"the profiled bound must be free: do not pass "
                         f"f{bound}")
    data = np.asarray(data, dtype=float).ravel()
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("data must be a non-empty array of finite values")
    used, _, _ = _truncation(data, truncate)
    if used.size < 2 or not np.ptp(used) > 0:
        raise ValueError("need at least two distinct values inside the "
                         "truncation window")

    fkw = {f'f{k}': v for k, v in fixed.items() if v is not None}
    fkw['truncate'] = truncate
    guess = {f'g{k}': v for k, v in
             dict(mode=gmode, sigma=gsigma, lower=glower, upper=gupper).items()
             if v is not None}
    others = [k for k in PARAMS if k != bound and fixed[k] is None]

    # The bound's distance from the data (and a fixed mode), in logs: the
    # coordinate the search and the root finding work in.
    sign = 1.0 if bound == 'upper' else -1.0
    if bound == 'upper':
        edge = max(float(used.max()), fmode if fmode is not None else -np.inf)
    else:
        edge = min(float(used.min()), fmode if fmode is not None else np.inf)
    span = float(np.ptp(used))
    u_far = np.log(_FAR * span) - np.log(10.0)  # fit_bounds' runaway zone
    u_near = np.log(_NEAR * span)

    def value(u):
        return edge + sign * np.exp(u)

    def coord(b):
        return np.log(sign * (b - edge))

    # -- the full fit, from the default start and any guesses -------------
    full = _quiet(fit_bounds, dist, data, 'MLE', **fkw, **guess)
    messages = []

    def starts_from(p):
        return {f'g{k}': getattr(p, k) for k in others}

    evals = {}                    # u -> nnlf of the profile fit there

    def profile_fit(u):
        # Warm start from the nearest point already evaluated.
        src = full
        if evals:
            keys = np.fromiter(evals, float)
            src = evals[keys[np.argmin(np.abs(keys - u))]][1]
        f = _quiet(fit_bounds, dist, data, 'MLE', **fkw,
                   **{f'f{bound}': value(u)}, **starts_from(src))
        evals[u] = (f.nnlf, f)
        return f

    def refit_from(f):
        nonlocal full
        g = _quiet(fit_bounds, dist, data, 'MLE', **fkw, **starts_from(f),
                   **{f'g{bound}': getattr(f, bound)})
        if g.nnlf < full.nnlf:
            full = g

    def lr(u):
        f = evals[u][1] if u in evals else profile_fit(u)
        if f.nnlf < full.nnlf - _REFIT_TOL:
            refit_from(f)
        return 2.0 * (f.nnlf - full.nnlf)

    def locate():
        """Where the full fit's bound is, in the search coordinate. A bound
        that ran away sits at the far end: the likelihood is still rising
        there."""
        away = bound in full.at_cap
        return away, (u_far if away
                      else min(coord(getattr(full, bound)), u_far))

    # The profile next to the fit should be >= 0. A profile point that
    # beats the full fit means it stopped short; lr() refits it from that
    # point, and the neighbourhood is checked again.
    for _ in range(5):
        before = full.nnlf
        ran_away, u_hat = locate()
        for u in (u_hat, u_hat - _STEP, min(u_hat + _STEP, u_far)):
            lr(u)
        if not full.nnlf < before - _REFIT_TOL:
            break
    ran_away, u_hat = locate()

    if calibration == 'chi2':
        crit = float(chi2.ppf(level, 1))
        lr_boot = None
    else:
        if ran_away:
            raise ValueError(
                f"calibration='bootstrap' needs a finite estimate, but the "
                f"{bound} bound runs away from the data (the likelihood "
                f"keeps rising): the interval is open on that side. Use "
                f"calibration='chi2', or a window with more data.")
        lr_boot, n_fail = _bootstrap_lr(dist, full, used, truncate, fkw,
                                        bound, others, n_boot, random_state)
        crit = float(np.quantile(lr_boot, level))
        if n_fail:
            messages.append(f"{n_fail} of {n_boot} bootstrap replicates had "
                            f"a fit that did not converge or identify every "
                            f"parameter (kept)")

    def crossing(direction):
        """Step from u_hat toward the data (direction -1) or away from it
        (+1) until LR exceeds crit; root-find between the last two steps.
        None if it never does before the near or far end."""
        u0, step = u_hat, _STEP
        while True:
            u1 = u0 + direction * step
            if direction > 0 and u1 >= u_far:
                return None if lr(u_far) <= crit else \
                    brentq(lambda u: lr(u) - crit, u0, u_far, xtol=_XTOL)
            if direction < 0 and u1 <= u_near:
                return None if lr(u_near) <= crit else \
                    brentq(lambda u: lr(u) - crit, u_near, u0, xtol=_XTOL)
            if lr(u1) > crit:
                a, b = sorted((u0, u1))
                return brentq(lambda u: lr(u) - crit, a, b, xtol=_XTOL)
            u0, step = u1, step * 1.5

    for _ in range(3):
        before = full.nnlf
        u_in = crossing(-1)
        u_out = crossing(+1) if not ran_away or lr(u_far) > crit else None
        if not full.nnlf < before - _REFIT_TOL:
            break
        # A better full fit turned up during the search: search again from
        # it (the points already evaluated are reused).
        ran_away, u_hat = locate()
    else:
        messages.append("the full fit kept improving during the search; "
                        "the interval may be off")
    if not full.success and not ran_away:
        messages.append(f"full fit: {full.message}")
    near_end = edge if u_in is None else float(value(u_in))
    far_end = sign * np.inf if u_out is None else float(value(u_out))
    if u_in is None:
        messages.append(f"the likelihood does not fall below the critical "
                        f"value before the {bound} bound reaches the data's "
                        f"edge, which is the interval's near end")

    if grid is not None:
        for b in np.atleast_1d(np.asarray(grid, dtype=float)):
            if sign * (b - edge) > 0:
                lr(float(coord(b)))

    # Final LR values against the (possibly improved) full fit.
    us = np.array(sorted(evals))
    bvals = value(us)
    lrs = np.array([2.0 * (evals[u][0] - full.nnlf) for u in us])
    order = np.argsort(bvals)

    out = BoundProfile()
    out.bound = bound
    out.estimate = sign * np.inf if ran_away else float(getattr(full, bound))
    lo, hi = sorted((near_end, far_end))
    out.interval = (float(lo), float(hi))
    out.level = float(level)
    out.critical = crit
    out.calibration = calibration
    out.fit = full
    out.profile = (bvals[order], lrs[order])
    out.lr_boot = lr_boot
    out.success = not messages
    out.message = "; ".join(messages) if messages else "ok"
    return out


def _bootstrap_lr(dist, full, used, truncate, fkw, bound, others, n_boot,
                  random_state):
    """LR at the fitted bound for n_boot samples drawn from the fitted
    model: same size as the data used, same truncation window, same fixed
    parameters. Returns (array of LR, number of replicates with a failed
    fit)."""
    rng = np.random.default_rng(random_state)
    frozen = dist.from_bounds(*full)
    _, lo, hi = _truncation(used, truncate)
    p_lo = 0.0 if lo is None else float(frozen.cdf(lo))
    p_hi = 1.0 if hi is None else float(frozen.cdf(hi))
    b_hat = getattr(full, bound)
    out, n_fail = [], 0
    for _ in range(int(n_boot)):
        if lo is None and hi is None:
            x = frozen.rvs(size=used.size, random_state=rng)
        else:
            x = frozen.ppf(rng.uniform(p_lo, p_hi, size=used.size))
        g = _quiet(fit_bounds, dist, x, 'MLE', **fkw,
                   **{f'f{bound}': b_hat},
                   **{f'g{k}': getattr(full, k) for k in others})
        f1 = _quiet(fit_bounds, dist, x, 'MLE', **fkw)
        f2 = _quiet(fit_bounds, dist, x, 'MLE', **fkw,
                    **{f'g{k}': getattr(g, k) for k in others},
                    **{f'g{bound}': b_hat})
        f = min((f1, f2), key=lambda r: r.nnlf)
        n_fail += not (g.success and f.success)
        out.append(max(2.0 * (g.nnlf - f.nnlf), 0.0))
    return np.array(out), n_fail
