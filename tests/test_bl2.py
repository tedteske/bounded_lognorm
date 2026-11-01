"""
Regression tests for bounded_lognorm.bl2.

Run directly:   python tests/test_bl2.py   (exits non-zero on failure)
Or under pytest: pytest tests/test_bl2.py

Author: Ted Teske with Claude (Anthropic), 2026.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bounded_lognorm.bl2 import (
    bounded_lognorm2_gen,
    lognorm_pdf, lognorm_params,
    _unnorm_pdf, _reference_unnorm,
)

#
# bounded_lognorm2 -- regression tests
#
# Author: Ted Teske with Claude (Anthropic), 2026.
#
# Run this cell after the class-definition cell. Prints a PASS/FAIL table.
#
# WHAT THIS GUARDS
# ----------------
# 1. Invariants that must hold at every precision (definition, normalization,
#    monotonicity, ppf round-trip, moments).
# 2. The PRECISION DIAL's promises: each position must meet the accuracy it
#    advertises (DIAL_TOLERANCES below), and accuracy must not get worse as you
#    move the dial toward 'exact'. If someone retunes _PRECISION_SETTINGS and
#    silently breaks a position, these fail.
# 3. Equivalence with the original implementation (Suite B, auto-skipped unless
#    `bl2_old` is defined).
#
# The suite constructs its OWN instances at known precisions -- it does NOT use
# whatever `bl2` the notebook happens to hold, because the whole point of the
# dial is that accuracy depends on it. Testing absolute accuracy against an
# unknown dial position is how you get a flaky suite.
#
# STRESS CASES: (1, 0.062, 0, 100) and (950, 1, -1000, 1000) are needle-thin
# peaks in wide domains -- the cases the effective-support logic exists to
# handle. They fail SILENTLY
# (wrong numbers, no exception) if that logic is broken. Do not remove them.
#

import numpy as np
from scipy.integrate import quad

# (mode, sigma, lower, upper)
TEST_CASES = [
    (5, 10, -5, 20),        # notebook's running example
    (3, 1, 0, 10),          # from the sigma sweep
    (1, 3, 0, 10),          # mode near lower bound
    (9, 1, 0, 10),          # mode near upper bound
    (75, 5, 70, 100),       # the fit example
    (0, 1, -4, 4),          # symmetric
    (-3, 2, -10, 0),        # negative support
    (1, 0.062, 0, 100),     # STRESS: needle-thin peak, wide domain
    (950, 1, -1000, 1000),  # STRESS: peak far from centre, wide domain
]

# Each dial position must meet these. Set generously above the measured error
# (roughly 3-20x margin) so the suite is not flaky, but tight enough to catch a
# real regression. Measured worst-case errors after the Phase 2 fixes
# (Nov 2026), for reference -- cdf is now at the gold floor everywhere:
#   fast     cdf 7e-14  ppf 2e-6
#   balanced cdf 8e-14  ppf 1e-7
#   precise  cdf 7e-14  ppf 4e-9
#   exact    cdf 8e-14  ppf 9e-14
#                                   (cdf_tol,  ppf_roundtrip_tol)
DIAL_TOLERANCES = {
    'fast':     (2e-7,  2e-5),
    'balanced': (1e-10, 1e-6),
    'precise':  (1e-11, 1e-7),
    'exact':    (1e-11, 1e-10),
}

NORM_TOL_AT_EXACT = 1e-9   # pdf-integrates-to-1, checked at the tight end

_results = []


def _check(name, case, passed, detail=""):
    _results.append((name, case, bool(passed), detail))


def _reference_density(x, mode, sigma, lower, upper):
    """Unnormalized density from the definition (Eq. 3), routed through the
    library's OWN reference function `_reference_unnorm` -- so the suite and the
    library share one definition, with no second copy to drift. Unnormalized on
    purpose: the gold-standard Z in _gold_cdf then comes from independent
    adaptive quadrature, keeping the cdf check a genuine outside check."""
    u, m = upper - lower, mode - lower
    mu1, s1 = lognorm_params(m, sigma)
    mu2, s2 = lognorm_params(u - m, sigma)
    return _reference_unnorm(x, mu1, s1, mu2, s2, lower, u)


def _gold_cdf(case, xs):
    """Normalized CDF via adaptive quadrature. points=[mode] tells QUADPACK
    where the peak is; without it the stress cases are missed."""
    mode, sigma, lower, upper = case
    g = lambda t: float(_reference_density(t, *case)[0])
    Z = quad(g, lower, upper, points=[mode], limit=800,
             epsabs=1e-15, epsrel=1e-13)[0]
    F = np.array([quad(g, lower, xx, points=[mode], limit=800,
                       epsabs=1e-15, epsrel=1e-13)[0] for xx in xs])
    return F / Z, Z


# Precompute the gold standard once per case (expensive; dial-independent).
_anchor = bounded_lognorm2_gen(precision='exact')
_GOLD = {}
for case in TEST_CASES:
    lo_x = float(np.ravel(_anchor.from_bounds(*case).ppf(0.001))[0])
    hi_x = float(np.ravel(_anchor.from_bounds(*case).ppf(0.999))[0])
    xs = np.linspace(lo_x, hi_x, 40)
    F, Z = _gold_cdf(case, xs)
    _GOLD[case] = (xs, F, Z)


# ---------------------------------------------------------------------------
# SUITE A -- invariants + per-position accuracy, at EVERY dial position
# ---------------------------------------------------------------------------
_cdf_worst_by_level = {}

for level in ('fast', 'balanced', 'precise', 'exact'):
    dist = bounded_lognorm2_gen(precision=level)
    cdf_tol, ppf_tol = DIAL_TOLERANCES[level]
    worst_cdf = 0.0

    for case in TEST_CASES:
        mode, sigma, lower, upper = case
        xs, ref_cdf, Zref = _GOLD[case]
        rv = dist.from_bounds(*case)

        # A1: pdf matches the definition (after normalization).
        ref_pdf = _reference_density(xs, *case) / Zref
        got_pdf = np.asarray(rv.pdf(xs)).ravel()
        scale = max(ref_pdf.max(), 1e-300)
        # pdf accuracy tracks the cdf tolerance (both set by n_quad via Z).
        _check(f"[{level}] A1 pdf==definition", case,
               np.max(np.abs(got_pdf - ref_pdf)) / scale < max(cdf_tol, 1e-9))

        # A2: pdf integrates to 1 (checked tightly only at 'exact').
        if level == 'exact':
            total = quad(
                lambda t: float(np.asarray(rv.pdf(t)).ravel()[0]),
                lower, upper, points=[mode], limit=800,
                epsabs=1e-13, epsrel=1e-11)[0]
            _check(f"[{level}] A2 pdf integrates to 1", case,
                   abs(total - 1.0) < NORM_TOL_AT_EXACT, f"{total:.12f}")

        # A3: cdf matches adaptive quadrature within this level's tolerance.
        got_cdf = np.asarray(rv.cdf(xs)).ravel()
        err = np.max(np.abs(got_cdf - ref_cdf))
        worst_cdf = max(worst_cdf, err)
        _check(f"[{level}] A3 cdf==quad", case, err < cdf_tol, f"err={err:.1e}")

        # A4: cdf boundary conditions and monotonicity (precision-independent).
        c_lo = np.asarray(rv.cdf(lower)).ravel()[0]
        c_hi = np.asarray(rv.cdf(upper)).ravel()[0]
        dense = np.asarray(rv.cdf(np.linspace(lower, upper, 2000))).ravel()
        _check(f"[{level}] A4 cdf bounds+monotone", case,
               abs(c_lo) < 1e-12 and abs(c_hi - 1.0) < 1e-12
               and np.all(np.diff(dense) >= -1e-12))

        # A5: ppf inverts cdf (round-trip) within this level's tolerance.
        q = np.linspace(0.005, 0.995, 100)
        rt = np.asarray(rv.cdf(rv.ppf(q))).ravel()
        _check(f"[{level}] A5 ppf round-trip", case,
               np.max(np.abs(rt - q)) < ppf_tol,
               f"err={np.max(np.abs(rt - q)):.1e}")

        # A6: median consistency.
        med = np.ravel(rv.median())[0]
        med_cdf = np.asarray(rv.cdf(med)).ravel()[0]
        _check(f"[{level}] A6 median cdf=0.5", case,
               abs(med_cdf - 0.5) < max(ppf_tol, 1e-6))

        # A7: rvs lands in support and tracks the cdf (sampling, dial-agnostic).
        r = np.asarray(rv.rvs(size=4000, random_state=0)).ravel()
        qx = np.quantile(r, [0.25, 0.5, 0.75])
        theo = np.asarray(rv.cdf(qx)).ravel()
        _check(f"[{level}] A7 rvs calibrated", case,
               np.all(r >= lower) and np.all(r <= upper)
               and np.max(np.abs(theo - [0.25, 0.5, 0.75])) < 0.05)

        # A8: moments finite, ordered, mean inside support.
        mu, var, sk, ku = rv.stats(moments='mvsk')
        _check(f"[{level}] A8 moments sane", case,
               np.all(np.isfinite([mu, var, sk, ku]))
               and var > 0 and lower < mu < upper)

    _cdf_worst_by_level[level] = worst_cdf


# ---------------------------------------------------------------------------
# SUITE C -- the dial actually moves: accuracy improves left -> right
# ---------------------------------------------------------------------------
_order = ['fast', 'balanced', 'precise', 'exact']
# Since the Phase 2 graded panels, every position reaches the gold
# reference's own accuracy (~6e-14, set by quad's epsrel=1e-13), so
# positions may tie there. Differences below this floor are noise, not a
# regression; a position falling ABOVE its neighbour by more still fails.
GOLD_CDF_FLOOR = 1e-13
_mono = all(_cdf_worst_by_level[a] >= _cdf_worst_by_level[b] - GOLD_CDF_FLOOR
            for a, b in zip(_order, _order[1:]))
_check("C1 cdf accuracy monotone in dial", None, _mono,
       " >= ".join(f"{_cdf_worst_by_level[l]:.1e}" for l in _order))

# 'exact' ppf must be dramatically better than 'fast' ppf (the dial's headline).
_f = bounded_lognorm2_gen(precision='fast')
_e = bounded_lognorm2_gen(precision='exact')
_c = (5, 10, -5, 20)
_q = np.linspace(0.01, 0.99, 50)
_rf, _re = _f.from_bounds(*_c), _e.from_bounds(*_c)
_ef = np.max(np.abs(np.asarray(_rf.cdf(_rf.ppf(_q))).ravel() - _q))
_ee = np.max(np.abs(np.asarray(_re.cdf(_re.ppf(_q))).ravel() - _q))
_check("C2 exact ppf >> fast ppf", None, _ee < _ef / 100.0,
       f"fast={_ef:.1e} exact={_ee:.1e}")

# C3: the fused fast path (_unnorm_pdf) equals the readable definition
# (_reference_unnorm) to ~machine precision. Dial-independent -- this is the
# guarantee that the optimization never drifts from Eq. 3. If someone edits the
# fused exponent and gets it subtly wrong, this fails even when every accuracy
# test still passes within tolerance.
#
# Compared only where the density is non-negligible (within ~10 orders of its
# peak). In the far tails the two are NOT bit-identical, and the fused form is
# the better one: it sums halved log-exponents, while the readable form
# multiplies two ~1e-200 densities and underflows to 0 first. That stability
# edge is a feature of the fast path, not a drift from the definition.
_pin_worst = 0.0
for case in TEST_CASES:
    mode, sigma, lower, upper = case
    u, m = upper - lower, mode - lower
    mu1, s1 = lognorm_params(m, sigma)
    mu2, s2 = lognorm_params(u - m, sigma)
    args = (mu1, s1, mu2, s2, lower, u)
    xs = np.linspace(lower, upper, 2000)
    fused = _unnorm_pdf(xs, *args)
    defn = _reference_unnorm(xs, *args)
    mask = defn > 1e-10 * defn.max()
    if np.any(mask):
        rel = np.abs(fused[mask] - defn[mask]) / defn[mask]
        _pin_worst = max(_pin_worst, float(rel.max()))
_check("C3 fast path == definition (Eq.3)", None, _pin_worst < 1e-12,
       f"max rel diff={_pin_worst:.1e}")


# ---------------------------------------------------------------------------
# SUITE B -- equivalence with the original (needs `bl2_old`; else skipped)
# ---------------------------------------------------------------------------
if 'bl2_old' in dir():
    ref = bounded_lognorm2_gen(precision='balanced')
    for case in TEST_CASES:
        xs, _, _ = _GOLD[case]
        p_old = np.asarray(bl2_old.pdf(xs, *case)).ravel()
        p_new = np.asarray(ref.from_bounds(*case).pdf(xs)).ravel()
        _check("B1 pdf==original", case,
               np.max(np.abs(p_old - p_new)) / max(p_old.max(), 1e-300) < 1e-6)
        c_old = np.asarray(bl2_old.cdf(xs, *case)).ravel()
        c_new = np.asarray(ref.from_bounds(*case).cdf(xs)).ravel()
        _check("B2 cdf==original", case,
               np.max(np.abs(c_old - c_new)) < 1e-5)
        # B3 (stats == original) was retired in Phase 2: the original grid-sum
        # moments were wrong (see SUITE D, D1), so equality with them is no
        # longer the goal. D1 checks the moments against a 30-digit reference.
else:
    _check("B (skipped: bl2_old not defined)", None, True)


# ---------------------------------------------------------------------------
# SUITE D -- Phase 2 math fixes (see docs/phase2_math_verification.md)
#
# Gold values below come from an independent 30-digit mpmath model of BL2
# (bisection root of the lognorm_params quartic; tanh-sinh quadrature with
# break points graded toward the mode and both bounds). They do not use any
# of the library's numerics, so these are genuine outside checks.
#
# Mutation-checked (Nov 2026): reverting each fix makes these fail --
#   stable lognorm_params      -> D1, D2, D3, D4
#   graded panels (geometric)  -> D1, D2, D3
#   quadrature moments         -> D1, D5, D7
#   (The equal panels within [a, b] are NOT test-guarded: they improve cdf
#   accuracy ~4x only for peaks narrower than 1e-8 of the coordinate scale,
#   i.e. below the float64 resolution floor where the warning already fires.
#   They are a cheap margin, not a requirement.)
#   _argcheck / resolution warning -> D6
# Infinite bounds and invalid ignore_checks inputs are rejected by TWO
# layers (_argcheck_final and lognorm_params' own validation); those D6
# checks fail only when both are reverted.
# ---------------------------------------------------------------------------
import warnings

# (mode, sigma, lower, upper): (mean, variance, skewness, excess kurtosis)
GOLD_MOMENTS = {
    (5, 10, -5, 20): (5.872972571882167, 19.054849479242379,
                      0.15954213718045105, -0.68478377366183439),
    (1, 3, 0, 10): (2.1396098352486822, 1.8396635083675718,
                    0.79512924305855025, 0.14209096331446325),
    (75, 5, 70, 100): (76.626189882159016, 10.139892202442778,
                       0.60129506294243963, -0.019570325749688087),
    (1, 0.062, 0, 100): (1.0028499746947212, 0.0038107946604830729,
                         0.091946465764579388, 0.0044743328551683669),
    (950, 1, -1000, 1000): (949.98538639650817, 0.99905538019237465,
                            -0.029227481656348268, 0.0003642045631913171),
    (0, 1, -1000, 1000): (0.0, 0.9999910001032486,
                          0.0, -1.0999796004540886e-5),
}

# Edge cases: sigma -> 0, bounds -> infinity, mode at a bound.
# (mode, sigma, lower, upper): [(q, x_q)] with cdf(x_q) = q exactly.
GOLD_EDGE_QUANTILES = {
    (5, 1e-07, 0, 10): [
        (0.01, 4.9999997673652125959), (0.25, 4.9999999325510249804),
        (0.5, 5.0), (0.75, 5.0000000674489750196),
        (0.99, 5.0000002326347874041)],
    (0, 1, -1e8, 1e8): [
        (0.01, -2.3263478740408397891), (0.25, -0.67448975019608151836),
        (0.5, 0.0), (0.75, 0.67448975019608151836),
        (0.99, 2.3263478740408394636)],
    (0.01, 1, 0, 10): [
        (0.01, 0.004345239470979309957), (0.25, 0.099342152853396613298),
        (0.5, 0.29006229483283872137), (0.75, 0.68243148648886028494),
        (0.99, 2.2643742273980705085)],
    (1e-06, 1, 0, 10): [
        (0.01, 0.000075363568448164057126), (0.25, 0.016037531077096011798),
        (0.5, 0.098290690280768484193), (0.75, 0.39478898373188596456),
        (0.99, 2.069617915780585469)],
    (9.999999, 1, 0, 10): [
        (0.01, 7.9303820842294751252), (0.25, 9.6052110162812569493),
        (0.5, 9.9017093097259152033), (0.75, 9.9839624689247608057),
        (0.99, 9.9999246364315713788)],
    (0, 1, -1e7, 3): [
        (0.01, -2.6572394326077586439), (0.25, -0.81573708299496315504),
        (0.5, -0.15887740416937144566), (0.75, 0.42902747846723737558),
        (0.99, 1.5459321163183662469)],
}
# Float64 resolution floor for these cases is ~2e-8 (the +-1e8 bounds case).
EDGE_CDF_TOL = 5e-8

# sigma/mode -> sigma_N (mode = 1).
GOLD_PARAMS = {
    1e-12: 9.9999999999999997989e-13, 1e-08: 9.9999999999999984592e-9,
    1e-4: 0.000099999998250000081146, 1.0: 0.56770116784363766186,
    1e4: 2.1465498943674091943, 1e8: 3.0348583776844012325,
    1e12: 3.7169222224798216248,
}

with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    for level in ('fast', 'balanced', 'precise', 'exact'):
        dist = bounded_lognorm2_gen(precision=level)
        _, ppf_tol = DIAL_TOLERANCES[level]

        # D1: moments match the 30-digit reference (all dial positions).
        for case, (m, v, s, k) in GOLD_MOMENTS.items():
            mm, vv, ss, kk = dist.from_bounds(*case).stats(moments='mvsk')
            em = abs(mm - m) / np.sqrt(v)
            ev = abs(vv - v) / v
            ok = em < 1e-10 and ev < 1e-10 and abs(ss - s) < 1e-10 \
                and abs(kk - k) < 1e-9
            _check(f"[{level}] D1 moments==gold", case, ok,
                   f"mean {em:.1e}sd var {ev:.1e} skew {abs(ss-s):.1e} "
                   f"kurt {abs(kk-k):.1e}")

        # D2: cdf at edge cases matches the reference quantiles.
        # D3: ppf round-trip holds at the edge cases too.
        for case, pairs in GOLD_EDGE_QUANTILES.items():
            q = np.array([p[0] for p in pairs])
            xq = np.array([p[1] for p in pairs])
            try:
                rv = dist.from_bounds(*case)
                err = np.max(np.abs(np.asarray(rv.cdf(xq)).ravel() - q))
                rt = np.max(np.abs(np.asarray(rv.cdf(rv.ppf(q))).ravel() - q))
            except Exception:                            # crash = failure
                err = rt = float('inf')
            _check(f"[{level}] D2 edge cdf==gold", case,
                   err < EDGE_CDF_TOL, f"err={err:.1e}")
            _check(f"[{level}] D3 edge ppf round-trip", case,
                   rt < max(ppf_tol, EDGE_CDF_TOL), f"err={rt:.1e}")

# D4: lognorm_params -- reference values, the quartic itself across 300
# decades of sigma/mode, and scale invariance (depends on sigma/mode only).
_worst = max(abs(float(lognorm_params(1.0, r)[1]) - s) / s
             for r, s in GOLD_PARAMS.items())
_check("D4 lognorm_params==gold", None, _worst < 1e-13, f"rel={_worst:.1e}")
_res = 0.0
for lr in range(-150, 151, 5):
    mu, sN = lognorm_params(1.0, 10.0 ** lr)
    t = float(sN) ** 2                      # g = e^t ; g^4 - g^3 = r^2
    _res = max(_res, abs(3 * t + np.log(np.expm1(t)) - 2 * lr * np.log(10))
               / max(1.0, abs(2 * lr * np.log(10))))
_check("D4 quartic residual, r=1e-150..1e150", None, _res < 1e-13,
       f"rel={_res:.1e}")
_s_ref = float(lognorm_params(1.0, 1.0)[1])
_check("D4 scale invariance", None,
       all(abs(float(lognorm_params(c, c)[1]) - _s_ref) < 1e-14
           for c in (1e-300, 1e-30, 1e30, 1e300)))

# D5: the mean is shift-equivariant: mean(X + c) = mean(X) + c.
_d = bounded_lognorm2_gen()
_base = (75, 5, 70, 100)
_m0, _v0 = _d.from_bounds(*_base).stats()
_shift_err = max(abs(float(_d.from_bounds(_base[0] + c, 5, _base[2] + c,
                                          _base[3] + c).mean()) - c - _m0)
                 / np.sqrt(_v0) for c in (-1e4, 1e3, 1e4))
_check("D5 mean shift-equivariant", None, _shift_err < 1e-10,
       f"err={_shift_err:.1e}sd")

# D6: argument guards.
def _raises(fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fn()
    except ValueError:
        return True
    return False

# Phase 3: the bounds form (from_bounds) raises on bad input, so a bad
# parameter set fails loudly; the native (c, s, loc, scale) API follows the
# scipy contract and returns NaN. Both halves are checked.
_check("D6 infinite bound rejected", None,
       _raises(lambda: _d.from_bounds(0, 1, -np.inf, np.inf))
       and _raises(lambda: _d.from_bounds(0, 1, -np.inf, 5)))
_check("D6 invalid params rejected by from_bounds", None,
       all(_raises(lambda a=a: _d.from_bounds(*a))
           for a in [(5, -1, 0, 10), (0, 1, 0, 10), (5, 0, -5, 20),
                     (30, 1, -5, 20), (5, 1, 20, -5), (5, np.inf, -5, 20)]))
_check("D6 native API returns NaN for invalid shapes", None,
       all(np.isnan(_d.pdf(0.5, *a)) and np.isnan(_d.cdf(0.5, *a))
           for a in [(1.2, 0.1), (0.0, 0.1), (1.0, 0.1), (0.5, 0.0),
                     (0.5, -1.0)]))
_ac = _d._argcheck
_check("D6 _argcheck encodes the constraints", None,
       bool(_ac(*_d.bounds_to_shapes(5, 10, -5, 20)[:2]))
       and bool(_ac(*_d.bounds_to_shapes(5, 1, None, None)[:2]))
       and not any(bool(_ac(*a)) for a in [(1.2, 0.1), (0.0, 0.1),
                                           (1.0, 0.1), (0.5, 0.0),
                                           (0.5, -1.0), (0.5, np.inf),
                                           (np.nan, 0.1)]))
with warnings.catch_warnings(record=True) as _w:
    warnings.simplefilter('always')
    _d.from_bounds(5, 1e-10, 0, 10).pdf(5)
_check("D6 resolution warning for tiny sigma", None,
       any(issubclass(w.category, RuntimeWarning) for w in _w))
with warnings.catch_warnings(record=True) as _w:
    warnings.simplefilter('always')
    _d.from_bounds(1e9 + 5, 1, 1e9, 1e9 + 10)        # s = 0.1, but |loc| huge
_check("D6 resolution warning for large |bounds|", None,
       any(issubclass(w.category, RuntimeWarning) for w in _w))

# D7: unbounded (None) moments -- the default-parameter case that used to
# return var(100, 1) = 2.68 -- match the reference and are location-free.
_m, _v, _, _ = GOLD_MOMENTS[(0, 1, -1000, 1000)]
_ok = all(abs(float(_d.from_bounds(mo, 1).var()) - _v) / _v < 1e-10
          and abs(float(_d.from_bounds(mo, 1).mean()) - mo) < 1e-10
          for mo in (0, 5, 100))
_check("D7 unbounded moments==gold", None, _ok)


# ---------------------------------------------------------------------------
# SUITE E -- Phase 3: scipy.stats conformance (see
# docs/phase3_api_conformance.md)
#
# BL2 is now a standard form on [0, 1] with shapes (c, s), moved by loc/scale.
# Gold values below come from the same kind of independent 40-digit mpmath
# model as Suite D (bisection root of the quartic, tanh-sinh quadrature over
# graded break points), in the original (mode, sigma, lower, upper) units.
#
# Mutation-checked (Nov 2026): each change below makes the listed tests fail --
#   drop _updated_ctor_param override        -> E1
#   _param_groups uses only the first (c, s) -> E2
#   old truncated-normal entropy             -> E4
#   _munp removed (n >= 5 back to scipy's
#     generic integration) or returns None   -> E5
#   _sf removed (scipy's 1 - cdf)            -> E6
#   _SUPPORT_TOL back to 1e-14               -> E6
#   _logpdf as log(_pdf)                     -> E7 (deep-tail logpdf)
#   _fitstart removed (scipy's c = 1 start)  -> E8
#   dropping _updated_ctor_param also fails A5, C2 and D3: every frozen
#   distribution silently falls back to 'balanced'.
# E3 (loc/scale equivariance) is a property check rather than a guard on one
# code path: loc/scale are applied by scipy, and any conversion slip in
# bounds_to_shapes already fails the gold suites.
# ---------------------------------------------------------------------------
import pickle

GOLD_ENTROPY = {     # differential entropy, nats
    (5, 10, -5, 20): 2.8642876021546915913,
    (1, 3, 0, 10): 1.6004062429322540248,
    (75, 5, 70, 100): 2.5196395034493657113,
    (1, 0.062, 0, 100): -1.3667309095103259314,
    (950, 1, -1000, 1000): 1.418394742587086973,
    (0, 1, -1000, 1000): 1.418934033233526616,
    (0.01, 1, 0, 10): 0.23580783905491950916,
    (9, 1, 0, 10): 0.98768492198512457026,
}
GOLD_RAW_MOMENTS = {   # E[X^n], n = 5..8
    (5, 10, -5, 20): (76455.785005417687142, 977928.56507218889028,
                      13006148.273331113983, 178536792.08306151163),
    (1, 3, 0, 10): (457.86525344185565187, 2272.0107833022334484,
                    11924.272038127001606, 65493.133381772724067),
    (75, 5, 70, 100): (2688596176.0642796856, 207870734517.8356692,
                       16102008322984.547693, 1249677556253449.8406),
    (-3, 2, -10, 0): (-1274.5167823283704734, 6632.964400403766534,
                      -36033.849675616938205, 202981.24212979675297),
    (950, 1, -1000, 1000): (773729989980674.54747, 735036251709071898.44,
                            6.9827833523338170175e+20,
                            6.6335935402823326422e+23),
}
GOLD_SF_TAIL = {       # (x, sf(x)); 9.99 lies past the effective support
    (1, 3, 0, 10): [(9.0, 2.0031432868087585302e-9),
                    (9.9, 6.2657127014251332825e-31),
                    (9.99, 3.9763956983119566365e-66)],
    (5, 10, -5, 20): [(19.0, 9.8417332401968253631e-7),
                      (19.9, 4.2064824920061267135e-17),
                      (19.99, 8.0961973504797929616e-33)],
    (3, 1, 0, 10): [(6.0, 0.00036602174407367985336),
                    (7.0, 5.0004863446488682695e-7),
                    (8.0, 1.6759978387377380373e-12)],
}

with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)

    # E1: freezing keeps the dial, and a frozen rv equals the native call.
    for level in ('fast', 'balanced', 'precise', 'exact'):
        dist = bounded_lognorm2_gen(precision=level)
        fz = dist.from_bounds(5, 10, -5, 20)
        c, s, loc, scale = dist.bounds_to_shapes(5, 10, -5, 20)
        xs = np.linspace(-6, 21, 50)
        q = np.linspace(0.01, 0.99, 9)
        same = (np.array_equal(fz.pdf(xs), dist.pdf(xs, c, s, loc, scale))
                and np.array_equal(fz.cdf(xs), dist.cdf(xs, c, s, loc, scale))
                and np.array_equal(fz.ppf(q), dist.ppf(q, c, s, loc, scale)))
        fz2 = pickle.loads(pickle.dumps(fz))
        _check(f"[{level}] E1 frozen keeps precision", None,
               fz.dist.precision == level and fz2.dist.precision == level
               and dist(c, s).dist.precision == level and same)

    _d = bounded_lognorm2_gen()

    # E2: array-valued shapes -- each column equals its own scalar call, for
    # pdf/cdf/sf/ppf/stats/moment/entropy, with 2-D broadcasting.
    cs = np.array([0.2, 0.5, 0.8, 0.5])          # repeated pair on purpose
    ss = np.array([0.05, 0.1, 0.2, 0.1])
    xg = np.linspace(0.05, 0.95, 7)[:, None]
    ok = True
    for meth in ('pdf', 'logpdf', 'cdf', 'sf', 'ppf'):
        A = getattr(_d, meth)(xg, cs, ss)
        ok &= A.shape == (7, 4) and all(
            np.array_equal(A[:, j], getattr(_d, meth)(xg[:, 0], cs[j], ss[j]))
            for j in range(4))
    for meth, args in (('stats', {'moments': 'mvsk'}), ('entropy', {})):
        A = np.array(getattr(_d, meth)(cs, ss, **args))
        B = np.array([getattr(_d, meth)(cs[j], ss[j], **args)
                      for j in range(4)]).T
        ok &= np.array_equal(A, B)
    ok &= np.array_equal(_d.moment(6, cs, ss),
                         [_d.moment(6, cs[j], ss[j]) for j in range(4)])
    r = _d.rvs(cs, ss, size=(2000, 4), random_state=0)
    ok &= r.shape == (2000, 4) and np.all(
        np.abs(np.median(r, axis=0) - _d.median(cs, ss)) < 0.01)
    _check("E2 array-valued shapes == scalar calls", None, ok)

    # E3: loc/scale equivariance -- the cached tables are keyed on (c, s)
    # only, so shifted/rescaled copies must agree to rounding.
    base = (75, 5, 70, 100)
    rb = _d.from_bounds(*base)
    xs = np.linspace(71, 99, 40)
    worst = 0.0
    for k, b0 in ((1.0, 0.0), (1.0, -1e3), (10.0, 0.0), (0.01, 5.0)):
        rk = _d.from_bounds(*(k * v + (b0 if i != 1 else 0.0)
                              for i, v in enumerate(base)))
        worst = max(worst,
                    np.max(np.abs(rk.cdf(k * xs + b0) - rb.cdf(xs))),
                    np.max(np.abs(k * rk.pdf(k * xs + b0) - rb.pdf(xs))
                           / rb.pdf(xs).max()),
                    abs(rk.entropy() - np.log(k) - rb.entropy()))
    _check("E3 loc/scale equivariance", None, worst < 1e-12,
           f"err={worst:.1e}")

    for level in ('fast', 'balanced', 'precise', 'exact'):
        dist = bounded_lognorm2_gen(precision=level)

        # E4: entropy is BL2's own differential entropy (was a truncated-
        # normal approximation through Phase 2).
        for case, H in GOLD_ENTROPY.items():
            e = abs(float(dist.from_bounds(*case).entropy()) - H)
            _check(f"[{level}] E4 entropy==gold", case, e < 1e-10,
                   f"err={e:.1e}")

        # E5: moment(n) for n = 5..8 (was None for n >= 5).
        for case, vals in GOLD_RAW_MOMENTS.items():
            rv = dist.from_bounds(*case)
            e = max(abs(float(rv.moment(n)) - v) / abs(v)
                    for n, v in zip((5, 6, 7, 8), vals))
            _check(f"[{level}] E5 moment(5..8)==gold", case, e < 1e-10,
                   f"rel={e:.1e}")

    # E5b: moment(n) agrees with stats() for n = 1..4 and with direct
    # quadrature (expect) for large n.
    rv = _d.from_bounds(75, 5, 70, 100)
    mu, var, sk, ku = rv.stats(moments='mvsk')
    m3c = sk * var ** 1.5
    m4c = (ku + 3) * var ** 2
    ok = (abs(rv.moment(1) - mu) < 1e-12 * abs(mu)
          and abs(rv.moment(2) - (var + mu ** 2)) < 1e-12 * rv.moment(2)
          and abs(rv.moment(3) - (m3c + 3 * mu * var + mu ** 3))
          < 1e-12 * rv.moment(3)
          and abs(rv.moment(4) - (m4c + 4 * mu * m3c + 6 * mu ** 2 * var
                                  + mu ** 4)) < 1e-12 * rv.moment(4))
    ok &= all(abs(rv.moment(n) / rv.expect(lambda x, n=n: x ** n) - 1)
              < 1e-9 for n in (10, 20))
    _check("E5 moment(n) consistent with stats/expect", None, ok)

    # E6: sf keeps relative accuracy deep into the upper tail (1 - cdf
    # bottoms out near 1e-16), down to the effective-support cutoff
    # (_SUPPORT_TOL = 1e-40 of the peak density); past it sf is 0.
    for level in ('fast', 'balanced', 'precise', 'exact'):
        dist = bounded_lognorm2_gen(precision=level)
        for case, pairs in GOLD_SF_TAIL.items():
            rv = dist.from_bounds(*case)
            ok, worst = True, 0.0
            for x, ref in pairs:
                got = float(rv.sf(x))
                if ref > 1e-35:
                    worst = max(worst, abs(got - ref) / ref)
                else:
                    ok &= got < 1e-40
            _check(f"[{level}] E6 sf upper tail==gold", case,
                   ok and worst < 1e-11, f"rel={worst:.1e}")

    # E7: logpdf -- the direct log form equals log of the fused pdf where
    # both are well defined, and stays finite where pdf underflows to 0.
    _lw = 0.0
    for case in TEST_CASES:
        mode, sigma, lower, upper = case
        rv = _d.from_bounds(*case)
        xs = np.linspace(lower, upper, 2001)[1:-1]
        p = rv.pdf(xs)
        lp = rv.logpdf(xs)
        m = p > 1e-250
        _lw = max(_lw, float(np.max(np.abs(lp[m] - np.log(p[m]))
                                    / np.maximum(1, np.abs(lp[m])))))
    _deep = _d.from_bounds(1, 0.062, 0, 100).logpdf([60.0, 99.0])
    _check("E7 logpdf == log(pdf), finite in deep tail", None,
           _lw < 1e-13 and np.all(np.isfinite(_deep))
           and np.all(_d.from_bounds(1, 0.062, 0, 100).pdf([60.0, 99.0])
                      == 0), f"rel={_lw:.1e}")

    # E8: fit -- scipy's generic MLE now works. With the bounds fixed it
    # recovers (c, s); with all four free it reaches a likelihood at least as
    # good as the true parameters' (that is what an MLE must do).
    true = (75, 5, 70, 100)
    data = _d.from_bounds(*true).rvs(size=3000, random_state=1)
    c0, s0, loc0, sc0 = _d.bounds_to_shapes(*true)
    try:
        c2, s2, _, _ = _d.fit(data, floc=loc0, fscale=sc0)
        ok2 = abs(c2 - c0) < 0.01 and abs(s2 - s0) / s0 < 0.05
        msg2 = f"c={c2:.4f}/{c0:.4f} s={s2:.4f}/{s0:.4f}"
    except Exception as exc:                         # crash = failure
        ok2, msg2 = False, repr(exc)[:80]
    try:
        p4 = _d.fit(data)
        ok4 = (bool(_d._argcheck(*p4[:2])) and p4[3] > 0
               and _d.nnlf(p4, data)
               <= _d.nnlf((c0, s0, loc0, sc0), data) + 1e-6)
        msg4 = f"bounds={_d.to_bounds(*p4)}"
    except Exception as exc:
        ok4, msg4 = False, repr(exc)[:80]
    _check("E8 fit (bounds fixed) recovers c, s", None, ok2, msg2)
    _check("E8 fit (all free) is an MLE", None, ok4, msg4)

    # E9: bridge round trip and generic rv_continuous methods.
    rv = _d.from_bounds(5, 10, -5, 20)
    rt = _d.to_bounds(*_d.bounds_to_shapes(5, 10, -5, 20))
    ok = np.allclose(rt, (5, 10, -5, 20), rtol=1e-15, atol=1e-14)
    ok &= rv.support() == (-5.0, 20.0)
    ok &= abs(rv.expect(lambda x: x) - rv.mean()) < 1e-9
    lo_i, hi_i = rv.interval(0.9)
    ok &= abs(rv.cdf(lo_i) - 0.05) < 1e-6 and abs(rv.cdf(hi_i) - 0.95) < 1e-6
    ok &= abs(rv.isf(0.1) - rv.ppf(0.9)) < 1e-12
    ok &= abs(np.exp(rv.logcdf(0.0)) - rv.cdf(0.0)) < 1e-15
    ok &= len(_d._shape_info()) == 2 and _d.numargs == 2
    _check("E9 bridge + generic methods", None, ok)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
_names = sorted({n for n, _, _, _ in _results})
_fail = [r for r in _results if not r[2]]

print(f"{'test':34s} {'passed':>8s}")
print("-" * 46)
for n in _names:
    rows = [r for r in _results if r[0] == n]
    print(f"{n:34s} {sum(r[2] for r in rows):4d}/{len(rows):<4d}")
print("-" * 46)
print(f"{'TOTAL':34s} {sum(r[2] for r in _results):4d}/{len(_results):<4d}")

if _fail:
    print("\nFAILURES:")
    for n, case, _, detail in _fail:
        print(f"  {n}  case={case}  {detail}")
    raise AssertionError(f"{len(_fail)} regression test(s) failed")
print("\nAll regression tests passed.")
