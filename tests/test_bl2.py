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
# handle, and the same pair the entropy() docstring flags. They fail SILENTLY
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
    lo_x = float(np.ravel(_anchor.ppf(0.001, *case))[0])
    hi_x = float(np.ravel(_anchor.ppf(0.999, *case))[0])
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

        # A1: pdf matches the definition (after normalization).
        ref_pdf = _reference_density(xs, *case) / Zref
        got_pdf = np.asarray(dist.pdf(xs, *case)).ravel()
        scale = max(ref_pdf.max(), 1e-300)
        # pdf accuracy tracks the cdf tolerance (both set by n_quad via Z).
        _check(f"[{level}] A1 pdf==definition", case,
               np.max(np.abs(got_pdf - ref_pdf)) / scale < max(cdf_tol, 1e-9))

        # A2: pdf integrates to 1 (checked tightly only at 'exact').
        if level == 'exact':
            total = quad(
                lambda t: float(np.asarray(dist.pdf(t, *case)).ravel()[0]),
                lower, upper, points=[mode], limit=800,
                epsabs=1e-13, epsrel=1e-11)[0]
            _check(f"[{level}] A2 pdf integrates to 1", case,
                   abs(total - 1.0) < NORM_TOL_AT_EXACT, f"{total:.12f}")

        # A3: cdf matches adaptive quadrature within this level's tolerance.
        got_cdf = np.asarray(dist.cdf(xs, *case)).ravel()
        err = np.max(np.abs(got_cdf - ref_cdf))
        worst_cdf = max(worst_cdf, err)
        _check(f"[{level}] A3 cdf==quad", case, err < cdf_tol, f"err={err:.1e}")

        # A4: cdf boundary conditions and monotonicity (precision-independent).
        c_lo = np.asarray(dist.cdf(lower, *case)).ravel()[0]
        c_hi = np.asarray(dist.cdf(upper, *case)).ravel()[0]
        dense = np.asarray(
            dist.cdf(np.linspace(lower, upper, 2000), *case)).ravel()
        _check(f"[{level}] A4 cdf bounds+monotone", case,
               abs(c_lo) < 1e-12 and abs(c_hi - 1.0) < 1e-12
               and np.all(np.diff(dense) >= -1e-12))

        # A5: ppf inverts cdf (round-trip) within this level's tolerance.
        q = np.linspace(0.005, 0.995, 100)
        rt = np.asarray(dist.cdf(dist.ppf(q, *case), *case)).ravel()
        _check(f"[{level}] A5 ppf round-trip", case,
               np.max(np.abs(rt - q)) < ppf_tol,
               f"err={np.max(np.abs(rt - q)):.1e}")

        # A6: median consistency.
        med = np.ravel(dist.median(*case))[0]
        med_cdf = np.asarray(dist.cdf(med, *case)).ravel()[0]
        _check(f"[{level}] A6 median cdf=0.5", case,
               abs(med_cdf - 0.5) < max(ppf_tol, 1e-6))

        # A7: rvs lands in support and tracks the cdf (sampling, dial-agnostic).
        r = np.asarray(dist.rvs(*case, size=4000, random_state=0)).ravel()
        qx = np.quantile(r, [0.25, 0.5, 0.75])
        theo = np.asarray(dist.cdf(qx, *case)).ravel()
        _check(f"[{level}] A7 rvs calibrated", case,
               np.all(r >= lower) and np.all(r <= upper)
               and np.max(np.abs(theo - [0.25, 0.5, 0.75])) < 0.05)

        # A8: moments finite, ordered, mean inside support.
        mu, var, sk, ku = dist.stats(*case, moments='mvsk')
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
_ef = np.max(np.abs(np.asarray(_f.cdf(_f.ppf(_q, *_c), *_c)).ravel() - _q))
_ee = np.max(np.abs(np.asarray(_e.cdf(_e.ppf(_q, *_c), *_c)).ravel() - _q))
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
        p_new = np.asarray(ref.pdf(xs, *case)).ravel()
        _check("B1 pdf==original", case,
               np.max(np.abs(p_old - p_new)) / max(p_old.max(), 1e-300) < 1e-6)
        c_old = np.asarray(bl2_old.cdf(xs, *case)).ravel()
        c_new = np.asarray(ref.cdf(xs, *case)).ravel()
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
            mm, vv, ss, kk = dist.stats(*case, moments='mvsk')
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
                err = np.max(np.abs(np.asarray(dist.cdf(xq, *case)).ravel()
                                    - q))
                rt = np.max(np.abs(np.asarray(
                    dist.cdf(dist.ppf(q, *case), *case)).ravel() - q))
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
_m0, _v0 = _d.stats(*_base)
_shift_err = max(abs(float(_d.mean(_base[0] + c, 5, _base[2] + c,
                                   _base[3] + c)) - c - _m0) / np.sqrt(_v0)
                 for c in (-1e4, 1e3, 1e4))
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

_check("D6 infinite bound rejected", None,
       _raises(lambda: _d.pdf(0, 0, 1, -np.inf, np.inf))
       and _raises(lambda: _d.cdf(0, 0, 1, -np.inf, 5)))
_check("D6 invalid params rejected under ignore_checks", None,
       _raises(lambda: _d.pdf(5, 5, -1, 0, 10, ignore_checks=True))
       and _raises(lambda: _d.pdf(5, 0, 1, 0, 10, ignore_checks=True)))
_ac = _d._argcheck
_check("D6 _argcheck encodes the constraints", None,
       bool(_ac(5, 10, -5, 20)) and bool(_ac(5, 1, None, None))
       and not any(bool(_ac(*a)) for a in [(5, -1, -5, 20), (5, 0, -5, 20),
                                           (30, 1, -5, 20), (5, 1, 20, -5),
                                           (5, np.inf, -5, 20)]))
with warnings.catch_warnings(record=True) as _w:
    warnings.simplefilter('always')
    _d.pdf(5, 5, 1e-10, 0, 10)
_check("D6 resolution warning for tiny sigma", None,
       any(issubclass(w.category, RuntimeWarning) for w in _w))

# D7: unbounded (None) moments -- the default-parameter case that used to
# return var(100, 1) = 2.68 -- match the reference and are location-free.
_m, _v, _, _ = GOLD_MOMENTS[(0, 1, -1000, 1000)]
_ok = all(abs(float(_d.var(mo, 1)) - _v) / _v < 1e-10
          and abs(float(_d.mean(mo, 1)) - mo) < 1e-10 for mo in (0, 5, 100))
_check("D7 unbounded moments==gold", None, _ok)


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
