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
# real regression. Measured worst-case errors (Nov 2026), for reference:
#   fast     cdf 5e-8   ppf 6e-6
#   balanced cdf 8e-12  ppf 4e-7
#   precise  cdf 4e-13  ppf 2e-8
#   exact    cdf 8e-14  ppf 7e-13
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
_mono = all(_cdf_worst_by_level[a] >= _cdf_worst_by_level[b] - 1e-15
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
        ok = True
        for fn in ('mean', 'var', 'std', 'entropy'):
            a = float(np.ravel(getattr(bl2_old, fn)(*case))[0])
            b = float(np.ravel(getattr(ref, fn)(*case))[0])
            ok &= abs(a - b) <= 1e-6 * max(abs(a), 1.0)
        _check("B3 stats==original", case, ok)
else:
    _check("B (skipped: bl2_old not defined)", None, True)


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
