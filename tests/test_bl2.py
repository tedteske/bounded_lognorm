"""
Regression tests for the BL2 distribution (bounded_lognorm._bl2).

The tests import the INSTALLED package (src layout), so install it first:
    pip install -e . --group test       (pip >= 25.1)
Then run:       pytest
Or directly:    python tests/test_bl2.py   (runs pytest on this file)

Every check of the pre-pytest script is one pytest item here, parametrized
over dial position and/or case, so a failure names the exact (level, case).
Gold values and tolerances are unchanged from that script.

Author: Ted Teske with Claude (Anthropic), 2026.
"""
import functools
import pickle
import sys
import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from bounded_lognorm._bl2 import (
    bounded_lognorm2_gen,
    lognorm_pdf, lognorm_params,
    _unnorm_pdf, _reference_unnorm,
)

#
# WHAT THIS GUARDS
# ----------------
# 1. Invariants that must hold at every precision (definition, normalization,
#    monotonicity, ppf round-trip, moments).
# 2. The PRECISION DIAL's promises: each position must meet the accuracy it
#    advertises (DIAL_TOLERANCES below), and accuracy must not get worse as you
#    move the dial toward 'exact'. If someone retunes _PRECISION_SETTINGS and
#    silently breaks a position, these fail.
# 3. Equivalence with the original implementation (Suite B, skipped unless a
#    module `bl2_old` is importable).
#
# The suite constructs its OWN instances at known precisions -- it does NOT use
# the package's shared `bl2` instance, because the whole point of the
# dial is that accuracy depends on it. Testing absolute accuracy against an
# unknown dial position is how you get a flaky suite.
#
# STRESS CASES: (1, 0.062, 0, 100) and (950, 1, -1000, 1000) are needle-thin
# peaks in wide domains -- the cases the effective-support logic exists to
# handle. They fail SILENTLY
# (wrong numbers, no exception) if that logic is broken. Do not remove them.
#

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

LEVELS = ('fast', 'balanced', 'precise', 'exact')

# Suites D and E (but not A) ran with RuntimeWarning silenced; keep that.
ignore_runtime_warnings = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _case_id(case):
    return ",".join(str(v) for v in case)


def _level_case_params(cases):
    """(level, case) pairs, ids like 'fast-5,10,-5,20'."""
    return [pytest.param(level, case, id=f"{level}-{_case_id(case)}")
            for level in LEVELS for case in cases]


def _gen(level='balanced'):
    """A fresh instance at a known dial position. The suite never uses a
    shared `bl2`, because accuracy depends on the dial."""
    return bounded_lognorm2_gen(precision=level)


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


@functools.cache
def _gold(case):
    """The gold standard per case (expensive; dial-independent), computed
    once per session on first use."""
    anchor = bounded_lognorm2_gen(precision='exact')
    lo_x = float(np.ravel(anchor.from_bounds(*case).ppf(0.001))[0])
    hi_x = float(np.ravel(anchor.from_bounds(*case).ppf(0.999))[0])
    xs = np.linspace(lo_x, hi_x, 40)
    F, Z = _gold_cdf(case, xs)
    return xs, F, Z


# ---------------------------------------------------------------------------
# SUITE A -- invariants + per-position accuracy, at EVERY dial position
# ---------------------------------------------------------------------------
A_PARAMS = _level_case_params(TEST_CASES)


@pytest.mark.parametrize("level, case", A_PARAMS)
def test_A1_pdf_matches_definition(level, case):
    cdf_tol, _ = DIAL_TOLERANCES[level]
    xs, _, Zref = _gold(case)
    rv = _gen(level).from_bounds(*case)
    ref_pdf = _reference_density(xs, *case) / Zref
    got_pdf = np.asarray(rv.pdf(xs)).ravel()
    scale = max(ref_pdf.max(), 1e-300)
    # pdf accuracy tracks the cdf tolerance (both set by n_quad via Z).
    err = np.max(np.abs(got_pdf - ref_pdf)) / scale
    assert err < max(cdf_tol, 1e-9), f"rel err={err:.1e}"


@pytest.mark.parametrize("case", TEST_CASES,
                         ids=lambda c: "exact-" + _case_id(c))
def test_A2_pdf_integrates_to_one(case):
    """Checked tightly, so only at 'exact'."""
    mode, sigma, lower, upper = case
    rv = _gen('exact').from_bounds(*case)
    total = quad(
        lambda t: float(np.asarray(rv.pdf(t)).ravel()[0]),
        lower, upper, points=[mode], limit=800,
        epsabs=1e-13, epsrel=1e-11)[0]
    assert abs(total - 1.0) < NORM_TOL_AT_EXACT, f"{total:.12f}"


@functools.cache
def _cdf_err(level, case):
    """Max |cdf - gold| over the gold grid. Shared by A3 and C1."""
    xs, ref_cdf, _ = _gold(case)
    rv = _gen(level).from_bounds(*case)
    got_cdf = np.asarray(rv.cdf(xs)).ravel()
    return np.max(np.abs(got_cdf - ref_cdf))


@pytest.mark.parametrize("level, case", A_PARAMS)
def test_A3_cdf_matches_quad(level, case):
    cdf_tol, _ = DIAL_TOLERANCES[level]
    err = _cdf_err(level, case)
    assert err < cdf_tol, f"err={err:.1e}"


@pytest.mark.parametrize("level, case", A_PARAMS)
def test_A4_cdf_bounds_and_monotone(level, case):
    """Precision-independent."""
    mode, sigma, lower, upper = case
    rv = _gen(level).from_bounds(*case)
    c_lo = np.asarray(rv.cdf(lower)).ravel()[0]
    c_hi = np.asarray(rv.cdf(upper)).ravel()[0]
    dense = np.asarray(rv.cdf(np.linspace(lower, upper, 2000))).ravel()
    assert abs(c_lo) < 1e-12 and abs(c_hi - 1.0) < 1e-12 \
        and np.all(np.diff(dense) >= -1e-12), \
        f"cdf(lower)={c_lo:.1e} cdf(upper)-1={c_hi - 1:.1e}"


@pytest.mark.parametrize("level, case", A_PARAMS)
def test_A5_ppf_round_trip(level, case):
    _, ppf_tol = DIAL_TOLERANCES[level]
    rv = _gen(level).from_bounds(*case)
    q = np.linspace(0.005, 0.995, 100)
    rt = np.asarray(rv.cdf(rv.ppf(q))).ravel()
    err = np.max(np.abs(rt - q))
    assert err < ppf_tol, f"err={err:.1e}"


@pytest.mark.parametrize("level, case", A_PARAMS)
def test_A6_median_cdf_is_half(level, case):
    _, ppf_tol = DIAL_TOLERANCES[level]
    rv = _gen(level).from_bounds(*case)
    med = np.ravel(rv.median())[0]
    med_cdf = np.asarray(rv.cdf(med)).ravel()[0]
    assert abs(med_cdf - 0.5) < max(ppf_tol, 1e-6), f"cdf(median)={med_cdf}"


@pytest.mark.parametrize("level, case", A_PARAMS)
def test_A7_rvs_calibrated(level, case):
    """rvs lands in support and tracks the cdf (sampling, dial-agnostic)."""
    mode, sigma, lower, upper = case
    rv = _gen(level).from_bounds(*case)
    r = np.asarray(rv.rvs(size=4000, random_state=0)).ravel()
    qx = np.quantile(r, [0.25, 0.5, 0.75])
    theo = np.asarray(rv.cdf(qx)).ravel()
    assert np.all(r >= lower) and np.all(r <= upper) \
        and np.max(np.abs(theo - [0.25, 0.5, 0.75])) < 0.05, \
        f"cdf at sample quartiles={theo}"


@pytest.mark.parametrize("level, case", A_PARAMS)
def test_A8_moments_sane(level, case):
    """Moments finite, ordered, mean inside support."""
    mode, sigma, lower, upper = case
    rv = _gen(level).from_bounds(*case)
    mu, var, sk, ku = rv.stats(moments='mvsk')
    assert np.all(np.isfinite([mu, var, sk, ku])) \
        and var > 0 and lower < mu < upper, f"mvsk={(mu, var, sk, ku)}"


# ---------------------------------------------------------------------------
# SUITE C -- the dial actually moves: accuracy improves left -> right
# ---------------------------------------------------------------------------
# Since the Phase 2 graded panels, every position reaches the gold
# reference's own accuracy (~6e-14, set by quad's epsrel=1e-13), so
# positions may tie there. Differences below this floor are noise, not a
# regression; a position falling ABOVE its neighbour by more still fails.
GOLD_CDF_FLOOR = 1e-13


def test_C1_cdf_accuracy_monotone_in_dial():
    worst = {level: max(_cdf_err(level, case) for case in TEST_CASES)
             for level in LEVELS}
    _order = list(LEVELS)
    mono = all(worst[a] >= worst[b] - GOLD_CDF_FLOOR
               for a, b in zip(_order, _order[1:]))
    assert mono, " >= ".join(f"{worst[l]:.1e}" for l in _order)


def test_C2_exact_ppf_beats_fast_ppf():
    """'exact' ppf must be dramatically better than 'fast' (the headline)."""
    _f = bounded_lognorm2_gen(precision='fast')
    _e = bounded_lognorm2_gen(precision='exact')
    _c = (5, 10, -5, 20)
    _q = np.linspace(0.01, 0.99, 50)
    _rf, _re = _f.from_bounds(*_c), _e.from_bounds(*_c)
    _ef = np.max(np.abs(np.asarray(_rf.cdf(_rf.ppf(_q))).ravel() - _q))
    _ee = np.max(np.abs(np.asarray(_re.cdf(_re.ppf(_q))).ravel() - _q))
    assert _ee < _ef / 100.0, f"fast={_ef:.1e} exact={_ee:.1e}"


def test_C3_fast_path_equals_definition():
    """The fused fast path (_unnorm_pdf) equals the readable definition
    (_reference_unnorm) to ~machine precision. Dial-independent -- this is
    the guarantee that the optimization never drifts from Eq. 3. If someone
    edits the fused exponent and gets it subtly wrong, this fails even when
    every accuracy test still passes within tolerance.

    Compared only where the density is non-negligible (within ~10 orders of
    its peak). In the far tails the two are NOT bit-identical, and the fused
    form is the better one: it sums halved log-exponents, while the readable
    form multiplies two ~1e-200 densities and underflows to 0 first. That
    stability edge is a feature of the fast path, not a drift from the
    definition.
    """
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
    assert _pin_worst < 1e-12, f"max rel diff={_pin_worst:.1e}"


# ---------------------------------------------------------------------------
# SUITE B -- equivalence with the original. Skipped unless an importable
# module `bl2_old` provides the pre-2026 instance as `bl2_old` (old API:
# bl2_old.pdf(x, mode, sigma, lower, upper)).
# ---------------------------------------------------------------------------
try:
    from bl2_old import bl2_old
except ImportError:
    bl2_old = None

needs_bl2_old = pytest.mark.skipif(bl2_old is None,
                                   reason="bl2_old (original) not importable")


@needs_bl2_old
@pytest.mark.parametrize("case", TEST_CASES, ids=_case_id)
def test_B1_pdf_matches_original(case):
    xs, _, _ = _gold(case)
    p_old = np.asarray(bl2_old.pdf(xs, *case)).ravel()
    p_new = np.asarray(_gen('balanced').from_bounds(*case).pdf(xs)).ravel()
    assert np.max(np.abs(p_old - p_new)) / max(p_old.max(), 1e-300) < 1e-6


@needs_bl2_old
@pytest.mark.parametrize("case", TEST_CASES, ids=_case_id)
def test_B2_cdf_matches_original(case):
    xs, _, _ = _gold(case)
    c_old = np.asarray(bl2_old.cdf(xs, *case)).ravel()
    c_new = np.asarray(_gen('balanced').from_bounds(*case).cdf(xs)).ravel()
    assert np.max(np.abs(c_old - c_new)) < 1e-5

# B3 (stats == original) was retired in Phase 2: the original grid-sum
# moments were wrong (see SUITE D, D1), so equality with them is no longer
# the goal. D1 checks the moments against a 30-digit reference.


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


@ignore_runtime_warnings
@pytest.mark.parametrize("level, case", _level_case_params(GOLD_MOMENTS))
def test_D1_moments_match_gold(level, case):
    """All dial positions."""
    m, v, s, k = GOLD_MOMENTS[case]
    mm, vv, ss, kk = _gen(level).from_bounds(*case).stats(moments='mvsk')
    em = abs(mm - m) / np.sqrt(v)
    ev = abs(vv - v) / v
    ok = em < 1e-10 and ev < 1e-10 and abs(ss - s) < 1e-10 \
        and abs(kk - k) < 1e-9
    assert ok, (f"mean {em:.1e}sd var {ev:.1e} skew {abs(ss-s):.1e} "
                f"kurt {abs(kk-k):.1e}")


@functools.cache
def _edge_errors(level, case):
    """(cdf error at the gold quantiles, ppf round-trip error); a crash
    counts as failure. Shared by D2 and D3."""
    pairs = GOLD_EDGE_QUANTILES[case]
    q = np.array([p[0] for p in pairs])
    xq = np.array([p[1] for p in pairs])
    try:
        rv = _gen(level).from_bounds(*case)
        err = np.max(np.abs(np.asarray(rv.cdf(xq)).ravel() - q))
        rt = np.max(np.abs(np.asarray(rv.cdf(rv.ppf(q))).ravel() - q))
    except Exception:                                    # crash = failure
        err = rt = float('inf')
    return err, rt


@ignore_runtime_warnings
@pytest.mark.parametrize("level, case",
                         _level_case_params(GOLD_EDGE_QUANTILES))
def test_D2_edge_cdf_matches_gold(level, case):
    err, _ = _edge_errors(level, case)
    assert err < EDGE_CDF_TOL, f"err={err:.1e}"


@ignore_runtime_warnings
@pytest.mark.parametrize("level, case",
                         _level_case_params(GOLD_EDGE_QUANTILES))
def test_D3_edge_ppf_round_trip(level, case):
    _, ppf_tol = DIAL_TOLERANCES[level]
    _, rt = _edge_errors(level, case)
    assert rt < max(ppf_tol, EDGE_CDF_TOL), f"err={rt:.1e}"


# D4: lognorm_params -- reference values, the quartic itself across 300
# decades of sigma/mode, and scale invariance (depends on sigma/mode only).
def test_D4_lognorm_params_match_gold():
    _worst = max(abs(float(lognorm_params(1.0, r)[1]) - s) / s
                 for r, s in GOLD_PARAMS.items())
    assert _worst < 1e-13, f"rel={_worst:.1e}"


def test_D4_quartic_residual_over_300_decades():
    _res = 0.0
    for lr in range(-150, 151, 5):
        mu, sN = lognorm_params(1.0, 10.0 ** lr)
        t = float(sN) ** 2                  # g = e^t ; g^4 - g^3 = r^2
        _res = max(_res,
                   abs(3 * t + np.log(np.expm1(t)) - 2 * lr * np.log(10))
                   / max(1.0, abs(2 * lr * np.log(10))))
    assert _res < 1e-13, f"rel={_res:.1e}"


def test_D4_scale_invariance():
    _s_ref = float(lognorm_params(1.0, 1.0)[1])
    assert all(abs(float(lognorm_params(c, c)[1]) - _s_ref) < 1e-14
               for c in (1e-300, 1e-30, 1e30, 1e300))


def test_D5_mean_shift_equivariant():
    """mean(X + c) = mean(X) + c."""
    _d = bounded_lognorm2_gen()
    _base = (75, 5, 70, 100)
    _m0, _v0 = _d.from_bounds(*_base).stats()
    _shift_err = max(
        abs(float(_d.from_bounds(_base[0] + c, 5, _base[2] + c,
                                 _base[3] + c).mean()) - c - _m0)
        / np.sqrt(_v0) for c in (-1e4, 1e3, 1e4))
    assert _shift_err < 1e-10, f"err={_shift_err:.1e}sd"


# D6: argument guards.
# Phase 3: the bounds form (from_bounds) raises on bad input, so a bad
# parameter set fails loudly; the native (c, s, loc, scale) API follows the
# scipy contract and returns NaN. Both halves are checked.
def _raises(fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fn()
    except ValueError:
        return True
    return False


def test_D6_infinite_bound_rejected():
    _d = bounded_lognorm2_gen()
    assert _raises(lambda: _d.from_bounds(0, 1, -np.inf, np.inf)) \
        and _raises(lambda: _d.from_bounds(0, 1, -np.inf, 5))


@pytest.mark.parametrize("args", [(5, -1, 0, 10), (0, 1, 0, 10),
                                  (5, 0, -5, 20), (30, 1, -5, 20),
                                  (5, 1, 20, -5), (5, np.inf, -5, 20)],
                         ids=_case_id)
def test_D6_invalid_params_rejected_by_from_bounds(args):
    _d = bounded_lognorm2_gen()
    assert _raises(lambda: _d.from_bounds(*args))


@pytest.mark.parametrize("args", [(1.2, 0.1), (0.0, 0.1), (1.0, 0.1),
                                  (0.5, 0.0), (0.5, -1.0)], ids=_case_id)
def test_D6_native_api_returns_nan_for_invalid_shapes(args):
    _d = bounded_lognorm2_gen()
    assert np.isnan(_d.pdf(0.5, *args)) and np.isnan(_d.cdf(0.5, *args))


def test_D6_argcheck_encodes_the_constraints():
    _d = bounded_lognorm2_gen()
    _ac = _d._argcheck
    assert bool(_ac(*_d.bounds_to_shapes(5, 10, -5, 20)[:2])) \
        and bool(_ac(*_d.bounds_to_shapes(5, 1, None, None)[:2])) \
        and not any(bool(_ac(*a)) for a in [(1.2, 0.1), (0.0, 0.1),
                                            (1.0, 0.1), (0.5, 0.0),
                                            (0.5, -1.0), (0.5, np.inf),
                                            (np.nan, 0.1)])


def test_D6_resolution_warning_for_tiny_sigma():
    _d = bounded_lognorm2_gen()
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter('always')
        _d.from_bounds(5, 1e-10, 0, 10).pdf(5)
    assert any(issubclass(w.category, RuntimeWarning) for w in _w)


def test_D6_resolution_warning_for_large_bounds():
    _d = bounded_lognorm2_gen()
    with warnings.catch_warnings(record=True) as _w:
        warnings.simplefilter('always')
        _d.from_bounds(1e9 + 5, 1, 1e9, 1e9 + 10)    # s = 0.1, |loc| huge
    assert any(issubclass(w.category, RuntimeWarning) for w in _w)


def test_D7_unbounded_moments_match_gold():
    """Unbounded (None) moments -- the default-parameter case that used to
    return var(100, 1) = 2.68 -- match the reference and are location-free."""
    _d = bounded_lognorm2_gen()
    _m, _v, _, _ = GOLD_MOMENTS[(0, 1, -1000, 1000)]
    assert all(abs(float(_d.from_bounds(mo, 1).var()) - _v) / _v < 1e-10
               and abs(float(_d.from_bounds(mo, 1).mean()) - mo) < 1e-10
               for mo in (0, 5, 100))


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


@ignore_runtime_warnings
@pytest.mark.parametrize("level", LEVELS)
def test_E1_frozen_keeps_precision(level):
    """Freezing keeps the dial, and a frozen rv equals the native call."""
    dist = bounded_lognorm2_gen(precision=level)
    fz = dist.from_bounds(5, 10, -5, 20)
    c, s, loc, scale = dist.bounds_to_shapes(5, 10, -5, 20)
    xs = np.linspace(-6, 21, 50)
    q = np.linspace(0.01, 0.99, 9)
    same = (np.array_equal(fz.pdf(xs), dist.pdf(xs, c, s, loc, scale))
            and np.array_equal(fz.cdf(xs), dist.cdf(xs, c, s, loc, scale))
            and np.array_equal(fz.ppf(q), dist.ppf(q, c, s, loc, scale)))
    fz2 = pickle.loads(pickle.dumps(fz))
    assert fz.dist.precision == level and fz2.dist.precision == level \
        and dist(c, s).dist.precision == level and same


@ignore_runtime_warnings
def test_E2_array_valued_shapes_equal_scalar_calls():
    """Each column equals its own scalar call, for pdf/cdf/sf/ppf/stats/
    moment/entropy, with 2-D broadcasting."""
    _d = bounded_lognorm2_gen()
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
    assert ok


@ignore_runtime_warnings
def test_E3_loc_scale_equivariance():
    """The cached tables are keyed on (c, s) only, so shifted/rescaled
    copies must agree to rounding."""
    _d = bounded_lognorm2_gen()
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
    assert worst < 1e-12, f"err={worst:.1e}"


@ignore_runtime_warnings
@pytest.mark.parametrize("level, case", _level_case_params(GOLD_ENTROPY))
def test_E4_entropy_matches_gold(level, case):
    """BL2's own differential entropy (was a truncated-normal approximation
    through Phase 2)."""
    H = GOLD_ENTROPY[case]
    e = abs(float(_gen(level).from_bounds(*case).entropy()) - H)
    assert e < 1e-10, f"err={e:.1e}"


@ignore_runtime_warnings
@pytest.mark.parametrize("level, case", _level_case_params(GOLD_RAW_MOMENTS))
def test_E5_moments_5_to_8_match_gold(level, case):
    """moment(n) for n = 5..8 (was None for n >= 5)."""
    vals = GOLD_RAW_MOMENTS[case]
    rv = _gen(level).from_bounds(*case)
    e = max(abs(float(rv.moment(n)) - v) / abs(v)
            for n, v in zip((5, 6, 7, 8), vals))
    assert e < 1e-10, f"rel={e:.1e}"


@ignore_runtime_warnings
def test_E5b_moment_consistent_with_stats_and_expect():
    """moment(n) agrees with stats() for n = 1..4 and with direct quadrature
    (expect) for large n."""
    rv = bounded_lognorm2_gen().from_bounds(75, 5, 70, 100)
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
    assert ok


@ignore_runtime_warnings
@pytest.mark.parametrize("level, case", _level_case_params(GOLD_SF_TAIL))
def test_E6_sf_upper_tail_matches_gold(level, case):
    """sf keeps relative accuracy deep into the upper tail (1 - cdf bottoms
    out near 1e-16), down to the effective-support cutoff (_SUPPORT_TOL =
    1e-40 of the peak density); past it sf is 0."""
    rv = _gen(level).from_bounds(*case)
    ok, worst = True, 0.0
    for x, ref in GOLD_SF_TAIL[case]:
        got = float(rv.sf(x))
        if ref > 1e-35:
            worst = max(worst, abs(got - ref) / ref)
        else:
            ok &= got < 1e-40
    assert ok and worst < 1e-11, f"rel={worst:.1e}"


@ignore_runtime_warnings
def test_E7_logpdf_equals_log_pdf_and_finite_in_deep_tail():
    """The direct log form equals log of the fused pdf where both are well
    defined, and stays finite where pdf underflows to 0."""
    _d = bounded_lognorm2_gen()
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
    assert _lw < 1e-13 and np.all(np.isfinite(_deep)) \
        and np.all(_d.from_bounds(1, 0.062, 0, 100).pdf([60.0, 99.0]) == 0), \
        f"rel={_lw:.1e}"


# E8: fit -- scipy's generic MLE now works. With the bounds fixed it
# recovers (c, s); with all four free it reaches a likelihood at least as
# good as the true parameters' (that is what an MLE must do).
E8_TRUE = (75, 5, 70, 100)


def _e8_data(_d):
    return _d.from_bounds(*E8_TRUE).rvs(size=3000, random_state=1)


@ignore_runtime_warnings
def test_E8_fit_bounds_fixed_recovers_c_s():
    _d = bounded_lognorm2_gen()
    data = _e8_data(_d)
    c0, s0, loc0, sc0 = _d.bounds_to_shapes(*E8_TRUE)
    c2, s2, _, _ = _d.fit(data, floc=loc0, fscale=sc0)
    assert abs(c2 - c0) < 0.01 and abs(s2 - s0) / s0 < 0.05, \
        f"c={c2:.4f}/{c0:.4f} s={s2:.4f}/{s0:.4f}"


@ignore_runtime_warnings
def test_E8_fit_all_free_is_an_mle():
    _d = bounded_lognorm2_gen()
    data = _e8_data(_d)
    c0, s0, loc0, sc0 = _d.bounds_to_shapes(*E8_TRUE)
    p4 = _d.fit(data)
    assert (bool(_d._argcheck(*p4[:2])) and p4[3] > 0
            and _d.nnlf(p4, data)
            <= _d.nnlf((c0, s0, loc0, sc0), data) + 1e-6), \
        f"bounds={_d.to_bounds(*p4)}"


@ignore_runtime_warnings
def test_E9_bridge_and_generic_methods():
    """Bridge round trip and generic rv_continuous methods."""
    _d = bounded_lognorm2_gen()
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
    assert ok


# ---------------------------------------------------------------------------
# SUITE F -- 0.2.0: tail quantiles (ppf and isf for q down to 1e-30)
#
# Through 0.1.0, isf was scipy's ppf(1 - q), and the grid positions inverted
# ppf from a spline in q itself: both were wrong by more than 100% below
# q ~ 1e-9. Tail quantiles are now inverted in log-probability space (grid
# positions) or by brentq on the tail's own integral ('exact').
#
# Gold values come from a new independent 40-digit mpmath model of the same
# kind as Suites D and E (bisection root of the quartic, tanh-sinh quadrature
# over break points graded toward the mode and both bounds), checked against
# the Suite D/E gold before use: lognorm_params to 4e-21, sf tail values to
# 7e-19 and an edge quantile to 2e-19. Quantiles were solved by Newton on
# log F and log S at 50 digits.
#
# Error metric: |x - x_gold| * pdf(x_gold) / q, the first-order relative
# error in q. Measured worst over these cases (Nov 2026):
#   fast 3e-3, balanced 2e-4, precise 2e-5, exact 6e-11.
#
# Mutation-checked (Nov 2026): each change below makes the listed tests fail --
#   _isf removed (scipy's ppf(1 - q))          -> F1 at every position
#   _isf_solve as ppf(1 - q) at 'exact'        -> F1 (exact)
#   _ppf_grid without the tail spline          -> F2 (grid positions)
#   tail S computed as 1 - F                   -> F1 (grid positions)
#   abrupt switch instead of the blend         -> F3 (fast)
# ---------------------------------------------------------------------------
GOLD_TAIL_QUANTILES = {   # (q, isf(q), ppf(q)), 40-digit mpmath model
    (5, 10, -5, 20): [
        (1e-3, 17.14099716357759405163, -3.596935953794623744872),
        (1e-6, 18.99793754649532788469, -4.61355808709492570582),
        (1e-9, 19.54945046302352517747, -4.855251816311656722902),
        (1e-12, 19.76933869624205201268, -4.93643087336219358688),
        (1e-16, 19.89308369596746637585, -4.975307604361726713801),
        (1e-20, 19.94553214240324514668, -4.989232470896235148707),
        (1e-25, 19.97430837787739101533, -4.995731185341927709462),
        (1e-30, 19.9869513707980467521, -4.998146900220186813749),
    ],
    (1, 3, 0, 10): [
        (1e-3, 6.916502939203951659028, 0.08231188650815343736407),
        (1e-6, 8.414427760944221602138, 0.01207919278374698375099),
        (1e-9, 9.045505753951378710033, 0.002807095712276413739848),
        (1e-12, 9.375692219974606017602, 0.0008259382733845558948425),
        (1e-16, 9.616311764991982323998, 0.0002023140218737868909328),
        (1e-20, 9.749501460241340453259, 0.00005882595015607486141832),
        (1e-25, 9.844134018468641998489, 0.00001483624147192571314464),
        (1e-30, 9.898332324840636440434, 0.000004281351481718289084631),
    ],
    (3, 1, 0, 10): [
        (1e-3, 5.782747002424193282662, 1.060704621493393883214),
        (1e-6, 6.918829542270133857244, 0.5452501317259286623339),
        (1e-9, 7.58192087787287343426, 0.3277556338515739324827),
        (1e-12, 8.028375088010816230472, 0.2139774879256014684515),
        (1e-16, 8.441316730096204003705, 0.131197838782269645662),
        (1e-20, 8.731893091241203764962, 0.08548302994915297513535),
        (1e-25, 8.992273539008675158228, 0.0530800466358182549125),
        (1e-30, 9.180742078819789925484, 0.03456542359495338227355),
    ],
    (9, 1, 0, 10): [
        (1e-3, 9.852090203980510051418, 5.977439464024890482004),
        (1e-6, 9.95993097622276148308, 4.654344655181778589069),
        (1e-9, 9.985096364877486023095, 3.8410182897389865931),
        (1e-12, 9.993481465405916407852, 3.269277370873043142251),
        (1e-16, 9.997476795510150874318, 2.716748436512615636394),
        (1e-20, 9.998902300998094746932, 2.309752961017998288442),
        (1e-25, 9.999565721976725198642, 1.92800481679525610349),
        (1e-30, 9.999811767608530891961, 1.638517587576972180788),
    ],
    (75, 5, 70, 100): [
        (1e-3, 87.65258364348061758041, 70.73189593406208160421),
        (1e-6, 92.00015608034917404082, 70.19886959940144310103),
        (1e-9, 94.26106737042896726643, 70.07407136540509671439),
        (1e-12, 95.65500266143371307072, 70.03242404767637004835),
        (1e-16, 96.84257150739049205562, 70.01255972436579458376),
        (1e-20, 97.61289304760743317214, 70.00546665560459556668),
        (1e-25, 98.25123488148331100623, 70.00216368377023227321),
        (1e-30, 98.67852128942726843169, 70.00093811906200312944),
    ],
    (1, 0.062, 0, 100): [
        (1e-3, 1.201312861026384937153, 0.8205804283521858739875),
        (1e-6, 1.315013178041464455348, 0.7314898886654940763434),
        (1e-9, 1.402610537613965051018, 0.6690021169581543209427),
        (1e-12, 1.477079116321956843743, 0.6197524239448872623002),
        (1e-16, 1.563907841948014958809, 0.5665192541084216782671),
        (1e-20, 1.641181229223216305263, 0.5227020053615891474361),
        (1e-25, 1.728423249489339697505, 0.4769985708527150958166),
        (1e-30, 1.808033695366894502736, 0.4385595664265351558458),
    ],
}

TAIL_TOL = {'fast': 3e-2, 'balanced': 2e-3, 'precise': 2e-4, 'exact': 1e-9}


def _tail_params():
    return [pytest.param(level, case, id=f"{level}-{_case_id(case)}")
            for level in LEVELS for case in GOLD_TAIL_QUANTILES]


@ignore_runtime_warnings
@pytest.mark.parametrize("level, case", _tail_params())
def test_F1_isf_upper_tail_matches_gold(level, case):
    rv = _gen(level).from_bounds(*case)
    rows = GOLD_TAIL_QUANTILES[case]
    q = np.array([r[0] for r in rows])
    xg = np.array([r[1] for r in rows])
    rel = np.abs(np.asarray(rv.isf(q)) - xg) * rv.pdf(xg) / q
    assert rel.max() < TAIL_TOL[level], \
        "rel. err in q: " + " ".join(f"{qq:.0e}:{e:.1e}" for qq, e in zip(q, rel))


@ignore_runtime_warnings
@pytest.mark.parametrize("level, case", _tail_params())
def test_F2_ppf_lower_tail_matches_gold(level, case):
    rv = _gen(level).from_bounds(*case)
    rows = GOLD_TAIL_QUANTILES[case]
    q = np.array([r[0] for r in rows])
    xg = np.array([r[2] for r in rows])
    rel = np.abs(np.asarray(rv.ppf(q)) - xg) * rv.pdf(xg) / q
    assert rel.max() < TAIL_TOL[level], \
        "rel. err in q: " + " ".join(f"{qq:.0e}:{e:.1e}" for qq, e in zip(q, rel))


@ignore_runtime_warnings
@pytest.mark.parametrize("level", LEVELS)
def test_F3_tail_quantiles_monotone(level):
    """ppf increases and isf decreases over q in [1e-40, 0.5], including
    across the blend between the tail and body splines (1e-4 to 1e-3)."""
    if level == 'exact':
        # brentq everywhere: no blend to probe, and each quantile is a solve.
        q = np.geomspace(1e-40, 0.5, 201)
    else:
        # A coarse sweep plus a dense cluster at each end of the blend: a
        # step backwards there is smaller than the progress between
        # neighbours of the coarse sweep.
        seams = [qs * (1 + np.linspace(-1e-2, 1e-2, 401))
                 for qs in (1e-4, 1e-3)]
        q = np.unique(np.concatenate([np.geomspace(1e-40, 0.5, 4001)]
                                     + seams))
    worst = 0.0
    for case in TEST_CASES:
        rv = _gen(level).from_bounds(*case)
        scale = case[3] - case[2]
        worst = max(worst,
                    -min(np.diff(rv.ppf(q)).min(), 0.0) / scale,
                    -min((-np.diff(rv.isf(q))).min(), 0.0) / scale)
    assert worst == 0.0, f"steps backwards by {worst:.1e} of scale"


@ignore_runtime_warnings
def test_F4_isf_shapes_and_body_agree_with_ppf():
    """Array-valued shapes broadcast like the other methods, and above the
    tails isf(q) is still ppf(1 - q)."""
    _d = bounded_lognorm2_gen()
    cs = np.array([0.2, 0.5, 0.8, 0.5])
    ss = np.array([0.05, 0.1, 0.2, 0.1])
    qg = np.array([[1e-20], [1e-5], [0.2]])
    A = _d.isf(qg, cs, ss)
    ok = A.shape == (3, 4) and all(
        np.array_equal(A[:, j], _d.isf(qg[:, 0], cs[j], ss[j]))
        for j in range(4))
    rv = _d.from_bounds(5, 10, -5, 20)
    qb = np.linspace(0.01, 0.99, 25)
    ok &= np.array_equal(rv.isf(qb), rv.ppf(1 - qb))
    assert ok


# ---------------------------------------------------------------------------
# SUITE G -- 0.2.0: fit_bounds (bounds-form fitting with fixed parameters)
#
# fit_bounds restores the original notebook's fit options: MLE or least
# squares on the empirical cdf, with any of mode, sigma, lower, upper held
# fixed. It is checked against independent fits rather than gold numbers:
# scipy's own MLE (fit) and scipy.optimize.curve_fit.
#
# Mutation-checked (Nov 2026): each change below makes the listed tests fail --
#   MLE objective sign flipped                  -> G1, G2, G4b, G4c
#   ecdf default 'step' instead of 'midpoint'   -> G3, G6
#   covariance s^2 = SSR / m (not m - p)        -> G3
#   free bounds not kept outside a fixed mode   -> G4c (G4b still passes:
#     the penalty on invalid trial points steers the optimizer back)
# ---------------------------------------------------------------------------
from scipy.optimize import curve_fit                     # noqa: E402

G_TRUE = (75, 5, 70, 100)


@functools.cache
def _g_data(n):
    return bounded_lognorm2_gen().from_bounds(*G_TRUE).rvs(
        size=n, random_state=1)


@ignore_runtime_warnings
def test_G1_mle_all_free_matches_native_fit():
    _d = bounded_lognorm2_gen()
    data = _g_data(3000)
    f = _d.fit_bounds(data)
    p4 = _d.fit(data)
    nat = _d.to_bounds(*p4)
    assert f.success and f.method == 'MLE' and f.cov is None
    assert f.nnlf <= _d.nnlf(p4, data) + 1e-6, (f.nnlf, _d.nnlf(p4, data))
    assert np.allclose(tuple(f), nat, rtol=1e-4), (tuple(f), nat)


@ignore_runtime_warnings
def test_G2_mle_bounds_fixed_at_least_as_good_as_native():
    _d = bounded_lognorm2_gen()
    data = _g_data(3000)
    f = _d.fit_bounds(data, flower=70, fupper=100)
    c0, s0, loc0, sc0 = _d.bounds_to_shapes(*G_TRUE)
    c2, s2, _, _ = _d.fit(data, floc=loc0, fscale=sc0)
    nat = _d.to_bounds(c2, s2, loc0, sc0)
    assert (f.lower, f.upper) == (70.0, 100.0) and f.free == ('mode', 'sigma')
    assert f.nnlf <= _d.nnlf((c2, s2, loc0, sc0), data) + 1e-9
    assert np.allclose(tuple(f)[:2], nat[:2], rtol=1e-3), (tuple(f), nat)


@ignore_runtime_warnings
def test_G3_lsq_matches_curve_fit():
    """Same optimum and the same pcov as scipy.optimize.curve_fit on the
    midpoint empirical cdf, in the bounds parametrization."""
    _d = bounded_lognorm2_gen()
    data = _g_data(3000)
    g = _d.fit_bounds(data, 'LSQ', flower=70)
    x, cnt = np.unique(data, return_counts=True)
    F = (np.cumsum(cnt) - cnt / 2) / data.size

    def model(x, mode, sigma, upper):
        return _d.cdf(x, *_d.bounds_to_shapes(mode, sigma, 70, upper))

    popt, pcov = curve_fit(model, x, F, p0=[g.mode, g.sigma, g.upper])
    ours = g.cov[np.ix_([0, 1, 3], [0, 1, 3])]
    rel = np.abs(ours - pcov) / np.sqrt(np.outer(np.diag(pcov),
                                                 np.diag(pcov)))
    assert np.allclose([g.mode, g.sigma, g.upper], popt, rtol=1e-7)
    assert rel.max() < 1e-4, f"cov rel diff {rel.max():.1e}"
    assert np.allclose(g.stderr, np.sqrt(np.diag(g.cov)))


@ignore_runtime_warnings
@pytest.mark.parametrize("fixed", [
    {'flower': 70.0},
    {'fmode': 75.0, 'flower': 70.0},
    {'fsigma': 5.0},
    {'flower': 70.0, 'fupper': 100.0},
], ids=lambda d: "+".join(d))
def test_G4_fixed_parameters_held_exactly(fixed):
    _d = bounded_lognorm2_gen()
    data = _g_data(500)
    g = _d.fit_bounds(data, 'LSQ', **fixed)
    names = {'fmode': 'mode', 'fsigma': 'sigma', 'flower': 'lower',
             'fupper': 'upper'}
    for k, v in fixed.items():
        assert getattr(g, names[k]) == v
        i = ('mode', 'sigma', 'lower', 'upper').index(names[k])
        assert np.all(g.cov[i] == 0) and np.all(g.cov[:, i] == 0)
    assert set(g.free) == {'mode', 'sigma', 'lower', 'upper'} - {
        names[k] for k in fixed}
    assert g.lower < g.mode < g.upper and g.sigma > 0
    assert g.lower < data.min() and g.upper > data.max()


@ignore_runtime_warnings
def test_G4b_fixed_mode_below_all_the_data():
    """A mode almost on the lower bound can lie below every sample. A free
    lower bound must then stay below the fixed mode, not just below the
    data."""
    _d = bounded_lognorm2_gen()
    data = _d.from_bounds(1e-6, 1, 0, 10).rvs(size=200, random_state=2)
    assert data.min() > 1e-6                       # the premise
    for method in ('LSQ', 'MLE'):
        # The upper bound is fixed too: with the mode this close to 0, a
        # sample says almost nothing about it (see G4c).
        g = _d.fit_bounds(data, method, fmode=1e-6, fupper=10)
        assert g.success and g.mode == 1e-6, (method, g, g.message)
        assert g.lower < 1e-6 < data.min() and g.upper > data.max()


def test_G4c_runaway_parameters_reported_as_failure():
    """Two models the data cannot support, each stopped at a cap and
    reported: a fixed mode well below data drawn around 75 (sigma runs
    away), and a free upper bound for a mode almost on the lower bound,
    which a sample barely constrains (this case also once crashed when the
    mode rounded onto the lower bound)."""
    _d = bounded_lognorm2_gen()
    for method in ('LSQ', 'MLE'):
        with pytest.warns(RuntimeWarning, match="sigma ran away"):
            g = _d.fit_bounds(_g_data(500), method, fmode=65.0)
        assert not g.success and g.mode == 65.0
    data = _d.from_bounds(1e-6, 1, 0, 10).rvs(size=200, random_state=2)
    for method in ('LSQ', 'MLE'):
        # (LSQ runs the upper bound away, MLE sigma: either is a failure.)
        with pytest.warns(RuntimeWarning, match="ran away toward its cap"):
            g = _d.fit_bounds(data, method)
        assert not g.success


@ignore_runtime_warnings
def test_G5_lsq_recovers_truth():
    g = bounded_lognorm2_gen().fit_bounds(_g_data(3000), 'LSQ', flower=70)
    assert abs(g.mode - 75) < 0.2 and abs(g.sigma / 5 - 1) < 0.05 \
        and abs(g.upper - 100) < 1.0, tuple(g)


@ignore_runtime_warnings
def test_G6_step_ecdf_is_the_original_and_pulls_upper_down():
    """'step' puts the largest value at ecdf = 1 (as the original fit did),
    which drags the fitted upper bound toward max(data)."""
    from bounded_lognorm._fit import _ecdf
    data = np.array([3.0, 1.0, 2.0, 2.0])
    x, Fs = _ecdf(data, 'step')
    _, Fm = _ecdf(data, 'midpoint')
    assert np.array_equal(x, [1, 2, 3])
    assert np.allclose(Fs, [0.25, 0.75, 1.0])
    assert np.allclose(Fm, [0.125, 0.5, 0.875])
    _d = bounded_lognorm2_gen()
    data = _g_data(500)
    up_m = _d.fit_bounds(data, 'LSQ', flower=70).upper
    up_s = _d.fit_bounds(data, 'LSQ', flower=70, ecdf='step').upper
    assert data.max() < up_s < up_m, (data.max(), up_s, up_m)


@pytest.mark.parametrize("kwargs, data_ok", [
    ({'flower': 72.0}, True),          # fixed lower above some data
    ({'fupper': 80.0}, True),          # fixed upper below some data
    ({'fmode': 60.0, 'flower': 65.0}, True),     # mode outside fixed bounds
    ({'fsigma': 0.0}, True),
    ({'fmode': 75, 'fsigma': 5, 'flower': 70, 'fupper': 100}, True),
    ({'method': 'MM'}, True),
    ({}, False),                       # one distinct value
], ids=["lower-in-data", "upper-in-data", "mode-outside", "sigma-0",
        "all-fixed", "bad-method", "constant-data"])
def test_G7_fit_bounds_rejects_bad_input(kwargs, data_ok):
    _d = bounded_lognorm2_gen()
    data = _g_data(500) if data_ok else np.full(10, 3.0)
    with pytest.raises(ValueError):
        _d.fit_bounds(data, **kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-rs"] + sys.argv[1:]))
