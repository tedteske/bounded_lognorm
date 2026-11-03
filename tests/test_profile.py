"""
Suite P (0.3.0): fit_bounds(truncate=...) and profile_bound, on both
families (BL and BL2 share the code, through the bounds bridge).

Checked against independent computations rather than gold numbers: the
conditional likelihood by hand and by quadrature, the profile's ends by
separate fit_bounds calls with the bound held there, and the mirror image of
the data. Coverage itself is a simulation study, too slow for the suite; it
is in docs/phase7_bound_intervals.md.

Mutation-checked (Nov 2026): each change below makes the listed tests fail --
  truncation term's sign flipped (+n log P -> -n log P)   -> P1, P3
  data outside the window not set aside                  -> P1, P2
  two-sided window from cdf(hi) + cdf(lo)                -> P1
  lower-truncated window from logcdf instead of logsf    -> P1, P3
  chi-squared critical value with 2 degrees of freedom   -> P5
  far end reported at the cap instead of inf             -> P7
  search coordinate's sign not flipped for 'lower'       -> P8
  full fit not refitted when a profile point beats it    -> P9

Author: Ted Teske with Claude (Anthropic), 2026.
"""
import functools
import sys
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi2

from bounded_lognorm import bl, bl2
from bounded_lognorm._fit import _log_window

FAMS = {'BL': bl, 'BL2': bl2}
TRUTH = (75, 5, 70, 100)
CRIT95 = float(chi2.ppf(0.95, 1))
# BL2 fits are ~10x slower than BL's (quadrature), so its samples are
# smaller.
N = {'BL': 500, 'BL2': 300}


def quiet(fn):
    @functools.wraps(fn)
    def wrapped(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return fn(*a, **k)
    return wrapped


@functools.cache
def _data(fam, n, seed=1):
    return FAMS[fam].from_bounds(*TRUTH).rvs(size=n, random_state=seed)


@functools.cache
def _unbounded_data():
    """Normal data: nothing above the median says where an upper bound is."""
    return 10.0 + np.random.default_rng(3).standard_normal(1000)


@functools.cache
@quiet
def _profile(fam, **kw):
    x = _data(fam, N[fam])
    return FAMS[fam].profile_bound(x, flower=70, **kw)


# --- fit_bounds(truncate=...) ---------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_P1_conditional_likelihood_by_hand(fam):
    """nnlf is -sum log f(x_i) + n log P(window), over the data inside the
    window only; the window's probability matches quadrature of the pdf."""
    d = FAMS[fam]
    x = _data(fam, N[fam])
    t1, t2 = np.percentile(x, [30, 90])
    for window in ((t1, None), (None, t2), (t1, t2)):
        f = d.fit_bounds(x, flower=70, truncate=window)
        lo = -np.inf if window[0] is None else window[0]
        hi = np.inf if window[1] is None else window[1]
        inside = x[(x > lo) & (x < hi)]
        fr = d.from_bounds(*f)
        p_quad = quad(fr.pdf, max(lo, f.lower), min(hi, f.upper),
                      epsabs=0, epsrel=1e-12, limit=200)[0]
        by_hand = -np.sum(fr.logpdf(inside)) + inside.size * np.log(p_quad)
        assert abs(f.nnlf - by_hand) < 1e-8 * abs(by_hand), (window, f.nnlf,
                                                             by_hand)
        assert f.truncate == (window[0], window[1])
        assert f.n_used == inside.size and f.n_excluded == x.size - inside.size
        args = d.bounds_to_shapes(*f)
        assert abs(_log_window(d, window[0], window[1], args)
                   - np.log(p_quad)) < 1e-9


@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_P2_data_outside_the_window_have_no_influence(fam):
    """Values below the window (e.g. a few very slow runs) leave the
    truncated fit exactly unchanged; they do move the untruncated one."""
    d = FAMS[fam]
    x = _data(fam, N[fam])
    t = float(np.median(x))
    dirty = np.concatenate([x, [70.01, 70.02, 70.05]])
    a = d.fit_bounds(x, flower=70, truncate=(t, None))
    b = d.fit_bounds(dirty, flower=70, truncate=(t, None))
    assert tuple(a) == tuple(b) and a.nnlf == b.nnlf
    assert b.n_excluded == a.n_excluded + 3
    assert d.fit_bounds(dirty, flower=70).upper != d.fit_bounds(
        x, flower=70).upper


@quiet
def test_P3_truncated_fit_recovers_the_truth():
    """Numerical, BL, n = 4000, seed 1: with only the data above the median,
    the profile interval for the upper bound contains the truth (100), and
    the conditional fit's upper piece matches the truth's: its sf, relative
    to the window, agrees with the truth's to 0.02 across the window."""
    x = bl.from_bounds(*TRUTH).rvs(size=4000, random_state=1)
    t = float(np.median(x))
    r = bl.profile_bound(x, flower=70, truncate=(t, None))
    assert r.interval[0] < 100 < r.interval[1], r
    fr, tr = bl.from_bounds(*r.fit), bl.from_bounds(*TRUTH)
    z = np.linspace(t, x.max(), 50)
    gap = np.abs(fr.sf(z) / fr.sf(t) - tr.sf(z) / tr.sf(t))
    assert gap.max() < 0.02, gap.max()


@pytest.mark.parametrize("kwargs", [
    dict(method='LSQ', truncate=(80, None)),
    dict(truncate=(90, 80)),
    dict(truncate=(np.inf, None)),
    dict(truncate=(200, None)),            # nothing inside
    dict(truncate=80),                     # not a pair
], ids=["lsq", "lo-above-hi", "infinite", "empty", "scalar"])
def test_P4_truncate_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        bl.fit_bounds(_data('BL', 500), flower=70, **kwargs)


# --- profile_bound ---------------------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_P5_ends_sit_at_the_chi2_critical_value(fam):
    """At each end, a separate fit with the bound held there has a
    likelihood ratio equal to chi2_1(0.95) = 3.841; just inside it is below,
    just outside above. The interval brackets the estimate, which is the
    full fit's bound, and lies beyond the data."""
    d = FAMS[fam]
    x = _data(fam, N[fam])
    r = _profile(fam)
    assert r.critical == pytest.approx(CRIT95, rel=1e-12)
    lo, hi = r.interval
    assert x.max() < lo < r.estimate < hi < np.inf
    assert r.estimate == r.fit.upper and r.success, r.message

    def lr(b):
        return 2 * (d.fit_bounds(x, flower=70, fupper=b,
                                 gmode=r.fit.mode,
                                 gsigma=r.fit.sigma).nnlf - r.fit.nnlf)

    for end, inward in ((lo, +1), (hi, -1)):
        assert lr(end) == pytest.approx(CRIT95, abs=2e-3), end
        gap = end - x.max()
        assert lr(end + inward * 0.01 * gap) < CRIT95
        assert lr(end - inward * 0.01 * gap) > CRIT95


@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_P6_level_orders_the_intervals(fam):
    """A 99% interval contains the 95% one, which contains an 80% one."""
    i80, i95, i99 = (_profile(fam, level=lv).interval
                     for lv in (0.80, 0.95, 0.99))
    assert i99[0] < i95[0] < i80[0] < i80[1] < i95[1] < i99[1]


@quiet
def test_P7_open_end_when_the_data_cannot_close_it():
    """Normal data, lower bound fixed at 0, only the data above the median:
    the likelihood keeps rising as the upper bound moves away, so the
    estimate and the interval's far end are inf, the near end finite and
    beyond the data."""
    y = _unbounded_data()
    r = bl.profile_bound(y, flower=0, truncate=(float(np.median(y)), None))
    assert r.estimate == np.inf and r.interval[1] == np.inf
    assert y.max() < r.interval[0] < np.inf
    assert 'upper' in r.fit.at_cap


@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_P8_lower_bound_is_the_mirror_image(fam):
    """Both families are symmetric under x -> -x with the bounds swapped,
    so the lower bound of -x has the mirror interval of the upper bound of
    x."""
    x = _data(fam, N[fam])
    up = _profile(fam)
    low = FAMS[fam].profile_bound(-x, 'lower', fupper=-70)
    assert low.estimate == pytest.approx(-up.estimate, rel=1e-6)
    assert low.interval[0] == pytest.approx(-up.interval[1], rel=1e-6)
    assert low.interval[1] == pytest.approx(-up.interval[0], rel=1e-6)


@quiet
def test_P9_full_fit_improved_by_a_better_profile_point(monkeypatch):
    """If the full fit stops short (here: it is replaced by a fit held at a
    bound 5 units off the maximum, which converges but is not the MLE),
    profile points beat it, and it must be refitted from the best of them:
    no LR may be negative, and the estimate and interval must be those of
    an undisturbed run."""
    import bounded_lognorm._profile as prof
    x = _data('BL', 500)
    good = _profile('BL')
    real = prof.fit_bounds
    calls = []

    def stops_short(dist, data, method, **kw):
        calls.append(kw)
        if len(calls) == 1:                       # the full fit
            return real(dist, data, method, **kw, fupper=good.estimate + 5)
        return real(dist, data, method, **kw)

    monkeypatch.setattr(prof, 'fit_bounds', stops_short)
    r = bl.profile_bound(x, flower=70)
    assert np.min(r.profile[1]) > -1e-6
    assert r.fit.nnlf == pytest.approx(good.fit.nnlf, abs=1e-6)
    assert r.estimate == pytest.approx(good.estimate, rel=1e-5)
    assert r.interval == pytest.approx(good.interval, rel=1e-5)


@pytest.mark.parametrize("kwargs", [
    dict(bound='middle'), dict(fupper=100.0), dict(level=1.5),
    dict(level=0.0), dict(method='LSQ'),
], ids=["bound", "profiled-bound-fixed", "level-1.5", "level-0", "lsq"])
def test_P10_profile_bound_rejects_bad_input(kwargs):
    with pytest.raises((ValueError, TypeError)):
        bl.profile_bound(_data('BL', 500), flower=70, **kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-rs"] + sys.argv[1:]))
