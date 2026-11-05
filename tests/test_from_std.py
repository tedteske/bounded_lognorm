"""
Suite S (0.4.0): from_std and sigma_from_std, on both families (they share
the code, through the bounds bridge).

Checked against independent computations where there are any: for the
symmetric BL, std and sigma are both explicit in sigma_N (S3), and for BL2
the standard deviation is integrated here from the density's definition with
the quartic for sigma_N solved here (S4), so neither uses the package's own
moments. The rest are round trips and properties (mirror image, location-
scale equivariance), plus the guard the uniqueness of the root rests on: std
increases with sigma over the whole search range (S5), and BL2's does not
beyond it (S6).

Mutation-checked (Nov 2026): each change below makes the listed tests fail --
  root polish skipped (bracket end returned)  -> S1-S4, S9, S10, S12
  root tolerance loosened (xtol 1e-13 -> 1e-6) -> S1-S3, S9, S10, S12
  search cap raised (1e12 -> 1e30)             -> S7
  unattainable std returned as the cap's sigma -> S7, S8
  cap measured from std, not (upper - lower)   -> S7
  gap sign flipped (target - std)              -> S1-S4, S7-S10, S12, S13
  only the first array element solved         -> S10
  std <= 0 accepted (fails later, wrong text)  -> S11
  solved on the other family                  -> S1-S3, S7, S9, S10, S12

Author: Ted Teske with Claude (Anthropic), 2026.
"""
import functools
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import log_ndtr

from bounded_lognorm import bl, bl2, bounded_lognorm2_gen, PRECISION_DIAL

FAMS = {'BL': bl, 'BL2': bl2}
# (mode, sigma, lower, upper): interior, near each bound, narrow and wide,
# far from the origin, and unbounded on one or both sides.
CASES = [
    (5, 1, 0, 10),
    (90, 5, 0, 100),
    (1, 0.062, 0, 100),
    (950, 1, -1000, 1000),
    (0.5, 3, 0, 1),
    (2e-4, 0.3, 0, 1),
    (70, 30, 0, 100),
    (0, 1, None, None),
    (0, 2, None, 3),
    (0, 2, -3, None),
]


def quiet(fn):
    @functools.wraps(fn)
    def wrapped(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return fn(*a, **k)
    return wrapped


def _std(d, mode, sigma, lower, upper):
    return float(d.from_bounds(mode, sigma, lower, upper).std())


# --- round trips ----------------------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("case", CASES, ids=str)
@quiet
def test_S1_from_std_has_the_requested_std(fam, case):
    d = FAMS[fam]
    mode, sigma, lower, upper = case
    target = 0.8 * _std(d, *case)
    rv = d.from_std(mode, target, lower, upper)
    assert isinstance(rv.dist, type(d))
    assert abs(float(rv.std()) / target - 1) < 1e-11


@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("case", CASES, ids=str)
@quiet
def test_S2_sigma_recovered_from_its_std(fam, case):
    d = FAMS[fam]
    mode, sigma, lower, upper = case
    got = d.sigma_from_std(mode, _std(d, *case), lower, upper)
    assert abs(got / sigma - 1) < 1e-11, got


# --- independent references -----------------------------------------------

def _sym_bl(a, sN):
    """Symmetric BL with half-width a, as explicit functions of sigma_N:
    std from the truncated-lognormal partial moments, P(k)/P(1) with
    P(k) = exp(k^2 s^2 / 2) Phi(-k s), and sigma from the quartic
    g^4 - g^3 = (sigma / a)^2 with g = exp(s^2)."""
    def ratio(k):
        return np.exp((k * k - 1) * sN * sN / 2
                      + log_ndtr(-k * sN) - log_ndtr(-sN))
    std = a * np.sqrt(1 - 2 * ratio(2) + ratio(3))
    sigma = a * np.exp(1.5 * sN * sN) * np.sqrt(np.expm1(sN * sN))
    return std, sigma


@pytest.mark.parametrize("sN", [0.05, 0.3, 1.0, 2.0, 3.0])
@quiet
def test_S3_symmetric_bl_against_its_explicit_form(sN):
    """sigma_N = 3 is sigma ~ 7e7 * (upper - lower), deep in the flat part
    where std is closing on the uniform's."""
    a, mode = 3.0, 5.0
    std, sigma = _sym_bl(a, sN)
    got = bl.sigma_from_std(mode, std, mode - a, mode + a)
    assert abs(got / sigma - 1) < 1e-9, (got, sigma)


def _sigmaN2(m, s):
    """sigma_N^2 = t solving 3 t + log(expm1(t)) = 2 log(s / m)."""
    rhs = 2 * np.log(s / m)
    tau = brentq(lambda tau: 3 * np.exp(tau) + np.log(np.expm1(np.exp(tau)))
                 - rhs, -200, 10, xtol=1e-15, rtol=1e-15)
    return np.exp(tau)


def _bl2_std_by_quadrature(c, s):
    """BL2 on [0, 1] from its definition: the density is proportional to
    the geometric mean of the two lognormal factors, each a Gaussian in the
    log of the distance to its bound measured in units of the mode's."""
    t1, t2 = _sigmaN2(c, s), _sigmaN2(1 - c, s)

    def f(y):
        return np.exp(-np.log(y / c) ** 2 / (4 * t1)
                      - np.log((1 - y) / (1 - c)) ** 2 / (4 * t2))
    kw = dict(points=[c], limit=200, epsabs=0, epsrel=1e-13)
    Z = quad(f, 0, 1, **kw)[0]
    m1 = quad(lambda y: y * f(y), 0, 1, **kw)[0] / Z
    m2 = quad(lambda y: (y - m1) ** 2 * f(y), 0, 1, **kw)[0] / Z
    return np.sqrt(m2)


@pytest.mark.parametrize("c, s", [(0.5, 0.2), (0.2, 0.1), (0.05, 0.3)])
@quiet
def test_S4_bl2_against_quadrature_of_its_definition(c, s):
    lower, width = 2.0, 5.0
    std = _bl2_std_by_quadrature(c, s) * width
    got = bl2.sigma_from_std(lower + c * width, std, lower, lower + width)
    assert abs(got / (s * width) - 1) < 1e-9, (got, s * width)


# --- why the root is unique, and why the search stops where it does -------

MONO_C = [1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9, 1 - 1e-9]
MONO_S = np.geomspace(1e-6, 1e12, 73)          # 4 per decade, to the cap


@pytest.mark.parametrize("gen", [bl] + [bounded_lognorm2_gen(precision=p)
                                        for p in PRECISION_DIAL],
                         ids=['BL'] + ['BL2-' + p for p in PRECISION_DIAL])
@quiet
def test_S5_std_increases_with_sigma_up_to_the_cap(gen):
    """The guard behind the unique root: on [0, 1], std strictly rises with
    s = sigma / (upper - lower) from 1e-6 to the search cap 1e12, for modes
    from 1e-12 to 1 - 1e-9 of the way across, at every dial position."""
    for c in MONO_C:
        sd = np.array([gen(c, s).std() for s in MONO_S])
        assert np.all(np.diff(sd) > 0), (c, MONO_S[np.argmin(np.diff(sd))])


@quiet
def test_S6_bl2_turns_down_beyond_the_cap():
    """Not a property of sigma_from_std but the reason for its cap: with the
    mode near a bound, BL2's std rises past the uniform value and then falls
    back (confirmed at 30 digits, docs/v0.4_from_std.md), so far beyond the
    cap a std can come from two sigmas. BL's keeps rising."""
    c = 1e-6
    sd2 = [bl2(c, s).std() for s in (1e12, 1e20, 1e24)]
    assert sd2[0] * np.sqrt(12) > 1            # already past the uniform's
    assert sd2[1] > sd2[0] and sd2[2] < sd2[1]  # then back down
    sd1 = [bl(c, s).std() for s in (1e12, 1e20, 1e24)]
    assert sd1[0] < sd1[1] < sd1[2] < 1 / np.sqrt(12)


@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_S7_attainable_up_to_the_cap_and_no_further(fam):
    d = FAMS[fam]
    c, lower, upper = 1e-6, 0.0, 1.0
    top = d(c, 1e12).std()          # std at the cap, sigma = 1e12 * width
    got = d.sigma_from_std(c, top * (1 - 1e-9), lower, upper)
    assert 1e11 < got <= 1e12
    with pytest.raises(ValueError, match="not attainable"):
        d.sigma_from_std(c, top * (1 + 1e-6), lower, upper)


@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_S8_uniform_std_is_not_attainable(fam):
    """A sample as spread out as a uniform one has no usable sigma: the
    error says so and gives the uniform's std."""
    d = FAMS[fam]
    uniform = 100 / np.sqrt(12)
    with pytest.raises(ValueError, match="uniform") as err:
        d.sigma_from_std(50, uniform, 0, 100)
    assert "%.6g" % uniform in str(err.value)
    with pytest.raises(ValueError, match="not attainable"):
        d.sigma_from_std(90, 40, 0, 100)


# --- unbounded sides, arrays, input checks, the dial ----------------------

@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_S9_unbounded_sides_attain_any_std(fam):
    """With a side unbounded the bound moves out with sigma, so std grows
    without limit: a std far larger than the finite side's distance is
    still found."""
    d = FAMS[fam]
    for lower, upper in [(None, None), (None, 3.0), (-3.0, None)]:
        for target in (1e-3, 1.0, 1e3):
            sigma = d.sigma_from_std(0.0, target, lower, upper)
            assert abs(_std(d, 0.0, sigma, lower, upper) / target - 1) \
                < 1e-11, (lower, upper, target)


@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_S10_arrays_broadcast_elementwise(fam):
    d = FAMS[fam]
    modes = np.array([[10.0], [50.0], [90.0]])
    stds = np.array([[3.0, 7.0]])
    got = d.sigma_from_std(modes, stds, 0, 100)
    assert got.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            assert got[i, j] == d.sigma_from_std(modes[i, 0], stds[0, j],
                                                 0, 100)
    rv = d.from_std(modes, stds, 0, 100)
    assert np.allclose(rv.std(), np.broadcast_to(stds, (3, 2)), rtol=1e-11)
    assert isinstance(d.sigma_from_std(50, 5, 0, 100), float)


@pytest.mark.parametrize("args, match", [
    ((50, 0, 0, 100), "std > 0"), ((50, -1, 0, 100), "std > 0"),
    ((50, [5, -5], 0, 100), "std > 0"),
    ((50, np.nan, 0, 100), "finite"), ((50, np.inf, 0, 100), "finite"),
    ((50, 5, -np.inf, 100), "finite"),
    ((150, 5, 0, 100), "lower < mode < upper"),
    ((0, 5, 0, 100), "lower < mode < upper"),
    ((100, 5, 0, 100), "lower < mode < upper"),
    (([50, 150], 5, 0, 100), "lower < mode < upper"),
])
def test_S11_invalid_input_raises(args, match):
    """Each with its own message: a bad std or mode would otherwise fail
    later, inside from_bounds, with a message about sigma."""
    for d in FAMS.values():
        with pytest.raises(ValueError, match=match):
            d.sigma_from_std(*args)
        with pytest.raises(ValueError, match=match):
            d.from_std(*args)


@quiet
def test_S12_solves_on_the_instance_own_precision():
    fast = bounded_lognorm2_gen(precision='fast')
    rv = fast.from_std(90, 5, 0, 100)
    assert rv.dist.precision == 'fast'
    assert abs(float(rv.std()) / 5 - 1) < 1e-11
    # and BL is not BL2: the same std needs a different sigma
    assert abs(bl.sigma_from_std(90, 5, 0, 100)
               / bl2.sigma_from_std(90, 5, 0, 100) - 1) > 0.05


# --- properties -------------------------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
@quiet
def test_S13_mirror_image_and_equivariance(fam):
    d = FAMS[fam]
    base = d.sigma_from_std(20, 6, 0, 100)
    # x -> -x: mode and bounds mirror, sigma and std unchanged
    assert abs(d.sigma_from_std(-20, 6, -100, 0) / base - 1) < 1e-12
    # x -> a + b x: sigma and std scale by b
    a, b = -7.0, 0.01
    got = d.sigma_from_std(a + 20 * b, 6 * b, a, a + 100 * b)
    assert abs(got / (base * b) - 1) < 1e-11
