# bounded-lognorm

[![PyPI](https://img.shields.io/pypi/v/bounded-lognorm)](https://pypi.org/project/bounded-lognorm/)
[![Python versions](https://img.shields.io/pypi/pyversions/bounded-lognorm)](https://pypi.org/project/bounded-lognorm/)
[![tests](https://github.com/tedteske/bounded_lognorm/actions/workflows/tests.yml/badge.svg)](https://github.com/tedteske/bounded_lognorm/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23897348.svg)](https://doi.org/10.5281/zenodo.23897348)

Bounded lognormal continuous probability distributions for `scipy.stats`.

The **bounded lognormal** is derived from the [lognormal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution). Unlike the normal and lognormal, its domain has both a lower and an upper bound, and it may be symmetric or skewed either way depending on its parameters. That makes it a natural fit for quantities with known limits, such as adult height, task durations with a hard deadline, or sprint times.

It is parametrized the way people tend to describe such quantities: by its **mode** (the most likely value), a spread **sigma**, and the **bounds**.

This package provides two forms of it, each a full `scipy.stats.rv_continuous` subclass on which every scipy method works (`pdf`, `cdf`, `ppf`, `rvs`, `stats`, `moment`, `entropy`, `fit`, freezing, and the rest):

- **`bl2`**, BL2: the smooth (C∞) geometric-mean form. Its CDF has no closed form, so it is computed by quadrature, with a speed/precision dial.
- **`bl`**, BL: the original piecewise form, a lognormal below the mode joined to a mirrored lognormal above it. Everything about it is in closed form, so it is exact and fast, but its density is only C¹ at the mode.

**They are different distributions, not two implementations of one.** With the same parameters their CDFs differ by 0.03–0.05 in typical cases and by up to about 0.26 with the mode near a bound. Pick one deliberately (see [Which one?](#which-one)) and do not mix results from the two.

## Installation

```
pip install bounded-lognorm
```

Requires Python 3.10+, NumPy 1.23.5+ and SciPy 1.9+.

## Quick start

```python
from bounded_lognorm import bl2      # or: from bounded_lognorm import bl

rv = bl2.from_bounds(mode=5, sigma=10, lower=-5, upper=20)   # frozen distribution
rv.pdf(0.0), rv.cdf(0.0), rv.ppf(0.9)
rv.mean(), rv.std(), rv.entropy()
sample = rv.rvs(size=1000, random_state=0)

c, s, loc, scale = bl2.fit(sample)                 # scipy's maximum-likelihood fit
mode, sigma, lower, upper = bl2.to_bounds(c, s, loc, scale)

bl2.from_bounds(mode=0, sigma=1)                   # None bound = unbounded on that side
```

## Fitting in the bounds form

`fit_bounds` fits mode, sigma and the bounds directly, by maximum likelihood or by least squares on the empirical CDF, with any of the four held fixed:

```python
fit = bl2.fit_bounds(sample, flower=-5)            # lower bound known, the rest fitted
mode, sigma, lower, upper = fit

fit = bl2.fit_bounds(sample, 'LSQ', fmode=5)       # least squares, mode held at 5
fit.stderr.upper                                   # LSQ also gives a covariance (see its docstring)
```

The data cannot always pin every parameter down. A fit whose sigma or a bound runs off toward infinity, or (since 0.2.1) whose free bound collapses onto the smallest or largest data value, is reported with `fit.success == False` and a warning rather than returned as if it were an answer. A collapsed bound is the sample extreme, not an estimate beyond it, and its standard error is NaN.

### An interval for a bound (0.3.0)

`profile_bound` gives a profile-likelihood confidence interval for the upper or lower bound, with the same fixed-parameter options as `fit_bounds` (maximum likelihood only):

```python
r = bl2.profile_bound(sample, 'upper', flower=-5)  # 95% by default
r.estimate, r.interval                             # an end the data cannot close is inf
```

`fit_bounds(..., truncate=(t, None))` and `profile_bound(..., truncate=(t, None))` fit by the likelihood conditional on the data above `t`, so that data below it (outliers far from the bound of interest, say) have no influence. A bound is a support parameter, and when the data say little about it the interval can be open on the far side. The critical value is chi-squared(1); how well that is calibrated, by simulation from both families (well at a few hundred observations, short of 95% at 100 when the bound is close to the data), is in [`docs/phase7_bound_intervals.md`](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase7_bound_intervals.md).

## From a standard deviation (0.4.0)

`sigma` is the spread of each lognormal piece before the bounds cut it off, so the distribution's own standard deviation is smaller. When what you have is a standard deviation, a sample's for instance, `from_std` takes it in place of sigma:

```python
rv = bl2.from_std(mode=90, std=3.7, lower=0, upper=100)    # rv.std() == 3.7
sigma = bl2.sigma_from_std(90, 3.7, 0, 100)                 # 4.305 (BL: 4.991)
```

There is no closed form for either family, so sigma is found by root finding, to 1e-13 relative. A standard deviation close to the uniform distribution's, (upper − lower)/√12, has no usable sigma and raises `ValueError`. Details, including why the search stops at sigma = 10¹² × (upper − lower), are in [`docs/v0.4_from_std.md`](https://github.com/tedteske/bounded_lognorm/blob/master/docs/v0.4_from_std.md).

`bl` has the same `from_bounds`, `from_std`, `sigma_from_std`, `fit`, `fit_bounds`, `profile_bound`, `to_bounds` and `bounds_to_shapes`; everything above works with `bl` in place of `bl2`.

## Which one?

Both take the same parameters and share the same API, so the choice is about the shape near the mode and about cost:

- **BL2** (`bl2`) is smooth everywhere. Use it when the density's curvature matters (for example as a likelihood whose derivatives you rely on), or to reproduce work done with `bounded-lognorm` 0.1.0.
- **BL** (`bl`) has closed forms for everything: density, CDF, survival function, quantiles in both tails, moments of any order, and entropy, all accurate to about 1e-13 or better. Its density has a kink in curvature at the mode (it is C¹ there, not C²). It suits Monte Carlo work and anything that needs many quantiles, and it reproduces the original notebook's BL (with the fixes listed in [ATTRIBUTION.md](https://github.com/tedteske/bounded_lognorm/blob/master/ATTRIBUTION.md)).

Results from one do not carry over to the other.

## Definitions

Let `f_L(x; m, σ)` be the lognormal density written in terms of its mode `m` and standard deviation `σ` (rather than the usual `μ_N`, `σ_N` of the underlying normal).

**BL2** is the normalized geometric mean of a lognormal rising from the lower bound and a mirrored lognormal falling to the upper bound:

```
g(x) = sqrt( f_L(x - lower;  mode - lower, σ) · f_L(upper - x;  upper - mode, σ) )

pdf(x) = g(x) / Z,   Z = ∫ g(x) dx  over (lower, upper)
```

Its mode is exactly `mode`. `sigma` is the standard deviation of each lognormal factor, not of BL2 itself; the two agree as the bounds move far from the mode.

The CDF has no elementary closed form, so it is computed by Gauss–Legendre quadrature over the region where the density is non-negligible, with the normalizing constant and a quantile grid cached per parameter set.

**BL** joins the two lognormals at the mode instead, scaling the upper one to meet the lower one there:

```
g(x) = f_L(x - lower;  mode - lower, σ)              for x <= mode
g(x) = k · f_L(upper - x;  upper - mode, σ)          for x >  mode,   k = f_L(mode - lower) / f_L(upper - mode)

pdf(x) = g(x) / Z
```

Each piece is a truncated lognormal, so the CDF, quantiles, moments and entropy all follow from the normal CDF. In both distributions `sigma` is the standard deviation of each untruncated lognormal piece, and the mode is exactly `mode`.

## Parametrization

Both distributions are exact location-scale families, so they follow the scipy convention: a standard form on [0, 1] with two shape parameters, moved by `loc` and `scale`.

- `loc = lower`, `scale = upper - lower`
- `c = (mode - lower) / (upper - lower)`, the relative mode, 0 < c < 1 (as in `scipy.stats.triang`)
- `s = sigma / (upper - lower)`, the relative sigma, s > 0

`from_bounds(mode, sigma, lower, upper)` converts and returns a frozen distribution; `bounds_to_shapes` and `to_bounds` convert either way. The native form works like any scipy distribution:

```python
bl2.pdf(5, c=0.4, s=0.4, loc=-5, scale=25)    # same as rv.pdf(5) above
```

`from_bounds` raises `ValueError` for invalid parameters. The native form follows scipy and returns NaN.

## Speed and precision (BL2)

BL2 has a speed/precision dial, set once, at construction:

```python
from bounded_lognorm import bounded_lognorm2_gen

exact = bounded_lognorm2_gen(precision='exact')
rv = exact.from_bounds(mode=5, sigma=10, lower=-5, upper=20)
```

- `'fast'`: plotting and exploration. `ppf` round-trip error about 2e-6.
- `'balanced'` (the default, used by `bl2`): ordinary work and fitting. About 1e-7.
- `'precise'`: about 4e-9.
- `'exact'`: `ppf` by root-finding, about 1e-13, but roughly 300 times slower per quantile. Use it for one-off high-accuracy quantiles, not for inverting thousands at once.

The CDF is accurate to about 1e-13 at every position, and `rvs` uses the quantile grid at every position. Frozen distributions keep their instance's setting.

Far-tail quantiles keep their relative accuracy: below a tail probability of 1e-3, `ppf` (lower tail) and `isf` (upper tail) are inverted in log-probability space, so `rv.isf(1e-20)` is accurate to about 2e-4 of its probability at `'balanced'` (6e-11 at `'exact'`), down to about 1e-30.

BL needs no dial: its closed forms are at full accuracy at no extra cost, with tail quantiles accurate relative to their probability down to about 1e-300. `bounded_lognorm_gen(precision=...)` accepts the same settings, so code written for BL2's dial runs unchanged, but they have no effect.

## Testing

```
pip install -e . --group test     # pip 25.1 or later
pytest
```

The suite checks BL2 against independent 30-, 40- and 50-digit mpmath reference values at every precision setting, and BL against an independent 50-digit model and against the original notebook implementation.

## Use cases

Two notebooks run top to bottom in Google Colab, installing the package from PyPI. Both use BL and BL2 side by side, always labelled:

- [What are the fastest possible 100m sprint times?](https://github.com/tedteske/bounded_lognorm/blob/master/notebooks/Applications_100m_Sprints.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tedteske/bounded_lognorm/blob/master/notebooks/Applications_100m_Sprints.ipynb) Fitting a bounded distribution to Olympic sprint speeds and reading the upper bound as the best possible runner, with `fit_bounds`, bootstrap refits, and (since 0.3.0) profile-likelihood intervals for the bound with `profile_bound`.
- [Monte Carlo with bounded lognormals](https://github.com/tedteske/bounded_lognorm/blob/master/notebooks/Applications_Monte_Carlo.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tedteske/bounded_lognorm/blob/master/notebooks/Applications_Monte_Carlo.ipynb) A schedule roll-up from three-point estimates, and an input whose most likely value sits near a bound: where the bound matters, where BL vs BL2 does and does not.

If you ran Monte Carlo simulations with the original notebook's BL, re-run them: its `rvs` returned draws grouped by side of the mode, which correlates independent variables when they are combined draw by draw (the second notebook shows the effect).

## Background and history

The original bounded lognormal (BL) is piecewise: a lognormal below the mode and a mirrored, rescaled lognormal above it. It has closed-form CDF and quantile functions, but its density is only C¹ at the mode, where the second derivative jumps. BL2 replaces the two pieces with their geometric mean, which is smooth everywhere. The two are different distributions, not two versions of one, so results computed with BL will not reproduce with BL2.

Version 0.1.0 shipped BL2 only. Since 0.2.0 the package also ships BL, ported from the [`bounded_lognorm.ipynb`](https://github.com/tedteske/bounded_lognorm/blob/master/notebooks/bounded_lognorm.ipynb) notebook with the same conventions as BL2. Code written against the notebook's `bounded_lognorm` instance needs one change: `bounded_lognorm.pdf(x, mode, sigma, lower, upper)` becomes `bl.from_bounds(mode, sigma, lower, upper).pdf(x)`; the [Phase 5 note](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase5_bl_port.md) has the full migration guide.

Derivations and verification notes:

- [Phase 1: why the CDF has no elementary closed form](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase1_cdf_derivation.md)
- [Phase 2: math verification and numerical fixes](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase2_math_verification.md)
- [Phase 3: scipy.stats conformance and migration from the old API](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase3_api_conformance.md)
- [Phase 4: packaging](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase4_packaging.md)
- [0.2.0: tail quantiles and bounds-form fitting](https://github.com/tedteske/bounded_lognorm/blob/master/docs/v0.2_tails_and_fit_bounds.md)
- [Phase 5: porting BL into the package](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase5_bl_port.md)
- [Phase 6: use cases, and the 0.2.1 fit_bounds fix](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase6_use_cases.md)
- [Phase 7: an interval for a fitted bound (0.3.0)](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase7_bound_intervals.md)

## Citing

If you use this package in published work, please cite it. GitHub's "Cite this repository" button gives APA and BibTeX from [CITATION.cff](https://github.com/tedteske/bounded_lognorm/blob/master/CITATION.cff). Releases are archived on Zenodo: [10.5281/zenodo.23897348](https://doi.org/10.5281/zenodo.23897348) cites all versions (and resolves to the latest); each version also has its own DOI, listed on that record, for citing the exact release you used (0.2.1: [10.5281/zenodo.23898464](https://doi.org/10.5281/zenodo.23898464); 0.2.0: [10.5281/zenodo.23897349](https://doi.org/10.5281/zenodo.23897349)).

## Attribution and license

The bounded lognormal distributions and their original implementations are by Ted Teske (2020–2024). The 2026 optimization, verification, scipy conformance, BL port and use cases were done by Ted Teske with assistance from Claude (Anthropic). See [ATTRIBUTION.md](https://github.com/tedteske/bounded_lognorm/blob/master/ATTRIBUTION.md) for the detailed breakdown and references.

MIT License. See [LICENSE](https://github.com/tedteske/bounded_lognorm/blob/master/LICENSE).
