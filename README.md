# bounded-lognorm

A bounded lognormal continuous probability distribution for `scipy.stats`.

The **bounded lognormal** is derived from the [lognormal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution). Unlike the normal and lognormal, its domain has both a lower and an upper bound, and it may be symmetric or skewed either way depending on its parameters. That makes it a natural fit for quantities with known limits, such as adult height, task durations with a hard deadline, or sprint times.

It is parametrized the way people tend to describe such quantities: by its **mode** (the most likely value), a spread **sigma**, and the **bounds**.

This package provides **BL2**, the smooth (C∞) form of the distribution, as a full `scipy.stats.rv_continuous` subclass. Every scipy method works: `pdf`, `cdf`, `ppf`, `rvs`, `stats`, `moment`, `entropy`, `fit`, freezing, and the rest.

## Installation

```
pip install bounded-lognorm
```

Requires Python 3.10+, NumPy 1.23.5+ and SciPy 1.9+.

## Quick start

```python
from bounded_lognorm import bl2

rv = bl2.from_bounds(mode=5, sigma=10, lower=-5, upper=20)   # frozen distribution
rv.pdf(0.0), rv.cdf(0.0), rv.ppf(0.9)
rv.mean(), rv.std(), rv.entropy()
sample = rv.rvs(size=1000, random_state=0)

c, s, loc, scale = bl2.fit(sample)                 # scipy's maximum-likelihood fit
mode, sigma, lower, upper = bl2.to_bounds(c, s, loc, scale)

bl2.from_bounds(mode=0, sigma=1)                   # None bound = unbounded on that side
```

## Definition

Let `f_L(x; m, σ)` be the lognormal density written in terms of its mode `m` and standard deviation `σ` (rather than the usual `μ_N`, `σ_N` of the underlying normal). BL2 is the normalized geometric mean of a lognormal rising from the lower bound and a mirrored lognormal falling to the upper bound:

```
g(x) = sqrt( f_L(x - lower;  mode - lower, σ) · f_L(upper - x;  upper - mode, σ) )

pdf(x) = g(x) / Z,   Z = ∫ g(x) dx  over (lower, upper)
```

Its mode is exactly `mode`. `sigma` is the standard deviation of each lognormal factor, not of BL2 itself; the two agree as the bounds move far from the mode.

The CDF has no elementary closed form, so it is computed by Gauss–Legendre quadrature over the region where the density is non-negligible, with the normalizing constant and a quantile grid cached per parameter set.

## Parametrization

BL2 is an exact location-scale family, so it follows the scipy convention: a standard form on [0, 1] with two shape parameters, moved by `loc` and `scale`.

- `loc = lower`, `scale = upper - lower`
- `c = (mode - lower) / (upper - lower)`, the relative mode, 0 < c < 1 (as in `scipy.stats.triang`)
- `s = sigma / (upper - lower)`, the relative sigma, s > 0

`bl2.from_bounds(mode, sigma, lower, upper)` converts and returns a frozen distribution; `bl2.bounds_to_shapes` and `bl2.to_bounds` convert either way. The native form works like any scipy distribution:

```python
bl2.pdf(5, c=0.4, s=0.4, loc=-5, scale=25)    # same as rv.pdf(5) above
```

`from_bounds` raises `ValueError` for invalid parameters. The native form follows scipy and returns NaN.

## Speed and precision

Set once, at construction:

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

## Testing

```
pip install -e . --group test     # pip 25.1 or later
pytest
```

The suite checks the distribution against independent 30- and 40-digit mpmath reference values at every precision setting.

## Background and history

The original bounded lognormal (BL) is piecewise: a lognormal below the mode and a mirrored, rescaled lognormal above it. It has closed-form CDF and quantile functions, but its density is only C¹ at the mode, where the second derivative jumps. BL2 replaces the two pieces with their geometric mean, which is smooth everywhere. The two are different distributions, not two versions of one, so results computed with BL will not reproduce with BL2. BL remains available in the [`bounded_lognorm.ipynb`](https://github.com/tedteske/bounded_lognorm/blob/master/bounded_lognorm.ipynb) notebook; this package ships BL2 only.

Derivations and verification notes:

- [Phase 1: why the CDF has no elementary closed form](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase1_cdf_derivation.md)
- [Phase 2: math verification and numerical fixes](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase2_math_verification.md)
- [Phase 3: scipy.stats conformance and migration from the old API](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase3_api_conformance.md)
- [Phase 4: packaging](https://github.com/tedteske/bounded_lognorm/blob/master/docs/phase4_packaging.md)

## Attribution and license

The bounded lognormal distributions and their original implementation are by Ted Teske (2020–2024). The 2026 optimization, verification and scipy conformance work was done by Ted Teske with assistance from Claude (Anthropic). See [ATTRIBUTION.md](https://github.com/tedteske/bounded_lognorm/blob/master/ATTRIBUTION.md) for the detailed breakdown and references.

MIT License. See [LICENSE](https://github.com/tedteske/bounded_lognorm/blob/master/LICENSE).
