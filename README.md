# bounded_lognorm
Bounded lognormal continuous probability distribution. 

The **bounded lognormal** is a continuous probability distribution derived from the [lognormal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution). Unlike the normal and lognormal distributions, its domain has both upper and lower bounds, and it may be either symmetrical or non-symmetrical, depending on its parameters. It therefore may be a good fit for many natural phenomena, for which there are known limits, for example adult male height. 

The lognormal probability density function (pdf) `f_L` is often defined in terms of `mu_N` and `sigma_N`, the mean and standard deviation of some normal distribution `N`.

If we redefine `f_L` in terms of mode `m` and standard deviation `sigma` as `f_L'`, then the bounded lognormal pdf `f_BL` is the piecewise function:

```
f_BL(x, m, sigma, upper) =
    f_L'(x, m, sigma) for x <= m, and
    f_L'(upper-x, upper-m, sigma) for x > m
```

where the second piece is scaled so that `f_BL` is continuous at the mode, and `(lower, upper)` are the lower and upper bounds of the domain.

The bounded lognormal distribution is implemented in Python as an instance of the [scipy.stats.rv_continuous](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous) class and inherits from it a collection of generic methods.

## Usage

BL2 follows the scipy.stats location-scale convention: a standard form on [0, 1] with shapes `c` (relative mode) and `s` (relative sigma), moved by `loc` = lower and `scale` = upper − lower. If you think in bounds, `from_bounds` does the conversion and returns a frozen distribution:

```python
from bounded_lognorm import bl2, bounded_lognorm2_gen

rv = bl2.from_bounds(mode=5, sigma=10, lower=-5, upper=20)
rv.pdf(0.0), rv.cdf(0.0), rv.ppf(0.9), rv.mean(), rv.entropy()
sample = rv.rvs(size=1000, random_state=0)

c, s, loc, scale = bl2.fit(sample)            # scipy's MLE
mode, sigma, lower, upper = bl2.to_bounds(c, s, loc, scale)

bl2.pdf(5, c=0.4, s=0.4, loc=-5, scale=25)    # native form, same as rv.pdf(5)
bl2.from_bounds(mode=0, sigma=1)              # None bound = unbounded side

exact = bounded_lognorm2_gen(precision='exact')   # speed/precision dial
```

See the class docstring for the precision dial and `docs/phase3_api_conformance.md` for the parametrization and migration notes.
