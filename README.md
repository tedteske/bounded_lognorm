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
