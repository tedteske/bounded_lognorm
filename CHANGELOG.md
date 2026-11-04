# Changelog

All notable changes to this project are documented here. The project follows [Semantic Versioning](https://semver.org/); while the version is 0.x, minor releases may change the API.

## 0.3.0 (unreleased)

- **`profile_bound`**: a profile-likelihood confidence interval for the upper or lower bound, on both `bl` and `bl2`, with `fit_bounds`' fixed-parameter options. The bound is held at trial values and the likelihood maximized over the other free parameters; the interval is where the likelihood ratio stays below the chi-squared(1) quantile. An end the data cannot close is reported as `inf`. It gives an interval where a bootstrap cannot, for a bound at or near the sample extreme. Maximum likelihood only. Calibration evidence, including where it falls short, is in `docs/phase7_bound_intervals.md`.
- **`fit_bounds(..., truncate=(lo, hi))`**: maximum likelihood conditional on a window of the data (either end open), so that data outside it have no influence on the fit. The result records the window and the numbers of data used and set aside (`.truncate`, `.n_used`, `.n_excluded`).
- `BoundsFit` also records which parameters ran away (`.at_cap`) and which bounds collapsed onto the data (`.collapsed`), previously only in `.message`.
- Nothing else changes: fits without `truncate` are identical to 0.2.1's.

## 0.2.1 (2026-11-02)

- **`fit_bounds` reports a bound that collapses onto the data.** With least squares, a free bound can converge onto the sample's smallest or largest value: the fit keeps improving as it approaches, as for the men's 100m speeds with the lower bound fixed at 0. 0.2.0 returned such fits with `success = True` and a standard error of ~0 for the bound, so a covariance interval was spuriously narrow. They now have `success = False`, a RuntimeWarning naming the bound, and NaN in that bound's `.cov` rows and columns and `.stderr`. The test is whether moving the bound 1000 times closer to the edge worsens the fit; every genuine fit examined worsens by 1e-2 or more and every collapse by 0 or less (`docs/v0.2_tails_and_fit_bounds.md` §2.1). Nothing else changes.

## 0.2.0 (2026-11-02)

- **BL, the piecewise bounded lognormal, joins the package** as `bl` (class `bounded_lognorm_gen`, module `bounded_lognorm._bl`), ported from the original notebook with BL2's conventions: standard form on [0, 1] with shapes `c` and `s`, `loc`/`scale`, the `from_bounds` / `bounds_to_shapes` / `to_bounds` bridge, `fit` and `fit_bounds`. Everything is in closed form: `pdf`, `cdf`, `sf`, `ppf`, `isf` (tails relative-accurate to ~1e-300), `logpdf`/`logcdf`/`logsf`, moments of any order, and entropy. **BL and BL2 are different distributions**; with the same parameters their CDFs differ by 0.03–0.05 typically and up to ~0.26 with the mode near a bound.
- BL accepts `precision=` for API parity with BL2; it has no effect.
- Behaviour changes from the notebook's BL (details in `docs/phase5_bl_port.md` and ATTRIBUTION.md): results come back in input order (the notebook returned left-of-mode results first for unsorted input, and `rvs` returned draws grouped by side of the mode); `rvs` honours `random_state`; skewness and kurtosis no longer cancel for narrow pieces (the notebook's kurtosis was off by up to 1e-2); entropy is exact instead of a truncated-normal approximation (off by 0.005–0.04 nats); `sf`, `isf` and the log forms keep relative accuracy in the tails; `moment(n)` works for every n.
- `from bounded_lognorm import bounded_lognorm` (the notebook's instance name) raises an ImportError explaining the migration, rather than aliasing to `bl` under the changed API.
- The bounds bridge, `fit_bounds`, `_argcheck`, `_fitstart` and `_shape_info` now live in a shared mixin (`bounded_lognorm._bridge`) used by both distributions; BL2's behaviour is unchanged.
- **Tail quantiles.** `isf` is now computed from the upper tail itself instead of scipy's `ppf(1 - q)`, and `ppf` keeps its accuracy deep in the lower tail. Both were wrong by more than 100% below q ~ 1e-9; they now hold their accuracy relative to q down to ~1e-30 (worst ~3e-3 at `'fast'`, 2e-4 `'balanced'`, 2e-5 `'precise'`, 6e-11 `'exact'`). Nothing changes above q = 1e-3, and `rvs` is unchanged.
- **`fit_bounds`**: fitting in the (mode, sigma, lower, upper) form, by MLE or least squares on the empirical CDF, with any parameters held fixed, as the original notebook's `fit` allowed. Least squares also returns curve_fit's covariance. Fits whose sigma or a bound runs away are reported as failures.
- `CITATION.cff`; README badges and sections on fitting, tails and citing.
- ATTRIBUTION.md: Wang et al. DOI verified (10.3141/2315-07).

## 0.1.0 (2026-11-01)

First release on PyPI, as `bounded-lognorm`.

- The BL2 (bounded lognormal 2) distribution as a `scipy.stats.rv_continuous` subclass: standard form on [0, 1] with shapes `c` (relative mode) and `s` (relative sigma), plus the `from_bounds` / `bounds_to_shapes` / `to_bounds` bridge for the (mode, sigma, lower, upper) form.
- A speed/precision dial (`precision='fast' | 'balanced' | 'precise' | 'exact'`), carried through freezing and pickling.
- Every generic scipy method works, including `fit`, `moment(n)` for any n, `entropy` and freezing.
- The implementation module is `bounded_lognorm._bl2`; `bounded_lognorm.bl2` is the distribution instance only, as with `scipy.stats.norm`.
- Changes from the pre-package notebook versions are listed in [ATTRIBUTION.md](ATTRIBUTION.md) and the Phase 2 and Phase 3 notes in `docs/`.
