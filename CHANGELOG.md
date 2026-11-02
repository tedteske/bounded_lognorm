# Changelog

All notable changes to this project are documented here. The project follows [Semantic Versioning](https://semver.org/); while the version is 0.x, minor releases may change the API.

## Unreleased (0.2.0)

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
