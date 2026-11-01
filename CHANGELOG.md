# Changelog

All notable changes to this project are documented here. The project follows [Semantic Versioning](https://semver.org/); while the version is 0.x, minor releases may change the API.

## 0.1.0 (unreleased)

First release on PyPI, as `bounded-lognorm`.

- The BL2 (bounded lognormal 2) distribution as a `scipy.stats.rv_continuous` subclass: standard form on [0, 1] with shapes `c` (relative mode) and `s` (relative sigma), plus the `from_bounds` / `bounds_to_shapes` / `to_bounds` bridge for the (mode, sigma, lower, upper) form.
- A speed/precision dial (`precision='fast' | 'balanced' | 'precise' | 'exact'`), carried through freezing and pickling.
- Every generic scipy method works, including `fit`, `moment(n)` for any n, `entropy` and freezing.
- The implementation module is `bounded_lognorm._bl2`; `bounded_lognorm.bl2` is the distribution instance only, as with `scipy.stats.norm`.
- Changes from the pre-package notebook versions are listed in [ATTRIBUTION.md](ATTRIBUTION.md) and the Phase 2 and Phase 3 notes in `docs/`.
