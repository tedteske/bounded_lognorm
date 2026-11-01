# BL2 — scipy.stats Conformance (Phase 3)

**Status:** implemented on top of Phase 2 commit `049ded7` (Nov 2026). Full test suite: 423/423 passing, including Suites A–D unchanged in their gold values and the new Suite E.

**Epistemic labels**, as in Phase 2: *Proof* = a complete argument. *Numerical* = measured against an independent mpmath reference (40 digits here); true for the cases tested, not a theorem.

**Goal.** Expose the distribution through scipy's private hooks (`_pdf`, `_cdf`, `_ppf`, …) instead of overriding the public `pdf`/`cdf`, so freezing, `fit` and every generic `rv_continuous` method work.

---

## 1. Parametrization: BL2 is a location-scale family (Proof)

Let u = upper − lower and y = (x − lower)/u, so a = x − lower = u·y and b = upper − x = u·(1 − y).

- `lognorm_params(m, σ)` returns (ln m + t, √t), where t solves 3t + log(expm1 t) = 2 log(σ/m). So t depends on σ/m alone (Phase 2, §4.2), and scaling m and σ by 1/u leaves s₁, s₂ unchanged and shifts μ₁, μ₂ by −ln u.
- A lognormal rescales as f_L(u·y; μ, s) = f_L(y; μ − ln u, s)/u.
- Hence g_x(x) = √(f_L(a) f_L(b)) = g_std(y)/u, where g_std is BL2 on [0, 1] with mode c = (mode − lower)/u and sigma s = σ/u.
- Then Z_x = ∫ g_x dx = ∫ g_std dy = Z_std, so **Z is scale-free**, and pdf_x(x) = pdf_std(y)/u. ∎

This is exactly scipy's location-scale form, so:

- `loc` = lower, `scale` = upper − lower, standard support [0, 1]
- shapes **`c`** = (mode − lower)/(upper − lower), 0 < c < 1 (the same convention as `scipy.stats.triang`), and **`s`** = σ/(upper − lower), s > 0
- σ is still the SD of each untruncated lognormal factor, not of BL2

**`None` bounds** exist only in the bounds-form bridge, `bl2.from_bounds(mode, sigma, lower=None, upper=None)`, which keeps the Phase 2 rule (None → mode ∓ 1000σ). The native API has no `None`: `loc` and `scale` are numbers, as in all of scipy. A two-sided unbounded BL2 is one fixed standard shape, c = 0.5 and s = 5e-4, moved by loc and scale.

**Errors.** `from_bounds` (and `bounds_to_shapes`) raise `ValueError` for infinite or inconsistent input, so a bad parameter set fails loudly. The native API follows scipy and returns NaN for invalid shapes.

## 2. Caching with array-valued shapes

scipy calls `_pdf(x, c, s)` etc. with c and s as arrays broadcast against x. The caches (`_panel_table`, `_cdf_grid`) need hashable floats.

- Every hook goes through `_by_params`, which groups the inputs by distinct (c, s) pair and evaluates each group through the cached scalar table. When all pairs are equal (the usual case: one parameter set, broadcast) a fast path skips `np.unique`.
- The cache is called in standard coordinates (lower = 0, upper = 1), so the key is (c, s, n_quad) only. **Shifted or rescaled copies of a distribution share one table**, which they could not before.
- Cost (Numerical, this machine): `pdf` at 1000 points is 0.13 ms with scalar shapes, 0.10 ms with an all-equal array, 0.5 ms with 4 distinct pairs. Each new pair builds a table once (~3 ms), so 100 distinct uncached pairs in one call take ~0.26 s. Calls with thousands of distinct shapes are slow; that is the price of exact per-set quadrature.

## 3. The precision dial survives freezing

scipy freezes by re-instantiating the class from `_updated_ctor_param()`. Without an override, every frozen distribution silently fell back to `'balanced'`. The override adds `precision`. Mutation check: removing it fails E1 and also A5, C2 and D3, because every `'exact'` ppf quietly became grid-based. Pickled frozen distributions keep the setting too.

## 4. Hooks implemented

- `_pdf`, `_logpdf` (direct log form, finite where `pdf` underflows; this matters for `fit` and `nnlf`)
- `_cdf`, `_sf` (integrated from the right, see §6), `_ppf` (grid or brentq, per the dial)
- `_rvs` (always the grid, per the Phase 2 decision; scipy's default would have used `_ppf`, i.e. brentq per sample at `'exact'`)
- `_stats` (mean, var, skew, kurtosis by graded Gauss–Legendre), `_munp` (§5), `_entropy` (§5)
- `_argcheck`, `_shape_info` (used by `scipy.stats.fit`), `_fitstart`

scipy supplies everything else: `logcdf`, `logsf`, `isf`, `median`, `interval`, `expect`, `nnlf`, `support`, `fit`, freezing. All public overrides from Phase 2 are gone.

## 5. Entropy and moments (Numerical)

- **Entropy** is now BL2's own differential entropy: H = ln Z − (1/Z)∫ g ln g, on the cached nodes, with ln g from the log form. scipy adds ln(scale). Worst error against the reference over 8 cases (including the needle, mode-near-bound and unbounded cases): 4e-14 nats at every dial position. The truncated-normal adaptation it replaces was close for near-symmetric cases (2.860 vs 2.864 at mode 5, σ 10 on (−5, 20)) but off by 0.18 nats with the mode near a bound (0.413 vs 0.236 at mode 0.01, σ 1 on (0, 10)).
- **`moment(n)`** works for every n. `_munp` returns E[Yⁿ] on [0, 1], where every term lies in [0, 1] and the sum cannot cancel; scipy maps it to loc and scale by the binomial expansion. Worst relative error for n = 5–8 over 5 cases: 2e-12. n = 1–4 agree with `stats()` to 1e-12; n = 10 and 20 agree with `expect` to 1e-9.

## 6. Tail accuracy: the effective-support cutoff (Numerical)

Making `_sf` integrate from the right was not enough on its own. The effective support, where the density exceeds 1e-14 of its peak, discarded tail mass at the ~1e-16 absolute level, so `sf` near the support edge was no better than `1 − cdf` (1e-4 relative error at sf = 1.7e-12).

- The cutoff is now `_SUPPORT_TOL = 1e-40`. Upper-tail `sf` holds ≤ 2e-13 relative error down to sf = 8e-33 at every dial position, where `1 − cdf` returns 0. Points past the new cutoff (sf ≈ 4e-66) still return 0.
- Lower-tail `cdf`/`logcdf` benefit the same way, since `cum` is summed from the left.
- Cost: none measurable. The table gains a few graded panels (56 vs 49 at mode 5, σ 10 on (−5, 20)), built once. A cutoff of 1e-100 also passed every test but adds 10 more panels for tails no test reaches.

## 7. fit

`bl2.fit(data)` is scipy's MLE (or `method='MM'`) over (c, s, loc, scale). `_fitstart` places the bounds 5% outside the data range, c at the median and s at the sample SD, because scipy's generic start sets every shape to 1, which is an invalid c.

- With bounds fixed (`floc`, `fscale`), 3000 draws from mode 75, σ 5 on (70, 100) recover c to 0.0002 and s to 1.4% in 0.3 s.
- All four free: converges in ~1.5 s to (mode 75.05, σ 5.14, lower 69.98, upper 96.94), with negative log-likelihood below that of the true parameters, as an MLE must.
- `scipy.stats.fit(bl2, data, bounds=...)` works through `_shape_info` (~5 s for the same data).

## 8. Migration from the Phase 2 API

- `bl2.pdf(x, mode, sigma, lower, upper)` → `bl2.from_bounds(mode, sigma, lower, upper).pdf(x)`
- Native form: `bl2.pdf(x, c, s, loc=lower, scale=upper - lower)`; `c, s, loc, scale = bl2.bounds_to_shapes(mode, sigma, lower, upper)`
- `bl2.stats(mode, sigma)` (unbounded) → `bl2.from_bounds(mode, sigma).stats()`
- `bl2.fit(data)` now returns (c, s, loc, scale); `bl2.to_bounds(*bl2.fit(data))` gives (mode, sigma, lower, upper)
- `ignore_checks` is gone. `normed=False` → `pdf * bl2.normalizing_constant(c, s)`
- `density_reference(x, c, s, loc, scale)` keeps the readable Eq. 3 form
- **Hazard:** an old positional call such as `bl2.pdf(x, 5, 10, -5, 20)` is now read as c = 5 and returns NaN rather than raising

## 9. Test coverage (Suite E, mutation-checked)

Suites A–D now call through `from_bounds`; their gold values are untouched. Reverting each change makes these fail:

- `_updated_ctor_param` override → E1 (and A5, C2, D3)
- grouping by (c, s) → E2
- entropy by quadrature (back to truncated normal) → E4
- `_munp` (removed, or None for n ≥ 5) → E5
- `_sf` (back to 1 − cdf) → E6; the 1e-40 cutoff (back to 1e-14) → E6
- `_logpdf` (back to log of pdf) → E7
- `_fitstart` (back to scipy's) → E8

E3 (loc/scale equivariance) and E9 (bridge round trip, generic methods) are property checks rather than guards on one code path. D6 was rewritten for the new error contract: `from_bounds` raises, the native API returns NaN.

## 10. Still open

- **Fitting by CDF curve.** The original `fit(MLE=False)` least-squares on the empirical CDF, and fixing `mode` or one bound alone, are not carried over. They could return as a `fit_bounds` helper if wanted.
- **Package namespace.** `from .bl2 import bl2` in `__init__.py` makes `bounded_lognorm.bl2` resolve to the instance, shadowing the submodule (pre-existing; `import bounded_lognorm.bl2 as m` returns the instance). Worth renaming before PyPI (Phase 4).
- **`isf` deep tail.** scipy's `isf` is `ppf(1 − q)`, so it loses relative accuracy for q below ~1e-12 even though `sf` does not. A dedicated `_isf` would invert `_sf`.
- From Phase 2, unchanged: as mode → lower the mass does not follow the mode (user-facing note); bounds → ∞ giving SD → σ remains a sketch plus numerics.
