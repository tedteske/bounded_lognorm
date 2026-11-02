# BL — Porting the Piecewise Bounded Lognormal into the Package (Phase 5)

**Status:** released. `bounded-lognorm` 0.2.0 was published to PyPI on 2 Nov 2026 from tag `v0.2.0` (merge commit `d0684fa` of PR #19); see §11. Implemented on top of master after PR #18 (`c6cc530`). Full test suite: 693 passed, 18 skipped (Suite B, which needs the original BL2), on Python 3.13 with the latest NumPy and SciPy and on Python 3.10 with NumPy 1.23.5 and SciPy 1.9.3. Suites A–G pass unchanged, gold values included.

**Epistemic labels**, as in the earlier phase notes: *Proof* = a complete argument. *Numerical* = measured against an independent 50-digit mpmath model of BL (§7), or against the original notebook code; true for the cases tested, not a theorem.

**Source.** The `bounded_lognorm_gen` class in the first code cell of `bounded_lognorm.ipynb` (Ted Teske, 2020–2021), Eq. 2: left of the mode a lognormal pdf at the lower bound; right of it a reflected lognormal pdf at the upper bound, scaled by k so the two meet at the mode; normalized by Z. Both pieces have mode `mode` and standard deviation `sigma` before truncation. The code is kept verbatim as the test oracle `tests/bl_original.py`.

**BL is not BL2.** They share parameters, conventions and the bounds bridge, and nothing else. The maximum cdf gap between them is 0.03–0.05 for typical parameters and 0.26 with the mode near a bound (Numerical; test H19 keeps it that way).

---

## 1. BL is a location-scale family (Proof)

Let u = upper − lower and y = (x − lower)/u, so x − lower = u·y and upper − x = u·(1 − y). Write m = mode − lower = c·u and r = upper − mode = (1 − c)·u, with c = m/u and s = σ/u.

1. **Shape parameters.** `lognorm_params(m, σ)` returns (ln m + t, √t), where t solves 3t + log(expm1 t) = 2 log(σ/m) (Phase 2 §4.2). So t depends on σ/m alone. For the left piece σ/m = s/c; for the right piece σ/r = s/(1 − c). Hence s₁ = √t₁ and s₂ = √t₂ depend on (c, s) only, and μ₁ = ln u + (ln c + t₁), μ₂ = ln u + (ln(1 − c) + t₂): each is ln u plus its standard-form value μᵢ*.
2. **Left piece.** A lognormal rescales as f_L(u·z; μ, s) = f_L(z; μ − ln u, s)/u. So f_L(x − lower; μ₁, s₁) = f_L(y; μ₁*, s₁)/u.
3. **Right piece.** Likewise f_L(upper − x; μ₂, s₂) = f_L(1 − y; μ₂*, s₂)/u.
4. **Continuity constant.** k = f_L(m; μ₁, s₁)/f_L(r; μ₂, s₂) = [f_L(c; μ₁*, s₁)/u]/[f_L(1 − c; μ₂*, s₂)/u] = k*: the factors 1/u cancel, so k is scale-free.
5. **Piece boundary.** x ≤ mode ⟺ y ≤ c.
6. **Normalizer.** The lognormal cdf rescales as F_L(u·z; μ, s) = F_L(z; μ − ln u, s). So Z = F_L(m; μ₁, s₁) + k·F_L(r; μ₂, s₂) = F_L(c; μ₁*, s₁) + k*·F_L(1 − c; μ₂*, s₂) = Z*: scale-free.

By 2–6, the unnormalized density in x is g*(y)/u piece by piece with the same Z, so pdf_x(x) = pdf*(y)/u, which is scipy's location-scale form with loc = lower, scale = upper − lower, and the standard form on [0, 1] with shapes c and s. Every step uses only x − lower and upper − x, so translation invariance is immediate. ∎

So BL takes exactly BL2's parametrization: `loc` = lower, `scale` = upper − lower, `c` = relative mode (0 < c < 1), `s` = relative sigma (s > 0), and the same `None` convention in `from_bounds` (None → mode ∓ 1000σ; both None is the fixed standard shape c = ½, s = 5e-4). Test H11 checks the equivariance numerically for pdf, cdf, ppf, entropy and std.

## 2. The mode-normalized closed form (Proof)

In standard coordinates let v = ln(y/c) left of the mode and v = ln((1 − y)/(1 − c)) right of it, so v ≤ 0 on both sides, and let aᵢ = sᵢ (so tᵢ = aᵢ²).

- **Density.** On the left, μ₁* = ln c + t₁ gives ln y − μ₁* = v − t₁, and then f_L(y)/f_L(c) = (c/y)·exp(−[(v − t₁)² − t₁²]/(2t₁)) = e^{−v}·e^{v − v²/(2t₁)} = exp(−v²/(2a₁²)). The same holds on the right. So

  **f(y) = f₀ · exp(−v² / (2aᵢ²))**, with f₀ the density at the mode.

  This is Eq. 2 rewritten, not an approximation; `density_reference` computes Eq. 2 as written with the original helper functions, and test H10 pins the two together to 1e-11.
- **Consequences.** ln f is strictly decreasing in |v| on each side, so BL is unimodal with its maximum exactly at the mode. f′ = 0 at the mode from both sides (C¹); f″ there is −f₀/(a₁²c²) from the left and −f₀/(a₂²(1 − c)²) from the right, which differ unless a₁c = a₂(1 − c), confirming Phase 2's "only C¹".
- **Piece masses.** Substituting y = c·eᵛ, the left mass is f₀·c·∫_{−∞}^0 e^{v − v²/(2t)} dv = f₀·c·a·R(a), where R(a) = Φ(−a)/φ(a) is the Mills ratio. So with Hᵢ = Lᵢ·aᵢ·R(aᵢ) (L₁ = c, L₂ = 1 − c): **f₀ = 1/(H₁ + H₂)** and the piece masses are **pᵢ = Hᵢ·f₀**. R is computed as erfcx(a/√2)·√(π/2), which neither overflows nor underflows.
- **Tails.** The mass of a piece beyond log-coordinate v (away from the mode) is f₀·L·a·e^{v − v²/(2t)}·R(a − v/a). So the cdf left of the mode is a·y·f(y)·R(a₁ − v/a₁), and the sf right of it is the mirror. Both are relative-accurate however small, and their logs are finite where they underflow.
- **Partial moments.** With V = ln(Y/c) on the left (V ~ N(t, t) conditioned on V ≤ 0), E[e^{jV} | left] = R((1 + j)a₁)/R(a₁); the exponential factors cancel exactly. This is the truncated-lognormal partial moment that the original `_stats` used (Wang et al.; Ford & McLaren), in a form that cannot overflow. The right piece is the same with a₂.
- **Entropy.** H = −ln f₀ + Σᵢ pᵢ·E[V²]/(2tᵢ) = −ln f₀ + ½ Σᵢ pᵢ·(1 + aᵢ² − aᵢ/R(aᵢ)), using E[Z | Z ≤ −a] = −1/R(a) and E[Z² | Z ≤ −a] = 1 + a/R(a). scipy adds ln(scale).

## 3. Evaluating the closed forms (Numerical)

Each closed form above is exact; three of them need care in floating point.

**cdf and sf near the mode.** Right of the mode the cdf is p₁ plus the mass between the mode and y, a sum of positive terms, so it never forms 1 − sf. That mass is f₀·(1 − c)·M(v) with M(v) = ∫_v^0 e^{u − u²/(2a²)} du. Far from the mode M is the closed form a·[R(a) − e^{v − v²/(2t)}R(a − v/a)], whose two terms differ by at least a factor ~0.6 there. Within |v| ≤ min(a, 1) that difference would cancel, so M is taken by 16-point Gauss–Legendre on [v, 0], where the integrand is smooth and varies by less than e^1.5 (exact to rounding). The sf left of the mode mirrors this. Near the mode v itself is computed as log1p((y − c)/c) or log1p((c − y)/(1 − c)), because y − c is exact there and ln((1 − y)/(1 − c)) would first round 1 − y: with the mode at c ≈ 1e-7 that alone cost ~1e-6 relative in v.

**Quantiles.** Each q is inverted on the piece whose own tail it lies in: v = a(β + a) with Φ(β) = q·Φ(−a)/p, solved in log probability (`ndtri_exp`), so ppf and isf stay relative-accurate down to q ≈ 1e-300. The other side of the mode is reached through 1 − q, which loses relative accuracy in q − p₁ when the near piece holds little mass. Two Newton steps on the accurate cdf (sf for isf) restore it: with the mode within 1e-6 of a bound (p₁ ≈ 8e-7), quantiles just past that piece go from ~1e-10 to ≤ 4e-16 relative error in q. `rvs` uses the unpolished inverse, whose error is far below sampling noise.

**Moments.** The original assembled central moments from raw moments, which cancels when a piece is narrow (a small): measured against the reference, its kurtosis is off by 1e-2 and 5e-3 and its skewness by 3e-6 at the two needle cases. The port takes moments about the mode instead: Y − c = −c·D₁ on the left and (1 − c)·D₂ on the right, with Dᵢ = 1 − e^{Vᵢ} ∈ [0, 1). Then A_k = E[(Y − c)^k] = p₁(−c)^k E[D₁^k] + p₂(1 − c)^k E[D₂^k], and the mean, variance, skewness and kurtosis follow from A₁…A₄ without large cancellation.

E[D^k] = Σⱼ C(k, j)(−1)ʲ R((1 + j)a)/R(a) is still an alternating sum, which loses about k·log₁₀(1/a) digits. It is evaluated by the first that applies:

1. **direct**, if the sum's computed condition number Σ|terms|/|sum| is at most 50;
2. **series**, if k·a ≤ 2: the same finite difference expanded in powers of a, Σ_{m≥k} (−1)^k k!·S(m, k)·R^{(m)}(a)·a^m/m!, with S the Stirling numbers of the second kind (as the coefficients of (e^{ax} − 1)^k) and R^{(m)} by its recurrence R^{(m)} = a R^{(m−1)} + (m − 1) R^{(m−2)}. An exact rearrangement; its terms decrease from the first when k·a is small;
3. **direct**, if its condition number is at most 1000 (error ≲ 2e-13);
4. otherwise **quadrature**: E[D^k] as the smooth integral ∫₀^∞ (1 − e^{−aw})^k e^{−aw − w²/2} dw / R(a), by `scipy.integrate.quad`.

The stats (k ≤ 4) never reach step 4: test H16 checks this over a ∈ [1e-6, 50]. Raw moments of order n ≥ 5 (`_munp`) need E[D^k] up to k = n, and step 4 is used for k ≳ 6 at moderate and large a. Worst relative error over a ∈ [1e-4, 5] and k ≤ 20 against a 40-digit quadrature: 7e-14 for k ≤ 4, 6e-13 overall. `_munp` sums only positive terms: E[Yⁿ] = p₁cⁿR((1 + n)a₁)/R(a₁) + p₂ Σ_k C(n, k) c^{n−k}(1 − c)^k E[D₂^k].

## 4. The four decisions

**(1) Location-scale: yes, proved in §1.** BL takes BL2's standard form and bridge unchanged.

**(2) Names.**

- Class `bounded_lognorm_gen`: the notebook's own class name, and the pattern BL2 already follows (`bounded_lognorm2_gen`). The scipy `name` is `'bounded_lognorm'`, as in the notebook.
- Instance **`bl`**, beside `bl2`, and module **`bounded_lognorm._bl`**, beside `_bl2`, as Phase 4 planned. The instance name and the module name differ, so the Phase 4 shadowing problem (an instance hiding its own submodule) does not arise; `tests/test_package.py` guards `bl2` and the same pattern holds for `bl`.
- **No alias for the notebook's `bounded_lognorm` instance.** Under the new signature an old call such as `bounded_lognorm.pdf(x, 5, 10, -5, 20)` is read as c = 5, s = 10, loc = −5, scale = 20 and returns NaN; with a mode inside (0, 1), say `bounded_lognorm.pdf(x, 0.5, 0.1, 0, 2)`, it returns a plausible wrong number. An alias would turn every such call site into silent NaN or wrong results. Instead `from bounded_lognorm import bounded_lognorm` (what `Applications_100m_Sprints.ipynb` does) raises an ImportError naming `bl.from_bounds` (test H20). It is raised as ImportError rather than AttributeError because `from … import` replaces an AttributeError from a module `__getattr__` with a generic "cannot import name" and drops the message.

**(3) Precision: accepted for API parity, no effect.** Every hook has a closed form, and the measured accuracy (§6) is at the float64 floor with no speed to trade: a dial would have nothing to set. `precision` is still accepted and validated against the same four positions, stored on the instance, and carried through freezing and pickling by the shared `_updated_ctor_param`, so that Phase 6 code written as `gen(precision=p)` runs on either family. Test H13 checks that all four positions give bit-identical results.

**(4) Intentional behaviour changes from the notebook** (also in ATTRIBUTION.md):

- *API*: `(c, s, loc, scale)` with `from_bounds` for the old form; scipy hooks instead of public overrides; no `ignore_checks`; NaN for invalid shapes natively, `ValueError` from `from_bounds`; no alias for the old instance name.
- *Output order*: elementwise everywhere. The original returned left-of-mode results first for unsorted input (Suite I5), and its `rvs`, being `ppf` of unsorted uniforms, returned draws grouped by side of the mode. That matters for Monte Carlo: pairing 10,000 draws each from two *independent* BLs (mode 5, σ 10 on (−5, 20) and mode 50, σ 3 on (40, 60)) gave a correlation of 0.59 with the original, against −0.02 now. Results from Monte Carlo runs that combined several notebook-BL variables elementwise are worth re-running.
- *`rvs`* honours `random_state` (the original used the global `np.random`).
- *Higher moments* about the mode; `moment(n)` for every n (was `None` for n ≥ 5).
- *Entropy* in closed form (the truncated-normal adaptation was off by 0.005–0.04 nats).
- *Tails*: `sf`, `isf`, `logpdf`, `logcdf`, `logsf` from their own ends.
- *Fitting*: scipy's `fit`, plus the shared `fit_bounds` (midpoint ecdf default; runaway detection). The histogram-pdf least-squares fallback is gone.
- *`lognorm_params`*: the stable Phase 2 solver.

**A correction to the brief.** The Phase 5 brief described the notebook's BL moments as "biased grid-sum moments with a sample-skewness correction". That describes BL2's original `_stats` (fixed in Phase 2). The notebook's BL already used closed-form truncated-lognormal partial moments combined with Raiford's formulas, and they are correct where well-conditioned: the mean and variance agree with the reference to 1e-11 or better for every non-needle case (Suite I3). What fails is the conditioning for narrow pieces (§3, Suite I4).

## 5. Hooks and shared code

- BL implements `_pdf`, `_logpdf`, `_cdf`, `_sf`, `_logcdf`, `_logsf`, `_ppf`, `_isf`, `_rvs`, `_stats`, `_munp`, `_entropy`. `_logcdf` and `_logsf` go beyond the brief's list because closed forms exist and make the log tails finite where the linear ones underflow (test H15).
- `_argcheck`, `_shape_info`, `_fitstart`, `_updated_ctor_param`, `from_bounds`, `bounds_to_shapes`, `to_bounds` and `fit_bounds` moved unchanged from `_bl2.py` into a mixin, `bounded_lognorm._bridge._BoundsBridge`, which both classes inherit. Suites A–G pass unchanged, gold values included.
- `fit_bounds` needed no change: `_fit.py` uses only `bounds_to_shapes`, `logpdf` and `cdf`.
- Per-(c, s) constants (two `lognorm_params` solves) are cached; array-valued shapes are grouped by distinct pair, as in BL2.

## 6. Accuracy (Numerical)

Against the 50-digit model over 15 cases (the nine Suite A cases, the mode at 0.01, 1e-6 and 9.99 on (0, 10), unbounded on one and both sides, and σ = 100 on (0, 10)):

- pdf, cdf, sf (relative; points across the support, in both deep tails, and at mode ∓ 1e-6σ): at the float64 floor set by representing x itself (|x|·eps·pdf/F), worst 3e-12 for the pdf and 3e-12 for the sf, none beyond the floor
- body quantiles, q from 0.01 to 0.99 (relative error in the nearer tail probability): worst 6e-13 (the (950, 1, −1000, 1000) case, at its floor); 2e-14 for the others
- tail quantiles, ppf and isf for q from 1e-3 to 1e-30: at the floor, none beyond it
- deep lower tail in standard form, q from 1e-50 down to subnormal 1e-320: 6e-15 relative in y
- quantiles just past a piece holding 8e-7 of the mass: 4e-16 relative in q (1e-10 without the Newton polish)
- raw moments E[Yⁿ], n = 1–8: 2e-14 relative
- mean (in units of σ) 7e-15, variance 4e-14 relative, skewness 7e-14, excess kurtosis 1e-13
- entropy: 9e-16 nats

Timing (this machine, 1000 points, mode 5, σ 10 on (−5, 20), frozen; scipy's argument handling included): pdf 0.12 ms, cdf 0.20 ms, sf 0.18 ms, ppf 0.63 ms, isf 0.60 ms per 1000 points; rvs 52 ms per million draws; a new (c, s) pair costs ~0.3 ms (constants plus stats). For comparison, BL2 at `'balanced'` takes 1.29 ms for the cdf and 115 ms per million draws; its grid ppf (0.14 ms) is faster once its grid is built.

## 7. Reference model

`tests/bl_mpmath_reference.py` (50 digits) shares no numerics with the library: the quartic g⁴ − g³ − (σ/m)² = 0 by bisection; the density exactly as Eq. 2 defines it, in the bounds coordinates; Z, moments and entropy by tanh-sinh quadrature over ~200 break points graded toward the mode and both bounds; cdf and sf both by quadrature from their own ends and by the textbook lognormal cdf, which must agree to 1e-28 (worst disagreement 1.5e-43); quantiles by bisection on log F and log S. Deep-tail quadratures are rescaled by the density at their inner end, because mpmath's stopping test is effectively absolute. The partial-moment evaluator was mapped separately against a 40-digit quadrature of E[D^k].

## 8. Tests (Suites H and I, mutation-checked)

`tests/test_bl.py`. Suite H compares against the reference and guards each numerical choice; Suite I compares against the original notebook code (`tests/bl_original.py`, verbatim), pinning what is unchanged and documenting what changed.

Mutation-checked (Nov 2026). Each change below makes the listed tests fail:

- the near-mode Gauss–Legendre branch of M(v) off (closed form everywhere) → H2, H4, H5b
- cdf right of the mode as 1 − sf → H2, H4, H5b
- sf as 1 − cdf → H2, H4, H15
- no Newton polish → H4, H5b
- quantile inversion through ndtri(exp(·)) instead of in log probability → H5
- `_isf` removed (scipy's ppf(1 − q)) → H4
- partial moments by the direct sum only → H6, H7, I4
- the series path off (series → direct) → H6, H7, I4
- the quadrature fallback replaced by the direct sum → H6
- entropy back to the truncated-normal adaptation → H8, I6
- `_logpdf` as log(pdf) → H15; `_logcdf`/`_logsf` removed → H15
- the log-coordinate without log1p near the mode → H2, H4, H5b
- `precision` not carried through freezing (in the shared mixin) → H13, and BL2's E1

Not test-guarded: the expm1 form of the closed-form inverse next to the mode (§3). The Newton polish corrects it for `ppf` and `isf`, so it only affects `rvs`, at the ~1e-10 level, far below sampling noise.

The other tests are property checks rather than guards on one code path: the definition (H10), location-scale equivariance (H11), input order and shape (H12) and draw order (H12b), precision without effect (H13), array-valued shapes (H14), moment paths (H16), `moment(n)` against `expect` (H17), `fit` and `fit_bounds` (H18), BL ≠ BL2 (H19), the migration error (H20) and the bridge and its errors (H21). Suite I pins agreement with the original where nothing changed (I1–I3: pdf, cdf, ppf, and mean and variance off the needles) and documents what did (I4 higher moments, I5 output order, I6 entropy).

## 9. Migration from the notebook

- `bounded_lognorm.pdf(x, mode, sigma, lower, upper)` → `bl.from_bounds(mode, sigma, lower, upper).pdf(x)`, and likewise for `cdf`, `ppf`, `rvs`, `stats`, `entropy`, …
- `from bounded_lognorm import bounded_lognorm` → `from bounded_lognorm import bl`
- `bounded_lognorm.stats(mode, sigma, lower, upper, moments='mvsk')` → `bl.from_bounds(mode, sigma, lower, upper).stats('mvsk')` (scalars, not 1-element arrays)
- `bounded_lognorm.fit(data, MLE=False, flower=0)` → `bl.fit_bounds(data, 'LSQ', flower=0)`; add `ecdf='step'` to reproduce the original empirical cdf exactly. `verbose=True` → the returned `BoundsFit` carries `.cov`, `.stderr`, `.nnlf`, `.success`.
- `bounded_lognorm.rvs(..., size=n)` → `rv.rvs(size=n, random_state=seed)`: draws now come in generated order and are reproducible.
- **Hazard:** `bl.pdf(x, mode, sigma, lower, upper)` without `from_bounds` is read as (c, s, loc, scale).

## 10. Still open

- As mode → a bound the mass need not follow the mode (as for BL2; user-facing note).
- Raw moments of order n ≥ ~6 use the quadrature fallback in part of the parameter space (§3); accurate (≤ 6e-13), but not closed form there.
- The 100m-sprints notebook still imports the notebook's instance; updating it is Phase 6.

## 11. Release record: 0.2.0 (Numerical)

- **CI on PR #19:** all 10 jobs passed (Ubuntu with Python 3.10–3.15 on the latest dependencies, Python 3.10 on NumPy 1.23.5 and SciPy 1.9.3, macOS and Windows on 3.13, and the build-and-check job). The macOS job first sat queued for 15 minutes without a runner and was cancelled with no steps run; the re-run passed. The same 10 jobs passed again on the merge commit `d0684fa`.
- **TestPyPI dry run:** the manually triggered publish run on the PR branch built the wheel and sdist, passed `twine check --strict` and the full suite against the built wheel, and uploaded both with digital attestations. In Google Colab (Python 3.13, Colab's own NumPy and SciPy), installing 0.2.0 from TestPyPI gave version 0.2.0 from the installed location, and BL's mean (5.849409133600799), entropy (2.733654201134262) and isf(1e-20) (19.74541097177343) and BL2's mean (5.872972571882167) for mode 5, σ 10 on (−5, 20) all agreed with the 50-digit reference to every printed digit. The full suite from the TestPyPI sdist, run against that install, gave 693 passed, 18 skipped. (`pip download --no-binary :all:` needs `--extra-index-url https://pypi.org/simple/` beside the TestPyPI index, so that it can fetch the build backend, hatchling, to read the sdist's metadata.)
- **PyPI release:** GitHub release `v0.2.0` at `d0684fa`; publish run #4 passed its tag check and the suite against the built wheel, waited for approval, and uploaded. https://pypi.org/project/bounded-lognorm/0.2.0/ A second manual run (#5) on master, started after the release, published to TestPyPI only, as manual runs do; its upload was byte-identical to the dry run's, so TestPyPI accepted it without change.
- **Check after release:** `pip install bounded-lognorm==0.2.0` into a clean Python 3.12 environment gave 0.2.0 and the same four values to every printed digit, and `from bounded_lognorm import bounded_lognorm` raised the migration ImportError.
- **Zenodo:** the GitHub integration was enabled for the repository before the release, so the release was archived with a DOI: version 0.2.0 is [10.5281/zenodo.23897349](https://doi.org/10.5281/zenodo.23897349), and the concept DOI for all versions, which always resolves to the latest, is [10.5281/zenodo.23897348](https://doi.org/10.5281/zenodo.23897348). The concept DOI is in `CITATION.cff` and the README badge. 0.1.0 predates the integration and has no DOI.

