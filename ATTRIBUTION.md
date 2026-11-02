## Attribution & References

### Provenance

This package and its notebooks contain work of three distinct kinds. They are separated below so that credit — and responsibility for errors — lands in the right place.

**A. Original work — Ted Teske (2020–2024), sole author**

- The bounded lognormal (BL) and bounded lognormal 2 (BL2) distributions themselves: the piecewise construction (eq. 2) and the geometric-mean construction (eq. 3).
- The mode/σ reparametrization `lognorm_params`, including the quartic root formulation.
- The `rv_continuous` subclass design and its full public API: `pdf`, `cdf`, `ppf`, `rvs`, `stats`, `moment`, `entropy`, `nnlf`, `fit`, and the supporting helper functions.
- For BL: the closed-form `cdf` and `ppf` built by concatenating the two lognormal pieces, and the moments as weighted truncated-lognormal partial moments (after Wang et al. and Ford & McLaren, ref. 12–13), combined across the pieces with Raiford's formulas for combined distributions (ref. 11).
- The adaptation of the truncated-normal entropy formula to BL and BL2.
- All example usage, plots, and the empirical comparison to the normal distribution.

**B. Collaborative work — Ted Teske with Claude (Anthropic), 2026**

- *Phase 1 CDF analysis:* the demonstration that the BL2 CDF admits no elementary closed form, via the observation that no change of variable linearizes both `ln x` and `ln(upper − x)` simultaneously (they are coupled by `e^w + e^v = u`); together with the catalogue of tractable special cases and the saddle-point route.
- *Performance refactor of the class:* hoisting the shape parameters and normalizing constant `Z` out of `pdf` (with `lru_cache`); vectorized composite Gauss–Legendre quadrature; restriction of all integration to an *effective support* located by bisection outward from the mode; the cached PCHIP cumulative grid backing `ppf`/`rvs`/`median`.
- The regression test cell.
- *Phase 2 math verification and fixes:* proofs that the mode of BL2 is exactly `mode` and that BL2 is C∞ (BL is only C¹), that Z ≤ 1 (a Bhattacharyya coefficient), and of the effective-support truncation bound; a stable, scale-free solver for the `lognorm_params` quartic (same equation, new numerics); mode-graded quadrature panels; moments by quadrature in place of the grid sum; argument guards; regression Suite D against a 30-digit reference. See `docs/phase2_math_verification.md`.
- *Phase 3 scipy.stats conformance:* the observation that BL2 is an exact location-scale family (Z is scale-free), and the resulting standard form on [0, 1] with shapes `c` = relative mode and `s` = relative sigma, `loc` = lower, `scale` = upper − lower; the `from_bounds` / `bounds_to_shapes` / `to_bounds` bridge and its `None`-bound convention; the `_pdf`/`_logpdf`/`_cdf`/`_sf`/`_ppf`/`_rvs`/`_stats`/`_munp`/`_entropy`/`_fitstart` hooks with caching grouped by distinct (c, s); carrying the precision dial through freezing; entropy by quadrature of −f ln f; the 1e-40 effective-support cutoff for tail accuracy; regression Suite E against a 40-digit reference. See `docs/phase3_api_conformance.md`.

- *Phase 5 port of BL into the package (0.2.0):* the proof that BL is an exact location-scale family, piece by piece; its standard form on [0, 1] with BL2's shapes `c`, `s` and the shared bounds bridge (moved into a mixin used by both); the mode-normalized closed form of the density, f(y) = f0·exp(−v²/2a²) with v the log-distance from the mode, and from it the Mills-ratio forms of the piece masses, of each tail (cdf from the left, sf from the right, and their logs), of the partial moments E[e^{jV}] = R((1+j)a)/R(a), and of the entropy; evaluating the partial moments about the mode, with a Stirling-series rearrangement for narrow pieces and a quadrature fallback for high orders, chosen by the computed condition number; quantile inversion in log probability with a Newton polish; elementwise evaluation preserving input order; regression Suites H (against an independent 50-digit model) and I (against the original notebook code, verbatim). See `docs/phase5_bl_port.md`.

**C. Third-party work — cited below**

Attribution suggestion, not a rule: the usual convention in published work is that an AI system is **not** listed as an author, since authorship carries accountability an AI cannot hold. If you publish this, the conventional form is to claim the work as your own and add an acknowledgement of AI assistance — e.g. *"Portions of the numerical implementation were developed with assistance from Claude (Anthropic); the author reviewed and verified all derivations and results."* Note also that some journals now require disclosure of AI use. Category B above is written to give you the specifics such a disclosure would need.

A caveat worth recording, in the same spirit: the Phase 1 non-closed-form result is an argument plus a symbolic check, **not a proof**. It shows no *elementary* antiderivative was found and explains structurally why one is not expected. It does not rule out a closed form in terms of special functions. Treat it as a well-supported working conclusion.

### Behavioural changes from the original

*Correction to an earlier version of this note.* It previously said the original's unnormalized density carried `1/(2π·√(s₁s₂))`. It did not: the original computed `sqrt(lognorm_pdf × lognorm_pdf)`, which gives the correct `1/√(2π·s₁s₂)`. The wrong constant existed only in an intermediate draft of the 2026 refactor and was fixed before release, so `normed=False` output is unchanged from the original.

Phase 2 (Nov 2026) intentionally changes these results:

- **`mean`, `var`, `std`, `stats`, `moment`, `entropy`.** The original grid-sum moments were biased (weights of `(upper − lower)/n` instead of `/(n − 1)`, never renormalized), applied a sample-skewness correction to a population moment, and aliased narrow peaks on a fixed 1001-point grid. They are now integrated by quadrature and match a 30-digit reference to ~1e-12. Example: `var(100, 1)` was 2.68; it is now 0.99999.
- **Extreme parameters.** sigma/mode outside roughly 1e-5 to 1e2 (sigma → 0, very wide bounds, mode very near a bound) previously lost accuracy or returned NaN; results there now agree with the reference down to the float64 resolution floor.
- **Invalid inputs raise.** Infinite bounds, and invalid parameters passed with `ignore_checks=True`, now raise `ValueError` instead of returning NaN or a plausible-looking number. A `RuntimeWarning` flags sigma below 1e-8 of the coordinate scale.

Phase 3 (Nov 2026) changes the public API. Details and a migration guide are in `docs/phase3_api_conformance.md`:

- **Signature.** Every method now takes `(c, s, loc, scale)` instead of `(mode, sigma, lower, upper)`. Old positional calls are read as the new parameters and return NaN (scipy's convention for invalid shapes) rather than raising. `bl2.from_bounds(mode, sigma, lower, upper)` reproduces the old parameter sets, and only it accepts `None` bounds.
- **`entropy`** is now BL2's own differential entropy, by quadrature. The original truncated-normal adaptation is retired: it was close for near-symmetric cases (2.860 vs 2.864 nats at mode 5, σ 10 on (−5, 20)) but off by 0.18 nats with the mode near a bound (0.413 vs 0.236 at mode 0.01, σ 1 on (0, 10)). Both figures use the corrected Phase 2 moments in the old formula.
- **`moment(n)`** returns a value for every n (it returned `None` for n ≥ 5).
- **`fit`** is scipy's generic estimator (MLE, or `method='MM'`) over (c, s, loc, scale), with `fc`/`fs`/`floc`/`fscale` to fix parameters. The original least-squares-on-the-empirical-CDF option (`MLE=False`) and fixing `mode` or one bound alone are not carried over.
- **`ignore_checks` and `normed`** are gone. `normalizing_constant(c, s)` returns Z; `pdf × Z` is the old `normed=False` density.
- **Tails.** `sf` is integrated from the right and the effective-support cutoff moved from 1e-14 to 1e-40 of the peak density, so upper-tail `sf` and lower-tail `cdf` keep ~1e-12 relative accuracy down to ~1e-32 (previously ~1e-16 absolute).

Phase 5 (Nov 2026) brings BL into the package. Relative to the notebook's `bounded_lognorm` (the first code cell of `bounded_lognorm.ipynb`), these results change on purpose; everything else (the density, cdf, ppf, and the mean and variance where they were well-conditioned) agrees with the original to its own precision. Details and a migration guide are in `docs/phase5_bl_port.md`:

- **API.** As BL2 in Phase 3: methods take `(c, s, loc, scale)`; `bl.from_bounds(mode, sigma, lower, upper)` reproduces the old parameter sets and alone accepts `None` bounds; scipy hooks replace the public overrides, so `expect`, `interval`, freezing and the rest work; `ignore_checks` is gone; invalid shapes give NaN in the native API and `ValueError` from `from_bounds`. The notebook's instance name is not aliased: `from bounded_lognorm import bounded_lognorm` raises an ImportError with the migration, because under the new signature an old call would run and return NaN or a wrong number.
- **Output order.** The original evaluated the points left of the mode first and concatenated, so `pdf`, `cdf` and `ppf` of unsorted input came back in the wrong order, and `rvs` returned its draws grouped by side of the mode (all left-of-mode draws first). Every method is now elementwise and keeps the input's order and shape; `rvs` returns draws in the order generated. The grouping mattered for Monte Carlo: 10,000 draws each from two independent BLs, paired elementwise, had a correlation of 0.59 with the original (−0.02 now).
- **`rvs`** honours `random_state` (the original drew from the global `np.random`).
- **Higher moments.** The original's closed forms were right but formed central moments as differences of raw moments, which cancel when a piece is narrow: skewness off by 3e-6 and kurtosis by up to 1e-2 at the needle cases (mode 1, σ 0.062 on (0, 100); mode 950, σ 1 on (−1000, 1000)). They are now taken about the mode, and agree with a 50-digit reference to ~1e-12. `moment(n)` returns a value for every n (it returned `None` for n ≥ 5).
- **`entropy`** is BL's own differential entropy, in closed form. The truncated-normal adaptation was off by 0.005–0.04 nats over the test cases (0.04 with the mode near a bound).
- **Tails.** `sf` and `isf` were `1 − cdf` and `ppf(1 − q)`, and `logpdf`/`logcdf`/`logsf` the logs of those; upper-tail values below ~1e-16 were lost and the logs went to −∞. Each tail is now computed from its own end, with relative accuracy down to the float64 range.
- **`fit`** is scipy's estimator over (c, s, loc, scale); fitting in the bounds form with any parameters fixed, by MLE or least squares on the empirical cdf, is `bl.fit_bounds` (shared with BL2, with the midpoint empirical cdf by default; see `docs/v0.2_tails_and_fit_bounds.md`). The original's fallback to least squares on a histogram of the pdf is not carried over.
- **`lognorm_params`** is the stable Phase 2 solver, shared with BL2. The original closed-form quartic root lost digits for small σ/mode (relative error up to ~2e-8 in the density at the needle cases).

### Link audit

Checked July 2026. Changes from the original reference list:

- **OSTI (Ford & McLaren)** — the `servlets/purl` deep link is fragile; the stable record is `osti.gov/biblio/5121047`. Updated below.
- **MathWave** — still live, but the site is unmaintained (last copyright 2017) and the domain is advertised for sale. Archive a copy if you rely on it.
- **`people.sc.fsu.edu/~jpeterson/numerical_quadrature`** — could not be resolved; appears dead. Replaced with a stable textbook reference (Davis & Rabinowitz) plus the SciPy docs.
- **`berkeley-stat159-f17.github.io/.../trapezoid..html`** — malformed URL (doubled period) and could not be resolved. Replaced with the NumPy/SciPy documentation, which is what the code actually uses.
- **`appliedbusinesseconomics.com/files\gvsplndrn01.pdf`** — could not be resolved; the backslash is also not a valid URL path separator. Recommend removing it or replacing it with the specific claim it supported.
- **Towards Data Science / Medium** — unstable and paywalled; retained only as an informal pointer, not a load-bearing citation.
- **Wang et al.** — the ResearchGate link is not canonical; cite the *Transportation Research Record* article of record. DOI 10.3141/2315-07 verified November 2026 against two independent records (Monash University research portal and TRB's TRID database), which agree on the title, all seven authors, issue 2315 (2012) and pages 66–72.

### References

**Distributions and statistics**

1. Log-normal distribution. https://en.wikipedia.org/wiki/Log-normal_distribution
2. Normal distribution. https://en.wikipedia.org/wiki/Normal_distribution
3. Truncated normal distribution. https://en.wikipedia.org/wiki/Truncated_normal_distribution — basis for the original `entropy` adaptation (replaced for BL2 in Phase 3 and for BL in Phase 5); its moment formulas also underlie the Phase 5 partial moments.
4. Mode (statistics). https://en.wikipedia.org/wiki/Mode_(statistics)
5. Probability density function. https://en.wikipedia.org/wiki/Probability_density_function
6. Cumulative distribution function. https://en.wikipedia.org/wiki/Cumulative_distribution_function
7. Quantile function. https://en.wikipedia.org/wiki/Quantile_function
8. Differential entropy. https://en.wikipedia.org/wiki/Differential_entropy
9. Inverse transform sampling. https://en.wikipedia.org/wiki/Inverse_transform_sampling
10. NIST/SEMATECH e-Handbook of Statistical Methods, §1.3.5.11 (measures of skewness and kurtosis). https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
11. Raiford, T. E. "Skewness of Combined Distributions." *Journal of the American Statistical Association*, vol. 37, no. 219, 1942, pp. 391–393. JSTOR, https://www.jstor.org/stable/2279009
12. Wang, Y., Dong, W., Zhang, L., Chin, D., Papageorgiou, M., Rose, G., and Young, W. "Speed Modeling and Travel Time Estimation Based on Truncated Normal and Lognormal Distributions." *Transportation Research Record*, no. 2315, 2012, pp. 66–72. https://doi.org/10.3141/2315-07
13. Ford, C. E., and McLaren, R. A. *Methods for Obtaining Distributions of Uranium Occurrence from Estimates of Geologic Features.* Union Carbide Corp. for the U.S. Dept. of Energy, 1980. OSTI, https://www.osti.gov/biblio/5121047

**Reparametrization (mode, σ → μ_N, σ_N)**

14. Exact quartic root adapted from the Wolfram Alpha result for `solve[g^4 - g^3 - sigma^2/m^2] for g`. https://www.wolframalpha.com (accessed 21 Oct. 2020).
15. Numerical fallback adapted from Warren Weckesser's answer at https://stackoverflow.com/questions/41464753 (accessed 21 Oct. 2020).
16. Scalar-coercion idiom from https://stackoverflow.com/questions/67097840

*Stack Overflow contributions are licensed CC BY-SA. Retaining the attributions above satisfies the attribution requirement; if this notebook is redistributed, note the share-alike term.*

**Numerical methods**

17. Davis, P. J., and Rabinowitz, P. *Methods of Numerical Integration*, 2nd ed. Academic Press, 1984. — Gauss–Legendre and composite quadrature.
18. Fritsch, F. N., and Carlson, R. E. "Monotone Piecewise Cubic Interpolation." *SIAM Journal on Numerical Analysis*, vol. 17, no. 2, 1980, pp. 238–246. — the PCHIP scheme used for the cached CDF/PPF grid.
19. `scipy.stats.rv_continuous`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
20. `scipy.optimize.curve_fit`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html — used by the original CDF-curve `fit` (not carried into Phase 3).
21. `scipy.interpolate.PchipInterpolator`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
22. `scipy.integrate.cumulative_trapezoid`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html
23. `numpy.polynomial.legendre.leggauss`. https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html
24. `functools.lru_cache`. https://docs.python.org/3/library/functools.html#functools.lru_cache

**Curve fitting to data**

25. MathWorks. "Fitting a Univariate Distribution Using Cumulative Probabilities." https://www.mathworks.com/help/stats/examples/fitting-a-univariate-distribution-using-cumulative-probabilities.html
26. "PDF vs CDF fitting." Mathematics Stack Exchange. https://math.stackexchange.com/questions/2365267
27. MathWave Technologies. "Distribution Fitting — Using Distribution Graphs." http://www.mathwave.com/articles/distribution-fitting-graphs.html *(site unmaintained; archive if relied upon)*

**Informal / superseded (retained for provenance only)**

28. Rejection-sampling approach, retained as a commented alternative in `rvs`: https://stackoverflow.com/questions/41470070

**Phase 5 (BL closed forms)**

29. Mills ratio. Abramowitz, M., and Stegun, I. A. *Handbook of Mathematical Functions*, §26.2 (normal probability function); computed here as `scipy.special.erfcx`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfcx.html
30. Stirling numbers of the second kind, as the coefficients of (eˣ − 1)ᵏ/k!. Graham, R. L., Knuth, D. E., and Patashnik, O. *Concrete Mathematics*, 2nd ed., §6.1 (Stirling numbers) and §7.4 (exponential generating functions). Addison-Wesley, 1994. — the series form of the BL partial moments for narrow pieces.
