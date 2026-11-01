## Attribution & References

### Provenance

This notebook contains work of three distinct kinds. They are separated below so that credit — and responsibility for errors — lands in the right place.

**A. Original work — Ted Teske (2020–2024), sole author**

- The bounded lognormal (BL) and bounded lognormal 2 (BL2) distributions themselves: the piecewise construction (eq. 2) and the geometric-mean construction (eq. 3).
- The mode/σ reparametrization `lognorm_params`, including the quartic root formulation.
- The `rv_continuous` subclass design and its full public API: `pdf`, `cdf`, `ppf`, `rvs`, `stats`, `moment`, `entropy`, `nnlf`, `fit`, and the supporting helper functions.
- The adaptation of the truncated-normal entropy formula to BL2.
- All example usage, plots, and the empirical comparison to the normal distribution.

**B. Collaborative work — Ted Teske with Claude (Anthropic), 2026**

- *Phase 1 CDF analysis:* the demonstration that the BL2 CDF admits no elementary closed form, via the observation that no change of variable linearizes both `ln x` and `ln(upper − x)` simultaneously (they are coupled by `e^w + e^v = u`); together with the catalogue of tractable special cases and the saddle-point route.
- *Performance refactor of the class:* hoisting the shape parameters and normalizing constant `Z` out of `pdf` (with `lru_cache`); vectorized composite Gauss–Legendre quadrature; restriction of all integration to an *effective support* located by bisection outward from the mode; the cached PCHIP cumulative grid backing `ppf`/`rvs`/`median`.
- The regression test cell.
- *Phase 2 math verification and fixes:* proofs that the mode of BL2 is exactly `mode` and that BL2 is C∞ (BL is only C¹), that Z ≤ 1 (a Bhattacharyya coefficient), and of the effective-support truncation bound; a stable, scale-free solver for the `lognorm_params` quartic (same equation, new numerics); mode-graded quadrature panels; moments by quadrature in place of the grid sum; argument guards; regression Suite D against a 30-digit reference. See `docs/phase2_math_verification.md`.

**C. Third-party work — cited below**

Attribution suggestion, not a rule: the usual convention in published work is that an AI system is **not** listed as an author, since authorship carries accountability an AI cannot hold. If you publish this, the conventional form is to claim the work as your own and add an acknowledgement of AI assistance — e.g. *"Portions of the numerical implementation were developed with assistance from Claude (Anthropic); the author reviewed and verified all derivations and results."* Note also that some journals now require disclosure of AI use. Category B above is written to give you the specifics such a disclosure would need.

A caveat worth recording, in the same spirit: the Phase 1 non-closed-form result is an argument plus a symbolic check, **not a proof**. It shows no *elementary* antiderivative was found and explains structurally why one is not expected. It does not rule out a closed form in terms of special functions. Treat it as a well-supported working conclusion.

### Behavioural changes from the original

*Correction to an earlier version of this note.* It previously said the original's unnormalized density carried `1/(2π·√(s₁s₂))`. It did not: the original computed `sqrt(lognorm_pdf × lognorm_pdf)`, which gives the correct `1/√(2π·s₁s₂)`. The wrong constant existed only in an intermediate draft of the 2026 refactor and was fixed before release, so `normed=False` output is unchanged from the original.

Phase 2 (Nov 2026) intentionally changes these results:

- **`mean`, `var`, `std`, `stats`, `moment`, `entropy`.** The original grid-sum moments were biased (weights of `(upper − lower)/n` instead of `/(n − 1)`, never renormalized), applied a sample-skewness correction to a population moment, and aliased narrow peaks on a fixed 1001-point grid. They are now integrated by quadrature and match a 30-digit reference to ~1e-12. Example: `var(100, 1)` was 2.68; it is now 0.99999.
- **Extreme parameters.** sigma/mode outside roughly 1e-5 to 1e2 (sigma → 0, very wide bounds, mode very near a bound) previously lost accuracy or returned NaN; results there now agree with the reference down to the float64 resolution floor.
- **Invalid inputs raise.** Infinite bounds, and invalid parameters passed with `ignore_checks=True`, now raise `ValueError` instead of returning NaN or a plausible-looking number. A `RuntimeWarning` flags sigma below 1e-8 of the coordinate scale.

### Link audit

Checked July 2026. Changes from the original reference list:

- **OSTI (Ford & McLaren)** — the `servlets/purl` deep link is fragile; the stable record is `osti.gov/biblio/5121047`. Updated below.
- **MathWave** — still live, but the site is unmaintained (last copyright 2017) and the domain is advertised for sale. Archive a copy if you rely on it.
- **`people.sc.fsu.edu/~jpeterson/numerical_quadrature`** — could not be resolved; appears dead. Replaced with a stable textbook reference (Davis & Rabinowitz) plus the SciPy docs.
- **`berkeley-stat159-f17.github.io/.../trapezoid..html`** — malformed URL (doubled period) and could not be resolved. Replaced with the NumPy/SciPy documentation, which is what the code actually uses.
- **`appliedbusinesseconomics.com/files\gvsplndrn01.pdf`** — could not be resolved; the backslash is also not a valid URL path separator. Recommend removing it or replacing it with the specific claim it supported.
- **Towards Data Science / Medium** — unstable and paywalled; retained only as an informal pointer, not a load-bearing citation.
- **Wang et al.** — the ResearchGate link is not canonical. Cite the *Transportation Research Record* article of record. I have **not** filled in a DOI, because I could not verify one and will not guess at it; please look it up.

### References

**Distributions and statistics**

1. Log-normal distribution. https://en.wikipedia.org/wiki/Log-normal_distribution
2. Normal distribution. https://en.wikipedia.org/wiki/Normal_distribution
3. Truncated normal distribution. https://en.wikipedia.org/wiki/Truncated_normal_distribution — basis for the `entropy` adaptation.
4. Mode (statistics). https://en.wikipedia.org/wiki/Mode_(statistics)
5. Probability density function. https://en.wikipedia.org/wiki/Probability_density_function
6. Cumulative distribution function. https://en.wikipedia.org/wiki/Cumulative_distribution_function
7. Quantile function. https://en.wikipedia.org/wiki/Quantile_function
8. Differential entropy. https://en.wikipedia.org/wiki/Differential_entropy
9. Inverse transform sampling. https://en.wikipedia.org/wiki/Inverse_transform_sampling
10. NIST/SEMATECH e-Handbook of Statistical Methods, §1.3.5.11 (measures of skewness and kurtosis). https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm
11. Raiford, T. E. "Skewness of Combined Distributions." *Journal of the American Statistical Association*, vol. 37, no. 219, 1942, pp. 391–393. JSTOR, https://www.jstor.org/stable/2279009
12. Wang, Y., et al. "Speed Modeling and Travel Time Estimation Based on Truncated Normal and Lognormal Distributions." *Transportation Research Record*, no. 2315, 2012, pp. 66–72. [DOI to be verified.]
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
20. `scipy.optimize.curve_fit`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
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
