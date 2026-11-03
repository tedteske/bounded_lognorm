# Phase 7 — An Interval for a Fitted Bound

**Status:** `profile_bound` and `fit_bounds(..., truncate=...)` implemented for both families on branch `phase7-bound-intervals`, proposed as 0.3.0 (not released). Tests: Suite P (`tests/test_profile.py`), mutation-checked. The sprints notebook has a new Part 5. The simulation scripts are not in the repository; their designs, seeds and results are below.

**Epistemic labels**, as in the earlier notes: *Proof* = a complete argument. *Argument* = a reasoned case that is not a complete proof. *Numerical* = measured, on the stated simulations or data; true for the cases run, not a theorem. Monte Carlo standard errors are given where they matter: about ±0.007 on a coverage of 0.95 from 1000 replicates, ±0.010 from 500.

**Families.** BL is the piecewise form (`bl`), BL2 the smooth one (`bl2`). They are different distributions, and every number below says which.

---

## 1. The question

Phase 6 left the men's 100m bound without an interval. Least squares collapses onto the record (9.63 s). The bootstrap cannot reach past the sample's maximum (Bickel & Freedman 1981). MLE puts the bound just past the record, 9.61 s (BL) and 9.62 s (BL2), but from fits that describe the data poorly.

Phase 7 asked three things:

1. Is a profile-likelihood interval for a bound calibrated, given that a bound is a support parameter?
2. How should the slow outliers that distort the MLE fits be handled?
3. Should this go in the package?

**Answers in brief:**

1. Yes, two-sided, with the chi-squared(1) critical value, where the model is right and the bound is identified. Coverage was 0.94–0.96 at every sprints-like truth, for both families and at n = 50 to 781. One-sided it is lopsided, and it under-covers modestly (0.92) in one weakly identified case. A parametric-bootstrap calibration is offered for those cases (§3.4).
2. Use a likelihood conditional on the fast runs: truncation at a threshold fixed in advance. Freeing the lower bound, trimming, and a robust objective all leave the fast end poorly fitted or collapse onto the record. With truncation, the men's data identify no upper bound at all, in either family. The profile likelihood then gives a one-sided statement: the best possible time is faster than about 9.4 s (§4–5).
3. In the package, as 0.3.0, because the method is general and needs the package's fitting machinery (§6).

**The men's conclusion changes.** Phase 6's "at or just past the record" rested on fits driven by a few very slow runs. §5 has the details.

## 2. Is the bound a regular parameter? (Argument)

Wilks' theorem (LR → χ²₁) needs the model to be regular in the parameter. A support parameter often is not. For a density that behaves like c(b − x)^(α−1) at its endpoint b, Smith (1985) shows three regimes:

- α > 2: MLE asymptotically normal and efficient
- 1 < α ≤ 2: non-standard rates
- α < 1: no local maximum

**Both families vanish faster than any power at each bound (Proof).** Take BL, upper piece. In standard form, f(y) = f₀ exp(−v²/(2a₂²)) with v = ln((1 − y)/(1 − c)), and v → −∞ as y → 1. Then f(y)/(1 − y)ᵏ = f₀ (1 − c)^(−k) exp(−v²/(2a₂²) − kv) → 0 for every k.

BL2's density is the normalized geometric mean √(f₁f₂). Here f₂ is the lognormal reflected at the upper bound, which vanishes like the above. Near the upper bound, f₁ is a lognormal density at a positive argument, so it is finite. So BL2 also vanishes faster than any power. The lower bound is the mirror image.

So formally α = ∞. Differentiating ∫ f dx under the integral sign with respect to the bound produces a boundary term f(b) = 0, and likewise for the second derivative. The score for b has finite variance: near the bound it grows like ln(b − x)/(b − x), and its square times f is integrable after the substitution b − x = e^(−t). So the information identity holds and the bound behaves like an ordinary parameter as n → ∞.

That is an argument, not a proof. Smith's theorem assumes f = (b − x)^(α−1) g(b − x) with g positive at 0, which these densities do not satisfy. The other classical conditions were not checked one by one. The argument also says nothing about how large n must be, and that is what decides calibration here.

**Two finite-sample features (Numerical):**

- **The near end.** The profile likelihood falls without limit as the bound approaches the sample extreme. This holds in every case run: the density at the largest observation goes to zero, and the other parameters cannot compensate. So the interval's near end is always strictly beyond the data, never at it. That is unlike the bootstrap, and it is what makes a bound at or near the sample extreme tractable.
- **The far end.** As the bound moves away, the upper piece tends to a fixed limit shape: BL's reflected lognormal becomes nearly normal with the same sigma. So the profile likelihood levels off at a finite value. If that plateau is within the critical value of the maximum, the interval is open on the far side: the data put no limit on how far away the bound is. If the plateau is the maximum, the estimate itself runs away. Both happen with the sprints data. `profile_bound` reports such an end as `inf`. At an infinite bound the true value sits on the edge of the parameter space, where the LR limit is a mixture (Self & Liang 1987), so the χ² value is conservative there.

## 3. Calibration by simulation (Numerical)

### 3.1 Design

For each truth, the procedure is:

- Draw a sample of the stated size.
- Fit by MLE with the lower bound fixed at 0, and the mode fixed where stated: the women's, at the true mode.
- Compute LR at the true bound: 2 × (profile nnlf at the truth − maximum nnlf).
- Coverage is P(LR ≤ 3.841), the 95% χ²₁ value.

The maximum is taken over two starts: the default, and the profile solution at the truth. The truncated runs add a third, far out. No replicate had a profile point beating the full fit by more than rounding. Seeds are `crc32(name)·1000 + i`.

Each truth is a fit to the sprints data, so the simulations sit in the part of parameter space that matters:

- **Full data:** the MLE fits (lower bound 0), and for the women also the LSQ-step fits, whose bound sits further from the data.
- **Truncated:** constrained conditional fits to the faster half at a chosen bound. For the men, at 9.0 s and 9.4 s. For the women, at 10.0 s and 10.4 s. Each simulated sample is fitted above its own median.
- **Small samples:** n = 50 and 100.

### 3.2 Two-sided coverage

Nominal 0.95. All are two-sided; the gap between the truth and the sample maximum is given in m/s.

**Full data, BL** (1000 replicates each):

- men, n = 781, MLE truth: 0.947 (truth 0.08 beyond the sample maximum, median)
- women, n = 691, MLE truth: 0.957 (gap 0.14)
- women, n = 691, LSQ-step truth: 0.954 (gap 0.26)

**Full data, BL2** (500 replicates each):

- men, MLE truth: 0.958 (gap 0.06)
- women, MLE truth: 0.960 (gap 0.07)
- women, LSQ-step truth: 0.942 (gap 0.11)

**Truncated at the sample median, BL** (500 replicates each):

- men, truth 9.0 s: 0.938 (gap 0.81; the estimate ran away in 10% of samples)
- men, truth 9.4 s: **0.920** (gap 0.39; 95% quantile of LR 4.69)
- women, truth 10.0 s: 0.956 (gap 0.65)
- women, truth 10.4 s: 0.948 (gap 0.30)

**Small samples, BL, full data:**

- women's LSQ-step truth, n = 50: 0.950
- [[S2, S3, S4]]

At 90%, the full-data and women's truncated cases covered 0.89–0.91. The men's truncated cases covered 0.884 and 0.854.

So the χ² interval holds its level, within Monte Carlo error, wherever the model is right and the bound is identified. That includes n = 50, and bounds 0.06 m/s from the data, closer than any of the sprints fits. The one clear exception is the men's truncated truth at 9.4 s, 2.5 standard errors below nominal. It is the weakly identified case nearest the real men's data.

### 3.3 One-sided: lopsided

Split by the sign of b̂ − b₀, misses are not shared evenly between the ends. The interval stops short on the data's side (the truth lies beyond its far end) 3.2–5.8% of the time. It overshoots on the far side 0.4–2.6% of the time. Each would be 2.5% if symmetric.

Examples:

- BL men: 4.1% vs 1.2%
- BL2 men: 3.8% vs 0.4%
- BL men, truncated at 9.4 s: 5.8% vs 2.2%

This is the familiar skewness of the signed root for a parameter bounded on one side by the data, here the sample maximum. So `profile_bound` reports a two-sided interval only. A one-sided limit is not offered, since χ² at level 0.90 would not give a 95% one-sided one.

### 3.4 Parametric-bootstrap calibration

The 95% quantile of LR at each truth is exactly what a parametric bootstrap from that truth estimates. Across the cases above it ranged from 3.45 to 4.69, against 3.84.

In practice the bootstrap is drawn from the *fitted* model, not the truth. `profile_bound(calibration='bootstrap')` does that: n_boot samples from the fit, with the same size, window and fixed parameters, and LR at the fitted bound. Whether it is calibrated was checked with a nested simulation. For each outer sample, the bootstrap critical value is computed from that sample's own fit, and LR at the truth is compared with it:

- [[N1]] BL men, full data, MLE truth, 200 outer × 100 inner
- [[N2]] BL women, LSQ-step truth, n = 50, 200 × 100
- [[N3]] BL men, truncated at the median, truth 9.4 s, 150 × 60

[[conclusion on bootstrap]]

The bootstrap needs a finite estimate to simulate from. When the bound runs away, as it does for the men's truncated fits, it is not available, and `profile_bound` says so. The open far end is then a statement in itself.

## 4. The slow outliers (Numerical)

With the lower bound fixed at 0, the MLE has to give density to a few very slow runs: men down to 50.94 s (1.96 m/s), women to 14.82 s. With one sigma on each side of the mode, it can only do so by widening the whole distribution. That leaves the fast end poorly fitted, and the bound is then set by the shape of the fit.

To measure what matters, the fits are compared by their **KS distance on the faster half of the data**: the fitted distribution conditional on x > median, against the runs above the median. All fits have lower bound 0; the women's have the mode fixed at 8.834 m/s unless stated.

**Men, best possible time (fast-half KS), BL / BL2:**

- MLE, all data: 9.61 s (0.267) / 9.62 s (0.257)
- LSQ, midpoint ECDF: collapses onto 9.63 s (0.113 / 0.111)
- MLE, lower bound free: the lower bound runs away (−8·10⁴, at the cap); 9.59 s (0.202) / 9.60 s (0.191)
- MLE, Tukey far-out trim (drops 8 runs slower than 12.09 s): 9.51 s (0.120) / 9.56 s (0.116)
- Robust, density power divergence (Basu et al. 1998), β = 0.1: 9.57 s (0.120) / 9.60 s (0.118)
- Robust, β = 0.3 and 0.5: collapse onto 9.63 s (0.087–0.097 / 0.083–0.095)
- Truncated likelihood, faster than the lower quartile, median or upper quartile: **not identified**; the bound runs away in both families (0.044–0.074 / 0.044–0.078)

**Women, mode fixed, BL / BL2:**

- MLE, all data: 10.57 s (0.117) / 10.67 s (0.129)
- LSQ, midpoint: 10.31 s (0.152) / 10.57 s (0.134)
- MLE, lower bound free: the lower bound runs away; 10.55 s (0.111) / 10.66 s (0.122)
- MLE, trimmed (10 runs slower than 14.05 s): 10.55 s (0.111) / 10.66 s (0.121)
- DPD, β = 0.1 / 0.3 / 0.5: BL 10.56 / 10.53 / 10.42 s (0.114 / 0.103 / 0.085); BL2 10.67 / 10.68 / 10.66 s (0.122 / 0.102 / 0.080)
- Truncated, faster than the median: 10.12 s (0.055) / 10.35 s (0.055)
- Truncated, faster than the upper quartile: 10.31 s (0.063) / 10.44 s (0.062)
- Truncated, faster than the lower quartile: 8.85 s (0.044) / 9.38 s (0.042). This window still includes runs below the mode, which the single sigma must reach.

**Women, mode free:**

- MLE: 10.66 s (BL) and 10.69 s (BL2), fast-half KS 0.17
- Truncated at the median: 10.52 s (BL) and 10.53 s (BL2), KS 0.036 for both. The mode moves to 8.63–8.70 m/s, below the window.
- Truncated at the upper quartile, BL: the mode goes to 0, a degenerate fit.

**Reading.** Freeing the lower bound does nothing useful: the data cannot place it, and the fast end stays poorly fitted. Trimming a stated handful of runs and mild down-weighting help a little. Strong down-weighting behaves like least squares and collapses onto the record. The truncated likelihood fits the fast end best, by a factor of 2–6 in KS distance. It does so with no assumption about the slow runs: they are not modelled, so they cannot distort the fit.

Its threshold is a choice. The analysis uses the median and the upper quartile, fixed before fitting, and reports both. That is the option adopted (§6).

## 5. What the sprints data say now (Numerical)

95% profile-likelihood intervals for the best possible time, two-sided, χ² calibrated. An open end is shown as 0 s: no limit on speed.

**Men, BL:**

- All data: estimate 9.61 s, interval [9.58, 9.62] s. Narrow, but from the poorly fitting MLE.
- Faster than the median (387 runs): estimate none (runs away), interval [0, 9.38] s
- Faster than the upper quartile (190 runs): estimate none, interval [0, 9.57] s

**Men, BL2:** [[BL2 men]]

**Women (mode fixed), BL:**

- All data: estimate 10.57 s, interval [10.50, 10.62] s
- Faster than the median: estimate 10.12 s, interval [0, 10.58] s
- Faster than the upper quartile: estimate 10.31 s, interval [0, 10.63] s

**Women, BL2:** [[BL2 women]]

**Men.** Using the fast runs alone, neither family identifies a best possible time. The likelihood rises steadily as the bound moves away. The profile does rule out bounds close to the record:

- LR is 8.3 at 9.55 s
- 6.3 at 9.50 s
- 4.1 at 9.40 s

So 9.55 s is rejected outright (BL, median threshold).

Under the weakest-calibrated simulation (§3.2, 95% quantile 4.69), the near end would be about 9.43 s instead of 9.38 s. With only the fastest quarter, the near end is 9.57 s. Fewer runs say less, but nothing different.

Is a runaway estimate plausible under a finite truth? In the simulation from a 9.0 s truth, 10% of samples ran away; from 9.4 s, 0.4%. The men's data look like the first.

**This replaces Phase 6's "at or just past the record."** That phrase came from fits that either collapse onto the record (least squares) or fit the fast end poorly because of a few very slow runs (MLE on all data). The defensible statement is now: the data put the best possible time below about 9.4 s, below the 9.58 s world record, and set no limit on how far below. A bounded lognormal fitted to the fast runs cannot locate a best possible time; it can only say the bound is well past the record.

**Women.** The truncated fits do identify a bound: 10.12 s (BL) and 10.35 s (BL2) at the median threshold. But the likelihood is nearly flat beyond it, so the intervals are open on the fast side too. The data say faster than about 10.6 s, below the 10.71 s fastest run and beyond Griffith-Joyner's 10.49 s record of 1988. The mode-fixed caveat of Phase 6 still applies.

**The data cover 1996–2016,** not 2004–2016 as the dataset's title and Phase 6 said: 1996 has 106 men's and 56 women's 100m runs, and 2000 has 160 and 135. The analysis is unaffected, but the notebook's description is corrected.

## 6. In the package (0.3.0, proposed)

**Why the package and not only the notebook.** The method is general: any `fit_bounds` user who reports a bound wants an interval, and the bootstrap fails exactly where it is most needed. It also depends on `fit_bounds`' machinery: the transforms, the caps that define "runs away", fixed parameters, warm starts. A notebook copy would duplicate all of that. The robust objective stays in the notebook: it is shown for comparison, and it was not the option adopted.

**API.**

- `dist.profile_bound(data, bound='upper'|'lower', level=0.95, *, calibration='chi2'|'bootstrap', n_boot=200, random_state=None, truncate=None, fmode=..., fsigma=..., flower=..., fupper=..., gmode=..., ..., grid=None)`, on `bl` and `bl2`. MLE only. Returns a `BoundProfile` with:
  - `.estimate`, `.interval` (±inf for an open end)
  - `.fit`, `.critical`, `.level`, `.calibration`
  - `.profile` (every point evaluated), `.lr_boot`, `.success`, `.message`
- `fit_bounds(..., truncate=(lo, hi))`, MLE only: the likelihood conditional on lo < x < hi, either end open. It records `.truncate`, `.n_used` and `.n_excluded`. Fits without it are unchanged.
- `BoundsFit.at_cap` and `.collapsed` name the parameters that ran away and the bounds that collapsed. Previously these were only in `.message`.

**Algorithm.** The search runs in u = log(distance of the bound from the data's edge):

- Step outward from the estimate by factors of 2, growing 1.5× per step, until LR passes the critical value.
- Then `brentq` on LR(u) − critical, to 1e-6 in u.
- Each profile fit is warm-started from the nearest point already evaluated.
- A profile point that beats the full fit triggers a refit from there, and the search is repeated.
- The far end is open when LR stays below the critical value out to fit_bounds' runaway zone (more than 10³ data ranges away). The near end is the data's edge only if LR stays below the critical value to within 1e-12 of the range; no case did.

The profile is assumed to fall away on each side without a second dip. That held in every case run, and it is documented.

**Cost.** Full data: about 15 fits; 1 s for BL and 10–15 s for BL2. Truncated: 2 s for BL and 1–2 min for BL2, since BL2's tail probability needs quadrature at every trial point.

**Tests: Suite P** (`tests/test_profile.py`, 24 tests). Each is checked against independent computations:

- the conditional likelihood by hand and by quadrature (P1)
- data outside the window having exactly no influence (P2)
- recovery of a truth (P3)
- bad input (P4, P11)
- each interval end sitting at the critical value, by separate fits, with LR below it just inside and above just outside (P5)
- nested levels (P6)
- an open end (P7)
- the lower bound as the mirror image of the upper (P8)
- the bootstrap's critical value, ends and reproducibility (P9)
- the refit of a full fit that stopped short (P10)

Mutation checks are listed in the file header. Each change makes the listed tests fail:

- the truncation term's sign
- data outside the window not set aside
- a wrong two-sided window
- `logcdf` in place of `logsf`
- χ² with 2 degrees of freedom
- an open end reported at the cap
- the sign not flipped for `lower`
- the bootstrap quantile at 1 − level
- no refit

Full suite: [[N passed]].

## 7. Phase 6's Colab check

Ted ran both Phase 6 use-case notebooks (`Applications_100m_Sprints.ipynb`, `Applications_Monte_Carlo.ipynb`) top to bottom in Google Colab on 2026-11-02, installing bounded-lognorm 0.2.1 from PyPI. The results were as expected. This is recorded in `docs/phase6_use_cases.md` §6.

## 8. Still open

- **Release.** 0.3.0 is not released. Until it is, the sprints notebook installs the package from this branch, so it runs in Colab for review. At release that line becomes `%pip install -q bounded-lognorm==0.3.0`, and `CITATION.cff` gets the version and date.
- **One-sided limits.** Not offered (§3.3). A calibrated version would bootstrap the signed root per side.
- **The truncation threshold.** It is chosen, not estimated. Results are shown at two thresholds. A principled choice is open, for example a threshold-stability plot as in extreme-value analysis.
- **The women's mode.** With the mode free, the median-truncated fits agree across families (10.52–10.53 s). Whether to adopt that over the fixed mode is Ted's call.
- **BL2 profiling under truncation is slow** (1–2 min). Caching the tail integrals across nearby parameter sets would help.

## References

- Basu, A., Harris, I. R., Hjort, N. L., and Jones, M. C. "Robust and efficient estimation by minimising a density power divergence." *Biometrika*, vol. 85, no. 3, 1998, pp. 549–559.
- Bickel, P. J., and Freedman, D. A. "Some Asymptotic Theory for the Bootstrap." *The Annals of Statistics*, vol. 9, no. 6, 1981, pp. 1196–1217.
- Self, S. G., and Liang, K.-Y. "Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions." *Journal of the American Statistical Association*, vol. 82, no. 398, 1987, pp. 605–610.
- Smith, R. L. "Maximum likelihood estimation in a class of non-regular cases." *Biometrika*, vol. 72, no. 1, 1985, pp. 67–90.
- Tukey, J. W. *Exploratory Data Analysis*. Addison-Wesley, 1977. (Outlier fences.)
- Venzon, D. J., and Moolgavkar, S. H. "A method for computing profile-likelihood-based confidence intervals." *Journal of the Royal Statistical Society, Series C*, vol. 37, no. 1, 1988, pp. 87–94.
- Wilks, S. S. "The large-sample distribution of the likelihood ratio for testing composite hypotheses." *The Annals of Mathematical Statistics*, vol. 9, no. 1, 1938, pp. 60–62.
