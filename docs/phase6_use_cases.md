# Phase 6 — Use Cases on BL and BL2

**Status:** two executed use-case notebooks in `notebooks/`, running on `bounded-lognorm` 0.2.1 from PyPI: `Applications_100m_Sprints.ipynb` (the 2020 notebook, migrated and re-examined) and `Applications_Monte_Carlo.ipynb` (two Monte Carlo cases agreed in advance). Phase 6 exposed one package bug, fixed and released first as 0.2.1 (PR #21; `docs/v0.2_tails_and_fit_bounds.md` §2.1; release record in §6 below). No other package code changed.

**Epistemic labels**, as in the earlier phase notes: *Proof* = a complete argument. *Numerical* = measured, here on the stated data and seeds; true for the cases run, not a theorem. Every number below is printed by the committed notebooks unless marked as measured separately. Those were measured with the original notebook code (`tests/bl_original.py`) or with scripts during this phase.

---

## 1. The 100m-sprints notebook: what survives the migration

The notebook fits a bounded distribution to Olympic 100m average speeds (2004–2016 heats, men n = 781, women n = 691) with the lower bound fixed at 0, and reads the fitted upper bound as the speed of the best possible runner. The original used least squares on the empirical CDF (`fit(MLE=False, flower=0)`, with `fmode` for the women) and ±2 standard errors from the least-squares covariance.

**The port is verified on real data (Numerical).** `bl.fit_bounds(speeds, 'LSQ', flower=0, fmode=..., ecdf='step')` reproduces the women's result exactly: upper bound 9.674 m/s, best possible time 10.34 s, covariance interval [10.25, 10.43] s. For the fictional runners, with a fixed seed, it matches the original code's fit to 4 decimals (measured separately). At the original's men's parameters BL's CDF in the package agrees with the original code's to 1.4e-14 (measured separately).

**Men: 9.44 s does not survive (Numerical).**

- The original's men's fit (mode 9.7526, sigma 0.3157, upper 10.593 m/s, i.e. 9.44 s) is not a least-squares optimum. Its squared error is 0.599. With the upper bound held at the same 10.593 the best fit reaches 0.198, and with it free 0.166. The gradient at the original's point is clearly nonzero (measured separately).
- With SciPy 1.18, the original code from its own default starting point stops at mode = upper = 10.03 m/s, below the fastest run: an invalid fit. Started from `gmode=9.8` or `gupper=10.5` it returns the 9.44 s point (measured separately). So the published number was where curve_fit happened to stop.
- The least-squares optimum is the boundary: the bound collapses onto the fastest speed in the data (Bolt, 9.63 s), for both families and both ECDF conventions. 0.2.0 returned this as a success with a standard error of 7e-21; 0.2.1 reports it (§6).
- MLE places the bound just past the record: 9.61 s (BL) and 9.62 s (BL2), from fits that describe the data poorly (below). The well-fitting least-squares fits and the poorly fitting MLE ones agree on the conclusion. In bootstrap resamples (400 for BL, 200 for BL2), the fitted bound is a median 0.03 s (BL) and 0.02 s (BL2) faster than each resample's own fastest run. In 95% of resamples it is 0.00–0.11 s and 0.01–0.08 s faster.
- A bootstrap of a bound at the sample maximum is not a confidence interval: a resample cannot contain a run faster than the data's fastest. The notebook therefore states the result as **at or just past the record in the data**, as agreed, and drops the original's reading that Bolt could have run 9.44 s.

**Women: the estimate depends on the method, the family and the fixed mode (Numerical).** Best possible time, lower bound 0, mode fixed at the most frequent speed (8.834 m/s):

- BL: 10.34 s (LSQ, step ECDF), 10.31 s (LSQ, midpoint), 10.57 s (MLE)
- BL2: 10.59 s (LSQ, step), 10.57 s (LSQ, midpoint), 10.67 s (MLE)

So the ECDF convention moves it by 0.02–0.03 s, the family by 0.25 s under LSQ and 0.10 s under MLE, and the method by 0.23 s (BL) or 0.08 s (BL2).

**The MLE fits describe the data poorly (Numerical).** Kolmogorov–Smirnov distances between each fit and the data:

- least squares: 0.06–0.09
- MLE: 0.16–0.23

With the lower bound fixed at 0, the likelihood must account for a few very slow runs (men down to 50.94 s, 1.96 m/s) with one sigma below the mode. It widens the whole distribution to do so, and the fitted density's peak falls to about 0.7–1.0 against a data peak near 2. So the MLE estimates of the bound rest on fits that miss the fast end, and least squares, the original's choice, is the better guide here. BL2 has the higher likelihood on both data sets (negative log-likelihood lower by about 90 for the men and 28 for the women), but neither MLE fit is good; under least squares the families fit about equally well. The first draft of the notebook said both MLE fits followed the bulk of the data; looking at the figure showed they did not, and the notebook now prints the KS distances and plots the least-squares fits.

- Bootstrap (mode held fixed, LSQ, step ECDF): BL [10.24, 10.48] s (400/400 identified), BL2 [10.51, 10.70] s (198/200). For BL this is a little wider than the covariance interval, and it does not overlap BL2's.
- With the mode free, the women's LSQ fit also collapses onto the record (found while testing the 0.2.1 fix). Re-taking the sample mode in each resample, as a bootstrap of the whole procedure must, 30 of 400 BL fits fail to identify a parameter and the rest span [9.46, 10.72] s. The interval with the mode held fixed is conditional on that choice; the notebook says so.

**The covariance interval understates the uncertainty (Numerical).** For the fictional runners (sums of 10 uniforms, n = 10,000, lower bound fixed at 0, BL, LSQ, step ECDF):

- over 100 independent samples (seeds 0–99), the fitted bound has mean 10.54 and standard deviation 0.69, ranging from 9.36 to 12.63
- the covariance standard error averages 0.024, 29 times smaller
- ±2 SE intervals contain the true 10 in 8% of samples
- a bootstrap of one sample (seed 0, 400 resamples) gives a standard deviation of 0.48, about two-thirds of the real one; BL2's (100 resamples, 97 identified) gives 0.52

The covariance is s²(JᵀJ)⁻¹, which treats the ECDF residuals as independent. They are not: each ECDF point contains all the points below it. This was already documented in v0.2 §2; the notebook now measures it and uses bootstrap refits instead.

**The fictional runners' bound is not 10 even with unlimited data (Numerical).** Fitted to 10⁶ runners, BL gives 10.49 (LSQ) and 11.13 (MLE); fitted to 2·10⁵, BL2 gives 10.45 and 10.61. A sum of uniforms is not a bounded lognormal, so the fitted bound belongs to the best-fitting member of the family, not to the runners. The original's qualitative point survives: the bound is far above the sample's best runner (8.66 for seed 0), around 10. But the honest uncertainty is about ±0.7, not ±0.03.

**What did not change anything.** The stable `lognorm_params` (Phase 2) moves no number in the notebook. The ECDF convention moves the women's estimate by 0.02–0.03 s and the runners' by about 0.01.

## 2. Keeping BL and BL2 apart

As agreed, both notebooks follow these rules:

- **The headline comparison is each family fitted to the same data.** A same-parameters comparison appears once in each notebook, labelled as such, to show they are different distributions.
- Every printed number carries the family's name, through a helper (`report()` in the sprints notebook). Every frozen distribution is named by family.
- Fixed styles throughout: BL solid blue (`#2a78d6`), BL2 dashed orange (`#eb6834`). The Monte Carlo comparators are dash-dot aqua (beta) and dotted yellow (unbounded lognormal). All four are validated together as a categorical palette (CVD separation and normal-vision floor pass). Aqua and yellow fall below 3:1 contrast on white, so every figure also has a legend, distinct line styles, and printed values.
- The README's "Which one?" stays the reference for choosing between them.

## 3. Where the notebooks live, and CI

- All notebooks moved from the repository root to `notebooks/` (agreed). The two development notebooks (`bounded_lognorm.ipynb`, `bounded_lognorm2.ipynb`) are deprecated and kept for history, each with a note at the top pointing to the package. Their Colab badges were updated to the new path.
- `notebooks/data/2004-2016-Short-Races.csv` is the sprint data, CC0 (Mirshams Shahshahani 2018, doi:10.7302/Z20V8B11), unmodified, with its citation and SHA-256 in `notebooks/data/README.md`. The notebook reads the repository copy first (locally, then from GitHub) and falls back to the Deep Blue link.
- **Executed examples, not tests.** The committed notebooks carry full-size outputs. Their run times are about 9 minutes (sprints, mostly bootstrap refits) and 1 minute (Monte Carlo), measured in a clean Python 3.13 environment with NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.0 and bounded-lognorm 0.2.1 from PyPI.
- A separate workflow, `.github/workflows/notebooks.yml`, executes both use-case notebooks on Python 3.12. It runs by hand and when the notebooks or the workflow change, with reduced sizes (`BOOTSTRAP_B=20`, `REPEATS=10`, `MC_N=20000`). Each notebook installs the pinned release itself (`%pip install bounded-lognorm==0.2.1`), as in Colab, so the job checks the notebooks against the released package, not the checkout. It is not part of the test matrix, which stays fast.
- The environment variables `BOOTSTRAP_B`, `REPEATS` and `MC_N` let a reader shorten a Colab run.

## 4. Monte Carlo case A: a schedule roll-up

Eight tasks in sequence, three-point estimates in days, most likely near the optimistic end (sums: minimums 39, most likely 48, maximums 114). Convention: sigma = (max − min)/6. In BL and BL2 sigma is the spread of each lognormal piece before truncation, so the actual standard deviation is smaller (task 1: sigma 1.33, BL sd 0.73, BL2 sd 0.81, measured separately). 200,000 independent draws per task, seed 1. Four comparators:

- BL
- BL2
- a beta with the same minimum, most likely, maximum and standard deviation as BL
- a lognormal from the minimum with the same mode and sigma and no maximum

**The unbounded comparator (Proof).** The lognormal from the minimum with mode m and standard deviation σ is, by Phase 5 §1, exactly BL's lower piece before truncation, `lognorm_params` solving the same moment conditions. So the comparison isolates the effect of the bound. The notebook solves (eᵗ − 1)e³ᵗ = (σ/m)² itself rather than calling the private `lognorm_params`; the density ratio to BL below the mode is constant to 1e-13 (Numerical).

`from_bounds(mode, sigma, lower, None)` is not this distribution. It moves the upper bound 1000σ away, and BL's upper piece, anchored at that bound, then has a thin, nearly normal right tail. The notebook says so, since it is an easy mistake; it was nearly made here.

**Results (Numerical), total duration:**

- BL: P50 53.0, P80 55.2, P95 57.4, P99 59.3 days; P(over 60 days) 0.0048
- BL2: P50 52.7, P80 55.2, P95 57.7, P99 59.9; P(over 60) 0.0091
- beta, same sd as BL: P80 53.7, P95 56.0; P(over 60) 0.0017
- lognormal, no maximum: P80 59.4, P95 64.5; P(over 60) 0.17; P(over 114, the sum of the maximums) 5e-6, longest draw 128

What they show:

- **The bound matters everywhere,** not only beyond the sum of the maximums: without it P80 is 4 days later.
- **BL vs BL2 hardly matters for the sum:** the same P80, and within 0.3 days up to P95. They part only far out (0.5% vs 0.9% beyond 60 days).
- **The shape matters more than the family:** the beta with BL's standard deviation is 1.5 days earlier at P80.

**Notebook-era draw order (Proof, then Numerical).** The original `rvs` was `ppf` of unsorted uniforms, and the original `ppf` returned left-of-mode results first. So its output was the draws below the mode, in generated order, then those above. Regrouping the package's draws the same way reproduces the original code's output to 4e-13 (measured separately, 20,000 draws). Emulated this way, two independent tasks have a correlation of +0.37, against +0.002 in generated order. The total keeps its mean (53.09 days), but its standard deviation goes from 2.50 to 4.72 days, P95 from 57.4 to 59.0, and P(over 60 days) from 0.5% to 2.2%. Results built from several notebook-era BL variables should be re-run.

## 5. Monte Carlo case B: an input with its mode near a bound

A rework fraction on [0, 0.4], most likely 2%, with a contingency triggered above 10%.

**Same parameters (Numerical):** sigma 5 points.

- P(> 10%): BL 0.044, BL2 0.056 (about a quarter higher)
- 95th percentile: 9.8% and 10.3%
- largest CDF gap: 0.062

With the mode at 0.4% and 0.04% the largest gap is 0.15 and 0.26. But it sits near the mode, so at the 10% threshold the difference shrinks (0.018 vs 0.023, then equal). Parameters are not portable between the families.

**Fitted to the same data (Numerical).** Data are 30 draws from 0.4·Beta(1.5, 11) (mode 1.9%, P(> 10%) = 0.091), a third distribution so that neither family is the truth. Bounds are fixed at 0 and 0.4; mode and sigma are fitted by MLE.

- On seed 0 the families land on different parameters: mode 0.30% for BL and 1.61% for BL2.
- On the same seed they give similar answers: P(> 10%) is 0.033 for BL and 0.041 for BL2.

Over 100 data sets (seeds 1–100), P(> 10%):

- BL minus BL2 on the same data: median −0.004, middle 90% −0.015 to +0.016
- error against the truth: BL median −0.017, middle 90% −0.067 to +0.093; BL2 median −0.007, middle 90% −0.058 to +0.079

With 30 observations the sampling error dominates the family choice, though the families occasionally differ by more.

## 6. Release record: 0.2.1 (Numerical)

- **The fix:** `fit_bounds` reports a free bound that collapses onto the data (`success = False`, a warning, NaN standard error). Detection asks whether moving the bound 1000× closer to its edge worsens the objective. Over 60 fits every collapse changed it by −3e-9 to 0 and every genuine fit by +1e-2 or more. Details and the mutation checks are in `docs/v0.2_tails_and_fit_bounds.md` §2.1. PR #21: 695 passed, 18 skipped.
- **CI on PR #21:** all 10 jobs passed. The macOS job was cancelled once without running, as for 0.2.0, and passed on re-run.
- **PyPI:** merged to master as `89a7ad2`; GitHub release `v0.2.1` published by Ted (this session cannot create releases); publish run passed and uploaded. https://pypi.org/project/bounded-lognorm/0.2.1/
- **Check after release:** `pip install bounded-lognorm==0.2.1` into a clean Python 3.13 environment (NumPy 2.5.4, SciPy 1.18.1) gave 0.2.1, and the collapse case returned `success=False` with a NaN standard error. BL's mean (5.849409133600799), entropy (2.733654201134262) and isf(1e-20) (19.74541097177343), and BL2's mean (5.872972571882167), for mode 5, σ 10 on (−5, 20), are identical to 0.2.0's to every printed digit.
- **Zenodo:** version 0.2.1 is [10.5281/zenodo.23898464](https://doi.org/10.5281/zenodo.23898464); the concept DOI [10.5281/zenodo.23897348](https://doi.org/10.5281/zenodo.23897348) resolves to the latest.

## 7. Still open

- A bound at the sample maximum has no bootstrap confidence interval. Whether to offer anything for that case (a profile-likelihood bound, say) is open; the notebooks state the limitation.
- The women's analysis rests on fixing the mode at the sample's most frequent speed, a fragile estimate for data recorded in hundredths of a second. A better-founded choice (a smoothed mode, or MLE with the mode free) would be a substantive change to the analysis, not a migration, so it is left as noted.
- Case A uses one sigma convention for three-point estimates. Others, such as sigma chosen so that BL's actual standard deviation is (max − min)/6, would change the absolute numbers but not the comparisons.
- From earlier phases: as the mode approaches a bound the mass need not follow the mode (user-facing note).
