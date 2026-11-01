# BL2 — Math Verification and Fixes (Phase 2)

**Status:** verification complete; fixes implemented on branch `phase2-math-fixes` (Nov 2026). Full test suite: 346/346 passing, including the new Suite D.

**Epistemic labels.** *Proof* = a complete argument (SymPy-confirmed where marked). *Numerical* = measured against an independent 30-digit mpmath reference; true for the cases tested, not a theorem. *Sketch* = an argument with a named gap.

**Reference model.** mpmath at 30 digits: the `lognorm_params` quartic solved by bisection on a stable form; integrals by tanh-sinh quadrature over ~130 break points graded toward the mode and both bounds. It shares no numerics with the library.

---

## 1. The mathematics holds

- **Eq. 1, lognormal pdf (Proof, SymPy).** It is the density of e^W, W ~ N(μ, s²): integrates to 1, mode e^(μ−s²), E[X^k] = exp(kμ + k²s²/2), Var = (e^(s²) − 1)e^(2μ+s²).
- **`lognorm_params` (Proof).** With g = e^(s²), mode = m gives μ = ln(m g) and Var = m² g³(g − 1), so SD = σ is g⁴ − g³ − σ²/m² = 0. The quartic is negative on (0, 1] and increasing for g ≥ 1, so it has exactly one positive root and that root exceeds 1. σ is the SD of each untruncated factor, not of BL2. The original closed-form root satisfies the quartic to 160 digits in exact arithmetic (Numerical).
- **Eq. 3 (Proof, SymPy).** √(f_L(a; μ₁, s₁) f_L(b; μ₂, s₂)) equals the fused `_unnorm_pdf` form exactly, with a = x − lower, b = upper − x.
- **Notation.** Eq. 2–3 in the notebook are printed for lower = 0 and without the 1/Z factor; the code shifts by `lower` and normalizes.
- **Record correction.** The original `pdf` already carried 1/√(2π s₁ s₂). The 1/(2π√(s₁s₂)) constant existed only in an intermediate draft of the 2026 refactor; `ATTRIBUTION.md` and the `bl2.py` header now say so.

## 2. Mode and smoothness (Proof)

- log g(a) = ½[h₁(a) + h₂(u − a)]. h₁′(a) = (ln m − ln a)/(s₁² a) and d/da h₂(u − a) = (ln(u − a) − ln(u − m))/(s₂²(u − a)); both have the sign of (m − a). So BL2 is strictly unimodal with its maximum exactly at `mode`. This also justifies the bisection in `_effective_support`.
- BL2 is C∞ on (lower, upper) and extends by zero to a C∞ function on ℝ. BL (Eq. 2) is only C¹ at the mode; its second derivative jumps by −0.119, −0.410 and −0.0044 for (m, σ, u) = (3, 1, 10), (1, 3, 10), (5, 10, 25) (Numerical).

## 3. Normalization (Proof + Numerical)

- Original and refactored code normalize consistently: Z is the integral of g over the support, and `cdf(upper, normed=False)` ≡ Z.
- Truncating to the effective support loses at most 1e-14 · g(mode) · ((a − lower) + (upper − b)) of mass, by unimodality.
- **Z ≤ 1 always** (Cauchy–Schwarz): Z is the Bhattacharyya coefficient of the left and reflected-right lognormals.

## 4. Defects found and fixed

1. **Moments (`_stats`) were wrong.** Weights used /n instead of /(n − 1) and were not renormalized (mean biased toward zero, not shift-equivariant); a sample-skewness correction was applied to a population moment; and a fixed 1001-point grid aliased narrow peaks. Example: `var(100, 1)` returned 2.68. **Fix:** moments integrated with the cdf's graded Gauss–Legendre rule (`_gl_moments`). Now within ~1e-12 of the reference at every dial position.
2. **`lognorm_params` lost precision** for σ/mode outside ~1e-5…1e2 (catastrophic cancellation; NaN beyond ~1e-9 and ~1e6; m⁸ overflow near 1e30). **Fix:** the same quartic solved for t = ln g as 3t + log(expm1 t) = 2 log(σ/mode), strictly increasing in t, bracketed analytically, solved by brentq. Scale-free; ~1e-14 relative from σ/mode = 1e-150 to 1e150.
3. **Quadrature under-resolved a mode near a bound** (four equal panels; Z error 5e-5 at mode 0.01 on (0, 10)). **Fix:** panel edges graded geometrically toward the mode and both bounds, plus equal panels within the effective support (`_graded_edges`), cached once per parameter set (`_panel_table`). The ppf grid uses the union of a uniform grid and panel-refined points.
4. **Guards.** `_argcheck` now encodes finite mode, σ > 0 and lower < mode < upper. Infinite bounds raise (use `None`). Invalid inputs under `ignore_checks=True` raise instead of returning NaN; `fit`'s `curve_fit` wrapper catches this. A `RuntimeWarning` flags σ < 1e-8 of the coordinate scale, where float64 resolution of x − lower limits accuracy.

## 5. Results after the fixes (Numerical)

Dial accuracy over the test suite (cdf vs reference; ppf as cdf(ppf(q)) round-trip):

- fast: cdf 7e-14, ppf 2e-6, cdf 0.4 ms per 1000 points
- balanced: cdf 8e-14, ppf 1e-7, cdf 0.9 ms per 1000 points
- precise: cdf 7e-14, ppf 4e-9, cdf 1.3 ms per 1000 points (ppf grid ~80 ms once per parameter set)
- exact: cdf 8e-14, ppf 9e-14, cdf 1.9 ms per 1000 points, ppf 0.3 ms per quantile

cdf accuracy is now at the reference floor at every position; the dial mainly trades speed and ppf accuracy.

Edge cases (σ → 0, bounds to ±1e8, mode within 1e-6 of a bound): cdf within 6e-9 of the reference at every dial position. The remaining floor is float64 resolution of x − lower.

## 6. Test coverage (Suite D, mutation-checked)

Reverting each fix makes these fail:

- stable `lognorm_params` → D1, D2, D3, D4
- graded panels → A5, D1, D2, D3
- quadrature moments → D1, D5, D7
- `_argcheck` / resolution warning → D6

Infinite-bound and invalid-input rejection is enforced by two layers, so those D6 checks fail only when both are reverted. The equal panels within the effective support are not test-guarded: their measured benefit (~4×) appears only below the float64 resolution floor.

## 7. Still open

- `moment(n)` returns `None` for n ≥ 5.
- `entropy` uses a truncated-normal approximation, not BL2's own entropy.
- As mode → lower the mass does not follow the mode (at mode 1e-8 on (0, 10): median 0.067, mean 0.266); worth a note in the user-facing docs.
- Bounds → ∞ giving SD → σ is a sketch (dominated-convergence step not written out) plus numerical confirmation: true SD 0.814, 0.959, 0.9996, 1.0000 at ±4, ±10, ±100, ±1000.
