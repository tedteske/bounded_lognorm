# BL2 — Closed-Form CDF Derivation (Phase 1)

**Distribution:** `bounded_lognorm2` (BL2), the geometric-mean ("non-piecewise") bounded lognormal from `bounded_lognorm2.ipynb`.

**Phase 1 goal:** Determine whether the BL2 CDF admits a closed form; if not, isolate precisely where the integral fails to close and map the tractable routes for Phase 2.

**Result (headline):** No elementary closed form exists in general. The obstruction is structural, not a solver artifact, and is confirmed symbolically. Several exact anchors and closed-form *approximations* do exist and are the natural Phase 2 targets.

---

## 1. Setup and normalization

Shift the domain so the lower bound sits at 0:

$$u \equiv \text{upper}-\text{lower}, \qquad m \equiv \text{mode}-\text{lower}, \qquad x \leftarrow x-\text{lower}, \qquad x\in(0,u).$$

Let $\Lambda(\cdot,\sigma)$ denote `lognorm_params` — the map from (mode, σ) to the underlying normal parameters, obtained by taking the real positive root $g$ of

$$g^4 - g^3 - \sigma^2/\text{mode}^2 = 0, \qquad s_N=\sqrt{\ln g}, \qquad \mu_N=\ln(\text{mode}\cdot g).$$

Define the two parameter sets used by the left and (reflected) right factors:

$$(\mu_1,s_1)=\Lambda(m,\sigma), \qquad (\mu_2,s_2)=\Lambda(u-m,\sigma).$$

The BL2 density is the **geometric mean** of a lognormal placed at the lower bound and a reflected lognormal placed at the upper bound:

$$f_{BL2}(x)=\frac{1}{Z}\,\sqrt{f_L(x;\mu_1,s_1)\;f_L(u-x;\mu_2,s_2)}.$$

Writing out both lognormal pdfs and taking the square root gives the **unnormalized density**

$$\boxed{\;g(x)=\frac{1}{\sqrt{2\pi\,s_1 s_2}}\cdot\frac{1}{\sqrt{x(u-x)}}\;\exp\!\left[-\frac{(\ln x-\mu_1)^2}{4s_1^2}-\frac{(\ln(u-x)-\mu_2)^2}{4s_2^2}\right]\;}$$

and the CDF

$$F(X)=\frac{1}{Z}\int_0^X g(x)\,dx, \qquad Z=\int_0^u g(x)\,dx.$$

*(This matches the notebook exactly: `pdf` builds `sqrt(y_left * y_right)` then divides by `Z = cdf(upper, ..., normed=False)`.)*

---

## 2. Where the integral fails to close

Substitute $w=\ln x$ (so $x=e^w$, $dx=e^w\,dw$). The prefactor $1/\sqrt{x}$ combines with $dx$ to give $e^{w/2}$, and the integral becomes

$$\int \frac{e^{w/2}}{\sqrt{\,u-e^{w}\,}}\;
\underbrace{\exp\!\left[-\frac{(w-\mu_1)^2}{4s_1^2}\right]}_{\text{Gaussian in }w}\;
\underbrace{\exp\!\left[-\frac{\big(\ln(u-e^{w})-\mu_2\big)^2}{4s_2^2}\right]}_{\text{transcendental in }w}\,dw.$$

- The **left** factor is a clean Gaussian in $w$ → would integrate to `erf`.
- The **right** factor contains $\ln(u-e^{w})$, which is **not linear or polynomial in $w$**. The product is therefore not Gaussian, and no elementary antiderivative exists.

The symmetric substitution $v=\ln(u-x)$ fixes the right factor but breaks the left in the identical way. **No single change of variable linearizes both logs simultaneously**, because $\ln x$ and $\ln(u-x)$ are constrained by

$$e^{w}+e^{v}=u,$$

which is not log-linear. This coupling — the reflection through the upper bound — is the entire obstruction. (Note it is *intrinsic* to the reflected-geometric-mean construction; it is not introduced by the `lognorm_params` reparametrization, which only sets the constants $\mu_i, s_i$.)

**Symbolic confirmation.** SymPy returns the integral of $g(x)$ unevaluated in both the general case and the symmetric case $\mu_1=\mu_2,\ s_1=s_2$ — consistent with the argument above rather than a failure to simplify.

---

## 3. Exact anchors and tractable regimes (Phase 2 material)

Although the full CDF is non-elementary, the following are exact or closed-form:

**(a) Single-lognormal limits.** When one factor is effectively constant over the region carrying the mass (e.g. `upper` far from the mode relative to σ), $g$ collapses to a single lognormal and

$$F(X)\to \tfrac12+\tfrac12\,\mathrm{erf}\!\left(\frac{\ln X-\mu_1}{\sqrt2\,s_1}\right),$$

an exact `erf` form usable for the lower tail / limiting checks.

**(b) Normal-approximation regime.** As both bounds recede from the mode (large "spread"), BL2 → Normal(mode, σ²) — the notebook already demonstrates this convergence. This yields a closed-form `erf` approximation that is asymptotically exact in that regime.

**(c) Symmetric special case, $m=u/2$.** Then $(\mu_1,s_1)=(\mu_2,s_2)$ and the density is symmetric about $u/2$. Even without an elementary antiderivative, this gives exact anchors:

$$F(u/2)=\tfrac12, \qquad F(u-x)=1-F(x),$$

halving the quadrature work and providing a correctness check.

**(d) Saddle-point / Laplace expansion (recommended core route).** Expand the "harder" factor's exponent about the dominant saddle $w^\*$ of the full integrand. Each truncation of $\ln(u-e^{w})$ to polynomial order turns the integrand into *Gaussian × polynomial*, which integrates **in closed form** to `erf` × (polynomial in the endpoint). This produces a controlled closed-form series approximation for $F$ with tunable accuracy.

**(e) Series in a coupling parameter.** When $s_2$ is small (right factor sharply peaked) or the two peaks are well separated, expand one factor around its peak and integrate term by term — again Gaussian moments, closed form per term.

---

## 4. Recommended Phase 2

1. Derive the saddle-point closed form from **3(d)**: leading `erf` term + first correction, with an explicit error bound.
2. Add the exact symmetric anchors **3(c)** and single-lognormal limits **3(a)** as boundary conditions / validation.
3. Benchmark the closed-form approximation against the existing `fixed_quad` `cdf` across the notebook's parameter sweeps (varying mode, σ, and bound spread), reporting max abs error and speedup.
4. If a uniform-accuracy closed form proves elusive, fall back to a hybrid: closed-form `erf` backbone (b) with a low-order saddle correction (d).

**Bottom line for Phase 1:** the exact CDF is irreducibly an integral; the productive path is a closed-form *approximation* built on the Gaussian left factor plus a polynomial expansion of the reflected log term, anchored by the exact symmetric and limiting cases.
