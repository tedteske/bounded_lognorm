"""
Independent 50-digit mpmath reference model of the piecewise BL (Eq. 2).

Generates the gold values pasted into tests/test_bl.py (Suite H). It shares
no numerics with the library:

  - lognorm_params by bisection on the quartic g^4 - g^3 - (sigma/m)^2 = 0
    (not the library's brentq on 3t + log(expm1 t)),
  - the density exactly as the notebook defines it, in the bounds
    coordinates (x - lower, upper - x), not the library's standard form:
    left of the mode a lognormal pdf at the lower bound; right of it a
    reflected lognormal pdf at the upper bound scaled by k so the two meet at
    the mode; normalized by Z,
  - Z, moments and entropy by tanh-sinh quadrature over break points graded
    toward the mode and both bounds (not the library's Mills-ratio forms),
  - cdf and sf both by quadrature (each from its own end) AND by the textbook
    lognormal cdf ncdf((ln y - mu)/s); the two must agree to 1e-28 relative
    or the script stops (the worst disagreement is printed at the end; the
    gold values are the closed-form ones),
  - quantiles by bisection/Newton on log F (lower) and log S (upper).

Not collected by pytest (no test_ prefix). Needs mpmath. Run:
    python tests/bl_mpmath_reference.py > gold.txt
Takes a few minutes.

Ted Teske with Claude (Anthropic), 2026 (Phase 5).
"""
import sys

import mpmath as mp

mp.mp.dps = 50


def lognorm_params_ref(m, sigma):
    """(mu, s) of the lognormal with mode m and SD sigma, by bisection on the
    quartic in g = exp(s^2)."""
    m, sigma = mp.mpf(m), mp.mpf(sigma)
    r2 = (sigma / m) ** 2
    lo, hi = mp.mpf(1), mp.mpf(2)
    while hi ** 4 - hi ** 3 - r2 < 0:
        hi *= 2
    for _ in range(400):
        mid = (lo + hi) / 2
        if mid ** 4 - mid ** 3 - r2 < 0:
            lo = mid
        else:
            hi = mid
    g = (lo + hi) / 2
    return mp.log(m * g), mp.sqrt(mp.log(g))


def f_lognorm(y, mu, s):
    if y <= 0:
        return mp.mpf(0)
    return mp.exp(-(mp.log(y) - mu) ** 2 / (2 * s * s)) / (y * s * mp.sqrt(2 * mp.pi))


def F_lognorm(y, mu, s):
    if y <= 0:
        return mp.mpf(0)
    return mp.ncdf((mp.log(y) - mu) / s)


class BLRef:
    """The piecewise BL with (mode, sigma, lower, upper) in the bounds form.
    None bounds follow the package convention: mode -/+ 1000 sigma."""

    def __init__(self, mode, sigma, lower=None, upper=None):
        mode, sigma = mp.mpf(mode), mp.mpf(sigma)
        lower = mode - 1000 * sigma if lower is None else mp.mpf(lower)
        upper = mode + 1000 * sigma if upper is None else mp.mpf(upper)
        self.mode, self.sigma, self.lower, self.upper = mode, sigma, lower, upper
        self.m = mode - lower           # mode measured from the lower bound
        self.r = upper - mode           # mode measured from the upper bound
        self.mu1, self.s1 = lognorm_params_ref(self.m, sigma)
        self.mu2, self.s2 = lognorm_params_ref(self.r, sigma)
        # continuity at the mode (Eq. 2)
        self.k = f_lognorm(self.m, self.mu1, self.s1) / \
            f_lognorm(self.r, self.mu2, self.s2)
        # break points: graded toward the mode (by sigma and geometrically)
        # and toward both bounds
        bp = set()
        for j in range(-12, 13):
            for d in (sigma * mp.mpf(2) ** j,):
                for x in (mode - d, mode + d):
                    if lower < x < upper:
                        bp.add(x)
        for j in range(1, 60):
            for x in (lower + self.m * mp.mpf(2) ** -j,
                      upper - self.r * mp.mpf(2) ** -j,
                      mode - self.m * mp.mpf(2) ** -j,
                      mode + self.r * mp.mpf(2) ** -j):
                if lower < x < upper:
                    bp.add(x)
        self.bp = sorted(bp)
        left = [lower] + [x for x in self.bp if x < mode] + [mode]
        right = [mode] + [x for x in self.bp if x > mode] + [upper]
        self.left_pts, self.right_pts = left, right
        self.ZL = mp.quad(self.g, left)
        self.ZR = mp.quad(self.g, right)
        self.Z = self.ZL + self.ZR
        # cross-check the quadrature against the closed-form lognormal cdf
        zl = F_lognorm(self.m, self.mu1, self.s1)
        zr = self.k * F_lognorm(self.r, self.mu2, self.s2)
        for a, b in ((self.ZL, zl), (self.ZR, zr)):
            self._agree(a, b, "Z", "Z")

    # unnormalized density, exactly Eq. 2
    def g(self, x):
        if x <= self.lower or x >= self.upper:
            return mp.mpf(0)
        if x <= self.mode:
            return f_lognorm(x - self.lower, self.mu1, self.s1)
        return self.k * f_lognorm(self.upper - x, self.mu2, self.s2)

    def pdf(self, x):
        return self.g(mp.mpf(x)) / self.Z

    # closed-form pieces (textbook lognormal cdf) -- used for quantiles
    def cdf_cf(self, x):
        x = mp.mpf(x)
        if x <= self.lower:
            return mp.mpf(0)
        if x >= self.upper:
            return mp.mpf(1)
        if x <= self.mode:
            return F_lognorm(x - self.lower, self.mu1, self.s1) / self.Z
        return 1 - self.sf_cf(x)

    def sf_cf(self, x):
        x = mp.mpf(x)
        if x >= self.upper:
            return mp.mpf(0)
        if x <= self.lower:
            return mp.mpf(1)
        if x >= self.mode:
            return self.k * F_lognorm(self.upper - x, self.mu2, self.s2) / self.Z
        return 1 - self.cdf_cf(x)

    # quadrature from each end -- the independent check
    # mp.quad's stopping test is effectively absolute, so a deep-tail
    # integral is rescaled by the density at its inner end before quadrature.
    def cdf_quad(self, x):
        x = mp.mpf(x)
        pts = [p for p in self.left_pts + self.right_pts[1:] if p < x] + [x]
        w = self.g(x) or mp.mpf(1)
        return mp.quad(lambda t: self.g(t) / w, pts) * w / self.Z

    def sf_quad(self, x):
        x = mp.mpf(x)
        pts = [x] + [p for p in self.left_pts + self.right_pts[1:] if p > x]
        w = self.g(x) or mp.mpf(1)
        return mp.quad(lambda t: self.g(t) / w, pts) * w / self.Z

    worst = mp.mpf(0)     # worst quadrature-vs-closed-form disagreement

    def _agree(self, a, b, what, x):
        if b != 0:
            d = abs(a / b - 1)
            BLRef.worst = max(BLRef.worst, d)
            if d > mp.mpf('1e-28'):
                sys.exit(f"{what} mismatch at {x}: {a} vs {b}")

    def checked_cdf(self, x):
        a, b = self.cdf_quad(x), self.cdf_cf(x)
        self._agree(a, b, "cdf", x)
        return b

    def checked_sf(self, x):
        a, b = self.sf_quad(x), self.sf_cf(x)
        self._agree(a, b, "sf", x)
        return b

    # quantiles: bisection in x on log F or log S (monotone), then polished
    def _invert(self, logq, upper_tail):
        lo, hi = self.lower, self.upper
        fn = (lambda x: mp.log(self.sf_cf(x))) if upper_tail else \
             (lambda x: mp.log(self.cdf_cf(x)))
        for _ in range(400):
            mid = (lo + hi) / 2
            val = fn(mid)
            if upper_tail:
                if val > logq:
                    lo = mid
                else:
                    hi = mid
            else:
                if val < logq:
                    lo = mid
                else:
                    hi = mid
            if hi - lo < mp.mpf(10) ** (-mp.mp.dps + 5) * (abs(lo) + 1):
                break
        return (lo + hi) / 2

    def ppf(self, q):
        return self._invert(mp.log(mp.mpf(q)), False)

    def isf(self, q):
        return self._invert(mp.log(mp.mpf(q)), True)

    # moments in standard coordinates y = (x - lower)/(upper - lower)
    def raw_moment_std(self, n):
        u = self.upper - self.lower
        f = lambda x: ((x - self.lower) / u) ** n * self.g(x)
        return (mp.quad(f, self.left_pts) + mp.quad(f, self.right_pts)) / self.Z

    def central(self, n, mean):
        f = lambda x: (x - mean) ** n * self.g(x)
        return (mp.quad(f, self.left_pts) + mp.quad(f, self.right_pts)) / self.Z

    def stats(self):
        mean = (mp.quad(lambda x: x * self.g(x), self.left_pts)
                + mp.quad(lambda x: x * self.g(x), self.right_pts)) / self.Z
        v = self.central(2, mean)
        sk = self.central(3, mean) / v ** mp.mpf(1.5)
        ku = self.central(4, mean) / v ** 2 - 3
        return mean, v, sk, ku

    def entropy(self):
        def h(x):
            gx = self.g(x)
            return 0 if gx == 0 else -(gx / self.Z) * mp.log(gx / self.Z)
        return mp.quad(h, self.left_pts) + mp.quad(h, self.right_pts)


def s(v, digits=20):
    return mp.nstr(v, digits)


# (mode, sigma, lower, upper); float inputs are converted exactly (mpf of
# the binary value), so the model sees what the library sees.
CASES = [
    (5, 10, -5, 20),        # the notebook's running example
    (3, 1, 0, 10),
    (1, 3, 0, 10),          # mode near the lower bound
    (9, 1, 0, 10),          # mode near the upper bound
    (75, 5, 70, 100),       # the fit example
    (0, 1, -4, 4),          # symmetric
    (-3, 2, -10, 0),        # negative support
    (1, 0.062, 0, 100),     # STRESS: needle-thin peak, wide domain
    (950, 1, -1000, 1000),  # STRESS: peak far from centre
    (0.01, 1, 0, 10),       # mode very near the lower bound
    (1e-6, 1, 0, 10),       # left piece carries ~1e-7 of the mass
    (9.99, 1, 0, 10),       # mode very near the upper bound
    (5, 1, None, None),     # unbounded both sides
    (5, 1, 0, None),        # unbounded above
    (5, 100, 0, 10),        # sigma far wider than the support
]
POINTS_FRAC = ['0.001', '0.01', '0.1', '0.25', '0.5', '0.75', '0.9',
               '0.99', '0.999']
BODY_Q = ['0.01', '0.1', '0.25', '0.5', '0.75', '0.9', '0.99']
TAIL_Q = ['1e-3', '1e-6', '1e-9', '1e-12', '1e-16', '1e-20', '1e-25',
          '1e-30']
# Deep lower tail in the standard form (loc 0, scale 1): (c, s) pairs.
DEEP_STD = [(0.4, 0.4), (0.04, 0.0248), (0.5, 0.0005)]
DEEP_Q = ['1e-50', '1e-100', '1e-200', '1e-300', '1e-310', '1e-320']


if __name__ == "__main__":
    import json
    conv = lambda v: None if v is None else mp.mpf(float(v))
    out = {"cases": [], "deep_std": []}
    for case in CASES:
        ref = BLRef(*(conv(v) for v in case))
        lo, up, mode = ref.lower, ref.upper, ref.mode
        rec = {"case": [None if v is None else str(v) for v in case]}
        xs = [lo + (up - lo) * mp.mpf(fr) for fr in POINTS_FRAC]
        xs += [mode - ref.sigma * mp.mpf(d) for d in ('3', '1', '0.1', '1e-6')]
        xs += [mode + ref.sigma * mp.mpf(d) for d in ('3', '1', '0.1', '1e-6')]
        xs = sorted(x for x in set(xs) if lo < x < up)
        rec["x"] = [s(x, 25) for x in xs]
        rec["pdf"] = [s(ref.pdf(x)) for x in xs]
        rec["cdf"] = [s(ref.checked_cdf(x)) for x in xs]
        rec["sf"] = [s(ref.checked_sf(x)) for x in xs]
        rec["ppf_body"] = [[str(q), s(ref.ppf(q), 25)] for q in BODY_Q]
        rec["ppf_tail"] = [[str(q), s(ref.ppf(q), 25)] for q in TAIL_Q]
        rec["isf_tail"] = [[str(q), s(ref.isf(q), 25)] for q in TAIL_Q]
        rec["raw_std"] = [s(ref.raw_moment_std(n)) for n in range(1, 9)]
        rec["stats"] = [s(v) for v in ref.stats()]
        rec["entropy"] = s(ref.entropy())
        rec["p_left"] = s(ref.ZL / ref.Z)
        out["cases"].append(rec)
        print(f"done {case}", file=sys.stderr)
    for c, sg in DEEP_STD:
        ref = BLRef(mp.mpf(c), mp.mpf(sg), 0, 1)
        # q as the float the library receives (subnormal q is inexact)
        out["deep_std"].append({"c": c, "s": sg, "ppf": [
            [q, s(ref.ppf(mp.mpf(float(q))), 25)] for q in DEEP_Q]})
    print("worst quadrature vs closed-form disagreement:",
          mp.nstr(BLRef.worst, 3), file=sys.stderr)
    json.dump(out, sys.stdout, indent=1)
