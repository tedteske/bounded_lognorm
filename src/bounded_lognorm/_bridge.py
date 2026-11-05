"""
The bounds bridge shared by every family in this package (BL2 and BL).

Both distributions are scipy location-scale families on [0, 1] with shapes
c (relative mode) and s (relative sigma), loc = lower and scale = upper -
lower. This mixin supplies, identically for both:

  - from_bounds / bounds_to_shapes / to_bounds, the (mode, sigma, lower,
    upper) <-> (c, s, loc, scale) bridge, with None meaning an unbounded
    side (mode -/+ 1000 sigma);
  - from_std / sigma_from_std, the same with a standard deviation in
    place of sigma (0.4.0; see _from_std.py);
  - fit_bounds, fitting in the bounds form (see _fit.py);
  - the scipy hooks that do not depend on the density: _shape_info,
    _fitstart, and _updated_ctor_param (which carries `precision` through
    freezing).

Moved here from _bl2.py unchanged in Phase 5 (0.2.0), when the piecewise BL
joined the package. Ted Teske with Claude (Anthropic), 2026.
"""
import warnings

import numpy as np

from ._fit import fit_bounds as _fit_bounds
from ._from_std import sigma_from_std as _sigma_from_std
from ._profile import profile_bound as _profile_bound

try:                                    # scipy >= 1.9; used by scipy.stats.fit
    from scipy.stats._distn_infrastructure import _ShapeInfo
except ImportError:                     # pragma: no cover
    _ShapeInfo = None

# The speed <-> precision dial positions, left to right. What each position
# does is up to the family: BL2 maps them to quadrature and grid settings;
# BL accepts them for API parity only (its closed forms need no dial).
PRECISION_DIAL = ('fast', 'balanced', 'precise', 'exact')

# Below this ratio of sigma to the coordinate scale, float64 resolution of
# (x - lower) alone limits cdf accuracy to worse than ~1e-8.
_RESOLUTION_WARN = 1e-8


class _BoundsBridge:
    """Mixin for an rv_continuous with shapes (c, s) on [0, 1] and a
    `precision` attribute. Put it before rv_continuous in the bases."""

    def _updated_ctor_param(self):
        # scipy freezes by re-instantiating from these; without `precision`
        # every frozen distribution would silently fall back to 'balanced'.
        d = super()._updated_ctor_param()
        d['precision'] = self.precision
        return d

    def _shape_info(self):
        return [_ShapeInfo("c", False, (0, 1), (False, False)),
                _ShapeInfo("s", False, (0, np.inf), (False, False))]

    def _argcheck(self, c, s):
        """
        0 < c < 1 and s > 0, both finite. Elementwise.

        Also warns when s < 1e-8: in standard coordinates float64 resolves
        1 - x only to ~1e-16, so accuracy is limited to roughly 1e-16 / s.
        (A large |loc| relative to sigma has the same effect in x - loc;
        only from_bounds can see that, and it warns for it.)
        """
        c, s = np.broadcast_arrays(np.asarray(c, dtype=float),
                                   np.asarray(s, dtype=float))
        cond = (np.isfinite(c) & np.isfinite(s)
                & (c > 0) & (c < 1) & (s > 0))
        if np.any(cond & (s < _RESOLUTION_WARN)):
            smin = float(np.min(s[cond]))
            warnings.warn(
                "s = %.1e: sigma is below 1e-8 of (upper - lower), so float64 "
                "resolution limits cdf/ppf accuracy to roughly %.0e."
                % (smin, 2.2e-16 / smin), RuntimeWarning, stacklevel=3)
        return cond

    def _fitstart(self, data, args=None):
        """
        Starting point for fit(): bounds just outside the data, mode at the
        median, sigma at the sample standard deviation. (scipy's generic
        start sets every shape to 1, which is an invalid c.)
        """
        data = np.asarray(data, dtype=float)
        lo, hi = float(np.min(data)), float(np.max(data))
        span = hi - lo if hi > lo else max(abs(hi), 1.0)
        loc = lo - 0.05 * span
        scale = 1.1 * span
        c = float(np.clip((np.median(data) - loc) / scale, 0.05, 0.95))
        s = float(max(np.std(data) / scale, 1e-3))
        return c, s, loc, scale

    def fit_bounds(self, data, method='MLE', *, fmode=None, fsigma=None,
                   flower=None, fupper=None, gmode=None, gsigma=None,
                   glower=None, gupper=None, ecdf='midpoint', truncate=None):
        """
        Fit in the bounds parametrization, holding any parameters fixed.

        Parameters
        ----------
        data : array_like
            The sample.
        method : 'MLE' or 'LSQ'
            'MLE' (default) maximizes the likelihood. 'LSQ' minimizes the
            squared distance between the cdf and the empirical cdf at the
            distinct data values -- the original notebook's fit(MLE=False).
        fmode, fsigma, flower, fupper : float, optional
            Hold that parameter fixed at this value. Any combination works,
            e.g. flower=0 alone, or fmode alone (which the native fit cannot
            express, since c depends on loc and scale).
        gmode, gsigma, glower, gupper : float, optional
            Starting guesses for free parameters. The defaults are the
            median, the standard deviation, and bounds 5% of the data range
            outside the data.
        ecdf : 'midpoint' or 'step'
            LSQ only. 'midpoint' (default) places the empirical cdf at the
            middle of each jump, the Hazen position (i - 1/2)/n. 'step' uses
            the top of each jump, i/n, as the original notebook did; it puts
            the largest value at exactly 1, which pulls a fitted upper bound
            toward max(data).
        truncate : (lo, hi), optional
            MLE only (0.3.0). Fit by the likelihood conditional on the
            window lo < x < hi, either end None for open: each density is
            divided by the window's probability, and data outside it are
            set aside (counted in .n_excluded). With (t, None) only the data
            above t inform the fit, and they are not distorted by data
            below it, e.g. a few very slow runs when the question is about
            the fastest possible one. The threshold should be chosen
            without looking at the fit (a sample quantile, say); profile
            a bound with the same window to see what the window's data
            can say about it.

        Returns
        -------
        BoundsFit
            Unpacks as (mode, sigma, lower, upper) and carries .method,
            .free, .nnlf, .success, .message, .truncate, .n_used,
            .n_excluded, and for LSQ .cov and .stderr.

        Notes
        -----
        Free bounds are kept outside the data (and outside a fixed mode), so
        every trial parameter set is valid; a fixed bound must already be.
        The optimizer runs on unconstrained transforms of the free
        parameters: Nelder-Mead for MLE, scipy's least_squares for LSQ.

        The LSQ covariance .cov is curve_fit's pcov in the bounds
        parametrization, s^2 (J^T J)^-1 with s^2 = SSR / (m - p), which is
        what the original fit returned. It treats the empirical-cdf
        residuals as independent, which they are not, so its standard
        errors understate the real uncertainty. For a confidence interval
        on, say, the upper bound, refit bootstrap resamples of the data.
        No covariance is offered for MLE: with a free bound the usual
        regularity conditions for the inverse Fisher information do not
        clearly hold.

        A fit is reported as failed (success False, a RuntimeWarning, and
        the reason in .message) when the data do not identify a parameter:
        sigma or a free bound running away from the data, or (since 0.2.1)
        a free bound collapsing onto the sample extreme. The second happens
        with LSQ when the fit keeps improving as the bound approaches the
        smallest or largest value, e.g. data that fall off steeply near
        their maximum with the lower bound fixed far below. The returned
        bound is then the sample extreme, not an estimate beyond it, and
        its .cov rows and columns and .stderr are NaN.

        Added in 0.2.0. For the native (c, s, loc, scale) parametrization
        and scipy's own options, use fit().
        """
        return _fit_bounds(self, data, method, fmode=fmode, fsigma=fsigma,
                           flower=flower, fupper=fupper, gmode=gmode,
                           gsigma=gsigma, glower=glower, gupper=gupper,
                           ecdf=ecdf, truncate=truncate)

    def profile_bound(self, data, bound='upper', level=0.95, *,
                      truncate=None, fmode=None, fsigma=None, flower=None,
                      fupper=None, gmode=None, gsigma=None, glower=None,
                      gupper=None, grid=None):
        """
        Profile-likelihood confidence interval for one bound (0.3.0).

        The bound is held at each trial value and the likelihood maximized
        over the other free parameters (fit_bounds with that bound fixed).
        The interval is the set of values whose likelihood ratio statistic
        LR(b) = 2 [nnlf_profile(b) - nnlf_max] is at most the chi-squared(1)
        quantile at `level` (3.84 for 95%).

        Parameters
        ----------
        data : array_like
        bound : 'upper' or 'lower'
            The bound to profile. It must be free (do not pass fupper for
            'upper').
        level : float
            Confidence level, default 0.95.
        truncate, fmode, fsigma, flower, fupper, gmode, gsigma, glower,
        gupper :
            As in fit_bounds (MLE). The same window and fixed values apply
            to every fit of the profile.
        grid : array_like, optional
            Further values of the bound at which to evaluate the profile,
            for plotting; values on the wrong side of the data are dropped.

        Returns
        -------
        BoundProfile
            .interval (lo, hi) in data units; an end the data cannot close
            is +-inf (the likelihood never falls far enough). .estimate is
            the maximum-likelihood bound (+-inf when it ran away). Also
            .fit (the full BoundsFit), .critical, .level, .profile (bound
            values and LR, sorted, every point evaluated), .success,
            .message.

        Notes
        -----
        MLE only: least squares on the empirical cdf has no likelihood,
        and its residuals are not independent, so the curvature of its
        objective is not a measure of uncertainty.

        A bound is a support parameter, so Wilks' theorem is not
        automatic. For BL and BL2 the density vanishes faster than any
        power at each bound, which argues for the regular case (Smith
        1985). In simulation from both families (docs/
        phase7_bound_intervals.md), two-sided 95% intervals covered the
        truth 92-96% of the time at the sprints' sample sizes (n ~ 700),
        full data or truncated, and 95% at n = 50 and 100 for one shape,
        but only 83-88% at n = 100 for another, with the bound close to
        the data and sigma wide. Misses are lopsided: the interval stops short on
        the data's side more often than it overshoots, so do not halve the
        level for a one-sided limit. A parametric-bootstrap critical value
        did no better in any case checked, so none is offered.

        The profile is assumed to fall away on each side of the maximum
        without a second dip; the search steps outward in the log of the
        distance from the data and stops at the first crossing. An end is
        reported open (inf) when LR stays below the critical value out to
        fit_bounds' cap on a bound's distance from the data (10^4 times
        the data range).
        """
        return _profile_bound(self, data, bound, level, truncate=truncate,
                              fmode=fmode, fsigma=fsigma, flower=flower,
                              fupper=fupper, gmode=gmode, gsigma=gsigma,
                              glower=glower, gupper=gupper, grid=grid)

    def from_bounds(self, mode, sigma, lower=None, upper=None):
        """
        Frozen distribution from the bounds parametrization.

        Parameters
        ----------
        mode : float
            x at the point of maximum density
        sigma : float
            spread around the mode (SD of each untruncated lognormal
            component), > 0
        lower, upper : float or None
            bounds of the support. None means unbounded on that side and is
            replaced by mode -/+ 1000 sigma, where the distribution is
            indistinguishable from the normal side. Infinite bounds are
            rejected; pass None.

        Returns
        -------
        A frozen rv with c = (mode - lower)/(upper - lower),
        s = sigma/(upper - lower), loc = lower, scale = upper - lower, at this
        instance's precision.

        Unlike the native API (which returns NaN for invalid parameters, as
        scipy does), this raises ValueError, so a bad parameter set fails
        loudly. Accepts arrays (broadcast elementwise).
        """
        c, s, loc, scale = self.bounds_to_shapes(mode, sigma, lower, upper)
        return self(c, s, loc=loc, scale=scale)

    def from_std(self, mode, std, lower=None, upper=None):
        """
        Frozen distribution with a given standard deviation (0.4.0).

        Like from_bounds, but with the standard deviation in place of
        sigma: the sigma is solved for so that the result's std() equals
        `std`, for this mode and these bounds. Use it when what you know
        is a spread -- a sample standard deviation, say -- rather than
        sigma, which is the spread of each lognormal piece before the
        bounds cut it off and is always larger than the std.

        Parameters
        ----------
        mode : float
            x at the point of maximum density
        std : float
            the standard deviation the distribution should have, > 0
        lower, upper : float or None
            bounds of the support, as in from_bounds (None: mode -/+ 1000
            sigma, for the sigma found)

        Returns
        -------
        from_bounds(mode, sigma_from_std(mode, std, lower, upper), lower,
        upper): a frozen rv at this instance's precision.

        Raises ValueError for invalid parameters, as from_bounds does, and
        when no sigma gives `std` (see sigma_from_std). Accepts arrays
        (broadcast elementwise). For the sigma itself, use sigma_from_std.

        Example
        -------
        Occupancy between 0 and 100%, most likely 90%, with an observed
        standard deviation of 3.7 points:

        >>> rv = bl.from_std(90, 3.7, 0, 100)
        >>> round(float(rv.std()), 10)
        3.7
        """
        sigma = self.sigma_from_std(mode, std, lower, upper)
        return self.from_bounds(mode, sigma, lower, upper)

    def sigma_from_std(self, mode, std, lower=None, upper=None):
        """
        The sigma whose distribution has standard deviation `std` (0.4.0).

        Solves from_bounds(mode, sigma, lower, upper).std() == std for
        sigma, with mode and bounds held fixed. There is no closed form,
        for either family (docs/v0.4_from_std.md), so this is a root find
        in log(sigma): brentq, to a relative tolerance of 1e-13 in sigma,
        on this instance's own std (so at its precision). Typically
        ~10-15 evaluations of std: about 5 ms for BL, 70 ms for BL2.

        Parameters
        ----------
        mode, lower, upper : float or None
            as in from_bounds
        std : float
            the target standard deviation, > 0

        Returns
        -------
        sigma : float (ndarray for array input, broadcast elementwise)

        Raises
        ------
        ValueError
            For invalid parameters (non-finite, std <= 0, not lower < mode
            < upper), and when `std` is not attainable: with both bounds
            finite, std rises with sigma toward (upper - lower)/sqrt(12),
            the uniform distribution's, which it approaches only
            logarithmically slowly; a std at or near that value has no
            usable sigma, and the error says how large std can get.

        Notes
        -----
        The root is unique because std increases with sigma. That is
        checked numerically, not proved: for modes from 1e-15 to 1 - 1e-15
        of the way across the support, at every dial position, std rises
        strictly up to sigma = 1e12 * (upper - lower), the end of the
        search. Beyond it BL2's std can turn down (mode within ~1e-3 of a
        bound, sigma above ~1e20 * (upper - lower)) after rising slightly
        past the uniform value, so a std is not always reached by a unique
        sigma there; BL's keeps rising. The search does not go that far.

        With an unbounded side (None) std grows without limit, so every
        std > 0 is attainable.
        """
        return _sigma_from_std(self, mode, std, lower, upper)

    @staticmethod
    def bounds_to_shapes(mode, sigma, lower=None, upper=None):
        """
        (mode, sigma, lower, upper) -> (c, s, loc, scale), validated.
        None bounds as in from_bounds. Useful with fit, e.g. to fix known
        bounds: dist.fit(data, floc=loc, fscale=scale).
        """
        normal_approx_factor = 1e+3
        mode = np.asarray(mode, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        lower = (mode - normal_approx_factor * sigma if lower is None
                 else np.asarray(lower, dtype=float))
        upper = (mode + normal_approx_factor * sigma if upper is None
                 else np.asarray(upper, dtype=float))
        if not np.all(np.isfinite(mode) & np.isfinite(sigma)
                      & np.isfinite(lower) & np.isfinite(upper)):
            raise ValueError(
                "mode, sigma, lower and upper must be finite; use None for "
                "an unbounded side")
        if not np.all((lower < mode) & (mode < upper) & (sigma > 0)):
            raise ValueError("Requires (lower < mode < upper) and (sigma > 0)")
        scale = upper - lower
        c = (mode - lower) / scale
        s = sigma / scale
        # Large |bounds| relative to sigma limit the resolution of x - loc.
        # (A small s on its own is reported by _argcheck instead.)
        bound = np.maximum(np.abs(lower), np.abs(upper))
        hit = (sigma < _RESOLUTION_WARN * bound) & ~(s < _RESOLUTION_WARN)
        if np.any(hit):
            r = float(np.min((sigma / bound)[hit]))
            warnings.warn(
                "sigma is %.1e of the coordinate scale (max |bound|); "
                "float64 resolution of x - lower limits cdf/ppf accuracy to "
                "roughly %.0e. Shift the problem nearer the origin if you "
                "need more." % (r, 2.2e-16 / r), RuntimeWarning, stacklevel=3)
        return c[()], s[()], lower[()], scale[()]

    @staticmethod
    def to_bounds(c, s, loc=0.0, scale=1.0):
        """(c, s, loc, scale) -> (mode, sigma, lower, upper). Inverse of
        bounds_to_shapes; e.g. dist.to_bounds(*dist.fit(data))."""
        c, s, loc, scale = (np.asarray(v, dtype=float)
                            for v in (c, s, loc, scale))
        return ((loc + c * scale)[()], (s * scale)[()], loc[()],
                (loc + scale)[()])

