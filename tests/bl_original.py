# The ORIGINAL piecewise BL, for Suite I of tests/test_bl.py.
#
# Verbatim from the first code cell of bounded_lognorm.ipynb (Ted Teske,
# 2020-2021), with only the optional plotting/example imports removed (they
# need sklearn, seaborn and pandas, and the class does not use them). Do not
# edit: Suite I compares the package against exactly this code, including
# its flaws.
#
#
# Author: Ted Teske 2020-2021
#

from scipy.stats import rv_continuous
from scipy.special import erf, erfinv
from scipy.optimize import curve_fit, minimize
import numpy as np




def lognorm_pdf(x, mu_N=0, sigma_N=1):
    """
    Log-normal (aka lognormal) probability density function (pdf).

    Parameters
    ----------
    x : {float, list or ndarray of floats}
        x-value(s)
    mu_N : float
        mean of underlying normal distribution
    sigma_N : float
        standard deviation of underlying normal distribution

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    # Creates an index of positive elements in x.
    x = np.atleast_1d(x)
    i = np.nonzero(x > 0)

    # Applies lognormal pdf to positive elements. All others evaluate to 0.
    y = np.zeros(len(x))
    y[i] = 1 / (x[i] * sigma_N * np.sqrt(2 * np.pi)) \
        * np.exp(-(np.log(x[i]) - mu_N)**2 / (2 * sigma_N**2))
    return y


def lognorm_cdf(x, mu_N=0, sigma_N=1):
    """
    Log-normal cumulative density function (cdf).

    Parameters
    ----------
    x : {float, list or ndarray of floats}
        x-value(s)
    mu_N : float
        mean of underlying normal distribution
    sigma_N : float
        standard deviation of underlying normal distribution

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    # Creates an index of positive elements in x.
    x = np.atleast_1d(x)
    i = np.nonzero(x > 0)

    # Applies lognormal cdf to positive elements. All others evaluate to 0.
    y = np.zeros(len(x))
    y[i] = 0.5 + 0.5 * erf((np.log(x[i]) - mu_N) / (np.sqrt(2) * sigma_N))
    return y


def lognorm_ppf(q, mu_N=0, sigma_N=1):
    """
    Log-normal percent-point function (ppf), also known as the quantile
    function, is the inverse of the cdf.

    Parameters
    ----------
    q : {float, list or ndarray of floats}
        quantile(s)
    mu_N : float
        mean of underlying normal distribution
    sigma_N : float
        standard deviation of underlying normal distribution

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    y = np.exp(mu_N + np.sqrt(2 * sigma_N**2) * erfinv(2 * q - 1))
    return y


def norm_pdf(x, mu_N=0, sigma_N=1):
    """
    Normal probability density function (pdf).

    Parameters
    ----------
    x : {float, list or ndarray of floats}
        x-value(s)
    mu_N : float
        mean
    sigma_N : float
        standard deviation

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Normal_distribution
    """
    y = 1 / (sigma_N * np.sqrt(2 * np.pi))
    y *= np.exp(-(x - mu_N)**2 / (2 * sigma_N**2))
    return y


def norm_cdf(x, mu_N=0, sigma_N=1):
    """
    Normal cumulative density function (cdf).

    Parameters
    ----------
    x : {float, list or ndarray of floats}
        x-value(s)
    mu_N : float
        mean
    sigma_N : float
        standard deviation

    Returns
    -------
    y : ndarray of float(s)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Normal_distribution
    """
    y = 0.5 * (1 + erf((x - mu_N) / (sigma_N * np.sqrt(2))))
    return y


def lognorm_params(mode, sigma):
    """
    Given user-defined mode and sigma of a lognormal distribution,
    calculates the mu and sigma of its underlying normal distribution.

    Parameters
    ----------
    mode : float
        mode of lognormal distribution
    sigma : float
        standard deviation of lognormal distribution

    Returns
    -------
    mu_N : float
    sigma_N : float

    References
    ----------
    [1] Exact form adapted from the result of the query:
        solve[g^4 - g^3 - sigma^2/m^2] for g
        at https://www.wolframalpha.com. Accessed 21 Oct. 2020.
    [2] Numerical approximation from Warren Weckesser's answer to question at
        https://stackoverflow.com/questions/41464753. Accessed 21 Oct. 2020.

    """
    # Converts parameters to floats in case exponents below too big for ints.
    # https://stackoverflow.com/questions/67097840
    def GetItem(inputobject):
        while type(inputobject) is np.ndarray:
            inputobject = inputobject[0]
        return inputobject

    mode, sigma = float(GetItem(mode)), float(GetItem(sigma))

    # Finds the roots of the polynomial g^4 - g^3 - sigma^2/mode^2,
    # where g = e^(variance of the associated Normal distribution),
    # and uses just the real, positive root.
    try:
        # This is the exact form [1].
        a = np.cbrt(np.sqrt(81 * mode**8 * sigma**4 +
                            768 * mode**6 * sigma**6) -
                    9 * mode**4 * sigma**2)
        b = np.cbrt(2 / 3) * sigma**2
        c = np.cbrt(2 / 3)**2 * 2
        d = np.sqrt((-16 * b / a) + (c * a / mode**2) + 1)

        g1 = d / 4
        g2 = (4 * b / a) - a / (np.cbrt(18) * mode**2)
        g3 = 1 / (2 * d)

        g = g1 + np.sqrt(g2 + g3 + 0.5) / 2 + 0.25  # the real, positive root

        # Whereas the above is the real, positive root (the one we'll use),
        # negative or complex roots are commented below:
        # g =  g1 - np.sqrt(g2 + g3 + 0.5) / 2 + 0.25
        # g = -g1 + np.sqrt(g2 - g3 + 0.5) / 2 + 0.25
        # g = -g1 - np.sqrt(g2 - g3 + 0.5) / 2 + 0.25

    except:
        # And this is a slightly slower, numerical approximation of g [2].
        # It is however much more concise.
        p = np.poly1d([1, -1, 0, 0, -(sigma / mode)**2])
        r = p.roots
        g = r[(r.imag == 0) & (r.real > 0)].real

    # Plugs g into the expressions for sigma_N and mu_N.
    sigma_N = np.sqrt(np.log(g))
    mu_N = np.log(mode * g)
    return mu_N, sigma_N


class bounded_lognorm_gen(rv_continuous):
    """
    Subclass of scipy.stats.rv_continuous [1]. This is used to generate a
    domain-bounded lognormal continuous probability distribution, which is
    fully described by the four parameters: mode, sigma, lower and upper
    bounds.

    The probability density function (pdf) concatenates two lognormal pdfs
        1. left-of-mode: lognormal pdf with loc at lower bound, and
        2. right-of-mode: reversed lognormal pdf with loc at upper bound and
            scaled so that the concatenated pdf is continuous at the mode;
    and scales the concatenated curve so that its integral equals 1.

    Methods
    -------
    same as scipy.stats.rv_continuous [1]

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.stats.rv_continuous.html
    """

    def _argcheck(self, *args):
        """
        Overrides default _argcheck to allow negative parameters. Returns
        condition array of 1s where arguments are correct and 0s where they
        are not.
        """
        cond = 1
        for arg in args:
            if arg is None:
                arg = 0
            cond = np.logical_and(cond, np.isfinite(arg))
        return cond

    def _argcheck_final(self, x, mode, sigma, lower, upper,
                        ignore_checks=False):
        """
        Provides further compatibility checks and conversions for parameters.
        """
        # If lower or upper parameter is set to None, converts to a scalar so
        # that the curve below or above mode, respectively, approximates the
        # corresponding side of a normal(mode, sigma^2) distribution.
        normal_approx_factor=1e+3
        if lower is None:
            lower = mode - normal_approx_factor * sigma
        if upper is None:
            upper = mode + normal_approx_factor * sigma

        # Converts parameters to scalars, if necessary.
        try:
            mode, sigma, lower, upper = mode[0], sigma[0], lower[0], upper[0]
        except:
            pass

        # Checks parameters for compatibility with distribution.
        if not ignore_checks:
            param_conditions = (lower < mode < upper) & (sigma > 0)
            if not param_conditions:
                error_text = "Requires (lower < mode < upper) and (sigma > 0)"
                raise ValueError(error_text)

        # Converts scalar, list, or Series input to array.
        x = np.atleast_1d(x)
        mode = np.atleast_1d(mode)
        return x, mode, sigma, lower, upper

    def pdf(self, x, mode, sigma, lower=None, upper=None, ignore_checks=False):
        """
        Probability density function (pdf).

        Parameters
        ----------
        x : {float, list, ndarray of floats}
            x-value(s)
        mode : float
            x at the point of maximum frequency [1]
        sigma : float
            indicates spread of data around mode. This approaches the standard
            deviation as lower and upper are increasingly distant from x.
        lower : float or None
            lower bound of the domain (x-values)
        upper : float or None
            upper bound of the domain (x-values)

        Returns
        -------
        relative likelihood [2] of x : ndarray of float(s)

        Note
        ----
        This is a wrapper for _pdf().

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Mode_(statistics)
        [2] https://en.wikipedia.org/wiki/Probability_density_function
        """
        # Checks parameters for compatibility with distribution and converts.
        args = self._argcheck_final(x, mode, sigma, lower, upper,
                                    ignore_checks=ignore_checks)
        x, mode, sigma, lower, upper = args

        # Shifts x, lower, upper, and mode so that lower is at 0;
        # allows negative x-values.
        x, mode, upper = x - lower, mode - lower, upper - lower

        # Concatenates left- and right-of-mode pdfs.
        y = self.concat_pdf(x, mode, sigma, upper)
        return y

    def logpdf(self, x, mode, sigma, lower=None, upper=None):
        """
        Log of the probability density function (pdf). See bounded_lognorm.pdf
        for a description of parameters.
        """
        with np.errstate(divide='ignore'):
            return np.log(self.pdf(x, mode, sigma, lower, upper))

    def cdf(self, x, mode, sigma, lower=None, upper=None, ignore_checks=False):
        """
        Cumulative density function (cdf).

        Parameters
        ----------
        x : {float, list, ndarray of floats}
            x-value(s)
        mode : float
            x at the point of maximum frequency [1]
        sigma : float
            indicates spread of data around mode. This approaches the standard
            deviation as lower and upper are increasingly distant from x.
        lower : float or None
            lower bound of the domain (x-values)
        upper : float or None
            upper bound of the domain (x-values)

        Returns
        -------
        probability(s) that random variable <= x : ndarray of float(s)

        Note
        ----
        This is a wrapper for _cdf().

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Mode_(statistics)
        [2] https://en.wikipedia.org/wiki/Cumulative_distribution_function
        """
        # Checks parameters for compatibility with distribution and converts.
        args = self._argcheck_final(x, mode, sigma, lower, upper,
                                    ignore_checks=ignore_checks)
        x, mode, sigma, lower, upper = args

        # Shifts x, lower, upper, and mode so that lower is at 0;
        # allows negative x-values.
        x, mode, upper = x - lower, mode - lower, upper - lower

        # Concatenates left- and right-of-mode cdfs.
        y = self.concat_cdf(x, mode, sigma, upper)
        return y

    def logcdf(self, x, mode, sigma, lower=None, upper=None):
        """
        Log of the cumulative density function (cdf). See bounded_lognorm.cdf
        for a description of parameters.
        """
        with np.errstate(divide='ignore'):
            return np.log(self.cdf(x, mode, sigma, lower, upper))

    def sf(self, x, mode, sigma, lower=None, upper=None):
        """
        Survival function, 1 - cdf. See bounded_lognorm.cdf
        for a description of parameters.
        """
        return 1.0 - self.cdf(x, mode, sigma, lower, upper)

    def logsf(self, x, mode, sigma, lower=None, upper=None):
        """
        Log of the survival function. See bounded_lognorm.cdf
        for a description of parameters.
        """
        with np.errstate(divide='ignore'):
            return np.log(self.sf(x, mode, sigma, lower, upper))

    def ppf(self, q, mode, sigma, lower=None, upper=None, ignore_checks=False):
        """
        Percent-point function (ppf), also known as the quantile function, is
        the inverse of the cdf.

        Parameters
        ----------
        q : {float, list, ndarray of floats}
            quantile(s), lower tail probability
        mode : float
            x at the point of maximum frequency [1]
        sigma : float
            indicates spread of data around mode. This approaches the standard
            deviation as lower and upper are increasingly distant from x.
        lower : float or None
            lower bound of the domain (x-values)
        upper : float or None
            upper bound of the domain (x-values)

        Returns
        -------
        x-value(s), such that P(X <= x) is q : ndarray of float(s) [2]

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Mode_(statistics)
        [2] https://en.wikipedia.org/wiki/Quantile_function
        """
        # Checks parameters for compatibility with distribution and converts.
        args = self._argcheck_final(q, mode, sigma, lower, upper,
                                    ignore_checks=ignore_checks)
        q, mode, sigma, lower, upper = args

        # Shifts lower, upper, and mode so that lower is at 0;
        # allows negative x-values.
        mode, upper = mode - lower, upper - lower

        # Concatenates left- and right-of-mode cdfs and adds back lower.
        y = self.concat_ppf(q, mode, sigma, upper) + lower
        return y

    def isf(self, q, mode, sigma, lower=None, upper=None):
        """
        Inverse survival function. See bounded_lognorm.ppf
        for a description of parameters.
        """
        return self.ppf(1.0 - q, mode, sigma, lower, upper)

    def rvs(self, mode, sigma, lower=None, upper=None, size=1,
            random_state=None):
        """
        Generates random variate(s) using inverse transform sampling [1].

        Parameters
        ----------
        mode : float
            x at the point of maximum frequency [2]
        sigma : float
            indicates spread of data around mode. This approaches the standard
            deviation as lower and upper are increasingly distant from x.
        lower : float or None
            lower bound of the domain (x-values)
        upper : float or None
            upper bound of the domain (x-values)
        size : int
            number of random variates to generate
        random_state : None, int, or numpy.random.RandomState
            not used; for backward compatibility only

        Returns
        -------
        ndarray of float(s)

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Inverse_transform_sampling
        [2] https://en.wikipedia.org/wiki/Mode_(statistics)
        """
        uniform_rands = np.random.uniform(size=int(size))
        return self.ppf(uniform_rands, mode, sigma, lower, upper)

    def _stats(self, mode, sigma, lower, upper, moments='mv'):
        """
        Calculates population mean('m'), variance('v'), skew('s'),
        and/or kurtosis('k'). Do not call this directly;
        call stats(), a wrapper for this function, instead.

        References
        ----------
        [1] Raiford, Theodore E. “Skewness of Combined Distributions.”
            Journal of the American Statistical Association,
            vol. 37, no. 219, 1942, pp. 391–393.
            JSTOR, www.jstor.org/stable/2279009.
            Accessed 18 Sept. 2020.
        [2] Wang, Yibing, et. al. "Speed Modeling and Travel Time Estimation
            Based on Truncated Normal and Lognormal Distributions."
            Transportation Research Record, No. 2315, 2012, pp. 66-72.
            ResearchGate, www.researchgate.net/publication/272774439.
            Accessed 18 Sept. 2020.
        [3] Ford and McLaren. "METHODS FOR OBTAINING DISTRIBUTIONS OF
            URANIUM OCCURRENCE FROM ESTIMATES OF GEOLOCJ:C FEATURES." Union
            Carbide Corporation for the Department of Energy, 1980.
            U.S. Department of Energy Office of Scientific and Technical
            Information, https://www.osti.gov/servlets/purl/5121047.
            Accessed 18 Sept. 2020.
        """
        def partial_moment(mu_N, sigma_N, k, n=1):
            """
            Conditional probability formula used here adapted from [2] p. 71
            and [3] p. 76:

            part = (norm_cdf(zb - n * sigma_N) - norm_cdf(za - n * sigma_N))
            part /= (norm_cdf(zb) - norm_cdf(za))

            where za and zb are cutoff points expressed in standard deviations

            but if we are only concerned with the right cutoff:

                za  = (np.log(0) - mu_N) / sigma_N
                    = (-np.inf - mu_N) / sigma_N
                    = -np.inf

                and norm_cdf(-np.inf) == 0

            then

                part = norm_cdf(z - n * sigma_N) / norm_cdf(z)
            """
            # lognormal moment-generating function
            moment = np.exp((n * mu_N) + (0.5 * n**2 * sigma_N**2))

            # adjustement for portion below cutoff
            z = (np.log(k) - mu_N) / sigma_N  # cutoff in standard deviations
            part = norm_cdf(z - n * sigma_N) / norm_cdf(z)
            return moment * part

        def partial_mean(mu_N, sigma_N, k):
            return partial_moment(mu_N, sigma_N, k, 1)

        def partial_var(mu_N, sigma_N, k):
            mom1 = partial_moment(mu_N, sigma_N, k, 1)
            mom2 = partial_moment(mu_N, sigma_N, k, 2)
            return mom2 - mom1**2

        def partial_skew(mu_N, sigma_N, k):
            mom1 = partial_moment(mu_N, sigma_N, k, 1)
            mom2 = partial_moment(mu_N, sigma_N, k, 2)
            mom3 = partial_moment(mu_N, sigma_N, k, 3)
            mu2 = mom2 - mom1**2
            mu3 = mom3 - mom1 * (mom1**2 + 3 * mu2)
            return mu3 / mu2**1.5

        def partial_kurt(mu_N, sigma_N, k):
            mom1 = partial_moment(mu_N, sigma_N, k, 1)
            mom2 = partial_moment(mu_N, sigma_N, k, 2)
            mom3 = partial_moment(mu_N, sigma_N, k, 3)
            mom4 = partial_moment(mu_N, sigma_N, k, 4)
            mu2 = mom2 - mom1**2
            mu3 = mom3 - mom1 *  (mom1**2 + 3 * mu2)
            mu4 = mom4 - mom1 * ((mom1**2 + 6 * mu2) * mom1 + 4 * mu3)
            return mu4 / mu2**2

        # Checks parameters for compatibility with distribution and converts.
        args = self._argcheck_final(0, mode, sigma, lower, upper)
        _, mode, sigma, lower, upper = args

        params_a = lognorm_params(mode - lower, sigma)
        params_b = lognorm_params(upper - mode, sigma)
        w = self.cdf(mode, mode, sigma, lower, upper)

        # mean from weighted partial means
        mu_a = partial_mean(*params_a, mode - lower) + lower
        mu_b = upper - partial_mean(*params_b, upper - mode)
        mu = w * mu_a + (1 - w) * mu_b

        # variance from weighted partial variances
        mu2_a = partial_var(*params_a, mode - lower)
        mu2_b = partial_var(*params_b, upper - mode)
        d_a, d_b = mu_a - mu, mu_b - mu
        mu2 =        w * (mu2_a + d_a**2)
        mu2 += (1 - w) * (mu2_b + d_b**2)

        # skew from weighted partial skews, adapted from [1]
        g1_a = partial_skew(*params_a, mode - lower)
        g1_b = -partial_skew(*params_b, upper - mode)
        g1 =        w * (g1_a * mu2_a**1.5 + d_a * (3 * mu2_a + d_a**2))
        g1 += (1 - w) * (g1_b * mu2_b**1.5 + d_b * (3 * mu2_b + d_b**2))
        g1 /= mu2**1.5

        # kurtosis from weighted partial kurtoses
        g2_a = partial_kurt(*params_a, mode - lower)
        g2_b = partial_kurt(*params_b, upper - mode)
        g2 = w * (g2_a * mu2_a**2
                  + d_a * (d_a * (6 * mu2_a + d_a**2)
                  + 4 * g1_a * mu2_a**1.5))
        g2 += (1 - w) * (g2_b * mu2_b**2
                         + d_b * (d_b * (6 * mu2_b + d_b**2)
                         + 4 * g1_b * mu2_b**1.5))
        g2 /= mu2**2
        g2 -= 3

        return mu, mu2, g1, g2

    def stats(self, mode, sigma, lower=None, upper=None, moments='mv'):
        """
        Calculates population mean('m'), variance('v'), skew('s'),
        and/or kurtosis('k').

        Parameters
        ----------
        mode : float
            x at the point of maximum frequency
        sigma : float
            indicates spread of data around mode. This approaches the standard
            deviation as lower and upper are increasingly distant from x.
        lower : float or None
            lower bound of the domain (x-values)
        upper : float or None
            upper bound of the domain (x-values)
        moments : str, optional
            composed of letters ['mvsk'] defining which moments to compute:
            'm' = mean,
            'v' = variance,
            's' = (Fisher's) skew,
            'k' = (Fisher's) kurtosis.
            (default is 'mv')

        Returns
        -------
        ndarray
        """
        output = []
        mu, mu2, g1, g2 = self._stats(mode, sigma, lower, upper)
        if 'm' in moments:
            output.append(mu)
        if 'v' in moments:
            output.append(mu2)
        if 's' in moments:
            output.append(g1)
        if 'k' in moments:
            output.append(g2)
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)

    def moment(self, n, mode, sigma, lower=None, upper=None):
        """
        Calculates n-th order non-central moment of distribution.

        Parameters
        ----------
        n : int, 0 <= n <= 4
            order of moment
        mode : float
            x at the point of maximum frequency
        sigma : float
            indicates spread of data around mode. This approaches the standard
            deviation as lower and upper are increasingly distant from x.
        lower : float or None
            lower bound of the domain (x-values)
        upper : float or None
            upper bound of the domain (x-values)

        Returns
        -------
        float
        """
        mu, mu2, g1, g2 = self._stats(mode, sigma, lower, upper)
        mu3 = g1 * mu2**1.5
        mu4 = (g2 + 3.0) * mu2**2

        if n == 0:
            return 1.0
        elif n == 1:
            return mu
        elif n == 2:
            return mu2 + mu**2
        elif n == 3:
            return mu3 + mu * (mu**2 + 3 * mu2)
        elif n == 4:
            return mu4 + mu * ((mu**2 + 6 * mu2) * mu + 4 * mu3)
        else:
            pass

    def support(self, mode, sigma, lower=None, upper=None):
        """
        End-points of the distribution's support. See bounded_lognorm.pdf for
        a description of the parameters mode, sigma, lower, upper.
        """
        # Checks parameters for compatibility with distribution and converts.
        args = self._argcheck_final(0, mode, sigma, lower, upper)
        _, mode, sigma, lower, upper = args

        return lower, upper

    def median(self, mode, sigma, lower=None, upper=None):
        """
        Median. See bounded_lognorm.pdf for a description of the parameters
        mode, sigma, lower, upper.
        """
        return self.ppf(0.5, mode, sigma, lower, upper)

    def mean(self, mode, sigma, lower=None, upper=None):
        """
        Mean. See bounded_lognorm.pdf for a description of the parameters
        mode, sigma, lower, upper.
        """
        return self.stats(mode, sigma, lower, upper, moments='m')

    def var(self, mode, sigma, lower=None, upper=None):
        """
        Variance. See bounded_lognorm.pdf for a description of the parameters
        mode, sigma, lower, upper.
        """
        return self.stats(mode, sigma, lower, upper, moments='v')

    def std(self, mode, sigma, lower=None, upper=None):
        """
        Standard deviation. See bounded_lognorm.pdf for a description of the
        parameters mode, sigma, lower, upper.
        """
        return np.sqrt(self.var(mode, sigma, lower, upper))

    def entropy(self, mode, sigma, lower=None, upper=None):
        """
        Differential entropy of the random variable.

        Parameters
        ----------
        mode : float
            x at the point of maximum frequency [1]
        sigma : float
            indicates spread of data around mode. This approaches the standard
            deviation as lower and upper are increasingly distant from x.
        lower : float or None
            lower bound of the domain (x-values)
        upper : float or None
            upper bound of the domain (x-values)

        Returns
        -------
        entropy in natural units of information (nats) [2]

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Mode_(statistics)
        [2] https://en.wikipedia.org/wiki/Differential_entropy
        [3] https://en.wikipedia.org/wiki/Truncated_normal_distribution
        """
        # Checks parameters for compatibility with distribution and converts.
        args = self._argcheck_final(0, mode, sigma, lower, upper)
        _, mode, sigma, lower, upper = args
        """
        The first draft uses the generic formula for differential
        entropy [2] and numeric integration:

            from scipy.integrate import quad
            from scipy.special import entr
            def pdf_times_logpdf(x):
                # Function to be integrated below.
                return entr(self.pdf(x, mode, sigma, lower, upper))
            return quad(pdf_times_logpdf, lower, upper)[0]

        but is found not to be robust to the following cases:
            mode, sigma, lower, upper = 1, .062, 0, 100
            mode, sigma, lower, upper = 950, 1, -1000, 1000

        The implemation below adapts the formula for entropy of the
        truncated normal distribution [3].
        """
        mu, var = self.stats(mode, sigma, lower, upper)
        std = np.sqrt(var)
        alpha = (lower - mu)/std
        beta = (upper - mu)/std
        Z = norm_cdf(beta) - norm_cdf(alpha)
        entropy = 0.5 * np.log(2 * np.pi * np.e * var * Z**2)
        entropy += (alpha * norm_pdf(alpha) - beta * norm_pdf(beta)) / (2 * Z)
        return entropy

    def concat_pdf(self, x, mode, sigma, upper):
        """
        Concatenates left- and right-of-mode pdfs.
        """
        # Left of mode.
        mu_N_left, sigma_N_left = lognorm_params(mode, sigma)
        x_left = np.append(x[x <= mode], mode, axis=0)
        y_left = lognorm_pdf(x_left, mu_N_left, sigma_N_left)

        # Right of mode.
        mu_N_right, sigma_N_right = lognorm_params(upper - mode, sigma)
        x_right = np.append(x[x > mode], mode, axis=0)
        y_right = lognorm_pdf(upper - x_right, mu_N_right, sigma_N_right)

        # Scales right side so that left and right sides continuous at mode.
        y_right = y_right * y_left[-1] / y_right[-1]

        # Concatenates left and right sides, dropping the extra modes
        # appended above.
        y = np.append(y_left[:-1], y_right[:-1], axis=0)

        # Normalizes concatenated pdf to area == 1.
        scale = self.cdf(mode, mode, sigma, 0, upper, ignore_checks=True)
        scale /= lognorm_cdf(mode, mu_N_left, sigma_N_left)
        y = y * scale
        return y

    def concat_cdf(self, x, mode, sigma, upper):
        """
        Concatenates left- and right-of-mode cdfs
        """
        # Appends upper for use in scaling below.
        x = np.append(x, np.atleast_1d(upper), axis=0)

        # Left of mode.
        mu_N_left, sigma_N_left = lognorm_params(mode, sigma)
        x_left = np.append(x[x <= mode], mode, axis=0)
        y_left = lognorm_cdf(x_left, mu_N_left, sigma_N_left)
        slope_left = lognorm_pdf(mode, mu_N_left, sigma_N_left)

        # Right of mode.
        mu_N_right, sigma_N_right = lognorm_params(upper - mode, sigma)
        x_right = np.append(x[x > mode], mode, axis=0)
        y_right = lognorm_cdf(upper - x_right, mu_N_right, sigma_N_right)
        slope_right = lognorm_pdf(upper - mode, mu_N_right, sigma_N_right)

        # Transforms right side so that curve is continuous at mode and
        # monotonic increasing.
        y_right = -y_right * slope_left / slope_right  # flips, matches slope
        y_right = y_right + y_left[-1] - y_right[-1]  # matches levels

        # Concatenates left and right sides, dropping the extra modes
        # appended above.
        y = np.append(y_left[:-1], y_right[:-1], axis=0)

        # Normalizes concatenated cdf to max == 1, dropping the extra upper
        # appended above.
        scale = 1 / y[-1]

        return y[:-1] * scale

    def concat_ppf(self, q, mode, sigma, upper):
        """
        Concatenates left- and right-of-mode ppfs
        """
        # Left of mode.
        mu_N_left, sigma_N_left = lognorm_params(mode, sigma)
        qmode_left = self.cdf(mode, mode, sigma, 0, upper)
        qmode_left2 = lognorm_cdf(mode, mu_N_left, sigma_N_left)
        q_left = q[q <= qmode_left] * qmode_left2/qmode_left
        y_left = lognorm_ppf(q_left, mu_N_left, sigma_N_left)

        # Right of mode.
        mu_N_right, sigma_N_right = lognorm_params(upper - mode, sigma)
        qmode_right = self.cdf(upper - mode, upper - mode, sigma, 0, upper)
        qmode_right2 = lognorm_cdf(upper - mode, mu_N_right, sigma_N_right)
        q_right = (1 - q[q > qmode_left]) * qmode_right2/qmode_right
        y_right = lognorm_ppf(q_right, mu_N_right, sigma_N_right)

        # Transforms right side so that curve is continuous at mode and
        # monotonic increasing.
        y_right = -y_right + upper

        # Concatenates left and right sides.
        y = np.append(y_left, y_right, axis=0)
        return y

    def nnlf(self, x, mode, sigma, lower=None, upper=None):
        """
        Negative loglikelihood function. See bounded_lognorm.pdf
        for a description of parameters.
        """
        return -np.sum(self.logpdf(x, mode, sigma, lower, upper), axis=0)

    def fit(self, data, MLE=True, verbose=False,
            fmode=None, fsigma=None, flower=None, fupper=None,
            gmode=None, gsigma=None, glower=None, gupper=None):
        """
        Estimates bounded_lognorm parameters from user-supplied random
        variates.

        Parameters
        ----------
        data : {list of floats, ndarray of floats}
            random variates
        MLE : boolean
            If True, use maximum likelihood estimation (default).
            If False, use least squares estimation.
        verbose : boolean
            If True, return verbose fit results.
        fmode, fsigma, flower, fupper : {None, float}
            parameters held to fixed constants
        gmode, gsigma, glower, gupper : {None, float}
            initial guesses, overridden by fixed parameters

        Returns
        -------
        shapes : ndarray
            shape parameters (mode, sigma, lower, upper)
        results : verbose output of fit algorithm used, if verbose=True
            If MLE==True, dictionary of optimization results
            If MLE==False, tuple containing
                popt : ndarray
                    estimated (i.e. non-fixed) parameters
                pcov : ndarray
                    estimated covariance matrix of popt

        Note
        ----
        "The diagonals [of pcov] provide the variance of the parameter
        estimate. To compute one standard deviation errors on the parameters
        use perr = np.sqrt(np.diag(pcov))." [1]

        References
        ----------
        [1] https://docs.scipy.org/doc/scipy/reference/generated
            /scipy.optimize.curve_fit.html
        """
        # Converts data to array and creates a container for shape parameters.
        data = np.asarray(data)
        sample_size = len(data)
        shapes = np.ones(4)

        # Collects fixed parameters in fparams.
        fparams = np.array([fmode, fsigma, flower, fupper])
        fmask = (fparams != np.array(None))
        shapes[fmask] = fparams[fmask]

        # Collects guessed parameters in gparams and applies default guesses
        # where none supplied by user.
        gparams = np.array([gmode, gsigma, glower, gupper])
        minmax_cushion = 0.0001  # Otherwise, fit tends to fail.
        gdefaults = np.array([np.median(data), data.std(),
                              data.min() * (1 - minmax_cushion),
                              data.max() * (1 + minmax_cushion)])
        gmask = (gparams != np.array(None))
        gparams[~gmask] = gdefaults[~gmask]
        gparams = np.array(gparams[~fmask], dtype='float64')

        # This is the objective function to be minimized by MLE below.
        # Returns the negative log-likelihood of the to-be-fitted pdf.
        def minimize_this(args):
            shapes[~fmask] = args
            return self.nnlf(data, *shapes)

        # This is the function to be fit by curve_fit below. Wraps the cdf
        # and pdf to hold fixed parameters constant.
        def curve_to_be_fit(x, *args):
            shapes[~fmask] = args
            if fit_cdf:
                return self.cdf(x, *shapes, ignore_checks=True)
            else:
                return self.pdf(x, *shapes, ignore_checks=True)

        # Why are these ignores necessary?
        with np.errstate(invalid='ignore', divide='ignore'):

            if MLE:  # maximum likelihood estimation
                results = minimize(minimize_this, gparams,
                                   method='Nelder-Mead')
                if not results.success:
                    raise Exception(results)
                else:
                    shapes[~fmask] = results.x

            else:  # non-linear least squares estimation

                try:  # Fits cdf.
                    # Converts data to ecdf.
                    xdata, counts = np.unique(data, return_counts=True)
                    ydata = np.cumsum(counts, dtype=np.double) / sample_size

                    # Fits estimated population cdf to ecdf.
                    fit_cdf = True
                    results = curve_fit(curve_to_be_fit, xdata, ydata,
                                        p0=gparams)

                except:  # If that doesn't work, tries fitting pdf instead.
                    # Converts data to histogram.
                    ydata, xdata = np.histogram(data, density=True,
                                                bins=sample_size)
                    # Sets xdata to center of bin edges.
                    xdata = np.mean([xdata[:-1], xdata[1:]], axis=0)

                    # Fits estimated population pdf to histogram.
                    fit_cdf = False
                    results = curve_fit(curve_to_be_fit, xdata, ydata,
                                        p0=gparams)

                shapes[~fmask] = results[0]

            if verbose:
                return shapes, results
            else:
                return shapes


bounded_lognorm = bounded_lognorm_gen(name='bounded_lognorm',
                                      shapes='mode, sigma, lower, upper')
