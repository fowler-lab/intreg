import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class IntReg:
    """
    Interval regression for censored, uncensored, and interval-censored data,
    using maximum likelihood estimation.

    Attributes:
        y_lower (array-like): Lower bound values of the intervals.
        y_upper (array-like): Upper bound values of the intervals.
    """

    def __init__(self, y_lower, y_upper):
        """
        Initialize model with data.

        Args:
            y_lower (array-like): Lower bounds of the intervals. Use -np.inf for left-censored values.
            y_upper (array-like): Upper bounds of the intervals. Use np.inf for right-censored values.
        """
        self.y_lower = y_lower
        self.y_upper = y_upper

    def log_L(self, params):
        """
        Compute the negative log-likelihood for the interval regression model.

        Args:
            params (array-like): Parameters to estimate. The first element is mu (mean), and the second is
            log(sigma) (log standard deviation to ensure positivity).

        Returns:
            float: Negative log-likelihood value.
        """
        # mu can be a scalar (for all observations) or vector (for each observation)
        mu = params[0]
        # sigma is optimized as log(sigma) to ensure positivity
        sigma = np.maximum(np.exp(params[1]), 1e-10)

        log_L = 0

        # likelihood function for point data
        points = self.y_upper == self.y_lower
        if np.any(points):
            w = (self.y_upper[points] - mu) ** 2 / sigma**2
            log_L += -0.5 * np.sum(w + np.log(2 * np.pi * sigma**2))

        # likelihood function for left-censored values
        left_censored = np.isin(self.y_lower, -np.inf)
        if np.any(left_censored):
            log_L += np.sum(norm.logcdf((self.y_upper[left_censored] - mu) / sigma))

        # likelihood function for right-censored values
        right_censored = np.isin(self.y_upper, np.inf)
        if np.any(right_censored):
            log_L += np.sum(
                np.log(1 - norm.cdf((self.y_lower[right_censored] - mu) / sigma))
            )

        # likelihood function for intervals
        interval_censored = ~left_censored & ~right_censored & ~points
        if np.any(interval_censored):
            log_L += np.sum(
                np.log(
                    norm.cdf((self.y_upper[interval_censored] - mu) / sigma)
                    - norm.cdf((self.y_lower[interval_censored] - mu) / sigma)
                )
            )

        return -log_L

    def _initial_params(self):
        """
        Generate automatic initial guesses for mu (mean) and log(sigma).

        Uses the mean of the midpoints (uncensored data) for mu and the standard deviation of the
        midpoints for sigma (log-transformed to ensure positivity).

        Returns:
            array: Initial guess for [mu, log(sigma)].
        """
        # Mean of uncensored data
        midpoints = (self.y_lower + self.y_upper) / 2.0
        valid_midpoints = np.where(np.isfinite(midpoints), midpoints, np.nan)
        mu = np.nanmean(valid_midpoints)

        # Standard deviation of the valid midpoints (log-transformed for positivity)
        sigma = np.nanstd(valid_midpoints)
        sigma = np.log(sigma)

        return np.array([mu, sigma])

    def fit(self, method="BFGS", initial_params=None):
        """
        Fit the interval regression model using maximum likelihood estimation.

        Args:
            method (str, optional): Optimization method to use. Defaults to "BFGS".
            initial_params (array-like, optional): Initial guesses for mu and log(sigma).
            If None, automatic initial guesses are generated.

        Returns:
            OptimizeResult: The result of the optimization process containing the estimated parameters.
        """
        if initial_params is None:
            initial_params = self._initial_params()

        result = minimize(self.log_L, initial_params, method=method)
        return result
