from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize
from scipy.special import log_ndtr
from scipy.stats import norm

ArrayLike = npt.ArrayLike
FloatArray = npt.NDArray[np.float64]


class IntReg:
    """
    Interval regression for censored, uncensored, and interval-censored data,
    using maximum likelihood estimation under a normal model.

    Attributes:
        y_lower (array-like): One-dimensional lower bounds for the observations.
            Use -np.inf for left-censored values.
        y_upper (array-like): One-dimensional upper bounds for the observations.
            Use np.inf for right-censored values.
        weights (array-like): Optional observation weights. If None, each
            observation has weight 1.
    """

    result: OptimizeResult

    def __init__(
        self,
        y_lower: ArrayLike,
        y_upper: ArrayLike,
        weights: Optional[ArrayLike] = None,
    ) -> None:
        """
        Initialize the model with interval-bound data.

        Args:
            y_lower (array-like): One-dimensional lower bounds. Use -np.inf
                for left-censored values.
            y_upper (array-like): One-dimensional upper bounds. Use np.inf
                for right-censored values.
            weights (array-like or None): Optional frequency or sampling
                weights. If None, all weights equal 1.

        Raises:
            ValueError: If the bounds or weights are not one-dimensional and
                of equal length, or if no observations are supplied.

        Returns:
            None
        """
        self.y_lower = np.asarray(y_lower, dtype=float)
        self.y_upper = np.asarray(y_upper, dtype=float)
        if self.y_lower.ndim != 1 or self.y_upper.ndim != 1:
            raise ValueError("y_lower and y_upper must be one-dimensional arrays.")
        if self.y_lower.shape != self.y_upper.shape:
            raise ValueError("y_lower and y_upper must have the same length.")
        if self.y_lower.size == 0:
            raise ValueError("At least one observation is required.")

        self.weights = (
            np.ones_like(self.y_lower)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if self.weights.shape != self.y_lower.shape:
            raise ValueError("weights must have the same length as the interval bounds.")
        self.L2_penalties: dict[str, Any] = {}


    @staticmethod
    def interval_prob(
        y_low: ArrayLike, y_high: ArrayLike, mu: float, sigma: float
    ) -> FloatArray:
        """
        Compute likelihood contributions on the probability scale.

        This method is retained for backwards compatibility. New code should
        use `interval_log_likelihood` to avoid underflow for small likelihoods.

        Args:
            y_low (array-like): Lower interval bounds (may include -np.inf).
            y_high (array-like): Upper interval bounds (may include np.inf).
            mu (float): Mean of the normal distribution.
            sigma (float): Standard deviation (must be > 0; automatically clipped).

        Returns:
            np.ndarray: Likelihood contribution (density or probability) per
                observation, shape (n,). Values are clipped to [1e-300, 1].
        """
        log_likelihood = IntReg.interval_log_likelihood(y_low, y_high, mu, sigma)
        return np.clip(np.exp(log_likelihood), 1e-300, 1.0)


    @staticmethod
    def interval_log_likelihood(
        y_low: ArrayLike, y_high: ArrayLike, mu: float, sigma: float
    ) -> FloatArray:
        """
        Compute per-observation log-likelihood contributions under a normal
        distribution for exact, left-censored, right-censored, and
        interval-censored observations.

        Args:
            y_low (array-like): Lower interval bounds. Use -np.inf for
                left-censored observations.
            y_high (array-like): Upper interval bounds. Use np.inf for
                right-censored observations.
            mu (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution.
                Values below 1e-10 are clipped to 1e-10.

        Returns:
            np.ndarray: Per-observation log-likelihood contributions, with
            shape (n,). Exact observations contribute log densities; censored
            and interval-censored observations contribute log probabilities.
        """
        y_low = np.asarray(y_low, dtype=float)
        y_high = np.asarray(y_high, dtype=float)
        sigma_value = float(np.maximum(sigma, 1e-10))
        return IntReg._log_likelihood_contributions(y_low, y_high, mu, sigma_value)


    @staticmethod
    def _log_interval_prob(lower: FloatArray, upper: FloatArray) -> FloatArray:
        """
        Compute the log probability of a standard normal value falling between
        the supplied lower and upper bounds.

        The calculation uses the CDF in the left tail and the survival function
        in the right tail to reduce loss of precision.

        Args:
            lower (array-like): Standardized lower bounds of the intervals.
            upper (array-like): Standardized upper bounds of the intervals.

        Returns:
            np.ndarray: Log probabilities for the intervals, with the same
            shape as the supplied bounds.
        """
        logcdf_upper = log_ndtr(upper)
        logcdf_lower = log_ndtr(lower)
        logsf_lower = log_ndtr(-lower)
        logsf_upper = log_ndtr(-upper)

        # Use CDFs in the left tail and survival functions in the right tail.
        use_cdf = upper <= 0
        log_base = np.where(use_cdf, logcdf_upper, logsf_lower)
        log_small = np.where(use_cdf, logcdf_lower, logsf_upper)
        delta = np.minimum(log_small - log_base, -np.finfo(float).eps)

        cutoff = -np.log(2.0)
        log_one_minus_exp = np.where(
            delta < cutoff,
            np.log1p(-np.exp(np.minimum(delta, cutoff))),
            np.log(-np.expm1(np.maximum(delta, cutoff))),
        )
        return cast(FloatArray, log_base + log_one_minus_exp)


    @staticmethod
    def _log_likelihood_contributions(
        y_low: FloatArray, y_high: FloatArray, mu: float, sigma: float
    ) -> FloatArray:
        """
        Return log-likelihood contributions for exact, left-censored,
        right-censored, and interval-censored observations.

        Args:
            y_low (array-like): Lower bounds of the observations. Use -np.inf
                for left-censored observations.
            y_high (array-like): Upper bounds of the observations. Use np.inf
                for right-censored observations.
            mu (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution.

        Returns:
            np.ndarray: Log-likelihood contribution for each observation,
            with shape (n,).
        """
        points = y_low == y_high
        left_cens = np.isneginf(y_low)
        right_cens = np.isposinf(y_high)
        interval_cens = ~(points | left_cens | right_cens)
        contributions = np.zeros_like(y_low, dtype=float)

        if np.any(points):
            contributions[points] = norm.logpdf(y_high[points], loc=mu, scale=sigma)
        if np.any(left_cens):
            z_upper = (y_high[left_cens] - mu) / sigma
            contributions[left_cens] = log_ndtr(z_upper)
        if np.any(right_cens):
            z_lower = (y_low[right_cens] - mu) / sigma
            contributions[right_cens] = log_ndtr(-z_lower)
        if np.any(interval_cens):
            z_lower = (y_low[interval_cens] - mu) / sigma
            z_upper = (y_high[interval_cens] - mu) / sigma
            contributions[interval_cens] = IntReg._log_interval_prob(z_lower, z_upper)

        return contributions


    def log_L(self, params: ArrayLike) -> float:
        """
        Compute the negative log-likelihood for the interval regression model.
        Exact observations use the normal density, while censored observations
        use their corresponding probabilities. The calculation is performed in
        log space to reduce numerical underflow.

        The parameter vector is interpreted as:
            params[0] = mu
            params[1] = log(sigma)

        Args:
            params (array-like): Model parameters [mu, log_sigma], where
                mu is the normal mean and log_sigma is the log standard
                deviation.

        Returns:
            float: Negative log-likelihood, with optional L2 regularisation.
        """
        params_array = np.asarray(params, dtype=float)
        mu = float(params_array[0])
        sigma = float(np.maximum(np.exp(params_array[1]), 1e-10))
        log_likelihood = self.interval_log_likelihood(
            self.y_lower, self.y_upper, mu, sigma
        )
        log_L = float(np.sum(self.weights * log_likelihood))

        if hasattr(self, "L2_penalties") and len(self.L2_penalties) > 0:
            log_L = self._apply_L2_regularisation(log_L, params_array)

        return float(-log_L)


    def _apply_L2_regularisation(self, log_L: float, params: ArrayLike) -> float:
        """
        Apply L2 (ridge) penalties to the log-likelihood. The penalties are
        scaled by the number of observations.

        Args:
            log_L (float): Unpenalized log-likelihood.
            params (array-like): Parameter vector [mu, log_sigma]. The first
                n_fixed parameters (one by default) receive the beta penalty;
                the final parameter receives the log(sigma) penalty.

        Returns:
            float: Regularized log-likelihood.
        """
        params_array = np.asarray(params, dtype=float)
        lambda_beta = self.L2_penalties.get("lambda_beta", 0.0)
        lambda_sigma = self.L2_penalties.get("lambda_sigma", 0.0)

        n_fixed = self.L2_penalties.get("n_fixed", len(params_array) - 1)
        beta = params_array[:n_fixed]  # Fixed effects (first n_fixed params)
        log_sigma = params_array[-1]  # log(sigma) is the last parameter

        # Compute scaled L2 penalties
        penalty_beta = (lambda_beta / len(self.y_lower)) * np.sum(np.square(beta))
        penalty_sigma = (lambda_sigma / len(self.y_lower)) * np.square(log_sigma)

        # Combine likelihood and regularization penalties
        return float(log_L - penalty_beta - penalty_sigma)

    def _initial_params(self) -> FloatArray:
        """
        Generate automatic initial guesses for mu and log(sigma).

        Finite interval midpoints are used to estimate mu and sigma. A small
        positive floor is applied to sigma so identical midpoints still give
        a finite starting value. If no finite midpoints are available, the
        starting values are mu=0 and sigma=1.

        Returns:
            array: Initial parameter vector [mu, log(sigma)].
        """
        # Mean of uncensored data
        midpoints = (self.y_lower + self.y_upper) / 2.0
        valid_midpoints = np.where(np.isfinite(midpoints), midpoints, np.nan)

        # Standard deviation of the valid midpoints (log-transformed for positivity)
        if np.all(np.isnan(valid_midpoints)):
            mu = 0.0
            sigma = 1.0
        else:
            mu = float(np.nanmean(valid_midpoints))
            sigma = max(float(np.nanstd(valid_midpoints)), 1e-6)
        log_sigma = float(np.log(sigma))

        return np.array([mu, log_sigma], dtype=float)

    def fit(
        self,
        method: str = "BFGS",
        initial_params: Optional[ArrayLike] = None,
        bounds: Any = None,
        options: Optional[dict[str, Any]] = None,
        L2_penalties: Optional[dict[str, Any]] = None,
    ) -> "IntReg":
        """
        Fit the interval regression model by maximum likelihood estimation.

        The optimization is performed over the parameters [mu, log(sigma)].
        The standard deviation is represented on the log scale to keep it
        positive.

        Initial parameters are either user-supplied or generated automatically.
        Optional L2 penalties may be applied to mu and log(sigma).

        Args:
            method (str, optional): Optimization method to use. Defaults to "BFGS".
            initial_params (array-like, optional): Initial values for
                [mu, log(sigma)]. If None, automatic initial values are used.
            bounds (array-like, optional): Bounds passed to
                scipy.optimize.minimize for the two parameters.
            options (dict, optional): Options passed to
                scipy.optimize.minimize.
            L2_penalties (dict or None): Optional regularisation strengths,
                such as {"lambda_beta": ..., "lambda_sigma": ...}. Defaults
                to None.

        Returns:
            IntReg: The fitted model instance, with the optimizer result
            available as `result`.
        """
        self.L2_penalties = L2_penalties or {}

        if initial_params is None:
            initial_params = self._initial_params()

        self.result = minimize(
            self.log_L, initial_params, method=method, bounds=bounds, options=options
        )

        return self
