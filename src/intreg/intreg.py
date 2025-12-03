import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class IntReg:
    """
    Interval regression for censored, uncensored, and interval-censored data,
    using maximum likelihood estimation under a normal model.

    Attributes:
        y_lower (array-like): Lower bound values of the observed intervals.
        y_upper (array-like): Upper bound values of the observed intervals.
        weights (array-like): Optional observation weights (defaults to 1 for all observations).
    """

    def __init__(self, y_lower, y_upper, weights=None):
        """
        Initialize the model with interval-bound data.

        Args:
            y_lower (array-like): Lower bounds of the intervals. Use -np.inf for left-censored values.
            y_upper (array-like): Upper bounds of the intervals. Use np.inf for right-censored values.
            weights (array-like or None): Optional frequency or sampling weights. If None, all weights equal 1.
        """
        self.y_lower = y_lower
        self.y_upper = y_upper

        self.weights = np.ones_like(self.y_lower) if weights is None else np.asarray(weights)


    @staticmethod
    def interval_prob(y_low, y_high, mu, sigma):
        """
        Compute the likelihood contribution for each observation, covering exact,
        left-censored, right-censored, and interval-censored outcomes under a
        normal distribution.

        Args:
            y_low (array-like): Lower interval bounds (may include -np.inf).
            y_high (array-like): Upper interval bounds (may include np.inf).
            mu (float): Mean of the normal distribution.
            sigma (float): Standard deviation (must be > 0; automatically clipped).

        Returns:
            np.ndarray: Likelihood contribution (density or probability) per
                observation, shape (n,). Values are clipped to avoid underflow.
        """
        y_low = np.asarray(y_low, dtype=float)
        y_high = np.asarray(y_high, dtype=float)
        sigma = np.maximum(sigma, 1e-10)

        # Identify censoring types
        points = (y_low == y_high)
        left_cens = np.isneginf(y_low)
        right_cens = np.isposinf(y_high)
        interval_cens = ~(points | left_cens | right_cens)

        p = np.zeros_like(y_low, dtype=float)

        # Point (exact) observations: PDF
        if np.any(points):
            p[points] = norm.pdf(y_high[points], mu, sigma)

        # Left-censored observations: CDF
        if np.any(left_cens):
            p[left_cens] = norm.cdf((y_high[left_cens] - mu) / sigma)

        # Right-censored observations: survival function (1 - CDF)
        if np.any(right_cens):
            p[right_cens] = 1 - norm.cdf((y_low[right_cens] - mu) / sigma)

        # Interval-censored → CDF difference
        if np.any(interval_cens):
            p[interval_cens] = (
                norm.cdf((y_high[interval_cens] - mu) / sigma)
                - norm.cdf((y_low[interval_cens] - mu) / sigma)
            )

        # Avoid underflow / log(0)
        p = np.clip(p, 1e-300, 1.0)
        return p


    def log_L(self, params):
        """
        Compute the negative log-likelihood for the interval regression model.

        The parameter vector is interpreted as:
            params[0] = mu
            params[1] = log(sigma)

        Args:
            params (array-like): Model parameters [mu, log_sigma].

        Returns:
            float: Negative log-likelihood (with optional regularisation).
        """
        mu = params[0]
        sigma = np.maximum(np.exp(params[1]), 1e-10)
        p = IntReg.interval_prob(self.y_lower, self.y_upper, mu, sigma)
        log_L = np.sum(self.weights * np.log(p))

        if hasattr(self, "L2_penalties") and len(self.L2_penalties) > 0:
            log_L = self._apply_L2_regularisation(log_L, params)

        return -log_L


    def _apply_L2_regularisation(self, log_L, params):
        """
        Apply L2 (ridge) penalties to the log-likelihood.
        These penalties are scaled by the number of observations.

        Args:
            log_L (float): Unpenalized log-likelihood.
            params (array-like): Parameter vector, where the first n_fixed
                parameters are penalized as 'beta' and the final parameter is
                log(sigma).

        Returns:
            float: Regularized log-likelihood.
        """
        lambda_beta = self.L2_penalties.get("lambda_beta", 0.0)
        lambda_sigma = self.L2_penalties.get("lambda_sigma", 0.0)

        n_fixed = self.L2_penalties.get("n_fixed", len(params) - 1)
        beta = params[:n_fixed]  # Fixed effects (first n_fixed params)
        log_sigma = params[-1]  # log(sigma) is the last parameter

        # Compute scaled L2 penalties
        penalty_beta = (lambda_beta / len(self.y_lower)) * np.sum(np.square(beta))
        penalty_sigma = (lambda_sigma / len(self.y_lower)) * np.square(log_sigma)

        # Combine likelihood and regularization penalties
        return log_L - penalty_beta - penalty_sigma

    def _initial_params(self):
        """
        Generate automatic initial guesses for mu and log(sigma).

        Midpoints of the intervals are used, and only finite midpoints
        contribute to the initial estimates. The mean of finite midpoints is
        mu, and their standard deviation (log-transformed) is log(sigma).

        Returns:
            array: Initial parameter vector [mu, log(sigma)].
        """
        # Mean of uncensored data
        midpoints = (self.y_lower + self.y_upper) / 2.0
        valid_midpoints = np.where(np.isfinite(midpoints), midpoints, np.nan)
        mu = np.nanmean(valid_midpoints)

        # Standard deviation of the valid midpoints (log-transformed for positivity)
        sigma = np.nanstd(valid_midpoints)
        sigma = np.log(sigma)

        return np.array([mu, sigma])

    def fit(
        self,
        method="BFGS",
        initial_params=None,
        bounds=None,
        options=None,
        L2_penalties=None,
    ):
        """
        Fit the interval regression model by maximum likelihood estimation.

        The optimization is performed over the parameters:
            [mu, log(sigma)]

        Initial parameters are either user-supplied or generated automatically.
        Optional L2 penalties may be applied to mu and/or log-sigma.

        Args:
            method (str, optional): Optimization method to use. Defaults to "BFGS".
            initial_params (array-like, optional): Initial guesses for beta, random effects, and log(sigma).
                If None, automatic initial guesses are generated.
            bounds (array-like, optional): bounds for sigma
            options (dict, optional): scipy minimisation options dictionary
            L2_penalties (dict or None): Regularisation strengths for fixed and random effects {lambda_beta:..., lambda_u:...}. Defaults to None

        Returns:
            IntReg: The fitted model instance containing the `result` attribute.
        """
        self.L2_penalties = L2_penalties or {}

        if initial_params is None:
            initial_params = self._initial_params()

        self.result = minimize(
            self.log_L, initial_params, method=method, bounds=bounds, options=options
        )

        return self
