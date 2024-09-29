import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class MeIntReg:
    """
    Mixed-effects interval regression for censored, uncensored, and interval-censored data,
    using maximum likelihood estimation.

    Attributes:
        y_lower (array-like): Lower bound values of the intervals.
        y_upper (array-like): Upper bound values of the intervals.
        X (array-like): Covariate matrix for fixed effects.
        clusters (array-like): Cluster labels for random effects.
    """

    def __init__(self, y_lower, y_upper, X, clusters):
        """
        Initialize the model with data.

        Args:
            y_lower (array-like): Lower bounds of the intervals. Use -np.inf for left-censored values.
            y_upper (array-like): Upper bounds of the intervals. Use np.inf for right-censored values.
            X (array-like): Covariate matrix for fixed effects.
            clusters (array-like): Cluster labels for grouping random effects.
        """
        self.y_lower = y_lower
        self.y_upper = y_upper
        self.X = X
        self.clusters = np.array(clusters)
        self.n_clusters = len(np.unique(clusters))

    def _compute_effects(self, params):
        """
        Extract fixed effects (beta), random effects (u), and compute the combined mean (mu).

        Args:
            params (array-like): Parameters containing beta, random effects, and log(sigma).

        Returns:
            tuple: (beta, u, mu), where:
                beta (array): Fixed effects coefficients.
                u (array): Random effects for each cluster.
                mu (array): Combined mean for each observation.
        """
        n_fixed = self.X.shape[1]
        beta = params[:n_fixed]  # Fixed effects
        u = params[n_fixed : n_fixed + self.n_clusters]  # Random effects for clusters

        # Map random effects to samples based on their clusters
        mu = np.dot(self.X, beta) + u[self.clusters]

        return beta, u, mu

    def log_L(self, params):
        """
        Compute the negative log-likelihood for the mixed-effects interval regression model.

        Args:
            params (array-like): Parameters to estimate. The first elements are beta (fixed effects coefficients),
            followed by the random effects for clusters, and the last element is log(sigma).

        Returns:
            float: Negative log-likelihood value.
        """
        _, _, mu = self._compute_effects(params)
        sigma = np.maximum(
            np.exp(params[-1]), 1e-10
        )  # Log-transformed sigma for positivity

        log_L = 0

        # Likelihood function for point data
        points = self.y_upper == self.y_lower
        if np.any(points):
            log_L += np.sum(norm.logpdf((self.y_upper[points] - mu[points]) / sigma))

        # Likelihood function for left-censored values
        left_censored = np.isin(self.y_lower, -np.inf)
        if np.any(left_censored):
            log_L += np.sum(
                norm.logcdf((self.y_upper[left_censored] - mu[left_censored]) / sigma)
            )

        # Likelihood function for right-censored values
        right_censored = np.isin(self.y_upper, np.inf)
        if np.any(right_censored):
            log_L += np.sum(
                np.log(
                    1
                    - norm.cdf(
                        (self.y_lower[right_censored] - mu[right_censored]) / sigma
                    )
                )
            )

        # Likelihood function for intervals
        interval_censored = ~left_censored & ~right_censored & ~points
        if np.any(interval_censored):
            log_L += np.sum(
                np.log(
                    norm.cdf(
                        (self.y_upper[interval_censored] - mu[interval_censored])
                        / sigma
                    )
                    - norm.cdf(
                        (self.y_lower[interval_censored] - mu[interval_censored])
                        / sigma
                    )
                )
            )

        return -log_L

    def _initial_params(self):
        """
        Generate automatic initial guesses for parameters.

        Uses linear regression coefficients as initial guesses for beta, zeros for random effects,
        and the log of the standard deviation of the midpoints for log(sigma).

        Returns:
            array: Initial guess for [beta, random effects, log(sigma)].
        """
        # Mean of uncensored data for initial beta estimate
        midpoints = (self.y_lower + self.y_upper) / 2.0
        valid_midpoints = np.where(np.isfinite(midpoints), midpoints, np.nan)

        # Solve linear regression for beta
        beta_init = np.linalg.lstsq(self.X, valid_midpoints, rcond=None)[0]

        # Initial guess for random effects as zeros
        u_init = np.zeros(self.n_clusters)

        # Standard deviation of the midpoints for sigma
        sigma_init = np.nanstd(valid_midpoints)
        log_sigma_init = np.log(sigma_init)

        return np.concatenate([beta_init, u_init, [log_sigma_init]])

    def fit(self, method="BFGS", initial_params=None):
        """
        Fit the mixed-effects interval regression model using maximum likelihood estimation.

        Args:
            method (str, optional): Optimization method to use. Defaults to "BFGS".
            initial_params (array-like, optional): Initial guesses for beta, random effects, and log(sigma).
            If None, automatic initial guesses are generated.

        Returns:
            OptimizeResult: The result of the optimization process containing the estimated parameters.
        """
        if initial_params is None:
            initial_params = self._initial_params()

        result = minimize(self.log_L, initial_params, method=method)
        return result
