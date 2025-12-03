import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class MeIntReg:
    """
    Mixed-effects interval regression for censored, uncensored, and interval-censored data,
    using maximum likelihood estimation under a normal model.

    Attributes:
        y_lower (array-like): Lower bound values of the intervals.
        y_upper (array-like): Upper bound values of the intervals.
        X (array-like): Covariate matrix for fixed effects.
        random_effects (array-like or None): Group labels determining random-effect membership.
        n_random_effects (int): Number of unique random-effect groups.
    """

    def __init__(self, y_lower, y_upper, X, random_effects=None):
        """
        Initialize the model with data.

        Args:
            y_lower (array-like): Lower bounds of the intervals. Use -np.inf for left-censored values.
            y_upper (array-like): Upper bounds of the intervals. Use np.inf for right-censored values.
            X (array-like): Covariate matrix for fixed effects.
            random_effects (array-like, optional): Group labels for random intercepts.
                If None, the model includes only fixed effects.
        """
        self.y_lower = y_lower
        self.y_upper = y_upper
        self.X = X

        if random_effects is not None:
            self.random_effects = np.array(random_effects)
            self.n_random_effects = len(np.unique(random_effects))
        else:
            self.random_effects = None
            self.n_random_effects = 0

    def _compute_effects(
        self, params: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Extract fixed effects (beta), random effects (u), and compute the linear predictor (mu).

        If random effects, mu is:

            mu = X @ beta + u[group_index]

        Otherwise:

            mu = X @ beta.

        Args:
            params (array-like): Parameter vector containing beta, random effects,
                and log(sigma).

        Returns:
            tuple: (beta, u, mu)
                beta (array): Fixed-effect coefficients.
                u (array or None): Random-effect coefficients (or None if not included).
                mu (array): Mean for each observation given fixed and random effects.
        """
        n_fixed = self.X.shape[1]
        beta = params[:n_fixed]

        if self.random_effects is not None:
            u = params[n_fixed : n_fixed + self.n_random_effects]
            mu = np.dot(self.X, beta) + u[self.random_effects]
        else:
            u = None
            mu = np.dot(self.X, beta)

        return beta, u, mu

    def log_L(self, params):
        """
        Compute the negative log-likelihood for the mixed-effects interval regression model.

        Censored, uncensored, and interval-censored contributions are computed
        using normal PDF/CDF relationships. Random intercepts shift the mean
        according to group membership.

        Args:
            params (array-like): Parameters to be estimated: [beta, u, log(sigma)].

        Returns:
            float: Negative log-likelihood (with optional L2 penalties).
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

        if hasattr(self, "L2_penalties") and len(self.L2_penalties) > 0:
            log_L = self._apply_L2_regularisation(log_L, params)
        return -log_L

    def _apply_L2_regularisation(self, log_L, params):
        """
        Apply L2 regularization (ridge penalties) to the log-likelihood.
        Each penalty is scaled by the number of observations.

        Args:
            log_L (float): Unpenalized log-likelihood value.
            params (array-like): Model parameters [beta, u, log(sigma)].

        Returns:
            float: Regularized log-likelihood value.
        """
        lambda_beta = self.L2_penalties.get("lambda_beta", 0.0)
        lambda_u = self.L2_penalties.get("lambda_u", 0.0)
        lambda_sigma = self.L2_penalties.get(
            "lambda_sigma", 0.0
        )  # New regularization term

        N = self.X.shape[0]

        n_fixed = self.X.shape[1]
        beta = params[:n_fixed]
        u = params[n_fixed : n_fixed + self.n_random_effects]
        log_sigma = params[-1]

        # Compute scaled L2 penalties
        penalty_beta = (lambda_beta / N) * np.sum(np.square(beta))
        penalty_u = (lambda_u / N) * np.sum(np.square(u))
        penalty_sigma = (lambda_sigma / N) * log_sigma**2  # Regularize log(sigma)

        # Combine likelihood and regularization penalties
        return log_L - penalty_beta - penalty_u - penalty_sigma

    def _initial_params(self):
        """
        Generate automatic initial values for all parameters.

        Initial beta is obtained from a least-squares regression of the finite
        midpoints of the intervals on X. Random effects are initialized to zero.
        The standard deviation of the finite midpoints is used to initialize log(sigma).

        Returns:
            array: Initial parameter vector [beta, random effects, log(sigma)].
        """
        # Mean of uncensored data for initial beta estimate
        midpoints = (self.y_lower + self.y_upper) / 2.0
        valid_mask = np.isfinite(midpoints)

        # Filter X and midpoints to exclude rows with non-finite midpoints
        X, midpoints = self.X[valid_mask], midpoints[valid_mask]

        # Solve linear regression for beta
        beta_init = np.linalg.lstsq(X, midpoints, rcond=None)[0]

        if self.random_effects is not None:
            u_init = np.zeros(self.n_random_effects)
        else:
            u_init = []

        # Standard deviation of the valid midpoints for sigma (log-transformed for positivity)
        sigma = np.nanstd(midpoints)
        sigma = np.log(sigma)

        return np.concatenate([beta_init, u_init, [sigma]])

    def fit(
        self,
        method="BFGS",
        initial_params=None,
        bounds=None,
        options=None,
        L2_penalties=None,
    ):
        """
        Fit the mixed-effects interval regression model using maximum likelihood estimation.

        If no initial parameters are supplied, they are generated automatically.
        Optional L2 penalties may be applied to fixed effects, random effects,
        and log(sigma).

        Args:
            method (str, optional): Optimization method passed to scipy.optimize.minimize.
                Defaults to "BFGS".
            initial_params (array-like, optional): Initial values for
                [beta, random effects, log(sigma)]. If None, automatic initial
                guesses are generated.
            bounds (array-like, optional): Parameter bounds passed directly to
                minimize, applying in order to all parameters.
            options (dict, optional): Options dictionary for the optimizer.
            L2_penalties (dict or None): Optional L2 regularisation strengths,
                e.g. {"lambda_beta": ..., "lambda_u": ..., "lambda_sigma": ...}.
                Defaults to None.

        Returns:
            MeIntReg: The fitted model instance, with the optimization result.
        """
        self.L2_penalties = L2_penalties or {}

        if initial_params is None:
            initial_params = self._initial_params()

        self.result = minimize(
            self.log_L, initial_params, method=method, bounds=bounds, options=options
        )
        return self
