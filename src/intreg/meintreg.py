from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from numpy.polynomial.hermite import hermgauss
from scipy.optimize import OptimizeResult, minimize

from jax import config
config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax
import jax.numpy as jnp
import jax.scipy as jsp

ArrayLike = npt.ArrayLike
FloatArray = npt.NDArray[np.float64]


class MeIntReg:
    """
    Mixed-effects interval regression for censored, uncensored, and interval-censored data,
    using marginal maximum likelihood under a Gaussian random-intercept model.

    Without random effects:
        [beta..., log_sigma]

    With random effects:
        [beta..., log_sigma, log_tau]

    where:
        beta  = fixed-effect coefficients
        sigma = residual standard deviation
        tau   = random-intercept standard deviation

    Model:
        y* = X beta + u_group + epsilon
        u_group ~ N(0, tau^2)
        epsilon ~ N(0, sigma^2)

    Random effects are integrated out using Gauss-Hermite quadrature.

    Attributes:
        y_lower (np.ndarray): Lower bounds of the observation intervals.
        y_upper (np.ndarray): Upper bounds of the observation intervals.
        X (np.ndarray): Fixed-effect design matrix.
        random_effects (np.ndarray or None): Encoded group labels, or None
            when the model has no random effects.
        n_random_effects (int): Number of random-effect groups.
        result (OptimizeResult): Optimization result created by `fit`.
    """

    result: OptimizeResult

    def __init__(
        self,
        y_lower: ArrayLike,
        y_upper: ArrayLike,
        X: ArrayLike,
        random_effects: Optional[ArrayLike] = None,
        quadrature_points: int = 20,
        n_jobs: int = -1,
    ) -> None:
        """
        Initialize the model.

        Args:
            y_lower (array-like): Lower interval bounds. Use -np.inf for
                left-censored observations.
            y_upper (array-like): Upper interval bounds. Use np.inf for
                right-censored observations.
            X (array-like): Fixed-effect design matrix, with one row per
                observation.
            random_effects (array-like or None): Optional group labels
                defining random-intercept membership. If None, only fixed
                effects are fitted.
            quadrature_points (int): Number of Gauss-Hermite quadrature points
                used to integrate the random effects. Defaults to 20.
            n_jobs (int): Retained for backwards compatibility.

        Returns:
            None
        """

        self.y_lower = np.asarray(y_lower, dtype=float)
        self.y_upper = np.asarray(y_upper, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.quadrature_points = quadrature_points
        self.n_jobs = n_jobs

        self._y_lower = jnp.asarray(self.y_lower)
        self._y_upper = jnp.asarray(self.y_upper)
        self._X = jnp.asarray(self.X)

        self.random_effects: Optional[Any] = None
        self.n_random_effects: int = 0
        self.group_indices: Optional[List[Any]] = None
        self._group_indices: Optional[List[Any]] = None
        self.gh_nodes: Optional[Any] = None
        self.gh_weights: Optional[Any] = None
        self.log_gh_weights: Optional[Any] = None
        self._gh_nodes: Optional[Any] = None
        self._log_gh_weights: Optional[Any] = None

        if random_effects is not None:
            random_effects = np.asarray(random_effects)
            _, group_codes = np.unique(random_effects, return_inverse=True)

            self.random_effects = group_codes
            self.n_random_effects = len(np.unique(group_codes))
            self.group_indices = [np.flatnonzero(group_codes == g) for g in range(self.n_random_effects)]
            self._group_indices = [jnp.asarray(indices, dtype=jnp.int32) for indices in self.group_indices]

            self.gh_nodes, self.gh_weights = hermgauss(quadrature_points)
            self.log_gh_weights = np.log(self.gh_weights)

            self._gh_nodes = jnp.asarray(self.gh_nodes)
            self._log_gh_weights = jnp.asarray(self.log_gh_weights)
        else:
            self.random_effects = None
            self.n_random_effects = 0
            self.group_indices = None
            self._group_indices = None
            self.gh_nodes = None
            self.gh_weights = None
            self.log_gh_weights = None
            self._gh_nodes = None
            self._log_gh_weights = None

        self.L2_penalties: Dict[str, float] = {}
        self._value_and_grad: Any = None
        self._hessian_fn: Any = None

    def _unpack_params(
        self, params: Any
    ) -> Tuple[Any, Any, Optional[Any]]:
        """
        Unpack the model parameter vector.

        Args:
            params (array-like): Model parameters. The first `X.shape[1]`
                values are fixed effects, followed by log_sigma and, when
                random effects are present, log_tau.

        Returns:
            tuple: Fixed-effect coefficients, log residual standard deviation,
            and log random-effect standard deviation (or None).
        """

        n_fixed = self.X.shape[1]
        beta = params[:n_fixed]
        log_sigma = params[n_fixed]
        log_tau = params[n_fixed + 1] if self.random_effects is not None else None

        return beta, log_sigma, log_tau

    @staticmethod
    def _log1mexp(x: Any) -> Any:
        """
        Compute log(1 - exp(x)) stably for x <= 0.

        Args:
            x (array-like): Values less than or equal to zero.

        Returns:
            jax.Array: Stable log(1 - exp(x)) values.
        """

        x = jnp.minimum(x, -1e-15)
        cutoff = -jnp.log(2.0)

        x_left = jnp.minimum(x, cutoff - 1e-15)
        x_right = jnp.maximum(x, cutoff)

        left = jnp.log1p(-jnp.exp(x_left))
        right = jnp.log(-jnp.expm1(x_right))

        return jnp.where(x < cutoff, left, right)

    @staticmethod
    def _log_interval_prob(lower: Any, upper: Any) -> Any:
        """
        Compute log[P(lower < Z < upper)] for Z ~ N(0, 1) stably.

        Args:
            lower (array-like): Standardized lower interval bounds.
            upper (array-like): Standardized upper interval bounds.

        Returns:
            jax.Array: Log probabilities for the intervals.
        """

        logcdf_upper = jsp.special.log_ndtr(upper)
        logcdf_lower = jsp.special.log_ndtr(lower)

        delta_cdf = jnp.minimum(logcdf_lower - logcdf_upper, -1e-15)
        log_prob_cdf = logcdf_upper + MeIntReg._log1mexp(delta_cdf)

        logsf_lower = jsp.special.log_ndtr(-lower)
        logsf_upper = jsp.special.log_ndtr(-upper)

        delta_sf = jnp.minimum(logsf_upper - logsf_lower, -1e-15)
        log_prob_sf = logsf_lower + MeIntReg._log1mexp(delta_sf)

        return jnp.where(upper <= 0, log_prob_cdf, log_prob_sf)

    @staticmethod
    def _normal_logpdf(y: Any, mu: Any, sigma: Any) -> Any:
        """
        Compute the log-density of a normal distribution.

        Args:
            y (array-like): Observed values.
            mu (array-like): Normal means.
            sigma (float or array-like): Normal standard deviation.

        Returns:
            jax.Array: Normal log-density values.
        """

        z = (y - mu) / sigma
        return -0.5 * z**2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)

    def _observation_loglik(self, indices: Any, mu: Any, sigma: Any) -> Any:
        """
        Compute the conditional log-likelihood for selected observations.

        Args:
            indices (array-like): Indices of observations to include.
            mu (array-like): Conditional means for the observations.
            sigma (float): Residual standard deviation.

        Returns:
            jax.Array: Sum of the log-likelihood contributions.
        """

        y_lower = self._y_lower[indices]
        y_upper = self._y_upper[indices]

        points = y_upper == y_lower
        left = jnp.isneginf(y_lower)
        right = jnp.isposinf(y_upper)
        interval = ~(points | left | right)

        safe_y = jnp.where(jnp.isfinite(y_upper), y_upper, 0.0)
        point_ll = self._normal_logpdf(safe_y, mu, sigma)
        point_ll = jnp.where(points, point_ll, 0.0)

        left_upper = jnp.where(left, y_upper, mu)
        z_upper = (left_upper - mu) / sigma
        left_ll = jnp.where(left, jsp.special.log_ndtr(z_upper), 0.0)

        right_lower = jnp.where(right, y_lower, mu)
        z_lower = (right_lower - mu) / sigma
        right_ll = jnp.where(right, jsp.special.log_ndtr(-z_lower), 0.0)

        safe_lower = jnp.where(interval, y_lower, 0.0)
        safe_upper = jnp.where(interval, y_upper, 1.0)

        interval_lower = (safe_lower - mu) / sigma
        interval_upper = (safe_upper - mu) / sigma
        interval_ll = self._log_interval_prob(interval_lower, interval_upper)
        interval_ll = jnp.where(interval, interval_ll, 0.0)

        return jnp.sum(point_ll + left_ll + right_ll + interval_ll)

    def _group_loglik(
        self, indices: Any, fixed_mu: Any, sigma: Any, tau: Any
    ) -> Any:
        """
        Marginal log-likelihood contribution for one random-effect group.
        All quadrature points are evaluated simultaneously.

        Args:
            indices (array-like): Indices belonging to the group.
            fixed_mu (array-like): Fixed-effect means for all observations.
            sigma (float): Residual standard deviation.
            tau (float): Random-intercept standard deviation.

        Returns:
            jax.Array: Marginal log-likelihood contribution for the group.
        """

        assert self._gh_nodes is not None
        assert self._log_gh_weights is not None

        y_lower = self._y_lower[indices]
        y_upper = self._y_upper[indices]
        group_fixed_mu = fixed_mu[indices]

        u = jnp.sqrt(2.0) * tau * self._gh_nodes
        mu = group_fixed_mu[None, :] + u[:, None]

        points = y_upper == y_lower
        left = jnp.isneginf(y_lower)
        right = jnp.isposinf(y_upper)
        interval = ~(points | left | right)

        safe_y = jnp.where(jnp.isfinite(y_upper), y_upper, 0.0)
        point_ll = self._normal_logpdf(safe_y[None, :], mu, sigma)
        point_ll = jnp.where(points[None, :], point_ll, 0.0)

        left_upper = jnp.where(left[None, :], y_upper[None, :], mu)
        z_upper = (left_upper - mu) / sigma
        left_ll = jnp.where(left[None, :], jsp.special.log_ndtr(z_upper), 0.0)

        right_lower = jnp.where(right[None, :], y_lower[None, :], mu)
        z_lower = (right_lower - mu) / sigma
        right_ll = jnp.where(right[None, :], jsp.special.log_ndtr(-z_lower), 0.0)

        safe_lower = jnp.where(interval, y_lower, 0.0)
        safe_upper = jnp.where(interval, y_upper, 1.0)

        interval_lower = (safe_lower[None, :] - mu) / sigma
        interval_upper = (safe_upper[None, :] - mu) / sigma
        interval_ll = self._log_interval_prob(interval_lower, interval_upper)
        interval_ll = jnp.where(interval[None, :], interval_ll, 0.0)

        node_loglik = jnp.sum(point_ll + left_ll + right_ll + interval_ll, axis=1)

        return jsp.special.logsumexp(self._log_gh_weights + node_loglik) - 0.5 * jnp.log(jnp.pi)

    def _apply_L2_regularisation(self, log_L: Any, params: Any) -> Any:
        """
        Apply optional L2 regularisation to the log-likelihood.

        Args:
            log_L (float or jax.Array): Unregularised log-likelihood.
            params (array-like): Model parameter vector containing fixed
                effects and log standard deviations.

        Returns:
            jax.Array: Regularised log-likelihood.
        """

        lambda_beta = self.L2_penalties.get("lambda_beta", 0.0)
        lambda_sigma = self.L2_penalties.get("lambda_sigma", 0.0)
        lambda_tau = self.L2_penalties.get("lambda_tau", 0.0)

        N = self.X.shape[0]
        beta, log_sigma, log_tau = self._unpack_params(params)

        penalty_beta = (lambda_beta / N) * jnp.sum(beta**2)
        penalty_sigma = (lambda_sigma / N) * log_sigma**2
        penalty_tau = (lambda_tau / N) * log_tau**2 if log_tau is not None else 0.0

        return log_L - penalty_beta - penalty_sigma - penalty_tau

    def _neg_loglik(self, params: Any) -> Any:
        """
        Compute the negative marginal log-likelihood.

        Args:
            params (array-like): Fixed-effect coefficients and log standard
                deviations, ordered as described in the class documentation.

        Returns:
            jax.Array: Negative marginal log-likelihood.
        """

        beta, log_sigma, log_tau = self._unpack_params(params)

        sigma = jnp.exp(log_sigma)
        fixed_mu = self._X @ beta

        if self.random_effects is None:
            indices = jnp.arange(self._y_lower.shape[0], dtype=jnp.int32)
            log_L = self._observation_loglik(indices, fixed_mu, sigma)
        else:
            assert log_tau is not None
            assert self._group_indices is not None
            tau = jnp.exp(log_tau)
            log_L = jnp.array(0.0, dtype=jnp.float64)

            for indices in self._group_indices:
                log_L += self._group_loglik(indices, fixed_mu, sigma, tau)

        if len(self.L2_penalties) > 0:
            log_L = self._apply_L2_regularisation(log_L, params)

        return -log_L

    def log_L(self, params: ArrayLike) -> float:
        """
        Compute the negative marginal log-likelihood for supplied parameters.

        Args:
            params (array-like): Fixed-effect coefficients and log standard
                deviations, ordered as described in the class documentation.

        Returns:
            float: Negative marginal log-likelihood.
        """

        return float(self._neg_loglik(jnp.asarray(params, dtype=jnp.float64)))

    def _initial_params(self) -> FloatArray:
        """
        Generate automatic initial parameter values.

        Without random effects:
            [beta..., log_sigma]

        With random effects:
            [beta..., log_sigma, log_tau]

        Returns:
            np.ndarray: Initial parameter vector for the model.
        """

        midpoints = (self.y_lower + self.y_upper) / 2.0
        valid_mask = np.isfinite(midpoints)

        X_valid = self.X[valid_mask]
        midpoints_valid = midpoints[valid_mask]

        beta_init = np.linalg.lstsq(X_valid, midpoints_valid, rcond=None)[0]

        sigma_init = max(np.nanstd(midpoints_valid), 1e-6)
        log_sigma_init = np.log(sigma_init)

        if self.random_effects is not None:
            tau_init = 0.5
            log_tau_init = np.log(tau_init)
            return np.concatenate([beta_init, [log_sigma_init, log_tau_init]])

        return np.concatenate([beta_init, [log_sigma_init]])

    def _objective_with_grad(self, params: ArrayLike) -> Tuple[float, FloatArray]:
        """
        Return the objective value and gradient for scipy.optimize.minimize.

        Args:
            params (array-like): Model parameter vector.

        Returns:
            tuple: Objective value as a float and gradient as a NumPy array.
        """

        if self._value_and_grad is None:
            self._value_and_grad = jax.jit(jax.value_and_grad(self._neg_loglik))

        value, gradient = self._value_and_grad(
            jnp.asarray(np.asarray(params, dtype=float), dtype=jnp.float64)
        )

        return float(value), np.asarray(gradient, dtype=float)

    def hessian(self, params: Optional[ArrayLike] = None) -> FloatArray:
        """
        Calculate the Hessian of the negative marginal log-likelihood.

        If params is None, use the fitted parameter vector.

        Args:
            params (array-like or None): Parameter vector at which to compute
                the Hessian. If None, the fitted parameters are used.

        Returns:
            np.ndarray: Hessian matrix of the negative marginal
            log-likelihood.

        Raises:
            ValueError: If params is None and the model has not been fitted.
        """

        if params is None:
            if not hasattr(self, "result"):
                raise ValueError("The model must be fitted before requesting the Hessian.")
            params = self.result.x

        if self._hessian_fn is None:
            self._hessian_fn = jax.jit(jax.hessian(self._neg_loglik))

        H = self._hessian_fn(jnp.asarray(params, dtype=jnp.float64))
        return np.asarray(H, dtype=float)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Remove compiled functions before serialization.

        Returns:
            dict: Model state with compiled JAX functions removed.
        """
        state = self.__dict__.copy()
        state["_value_and_grad"] = None
        state["_hessian_fn"] = None
        return state


    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restore model state after deserialization.

        Args:
            state (dict): Previously serialized model state.

        Returns:
            None
        """
        self.__dict__.update(state)
        self._value_and_grad = None
        self._hessian_fn = None

    def fit(
        self,
        method: str = "BFGS",
        initial_params: Optional[ArrayLike] = None,
        bounds: Any = None,
        options: Optional[Dict[str, Any]] = None,
        L2_penalties: Optional[Dict[str, float]] = None,
    ) -> "MeIntReg":
        """
        Fit the interval-regression model using marginal maximum likelihood.

        Parameters are ordered as [beta..., log_sigma] without random effects
        and [beta..., log_sigma, log_tau] with random effects.

        Args:
            method (str): Optimization method passed to
                scipy.optimize.minimize. Defaults to "BFGS".
            initial_params (array-like or None): Optional initial parameter
                vector. If None, initial values are generated automatically.
            bounds (sequence or None): Optional parameter bounds passed to
                scipy.optimize.minimize.
            options (dict or None): Options passed to
                scipy.optimize.minimize.
            L2_penalties (dict or None): Optional regularisation strengths,
                such as {"lambda_beta": ..., "lambda_sigma": ...,
                "lambda_tau": ...}.

        Returns:
            MeIntReg: The fitted model instance, with the optimization result
            available as `result`.
        """

        self.L2_penalties = dict(L2_penalties or {})

        if initial_params is None:
            initial_params = self._initial_params()

        initial_params = np.asarray(initial_params, dtype=float)

        self._value_and_grad = jax.jit(jax.value_and_grad(self._neg_loglik))
        self._hessian_fn = None
        self._value_and_grad(jnp.asarray(initial_params, dtype=jnp.float64))

        value0, grad0 = self._objective_with_grad(initial_params)

        print("INITIAL OBJECTIVE:", value0)
        print("INITIAL OBJECTIVE FINITE:", np.isfinite(value0))
        print("INITIAL GRADIENT FINITE:", np.isfinite(grad0).all())
        print("INITIAL MAX |GRAD|:", np.max(np.abs(grad0)))

        self.result = minimize(
            self._objective_with_grad,
            initial_params,
            method=method,
            jac=True,
            bounds=bounds,
            options=options,
        )

        return self
