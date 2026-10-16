import pytest
import numpy as np
import pickle
from intreg import MeIntReg
from scipy.optimize import OptimizeResult


@pytest.fixture
def simple_point_data():
    """Fixture for simple point data (uncensored) with covariates."""
    y_lower = np.array([1, 1, 2])
    y_upper = np.array([1, 1, 2])
    X = np.array([[1, 0], [1, 1], [1, 2]])  # Covariates with intercept and slope
    random_effects = np.array([0, 1, 1])
    return y_lower, y_upper, X, random_effects


@pytest.fixture
def left_censored_data():
    """Fixture for left-censored data."""
    y_lower = np.array([-np.inf, 0, 1])
    y_upper = np.array([1, 1, 2])
    X = np.array([[1, 0], [1, 1], [1, 2]])
    random_effects = np.array([0, 1, 1])
    return y_lower, y_upper, X, random_effects


@pytest.fixture
def right_censored_data():
    """Fixture for right-censored data."""
    y_lower = np.array([1, 1, 2])
    y_upper = np.array([np.inf, 2, 3])
    X = np.array([[1, 0], [1, 1], [1, 2]])
    random_effects = np.array([0, 1, 1])
    return y_lower, y_upper, X, random_effects


@pytest.fixture
def mixed_interval_and_point_data():
    """Fixture for mixed interval-censored and point data."""
    y_lower = np.array([1, 2, 3, 3])
    y_upper = np.array([1, np.inf, 4, 3])
    X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
    random_effects = np.array([0, 1, 1, 2])
    return y_lower, y_upper, X, random_effects


@pytest.fixture
def clear_interval_censored_data():
    """Fixture for interval-censored data with clear midpoints."""
    y_lower = np.array([1, 2, 3, 4])
    y_upper = np.array([2, 3, 4, 5])
    X = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
    random_effects = np.array([0, 1, 1, 2])
    return y_lower, y_upper, X, random_effects


def test_initial_params(
    simple_point_data,
    left_censored_data,
    right_censored_data,
    mixed_interval_and_point_data,
    clear_interval_censored_data,
):
    """Test automatic starting values for models with random intercepts."""

    test_cases = [
        (simple_point_data, "simple point data"),
        (left_censored_data, "left-censored data"),
        (right_censored_data, "right-censored data"),
        (mixed_interval_and_point_data, "mixed interval and point data"),
        (clear_interval_censored_data, "interval-censored data"),
    ]

    for (y_lower, y_upper, X, random_effects), description in test_cases:
        model = MeIntReg(y_lower, y_upper, X, random_effects)
        initial_params = model._initial_params()

        midpoints = (y_lower + y_upper) / 2.0
        valid_mask = np.isfinite(midpoints)
        expected_beta = np.linalg.lstsq(X[valid_mask], midpoints[valid_mask], rcond=None)[0]
        expected_log_sigma = np.log(max(np.nanstd(midpoints[valid_mask]), 1e-6))

        assert initial_params.shape == (X.shape[1] + 2,)
        assert np.allclose(
            initial_params[: X.shape[1]], expected_beta, atol=1e-4
        ), f"Failed for {description}: fixed-effect starting values are incorrect"
        assert np.isclose(
            initial_params[-2], expected_log_sigma, atol=1e-4
        ), f"Failed for {description}: log(sigma) starting value is incorrect"
        assert np.isclose(
            initial_params[-1], np.log(0.5), atol=1e-4
        ), f"Failed for {description}: log(tau) starting value is incorrect"


def test_initial_params_without_random_effects(simple_point_data):
    """Test automatic starting values when random effects are omitted."""
    y_lower, y_upper, X, _ = simple_point_data
    model = MeIntReg(y_lower, y_upper, X)
    initial_params = model._initial_params()

    assert initial_params.shape == (X.shape[1] + 1,)
    assert np.all(np.isfinite(initial_params))


def test_no_random_effects_likelihood_gradient_and_hessian():
    """Test likelihood derivatives when the model has no random effects."""
    y_lower = np.array([1.0, -np.inf, 2.0, 0.0])
    y_upper = np.array([1.0, 0.0, np.inf, 2.0])
    X = np.ones((4, 1))
    model = MeIntReg(y_lower, y_upper, X)
    likelihood_params = np.array([0.5, 0.0])

    # Exercise point, left-censored, right-censored, and interval branches.
    assert np.isfinite(model.log_L(likelihood_params))
    _, censored_gradient = model._objective_with_grad(likelihood_params)
    assert np.all(np.isfinite(censored_gradient))

    grouped_model = MeIntReg(
        y_lower, y_upper, X, random_effects=np.array([0, 1, 1, 2])
    )
    grouped_params = grouped_model._initial_params()
    _, grouped_gradient = grouped_model._objective_with_grad(grouped_params)
    assert np.all(np.isfinite(grouped_gradient))

    # Use finite point data to check derivatives independently of censoring masks.
    point_model = MeIntReg([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], np.ones((3, 1)))
    params = np.array([2.0, 0.0])
    objective, gradient = point_model._objective_with_grad(params)
    assert np.isfinite(objective)
    assert gradient.shape == params.shape
    assert np.all(np.isfinite(gradient))

    point_model.result = OptimizeResult(x=params)
    hessian = point_model.hessian()
    assert hessian.shape == (len(params), len(params))
    assert np.all(np.isfinite(hessian))


def test_hessian_requires_fit_when_params_are_omitted(simple_point_data):
    """Test Hessian reports a clear error before fitting without parameters."""
    y_lower, y_upper, X, _ = simple_point_data
    model = MeIntReg(y_lower, y_upper, X)

    with pytest.raises(ValueError, match="must be fitted"):
        model.hessian()


def test_serialization_clears_compiled_functions():
    """Test compiled JAX functions are recreated after serialization."""
    y_lower = np.array([1.0, 2.0, 3.0])
    X = np.ones((3, 1))
    model = MeIntReg(y_lower, y_lower, X)
    params = np.array([2.0, 0.0])
    model._objective_with_grad(params)
    model.hessian(params)

    restored_model = pickle.loads(pickle.dumps(model))
    assert restored_model._value_and_grad is None
    assert restored_model._hessian_fn is None
    assert np.isfinite(restored_model.log_L(params))


def test_log_L(
    simple_point_data,
    left_censored_data,
    right_censored_data,
    mixed_interval_and_point_data,
    clear_interval_censored_data,
):
    """Test the negative marginal log-likelihood for each observation type."""

    test_cases = [
        (simple_point_data, "point data"),
        (right_censored_data, "right-censored data"),
        (left_censored_data, "left-censored data"),
        (mixed_interval_and_point_data, "mixed interval and point data"),
        (clear_interval_censored_data, "interval-censored data"),
    ]

    for (y_lower, y_upper, X, random_effects), description in test_cases:
        model = MeIntReg(y_lower, y_upper, X, random_effects)
        params = model._initial_params()
        negative_log_likelihood = model.log_L(params)

        assert isinstance(
            negative_log_likelihood, float
        ), f"log_L output should be a float for {description}"
        assert np.isfinite(
            negative_log_likelihood
        ), f"log_L output should be finite for {description}"


@pytest.mark.parametrize(
    "fixture_name",
    [
        "simple_point_data",
        "left_censored_data",
        "right_censored_data",
        "mixed_interval_and_point_data",
        "clear_interval_censored_data",
    ],
)
def test_fit_returns_finite_result(request, fixture_name):
    """Test fitting returns finite parameter estimates for each data type."""
    y_lower, y_upper, X, random_effects = request.getfixturevalue(fixture_name)
    model = MeIntReg(y_lower, y_upper, X, random_effects).fit()

    assert isinstance(model.result, OptimizeResult)
    assert model.result.x.shape == (X.shape[1] + 2,)
    assert np.all(np.isfinite(model.result.x))
    assert np.isfinite(model.result.fun)


def test_meintreg_regularisation():
    """Test L2 penalties for fixed effects and both standard deviations."""
    y_lower = np.array([1.0, 1.0, 2.0, 2.0])
    y_upper = y_lower.copy()  # exact values
    X = np.ones((4, 1))  # intercept-only fixed effect
    groups = np.array([0, 0, 1, 1])  # random intercept groups
    model = MeIntReg(y_lower, y_upper, X, random_effects=groups)
    model.L2_penalties = {
        "lambda_beta": 2.0,
        "lambda_sigma": 4.0,
        "lambda_tau": 3.0,
    }

    # params = [beta, log_sigma, log_tau]
    params = np.array([1.5, -1.0, -0.5])
    unpenalised = -10.0
    expected = unpenalised - (
        (2.0 / 4) * 1.5**2
        + (4.0 / 4) * (-1.0) ** 2
        + (3.0 / 4) * (-0.5) ** 2
    )
    direct = model._apply_L2_regularisation(unpenalised, params)
    assert np.isclose(direct, expected)

    no_penalty = MeIntReg(y_lower, y_upper, X, groups).log_L(params)
    model.L2_penalties = {
        "lambda_beta": 50.0,
        "lambda_sigma": 0.0,
        "lambda_tau": 0.0,
    }
    with_penalty = model.log_L(params)
    assert with_penalty > no_penalty  # more penalty increases negative log-likelihood

    model = MeIntReg(y_lower, y_upper, X, groups)
    model.fit(
        L2_penalties={
            "lambda_beta": 1.0,
            "lambda_sigma": 1.0,
            "lambda_tau": 1.0,
        }
    )
    assert hasattr(model, "result")
    assert model.result.x is not None
