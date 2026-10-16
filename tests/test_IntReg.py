import importlib
import importlib.metadata
import pytest
import numpy as np
import intreg
from intreg import IntReg
from scipy.optimize import OptimizeResult
from scipy.stats import norm
from importlib.metadata import PackageNotFoundError


@pytest.fixture
def simple_point_data():
    """Fixture for simple point data (uncensored)."""
    y_lower = np.array([1, 1, 2])
    y_upper = np.array([1, 1, 2])
    return y_lower, y_upper


@pytest.fixture
def left_censored_data():
    """Fixture for left-censored data."""
    y_lower = np.array([-np.inf, 0, 1])
    y_upper = np.array([1, 1, 2])
    return y_lower, y_upper


@pytest.fixture
def right_censored_data():
    """Fixture for right-censored data."""
    y_lower = np.array([1, 1, 2,])
    y_upper = np.array([np.inf, 2, 3])
    return y_lower, y_upper


@pytest.fixture
def mixed_interval_and_point_data():
    """Fixture for mixed interval and point data."""
    y_lower = np.array([1, 2, 3])
    y_upper = np.array([1, 3, 4])
    return y_lower, y_upper


@pytest.fixture
def clear_interval_censored_data():
    """Fixture for clear interval-censored data with obvi ous midpoints."""
    y_lower = np.array([1, 2, 3, 4])
    y_upper = np.array([2, 3, 4, 5])
    return y_lower, y_upper


def test_initial_params(
    simple_point_data,
    left_censored_data,
    right_censored_data,
    clear_interval_censored_data,
):
    """Test _initial_params method with all types of mock data (point, left-censored, right-censored, and interval-censored)."""

    # Define test cases as tuples: (data, expected_mu, expected_log_sigma_func)
    # We define expected values based on the interpretation of each dataset.
    test_cases = [
        (
            simple_point_data,
            lambda midpoints: np.nanmean(midpoints),
            lambda midpoints: np.log(np.nanstd(midpoints)),
        ),
        (
            left_censored_data,
            lambda midpoints: np.nanmean(midpoints),
            lambda midpoints: np.log(np.nanstd(midpoints)),
        ),
        (
            right_censored_data,
            lambda midpoints: np.nanmean(midpoints),
            lambda midpoints: np.log(np.nanstd(midpoints)),
        ),
        (
            clear_interval_censored_data,
            lambda midpoints: np.nanmean(midpoints),
            lambda midpoints: np.log(np.nanstd(midpoints)),
        ),
    ]

    for (y_lower, y_upper), expected_mu_func, expected_log_sigma_func in test_cases:
        model = IntReg(y_lower, y_upper)

        # Calculate midpoints for uncensored data
        midpoints = (y_lower + y_upper) / 2.0
        midpoints = np.where(np.isfinite(midpoints), midpoints, np.nan)

        # Calculate expected values
        expected_mu = expected_mu_func(midpoints)
        expected_log_sigma = expected_log_sigma_func(midpoints)

        # Call _initial_params and get the output
        initial_mu, initial_log_sigma = model._initial_params()

        # Assertions to check that _initial_params matches expected mu and log(sigma)
        assert np.isclose(
            initial_mu, expected_mu, atol=1e-4
        ), f"Failed for data {y_lower, y_upper}: expected mu {expected_mu}, got {initial_mu}"
        assert np.isclose(
            initial_log_sigma, expected_log_sigma, atol=1e-4
        ), f"Failed for data {y_lower, y_upper}: expected log(sigma) {expected_log_sigma}, got {initial_log_sigma}"


def test_initial_params_handles_zero_spread_and_no_finite_midpoints():
    """Test initial values remain finite for degenerate and censored data."""
    same_points = np.array([2.0, 2.0, 2.0])
    model = IntReg(same_points, same_points)
    initial_params = model._initial_params()
    assert np.all(np.isfinite(initial_params))
    assert np.isclose(initial_params[0], 2.0)
    assert np.isclose(initial_params[1], np.log(1e-6))

    model = IntReg([-np.inf, -np.inf], [0.0, 1.0])
    initial_params = model._initial_params()
    assert np.all(np.isfinite(initial_params))
    assert np.allclose(initial_params, [0.0, 0.0])


def test_input_arrays_must_have_matching_lengths():
    """Test bounds and weights are checked against the observation count."""
    with pytest.raises(ValueError, match="one-dimensional"):
        IntReg([[1.0, 2.0]], [[1.0, 2.0]])

    with pytest.raises(ValueError, match="same length"):
        IntReg([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="At least one observation"):
        IntReg([], [])

    with pytest.raises(ValueError, match="weights"):
        IntReg([1.0, 2.0], [1.0, 2.0], weights=[1.0])


def test_package_version_fallback_when_metadata_is_missing(monkeypatch):
    """Test package version defaults when distribution metadata is unavailable."""
    def missing_version(_):
        raise PackageNotFoundError("intreg")

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", missing_version)
        importlib.reload(intreg)
        assert intreg.__version__ == "0.0.0"

    importlib.reload(intreg)


def test_log_L(
    simple_point_data,
    left_censored_data,
    right_censored_data,
    clear_interval_censored_data,
):
    """Test log_L using parameters generated by _initial_params on all data types."""

    # Define test cases as tuples: (data, description)
    test_cases = [
        (simple_point_data, "point data"),
        (right_censored_data, "right-censored data"),
        (left_censored_data, "left-censored data"),
        (clear_interval_censored_data, "interval-censored data"),
    ]

    for (y_lower, y_upper), description in test_cases:
        model = IntReg(y_lower, y_upper)

        params = model._initial_params()
        print ('params', params)
        log_likelihood = model.log_L(params)

        assert isinstance(
            log_likelihood, float
        ), f"log_L output should be float for {description}"
        assert np.isfinite(
            log_likelihood
        ), f"log_L output should be finite for {description}"

        print(log_likelihood)
        assert (
            log_likelihood > 0
        ), f"log_L output should be positive for {description}, got {log_likelihood}"


def test_log_L_is_finite_for_extreme_intervals():
    """Test log-space likelihood remains finite for narrow tail intervals."""
    y_lower = np.array([40.0, -np.inf, 40.0])
    y_upper = np.array([40.000001, 0.0, np.inf])
    model = IntReg(y_lower, y_upper)

    negative_log_likelihood = model.log_L(np.array([0.0, 0.0]))
    assert np.isfinite(negative_log_likelihood)


def test_interval_log_likelihood_returns_each_observation_contribution():
    """Test log-likelihood contributions for each censoring type."""
    y_lower = np.array([1.0, -np.inf, 2.0, 40.0])
    y_upper = np.array([1.0, 0.0, np.inf, 40.000001])
    contributions = IntReg.interval_log_likelihood(
        y_lower, y_upper, mu=0.0, sigma=1.0
    )

    assert contributions.shape == y_lower.shape
    assert np.all(np.isfinite(contributions))
    assert np.isclose(contributions[0], norm.logpdf(1.0))
    assert np.isclose(contributions[1], norm.logcdf(0.0))
    assert np.isclose(contributions[2], norm.logsf(2.0))


def test_interval_prob_remains_compatible_with_log_likelihood_method():
    """Test the probability-scale helper remains available for existing callers."""
    y_lower = np.array([1.0, -np.inf, 2.0])
    y_upper = np.array([1.0, 0.0, np.inf])

    probabilities = IntReg.interval_prob(y_lower, y_upper, mu=0.0, sigma=1.0)
    log_likelihood = IntReg.interval_log_likelihood(
        y_lower, y_upper, mu=0.0, sigma=1.0
    )

    assert np.allclose(probabilities, np.clip(np.exp(log_likelihood), 1e-300, 1.0))


def test_simple_point_data_fit(simple_point_data):
    y_lower, y_upper = simple_point_data
    model = IntReg(y_lower, y_upper).fit()
    result = model.result
    print (result.x[0], result.x[1])
    assert isinstance(result, OptimizeResult)
    assert np.isclose(result.x[0], 1.3, atol=1e-1)  # Expect mu near 1
    assert np.isclose(result.x[1], -0.8, atol=1e-1)  # Expect sigma near -0.75


def test_left_censored_data_fit(left_censored_data):
    y_lower, y_upper = left_censored_data
    model = IntReg(y_lower, y_upper).fit()
    result = model.result
    print (result.x[0], result.x[1])
    assert isinstance(result, OptimizeResult)
    assert np.isclose(result.x[0], 0.9, atol=1e-1)  # Expect mu near 0.9
    assert np.isclose(result.x[1], -1.8, atol=1e-1)  # Expect sigma near -1.8


def test_right_censored_data_fit(right_censored_data):
    y_lower, y_upper = right_censored_data
    model = IntReg(y_lower, y_upper).fit()
    result = model.result
    assert isinstance(result, OptimizeResult)
    assert np.isclose(result.x[0], 2, atol=1e-1)  # Expect mu near 1.5
    assert np.isclose(result.x[1], -1.9, atol=1e-1)  # Expect sigma near -1.9


def test_mixed_interval_and_point_data_fit(mixed_interval_and_point_data):
    y_lower, y_upper = mixed_interval_and_point_data
    model = IntReg(y_lower, y_upper).fit()
    result = model.result
    print(result.x[0], result.x[1])
    assert isinstance(result, OptimizeResult)
    assert np.isclose(result.x[0], 2.3, atol=1e-1)  # Expect mu around 2.5
    assert np.isclose(result.x[1], 0.02, atol=1e-1)  # Expect log(sigma) = 0.02


def test_clear_interval_censored_data_fit(clear_interval_censored_data):
    y_lower, y_upper = clear_interval_censored_data
    model = IntReg(y_lower, y_upper).fit()
    result = model.result
    print(result.x[0], result.x[1])
    assert isinstance(result, OptimizeResult)
    assert np.isclose(result.x[0], 3, atol=1e-1)  # Expect mu around 2.5
    assert np.isclose(result.x[1], 0.08, atol=1e-1)  # Expect log(sigma) = 0.08


def test_regularisation():
    y_lower = np.array([1, 2, 3])
    y_upper = np.array([1, 2, 3])
    model = IntReg(y_lower, y_upper)

    model.L2_penalties = {"lambda_beta": 2.0, "lambda_sigma": 4.0, "n_fixed": 1}
    params = np.array([2.0, -1.0])  # mu, log_sigma
    unpenalised = -5.0
    direct = model._apply_L2_regularisation(unpenalised, params)

    # expected penalties:
    # beta:  (2/3) * 4 = 8/3
    # sigma: (4/3) * 1 = 4/3
    expected = unpenalised - (8/3 + 4/3)
    assert np.isclose(direct, expected)

    # ----- verify log_L applies penalties -----
    no_penalty = IntReg(y_lower, y_upper).log_L(params)
    model.L2_penalties = {"lambda_beta": 100.0, "lambda_sigma": 0.0, "n_fixed": 1}
    with_penalty = model.log_L(params)
    assert with_penalty > no_penalty  # penalty makes negative log-likelihood larger

    # ----- verify fit() runs with penalties -----
    model = IntReg(y_lower, y_upper)
    model.fit(L2_penalties={"lambda_beta": 1.0, "lambda_sigma": 1.0, "n_fixed": 1})
    assert hasattr(model, "result")
    assert model.result.x is not None 
