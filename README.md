[![codecov](https://codecov.io/gh/fowler-lab/intreg/branch/main/graph/badge.svg?token=EECMnKifvc)](https://codecov.io/gh/fowler-lab/intreg)
[![DOI](https://zenodo.org/badge/856350662.svg)](https://doi.org/10.5281/zenodo.14889276)

# intreg

**intreg** is a lightweight Python package for fitting **interval regression models** to censored and interval-bounded data using **maximum likelihood estimation**.

It provides:

- Interval regression (`IntReg`) for left-, right-, exact-, and interval-censored data
- Mixed-effects interval regression (`MeIntReg`) using random intercepts
- Optional L2 regularisation for fixed and random effects
- A clean NumPy/SciPy-based API suitable for modelling, research, and simulation

The package is useful for modelling censored continuous outcomes such as MIC values, detection limits, truncated biological measurements, and any interval-coded response.

---

## 📦 Installation

### Install from PyPI

```bash
pip install intreg
```

### Install from source (developer mode)

```bash
pip install -e .
```

---

## 🧩 Package Structure

The package exposes two core classes:

| Class          | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| **`IntReg`**   | Interval regression without random effects. Handles all types of censoring. |
| **`MeIntReg`** | Mixed-effects interval regression with random intercepts                    |

Both models:

- use a Gaussian likelihood
- parameterise variance via `log(sigma)` for stability
- accept optional `L2_penalties={}` to regularise parameters
- optimise using `scipy.optimize.minimize`

Project layout (under `src/`):

```
intreg/
    __init__.py
    intreg.py
    meintreg.py
```

---

## 🔧 Basic Usage

### 1. Interval Regression (`IntReg`)

```python
import numpy as np
from intreg import IntReg

y_lower = np.array([1.0, 2.0, -np.inf, 3.0])
y_upper = np.array([1.0, 2.5, 1.5, np.inf])

model = IntReg(y_lower, y_upper)
model.fit()

print(model.result.x)  # estimated [mu, log_sigma]
```

### 2. Mixed-Effects Interval Regression (`MeIntReg`)

```python
import numpy as np
from intreg import MeIntReg

y_low = np.array([1.0, 2.0, 1.5, -np.inf])
y_high = np.array([1.0, 2.2, np.inf, 1.0])

X = np.column_stack([np.ones(4)])  # intercept-only model
groups = np.array([0, 0, 1, 1])

model = MeIntReg(y_low, y_high, X, random_effects=groups)
model.fit()

print(model.result.x)  # [beta, random_effects, log_sigma]
```

---

## 🧪 Testing

Tests are stored under:

```
temp/tests/
```

Run them using:

```bash
pytest
```

---

## 📄 License

MIT License.
