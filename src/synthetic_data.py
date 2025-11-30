# src/synthetic_data.py
import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=2000):
    """
    Generate synthetic correlated dataset as in Strobl et al.
    """
    # Covariance matrix for 12 variables
    cov = np.eye(12)
    cov[:4, :4] = 0.9  # block-correlated variables
    np.fill_diagonal(cov[:4, :4], 1)

    X = np.random.multivariate_normal(mean=np.zeros(12), cov=cov, size=n_samples)

    # True coefficients
    beta = np.array([5, 5, 2, 0, -5, -5, -2, 0, 0, 0, 0, 0])

    noise = np.random.normal(0, 0.5, size=n_samples)
    y = X @ beta + noise

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(12)])
    df["y"] = y

    return df
