# src/conditional_importance.py
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def conditional_permutation_importance(model, X, y, feature_index, n_repeats=10):
    """
    Approximate conditional permutation importance using leaf-based grouping.
    Works with both Pandas DataFrames and NumPy arrays.
    """
    # Detect if X is a DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    
    # Make a copy
    X = X.copy()
    
    baseline_pred = model.predict(X)
    baseline_mse = mean_squared_error(y, baseline_pred)

    # Get leaf assignments (n_samples × n_trees)
    leaves = model.apply(X)

    importances = []

    for _ in range(n_repeats):
        X_permuted = X.copy()

        for tree_id in range(leaves.shape[1]):
            leaf_ids = leaves[:, tree_id]

            for leaf in np.unique(leaf_ids):
                mask = (leaf_ids == leaf)  # boolean mask for this leaf

                if mask.sum() > 1:
                    # Extract values to permute
                    if is_dataframe:
                        values = X_permuted.iloc[mask, feature_index].to_numpy()
                    else:
                        values = X_permuted[mask, feature_index]

                    permuted = np.random.permutation(values)

                    # Assign back
                    if is_dataframe:
                        X_permuted.iloc[mask, feature_index] = permuted
                    else:
                        X_permuted[mask, feature_index] = permuted

        perm_pred = model.predict(X_permuted)
        perm_mse = mean_squared_error(y, perm_pred)
        importances.append(perm_mse - baseline_mse)

    return np.mean(importances)
