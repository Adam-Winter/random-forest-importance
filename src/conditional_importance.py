# src/conditional_importance.py
import numpy as np
from sklearn.metrics import mean_squared_error

def conditional_permutation_importance(model, X, y, feature_index, n_repeats=10):
    """
    Approximate conditional permutation importance using leaf-based grouping.
    Works with both pandas DataFrames and numpy arrays.
    """
    X = X.copy()
    baseline_pred = model.predict(X)
    baseline_mse = mean_squared_error(y, baseline_pred)

    # Leaf assignments: (n_samples × n_trees)
    leaves = model.apply(X)

    importances = []

    for _ in range(n_repeats):
        X_permuted = X.copy()

        # Loop over trees
        for tree_id in range(leaves.shape[1]):
            leaf_ids = leaves[:, tree_id]

            # Loop over each leaf
            for leaf in np.unique(leaf_ids):
                mask = (leaf_ids == leaf)  # boolean mask (numpy array)

                if mask.sum() > 1:  # must have enough samples to permute
                    # Extract values using boolean indexing
                    values = X_permuted.iloc[mask, feature_index].values
                    permuted_values = np.random.permutation(values)

                    # Assign back
                    X_permuted.iloc[mask, feature_index] = permuted_values

        # Compute new prediction error
        perm_pred = model.predict(X_permuted)
        perm_mse = mean_squared_error(y, perm_pred)

        importances.append(perm_mse - baseline_mse)

    return np.mean(importances)
