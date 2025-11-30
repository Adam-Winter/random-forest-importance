# src/conditional_importance.py
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

def conditional_permutation_importance(model, X, y, feature_index, n_repeats=10):
    """
    Approximate conditional permutation importance:
    - group samples by RF leaf index → preserves some correlations
    - permute inside each leaf → preserves relationship with correlated vars
    """
    X = X.copy()
    baseline_pred = model.predict(X)
    baseline_mse = mean_squared_error(y, baseline_pred)

    # Get leaf assignments from all trees
    leaves = model.apply(X)

    importances = []

    for _ in range(n_repeats):
        X_permuted = X.copy()

        # permute within each leaf group to preserve correlations
        for tree_id in range(leaves.shape[1]):
            leaf_ids = leaves[:, tree_id]
            for leaf in np.unique(leaf_ids):
                mask = (leaf_ids == leaf)
                if mask.sum() > 1:
                    X_permuted.loc[mask, feature_index] = (
                        np.random.permutation(X.loc[mask, feature_index].values)
                    )

        perm_pred = model.predict(X_permuted)
        perm_mse = mean_squared_error(y, perm_pred)

        importances.append(perm_mse - baseline_mse)

    return np.mean(importances)
