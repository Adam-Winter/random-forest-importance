# src/california_housing_analysis.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import permutation_importance

from src.utils import train_rf, plot_importances
from src.conditional_importance import conditional_permutation_importance

def run_california_housing_analysis():
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    feature_names = X.columns.tolist()

    model = train_rf(X, y, n_estimators=50)

    # Gini importance
    gini_importances = model.feature_importances_

    # Permutation importance
    perm = permutation_importance(
        model, X, y, n_repeats=5, random_state=42
    )
    perm_importances = perm.importances_mean

    # Conditional permutation importance
    cond_importances = []
    for i in range(X.shape[1]):
        cond_importances.append(
            conditional_permutation_importance(model, X, y, i, n_repeats=3)
        )

    plot_importances(feature_names, gini_importances, "California Housing: Gini Importance", "plots/ca_gini.png")

    plot_importances(feature_names, perm_importances, "California Housing: Permutation Importance", "plots/ca_perm.png")

    plot_importances(feature_names, cond_importances, "California Housing: Conditional Importance", "plots/ca_cond.png")

    print("California Housing analysis complete.")
