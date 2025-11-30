# src/wine_quality_analysis.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import train_rf, plot_importances
from src.conditional_importance import conditional_permutation_importance

def load_wine_data(path="data/winequality-red.csv"):
    df = pd.read_csv(path, sep=";")
    return df

def run_wine_analysis():
    df = load_wine_data()
    X = df.drop(columns="quality")
    y = df["quality"]

    feature_names = X.columns.tolist()

    model = train_rf(X, y)

    # Gini importance
    gini_importances = model.feature_importances_

    # Permutation importance
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_importances = perm.importances_mean

    # Conditional importance
    cond_importances = []
    for i in range(X.shape[1]):
        imp = conditional_permutation_importance(model, X, y, feature_index=i)
        cond_importances.append(imp)

    # Plot results
    plot_importances(feature_names, gini_importances,
                     "Gini Importance", "plots/gini.png")
    plot_importances(feature_names, perm_importances,
                     "Permutation Importance", "plots/perm.png")
    plot_importances(feature_names, cond_importances,
                     "Conditional Permutation Importance", "plots/cond.png")

    print("Analysis complete.")
