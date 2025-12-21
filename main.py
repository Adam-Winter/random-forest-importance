# main.py
from src.synthetic_data import generate_synthetic_data
from src.utils import train_rf, plot_importances
from src.conditional_importance import conditional_permutation_importance
from src.wine_quality_analysis import run_wine_analysis
from src.california_housing_analysis import run_california_housing_analysis

import pandas as pd
from sklearn.inspection import permutation_importance

def run_synthetic():
    df = generate_synthetic_data()
    X = df.drop(columns="y")
    y = df["y"]
    model = train_rf(X, y)

    gini = model.feature_importances_
    perm = permutation_importance(model, X, y).importances_mean

    cond = []
    for i in range(X.shape[1]):
        cond.append(
            conditional_permutation_importance(model, X, y, i)
        )

    plot_importances(X.columns, gini, "Synthetic: Gini", "plots/synth_gini.png")
    plot_importances(X.columns, perm, "Synthetic: Permutation", "plots/synth_perm.png")
    plot_importances(X.columns, cond, "Synthetic: Conditional", "plots/synth_cond.png")

def main():
    print("Running synthetic dataset analysis...")
    run_synthetic()

    print("\nRunning Wine Quality analysis...")
    run_wine_analysis()

    print("\nRunning California Housing analysis...")
    run_california_housing_analysis()

if __name__ == "__main__":
    main()
