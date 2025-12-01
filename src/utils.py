# src/utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def train_rf(X, y, n_estimators=50, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def plot_importances(names, values, title, save_path=None):
    idx = np.argsort(values)
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(names)[idx], np.array(values)[idx])
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
