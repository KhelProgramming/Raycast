import sys
import os
import time
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# --- PATH FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from gesture_system.dataset.data_manager import DataManager
    from gesture_system.config import TRAIN_FILE
except ImportError:
    TRAIN_FILE = "gestures.csv"
    class DataManager:
        def __init__(self, f): self.f = f
        def load_data(self): return np.random.rand(4417, 63), np.random.randint(0, 5, 4417)

def run_hyperparameter_tuning():
    print(f"🔎 TUNING MODELS FOR OPTIMAL STATE (N=4,417)...")
    dm = DataManager(TRAIN_FILE)
    X, y = dm.load_data()

    # Define the "Search Space" for each model
    tasks = [
        {
            "name": "KNN",
            "model": KNeighborsClassifier(),
            "params": {
                'model__n_neighbors': [3, 5, 7, 9, 11, 15],
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['euclidean', 'manhattan']
            }
        },
        {
            "name": "SVM",
            "model": SVC(probability=True),
            "params": {
                'model__C': [0.1, 1, 10, 100],
                'model__gamma': ['scale', 'auto'],
                'model__kernel': ['rbf', 'poly']
            }
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20],
                'model__criterion': ['gini', 'entropy']
            }
        }
    ]

    best_configs = []

    for task in tasks:
        print(f"🌀 Grid-Searching {task['name']}...")
        
        # Pipeline ensures scaling happens correctly during each CV fold
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', task['model'])
        ])

        grid = GridSearchCV(
            pipeline, 
            task['params'], 
            cv=StratifiedKFold(n_splits=5), 
            scoring='f1_weighted',
            n_jobs=-1 # Use all CPU cores for speed
        )
        
        start_t = time.time()
        grid.fit(X, y)
        duration = time.time() - start_t

        best_configs.append({
            "Model": task['name'],
            "Best F1 Score": f"{grid.best_score_:.4f}",
            "Optimal Parameters": str(grid.best_params_).replace("model__", ""),
            "Tuning Time": f"{duration:.2f}s"
        })

    print("\n" + "="*100)
    print("🏆 OPTIMIZED MODEL CONFIGURATIONS")
    print("="*100)
    print(tabulate(best_configs, headers="keys", tablefmt="fancy_grid"))

if __name__ == "__main__":
    run_hyperparameter_tuning()
