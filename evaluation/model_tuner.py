import sys, os, time
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# --- PATH FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_pro_level_tuning():
    print(f"🔎 TUNING THE BIG FIVE WITH OVERFITTING ANALYSIS...")
    TRAIN_FILE = "gesture_train_pruned.csv"
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ Error: {TRAIN_FILE} not found!")
        return

    df = pd.read_csv(TRAIN_FILE)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    tasks = [
        {"name": "KNN", "model": KNeighborsClassifier(), 
         "params": {'model__n_neighbors': [3, 7, 9, 15], 'model__weights': ['uniform', 'distance']}},
        {"name": "SVM", "model": SVC(), 
         "params": {'model__C': [0.1, 1, 10], 'model__kernel': ['rbf']}},
        {"name": "Random Forest", "model": RandomForestClassifier(random_state=42), 
         "params": {'model__n_estimators': [50, 100, 200], 'model__max_depth': [None, 10, 20]}},
        {"name": "Logistic Reg", "model": LogisticRegression(max_iter=2000, multi_class='multinomial'), 
         "params": {'model__C': [0.1, 1.0, 10.0], 'model__solver': ['lbfgs']}},
        {"name": "Gradient Boost", "model": GradientBoostingClassifier(random_state=42), 
         "params": {'model__n_estimators': [50, 100], 'model__learning_rate': [0.1], 'model__max_depth': [3, 5]}}
    ]

    best_configs = []
    for task in tasks:
        print(f"🌀 Tuning {task['name']}...")
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', task['model'])])
        
        # KEY UPGRADE: return_train_score=True
        grid = GridSearchCV(pipeline, task['params'], cv=StratifiedKFold(n_splits=5), 
                            scoring='f1_weighted', n_jobs=-1, return_train_score=True)
        
        start_t = time.time()
        grid.fit(X, y)
        
        # Calculate Overfitting Gap
        train_f1 = grid.cv_results_['mean_train_score'][grid.best_index_]
        val_f1 = grid.best_score_
        overfit_gap = train_f1 - val_f1

        best_configs.append({
            "Model": task['name'],
            "Train F1": f"{train_f1:.4f}",
            "Best F1 (CV)": f"{val_f1:.4f}",
            "Overfit Gap": f"{overfit_gap:.4f}", # The "Panel Pleaser" column
            "Optimal Parameters": str(grid.best_params_).replace("model__", ""),
            "Tuning Time": f"{time.time() - start_t:.2f}s"
        })

    print("\n" + "="*140)
    print("🏆 RESEARCH CONFIGURATIONS & GENERALIZATION ANALYSIS")
    print("="*140)
    print(tabulate(best_configs, headers="keys", tablefmt="fancy_grid"))

if __name__ == "__main__":
    run_pro_level_tuning()
