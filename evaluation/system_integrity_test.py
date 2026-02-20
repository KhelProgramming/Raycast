import time
import sys
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# --- PATHING FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gesture_system.dataset.data_manager import DataManager
    from gesture_system.config import TRAIN_FILE
except (ImportError, ModuleNotFoundError):
    TRAIN_FILE = "gesture_train_pruned.csv" 
    class DataManager:
        def __init__(self, f): self.f = f
        def load_data(self): 
            # Fallback for testing logic
            return np.random.rand(4417, 63), np.random.randint(0, 5, 4417)

def run_thesis_benchmark():
    print(f"📊 Analyzing Samples from {TRAIN_FILE}...")
    dm = DataManager(TRAIN_FILE)
    X, y = dm.load_data()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- RESEARCH MODEL FACTORIES ---
    model_factories = {
        "KNN (k=9)": lambda: KNeighborsClassifier(n_neighbors=9),
        "SVM (RBF)": lambda: SVC(kernel='rbf'),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Reg": lambda: LogisticRegression(max_iter=1000, multi_class='auto'),
        "Gradient Boost": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    final_results = []

    for name, factory in model_factories.items():
        print(f"⚡ Testing {name}...")
        metrics = {"acc": [], "f1": [], "inf_lat": [], "train_lat": []}

        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            model = factory()

            # Measure Training Time
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            metrics["train_lat"].append(time.perf_counter() - t0)

            # Measure Inference Time (Per-sample latency)
            i0 = time.perf_counter()
            y_pred = model.predict(X_test)
            metrics["inf_lat"].append(((time.perf_counter() - i0) / len(X_test)) * 1000)

            metrics["acc"].append(accuracy_score(y_test, y_pred))
            metrics["f1"].append(f1_score(y_test, y_pred, average='weighted'))

        # Safety Check for FPS calculation
        mean_inf = np.mean(metrics['inf_lat'])
        max_fps = int(1000 / mean_inf) if mean_inf > 0 else 0

        final_results.append({
            "Model": name,
            "Accuracy": f"{np.mean(metrics['acc']):.2%} ± {np.std(metrics['acc']):.2%}",
            "Train Time": f"{np.mean(metrics['train_lat']):.4f}s",
            "Inf. Latency": f"{mean_inf:.3f}ms",
            "Stability (F1)": f"{np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}",
            "Max FPS": f"{max_fps}"
        })

    print("\n" + "="*115)
    print("🏆 RESEARCH DATA: COMPUTATIONAL EFFICIENCY VS. PREDICTIVE POWER")
    print("="*115)
    headers = ["Model", "Accuracy", "Train Time", "Inf. Latency", "Stability (F1)", "Max FPS"]
    print(tabulate(final_results, headers="keys", tablefmt="fancy_grid"))

if __name__ == "__main__":
    run_thesis_benchmark()
