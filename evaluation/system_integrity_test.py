import time
import sys
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

# --- RESEARCH MODELS ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# --- PATHING ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gesture_system.dataset.data_manager import DataManager
    from gesture_system.config import TRAIN_FILE
except (ImportError, ModuleNotFoundError):
    TRAIN_FILE = os.path.join("gesture_system", "dataset", "gesture_train_refined.csv")
    class DataManager:
        def __init__(self, f): self.f = f
        def load_data(self): 
            df = pd.read_csv(self.f)
            return df.iloc[:, 1:].values, df.iloc[:, 0].values

def run_thesis_benchmark():
    dm = DataManager(TRAIN_FILE)
    X, y = dm.load_data()
    unique_labels = sorted(np.unique(y))
    print(f"📊 Analyzing {len(X):,} Samples from {TRAIN_FILE}...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model_factories = {
        "KNN (k=9)": lambda: KNeighborsClassifier(n_neighbors=9),
        "SVM (RBF)": lambda: SVC(kernel='rbf', class_weight='balanced', probability=True),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Logistic Reg": lambda: LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Gradient Boost": lambda: GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    final_results = []

    for name, factory in model_factories.items():
        print(f"⚡ Testing {name}...")
        metrics = {"acc": [], "f1": [], "inf_lat": [], "train_lat": []}
        
        # New: Tracking per-class recall across folds
        per_class_recall = {label: [] for label in unique_labels}

        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            model = factory()

            # Measure Training
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            metrics["train_lat"].append(time.perf_counter() - t0)

            # Measure Inference
            i0 = time.perf_counter()
            y_pred = model.predict(X_test)
            metrics["inf_lat"].append(((time.perf_counter() - i0) / len(X_test)) * 1000)

            metrics["acc"].append(accuracy_score(y_test, y_pred))
            metrics["f1"].append(f1_score(y_test, y_pred, average='weighted'))
            
            # Extract per-class recall for this specific fold
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            for label in unique_labels:
                per_class_recall[label].append(report[str(label)]['recall'])

        # Print per-class summary (Mean ± Std)
        print(f"\n📊 {name} - Per-Class Recall (Generalization Metric):")
        class_table = []
        for label in unique_labels:
            mean_rec = np.mean(per_class_recall[label])
            std_rec = np.std(per_class_recall[label])
            class_table.append([label, f"{mean_rec:.4f} ± {std_rec:.4f}"])
        print(tabulate(class_table, headers=["Gesture", "Recall"], tablefmt="simple"))

        mean_inf = np.mean(metrics['inf_lat'])
        final_results.append({
            "Model": name,
            "Accuracy": f"{np.mean(metrics['acc']):.2%} ± {np.std(metrics['acc']):.2%}",
            "Train Time": f"{np.mean(metrics['train_lat']):.4f}s",
            "Inf. Latency": f"{mean_inf:.3f}ms",
            "Stability (F1)": f"{np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}",
            "Max FPS": f"{int(1000/mean_inf) if mean_inf > 0 else 0}"
        })

    print("\n" + "="*115)
    print("    FINAL BENCHMARK: COMPUTATIONAL EFFICIENCY VS. PER-CLASS STABILITY")
    print("="*115)
    print(tabulate(final_results, headers="keys", tablefmt="fancy_grid"))

if __name__ == "__main__":
    run_thesis_benchmark()
