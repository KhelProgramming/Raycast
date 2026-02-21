import sys, os, time
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- RESEARCH MODELS ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# --- 1. FILE PATH SLOTS ---
DATASETS = {
    "Khel": "gesture_system/dataset/samples/gesture_train_refined.csv",
    "Don": "gesture_system/dataset/samples/gesture_data_preprocessed_don_refined.csv",
    "Bryan": "gesture_system/dataset/samples/gesture_data_preprocessed_bryan_refined.csv",
    "Louise": "gesture_system/dataset/samples/gesture_data_preprocessed_louise2_refined.csv"
}

def run_elite_rotation_audit():
    print("\n" + "="*110)
    print("🔬 ELITE LEAVE-ONE-USER-OUT (LOUO) BENCHMARK: INTER-SUBJECT RELIABILITY")
    print("="*110)
    
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"❌ Missing file for {name}: {path}")
            return

    all_fold_metrics = []
    names = list(DATASETS.keys())
    
    # Track global accuracies for statistical outlier detection
    global_accuracies = []

    for i, test_user in enumerate(names):
        train_users = [n for n in names if n != test_user]
        
        # Load and Merge Training Data
        train_dfs = [pd.read_csv(DATASETS[u]) for u in train_users]
        train_df = pd.concat(train_dfs).reset_index(drop=True)
        test_df = pd.read_csv(DATASETS[test_user])

        # [cite_start]Features/Labels (Label is Col 0) [cite: 5, 11]
        X_train, y_train = train_df.iloc[:, 1:].values, train_df.iloc[:, 0].values
        X_test, y_test = test_df.iloc[:, 1:].values, test_df.iloc[:, 0].values
        labels = np.unique(y_test)

        # Scaling: Fit ONLY on Train
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # ✅ RESEARCH CONFIGURATION
        models = {
            "SVM (Balanced)": SVC(C=1, kernel='rbf', class_weight='balanced'),
            "RF (Balanced)": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "KNN (k=3)": KNeighborsClassifier(n_neighbors=3, weights='uniform')
        }

        print(f"\n🌀 ROUND {i+1}: Training on {', '.join(train_users)} | Testing on '{test_user}'")

        for model_name, model in models.items():
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            global_accuracies.append(acc)

            # 👉 PER-CLASS RECALL: The "Golden Insight"
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            recalls = {str(lbl): f"{report[str(lbl)]['recall']:.2f}" for lbl in labels}

            all_fold_metrics.append({
                "Subject": test_user,
                "Model": model_name,
                "Acc": acc,
                "F1": f1,
                "Recalls": recalls
            })
            print(f"   ✅ {model_name} Acc: {acc:.2%} | F1: {f1:.3f}")

    # 📊 STATISTICAL OUTLIER ANALYSIS
    mean_acc = np.mean(global_accuracies)
    std_acc = np.std(global_accuracies)
    threshold = mean_acc - (2 * std_acc)

    final_table = []
    for m in all_fold_metrics:
        # Determine Status based on statistical variance, not random 85%
        status = "✅ STABLE" if m['Acc'] >= threshold else "⚠️ DOMAIN SHIFT"
        final_table.append([
            m['Subject'], m['Model'], f"{m['Acc']:.2%}", f"{m['F1']:.3f}", status, m['Recalls']
        ])

    print("\n" + "="*140)
    print(f"📊 FINAL INTER-SUBJECT REPORT (Mean Acc: {mean_acc:.2%} | Lower Bound: {threshold:.2%})")
    print("="*140)
    headers = ["Test Subject", "Model", "Accuracy", "F1", "Status", "Per-Class Recall"]
    print(tabulate(final_table, headers=headers, tablefmt="fancy_grid"))

if __name__ == "__main__":
    run_elite_rotation_audit()
