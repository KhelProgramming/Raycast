import time
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Sklearn
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# --- SMART IMPORT FIX ---
# This allows the script to run even if you are inside the folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try importing as a package (Best for main.py)
    from gesture_system.config import TRAIN_FILE
    from gesture_system.dataset.data_manager import DataManager
except ModuleNotFoundError:
    # Fallback: Import locally (Best for running this script directly)
    from config import TRAIN_FILE
    from dataset.data_manager import DataManager

def perform_analysis():
    print("🚀 INITIALIZING FULL RESEARCH SUITE...")
    
    # --- 1. DATA LOADING & PREPROCESSING ---
    dm = DataManager(TRAIN_FILE)
    X, y = dm.load_data()
    if X is None: 
        print("❌ No data found. Aborting.")
        return

    # --- FIX: ENCODE LABELS TO NUMBERS ---
    # This fixes the "ValueError: invalid literal for int() with base 10: 'Hold'"
    le = LabelEncoder()
    y = le.fit_transform(y) 
    print(f"✅ Labels successfully encoded: {list(le.classes_)}")

    # Scale data (Critical for SVM/KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split: 80% Training/Tuning, 20% Final Validation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- 2. DEFINE BATTLE CONFIGURATIONS ---
    battle_configs = [
        {
            "name": "KNN",
            "model": KNeighborsClassifier(),
            "params": {
                'n_neighbors': [3, 5, 9, 15, 21], 
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        {
            "name": "SVM (RBF)",
            "model": SVC(probability=True),
            "params": {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf'] 
            }
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'criterion': ['gini', 'entropy']
            }
        }
    ]

    results_data = []
    best_overall_model = None
    best_overall_f1 = 0
    best_model_name = ""

    # --- 3. EXECUTE BATTLE ROYALE ---
    for config in battle_configs:
        print(f"\n⚡ Tuning {config['name']}... (This may take a moment)")
        
        # Grid Search with 5-Fold CV
        grid = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        
        # --- LATENCY TEST (Avg of 1000 iterations for precision) ---
        start_time = time.time()
        for _ in range(1000):
            best_model.predict([X_test[0]]) # Predict single sample
        latency_ms = ((time.time() - start_time) / 1000) * 1000

        # --- ACCURACY TEST ---
        y_pred = best_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')

        # Check if this is the new King
        if test_f1 > best_overall_f1:
            best_overall_f1 = test_f1
            best_overall_model = best_model
            best_model_name = config['name']

        # Save data for the table/graphs
        results_data.append({
            "Algorithm": config['name'],
            "5-Fold CV Score": grid.best_score_,
            "Test Accuracy": test_acc,
            "Test F1-Score": test_f1,
            "Latency (ms)": latency_ms,
            "Best Params": str(grid.best_params_)
        })

    # --- 4. REPORTING ---
    df_results = pd.DataFrame(results_data)
    
    print("\n" + "="*80)
    print("🏆 FINAL RESEARCH DATA 🏆")
    print("="*80)
    print(tabulate(df_results, headers="keys", tablefmt="grid", showindex=False))
    
    print(f"\n🥇 The Winner is: {best_model_name} with F1-Score: {best_overall_f1:.4f}")
    
    # --- 5. VISUALIZATION GENERATION ---
    print("\n🎨 Generating Research Plots...")
    sns.set_style("whitegrid")

    # Plot 1: Accuracy/F1 Comparison
    plt.figure(figsize=(10, 6))
    melted_df = df_results.melt(id_vars="Algorithm", value_vars=["Test Accuracy", "Test F1-Score"], var_name="Metric", value_name="Score")
    sns.barplot(data=melted_df, x="Algorithm", y="Score", hue="Metric", palette="viridis")
    plt.ylim(0.8, 1.0) # Zoom in on the top percentages
    plt.title("Algorithm Accuracy Comparison")
    plt.ylabel("Score (0.0 - 1.0)")
    plt.savefig("benchmark_accuracy.png")
    print("✅ Saved 'benchmark_accuracy.png'")

    # Plot 2: Latency Comparison (Lower is Better)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_results, x="Algorithm", y="Latency (ms)", palette="magma")
    plt.title("Inference Latency (Speed Test)")
    plt.ylabel("Time per Prediction (ms)")
    for i, v in enumerate(df_results["Latency (ms)"]):
        plt.text(i, v + 0.05, f"{v:.2f}ms", ha='center', va='bottom', fontweight='bold')
    plt.savefig("benchmark_latency.png")
    print("✅ Saved 'benchmark_latency.png'")

    # Plot 3: Confusion Matrix for the WINNER
    plt.figure(figsize=(10, 8))
    y_pred_best = best_overall_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    # Convert numbers back to names for the graph
    target_names = le.inverse_transform(best_overall_model.classes_)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix: {best_model_name} (Best Model)")
    plt.ylabel("Actual Gesture")
    plt.xlabel("Predicted Gesture")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("best_model_confusion_matrix.png")
    print("✅ Saved 'best_model_confusion_matrix.png'")

    print("\n✨ ALL TESTS COMPLETE. Check your folder for the images!")

if __name__ == "__main__":
    perform_analysis()
