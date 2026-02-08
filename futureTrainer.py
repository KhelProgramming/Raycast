from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from gesture_system.dataset.data_manager import DataManager

class Trainer:
    def __init__(self, data_path):
        self.data_manager = DataManager(data_path)
        self.model = None
        self.scaler = None

    def train(self):
        """Loads data and fits the BEST MODEL (SVM-RBF)."""
        X, y = self.data_manager.load_data()
        
        if X is None or len(X) == 0:
            print("⚠️ No training data found.")
            return None, None

        # 1. Scale the Data (Crucial for SVM)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Train with the "Battle Royale" Winner Settings
        print("🧠 Training Optimized SVM (RBF) Model...")
        self.model = SVC(
            C=100,           # High penalty for errors (Strict)
            gamma=0.1,       # localized influence
            kernel='rbf',    # Captures complex non-linear relationships
            probability=True # Allows us to get confidence scores if needed
        )
        self.model.fit(X_scaled, y)
        
        print(f"✅ Model Trained on {len(X)} samples. Latency expected: ~0.2ms")
        return self.model, self.scaler
