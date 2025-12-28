from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from gesture_system.dataset.data_manager import DataManager

class Trainer:
    def __init__(self, data_path):
        self.data_manager = DataManager(data_path)
        self.model = None
        self.scaler = None

    def train(self):
        """Loads data and fits the KNN model."""
        X, y = self.data_manager.load_data()
        
        if X is None or len(X) == 0:
            print("⚠️ No training data found.")
            return None, None

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = KNeighborsClassifier(n_neighbors=15)
        self.model.fit(X_scaled, y)
        
        print(f"✅ Model Trained on {len(X)} samples")
        return self.model, self.scaler