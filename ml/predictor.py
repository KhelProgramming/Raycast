class Predictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, landmarks_flattened):
        """Predicts the gesture class from preprocessed landmarks."""
        if self.model is None or self.scaler is None:
            return "unknown"
            
        row_scaled = self.scaler.transform([landmarks_flattened])
        pred = self.model.predict(row_scaled)[0]
        return pred