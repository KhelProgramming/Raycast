import pandas as pd
import os
import sys

class DataManager:
    """
    Handles loading and saving of gesture data from the samples folder.
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads the CSV data for training."""
        if not os.path.exists(self.file_path):
            print(f"❌ Error: Data file not found at {self.file_path}")
            # Return empty structure or raise error depending on preference
            return None, None

        print(f"📂 Loading dataset from: {self.file_path}")
        try:
            data = pd.read_csv(self.file_path)
            X = data.drop("label", axis=1).values
            y = data["label"].values
            return X, y
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return None, None