import os
import sys
import numpy as np
import pandas as pd

# --- SMART IMPORT FIX ---
# Allows running from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gesture_system.config import TRAIN_FILE
    from gesture_system.dataset.data_manager import DataManager
except ModuleNotFoundError:
    from config import TRAIN_FILE
    from dataset.data_manager import DataManager

def prune_dataset(threshold=0.05):
    """
    threshold (float): How different a frame needs to be to be kept.
                       0.05 is a good balance for normalized landmarks.
    """
    print("🧹 STARTING DATA PRUNING OPERATION...")
    
    # 1. Load Data
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ Error: Could not find {TRAIN_FILE}")
        return

    df = pd.read_csv(TRAIN_FILE)
    original_count = len(df)
    print(f"📦 Original Dataset Size: {original_count} samples")

    # 2. Setup output list
    pruned_data = []
    
    # 3. Process each gesture label separately
    # (We don't want to compare a 'Left_Click' to a 'Right_Click')
    for label, group in df.groupby("label"):
        print(f"   ...Processing '{label}' (Count: {len(group)})")
        
        # Convert features to numpy array (excluding label column)
        features = group.drop("label", axis=1).values
        
        # Always keep the first frame of a gesture
        kept_indices = [0] 
        last_kept_features = features[0]
        
        # Iterate through the rest
        for i in range(1, len(features)):
            current_features = features[i]
            
            # Calculate Euclidean Distance between this frame and the LAST KEPT frame
            # (Dist = sqrt((x2-x1)^2 + (y2-y1)^2 + ...))
            distance = np.linalg.norm(current_features - last_kept_features)
            
            # If distance is BIGGER than threshold, it's a new unique movement. Keep it.
            if distance > threshold:
                kept_indices.append(i)
                last_kept_features = current_features
        
        # Add the kept rows to our master list
        pruned_group = group.iloc[kept_indices]
        pruned_data.append(pruned_group)
        print(f"      -> Kept {len(pruned_group)} / {len(group)} ({len(group)-len(pruned_group)} dropped)")

    # 4. Combine and Save
    final_df = pd.concat(pruned_data)
    
    # Create new filename
    directory = os.path.dirname(TRAIN_FILE)
    filename = os.path.basename(TRAIN_FILE)
    new_filename = filename.replace(".csv", "_pruned.csv")
    output_path = os.path.join(directory, new_filename)
    
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("✨ PRUNING COMPLETE ✨")
    print("="*50)
    print(f"📉 Reduced from {original_count} to {len(final_df)} samples.")
    print(f"💾 Saved clean version to: {output_path}")
    print(f"🗑️  Removed {original_count - len(final_df)} duplicate frames.")
    print("="*50)
    print("👉 NOTE: Go to 'config.py' and change TRAIN_FILE to this new file to use it!")

if __name__ == "__main__":
    prune_dataset(threshold=0.05) # Adjustable: Higher = Stricter Pruning
