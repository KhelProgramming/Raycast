import os
import sys
import pandas as pd

# --- SMART IMPORT FIX ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gesture_system.config import TRAIN_FILE
except (ImportError, ModuleNotFoundError):
    TRAIN_FILE = os.path.join("gesture_system", "dataset", "gestures.csv")

def refine_dataset_for_realism():
    """
    Final refinement based on production-grade principles:
    1. Removes exact duplicates to stop static-bias.
    2. Uses Temporal Spread for downsampling (no clustering).
    3. Preserves motion transitions for real-time smoothness.
    """
    print("🚀 REFINING DATASET FOR REAL-TIME REALISM...")
    
    if not os.path.exists(TRAIN_FILE):
        print(f"❌ Error: {TRAIN_FILE} not found!")
        return

    df = pd.read_csv(TRAIN_FILE)
    original_count = len(df)
    
    # ✅ STEP 1: REMOVE ONLY EXACT DUPLICATES
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_count = len(df)
    
    print(f"📊 Original Size: {original_count}")
    print(f"📊 De-duplicated: {dedup_count} (Removed {original_count - dedup_count})")
    
    # ✅ STEP 2: CLASS DISTRIBUTION AUDIT
    counts = df['label'].value_counts()
    gesture_counts = counts.drop('idle', errors='ignore')
    
    if not gesture_counts.empty:
        max_gesture_size = gesture_counts.max()
        idle_size = counts.get('idle', 0)
        
        # ✅ STEP 3: TEMPORAL SPREAD DOWNSAMPLING
        # Only trigger if idle is extreme (> 4x largest gesture)
        if idle_size > (4 * max_gesture_size):
            target_idle = 2 * max_gesture_size
            print(f"⚠️ Idle dominance detected ({idle_size} samples).")
            print(f"⚖️ Applying Temporal Spread: Keeping {target_idle} samples across the timeline...")
            
            idle_subset = df[df.label == 'idle']
            # Logic: Slice the dataframe to get even spacing
            step = max(1, idle_size // target_idle)
            idle_group = idle_subset.iloc[::step].head(target_idle)
            
            action_groups = df[df.label != 'idle']
            df = pd.concat([idle_group, action_groups]).sample(frac=1).reset_index(drop=True)

    output_path = TRAIN_FILE.replace(".csv", "_refined.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ REFINED DATASET SAVED: {output_path}")
    print("👉 CRITICAL: Your models MUST use 'class_weight=balanced' now!")

if __name__ == "__main__":
    refine_dataset_for_realism()
