import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# All the Model Imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA_FILE = "user_profile.csv"

# --- 1. THE GEOMETRIC EXTRACTOR ---
def extract_geometric_features(landmarks):
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    palm_width = np.linalg.norm(points[5] - points[17])
    if palm_width < 1e-6: palm_width = 1.0 

    wrist = points[0]
    extensions = [np.linalg.norm(wrist - points[idx]) / palm_width for idx in [4, 8, 12, 16, 20]]
    thumb_tip = points[4]
    pinches = [np.linalg.norm(thumb_tip - points[idx]) / palm_width for idx in [8, 12, 16, 20]]
    spreads = [np.linalg.norm(points[i] - points[j]) / palm_width for i, j in [(8, 12), (12, 16), (16, 20)]]
    thumb_to_pinky_base = np.linalg.norm(points[4] - points[17]) / palm_width
    
    return extensions + pinches + spreads + [thumb_to_pinky_base]

# --- 2. THE LIVE TESTER ---
def run_transition_tester():
    if not os.path.exists(DATA_FILE):
        print("⚠️ No user_profile.csv found. Please run the calibration script first.")
        return

    print("Loading data and training all 6 models. Please wait...")
    df = pd.read_csv(DATA_FILE)
    X = df.drop('label', axis=1).values
    y = df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the champions for our Hybrid
    mlp_champ = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    rf_champ = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # Build the Hybrid Ensemble
    hybrid_model = VotingClassifier(
        estimators=[('MLP', mlp_champ), ('RF', rf_champ)],
        voting='soft'
    )

    # Initialize all models in a dictionary
    models_dict = {
        '1': ("KNN", KNeighborsClassifier(n_neighbors=5, weights='distance')),
        '2': ("SVM", SVC(kernel='rbf', C=10, gamma='scale', probability=True)),
        '3': ("Random Forest", rf_champ),
        '4': ("MLP Neural Net", mlp_champ),
        '5': ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        '6': ("HYBRID (MLP + RF)", hybrid_model)
    }

    # Train them all instantly
    for key, (name, model) in models_dict.items():
        model.fit(X_scaled, y)
        print(f"✅ {name} trained!")

    print("\n" + "="*50)
    print("🚀 LIVE TRANSITION TESTER READY")
    print("Press 1-6 to switch models. Press 'q' to quit.")
    print("="*50)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    current_key = '6' # Default to our new Hybrid!

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw UI Background
        cv2.rectangle(frame, (0, 0), (420, 180), (0, 0, 0), -1)

        model_name, active_model = models_dict[current_key]
        cv2.putText(frame, f"MODEL: {model_name} (Press 1-6)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            
            features = extract_geometric_features(results.multi_hand_landmarks[0])
            features_scaled = scaler.transform([features])
            
            # Get the exact probability spread
            probs = active_model.predict_proba(features_scaled)[0]
            classes = active_model.classes_
            
            # Sort the predictions to show the top 3 highest guesses
            sorted_indices = np.argsort(probs)[::-1]
            
            y_offset = 70
            for i in range(3):
                idx = sorted_indices[i]
                gesture_name = classes[idx]
                confidence = probs[idx] * 100
                
                # Color code: Green if >= 80%, Yellow if transitioning, Gray otherwise
                if i == 0 and confidence >= 80.0:
                    color = (0, 255, 0)
                elif i == 0:
                    color = (0, 200, 255)
                else:
                    color = (150, 150, 150)
                
                text = f"{gesture_name}: {confidence:.1f}%"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw a visual bar for the probability
                bar_length = int(confidence * 2) # Scale for pixels
                cv2.rectangle(frame, (230, y_offset - 15), (230 + bar_length, y_offset), color, -1)
                
                y_offset += 40

        cv2.imshow("Transition Tester", frame)

        # Keyboard controls for swapping models live
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif chr(key) in models_dict.keys():
            current_key = chr(key)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_transition_tester()
