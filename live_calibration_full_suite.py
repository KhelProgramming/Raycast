import cv2
import mediapipe as mp
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
 
# --- 1. THE GEOMETRIC EXTRACTOR ---
def extract_geometric_features(landmarks):
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    palm_width = np.linalg.norm(points[5] - points[17])
    if palm_width < 1e-6: palm_width = 1.0 

    wrist = points[0]
    # Distances from wrist to tips
    extensions = [np.linalg.norm(wrist - points[idx]) / palm_width for idx in [4, 8, 12, 16, 20]]
    # Pinch distances (Thumb tip to others)
    thumb_tip = points[4]
    pinches = [np.linalg.norm(thumb_tip - points[idx]) / palm_width for idx in [8, 12, 16, 20]]
    # Finger spreads
    spreads = [np.linalg.norm(points[i] - points[j]) / palm_width for i, j in [(8, 12), (12, 16), (16, 20)]]
    # Thumb flexion
    thumb_to_pinky_base = np.linalg.norm(points[4] - points[17]) / palm_width
    
    return extensions + pinches + spreads + [thumb_to_pinky_base]

# --- 2. FULL SUITE CALIBRATION SYSTEM ---
def run_full_calibration():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    # 🎯 ALL 7 GESTURES: 2 angles each for maximum robustness
    CALIBRATION_TASKS = [
        {"label": "Idle", "instruction": "Open palm - Flat"},
        {"label": "Idle", "instruction": "Open palm - Tilted"},
        {"label": "Left Click", "instruction": "Index-Thumb Pinch - Flat"},
        {"label": "Left Click", "instruction": "Index-Thumb Pinch - Tilted"},
        {"label": "Right Click", "instruction": "Middle-Thumb Pinch - Flat"},
        {"label": "Right Click", "instruction": "Middle-Thumb Pinch - Tilted"},
        {"label": "Hold", "instruction": "Index-Thumb CLOSED - Flat"},
        {"label": "Hold", "instruction": "Index-Thumb CLOSED - Tilted"},
        {"label": "Toggle", "instruction": "Ring-Thumb Pinch - Flat"},
        {"label": "Undo", "instruction": "Fist (All fingers curled)"},
        {"label": "Redo", "instruction": "Peace Sign (Index & Middle out)"}
    ]
    
    current_task_idx = 0
    X_train, y_train = [], []
    model = KNeighborsClassifier(n_neighbors=5)
    scaler = StandardScaler()
    is_trained = False

    state = "WAITING"
    timer_start = 0
    SETUP_DURATION = 4.0    # Giving you 4 seconds to adjust
    CAPTURE_DURATION = 1.0  # 1 second high-speed burst

    print("\n" + "="*50)
    print("🚀 FULL SUITE CALIBRATION (7 GESTURES)")
    print("="*50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- UX & LOGIC STATE MACHINE ---
        if state == "DONE":
            cv2.putText(frame, "✅ SYSTEM CALIBRATED & LOCKED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if results.multi_hand_landmarks and is_trained:
                features = extract_geometric_features(results.multi_hand_landmarks[0])
                scaled_features = scaler.transform([features])
                
                probs = model.predict_proba(scaled_features)[0]
                conf = np.max(probs)
                gesture = model.classes_[np.argmax(probs)]
                
                color = (0, 255, 0) if conf >= 0.80 else (0, 255, 255)
                status = "FIRED" if conf >= 0.80 else "TRANSITIONING"
                cv2.putText(frame, f"{status}: {gesture} ({conf*100:.0f}%)", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

        else:
            task = CALIBRATION_TASKS[current_task_idx]
            
            if state == "WAITING":
                cv2.putText(frame, f"TASK {current_task_idx + 1}/{len(CALIBRATION_TASKS)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"POSE: {task['label']}", (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)
                cv2.putText(frame, f"INSTRUCTION: {task['instruction']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, "Press SPACE when ready...", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    state = "SETUP"
                    timer_start = time.time()

            elif state == "SETUP":
                time_left = SETUP_DURATION - (time.time() - timer_start)
                cv2.putText(frame, f"READY IN: {time_left:.1f}s", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 165, 255), 2)
                if time_left <= 0:
                    state = "CAPTURING"
                    timer_start = time.time()

            elif state == "CAPTURING":
                time_left = CAPTURE_DURATION - (time.time() - timer_start)
                cv2.putText(frame, "🔴 RECORDING BURST...", (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                if results.multi_hand_landmarks:
                    features = extract_geometric_features(results.multi_hand_landmarks[0])
                    X_train.append(features)
                    y_train.append(task['label'])
                
                if time_left <= 0:
                    current_task_idx += 1
                    if current_task_idx >= len(CALIBRATION_TASKS):
                        print("\n🔄 Training KNN Model...")
                        X_scaled = scaler.fit_transform(X_train)
                        model.fit(X_scaled, y_train)
                        is_trained = True
                        state = "DONE"
                    else:
                        state = "WAITING"

        if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.imshow("Full Suite Calibration", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_full_calibration()
