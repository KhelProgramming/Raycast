import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

DATA_FILE = "user_profile.csv"

# --- 1. THE 3D VECTOR MATH HELPER ---
def calculate_angle(a, b, c):
    """Calculates the 3D angle at joint 'b' in degrees."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0) # Prevent floating point errors
    return np.degrees(np.arccos(cosine_angle))

def get_normalized_points(landmarks):
    """
    Takes raw MediaPipe landmarks and shifts them so the wrist is always at (0, 0, 0).
    """
    # 1. Convert the raw MediaPipe landmarks into a flat numpy array
    raw_points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    # 2. Grab the wrist's coordinates (the wrist is always landmark 0)
    wrist = raw_points[0]
    
    # 3. Subtract the wrist's position from every single joint in the hand
    # This magically shifts the entire hand so the wrist becomes 0, 0, 0
    normalized_points = raw_points - wrist
    
    return normalized_points

# --- 2. THE UPGRADED GEOMETRIC EXTRACTOR ---
def extract_geometric_features(landmarks):
    # USE THE NEW NORMALIZER HERE!
    points = get_normalized_points(landmarks)
    
    # The rest of your exact same 18-feature math stays here:
    
    # 📏 Original 13 Distance Features
    palm_width = np.linalg.norm(points[5] - points[17])
    if palm_width < 1e-6: palm_width = 1.0 

    # Note: points[0] is now mathematically [0.0, 0.0, 0.0], perfectly anchored!
    wrist = points[0] 
    extensions = [np.linalg.norm(wrist - points[idx]) / palm_width for idx in [4, 8, 12, 16, 20]]
    thumb_tip = points[4]
    pinches = [np.linalg.norm(thumb_tip - points[idx]) / palm_width for idx in [8, 12, 16, 20]]
    spreads = [np.linalg.norm(points[i] - points[j]) / palm_width for i, j in [(8, 12), (12, 16), (16, 20)]]
    thumb_to_pinky_base = np.linalg.norm(points[4] - points[17]) / palm_width
    
    # 📐 New 5 Curl Angle Features 
    thumb_angle = calculate_angle(points[1], points[2], points[3]) / 180.0
    index_angle = calculate_angle(points[5], points[6], points[7]) / 180.0
    middle_angle = calculate_angle(points[9], points[10], points[11]) / 180.0
    ring_angle = calculate_angle(points[13], points[14], points[15]) / 180.0
    pinky_angle = calculate_angle(points[17], points[18], points[19]) / 180.0
    
    angles = [thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle]
    
    return extensions + pinches + spreads + [thumb_to_pinky_base] + angles

# --- 3. THE CALIBRATION ENGINE ---
def run_master_calibration():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    # 🎯 ALL 7 GESTURES + THE NEW "UNKNOWN" NOISE CLASS
    CALIBRATION_TASKS = [
        {"label": "Idle", "instruction": "Open palm - Flat"},
        {"label": "Idle", "instruction": "Open palm - Tilted"},
        {"label": "Left Click", "instruction": "Index-Thumb Pinch - Flat"},
        {"label": "Left Click", "instruction": "Index-Thumb Pinch - Tilted"},
        {"label": "Right Click", "instruction": "Middle-Thumb Pinch - Flat"},
        {"label": "Right Click", "instruction": "Middle-Thumb Pinch - Tilted"},
        {"label": "Toggle", "instruction": "Ring-Thumb Pinch"},
        {"label": "Undo", "instruction": "Fist (All fingers curled)"},
        {"label": "Redo", "instruction": "Peace Sign (Index & Middle out)"},
        # THE MAGIC SAUCE: Negative Training Data
        {"label": "Unknown", "instruction": "Wiggle fingers, scratch nose, random shapes!"},
        {"label": "Unknown", "instruction": "Half-closed hand, transition shapes, move around!"}
    ]
    
    current_task_idx = 0
    X_train, y_train = [], []

    state = "WAITING"
    timer_start = 0
    SETUP_DURATION = 3.0    # 3 seconds to get your hand in position
    CAPTURE_DURATION = 1.5  # 1.5 seconds of rapid data collection

    print("\n" + "="*50)
    print("📸 PHASE 2: MASTER DATA COLLECTION (ANGLES + NOISE)")
    print("="*50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        task = CALIBRATION_TASKS[current_task_idx]
        
        if state == "WAITING":
            cv2.putText(frame, f"TASK {current_task_idx + 1}/{len(CALIBRATION_TASKS)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"POSE: {task['label']}", (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(frame, f"DO: {task['instruction']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, "Press SPACE to start timer...", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if cv2.waitKey(1) & 0xFF == ord(' '):
                state = "SETUP"
                timer_start = time.time()

        elif state == "SETUP":
            time_left = SETUP_DURATION - (time.time() - timer_start)
            cv2.putText(frame, f"HOLD STILL IN: {time_left:.1f}s", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 165, 255), 2)
            if time_left <= 0:
                state = "CAPTURING"
                timer_start = time.time()

        elif state == "CAPTURING":
            time_left = CAPTURE_DURATION - (time.time() - timer_start)
            cv2.putText(frame, "🔴 RECORDING DATA...", (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            
            if results.multi_hand_landmarks:
                # USING THE NEW UPGRADED EXTRACTOR
                features = extract_geometric_features(results.multi_hand_landmarks[0])
                X_train.append(features)
                y_train.append(task['label'])
            
            if time_left <= 0:
                current_task_idx += 1
                if current_task_idx >= len(CALIBRATION_TASKS):
                    break # All tasks done, exit loop
                else:
                    state = "WAITING"

        cv2.imshow("Master Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # --- SAVE TO CSV ---
    if len(X_train) > 0:
        df = pd.DataFrame(X_train)
        df['label'] = y_train
        df.to_csv(DATA_FILE, index=False)
        print(f"\n✅ SUCCESS! {len(X_train)} frames of 18-feature data saved to {DATA_FILE}")
        print("You are now ready to run the Live Transition Tester!")
    else:
        print("\n⚠️ No data was captured. Please try again.")

if __name__ == "__main__":
    run_master_calibration()
