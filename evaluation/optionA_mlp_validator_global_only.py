import os
import time
import json
from typing import List, Tuple, Optional

import cv2
import numpy as np
import joblib
import mediapipe as mp


# ======================================================================================
# PATHS
# ======================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(HERE, "artifacts")

SCALER_PATH = os.path.join(ART_DIR, "validator_scaler.joblib")
LE_PATH = os.path.join(ART_DIR, "validator_label_encoder.joblib")
GLOBAL_MLP_PATH = os.path.join(ART_DIR, "validator_MLP.joblib")

OUT_DIR = os.path.join(ART_DIR, "optionA_global_only")
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================================================
# CONFIG (includes Toggle)
# ======================================================================================
GESTURES = ["Idle", "Hold", "Left Click", "Right Click", "Undo", "Redo", "Toggle"]

CALIB_SECONDS = 5.0
VAL_MIN_TOP1_PROB = 0.70
VAL_MIN_MARGIN = 0.20

STRICT_ZERO_DROPS = False
MAX_DROPS = 3


# ======================================================================================
# WINDOW HELPERS
# ======================================================================================
def maximize_window(win_name: str):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass


# ======================================================================================
# GEO18 FEATURES
# ======================================================================================
EPS = 1e-9

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + EPS
    cosang = float(np.dot(ba, bc) / denom)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))

def get_normalized_points(hand_landmarks) -> np.ndarray:
    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float64)
    return raw - raw[0]

def extract_geometric18(hand_landmarks) -> List[float]:
    p = get_normalized_points(hand_landmarks)
    palm_width = np.linalg.norm(p[5] - p[17])
    if palm_width < 1e-6:
        palm_width = 1.0

    wrist = p[0]
    extensions = [np.linalg.norm(wrist - p[i]) / palm_width for i in [4, 8, 12, 16, 20]]
    thumb_tip = p[4]
    pinches = [np.linalg.norm(thumb_tip - p[i]) / palm_width for i in [8, 12, 16, 20]]
    spreads = [np.linalg.norm(p[i] - p[j]) / palm_width for i, j in [(8, 12), (12, 16), (16, 20)]]
    thumb_to_pinky_base = np.linalg.norm(p[4] - p[17]) / palm_width
    angles = [
        calculate_angle(p[1], p[2], p[3]) / 180.0,
        calculate_angle(p[5], p[6], p[7]) / 180.0,
        calculate_angle(p[9], p[10], p[11]) / 180.0,
        calculate_angle(p[13], p[14], p[15]) / 180.0,
        calculate_angle(p[17], p[18], p[19]) / 180.0,
    ]
    return extensions + pinches + spreads + [thumb_to_pinky_base] + angles


# ======================================================================================
# GLOBAL MLP
# ======================================================================================
def top1_margin(probs: np.ndarray) -> Tuple[int, float, float]:
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    p1 = float(probs[top])
    p2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    return top, p1, (p1 - p2)

class GlobalMLP:
    def __init__(self):
        for p in (SCALER_PATH, LE_PATH, GLOBAL_MLP_PATH):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing artifact: {p}")
        self.scaler = joblib.load(SCALER_PATH)
        self.le = joblib.load(LE_PATH)
        self.mlp = joblib.load(GLOBAL_MLP_PATH)

    def predict(self, f18: List[float]) -> Tuple[str, float, float]:
        Xs = self.scaler.transform([f18])
        probs = self.mlp.predict_proba(Xs)[0]
        top, p1, margin = top1_margin(probs)
        label = self.le.inverse_transform([top])[0]
        return label, p1, margin


# ======================================================================================
# CALIBRATION SESSION (single window, single camera)
# ======================================================================================
class CalibrationSession:
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.cap = cv2.VideoCapture(0)
        maximize_window(self.window_title)

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.hands.close()
        except Exception:
            pass

    def _draw_text(self, frame, lines: List[Tuple[str, Tuple[int,int,int]]]):
        y = 35
        for txt, col in lines:
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            y += 35

    def calibrate_one(self, gm: GlobalMLP, target_label: str) -> np.ndarray:
        capturing = False
        start_t = 0.0
        feats: List[List[float]] = []
        drops = 0

        status = "WAITING: press SPACE to start"
        status_col = (200, 200, 200)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)

            # Live preview (always)
            if res.multi_hand_landmarks:
                f18_preview = extract_geometric18(res.multi_hand_landmarks[0])
                pred, p1, margin = gm.predict(f18_preview)
                live = f"LIVE: {pred} p={p1:.2f} m={margin:.2f}"
            else:
                live = "LIVE: NO HAND"

            if not capturing:
                self._draw_text(frame, [
                    (f"TARGET: {target_label}", (255, 255, 255)),
                    ("Press SPACE to start 5s capture | Press Q to quit", (200, 200, 200)),
                    (status, status_col),
                    (live, (180, 180, 255)),
                ])
            else:
                elapsed = time.time() - start_t
                remaining = max(0.0, CALIB_SECONDS - elapsed)

                frame_status = "NO HAND"
                frame_col = (0, 0, 255)

                if res.multi_hand_landmarks:
                    f18 = extract_geometric18(res.multi_hand_landmarks[0])
                    pred, p1, margin = gm.predict(f18)

                    cond = (pred == target_label and p1 >= VAL_MIN_TOP1_PROB and margin >= VAL_MIN_MARGIN)
                    frame_status = f"{pred} p={p1:.2f} m={margin:.2f} {'OK' if cond else 'BAD'}"
                    frame_col = (0, 255, 0) if cond else (0, 0, 255)

                    feats.append(f18)

                    if not cond:
                        drops += 1
                        if STRICT_ZERO_DROPS or (drops > MAX_DROPS):
                            # fail attempt
                            capturing = False
                            feats = []
                            drops = 0
                            status = "FAILED ❌ (press SPACE to retry)"
                            status_col = (0, 0, 255)

                self._draw_text(frame, [
                    (f"TARGET: {target_label}", (255, 255, 255)),
                    (f"CAPTURING... {remaining:0.1f}s left | Drops: {drops}", (200, 200, 200)),
                    (frame_status, frame_col),
                    (live, (180, 180, 255)),
                ])

                if capturing and remaining <= 0.001 and len(feats) > 0:
                    arr = np.array(feats, dtype=np.float64)
                    status = f"ACCEPTED ✅ ({arr.shape[0]} frames)"
                    status_col = (0, 255, 0)

                    # flash success
                    self._draw_text(frame, [(status, status_col)])
                    cv2.imshow(self.window_title, frame)
                    cv2.waitKey(350)
                    return arr

            cv2.imshow(self.window_title, frame)
            k = cv2.waitKey(1) & 0xFF

            if k in (ord("q"), ord("Q")):
                raise SystemExit("Quit")

            if (not capturing) and k == 32:  # SPACE
                capturing = True
                start_t = time.time()
                feats = []
                drops = 0
                status = "STARTED..."
                status_col = (200, 200, 200)


def run_calibration_save_only():
    gm = GlobalMLP()

    # sanity: ensure gestures exist in global label encoder
    missing = [g for g in GESTURES if g not in set(gm.le.classes_)]
    if missing:
        raise ValueError(
            f"These gestures are missing in GLOBAL label encoder: {missing}\n"
            f"Available: {list(gm.le.classes_)}"
        )

    session = CalibrationSession("Calibration (Option A)")
    collected = {}
    try:
        for g in GESTURES:
            Xg = session.calibrate_one(gm, g)
            np.save(os.path.join(OUT_DIR, f"{g.replace(' ', '_')}_features.npy"), Xg)
            collected[g] = {"n_frames": int(Xg.shape[0])}
    finally:
        session.close()

    meta = {
        "option": "A_global_only",
        "validator": "MLP",
        "gestures": GESTURES,
        "policy": {
            "calib_seconds": CALIB_SECONDS,
            "val_min_prob": VAL_MIN_TOP1_PROB,
            "val_min_margin": VAL_MIN_MARGIN,
            "strict_zero_drops": STRICT_ZERO_DROPS,
            "max_drops": MAX_DROPS,
        },
        "collected": collected,
    }
    with open(os.path.join(OUT_DIR, "session_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n✅ Option A calibration done.")
    print(f"💾 Saved to: {OUT_DIR}")


def live_global_only():
    gm = GlobalMLP()

    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    WIN = "Live (Option A: Global MLP Only)"
    maximize_window(WIN)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            text = "NO HAND"
            color = (0, 0, 255)

            if res.multi_hand_landmarks:
                f18 = extract_geometric18(res.multi_hand_landmarks[0])
                pred, p1, margin = gm.predict(f18)
                text = f"GLOBAL(MLP): {pred} p={p1:.2f} m={margin:.2f}"
                color = (0, 255, 0)

            cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.putText(frame, "Press Q to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow(WIN, frame)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "live"
    if mode == "calibrate":
        run_calibration_save_only()
    elif mode == "live":
        live_global_only()
    else:
        print("Usage: python optionA_mlp_validator_global_only.py [calibrate|live]")