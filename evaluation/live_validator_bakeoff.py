import os
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# First time (train + live)
# python gesture_system/evaluation/live_validator_bakeoff.py

# Later (ONLY live, no retrain)
# python gesture_system/evaluation/live_validator_bakeoff.py --live

# Retrain only when needed
# python gesture_system/evaluation/live_validator_bakeoff.py --train --force_retrain


# ======================================================================================
# CONFIG (edit paths if needed)
# ======================================================================================

DATASET_FILES = [
    "gesture_system/dataset/samples/KHEL_GEO18.csv",
    "gesture_system/dataset/samples/BRYAN_GEO18.csv",
    "gesture_system/dataset/samples/DON_GEO18.csv",
    "gesture_system/dataset/samples/LOUISE_GEO18.csv",
]

# If Toggle is inconsistent, keep True for the global validator.
REMOVE_TOGGLE_DEFAULT = False

ARTIFACT_DIR = "gesture_system/evaluation/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Saved artifacts (so you don't retrain again)
SCALER_PATH = os.path.join(ARTIFACT_DIR, "validator_scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(ARTIFACT_DIR, "validator_label_encoder.joblib")
META_PATH = os.path.join(ARTIFACT_DIR, "validator_model_meta.json")

# Each tuned model saved separately
MODEL_PATHS = {
    "RF": os.path.join(ARTIFACT_DIR, "validator_RF.joblib"),
    "MLP": os.path.join(ARTIFACT_DIR, "validator_MLP.joblib"),
    "SVM": os.path.join(ARTIFACT_DIR, "validator_SVM.joblib"),
    "KNN": os.path.join(ARTIFACT_DIR, "validator_KNN.joblib"),
    "LOGREG": os.path.join(ARTIFACT_DIR, "validator_LOGREG.joblib"),
    "HYBRID_RF_MLP": os.path.join(ARTIFACT_DIR, "validator_HYBRID_RF_MLP.joblib"),
}

TRIAL_LOG_PATH = os.path.join(ARTIFACT_DIR, "validator_trials.csv")

# Live validation thresholds (tune later)
MIN_TOP1_PROB_DEFAULT = 0.70
MIN_MARGIN_DEFAULT = 0.20
REQUIRED_STABLE_FRAMES_DEFAULT = 12


# ======================================================================================
# 18-FEATURE EXTRACTOR (matches your GEO18 design)
# ======================================================================================

EPS = 1e-9

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """3D angle at joint b, degrees."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + EPS
    cosine_angle = np.dot(ba, bc) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

def get_normalized_points(hand_landmarks) -> np.ndarray:
    """Wrist-relative normalization (wrist at 0,0,0)."""
    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float64)
    wrist = raw[0].copy()
    return raw - wrist

def extract_geometric_features(hand_landmarks) -> List[float]:
    """
    Returns 18 features:
      extensions(5) + pinches(4) + spreads(3) + thumb_to_pinky_base(1) + angles(5)
    """
    points = get_normalized_points(hand_landmarks)

    palm_width = np.linalg.norm(points[5] - points[17])
    if palm_width < 1e-6:
        palm_width = 1.0

    wrist = points[0]  # now [0,0,0]
    extensions = [np.linalg.norm(wrist - points[idx]) / palm_width for idx in [4, 8, 12, 16, 20]]

    thumb_tip = points[4]
    pinches = [np.linalg.norm(thumb_tip - points[idx]) / palm_width for idx in [8, 12, 16, 20]]

    spreads = [np.linalg.norm(points[i] - points[j]) / palm_width for i, j in [(8, 12), (12, 16), (16, 20)]]

    thumb_to_pinky_base = np.linalg.norm(points[4] - points[17]) / palm_width

    thumb_angle = calculate_angle(points[1], points[2], points[3]) / 180.0
    index_angle = calculate_angle(points[5], points[6], points[7]) / 180.0
    middle_angle = calculate_angle(points[9], points[10], points[11]) / 180.0
    ring_angle = calculate_angle(points[13], points[14], points[15]) / 180.0
    pinky_angle = calculate_angle(points[17], points[18], points[19]) / 180.0

    angles = [thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle]

    return extensions + pinches + spreads + [thumb_to_pinky_base] + angles


# ======================================================================================
# DATA LOADING
# ======================================================================================

def load_merged_dataset(files: List[str], remove_toggle: bool) -> Tuple[np.ndarray, np.ndarray]:
    dfs = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing dataset file: {f}")
        df = pd.read_csv(f)
        if "label" not in df.columns:
            raise ValueError(f"{f} must contain a 'label' column.")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    if remove_toggle:
        merged = merged[merged["label"] != "Toggle"].copy()

    # Take everything except label as X (works if columns are "0".."17" strings too)
    X = merged.drop(columns=["label"]).values.astype(np.float64)
    y = merged["label"].astype(str).values
    return X, y


# ======================================================================================
# TRAINING + SAVING (Peak params)
# ======================================================================================

@dataclass
class ValidatorArtifacts:
    label_encoder: LabelEncoder
    scaler: StandardScaler
    models: Dict[str, object]
    meta: Dict[str, dict]

def _save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def artifacts_exist() -> bool:
    required = [SCALER_PATH, LABEL_ENCODER_PATH, META_PATH] + list(MODEL_PATHS.values())
    return all(os.path.exists(p) for p in required)

def save_artifacts(art: ValidatorArtifacts) -> None:
    joblib.dump(art.scaler, SCALER_PATH)
    joblib.dump(art.label_encoder, LABEL_ENCODER_PATH)

    for name, model in art.models.items():
        joblib.dump(model, MODEL_PATHS[name])

    _save_json(META_PATH, art.meta)

    print("\n" + "=" * 90)
    print("💾 Saved validator artifacts:")
    print(f"  scaler        -> {SCALER_PATH}")
    print(f"  label encoder -> {LABEL_ENCODER_PATH}")
    print(f"  meta          -> {META_PATH}")
    for k, p in MODEL_PATHS.items():
        print(f"  model {k:12} -> {p}")
    print("=" * 90)

def load_artifacts() -> ValidatorArtifacts:
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    meta = _load_json(META_PATH)

    models = {}
    for name, path in MODEL_PATHS.items():
        models[name] = joblib.load(path)

    return ValidatorArtifacts(label_encoder=le, scaler=scaler, models=models, meta=meta)

def train_peak_models(X: np.ndarray, y_str: np.ndarray) -> ValidatorArtifacts:
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # NOTE: all tuned models must support predict_proba for live gating.
    # - SVM: probability=True
    # - KNN: has predict_proba
    # - LOGREG: has predict_proba
    # - MLP: has predict_proba
    candidates: Dict[str, Tuple[object, dict]] = {
        "RF": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "n_estimators": [200, 400, 700, 1000],
                "max_depth": [None, 10, 15, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        ),
        "SVM": (
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
            {
                "C": np.logspace(-1, 2, 12),     # 0.1..100
                "gamma": np.logspace(-3, 0, 10), # 0.001..1
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": list(range(3, 26, 2)),
                "weights": ["uniform", "distance"],
                "p": [1, 2],  # L1 / L2
            },
        ),
        "LOGREG": (
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
                n_jobs=-1,
            ),
            {
                "C": np.logspace(-2, 2, 12),  # 0.01..100
            },
        ),
        "MLP": (
            MLPClassifier(
                max_iter=1800,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            ),
            {
                "hidden_layer_sizes": [(64, 32), (128, 64, 32), (128, 128, 64), (256, 128, 64)],
                "activation": ["relu", "tanh"],
                "alpha": np.logspace(-6, -2, 9),             # L2
                "learning_rate_init": np.logspace(-4, -2, 8),# 1e-4..1e-2
                "batch_size": [128, 256, 512],
            },
        ),
    }

    best_models: Dict[str, object] = {}
    meta: Dict[str, dict] = {}

    print("\n" + "=" * 90)
    print("🏋️ TRAINING PEAK VALIDATOR MODELS (RandomizedSearchCV)")
    print("=" * 90)

    for name, (model, param_space) in candidates.items():
        print(f"\n🔧 Tuning {name} ...")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_space,
            n_iter=30,                  # increase to 60+ if you want heavier tuning
            scoring="f1_weighted",
            cv=cv,
            verbose=0,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(Xs, y)

        best = search.best_estimator_
        best_models[name] = best

        meta[name] = {
            "best_params": search.best_params_,
            "best_cv_f1_weighted": float(search.best_score_),
            "n_iter": 30,
            "cv_folds": 5,
            "scoring": "f1_weighted",
        }

        print(f"✅ {name} best CV F1(w): {search.best_score_:.4f}")
        print(f"   params: {search.best_params_}")

    # Hybrid: tuned RF + tuned MLP soft-vote
    rf = best_models["RF"]
    mlp = best_models["MLP"]
    hybrid = VotingClassifier(estimators=[("RF", rf), ("MLP", mlp)], voting="soft")
    hybrid.fit(Xs, y)

    best_models["HYBRID_RF_MLP"] = hybrid
    meta["HYBRID_RF_MLP"] = {"note": "soft vote of tuned RF + tuned MLP"}

    print("\n✅ Hybrid trained: HYBRID_RF_MLP")

    return ValidatorArtifacts(label_encoder=le, scaler=scaler, models=best_models, meta=meta)


# ======================================================================================
# LIVE VALIDATOR BAKEOFF
# ======================================================================================

def top1_margin(probs: np.ndarray) -> Tuple[int, float, float]:
    """Returns (top_idx, top1_prob, margin=top1-top2)."""
    order = np.argsort(probs)[::-1]
    top = order[0]
    p1 = float(probs[top])
    p2 = float(probs[order[1]]) if len(order) > 1 else 0.0
    return int(top), p1, (p1 - p2)

def ensure_trial_log(path: str) -> None:
    if os.path.exists(path):
        return
    cols = [
        "timestamp",
        "tester_name",
        "expected_label",
        "model_name",
        "accepted",
        "time_to_accept_sec",
        "stable_frames_required",
        "min_top1_prob",
        "min_margin",
        "final_top1_prob",
        "final_margin",
        "final_predicted_label",
    ]
    pd.DataFrame(columns=cols).to_csv(path, index=False)

def append_trial(path: str, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)

def run_live_validator(art: ValidatorArtifacts,
                       min_top1_prob: float,
                       min_margin: float,
                       stable_frames: int) -> None:
    ensure_trial_log(TRIAL_LOG_PATH)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)

    # Order shown in UI
    preferred = ["RF", "MLP", "SVM", "KNN", "LOGREG", "HYBRID_RF_MLP"]
    model_keys = [k for k in preferred if k in art.models] + [k for k in art.models.keys() if k not in preferred]

    current_model_idx = model_keys.index("RF") if "RF" in model_keys else 0
    expected_label: Optional[str] = None
    tester_name: str = "anonymous"

    stable_count = 0
    start_time = None

    print("\n" + "=" * 70)
    print("🧪 LIVE VALIDATOR BAKE-OFF")
    print("=" * 70)
    print("Controls:")
    print("  [1-9]  switch model (by index shown on screen)")
    print("  [e]    set Expected Label (type in console)")
    print("  [n]    set Tester Name (type in console)")
    print("  [r]    reset stability counter")
    print("  [q]    quit")
    print("\nValidation rule to ACCEPT:")
    print(f"  predicted == expected AND top1_prob >= {min_top1_prob} AND margin >= {min_margin}")
    print(f"  for {stable_frames} consecutive frames\n")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # UI background
        cv2.rectangle(frame, (0, 0), (560, 240), (0, 0, 0), -1)

        model_name = model_keys[current_model_idx]
        model = art.models[model_name]

        cv2.putText(frame, f"MODEL [{current_model_idx+1}/{len(model_keys)}]: {model_name}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        exp_txt = expected_label if expected_label else "(not set)"
        cv2.putText(frame, f"EXPECTED: {exp_txt}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.putText(frame, f"TESTER: {tester_name}", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.putText(frame, f"STABLE: {stable_count}/{stable_frames}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if stable_count > 0 else (200, 200, 200), 2)

        # show model selector list
        y_menu = 140
        for i, k in enumerate(model_keys[:9]):
            marker = ">" if i == current_model_idx else " "
            cv2.putText(frame, f"{marker} {i+1}) {k}", (10, y_menu),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            y_menu += 18

        final_pred_label = "-"
        final_p1 = 0.0
        final_margin = 0.0
        accepted = False

        if results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            feats = extract_geometric_features(results.multi_hand_landmarks[0])
            X_live = art.scaler.transform([feats])

            # predict probabilities
            probs = model.predict_proba(X_live)[0]
            top_idx, p1, margin = top1_margin(probs)

            pred_label = art.label_encoder.inverse_transform([top_idx])[0]
            final_pred_label = pred_label
            final_p1 = float(p1)
            final_margin = float(margin)

            # show top3
            order = np.argsort(probs)[::-1]
            y0 = 145
            x0 = 330
            for i in range(min(3, len(order))):
                idx = order[i]
                lbl = art.label_encoder.inverse_transform([idx])[0]
                conf = float(probs[idx]) * 100.0
                color = (0, 255, 0) if i == 0 and conf >= 80 else ((0, 200, 255) if i == 0 else (150, 150, 150))
                cv2.putText(frame, f"{i+1}) {lbl}: {conf:5.1f}%", (x0, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                y0 += 25

            meets_prob = (final_p1 >= min_top1_prob)
            meets_margin = (final_margin >= min_margin)
            meets_label = (expected_label is not None and pred_label == expected_label)

            if meets_prob and meets_margin and meets_label:
                stable_count += 1
                if start_time is None:
                    start_time = time.time()
            else:
                stable_count = 0
                start_time = None

            if stable_count >= stable_frames:
                accepted = True
                t_accept = time.time() - start_time if start_time else 0.0

                row = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tester_name": tester_name,
                    "expected_label": expected_label,
                    "model_name": model_name,
                    "accepted": True,
                    "time_to_accept_sec": round(float(t_accept), 3),
                    "stable_frames_required": stable_frames,
                    "min_top1_prob": min_top1_prob,
                    "min_margin": min_margin,
                    "final_top1_prob": round(final_p1, 4),
                    "final_margin": round(final_margin, 4),
                    "final_predicted_label": final_pred_label,
                }
                append_trial(TRIAL_LOG_PATH, row)

                # reset for next trial
                stable_count = 0
                start_time = None

        status = "ACCEPT ✅" if accepted else "..."
        cv2.putText(frame, f"STATUS: {status}", (330, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if accepted else (200, 200, 200), 2)

        cv2.putText(frame, f"P1={final_p1:.2f}  MARGIN={final_margin:.2f}", (330, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.putText(frame, f"PRED: {final_pred_label}", (330, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        cv2.imshow("Validator Bakeoff", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            stable_count = 0
            start_time = None
        elif key == ord("e"):
            cap.release()
            cv2.destroyAllWindows()
            print("\nAvailable labels:", list(art.label_encoder.classes_))
            expected_label = input("Type expected label EXACTLY: ").strip()
            if expected_label not in set(art.label_encoder.classes_):
                print("⚠️ Label not recognized. Expected label cleared.")
                expected_label = None
            cap = cv2.VideoCapture(0)
        elif key == ord("n"):
            cap.release()
            cv2.destroyAllWindows()
            tester_name = input("\nType tester name: ").strip() or "anonymous"
            cap = cv2.VideoCapture(0)
        elif ord("1") <= key <= ord("9"):
            idx = (key - ord("1"))
            if 0 <= idx < len(model_keys):
                current_model_idx = idx

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print(f"\n✅ Trials logged to: {TRIAL_LOG_PATH}")


# ======================================================================================
# ENTRYPOINT
# ======================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Train + tune models and save artifacts.")
    ap.add_argument("--live", action="store_true", help="Run live validator UI (loads saved artifacts).")
    ap.add_argument("--force_retrain", action="store_true", help="Retrain even if artifacts exist.")
    ap.add_argument("--remove_toggle", action="store_true", default=REMOVE_TOGGLE_DEFAULT, help="Remove Toggle rows from merged dataset.")
    ap.add_argument("--min_top1_prob", type=float, default=MIN_TOP1_PROB_DEFAULT)
    ap.add_argument("--min_margin", type=float, default=MIN_MARGIN_DEFAULT)
    ap.add_argument("--stable_frames", type=int, default=REQUIRED_STABLE_FRAMES_DEFAULT)
    args = ap.parse_args()

    # Default behavior: if no flags, do both
    do_train = args.train or (not args.train and not args.live)
    do_live = args.live or (not args.train and not args.live)

    # Train if needed
    if do_train:
        if artifacts_exist() and not args.force_retrain:
            print("✅ Artifacts already exist — skipping retrain (use --force_retrain to override).")
        else:
            X, y = load_merged_dataset(DATASET_FILES, args.remove_toggle)
            print(f"Loaded merged dataset: X={X.shape}, y={len(y)} samples")
            print("Label counts:\n", pd.Series(y).value_counts())

            art = train_peak_models(X, y)
            save_artifacts(art)

    # Live
    if do_live:
        if not artifacts_exist():
            raise RuntimeError("No saved artifacts found. Run with --train first (or run with no flags).")

        art = load_artifacts()
        run_live_validator(
            art,
            min_top1_prob=args.min_top1_prob,
            min_margin=args.min_margin,
            stable_frames=args.stable_frames
        )

if __name__ == "__main__":
    main()