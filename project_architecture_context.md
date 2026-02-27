# Project Architecture Context: Touch Gesture System

## 1. System Overview
The "Touch" gesture system is a highly modular, thread-concurrent computer vision application designed to map human hand gestures to system-level mouse and keyboard controls. 

At its core, the system acts as an invisible, intelligent input layer living between a live webcam feed and the host operating system. It achieves this by utilizing MediaPipe for 3D hand landmark extraction, geometric normalization for spatial invariance, and a machine learning classifier (currently K-Nearest Neighbors) to rapidly categorize hand poses into discrete executable commands. 

The application architecture heavily leverages the Model-View-Controller (MVC) paradigm:
- **Vision/ML Layer (Model):** Captures frames, extracts landmarks, scaling/normalizing data, and predicting classes in isolated threads.
- **Control Layer (Controller):** Maintains dynamic state (such as mode toggling, sticky click-and-drag mechanics, and adaptive DPI scaling based on Z-axis depth).
- **UI Layer (View):** Built entirely with PyQt5, overlaying transparent, non-blocking visual feedback constructs onto the user's screen (virtual keyboard, particle effects, and status bar).

The orchestration of these components is centralized in the `AirController`, which establishes a continuous, multi-threaded pipeline designed to decouple camera I/O, heavy ML inference, and operating system calls to maintain a smooth, non-blocking 30+ FPS user experience.

---

## 2. Setup & Environment Requirements

### Hardware Requirements
- **Webcam:** Capable of at least 1280x720 resolution at 30 True FPS.
- **CPU:** Standard modern multi-core processor (required for parallel processing threads).

### System Dependencies & Libraries
To run the system or benchmark modules from scratch, the following dependencies are strictly required. 
- `opencv-python` (cv2) - Camera interfacing and image matrix manipulation.
- `mediapipe` - Core 21-point 3D hand landmark recognition framework.
- `scikit-learn` - Machine learning (KNN, SVM, RF), standard scaling, grid search, and metrics.
- `PyQt5` - UI rendering (Status Bar, Virtual Keyboard, Particles) and primary application event loop.
- `pyautogui` - OS-level execution of mouse movements, clicks, drags, and keyboard shortcuts.
- `numpy` & `pandas` - Array manipulation, geometric math, and dataset management.
- `matplotlib` & `seaborn` & `tabulate` - Specifically required for the `benchmark_visualizer.py` tool.

### Setup Instructions
1. **Ensure Python 3.8 - 3.10 is installed** (MediaPipe compatibility is best optimized here).
2. **Open a terminal** and strictly initialize a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install the required packages**:
   ```bash
   pip install opencv-python mediapipe scikit-learn PyQt5 pyautogui numpy pandas matplotlib seaborn tabulate
   ```
4. **Data Prerequisites**: Ensure that standard training data exists at `dataset/samples/gesture_data_preprocessed_louise2_geometric.csv` as mapped in `config.py`.

---

## 3. Machine Learning Pipeline

### Data Preprocessing
The model does not classify raw image pixels; it operates on 3D geometric skeletons. 
When MediaPipe generates a `multi_hand_landmarks` object containing 21 vertices (X, Y, Z), it is intercepted by `preprocessing/normalizer.py`.
1. **Translation:** The coordinates are shifted so the wrist (landmark 0) acts as the absolute `(0,0,0)` origin.
2. **Scaling:** A normalization factor is derived from the Euclidean distance between the wrist and the middle finger MCP joint (landmark 12/9 approximation). All coordinates are divided by this factor, rendering the input scale-invariant (a hand close to the camera reads identically to a hand far away).
3. **Flattening:** The resulting 3D coordinate array is flattened into a 1D vector (63 features) suitable for tabular ML models.

### Training & Classification
- The `Trainer` class (`ml/trainer.py`) initializes during startup. It loads the `TRAIN_FILE` defined in `config.py`. 
- **Standardization:** A `StandardScaler` fits to the data to ensure 0-mean and unit variance, a strict requirement for distance-based algorithms.
- **Algorithm:** The default production algorithm is a `KNeighborsClassifier(n_neighbors=15)`. While `benchmark_visualizer.py` highlights tests against Random Forests and SVMs (with hyperparameter GridSearch tuning), KNN provides an optimal balance of fast inference latency and accuracy for real-time 30-FPS evaluation without requiring heavy GPU compute.
- **Inference:** In real-time (`ml/predictor.py`), incoming flattened spatial coordinates are standardized via the fitted scaler, then passed to the `.predict()` method to yield one of the discrete gesture classes.

---

## 4. Concurrency & Threading
To prevent heavy ML interference from blocking the camera feed or hanging the UI, `AirController` spins up 5 concurrent `threading.Thread` instances managed via thread-safe `queue.Queue` structures and `threading.Lock()` mutexes.

1. **`camera_thread` (Daemon):** Binds to `cv2.VideoCapture(0)`. Continuously extracts frames, flips them for mirror viewing, updates the UI preview, and pushes the pure `np.ndarray` frame into a capped `frame_queue` (maxsize 2 to prevent memory bloat/frame lag).
2. **`processing_thread` (Daemon):** Pulls from `frame_queue`, calculates FPS, and passes the frame to MediaPipe `HandTracker`. Using the lock, it calculates mapped screen coordinates (`min(y / 0.7, 1.0) * screen_h`), updates adaptive DPI factors (`get_hand_distance`), pushes standardized rows to label-specific `landmark_queues` ("Left" / "Right", maxsize 1), and heavily updates the core `hands_data` dictionary.
3. **`classifier_thread (Left)` (Daemon):** Constantly monitors the Left `landmark_queue`. Upon spotting a new data point, pushes it through the `Predictor`, storing the result back in the locked `hands_data["Left"]["prediction"]`.
4. **`classifier_thread (Right)` (Daemon):** Identical to above, ensuring independent and concurrent prediction for dual-hand usage.
5. **`action_thread` (Daemon):** Throttled loop (~10ms sleep). Filters for "mouse" mode. Evaluates the `hands_data` object and securely fires system OS operations via the `MouseOrchestrator`, intentionally breaking its loop after one valid hand acts to prevent dual-hand collision (e.g., unintended double clicks).

*(Note: Safety signals utilizing PyQt5's internal `pyqtSignal` architecture safely bridge thread boundaries from these background workers to the Main UI Thread).*

---

## 5. Core Functionalities & Mapping

### Mode Management (`ModeManager`)
At the core of the system is a state machine routing between three phases: **Standby**, **Mouse**, and **Keyboard**.
- **Toggle Gesture:** Holding the "toggle" pose securely for `HOLD_TOGGLE_TIME` (config: 1.0s) oscillates the system state. A strict grace period/flicker threshold of 0.3s ensures dropping a frame doesn't ruin the toggle charge.
- **Keyboard Auto-Trigger:** Having two distinct hands visible and predicting the "idle" gesture for a sustained 2.0s triggers Keyboard Mode. Losing that state for 3.0s reverts the system.

*(Input strings `.lower()` sanitization is applied dynamically against predictions to prevent strict casing issues.)*

### Mouse Orchestrator (`control/mouse/`)
In "mouse" mode, the following commands execute:
- **Movement (Hover/Idle mapped):** `MouseMovement` applies a deque-based smoothing algorithm (lookback of 3 frames) to the absolute hand deltas multiplied by the adaptive DPI.
- **Adaptive DPI:** Z-Axis spatial depth is calculated using negative screen Euclidean distance of the wrist to MCP. Pre-mapped states in `config.py` ('very_close' -> 400 DPI up to 'far' -> 3200 DPI) determine the exact granular control bounds.
- **Clicks (`left_click`, `right_click`):** Executed as one-shot signals protected by a standard 0.3s temporal cooldown.
- **Drag (`hold`):** Implements "sticky state". Initiates a `mouseDown`. Requires another valid gesture or a strict 0.4s grace period timeout of non-"hold" state to trigger `mouseUp`.
- **Shortcuts (`undo`, `redo`):** Emulates strict physical `ctrl` down + `z/y` down behaviors via explicit `pyautogui.keyDown/Up` blocks rather than simple hotkey strings perfectly ensuring Windows compatibility.

### Virtual Keyboard (`control/keyboard/`)
A floating, translucent GUI overlay utilizing `QPainter` shapes.
- The `action_thread` calculates UI rectangle collision mapping between the dynamic predicted virtual hand screen-space pointer and physical PyQt Key QRect interfaces.
- Uses `Counter(valid_keys).most_common(1)` polling across an internal frame deque (7 frames deep) combined with the "press" gesture logic to execute key actions, ensuring highly accurate typing despite slight hand shaking.

---

## 6. Component File Map

- `/` (Root)
  - `main.py`: Dependency injection core; initiates PyQt QApplication and instantiates `AirController`.
  - `config.py`: Central source of truth constants (Screen/Camera Dimensions, UI Colors, Cooldown Timers, Static File Paths).
  - `benchmark_visualizer.py`: Exhaustive validation tool utilizing Pandas and Seaborn to test and chart precision/latency arrays of KNN vs SVM vs Random Forests models against dataset splits.
- `/core/`
  - `air_controller.py`: The orchestrator node. Manages global dictionary models, thread pools, locks, and component injection maps.
  - `mode_manager.py`: State timer manager evaluating boolean conditions to route UI context paths.
  - `signals.py`: Explicit mapping of PyQt5 Cross-Thread safe emitters (update_fps, update_particles, etc).
- `/control/`
  - `/mouse/orchestrator.py, dpi.py, movement.py, click.py, drag.py`: High-level PyAutoGUI execution files handling relative math and system triggers.
  - `/keyboard/virtual_keyboard.py, key_actions.py`: The bespoke QPainter transparent system layout algorithm routing into strict standard OS keyboard events. 
- `/ml/`
  - `trainer.py`: Startup loader mapping raw CSV features into the trained `.fit()` standardized KNN memory model.
  - `predictor.py`: The latency-critical real-time inference wrapper yielding single-string commands. 
- `/preprocessing/`
  - `normalizer.py`: Math-heavy node manipulating linear algebra translations for spatial and depth invariance matrices.
- `/vision/`
  - `camera.py`: OpenCV abstraction handling resolution sizing and automated view mirroring.
  - `hand_tracker.py`: Direct implementation interface parsing the actual MediaPipe `Hands` solution pipeline.
- `/ui/`
  - `status_bar.py, particle_overlay.py, camera_preview.py`: Visual Qt feedback rendering elements executing on the main application thread decoupled from the core CV pipelines.

---

## 7. Current State & Next Steps

### Current State
The codebase is incredibly robust with high modularity and clean abstractions. Thread concurrency guarantees strong non-blocking performance while scaling limits CPU thrash. Recent bug fixes involving capitalization errors (`.lower()` application on labels) are successful, and the dataset labels via `LabelEncoder` during benchmark routines correctly parse the string elements. Windows-compliant physical OS undo/redo handlers are currently perfectly stable. 

### Recommended Next Steps / Optimizations
1. **Model Persistence (High Priority):** Currently, the pipeline natively parses CSV sheets and retrains the entire KNN framework utilizing `trainer.py` *every single time* the application boots. For a large dataset, this introduces significant startup block. 
   - **Fix:** Implement `joblib` or `pickle` model serialization in `trainer.py`. If a `.pkl` model file exists, load it immediately. Only re-parse and train from CSV if the `TRAIN_FILE` timestamp changes or if strictly commanded.
2. **Dynamic UI/Resolution Constraints:** While screen coordinates are parsed dynamically relative to the OS default GUI space (`screen.width(), screen.height()`), multiple monitor configurations or severe scaling variations (4K monitors scaled to 150% UI in Windows) might cause discrepancy between logical and physical DPI bounds when `pyautogui` attempts absolute mapping. Further mapping abstraction may be required.
3. **Expanded Grid Calibration:** Implement an application-front-facing calibration mode capturing users' unique hand dimensions (Wrist to MCP variance) mapping it securely directly to `get_hand_distance()` baseline algorithms ensuring Z-depth tracking works perfectly for any user's biological sizing out of the box.
