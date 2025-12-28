import threading
import queue
import time
from PyQt5.QtWidgets import QApplication

# --- MODULAR IMPORTS ---
from gesture_system.config import TRAIN_FILE, CAMERA_WIDTH, CAMERA_HEIGHT
from gesture_system.vision.camera import Camera
from gesture_system.vision.hand_tracker import HandTracker
from gesture_system.preprocessing.normalizer import preprocess_landmarks, get_hand_distance
from gesture_system.ml.trainer import Trainer
from gesture_system.ml.predictor import Predictor
from gesture_system.control.mouse.orchestrator import MouseOrchestrator
from gesture_system.core.mode_manager import ModeManager

class AirController:
    def __init__(self, app, signals_cls, overlay_cls, status_cls, keyboard_cls, preview_cls):
        self.app = app
        
        # 1. CORE
        self.signals = signals_cls()
        self.lock = threading.Lock()
        self.running = True
        
        # 2. VISION & ML
        self.camera = None 
        self.tracker = HandTracker()
        
        trainer = Trainer(TRAIN_FILE)
        model, scaler = trainer.train()
        self.predictor = Predictor(model, scaler)
        
        # 3. CONTROLLERS
        self.mouse_orchestrator = MouseOrchestrator()
        self.mode_manager = ModeManager(self.signals, self.mouse_orchestrator)
        
        # 4. UI
        self.particle_overlay = overlay_cls()
        self.status_bar = status_cls()
        self.keyboard = keyboard_cls()
        self.camera_preview = preview_cls()
        
        # 5. CONNECTIONS
        self.signals.update_particles.connect(self._update_particles)
        self.signals.show_keyboard.connect(self.keyboard.activate)
        self.signals.hide_keyboard.connect(self.keyboard.deactivate)
        self.signals.update_status.connect(self.status_bar.update_mode)
        self.signals.update_predictions.connect(self.status_bar.update_gestures)
        self.signals.update_dpi.connect(self.status_bar.update_dpi)
        self.signals.update_fps.connect(self.status_bar.update_fps_display)
        
        # 6. DATA STRUCTURES
        self.frame_queue = queue.Queue(maxsize=2)
        self.landmark_queues = {"Left": queue.Queue(maxsize=1), "Right": queue.Queue(maxsize=1)}
        self.hands_data = {
            "Left": {"prediction": "idle", "detected": False, "landmarks": None, "pos": (0,0), "screen_pos": (0,0)},
            "Right": {"prediction": "idle", "detected": False, "landmarks": None, "pos": (0,0), "screen_pos": (0,0)}
        }
        
        screen = QApplication.primaryScreen().geometry()
        self.screen_w, self.screen_h = screen.width(), screen.height()

    def _update_particles(self, data):
        for hand, (x, y) in data.items():
            self.particle_overlay.add_particle(hand, x, y)

    def camera_thread(self):
        self.camera = Camera()
        while self.running:
            ret, frame = self.camera.read()
            if not ret: break
            
            self.camera_preview.update_frame(frame)
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            time.sleep(0.01)
        self.camera.release()

    def processing_thread(self):
        prev_time = 0
        
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue
            
            frame = self.frame_queue.get()
            
            # FPS Calculation
            curr_time = time.time()
            fps = 0
            if prev_time != 0:
                delta = curr_time - prev_time
                if delta > 0:
                    fps = int(1 / delta)
            prev_time = curr_time
            self.signals.update_fps.emit(fps)

            # Processing
            result = self.tracker.process(frame)
            h, w, _ = frame.shape
            
            with self.lock:
                self.hands_data["Left"]["detected"] = False
                self.hands_data["Right"]["detected"] = False
                
                if result.multi_hand_landmarks:
                    for landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                        label = handedness.classification[0].label
                        
                        idx_tip = landmarks.landmark[8]
                        ix, iy = int(idx_tip.x * w), int(idx_tip.y * h)
                        
                        compressed_y = idx_tip.y / 0.7
                        screen_x = int(idx_tip.x * self.screen_w)
                        screen_y = int(min(compressed_y, 1.0) * self.screen_h)
                        
                        # DPI Update
                        dpi_factor = get_hand_distance(landmarks)
                        dpi, dpi_label = self.mouse_orchestrator.process_hand(landmarks, dpi_factor)
                        
                        self.signals.update_dpi.emit(dpi, dpi_label)
                        self.signals.update_particles.emit({label: (screen_x, screen_y)})
                        
                        row = preprocess_landmarks(landmarks.landmark)
                        if not self.landmark_queues[label].full():
                            self.landmark_queues[label].put(row)
                        
                        self.hands_data[label].update({
                            "detected": True,
                            "landmarks": landmarks,
                            "pos": (ix, iy),
                            "screen_pos": (screen_x, screen_y)
                        })
                
                self.mode_manager.update(self.hands_data)
                
                if self.mode_manager.current_mode == "keyboard":
                    self.keyboard.check_hover(self.hands_data)
                    
            time.sleep(0.01)

    def classifier_thread(self, label):
        while self.running:
            if self.landmark_queues[label].empty():
                time.sleep(0.01)
                continue
            
            row = self.landmark_queues[label].get()
            pred = self.predictor.predict(row)
            
            with self.lock:
                self.hands_data[label]["prediction"] = pred
            
            if self.hands_data["Left"]["detected"] or self.hands_data["Right"]["detected"]:
                 preds = {l: self.hands_data[l]["prediction"] for l in ["Left", "Right"]}
                 self.signals.update_predictions.emit(preds)

    def action_thread(self):
        while self.running:
            time.sleep(0.01)
            # Only run actions in MOUSE mode
            if self.mode_manager.current_mode != "mouse":
                continue
            
            with self.lock:
                # Priority: Right hand acts, then Left hand
                # Or simply: check both. Using a loop handles both.
                for label in ["Right", "Left"]:
                    if self.hands_data[label]["detected"]:
                        pred = self.hands_data[label]["prediction"]
                        
                        # ✅ UPDATED: Simply pass the gesture to Orchestrator
                        # The Orchestrator now manages sticky state and one-shots
                        self.mouse_orchestrator.process_gesture(pred)
                        
                        # We break after first valid hand to prevent double-clicking 
                        # if both hands are gesturing
                        break

    def run(self):
        threads = [
            threading.Thread(target=self.camera_thread, daemon=True),
            threading.Thread(target=self.processing_thread, daemon=True),
            threading.Thread(target=self.classifier_thread, args=("Left",), daemon=True),
            threading.Thread(target=self.classifier_thread, args=("Right",), daemon=True),
            threading.Thread(target=self.action_thread, daemon=True)
        ]
        for t in threads: t.start()
        print("🎮 Air Controller Running")

    def stop(self):
        self.running = False
        self.mouse_orchestrator.deactivate()
        QApplication.quit()