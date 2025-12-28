import time
from gesture_system.config import HOLD_TOGGLE_TIME

class ModeManager:
    """
    Handles logic for switching between Standby, Mouse, and Keyboard modes
    based on gestures and timing.
    """
    def __init__(self, signals, mouse_controller):
        self.signals = signals
        self.mouse = mouse_controller
        self.current_mode = "standby"
        
        # State variables
        self.toggle_gesture_start = None
        self.last_mode_switch = 0
        self.mode_switch_cooldown = 1.5
        
        # Grace period variables
        self.last_toggle_seen = 0
        self.flicker_threshold = 0.3
        
        # Keyboard Timers
        self.kb_activation_start = None
        self.kb_activation_delay = 2.0
        self.kb_deactivation_start = None
        self.kb_deactivation_delay = 3.0

    def update(self, hands_data):
        now = time.time()
        left = hands_data["Left"]
        right = hands_data["Right"]
        
        # 1. TOGGLE GESTURE LOGIC (Case-Insensitive Fix)
        # We convert prediction to string and lower() it to match "toggle" safely
        left_pred = str(left["prediction"]).lower()
        right_pred = str(right["prediction"]).lower()
        
        current_frame_is_toggle = (left["detected"] and left_pred == "toggle") or \
                                  (right["detected"] and right_pred == "toggle")
        
        if current_frame_is_toggle:
            self.last_toggle_seen = now
            
            if self.toggle_gesture_start is None:
                self.toggle_gesture_start = now
                print("⏱️  Toggle started...")
            
            # Calculate duration
            duration = now - self.toggle_gesture_start
            
            # Print progress every 0.5s
            if duration > 0.5 and duration % 0.5 < 0.1:
                print(f"⏳ Toggle Hold: {duration:.1f}s / {HOLD_TOGGLE_TIME}s")
                
            if duration >= HOLD_TOGGLE_TIME:
                if now - self.last_mode_switch >= self.mode_switch_cooldown:
                    self._switch_mode_toggle()
                    self.last_mode_switch = now
                    self.toggle_gesture_start = None
        
        else:
            # GRACE PERIOD LOGIC
            if self.toggle_gesture_start is not None:
                if (now - self.last_toggle_seen) > self.flicker_threshold:
                    print("❌ Toggle cancelled (Gesture lost)")
                    self.toggle_gesture_start = None

        # 2. TWO IDLE HANDS (For Keyboard Mode)
        # Also using .lower() for safety
        two_idle = (left["detected"] and right["detected"] and \
                    left_pred == "idle" and right_pred == "idle")

        # Activation Logic
        if two_idle and self.current_mode != "keyboard":
            self.kb_deactivation_start = None
            if self.kb_activation_start is None:
                self.kb_activation_start = now
                print(f"✌️ Two hands idle... {self.kb_activation_delay}s")
            elif now - self.kb_activation_start >= self.kb_activation_delay:
                self._activate_keyboard()
                self.kb_activation_start = None
        else:
            self.kb_activation_start = None

        # Deactivation Logic
        if self.current_mode == "keyboard" and not two_idle:
            if self.kb_deactivation_start is None:
                self.kb_deactivation_start = now
                print(f"⚠️ Hands lost/moved... {self.kb_deactivation_delay}s to close")
            elif now - self.kb_deactivation_start >= self.kb_deactivation_delay:
                self._deactivate_keyboard()
                self.kb_deactivation_start = None
        elif self.current_mode == "keyboard" and two_idle:
             self.kb_deactivation_start = None

    def _switch_mode_toggle(self):
        print(f"🔄 SWITCHING MODE. Current: {self.current_mode}")
        
        if self.current_mode == "standby":
            self._set_mode("mouse")
            self.mouse.activate()
        elif self.current_mode == "mouse":
            self._set_mode("standby")
            self.mouse.deactivate()
        elif self.current_mode == "keyboard":
            self._deactivate_keyboard()
            self._set_mode("mouse")
            self.mouse.activate()

    def _activate_keyboard(self):
        if self.current_mode == "mouse":
            self.mouse.deactivate()
        self.signals.show_keyboard.emit()
        self._set_mode("keyboard")
        print("⌨️ Keyboard Mode Active")

    def _deactivate_keyboard(self):
        self.signals.hide_keyboard.emit()
        self._set_mode("standby")
        print("⌨️ Keyboard Closed")

    def _set_mode(self, mode):
        self.current_mode = mode
        self.signals.update_status.emit(mode)