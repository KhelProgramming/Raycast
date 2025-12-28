import time
import pyautogui
from .dpi import AdaptiveDPI
from .movement import MouseMovement
from .click import MouseClick
from .drag import MouseDrag

class MouseOrchestrator:
    def __init__(self):
        self.active = False
        
        # Sub-components
        self.dpi = AdaptiveDPI()
        self.movement = MouseMovement()
        self.clicker = MouseClick()
        self.dragger = MouseDrag()
        
        # Cooldowns
        self.last_shortcut_time = 0
        self.shortcut_cooldown = 0.5 # Reduced slightly for faster repeated undo/redo
        
        # Sticky Hold Logic
        self.last_hold_seen = 0
        self.hold_grace_period = 0.4

    def activate(self):
        if not self.active:
            self.active = True
            self.movement.reset()
            print("🖱️  MOUSE ORCHESTRATOR ACTIVATED")

    def deactivate(self):
        if self.active:
            self.active = False
            self.dragger.stop()
            self.movement.reset()
            print("🖱️  MOUSE ORCHESTRATOR DEACTIVATED")

    def process_hand(self, landmarks, z_distance):
        """Updates movement and DPI."""
        current_dpi = self.dpi.calculate(z_distance)
        
        if self.active:
            idx_x = landmarks.landmark[8].x 
            idx_y = landmarks.landmark[8].y
            self.movement.move(idx_x, idx_y, current_dpi)

        return current_dpi, self.dpi.get_label()

    def process_gesture(self, gesture_name):
        """
        Manages state persistence for Holds and triggers one-shot actions.
        """
        if not self.active:
            return

        cmd = gesture_name.lower()
        now = time.time()

        # --- 1. STICKY HOLD LOGIC ---
        if cmd == "hold":
            self.last_hold_seen = now
            self.dragger.start()
        else:
            if self.dragger.is_holding:
                if now - self.last_hold_seen > self.hold_grace_period:
                    self.dragger.stop()
                    print("🖱️  HOLD RELEASED (Timeout)")
            
        # --- 2. INSTANT ACTIONS (Clicks) ---
        if not self.dragger.is_holding:
            if cmd == "left_click":
                self.clicker.left()
            elif cmd == "right_click":
                self.clicker.right()

        # --- 3. SHORTCUTS (Undo/Redo) - WINDOWS ROBUST FIX ---
        if cmd == "undo":
            if now - self.last_shortcut_time > self.shortcut_cooldown:
                print(f"⚡ WINDOWS UNDO (Ctrl+Z)")
                
                # Robust Method: Explicit Key Down/Up
                pyautogui.keyDown('ctrl')
                pyautogui.press('z')
                pyautogui.keyUp('ctrl')
                
                self.last_shortcut_time = now
                
        elif cmd == "redo":
            if now - self.last_shortcut_time > self.shortcut_cooldown:
                print(f"⚡ WINDOWS REDO (Ctrl+Y)")
                
                # Robust Method: Explicit Key Down/Up
                pyautogui.keyDown('ctrl')
                pyautogui.press('y')
                pyautogui.keyUp('ctrl')
                
                # OPTIONAL: If Ctrl+Y doesn't work for your specific app, 
                # uncomment the lines below to try Ctrl+Shift+Z instead:
                # pyautogui.keyDown('ctrl')
                # pyautogui.keyDown('shift')
                # pyautogui.press('z')
                # pyautogui.keyUp('shift')
                # pyautogui.keyUp('ctrl')
                
                self.last_shortcut_time = now