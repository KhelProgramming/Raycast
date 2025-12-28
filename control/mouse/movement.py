import pyautogui
import time
from collections import deque

class MouseMovement:
    def __init__(self):       
        pyautogui.FAILSAFE = False
        
        self.prev_x = None
        self.prev_y = None
        self.smoothing = deque(maxlen=3)
        self.last_seen = 0
        self.timeout = 0.5
        
        # Get screen size once
        self.screen_w, self.screen_h = pyautogui.size()

    def move(self, x, y, dpi):
        """
        Calculates relative movement based on DPI and moves cursor.
        x, y: Normalized coordinates (0.0 to 1.0) or raw camera coords depending on logic.
        Current logic assumes these are incoming relative positions.
        """
        now = time.time()
        
        # Reset if hand lost for too long
        if now - self.last_seen > self.timeout:
            self.prev_x, self.prev_y = None, None
            self.smoothing.clear()

        if self.prev_x is not None:
            # Calculate delta
            dx = (x - self.prev_x) * dpi
            dy = (y - self.prev_y) * dpi
            
            # Smooth
            self.smoothing.append((dx, dy))
            avg_dx = sum(d[0] for d in self.smoothing) / len(self.smoothing)
            avg_dy = sum(d[1] for d in self.smoothing) / len(self.smoothing)
            
            # Apply movement if significant
            if abs(avg_dx) > 1 or abs(avg_dy) > 1:
                pyautogui.moveRel(int(avg_dx), int(avg_dy), _pause=False)
        
        self.prev_x, self.prev_y = x, y
        self.last_seen = now
        
    def reset(self):
        self.prev_x = None
        self.prev_y = None
        self.smoothing.clear()