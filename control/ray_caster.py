import numpy as np

class RayCaster:
    def __init__(self):
        # VR RAYCAST SETTINGS
        # Since we normalize the vector (length=1.0), sensitivity acts as "Reach Length"
        # Start with much lower values - you can tune from here
        self.sensitivity_x = 0.3  # Reduced from 1.8
        self.sensitivity_y = 0.3  # Reduced from 2.2
        
        # Smoothing (0.0 to 1.0)
        # 0.5 = Balanced smoothing
        self.alpha = 0.5
        
        self.prev_x = 0.5
        self.prev_y = 0.5

    def cast(self, landmarks):
        """
        Implements a VR-style Raycast:
        Origin = Wrist (Stable pivot)
        Direction = Wrist -> Index Knuckle (Stable angle, unaffected by clicking)
        """
        # Landmark 0 = Wrist (The Anchor)
        # Landmark 5 = Index Finger MCP (The Aiming Point - Knuckle)
        wrist = landmarks.landmark[0]
        target = landmarks.landmark[5] 
        
        # 1. Calculate Un-normalized Direction Vector
        # X: Standard (right = positive)
        # Y: INVERTED because MediaPipe Y increases downward, but we want upward pointing
        vec_x = target.x - wrist.x
        vec_y = wrist.y - target.y  # FLIPPED: Now pointing up gives positive values
        
        # 2. Normalize the Vector (Make length = 1.0)
        # This fixes the "finger curling" issue. Shortening the finger 
        # (by clicking) won't pull the cursor back, because we only care about ANGLE.
        magnitude = np.sqrt(vec_x**2 + vec_y**2)
        
        if magnitude > 0:
            dir_x = vec_x / magnitude
            dir_y = vec_y / magnitude
        else:
            dir_x, dir_y = 0, 0

        # 3. Project the Ray
        # Formula: ScreenPos = Origin + (Direction * Reach/Sensitivity)
        # We start at the wrist and extend out along the angle.
        raw_x = wrist.x + (dir_x * self.sensitivity_x)
        raw_y = wrist.y - (dir_y * self.sensitivity_y)  # SUBTRACT because MediaPipe Y is inverted
        
        # 4. Center Offset (Calibration)
        # Fine-tune vertical aiming to feel natural
        raw_y += 0.05  # Reduced from 0.2 - smaller adjustment
        
        # 5. Clamp to Screen Limits (0.0 to 1.0)
        target_x = max(0.0, min(1.0, raw_x))
        target_y = max(0.0, min(1.0, raw_y))
        
        # 6. Exponential Smoothing
        self.prev_x = (self.prev_x * self.alpha) + (target_x * (1 - self.alpha))
        self.prev_y = (self.prev_y * self.alpha) + (target_y * (1 - self.alpha))
        
        return self.prev_x, self.prev_y