from collections import deque
from gesture_system.config import DPI_LEVELS, Z_RANGES

class AdaptiveDPI:
    def __init__(self):
        self.dpi_levels = DPI_LEVELS
        self.history = deque(maxlen=15)
        self.current_dpi = 800
        self.last_z = 0.0
        
        print("🎯 DPI Component initialized")

    def calculate(self, z_distance):
        """Calculates DPI based on hand Z-distance."""
        if z_distance < Z_RANGES['very_close']:
            target = 400
        elif z_distance < Z_RANGES['close']:
            target = 800
        elif z_distance < Z_RANGES['medium']:
            target = 1600
        else:
            target = 3200
        
        self.history.append(target)
        smoothed = sum(self.history) / len(self.history)
        
        self.current_dpi = int(smoothed)
        self.last_z = z_distance
        return self.current_dpi

    def get_label(self):
        """Returns readable label for UI."""
        if self.last_z < Z_RANGES['very_close']: return "VERY CLOSE"
        if self.last_z < Z_RANGES['close']: return "CLOSE"
        if self.last_z < Z_RANGES['medium']: return "MEDIUM"
        return "FAR"