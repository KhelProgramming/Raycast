import pyautogui
import time
from gesture_system.config import CLICK_COOLDOWN

class MouseClick:
    def __init__(self):
        self.last_left = 0
        self.last_right = 0
        self.cooldown = CLICK_COOLDOWN

    def left(self):
        if time.time() - self.last_left >= self.cooldown:
            pyautogui.click()
            self.last_left = time.time()
            print("🖱️  LEFT CLICK")

    def right(self):
        if time.time() - self.last_right >= self.cooldown:
            pyautogui.rightClick()
            self.last_right = time.time()
            print("🖱️  RIGHT CLICK")