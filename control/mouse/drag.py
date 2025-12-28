import pyautogui

class MouseDrag:
    def __init__(self):
        self.is_holding = False

    def start(self):
        if not self.is_holding:
            pyautogui.mouseDown()
            self.is_holding = True
            print("🖱️  HOLD START")

    def stop(self):
        if self.is_holding:
            pyautogui.mouseUp()
            self.is_holding = False
            print("🖱️  HOLD STOP")