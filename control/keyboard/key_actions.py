import pyautogui

def press_key(key):
    """Executes the key press via PyAutoGUI."""
    if key == "BACK":
        pyautogui.press('backspace')
    elif key == "SPACE":
        pyautogui.press('space')
    else:
        pyautogui.write(key.lower())