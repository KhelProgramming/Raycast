# =============================================================================
# CONFIGURATION
# Central source of truth for all constants, paths, and settings.
# =============================================================================

import os

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'samples')
TRAIN_FILE = os.path.join(DATA_DIR, 'gesture_train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'gesture_test.csv')

# --- CAMERA ---
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS = 30

# --- MOUSE SETTINGS ---
DPI_LEVELS = [400, 800, 1600, 3200]
CLICK_COOLDOWN = 0.3
HOLD_TOGGLE_TIME = 1.0

# --- Z-DEPTH CALIBRATION ---
Z_RANGES = {
    'very_close': -0.25,
    'close': -0.20,
    'medium': -0.08,
    'far': -0.04
}

# --- UI COLORS ---
COLORS = {
    'keyboard_active': '#4CFF88',
    'keyboard_idle': '#3A3A3A',
    'text': '#FFFFFF',
    'dpi_green': '#00FF00',
    'dpi_yellow': '#FFFF00',
    'dpi_orange': '#FFA500',
    'dpi_red': '#FF0000'
}

# --- GESTURES ---
GESTURE_ACTIONS = {
    'toggle': 'TOGGLE_MODE',
    'left_click': 'LEFT_CLICK',
    'right_click': 'RIGHT_CLICK',
    'hold': 'DRAG',
    'press': 'PRESS_KEY',
    'undo': 'UNDO',
    'redo': 'REDO',
    'idle': 'IDLE'
}