from PyQt5.QtCore import QObject, pyqtSignal

class SignalEmitter(QObject):
    update_particles = pyqtSignal(dict)
    show_keyboard = pyqtSignal()
    hide_keyboard = pyqtSignal()
    update_status = pyqtSignal(str)
    update_predictions = pyqtSignal(dict)
    update_dpi = pyqtSignal(int, str)
    exit_system = pyqtSignal()
    
    # NEW: Signal for FPS
    update_fps = pyqtSignal(int)