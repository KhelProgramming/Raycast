import sys
from PyQt5.QtWidgets import QApplication

# Import the core controller
from gesture_system.core.air_controller import AirController

# Import the UI/Helper classes 
# (These must exist in their folders, copied from your original code)
from gesture_system.ui.particle_overlay import ParticleOverlay
from gesture_system.ui.status_bar import StatusBar
from gesture_system.ui.camera_preview import CameraPreview
from gesture_system.control.keyboard.virtual_keyboard import VirtualKeyboard
from gesture_system.core.signals import SignalEmitter
from gesture_system.ui.exit_handler import ExitHandler

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Dependency Injection
    controller = AirController(
        app, 
        SignalEmitter, 
        ParticleOverlay, 
        StatusBar, 
        VirtualKeyboard, 
        CameraPreview
    )
    
    # Setup Exit Handler
    exit_handler = ExitHandler(controller)
    
    # Blast off
    controller.run()
    
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        controller.stop()
        print("\n⚠️  Stopped")

if __name__ == "__main__":
    main()