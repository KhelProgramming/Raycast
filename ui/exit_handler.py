from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtCore import Qt as QtKey
from PyQt5.QtCore import Qt

class ExitHandler(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setGeometry(0, 0, 150, 80)
        self.setWindowTitle("Exit: Press Q/ESC")
        self.show()
        self.activateWindow()
        self.setFocus()
    
    def keyPressEvent(self, event):
        if event.key() == QtKey.Key_Q:
            print("\n👋 Q pressed - Exiting...")
            self.controller.stop()
        elif event.key() == QtKey.Key_Escape:
            print("\n👋 ESC pressed - Exiting...")
            self.controller.stop()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(0, 0, self.width(), self.height(), QColor(50, 50, 50))
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(10, 30, "Press Q or ESC")
        painter.drawText(10, 50, "to exit")