from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QPainterPath
from gesture_system.config import COLORS

class StatusBar(QWidget):
    def __init__(self):
        super().__init__()
        self.current_mode = "STANDBY"
        self.current_dpi = 800
        self.distance_label = "MEDIUM"
        self.fps = 0 # Store FPS value
        self.initUI()
    
    def initUI(self):
        screen = QApplication.primaryScreen().geometry()
        width, height = 350, 110
        x = (screen.width() - width) // 2
        y = 10
        
        self.setGeometry(x, y, width, height)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Mode Label
        self.mode_label = QLabel("MODE: STANDBY", self)
        self.mode_label.setGeometry(10, 20, 330, 25)
        self.mode_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setStyleSheet(f"color: {COLORS['text']};")
        
        # Gesture Label
        self.gesture_label = QLabel("", self)
        self.gesture_label.setGeometry(10, 45, 330, 20)
        self.gesture_label.setFont(QFont("Arial", 8))
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setStyleSheet("color: #aaaaaa;")
        
        # DPI Label
        self.dpi_label = QLabel("DPI: 800 | MEDIUM", self)
        self.dpi_label.setGeometry(10, 70, 330, 25)
        self.dpi_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.dpi_label.setAlignment(Qt.AlignCenter)
        self.dpi_label.setStyleSheet(f"color: {COLORS['dpi_yellow']};")

        # ✅ NEW: FPS Label (Top Right Corner)
        self.fps_label = QLabel("FPS: 0", self)
        self.fps_label.setGeometry(280, 5, 60, 15)
        self.fps_label.setFont(QFont("Arial", 7, QFont.Bold))
        self.fps_label.setAlignment(Qt.AlignRight)
        self.fps_label.setStyleSheet("color: #00FF00;")
        
        self.show()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 10, 10)
        painter.fillPath(path, QBrush(QColor(0, 0, 0, 180)))
        painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
        painter.drawPath(path)
    
    def update_mode(self, mode):
        self.current_mode = mode.upper()
        color_map = {"KEYBOARD": COLORS['keyboard_active'], "MOUSE": COLORS['dpi_orange'], "STANDBY": "#888888"}
        color = color_map.get(self.current_mode, "#888888")
        self.mode_label.setText(f"MODE: {self.current_mode}")
        self.mode_label.setStyleSheet(f"color: {color};")
    
    def update_gestures(self, predictions):
        text = " | ".join([f"{h}: {g.upper()}" for h, g in predictions.items() if g != "idle"])
        self.gesture_label.setText(text if text else "")
    
    def update_dpi(self, dpi, distance_label):
        self.current_dpi = dpi
        self.distance_label = distance_label
        
        if dpi <= 400: color = COLORS['dpi_green']
        elif dpi <= 800: color = COLORS['dpi_yellow']
        elif dpi <= 1600: color = COLORS['dpi_orange']
        else: color = COLORS['dpi_red']
        
        self.dpi_label.setText(f"DPI: {dpi} | {distance_label}")
        self.dpi_label.setStyleSheet(f"color: {color};")

    #  NEW: Update FPS text
    def update_fps_display(self, fps):
        self.fps = fps
        color = "#00FF00" if fps > 24 else "#FF0000"
        self.fps_label.setText(f"FPS: {fps}")
        self.fps_label.setStyleSheet(f"color: {color};")