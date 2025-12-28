import cv2
import threading
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QPainterPath, QImage, QPixmap

class CameraPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.lock = threading.Lock()
        
        self.initUI()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(33)
    
    def initUI(self):
        screen = QApplication.primaryScreen().geometry()
        width, height = 320, 240
        x = screen.width() - width - 10
        y = 10
        
        self.setGeometry(x, y, width, height)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.show()
    
    def update_frame(self, frame):
        with self.lock:
            resized = cv2.resize(frame, (320, 240))
            self.current_frame = resized
    
    def update_display(self):
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 10, 10)
        painter.fillPath(path, QBrush(QColor(20, 20, 20, 230)))
        painter.setPen(QPen(QColor(255, 255, 255, 150), 3))
        painter.drawPath(path)
        
        with self.lock:
            if self.current_frame is not None:
                h, w, ch = self.current_frame.shape
                bytes_per_line = ch * w
                rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                painter.setClipPath(path)
                painter.drawPixmap(5, 5, 310, 230, pixmap)
        
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(10, 20, "CAMERA PREVIEW")