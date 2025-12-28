import threading
from collections import deque
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush

class ParticleOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.particles = {"Left": deque(maxlen=50), "Right": deque(maxlen=50)}
        self.max_age = 30
        self.lock = threading.Lock()
        
        self.initUI()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_particles)
        self.timer.start(16)
    
    def initUI(self):
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.show()
    
    def add_particle(self, hand, x, y):
        with self.lock:
            self.particles[hand].append([x, y, 0])
    
    def update_particles(self):
        with self.lock:
            for hand in self.particles:
                for particle in self.particles[hand]:
                    particle[2] += 1
                while self.particles[hand] and self.particles[hand][0][2] >= self.max_age:
                    self.particles[hand].popleft()
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        with self.lock:
            for hand, particles in self.particles.items():
                for x, y, age in particles:
                    alpha = int(255 * (1 - age / self.max_age))
                    size = int(15 * (1 - age / self.max_age))
                    color = QColor(200, 100, 255, alpha)
                    painter.setBrush(QBrush(color))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(QPoint(x, y), size, size)