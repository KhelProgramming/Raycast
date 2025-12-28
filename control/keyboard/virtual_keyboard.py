import time
import threading
from collections import deque, Counter
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF, QPoint
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QPainterPath

from gesture_system.config import COLORS
from .key_actions import press_key

class VirtualKeyboard(QWidget):
    def __init__(self):
        super().__init__()
        
        self.keys_layout = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BACK'],
            ['SPACE']
        ]
        
        self.key_multipliers = {'BACK': 1.5, 'SPACE': 8.0}
        
        # Colors
        self.bg_color = QColor(30, 30, 30, 240)
        self.key_idle_color = QColor(58, 58, 58)
        self.key_hover_color = QColor(120, 195, 255)
        self.key_active_color = QColor(76, 255, 136)
        self.text_color = QColor(255, 255, 255)
        self.border_color = QColor(255, 255, 255, 32)
        
        self.active = False
        self.key_rects = {}
        self.hovered_keys = {}
        self.pressed_keys = {}
        
        self.lock = threading.Lock()
        
        # Hover queue
        self.hover_queues = {"Left": deque(maxlen=7), "Right": deque(maxlen=7)}
        self.last_press_time = {"Left": 0, "Right": 0}
        self.press_cooldown = 0.5 
        
        self.initUI()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.check_hover_triggers)
        self.update_timer.start(30)
    
    def initUI(self):
        screen = QApplication.primaryScreen().geometry()
        width = int(screen.width() * 0.75)
        height = int(screen.height() * 0.35)
        x = (screen.width() - width) // 2
        y = screen.height() - height - 50
        
        self.setGeometry(x, y, width, height)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.compute_key_positions()
    
    def compute_key_positions(self):
        outer_margin = 20
        row_spacing = 12
        key_spacing = 10
        
        inner_width = self.width() - outer_margin * 2
        inner_height = self.height() - outer_margin * 2
        
        num_rows = len(self.keys_layout)
        row_height = (inner_height - row_spacing * (num_rows - 1)) // num_rows
        
        self.key_rects = {}
        
        for row_idx, row in enumerate(self.keys_layout):
            total_multipliers = sum([self.key_multipliers.get(key, 1.0) for key in row])
            total_spacing = key_spacing * (len(row) - 1)
            base_unit = (inner_width - total_spacing) / total_multipliers
            
            row_width = sum([base_unit * self.key_multipliers.get(key, 1.0) for key in row]) + total_spacing
            
            start_x = outer_margin + (inner_width - row_width) // 2
            start_y = outer_margin + row_idx * (row_height + row_spacing)
            
            current_x = int(start_x)
            for key in row:
                key_width = int(base_unit * self.key_multipliers.get(key, 1.0))
                self.key_rects[key] = QRect(int(current_x), int(start_y), int(key_width), int(row_height))
                current_x += key_width + key_spacing

    def activate(self):
        if not self.active:
            self.active = True
            self.show()
            print("⌨️  KEYBOARD ACTIVATED")
    
    def deactivate(self):
        if self.active:
            self.active = False
            self.hide()
            with self.lock:
                self.hovered_keys = {}
                self.pressed_keys = {}
            print("⌨️  KEYBOARD DEACTIVATED")
    
    def check_hover(self, hands_data):
        if not self.isVisible(): return {}
        
        current_time = time.time()
        new_hovers = {}
        
        with self.lock:
            for hand_label, hand_data in hands_data.items():
                if not hand_data["detected"]: continue
                
                screen_x, screen_y = hand_data["screen_pos"]
                gesture = hand_data["prediction"]
                
                widget_pos = self.mapFromGlobal(QPoint(screen_x, screen_y))
                
                # Check intersection
                hovered_key = None
                for key, rect in self.key_rects.items():
                    if rect.contains(widget_pos):
                        hovered_key = key
                        new_hovers[hand_label] = key
                        break
                
                self.hover_queues[hand_label].append(hovered_key)
                
                # Check Trigger
                if gesture.lower() == "press":
                    if current_time - self.last_press_time[hand_label] >= self.press_cooldown:
                        valid_keys = [k for k in self.hover_queues[hand_label] if k is not None]
                        if valid_keys:
                            most_common_key = Counter(valid_keys).most_common(1)[0][0]
                            
                            # Execute Action
                            press_key(most_common_key)
                            
                            self.last_press_time[hand_label] = current_time
                            key_id = f"{hand_label}_{most_common_key}"
                            self.pressed_keys[key_id] = current_time
                            self.hover_queues[hand_label].clear()
            
            self.hovered_keys = new_hovers
        
        self.update()
        return new_hovers
    
    def check_hover_triggers(self):
        current_time = time.time()
        with self.lock:
            for key_id in list(self.pressed_keys.keys()):
                if (current_time - self.pressed_keys[key_id]) > 0.3:
                    del self.pressed_keys[key_id]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # BG
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 20, 20)
        painter.fillPath(path, QBrush(self.bg_color))
        
        # Keys
        with self.lock:
            for key, rect in self.key_rects.items():
                key_pressed = any(k.endswith(f"_{key}") for k in self.pressed_keys)
                
                if key_pressed: color = self.key_active_color
                elif key in self.hovered_keys.values(): color = self.key_hover_color
                else: color = self.key_idle_color
                
                key_path = QPainterPath()
                rect_f = QRectF(rect)
                key_path.addRoundedRect(rect_f, 10, 10)
                
                painter.fillPath(key_path, QBrush(color))
                painter.setPen(QPen(self.border_color, 1))
                painter.drawPath(key_path)
                
                painter.setPen(QPen(self.text_color))
                painter.setFont(QFont("Arial", 14, QFont.Bold))
                painter.drawText(rect, Qt.AlignCenter, key)