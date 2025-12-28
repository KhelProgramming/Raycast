import cv2
from gesture_system.config import CAMERA_WIDTH, CAMERA_HEIGHT

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        if not self.cap.isOpened():
            print("❌ Camera failed to initialize")
        else:
            print("📷 Camera initialized")

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip immediately for mirror view
            return True, cv2.flip(frame, 1)
        return False, None

    def release(self):
        self.cap.release()