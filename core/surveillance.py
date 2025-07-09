import cv2
import threading
from threat_detector import ThreatAnalyzer
from face_db import FaceDatabase
from fire_analyzer import FireDetector

class SurveillanceSystem:
    def __init__(self):
        self.threat_analyzer = ThreatAnalyzer()
        self.face_db = FaceDatabase()
        self.fire_detector = FireDetector()
        self.alarm_status = False

    def process_frame(self, frame):
        # Parallel processing
        threads = [
            threading.Thread(target=self._process_faces, args=(frame.copy(),)),
            threading.Thread(target=self._process_threats, args=(frame.copy(),)),
            threading.Thread(target=self._process_fire, args=(frame.copy(),))
        ]
        [t.start() for t in threads]
        [t.join() for t in threads]
        
        return frame

    def _process_faces(self, frame):
        self.face_db.process(frame)

    def _process_threats(self, frame):
        self.threat_analyzer.detect(frame)

    def _process_fire(self, frame):
        if self.fire_detector.analyze(frame):
            self._trigger_alarm("FIRE")

    def _trigger_alarm(self, alert_type):
        import pygame
        pygame.mixer.init()
        pygame.mixer.Sound("alarm.wav").play()
        print(f"ALERT: {alert_type} detected!")

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    system = SurveillanceSystem()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = system.process_frame(frame)
        cv2.imshow("Surveillance", frame)
        
        if cv2.waitKey(1) == 27: break
    
    cap.release()
    cv2.destroyAllWindows()