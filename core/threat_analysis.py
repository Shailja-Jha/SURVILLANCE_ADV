from ultralytics import YOLO
import onnxruntime as ort
import cv2
import numpy as np

class ThreatAnalyzer:
    def __init__(self):
        self.weapon_model = YOLO('models/yolov9_weapons.pt')
        self.violence_model = ort.InferenceSession('models/stgcn_fighting.onnx')

    def detect(self, frame):
        self._detect_weapons(frame)
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 0
            
        if self.frame_count % 10 == 0:
            self._detect_violence(frame)

    def _detect_weapons(self, frame):
        results = self.weapon_model(frame)[0]
        for box in results.boxes:
            if box.conf > 0.85:
                self._draw_alert(frame, box.xyxy[0], "WEAPON")

    def _detect_violence(self, frame):
        # ST-GCN requires pose estimation preprocessing
        # Simplified example - use actual pose keypoints in production
        dummy_input = np.random.rand(1, 3, 300, 17, 2).astype(np.float32)
        outputs = self.violence_model.run(None, {'input': dummy_input})
        if outputs[0][0][1] > 0.9:
            self._draw_alert(frame, None, "FIGHTING")

    def _draw_alert(self, frame, bbox, alert_type):
        cv2.putText(frame, f"ALERT: {alert_type}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if bbox is not None:
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)