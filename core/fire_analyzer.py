import torch
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet
from pygame import mixer

class FireDetector:
    def __init__(self):
        # Initialize EfficientNet-B7
        self.model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2)
        self.model.load_state_dict(torch.load('models/efficientnet_b7.pth'))
        self.model.eval()
        
        # Preprocessing parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Alarm system
        mixer.init()
        self.alarm = mixer.Sound('alarms/fire_alarm.wav')
        self.alarm_playing = False

    def analyze(self, frame):
        # Preprocess frame
        input_tensor = self._preprocess(frame)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            fire_prob = torch.nn.functional.softmax(outputs, dim=1)[0][1].item()
        
        # Trigger alarm if fire detected
        if fire_prob > 0.92:  # 92% confidence threshold
            self._trigger_alarm(frame, fire_prob)
            return True
        elif self.alarm_playing:
            self.alarm.stop()
            self.alarm_playing = False
        return False

    def _preprocess(self, frame):
        # Convert to RGB and resize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (600, 600))  # EfficientNet-B7 input size
        
        # Normalize and convert to tensor
        img = (img / 255.0 - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        return torch.FloatTensor(img).unsqueeze(0)

    def _trigger_alarm(self, frame, prob):
        # Visual alert
        cv2.putText(frame, f"FIRE DETECTED: {prob:.2%}", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Audio alert (non-blocking)
        if not self.alarm_playing:
            self.alarm.play(loops=-1)  # Loop until stopped
            self.alarm_playing = True

    def cleanup(self):
        if self.alarm_playing:
            self.alarm.stop()