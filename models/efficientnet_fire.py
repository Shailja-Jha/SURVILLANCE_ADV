import torch
import cv2
import numpy as np
from efficientnet_pytorch import EfficientNet

class FireDetector:
    def __init__(self, device='cuda:0'):
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        self.model._fc = torch.nn.Linear(2560, 2)  # Modify last layer
        self.model.load_state_dict(torch.load('models/efficientnet_b7.pth'))
        self.model.to(device)
        self.model.eval()
        
        self.device = device
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def detect(self, frame):
        input_tensor = self._preprocess(frame)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs[0][1].item()  # Return fire probability

    def _preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (600, 600))
        img = (img / 255.0 - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).float().unsqueeze(0).to(self.device)