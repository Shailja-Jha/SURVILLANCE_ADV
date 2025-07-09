import torch
from ultralytics import YOLO

# Test fire model
fire_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b7', pretrained=True)
fire_model.load_state_dict(torch.load('models/efficientnet_fire.pth'))
print("Fire model loaded successfully")

# Test weapon model
weapon_model = YOLO('models/yolov8n_weapons.pt')
print("Weapon model loaded successfully")