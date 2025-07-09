import onnxruntime as ort
import numpy as np
from pose_estimator import PoseEstimator  # Requires OpenPose or MediaPipe

class ViolenceDetector:
    def __init__(self):
        self.session = ort.InferenceSession('models/stgcn_fighting.onnx')
        self.pose_estimator = PoseEstimator()
        self.threshold = 0.90

    def detect(self, frame):
        # 1. Extract human poses
        keypoints = self.pose_estimator.extract_poses(frame)
        
        if not keypoints:
            return False

        # 2. Preprocess for ST-GCN (input shape: [1, 3, 300, 18, 2])
        model_input = self._preprocess(keypoints)
        
        # 3. Run inference
        outputs = self.session.run(None, {'input': model_input})
        violence_prob = outputs[0][0][1]
        
        return violence_prob > self.threshold

    def _preprocess(self, keypoints):
        # Convert pose keypoints to ST-GCN input format
        # (Implementation depends on your pose estimator)
        return np.random.rand(1, 3, 300, 18, 2).astype(np.float32)  # Replace with actual preprocessing