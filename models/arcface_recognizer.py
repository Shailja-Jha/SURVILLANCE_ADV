import insightface
import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self, db_path='known_faces'):
        self.model = insightface.app.FaceAnalysis(name='arcface_r100_v1')
        self.model.prepare(ctx_id=0)
        self.face_db = self._load_face_db(db_path)

    def _load_face_db(self, path):
        return {
            person.split('.')[0]: self._get_embedding(cv2.imread(f"{path}/{person}"))
            for person in os.listdir(path)
        }

    def _get_embedding(self, image):
        faces = self.model.get(image)
        return faces[0].embedding if faces else None

    def recognize(self, frame):
        faces = self.model.get(frame)
        results = []
        
        for face in faces:
            identity = "Unknown"
            max_sim = 0
            
            for name, db_emb in self.face_db.items():
                sim = self.model.compute_sim(face.embedding, db_emb)
                if sim > 0.6 and sim > max_sim:
                    max_sim = sim
                    identity = name
            
            results.append({
                'bbox': face.bbox.astype(int),
                'identity': identity,
                'confidence': max_sim
            })
        
        return results