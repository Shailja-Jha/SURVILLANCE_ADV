import insightface
import cv2
import os

class FaceDatabase:
    def __init__(self, db_path="known_faces"):
        self.model = insightface.app.FaceAnalysis()
        self.model.prepare(ctx_id=0)
        self.known_faces = self._load_known_faces(db_path)

    def _load_known_faces(self, path):
        faces = {}
        for img_file in os.listdir(path):
            img = cv2.imread(f"{path}/{img_file}")
            faces[img_file.split('.')[0]] = self.model.get(img)[0].embedding
        return faces

    def process(self, frame):
        faces = self.model.get(frame)
        for face in faces:
            identity = self._recognize(face)
            self._draw_result(frame, face.bbox, identity)

    def _recognize(self, face):
        for name, emb in self.known_faces.items():
            if self.model.compute_sim(face.embedding, emb) > 0.6:
                return name
        return "Unknown"

    def _draw_result(self, frame, bbox, name):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.putText(frame, name, (int(bbox[0]), int(bbox[1])-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)