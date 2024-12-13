import os
import cv2
import numpy as np
import onnxruntime as ort
from numpy import ndarray
from functools import partial
from face_det import RetinaFace
from face_emb import ArcFace
from face_emb.utils import norm_crop
from errors import errors

ort.set_default_logger_severity(3)

class Verification:
    def __init__(self, arc_face, retina_face):
        self.has_error = False

        self.arc_face = self._initialize_model(arc_face, "ArcFace embedding")
        self.face_dm = self._initialize_model(retina_face, "RetinaFace detector")

    def _initialize_model(self, model, model_name):
        if self.has_error:
            return None
        try:
            print(f"{model_name} loaded.")
            return model
        except Exception as e:
            self.make_error(errors.ERROR_1005_UNEXPECTED_ERROR, f"initializing {model_name}")
            print(e)

    def read_image(self, image_path: str):
        self.check_path(image_path, p_type="reference")
        if self.has_error:
            return None

        try:
            image = cv2.imread(image_path)
            if image is None or image.ndim != 3:
                self.make_error(errors.ERROR_1003_UNEXPECTED_PARAMETER_ERROR, "reading image")
                return None

            print(f"Image loaded successfully from {image_path}.")
            return image
        except Exception as e:
            self.make_error(errors.ERROR_1005_UNEXPECTED_ERROR, "reading image")
            print(e)
            return None

    def make_error(self, error, sources: str):
        self.has_error = True
        print(f"Error: {error[0].format(sources=sources)}")

    def check_path(self, path: str, p_type: str = "user"):
        if not path:
            self.make_error(errors.ERROR_1006_PARAMETER_NOT_GIVEN, f"{p_type} image/video path")
            return

        if not os.path.exists(path):
            self.make_error(errors.ERROR_1001_SOURCE_NOT_FOUND_ERROR, f"{p_type} image/video path")
        else:
            print(f"{p_type} path {path} exists.")

        if not os.path.isfile(path):
            self.make_error(errors.ERROR_1002_FILE_NOT_FOUND_ERROR, f"{p_type} image/video path")
        else:
            print(f"{p_type} file {os.path.basename(path)} found in {os.path.dirname(path)}.")

    def calculate_similarity(self, img_path1: str, img_path2: str):
        img1 = self.read_image(img_path1)
        if self.has_error:
            print("Error loading the first image.")
            return None

        face1, bbox1 = self.detection(img1)
        if not bbox1.size or self.has_error:
            print("No face detected in the first image.")
            return None

        emb1 = self.calculate_embedding(face1)
        if emb1 is None or self.has_error:
            print("Error calculating embedding for the first image.")
            return None

        img2 = self.read_image(img_path2)
        if self.has_error:
            print("Error loading the second image.")
            return None

        face2, bbox2 = self.detection(img2)
        if not bbox2.size or self.has_error:
            print("No face detected in the second image.")
            return None

        emb2 = self.calculate_embedding(face2)
        if emb2 is None or self.has_error:
            print("Error calculating embedding for the second image.")
            return None

        try:
            similarity = self.arc_face.compute_sim(emb1, emb2)
            print(f"Similarity score: {similarity}")
            return similarity
        except Exception:
            self.make_error(errors.ERROR_1005_UNEXPECTED_ERROR, "computing similarity between embeddings")
            return None

    def detection(self, image):
        boxes, points = self.face_dm.detect(image, max_num=1)
        if not boxes.size:
            return [], np.empty((0, 4))

        bbox = boxes[0, :4]
        kps = points[0] if points is not None else None
        face = norm_crop(image, kps)

        return face, bbox

    def calculate_embedding(self, face_image: ndarray):
        if face_image is None or face_image.ndim != 3:
            self.make_error(errors.ERROR_1006_PARAMETER_NOT_GIVEN, "face image")
            return None

        try:
            embedding = self.arc_face.get_feat(face_image)
            if embedding.shape[1] == 512:
                print("Embedding calculated successfully.")
                return embedding
            else:
                self.make_error(errors.ERROR_1003_UNEXPECTED_PARAMETER_ERROR, "embedding shape")
        except Exception:
            self.make_error(errors.ERROR_1005_UNEXPECTED_ERROR, "embedding calculation")

        return None

if __name__ == "__main__":
    image_path = "./file/test1.jpg"
    arc_face = ArcFace(model_file="./trained_model/embedding.onnx")
    retina_face = RetinaFace(
        model_file="./trained_model/detector.onnx",
        nms_thresh=0.4,
        det_thresh=0.5,
        input_size=(640, 480),
    )

    verifier = Verification(arc_face=arc_face, retina_face=retina_face)

    img = verifier.read_image(image_path)
    if img is not None:
        face, bbox = verifier.detection(img)
        emb = verifier.calculate_embedding(face)
        print(emb)

        if face is not None:
            cv2.imshow("Detected Face", face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        sim = verifier.calculate_similarity(image_path, image_path)
        print(sim)
