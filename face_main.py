from face_det import RetinaFace
from face_emb import ArcFace
import os
import time
from functools import partial

import cv2
import numpy as np
import onnxruntime as ort

from numpy import ndarray

from errors import errors

from face_emb.utils import norm_crop

ort.set_default_logger_severity(3)


class Verification:
    def __init__(
            self,
            arc_face,
            retina_face,
            ref_img_path: str = None,
            sim_threshold: float = 0.35
    ):

        self.ref_img_path = ref_img_path
        self.ref_img = None
        self.embed_ref = None
        self.face_ref = None

        # self.usr_img_path = None
        # self.usr_img = None
        # self.usr_frames_np = None

        self.sim_threshold = sim_threshold
        self.sim = 0

        self.flag_verification = None
        self.flag_existence = None

        # Controlling flag
        self.has_error = False

        if not self.has_error:
            try:
                self.arc_face = arc_face
                print("Embedding loaded.")
            except:
                self.make_error(
                    error=errors.ERROR_1005_UNEXPECTED_ERROR,
                    sources="initializing ArcFace embedding.",
                )

        # Loading face detector model
        if not self.has_error:
            try:
                self.face_dm = retina_face
                print("Detector loaded.")
            except:
                self.make_error(
                    error=errors.ERROR_1005_UNEXPECTED_ERROR,
                    sources="initializing RetinaFace detector.",
                )

    def read_image(self, image_path: str):
        """
        Reads an image from the given path and stores it as `ref_img`.

        Args:
            image_path (str): Path to the image file.

        Returns:
            bool: True if successful, False otherwise.
        """
        self.check_path(image_path, p_type="reference")
        if not self.has_error:
            try:
                image = cv2.imread(image_path)
                if image is None or len(np.shape(image)) != 3:
                    self.make_error(
                        error=errors.ERROR_1003_UNEXPECTED_PARAMETER_ERROR,
                        sources="reading image",
                    )
                    return None
                self.ref_img = image
                print(f"Image loaded successfully from {image_path}.")
                return image
            except:
                self.make_error(
                    error=errors.ERROR_1005_UNEXPECTED_ERROR,
                    sources="reading image",
                )
                return None
        return None

    def make_error(self, error, sources: str):
        self.has_error = True
        print(f"Error: {error[0].format(sources=sources)}")

    def check_path(self, path: str, p_type: str = "user"):
        """
        Check if the provided video path is valid.

        Raises:
            ValueError: If the video path is invalid.
        """
        if path is not None:
            # Check path is valid and found.
            if not os.path.exists(path):
                self.make_error(
                    error=errors.ERROR_1001_SOURCE_NOT_FOUND_ERROR,
                    sources=f"{p_type} image/video path",
                )
            else:
                print(f"{p_type} path {path} exist.")

            # Check path contains a file
            if not os.path.isfile(path):
                self.make_error(
                    error=errors.ERROR_1002_FILE_NOT_FOUND_ERROR,
                    sources=f"{p_type} image/video path",
                )
            else:
                print(
                    f"{p_type} file {os.path.basename(path)} found in {os.path.dirname(path)}."
                )

        else:
            self.make_error(
                error=errors.ERROR_1006_PARAMETER_NOT_GIVEN,
                sources=f"{p_type} image/video path",
            )

    def single_frame_verification(self, usr_img_path: str = None):

        # Check user image path and file
        self.usr_img_path = usr_img_path
        self.check_path(self.usr_img_path, p_type="user")

        # Read user image
        if not self.has_error:
            try:
                usr_img = cv2.imread(self.usr_img_path)
                print("User image read successfully.")

                # Check reference image array is RGB
                if len(np.shape(usr_img)) == 3:
                    self.usr_img = usr_img
                else:
                    self.make_error(
                        error=errors.ERROR_1003_UNEXPECTED_PARAMETER_ERROR,
                        sources="user image shape",
                    )

            except:
                self.make_error(
                    error=errors.ERROR_1005_UNEXPECTED_ERROR,
                    sources="reading user image",
                )

        # single frame verification
        if not self.has_error:
            try:
                face_usr, bbox_usr = self.detection(self.usr_img)

                if len(bbox_usr) != 0:
                    print("Face detected in user image.")
                    self.flag_existence = True

                    try:
                        embed_usr = self.arc_face.get_feat(face_usr)
                        if embed_usr.shape[1] == 512:
                            print(
                                "User embedding calculated successfully."
                            )

                            try:
                                self.sim = self.arc_face.compute_sim(
                                    self.embed_ref, embed_usr
                                )

                                if self.sim >= self.sim_threshold:
                                    self.flag_verification = True
                                    print("Similarity is over threshold.")
                                else:
                                    self.flag_verification = False
                                    print("Similarity is under threshold.")

                            except:
                                self.make_error(
                                    error=errors.ERROR_1005_UNEXPECTED_ERROR,
                                    sources="computing similarity",
                                )

                        else:
                            self.make_error(
                                error=errors.ERROR_1003_UNEXPECTED_PARAMETER_ERROR,
                                sources="user embedding shape",
                            )
                    except:
                        self.make_error(
                            error=errors.ERROR_1005_UNEXPECTED_ERROR,
                            sources="user embedding calculation",
                        )

                else:
                    self.make_error(
                        error=errors.ERROR_1005_UNEXPECTED_ERROR,
                        sources="no face detected in user image",
                    )

            except:
                self.make_error(
                    error=errors.ERROR_1005_UNEXPECTED_ERROR,
                    sources="single frame verification",
                )

        return True

    def calculate_similarity(self, img_path1: str, img_path2: str):
        """
        Calculate similarity between two images by detecting faces, computing embeddings,
        and comparing the embeddings.

        Args:
            img_path1 (str): Path to the first image.
            img_path2 (str): Path to the second image.

        Returns:
            float: Similarity score between the two embeddings.
        """
        # Read the first image
        img1 = self.read_image(img_path1)
        if img1 is None or self.has_error:
            print("Error loading the first image.")
            return None

        # Detect face in the first image
        face1, bbox1 = self.detection(img1)
        if len(bbox1) == 0 or self.has_error:
            print("No face detected in the first image.")
            return None

        # Compute embedding for the first face
        emb1 = self.calculate_embedding(face1)
        if emb1 is None or self.has_error:
            print("Error calculating embedding for the first image.")
            return None

        # Read the second image
        img2 = self.read_image(img_path2)
        if img2 is None or self.has_error:
            print("Error loading the second image.")
            return None

        # Detect face in the second image
        face2, bbox2 = self.detection(img2)
        if len(bbox2) == 0 or self.has_error:
            print("No face detected in the second image.")
            return None

        # Compute embedding for the second face
        emb2 = self.calculate_embedding(face2)
        if emb2 is None or self.has_error:
            print("Error calculating embedding for the second image.")
            return None

        # Compute similarity between the two embeddings
        try:
            similarity = self.arc_face.compute_sim(emb1, emb2)
            print(f"Similarity score: {similarity}")
            return similarity
        except:
            self.make_error(
                error=errors.ERROR_1005_UNEXPECTED_ERROR,
                sources="computing similarity between embeddings",
            )
            return None

    def detection(self, image):
        boxes, points = self.face_dm.detect(image, max_num=1)
        bbox = np.empty((1, 4))
        face = []

        if len(boxes) != 0:
            for idx in range(boxes.shape[0]):
                bbox = boxes[idx, 0:4]

                kps = None
                if points is not None:
                    kps = points[idx]

                face = norm_crop(image, kps)

            return face, bbox
        else:
            return face, np.empty((0, 4))

    def calculate_embedding(self, face_image: ndarray):
        """
        Calculate the embedding for the provided face image.

        Args:
            face_image (ndarray): Cropped face image as a NumPy array.

        Returns:
            ndarray: The embedding vector of the face image.
        """
        if face_image is None or len(np.shape(face_image)) != 3:
            self.make_error(
                error=errors.ERROR_1006_PARAMETER_NOT_GIVEN,
                sources="face image",
            )
            return None

        try:
            embedding = self.arc_face.get_feat(face_image)
            if embedding.shape[1] == 512:
                print("Embedding calculated successfully.")
                return embedding
            else:
                self.make_error(
                    error=errors.ERROR_1003_UNEXPECTED_PARAMETER_ERROR,
                    sources="embedding shape",
                )
        except:
            self.make_error(
                error=errors.ERROR_1005_UNEXPECTED_ERROR,
                sources="embedding calculation",
            )

        return None

    def get_result(self):
        return {
            "flag_existence": self.flag_existence,
            "flag_verification": self.flag_verification,
            "similarity": self.sim,
            "has_error": self.has_error,
        }

if __name__ == "__main__":

    image_path = "./file/test1.jpg"
    arc_face = ArcFace(
            model_file="./trained_model/embedding.onnx"
        )
    retina_face = RetinaFace(
        model_file="./trained_model/detector.onnx",
        nms_thresh=0.4,
        det_thresh=0.5,
        input_size=(640, 480),
    )

    verif = Verification(
        arc_face=arc_face,
        retina_face=retina_face,
        ref_img_path=image_path,
    )

    img = verif.read_image(image_path)
    face, bbox = verif.detection(img)
    emb = verif.calculate_embedding(face)
    print(emb)
    if face is not None:
        cv2.imshow("Detected Face", face)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

    sim = verif.calculate_similarity(image_path, image_path)
    print(sim)

