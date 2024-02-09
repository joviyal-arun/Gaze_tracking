from typing import List
import FLM_Module
import dlib
import numpy as np
import yacs.config
import pdb
from face_detection import FaceDetection
from ..common import Face
import cv2
import mediapipe

class LandmarkEstimator:

    def __init__(self, config: yacs.config.CfgNode):

        self.mode = config.face_detector.mode
        self.config = config
        self.detector = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=3,static_image_mode=False)

        if self.mode == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(config.face_detector.dlib.model)

        elif self.mode =='openvino':
            self.face_model_path=self.config.face_detector.openvino.model
            self.face_model = FaceDetection(model_name=self.face_model_path)
            self.face_model.load_model()
            self.face_model.check_model()

        elif self.mode =='medipipe':
            self.detector = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=3,static_image_mode=False)

        else:
            raise ValueError

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        
        if self.mode == 'dlib':
            return self._detect_faces_dlib(image)
        elif self.mode=='medipipe':
            return self._detect_faces_mediapipe(image)
        elif self.mode=='openvino':
            return self._detect_faces_openvino(image)

    

    def _detect_faces_openvino(self, image: np.ndarray) -> List[Face]:
        
        #self.detector = mediapipe.solutions.face_mesh.FaceMesh(max_num_faces=3,static_image_mode=False)
        
        facial_land_mark = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                        296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                        380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        detected = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                facial_landmarks_68 = np.zeros((68, 2), dtype=np.float64)
                for index,data in enumerate(facial_land_mark):
                    output_list=[pts[data][0],pts[data][1]]
                    output=np.array([output_list], dtype=np.float64)
                    facial_landmarks_68[index]=output
                detected.append(Face(bbox, facial_landmarks_68))
        return detected

    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Face]:
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        detected = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                detected.append(Face(bbox, pts))
        return detected

    def _detect_faces_dlib(self, image: np.ndarray) -> List[Face]:
        
        bboxes = self.detector(image[:, :, ::-1], 1)
        detected = []

        for bbox in bboxes:
            predictions = self.predictor(image[:, :, ::-1], bbox)
            landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                 dtype=np.float64)
            bbox = np.array([[bbox.left(), bbox.top()],
                             [bbox.right(), bbox.bottom()]],
                            dtype=np.float64)
            detected.append(Face(bbox, landmarks))

        return detected
