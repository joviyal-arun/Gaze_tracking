import cv2
import mediapipe as mp
import time
from gaze_estimation.gaze_estimator.common import Face
import numpy as np

NUM_FACE = 2


class FaceLandMarks:


    def __init__(self, staticMode=False,maxFace=NUM_FACE, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = False
        self.maxFace =  maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        #self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,max_num_faces=self.maxFace,min_detection_confidence=self.minDetectionCon,min_tracking_confidence=self.minTrackCon)
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=False,max_num_faces=3)
        # self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceLandmark(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        detected = []
        h, w = img.shape[:2]
        if self.results.multi_face_landmarks:
            # for faceLms in self.results.multi_face_landmarks:
            #     if draw:
            #         self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)

            #     face = []
            #     for id, lm in enumerate(faceLms.landmark):
            #         ih, iw, ic = img.shape
            #         x, y = int(lm.x * iw), int(lm.y * ih)
            #         face.append([x,y])
            #     faces.append(face)

            for prediction in self.results.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                detected.append(Face(bbox, pts))
                
            return detected

    def findFaceLandmark_final(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)

        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x,y])
                faces.append(face)
                
        return img, faces



# def main():
#     cap = cv2.VideoCapture(0)
#     pTime = 0
#     detector = FaceLandMarks()
#     while True:
#         success, img = cap.read()
#         img, faces = detector.findFaceLandmark(img)
#         if len(faces)!=0:
#             print(len(faces))
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime
#         cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.imshow("Test", img)
#         cv2.waitKey(1)


# if __name__ == "__main__":
#     main()