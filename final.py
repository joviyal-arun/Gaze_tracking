from memory_profiler import profile
import datetime
import logging
import pathlib
from typing import Optional
import FLM_Module
import cv2
import numpy as np
import yacs.config
import tensorflow as tf
from tensorflow import keras
from face_detection import FaceDetection
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from gaze_estimation import GazeEstimationMethod, GazeEstimator
from gaze_estimation.gaze_estimator.common import (Face, FacePartsName,
                                                   Visualizer)
from gaze_estimation.utils import load_config
import pdb
import os
import pandas as pd
import pickle
#------------------------FLM MODULE --------
import mediapipe as mp
import time
from gaze_estimation.gaze_estimator.common import Face
import numpy as np
#------------------------FLM MODULE --------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Demo:

    QUIT_KEYS = {27, ord('q')}

    def __init__(self, config: yacs.config.CfgNode):

        self.config = config
        self.gaze_estimator = GazeEstimator(config)
        self.visualizer = Visualizer(self.gaze_estimator.camera)

        self.cap = self._create_capture()
        self.output_dir = self._create_output_dir()
        self.writer = self._create_video_writer()

        self.stop = False
        self.show_bbox = self.config.demo.show_bbox
        self.show_head_pose = self.config.demo.show_head_pose
        self.show_landmarks = self.config.demo.show_landmarks
        self.show_normalized_image = self.config.demo.show_normalized_image
        self.show_template_model = self.config.demo.show_template_model
        self.m1=keras.models.load_model("high_quality_open_multiple_CNN_19_05_10epoch.h5")

        #--------------------- FLM MODULE ----------------
        self.staticMode = False
        self.maxFace =  2
        self.minDetectionCon = 0.5
        self.minTrackCon = 0.5
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=False,max_num_faces=3)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        #--------------------- FLM MODULE ----------------

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
        
    def run_image_processing(self):

        index=0

        total_file=os.listdir('video_input/frames')

        for single_image in total_file:
            
            frame =cv2.imread('video_input/frames/'+single_image)
            undistorted = cv2.undistort(
                frame, self.gaze_estimator.camera.camera_matrix,
                self.gaze_estimator.camera.dist_coefficients)

            self.visualizer.set_image(frame.copy())
            
            faces = self.gaze_estimator.detect_faces(undistorted)
            
            for face in faces:
                try:

                    self.gaze_estimator.estimate_gaze(undistorted, face)
                    self._draw_face_bbox(face)
                    self._draw_head_pose(face)
                    self._draw_landmarks(face)
                    self._draw_face_template_model(face)
                    self._draw_gaze_vector(face)
                    self._display_normalized_image(face)

                    start_point=(self.co_ordinates[0][0],self.co_ordinates[0][1])
                    end_point=(self.co_ordinates[1][0],self.co_ordinates[1][1])

                    cropped_face_image =frame[self.co_ordinates[0][1]:self.co_ordinates[1][1],self.co_ordinates[0][0]:self.co_ordinates[1][0]]

                    cropped_face_images = cropped_face_image.copy()

                    detector = FLM_Module.FaceLandMarks()
                    img, mask_faces = detector.findFaceLandmark_final(cropped_face_images)

                    facial_landmarks=mask_faces[0]

                    # Left eye crop:-

                    left_eye_start_x=facial_landmarks[30]
                    left_eye_end_x=facial_landmarks[133]
                                        
                    x,y=left_eye_start_x

                    width=left_eye_end_x[0]-x
                    height=left_eye_end_x[1]-y

                    delta_w = int(0.2 * width)
                    delta_h = int(0.9 * height)

                    # Increase the size of the bounding box
                    x -= delta_w
                    y -= delta_h
                    width += 2 * delta_w
                    height += 2 * delta_h

                    left_crop=cropped_face_image[y:y+height,x:x+width]
                    cv2.imwrite('left_crop.jpg',left_crop)

                    # Right Eye Crop:-

                    right_eye_start_x=facial_landmarks[441]
                    right_eye_end_x=facial_landmarks[255]

                    x,y=right_eye_start_x
                    width=right_eye_end_x[0]-x
                    height=right_eye_end_x[1]-y

                    delta_w = int(0.2 * width)
                    delta_h = int(0.7 * height)

                    x -= delta_w
                    y -= delta_h
                    width += 2 * delta_w
                    height += 2 * delta_h

                    right_crop=cropped_face_image[y:y+height,x:x+width]
                    # cv2.imwrite('output/right_eye_crop_'+str(index)+'.jpg', right_crop)

                    # Eye Blink Classification:-

                    left_eye_image='left_crop.jpg'
                    
                    # Use the trained model to make predictions on new images
                    new_image = keras.preprocessing.image.load_img(left_eye_image, target_size=(100, 100))
                    new_image_arr = keras.preprocessing.image.img_to_array(new_image)
                    new_image_arr /= 255.0
                    prediction = self.m1.predict(tf.expand_dims(new_image_arr, axis=0))
                    output = prediction[0][0]
                    if output<=0.5:
                        output_string='eyes_close'
                    else:
                        output_string='eyes_open'
                
                    thickness=1
                    fontScale=1
                    color=(255,255,255)

                    if self.yaw < 25 or self.yaw<-25:
                        if self.yaw_eg <12  and  self.yaw_eg>-1 or self.yaw_eg >-2  and self.yaw_eg < -11: 
                            cv2.putText(self.visualizer.image,"gazing else where", (0,80), cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                        else:
                            cv2.putText(self.visualizer.image,"gazing", (0,80), cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    

                    cv2.rectangle(self.visualizer.image, start_point, end_point, color, thickness)
                    cv2.putText(self.visualizer.image, output_string, (0, 60),cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    cv2.putText(self.visualizer.image, "eye_pitch:{:.2f},eye_yaw:{:.2f}".format(self.pitch_eg,self.yaw_eg), (0,20), cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    cv2.putText(self.visualizer.image, "pitch:{:.2f},roll:{:.2f},yaw:{:.2f}".format(self.pitch,self.roll,self.yaw), (0,40), cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    cv2.imshow('output',self.visualizer.image)
                    # cv2.imwrite('video_output/68_land_mark/'+str(index)+'.jpg',self.visualizer.image)

                except Exception as e:
                    print("Issue in For loop :- ",e)
                    pass
            index+=1

    @profile
    def run(self) -> None:
        
        index=0

        # output_dictionary={'file_name':[],'pitch':[],'roll':[],'yaw':[],'eye_pitch':[],'eye_yaw':[],'magnitude':[],'label':[],'predicted_label':[],'actual_label':[]}

        output_dictionary={'pitch':[],'roll':[],'yaw':[],'eye_pitch':[],'eye_yaw':[],'magnitude':[],'actual_label':[]}

        filename = 'linear_model.sav'
        load_model = pickle.load(open(filename, 'rb'))

        while True:

            if self.config.demo.display_on_screen:
                self._wait_key()
                if self.stop:
                    break

            ok, frame = self.cap.read()

            if not ok:
                break

            undistorted = cv2.undistort(
                frame, self.gaze_estimator.camera.camera_matrix,
                self.gaze_estimator.camera.dist_coefficients)

            self.visualizer.set_image(frame.copy())
            
            faces = self.gaze_estimator.detect_faces(undistorted)
            
            for face in faces:
                try:

                    self.gaze_estimator.estimate_gaze(undistorted, face)
                    self._draw_face_bbox(face)
                    self._draw_head_pose(face)
                    self._draw_landmarks(face)
                    self._draw_face_template_model(face)
                    self._draw_gaze_vector(face)
                    self._display_normalized_image(face)

                    start_point=(self.co_ordinates[0][0],self.co_ordinates[0][1])
                    end_point=(self.co_ordinates[1][0],self.co_ordinates[1][1])

                    cropped_face_image =frame[self.co_ordinates[0][1]:self.co_ordinates[1][1],self.co_ordinates[0][0]:self.co_ordinates[1][0]]

                    cropped_face_images = cropped_face_image.copy()
                    #--------------------- FLM MODULE ----------
                    # detector = FLM_Module.FaceLandMarks()
                    # img, mask_faces = detector.findFaceLandmark_final(cropped_face_images)
                    # facial_landmarks=mask_faces[0]
                    
                    self.imgRGB = cv2.cvtColor(cropped_face_images, cv2.COLOR_BGR2RGB)
                    self.results = self.faceMesh.process(self.imgRGB)
                    faces = []
                    if self.results.multi_face_landmarks:
                        for faceLms in self.results.multi_face_landmarks:
                            self.mpDraw.draw_landmarks(cropped_face_images, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                            face = []
                            for id, lm in enumerate(faceLms.landmark):
                                ih, iw, ic = cropped_face_images.shape
                                x, y = int(lm.x * iw), int(lm.y * ih)
                                face.append([x,y])
                            faces.append(face)


                    facial_landmarks=faces[0]
                    #--------------------- FLM MODULE ----------

                    

                    # Left eye crop:-

                    left_eye_start_x=facial_landmarks[30]
                    left_eye_end_x=facial_landmarks[133]
                                        
                    x,y=left_eye_start_x

                    width=left_eye_end_x[0]-x
                    height=left_eye_end_x[1]-y

                    delta_w = int(0.2 * width)
                    delta_h = int(0.9 * height)

                    # Increase the size of the bounding box
                    x -= delta_w
                    y -= delta_h
                    width += 2 * delta_w
                    height += 2 * delta_h

                    left_crop=cropped_face_image[y:y+height,x:x+width]
                    cv2.imwrite('left_crop.jpg',left_crop)

                    # Right Eye Crop:-

                    right_eye_start_x=facial_landmarks[441]
                    right_eye_end_x=facial_landmarks[255]

                    x,y=right_eye_start_x
                    width=right_eye_end_x[0]-x
                    height=right_eye_end_x[1]-y

                    delta_w = int(0.2 * width)
                    delta_h = int(0.7 * height)

                    x -= delta_w
                    y -= delta_h
                    width += 2 * delta_w
                    height += 2 * delta_h

                    right_crop=cropped_face_image[y:y+height,x:x+width]
                    # cv2.imwrite('output/right_eye_crop_'+str(index)+'.jpg', right_crop)

                    # Eye Blink Classification:-

                    left_eye_image='left_crop.jpg'
                    
                    # Use the trained model to make predictions on new images
                    new_image = keras.preprocessing.image.load_img(left_eye_image, target_size=(100, 100))
                    new_image_arr = keras.preprocessing.image.img_to_array(new_image)
                    new_image_arr /= 255.0
                    prediction = self.m1.predict(tf.expand_dims(new_image_arr, axis=0))
                    output = prediction[0][0]
                    if output<=0.5:
                        output_string='eyes_close'
                    else:
                        output_string='eyes_open'
                
                    thickness=1
                    fontScale=1
                    color=(255,255,255)


                    output_dictionary['pitch'].append(self.pitch)
                    output_dictionary['roll'].append(self.roll)
                    output_dictionary['yaw'].append(self.yaw)

                    output_dictionary['eye_pitch'].append(self.pitch_eg)
                    output_dictionary['eye_yaw'].append(self.yaw_eg)

                    # output_label='head_straight_eye_top_30'
                    # output_dictionary['label'].append(output_label)

                    actual_label='gazing_else_where'
                    output_dictionary['actual_label'].append(actual_label)

                    # file_name=str(index)+'.jpg'
                    # output_dictionary['file_name'].append(file_name)

                    # Classification 
                    single_prediction=np.array([[self.pitch,self.roll,self.yaw,self.pitch_eg,self.yaw_eg]])
                    y_pred = load_model.predict(single_prediction)[0]


                    cv2.putText(self.visualizer.image, y_pred, (0, 100),cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    cv2.rectangle(self.visualizer.image, start_point, end_point, color, thickness)
                    cv2.putText(self.visualizer.image, output_string, (0, 60),cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    cv2.putText(self.visualizer.image, "eye_pitch:{:.2f},eye_yaw:{:.2f}".format(self.pitch_eg,self.yaw_eg), (0,20), cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    cv2.putText(self.visualizer.image, "pitch:{:.2f},roll:{:.2f},yaw:{:.2f}".format(self.pitch,self.roll,self.yaw), (0,40), cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    
                    Threshold_value=12
                    distance=round(self.visualizer.distance)
                    output_dictionary['magnitude'].append(distance)



                    
                    # if distance>=0 and distance<=Threshold_value:
                    #     output_label ='gazing'
                    #     output_dictionary['predicted_label'].append(output_label)
                    #     cv2.putText(self.visualizer.image,output_label, (0,120), cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    # else:
                    #     output_label ='gazing else where'
                    #     output_dictionary['predicted_label'].append(output_label)
                    #     cv2.putText(self.visualizer.image,output_label, (0,120), cv2.FONT_HERSHEY_PLAIN, fontScale, color, thickness)
                    

                    # NAVEEN commenting imshow for testing purpose
                    cv2.imshow('output',self.visualizer.image)
                    #print("FRAME COMPLETED")
                    #db.set_trace()
                    # NAVEEN commenting imshow for testing purpose


                    # cv2.imwrite('output/images/frame_by_frame_gaze_else_where/'+str(index)+'.jpg',self.visualizer.image) 

                except Exception as e:
                    print('error is',e)
                    pass
            index+=1
            #break
            if self.config.demo.use_camera:
                self.visualizer.image = self.visualizer.image[:, ::-1]
            # if self.writer:
            #     self.writer.write(self.visualizer.image)
            # if self.config.demo.display_on_screen:
            #     thickness=1
            #     fontScale=1
            #     color=(255,255,255)
        self.cap.release()

        # df=pd.DataFrame(output_dictionary)
        # df.to_csv('output/csv/frame_by_frame_gaze_else_where.csv',index=True,header=True)
        if self.writer:
            self.writer.release()

    def _create_capture(self) -> cv2.VideoCapture:
        if self.config.demo.use_camera:
            cap = cv2.VideoCapture(0)
        elif self.config.demo.video_path:
            print('video path',self.config.demo.video_path)
            cap = cv2.VideoCapture(self.config.demo.video_path)
        else:
            raise ValueError
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.gaze_estimator.camera.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.gaze_estimator.camera.height)
        return cap

    def _create_output_dir(self) -> Optional[pathlib.Path]:
        if not self.config.demo.output_dir:
            return
        output_dir = pathlib.Path(self.config.demo.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @staticmethod
    def _create_timestamp() -> str:
        dt = datetime.datetime.now()
        return dt.strftime('%Y%m%d_%H%M%S')

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        if not self.output_dir:
            return None
        ext = self.config.demo.output_file_extension
        if ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        elif ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            raise ValueError
        output_path = self.output_dir / f'{self._create_timestamp()}.{ext}'
        writer = cv2.VideoWriter(output_path.as_posix(), fourcc, 30,
                                 (self.gaze_estimator.camera.width,
                                  self.gaze_estimator.camera.height))
        if writer is None:
            raise RuntimeError
        return writer

    def _wait_key(self) -> None:
        key = cv2.waitKey(self.config.demo.wait_time) & 0xff
        if key in self.QUIT_KEYS:
            self.stop = True
        elif key == ord('b'):
            self.show_bbox = not self.show_bbox
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('h'):
            self.show_head_pose = not self.show_head_pose
        elif key == ord('n'):
            self.show_normalized_image = not self.show_normalized_image
        elif key == ord('t'):
            self.show_template_model = not self.show_template_model

    def _draw_face_bbox(self, face: Face) -> None:
        if not self.show_bbox:
            return
        self.visualizer.draw_bbox(face.bbox)        
        self.co_ordinates=self.visualizer.bbox[0],self.visualizer.bbox[1]

    def _draw_head_pose(self, face: Face) -> None:
        if not self.show_head_pose:
            return
        # Draw the axes of the model coordinate system
        length = self.config.demo.head_pose_axis_length
        self.visualizer.draw_model_axes(face, length, lw=2)

        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        self.pitch, self.yaw, self.roll = face.change_coordinate_system(euler_angles)
        # logger.info(f'[head] pitch: {self.pitch:.2f}, yaw: {self.yaw:.2f}, '
        #             f'roll: {self.roll:.2f}, distance: {face.distance:.2f}')

    def _draw_landmarks(self, face: Face) -> None:
        if not self.show_landmarks:
            return
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face: Face) -> None:
        if not self.show_template_model:
            return
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _display_normalized_image(self, face: Face) -> None:
        if not self.config.demo.display_on_screen:
            return
        if not self.show_normalized_image:
            return
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            reye = face.reye.normalized_image
            leye = face.leye.normalized_image
            normalized = np.hstack([reye, leye])
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            normalized = face.normalized_image
        else: #eth xgaze
            normalized = face.normalized_image
            #raise ValueError
        if self.config.demo.use_camera:
            normalized = normalized[:, ::-1]
        cv2.imshow('normalized', normalized)

    def _draw_gaze_vector(self, face: Face) -> None:
        length = self.config.demo.gaze_visualization_length
        if self.config.mode == GazeEstimationMethod.MPIIGaze.name:
            for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                eye = getattr(face, key.name.lower())
                self.visualizer.draw_3d_line(
                    eye.center, eye.center + length * eye.gaze_vector)
                self.pitch_eg, self.yaw_eg = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                # logger.info(
                #     f'[{key.name.lower()}] pitch: {self.pitch_e:.2f}, yaw: {self.yaw_e:.2f}')
        elif self.config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            self.pitch_eg, self.yaw_eg = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            #logger.info(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        else: # eth xgaze

            self.visualizer.draw_3d_line(
                face.center, face.center + length * face.gaze_vector)
            
            # Draw line process:--

   


            self.pitch_eg, self.yaw_eg = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            #logger.info(f'[face] pitch: {self.pitch:.2f}, yaw: {self.yaw:.2f}')
            #raise ValueError


def main():
    config = load_config()
    demo = Demo(config)
    demo.run()
    # demo.run_image_processing()


if __name__ == '__main__':
    main()
