import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)  # Change the parameter to a video file path if you want to process a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe face mesh
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks for left and right eyes
            left_eye_landmarks = []
            right_eye_landmarks = []

            for id, landmark in enumerate(face_landmarks.landmark):
                if id in [33, 133, 160, 144, 153, 145, 155, 159, 154]:
                    left_eye_landmarks.append([landmark.x, landmark.y, landmark.z])
                elif id in [362, 263, 467, 466, 388, 387, 386, 385, 384]:
                    right_eye_landmarks.append([landmark.x, landmark.y, landmark.z])

            # Perform gaze classification using the eye landmarks
            left_eye_features = np.array(left_eye_landmarks).flatten()
            right_eye_features = np.array(right_eye_landmarks).flatten()
            
            # Use your trained model to classify the gaze direction
            gaze_prediction = your_gaze_model.predict([left_eye_features, right_eye_features])
            gaze_label = your_gaze_labels[gaze_prediction]

            # Draw the gaze label on the frame
            cv2.putText(frame, gaze_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
