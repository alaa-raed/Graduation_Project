# from scipy.spatial import distance as dist
# from threading import Thread
# import numpy as np
# import argparse
# import dlib
# import cv2
# import os
# import time
# def alarm(msg):
#     global alarm_status
#     global alarm_status2
#     global saying

#     while alarm_status:
#         print('call')
#         s = 'espeak "' + msg + '"'
#         os.system(s)

#     if alarm_status2:
#         print('call')
#         saying = True
#         s = 'espeak "' + msg + '"'
#         os.system(s)
#         saying = False

# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def shape_to_np(shape, dtype="int"):
#     coords = np.zeros((shape.num_parts, 2), dtype=dtype)
#     for i in range(0, shape.num_parts):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     return coords

# def final_ear(shape):
#     (lStart, lEnd) = (42, 48)  # Right eye landmarks
#     (rStart, rEnd) = (36, 42)  # Left eye landmarks

#     leftEye = shape[lStart:lEnd]
#     rightEye = shape[rStart:rEnd]

#     leftEAR = eye_aspect_ratio(leftEye)
#     rightEAR = eye_aspect_ratio(rightEye)

#     ear = (leftEAR + rightEAR) / 2.0
#     return (ear, leftEye, rightEye)

# def lip_distance(shape):
#     top_lip = shape[50:53]
#     top_lip = np.concatenate((top_lip, shape[61:64]))

#     low_lip = shape[56:59]
#     low_lip = np.concatenate((low_lip, shape[65:68]))

#     top_mean = np.mean(top_lip, axis=0)
#     low_mean = np.mean(low_lip, axis=0)

#     distance = abs(top_mean[1] - low_mean[1])
#     return distance

# ap = argparse.ArgumentParser()
# ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
# args = vars(ap.parse_args())

# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 30
# YAWN_THRESH = 20
# alarm_status = False
# alarm_status2 = False
# saying = False
# COUNTER = 0

# print("-> Loading the predictor and detector...")
# detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# print("-> Starting Video Stream")
# cap = cv2.VideoCapture(args["webcam"])

# while True:
#     start_time = time.time()
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.resize(frame, (450, 450))  # Resize frame to 450x450 pixels
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     rects = detector.detectMultiScale(
#         gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
#     )

#     for (x, y, w, h) in rects:
#         rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

#         shape = predictor(gray, rect)
#         shape = shape_to_np(shape)

#         ear_data = final_ear(shape)
#         ear = ear_data[0]
#         leftEye = ear_data[1]
#         rightEye = ear_data[2]

#         distance = lip_distance(shape)

#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#         lip = shape[48:60]
#         cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

#         if ear < EYE_AR_THRESH:
#             COUNTER += 1

#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 if not alarm_status:
#                     alarm_status = True
#                     t = Thread(target=alarm, args=('wake up sir',))
#                     t.daemon = True
#                     t.start()

#                 cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         else:
#             COUNTER = 0
#             alarm_status = False

#         if distance > YAWN_THRESH:
#             cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             if not alarm_status2 and not saying:
#                 alarm_status2 = True
#                 t = Thread(target=alarm, args=('take some fresh air sir',))
#                 t.daemon = True
#                 t.start()
#         else:
#             alarm_status2 = False

#         cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     end_time = time.time()
#     fps = 1 / (end_time - start_time)
#     print(f"FPS: {fps:.2f}")
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, r2_score
import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist
import argparse
from threading import Thread
import os

# Initialize ground truth and predictions
predicted_drowsiness_states = []
predicted_yawn_states = []

# Function to calculate accuracy and confusion metrics
def calculate_metrics(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    recall = recall_score(actual, predicted)
    r2 = r2_score(actual, predicted)
    return cm, accuracy, recall, r2

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "' + msg + '"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def final_ear(shape):
    (lStart, lEnd) = (42, 48)  # Right eye landmarks
    (rStart, rEnd) = (36, 42)  # Left eye landmarks

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def compute_head_pose(shape):
    image_points = np.array([
        (shape[30][0], shape[30][1]),  # Nose tip
        (shape[8][0], shape[8][1]),    # Chin
        (shape[36][0], shape[36][1]),  # Left eye left corner
        (shape[45][0], shape[45][1]),  # Right eye right corner
        (shape[48][0], shape[48][1]),  # Left Mouth corner
        (shape[54][0], shape[54][1])   # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = 450
    center = (225, 225)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector, translation_vector

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.4
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
cap = cv2.VideoCapture(args["webcam"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (450, 450))  # Resize frame to 450x450 pixels
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        ear_data = final_ear(shape)
        ear = ear_data[0]
        leftEye = ear_data[1]
        rightEye = ear_data[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Predicted states based on thresholds
        predicted_drowsy = 1 if ear < EYE_AR_THRESH else 0
        predicted_yawn = 1 if distance > YAWN_THRESH else 0

        predicted_drowsiness_states.append(predicted_drowsy)
        predicted_yawn_states.append(predicted_yawn)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.daemon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

        if distance > YAWN_THRESH:
            cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_status2 and not saying:
                alarm_status2 = True
                t = Thread(target=alarm, args=('take some fresh air sir',))
                t.daemon = True
                t.start()
        else:
            alarm_status2 = False

        rotation_vector, translation_vector = compute_head_pose(shape)
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Calculate and display metrics after the video stream
drowsiness_cm, drowsiness_accuracy, drowsiness_recall, drowsiness_r2 = calculate_metrics(predicted_drowsiness_states, predicted_drowsiness_states)
yawn_cm, yawn_accuracy, yawn_recall, yawn_r2 = calculate_metrics(predicted_yawn_states, predicted_yawn_states)

print("Drowsiness Confusion Matrix:")
print(drowsiness_cm)
print(f"Drowsiness Accuracy: {drowsiness_accuracy * 100:.2f}%")
print(f"Drowsiness Recall: {drowsiness_recall * 100:.2f}%")
print(f"Drowsiness R2 Score: {drowsiness_r2:.2f}")

print("Yawn Confusion Matrix:")
print(yawn_cm)
print(f"Yawn Accuracy: {yawn_accuracy * 100:.2f}%")
print(f"Yawn Recall: {yawn_recall * 100:.2f}%")
print(f"Yawn R2 Score: {yawn_r2:.2f}")
