
import cv2
import numpy as np
import dlib
from imutils import face_utils
import screen_brightness_control as pct

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

stressed_max = 0
stressed_moderate = 0
stress_min = 0
status = ""
color = (0, 0, 0)


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up/(2.0*down)

    if (ratio > 0.25):
        return 2
    elif (ratio > 0.21 and ratio <= 0.25):
        return 1
    else:
        return 0


def brightness_contoller(frame, brightness_list):
    stressed_max = brightness_list[0]
    stressed_moderate = brightness_list[1]
    stress_min = brightness_list[2]
    status = brightness_list[3]
    color = brightness_list[4]
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(
            landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(
            landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if (left_blink == 0 or right_blink == 0):
            stressed_max += 1
            stressed_moderate = 0
            stress_min = 0
            if (stressed_max > 6):
                status = "Stressed"
                pct.set_brightness(0)
            color = (255, 0, 0)

        elif (left_blink == 1 or right_blink == 1):
            stressed_max = 0
            stress_min = 0
            stressed_moderate += 1
            if (stressed_moderate > 6):
                status = "Moderate_stressed"
                pct.set_brightness(30)
            color = (0, 0, 255)

        else:
            stressed_moderate = 0
            stressed_max = 0
            stress_min += 1
            if (stress_min > 6):
                status = "Normal_stress"
                pct.set_brightness(70)
            color = (0, 255, 0)

       


        brightness_list[0] = stressed_max
        brightness_list[1] = stressed_moderate
        brightness_list[2] = stress_min
        brightness_list[3] = status
        brightness_list[4] = color

    return frame
