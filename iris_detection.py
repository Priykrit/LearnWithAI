import cv2 as cv
import numpy as np
import mediapipe as mp
import math

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

L_H_LEFT = [33]    
L_H_RIGHT = [133]   
R_H_LEFT = [362]    
R_H_RIGHT = [263]   


def euclidean_distance(point1, point2):
    x1, y1 =point1.ravel()
    x2, y2 =point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    iris_position =""
    if ratio <= 0.42:
        iris_position="right"
    elif ratio > 0.42 and ratio <= 0.57:
        iris_position="center"
    else:
        iris_position = "left"
    return iris_position, ratio


def iris_estimation(frame,iris_list):
    face_mesh = iris_list[0]
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)   
    img_h, img_w = frame.shape[:2]
    results = face_mesh.process(rgb_frame)
    iris_list[0]=face_mesh
    if results.multi_face_landmarks:
        mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx,r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

           
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
            
           


            

        iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
        left_eye_left = mesh_points[L_H_LEFT]
        left_eye_right= mesh_points[L_H_RIGHT]
        left_eye_left_x = left_eye_left[0][0]
        left_eye_left_y = left_eye_left[0][1]
        left_eye_right_x = left_eye_right[0][0]
        left_eye_right_y = left_eye_right[0][1]

        right_eye_left = mesh_points[R_H_LEFT]
        right_eye_right= mesh_points[R_H_RIGHT]
        right_eye_left_x = right_eye_left[0][0]
        right_eye_left_y = right_eye_left[0][1]
        right_eye_right_x = right_eye_right[0][0]
        right_eye_right_y = right_eye_right[0][1]



           
        if((left_eye_right_x-left_eye_left_x)>=3*(center_right[0]-left_eye_left_x)):
            cv.putText(frame, "Plese look onto screen", (100,100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0),3)
        if((right_eye_right_x-right_eye_left_x)>=3*(right_eye_right_x-center_left[0])):
            cv.putText(frame, "Plese look onto screen", (100,100), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0),3)
        iris_list[0]=face_mesh
        return frame
    else:
        return frame