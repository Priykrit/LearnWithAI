
import cv2
import numpy as np
import dlib

# cap = cv2.VideoCapture(0)


# def POINTS(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)

# cv2.namedWindow("Frame")
# cv2.setMouseCallback("Frame", POINTS)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# while True:
#     _, frame = cap.read()
    

#     key = cv2.waitKey(1)
#     if key == 27:
#         break
def side_face_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.flip(gray,1)
    frame = cv2.flip(frame,1)
    faces = detector(gray)
    for face in faces:
        # x1 = face.left()
        # y1 = face.top()
        # x2 = face.right()
        # y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        # for n in range(0, 68):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        pointx1 = landmarks.part(1).x
        pointx29 = landmarks.part(29).x
        pointx15 = landmarks.part(15).x
        pointx2 = landmarks.part(2).x
        pointx30 = landmarks.part(30).x
        pointx14 = landmarks.part(14).x
        pointx3 = landmarks.part(3).x
        pointx33 = landmarks.part(33).x
        pointx13 = landmarks.part(13).x
        pointy1 = landmarks.part(1).y
        pointy29 = landmarks.part(29).y
        pointy15 = landmarks.part(15).y
        pointy2 = landmarks.part(2).y
        pointy30 = landmarks.part(30).y
        pointy14 = landmarks.part(14).y
        pointy3 = landmarks.part(3).y
        pointy33 = landmarks.part(33).y
        pointy13 = landmarks.part(13).y
        if((pointx29-pointx1)>=3*(pointx15-pointx29)):
            cv2.putText(frame, "Plese look onto screen", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0),3)
        elif((pointx30-pointx2)>=3*(pointx14-pointx30)):
            cv2.putText(frame,  "Plese look onto screen", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0),3)
        elif((pointx33-pointx3)>=3*(pointx13-pointx33)):
            cv2.putText(frame,  "Plese look onto screen", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0),3)
        if(3*(pointx29-pointx1)<=(pointx15-pointx29)):
            cv2.putText(frame,  "Plese look onto screen", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0),3)
        elif(3*(pointx30-pointx2)<=(pointx14-pointx30)):
            cv2.putText(frame,  "Plese look onto screen", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0),3)
        elif(3*(pointx33-pointx3)<=(pointx13-pointx33)):
            cv2.putText(frame,  "Plese look onto screen", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0),3)
            
    return frame