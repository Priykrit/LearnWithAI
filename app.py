from flask import Flask,render_template,Response
import cv2
import side_face_detection as sfd
import Drowsyness as dwsy
from PIL import Image as im
app=Flask(__name__)
camera=cv2.VideoCapture(0)


sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

drowsy_list = [sleep,drowsy,active,status,color]



def side_face():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame1 = sfd.side_face_detector(frame)
            # data = im.fromarray(frame)
            ret,buffer=cv2.imencode('.jpg',frame1)
            frame2=buffer.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')



def drowsy_face():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            frame1 = dwsy.drowsyness_detector(frame,drowsy_list)
            # data = im.fromarray(frame)
            ret,buffer=cv2.imencode('.jpg',frame1)
            frame2=buffer.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interview')
def interview():
    return render_template('interview.html')

@app.route('/nightstudy')
def nightstudy():
    return render_template('nightstudy.html')


@app.route('/videoforInterview')
def videoforInterview():
    frame= Response(side_face(),mimetype='multipart/x-mixed-replace; boundary=frame')
    return frame

@app.route('/videoforNightstudy')
def videoforNightstudy():
    frame= Response(drowsy_face(),mimetype='multipart/x-mixed-replace; boundary=frame')
    return frame


if __name__=="__main__":
    app.run(debug=True)
