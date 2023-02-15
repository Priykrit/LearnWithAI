from flask import Flask,render_template,Response
import cv2
import side_face_detection as sfd
from PIL import Image as im
app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interview')
def index():
    return render_template('interview.html')

@app.route('/video')
def video():
    frame= Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return frame


if __name__=="__main__":
    app.run(debug=True)
