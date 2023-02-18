import cv2


    
def obj_search(frame,web_list):
    model=web_list[0]
    frame= cv2.flip(frame,1)
    results=model(frame)
    for index,row in results.pandas().xyxy[0].iterrows():
        x1=int(row[ 'xmin' ])
        y1=int (row[ 'ymin' ])
        x2=int(row[ 'xmax' ])
        y2=int (row[ 'ymax' ])
        b=str(row['name'])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)
        cv2.putText(frame,b,(x1,y1),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0,),2)
    return frame