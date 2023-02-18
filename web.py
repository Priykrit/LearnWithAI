import cv2

from selenium import webdriver

    
def wb_search(frame,web_list):
    model=web_list[0]
    frame= cv2.flip(frame,1)
    results=model(frame)
    list=[]
    for index,row in results.pandas().xyxy[0].iterrows():
        x1=int(row[ 'xmin' ])
        y1=int (row[ 'ymin' ])
        x2=int(row[ 'xmax' ])
        y2=int (row[ 'ymax' ])
        b=str(row['name'])
        if 'person' not in b:
            list.append([x1,y1,x2,y2,b])

    list1 =[]
    for box_id in list:
        x,y,w,h,names=box_id
        cv2.rectangle(frame,(x,y),(w,h),(255,0,255),2)
        cv2.putText(frame,names,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0,),2)
        list1.append(names)

    for z in list1:
        topic_search=z
        topic_search = topic_search.replace(' ','+')

        browser = webdriver.Edge('msedgedriver.exe')

        for i in range(1):
            elements = browser.get("https://www.bing.com/search?q="+
            topic_search+"&start"+str(i))
    return frame