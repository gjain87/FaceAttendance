import imp
from multiprocessing.spawn import import_main_path
from pickle import NONE
from tkinter import font
import cv2
from cv2 import putText
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
import face_recognition
import os
from datetime import datetime

path='images attendence'
images=[]
classnames=[]
mylist=os.listdir(path)
print(mylist)
for i in mylist:
    curr=cv2.imread(f'{path}/{i}')
    images.append(curr)
    classnames.append(os.path.splitext(i)[0])
print(classnames)  

def findencodings(images):
    encodelist=[]
    for i in images:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(i)[0]
        encodelist.append(encode)
    return encodelist

def markattendance(name):
    with open('attendance.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dt=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')

# markattendance('Gaurav') 




    



encodelistknown=findencodings(images)
print('Encoding complete')

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faceloccurr=face_recognition.face_locations(imgS)
    encodecurr=face_recognition.face_encodings(imgS,faceloccurr)

    for encodeface,faceloc in zip(encodecurr,faceloccurr):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        faceDis=face_recognition.face_distance(encodelistknown,encodeface)
        # print(faceDis)
        matchindex=np.argmin(faceDis)

        if matches[matchindex]:
            name=classnames[matchindex].upper()
            # print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markattendance(name)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)





