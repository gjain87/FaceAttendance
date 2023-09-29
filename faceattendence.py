from unittest import result
import cv2
import numpy as np
from scipy.misc import face
import face_recognition
img1=face_recognition.load_image_file('dp.jpg')
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=face_recognition.load_image_file('photo.jpeg')
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
faceloc=face_recognition.face_locations(img1)[0]
faceloc2=face_recognition.face_locations(img2)[0]
encodeimg1=face_recognition.face_encodings(img1)[0]
encodeimg2=face_recognition.face_encodings(img2)[0]
cv2.rectangle(img1,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
cv2.rectangle(img2,(faceloc2[3],faceloc2[0]),(faceloc2[1],faceloc2[2]),(255,0,255),2)
facedis=face_recognition.face_distance([encodeimg1],encodeimg2)

results=face_recognition.compare_faces([encodeimg1],encodeimg2)
print(results,facedis)
# print(faceloc)
cv2.imshow('Gaurav',img1)
cv2.imshow('Gaurav',img2)
cv2.waitKey(0)
