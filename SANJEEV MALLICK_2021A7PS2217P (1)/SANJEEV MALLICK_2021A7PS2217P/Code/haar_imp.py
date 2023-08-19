import img as img
import numpy as np
import face_recognition as fr
import cv2 as cv
import base64
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import mysql.connector
import cv2


f_cascade = cv2.CascadeClassifier("face.xml")
e_cascade = cv2.CascadeClassifier("eye.xml")

image = cv2.imread("actor.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = f_cascade.detectMultiScale(gray, 1.3, 5)


for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = e_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',image)

cv2.waitKey(0)

cv2.destroyAllWindows()