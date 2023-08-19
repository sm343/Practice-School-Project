import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

Tk().withdraw()
load_image = askopenfilename()

target_image = fr.load_image_file(load_image)
face_locations = fr.face_locations(target_image)
target_encodings = fr.face_encodings(target_image, face_locations)

# Load Haar Cascade classifier for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Searches for occurrences of known faces
def encode_faces(folder):
    list_people_encoding = []

    for filename in os.listdir(folder):
        known_image = fr.load_image_file(os.path.join(folder, filename))
        known_encoding = fr.face_encodings(known_image)[0]

        list_people_encoding.append((known_encoding, filename))

    return list_people_encoding

def find_target_faces():
    # Detect faces using Haar Cascade
    gray_image = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
    face_locations_haar = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in face_locations_haar:
        # Convert Haar Cascade face location format to face_recognition format (top, right, bottom, left)
        face_location = (y, x + w, y + h, x)
        target_encoding = fr.face_encodings(target_image, [face_location])[0]  # Get the face encoding of each detected face

        for person in encode_faces('Image_Database/'):
            encoded_face = person[0]
            filename = person[1]
            is_target_face = fr.compare_faces([encoded_face], target_encoding, tolerance=0.60)[0]  # Compare faces

            if is_target_face:
                label = filename
                create_frame(face_location, label)

def create_frame(location, label):
    top, right, bottom, left = location

    cv.rectangle(target_image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv.rectangle(target_image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
    cv.putText(target_image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

def render_image():
    rgb_img = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
    cv.imshow('Face Recognition', rgb_img)
    cv.waitKey(0)

find_target_faces()
render_image()
