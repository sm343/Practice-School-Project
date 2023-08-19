# Creator @ Sanjeev Mallick
# Face recognition system project
# Complete source code

# Importing the necessary modules in my project

import face_recognition as fr
import cv2 as cv
import base64
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import mysql.connector

# Hide the Tkinter main window
Tk().withdraw()

# Open file dialog and select an image
load_image = askopenfilename()

# Load the target image and detect face locations and encodings
target_image = fr.load_image_file(load_image)
face_locations = fr.face_locations(target_image)
target_encodings = fr.face_encodings(target_image, face_locations)

# Load Haar Cascade classifier for face detection

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a MySQL connection

db_connection = mysql.connector.connect(
    host='Your Hostname in MySQL',
    user='Your Username in MySQL ',
    password='Your Password in MySQL',
    database='Your database in MySQL'
)

db_cursor = db_connection.cursor()

# Create the 'known_faces' table if it doesn't exist
db_cursor.execute('''
    CREATE TABLE IF NOT EXISTS known_faces (
        id INT AUTO_INCREMENT PRIMARY KEY,
        filename VARCHAR(255),
        encoding TEXT
    )
''')

# Function to insert a face into the database

def insert_face_to_database(filename, encoding):
    # Convert the binary encoding to a base64 encoded string
    encoding_base64 = base64.b64encode(encoding).decode('utf-8')
    # Insert a new face into the database
    sql_query = 'INSERT INTO known_faces (filename, encoding) VALUES (%s, %s)'
    db_cursor.execute(sql_query, (filename, encoding_base64))
    db_connection.commit()

# Function to fetch all known faces and their encodings from the database

def fetch_known_faces():
    db_cursor.execute('SELECT filename, encoding FROM known_faces')
    return db_cursor.fetchall()

# Function to encode faces from a folder and store them in the database

def encode_faces_in_database(folder):
    for filename in os.listdir(folder):
        known_image = fr.load_image_file(os.path.join(folder, filename))
        known_encoding = fr.face_encodings(known_image)[0]
        insert_face_to_database(filename, known_encoding)  # Store the face encoding in the database

# Function to find and label target faces in the image

def find_target_faces():
    for face_location in face_locations:
        # Convert face location format to face_recognition format (top, right, bottom, left)
        top, right, bottom, left = face_location
        target_encoding = fr.face_encodings(target_image, [face_location])[0]  # Get the face encoding of each detected face

        for filename, encoding_base64 in fetch_known_faces():
            # Convert base64 encoded string back to binary encoding
            encoding_bytes = base64.b64decode(encoding_base64.encode('utf-8'))
            known_encoding = fr.face_encodings(target_image, [face_location])[0]
            is_target_face = fr.compare_faces([known_encoding], target_encoding, tolerance=0.60)[0]  # Compare faces

            if is_target_face:
                label = filename
                create_frame(face_location, label)

# Function to create a frame around the detected face and display the image

def create_frame(location, label):
    top, right, bottom, left = location

    cv.rectangle(target_image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv.rectangle(target_image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
    cv.putText(target_image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

# Function to display the image

def render_image():
    rgb_img = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
    cv.imshow('Face Recognition', rgb_img)
    cv.waitKey(0)

# Encode faces from the 'Image_Database' folder and store them in the database
encode_faces_in_database('Image_Database/')

# Find and label target faces in the image
find_target_faces()

# Render the image with labeled faces
render_image()

# Close the database connection
db_connection.close()


