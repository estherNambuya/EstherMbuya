#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import cv2
import time
import os
from tkinter import messagebox
# This is an example of running face recognition on live video from an ip webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from an ip webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
#


# Get a reference to an ip webcam #0 (the default one)
ip="https://192.168.43.1:8080/video"
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
#Edwin_image= face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Parents\\Edwin.jpg")
#Edwin_face_encoding = face_recognition.face_encodings(Edwin_image)[0]

#Kip_image= face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Parents\\Kip.jpg")
#Kip_face_encoding = face_recognition.face_encodings(Kip_image)[0]

Shell_image= face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Parents\\Shell.jpg")
Shell_face_encoding = face_recognition.face_encodings(Shell_image)[0]

# Load a second sample picture and learn how to recognize it.
Nelly_image = face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Parents\\Nelly.jpg")
Nelly_face_encoding = face_recognition.face_encodings(Nelly_image)[0]

# Load a second sample picture and learn how to recognize it.
Esther_image = face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Parents\\Davy.jpg")
Esther_face_encoding = face_recognition.face_encodings(Esther_image)[0]

Jeff_image= face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Parents\\Jeff.jpg")
Jeff_face_encoding = face_recognition.face_encodings(Jeff_image)[0]

# Load a second sample picture and learn how to recognize it.
Babu_image = face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Parents\\Babu.jpg")
Babu_face_encoding = face_recognition.face_encodings(Babu_image)[0]

# Load a second sample picture and learn how to recognize it.
Nelson_image = face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Parents\\Tanui.jpg")
Nelson_face_encoding = face_recognition.face_encodings(Nelson_image)[0]

#Frank_image= face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Parents\\Frank.jpg")
#Frank_face_encoding = face_recognition.face_encodings(Frank_image)[0]

#X_image= face_recognition.load_image_file("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Esther\\Melissa.jpg")
#X_face_encoding = face_recognition.face_encoding(X_image)[0]

# Load a second sample picture and learn how to recognize it.

# Create arrays of known face encodings and their names
known_face_encodings = [
    #Frank_face_encoding,

    #Edwin_face_encoding,

    #Kip_face_encoding,

    Babu_face_encoding,

    Esther_face_encoding,

    Nelly_face_encoding,

    Shell_face_encoding,

    Nelson_face_encoding,

    Jeff_face_encoding,

    #X_face_encoding
]
known_face_names = [
    #"Frank",

    #"Edwin",

    #"Mr. Kip",

    "Babu",

    "Esther",

    "Nelly",

    "Jacinta Shell",

    "Tanui",

    "Jeff"

    #"Melissa"
]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)




        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            #if name == 'Mr. Kip':
                #img = cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Watoto\\6.JPG")
                #cv2.imshow('mtoto wa ' + name, img)

            if name == 'Nelly':
                img = cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Watoto\\14.JPG")
                cv2.imshow('mtoto wa ' + name, img)

            elif name == 'Jacinta Shell':
                img = cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Watoto\\2.JPG")
                cv2.imshow('mtoto wa ' + name,img)

            elif name == 'Babu':
                img=cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Watoto\\15.JPG")
                cv2.imshow('mtoto wa ' + name, img)

            #elif name == 'Edwin':
                #img = cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Watoto\\4.JPG")
                #cv2.imshow('mtoto wa ' + name, img)

            elif name == 'Esther':
                img = cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Watoto\\16.JPG")
                cv2.imshow('mtoto wa ' + name, img)

            #elif name == 'Frank':
                #img = cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Watoto\\12.JPG")
                #cv2.imshow('watoto wa ' + name,img)

            #elif name == 'Melissa':
                #img = cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Esther\\8.JPG")
                #cv2.imshow('mtoto wa ' + name,img)

            elif name == 'Tanui':
                img = cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Watoto\\7.JPG")
                cv2.imshow('mtoto wa ' + name,img)

            elif name == 'Jeff':
                img = cv2.imread("C:\\Users\\user\\.PyCharmCE2018.3\\system\\python_stubs\\-1839912465\\PIL\\Watoto\\1.JPG")
                cv2.imshow('mtoto wa ' + name, img)

        elif False in matches:
            name = "Stranger!"
            import sqlite3

            conn = sqlite3.connect('C:\\Users\\user\\python-pusher-traffic-monitor\\pythonsqlite.db')
            if not os.path.exists('./dataset'):
                os.makedirs('./dataset')
            c = conn.cursor()
            # face_cascade = cv2.CascadeClassifier("C:\\Program Files\\haarcascade_frontalface_default.xml")
            #cap = cv2.VideoCapture(0)
            # uname = input("Enter your name: ")
            # c.execute('INSERT INTO users (name) VALUES (?)', (uname,))
            uid = c.lastrowid
            sampleNum = 0
            while True:
                #ret, img = cap.read()
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                # for (x, y, w, h) in img:

                sampleNum = sampleNum + 1
                cv2.imwrite("dataset/Stranger." + str(uid) + "." + str(sampleNum) + ".jpg", frame)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.waitKey(20)
                #cv2.imshow('img', img)
                #cv2.waitKey(1);
                os.system("C:\\Users\\user\\.PyCharmCE2018.3\\system\python_stubs\\-1839912465\\PIL\\Alarm\\Babies.m4a")
                if sampleNum >5:
                    break
            #video_capture.release()
            #conn.commit()
            #conn.close()
                #time.sleep(20)

                #cv2.imwrite("C:\\Users\\user\\Desktop\\Strangers\\FORBIDDEN_4.png",frame)
                #os.system("C:\\Users\\user\\.PyCharmCE2018.3\\system\python_stubs\\-1839912465\\PIL\\Esther\\Babies.m4a")




        # Draw a box around the face
        cv2.rectangle(frame, (left-13,top-34), (right+13, bottom +34), (127, 0, 255), 5)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, top-45), (right, top), (153, 0, 153), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+12, top-2), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




