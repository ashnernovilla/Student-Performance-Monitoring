# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 20:29:57 2021

@author: ASHNER_NOVILLA
"""

# https://stackoverflow.com/questions/65273118/why-is-tensorflow-not-recognizing-my-gpu-after-conda-install

# Basic Libracry
import cv2
import numpy as np
import os
from scipy.spatial import distance
from datetime import datetime
import pandas as pd

# This library is for the face recognition
import face_recognition

# This library is for the face behavior 
from deepface import DeepFace
font = cv2.FONT_HERSHEY_SIMPLEX

# This library is for detecting the drowsyness of the person
import dlib

# Path of the known Images (You can put the image of the employee in this folder - Path is already declared)
path = r'C:\Users\ASHNER_NOVILLA\Pictures\PictureTest'
images = []
classnames = []
myList = os.listdir(path)
print(myList)

cv2.startWindowThread()

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg) #Append all the imgaes on the directory
    classnames.append(os.path.splitext(cl)[0]) #Split the name of the images to its entension

print(classnames)

# Encoding the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Calculating the Eye Aspect Ratio
def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

# Attendance logging - This can be function can be replaced if the location will be on the sql database
# For PoC the developer used csv for easy access and checking
def markAttendance(name, behavior, drowsiness):
    with open(r'D:\Disk_Drive\DocumentsBuckUp\360DigiDeepLearning\CapstoneII\Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # if name not in nameList:  #This is if we're only accepting one user log-in at a day
        #     now = datetime.now()
        #     dtString = now.strftime('%H:%M:%S')
        #     f.writelines(f'\n{name},{behavior},{dtString}')  
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        f.writelines(f'\n{name},{behavior},{drowsiness},{dtString}')
 
# Find the Macthes between our encodings
encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))  #Number of images on the folder      

import mediapipe as mp # Import mediapipe
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

import pickle
import json

with open(r'D:\Disk_Drive\DocumentsBuckUp\360DigiDeepLearning\CapstoneIII\body_language.pkl', 'rb') as f:
    model = pickle.load(f)
    
def markPosture_json(new_data):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')

    dictionary ={
        "Body Posture" : new_data,
        "DateTime" : dtString,
                }
    # Serializing json 
    json_object = json.dumps(dictionary, indent = 4)
    
    with open(r'D:\Disk_Drive\DocumentsBuckUp\360DigiDeepLearning\CapstoneIII\PostureLog.json', "a") as outfile:
        outfile.write(json_object)
        
def markPosture(body_language_class):
    with open(r'D:\Disk_Drive\DocumentsBuckUp\360DigiDeepLearning\CapstoneIII\PostureLog.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])  
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        f.writelines(f'\n{body_language_class}, {dtString}')
    
model

# Accesing the camera
cap = cv2.VideoCapture(0) #Activating the camera - 0 for main camera and other numbers to the extended camera
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError ("Can't Open WebCam")

# The model for the face behavior
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# print(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# The model for the face marks for eye blink detection
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(r'D:\Disk_Drive\DocumentsBuckUp\360DigiDeepLearning\CapstoneII\shape_predictor_68_face_landmarks.dat')

# Realtime Video Capture
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, img = cap.read()
        
        try:  #In this try we on perform the code if we detect a know person if not the code will proceed to the except
            result = DeepFace.analyze(img, actions = ['emotion'])  #This is the function to call the emotions on deepface library
            behavior = result['dominant_emotion']    #This is the function to call the dominant emotions on deepface library
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #Converting the captured image to gray scale for faster processing
            # faces = facecascade.detectMultiScale(gray,1.1,4)        
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # We're scaling the image to quickly match all the images
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            
            #### This is the start of body posture
            # Recolor Feed
            imagess = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagess.flags.writeable = False        
            
            # Make Detections
            resultss = holistic.process(imagess)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            imagess.flags.writeable = True   
            imagess = cv2.cvtColor(imagess, cv2.COLOR_RGB2BGR)
            
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(img, resultss.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )
            
            # 2. Right hand
            mp_drawing.draw_landmarks(img, resultss.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )
    
            # 3. Left Hand
            mp_drawing.draw_landmarks(img, resultss.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                     )
    
            # 4. Pose Detections
            mp_drawing.draw_landmarks(img, resultss.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
            # Export coordinates
            try:
                # Extract Pose landmarks
                posess = resultss.pose_landmarks.landmark
                pose_rowss = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in posess]).flatten())
                
                # Extract Face landmarks
                facess = resultss.face_landmarks.landmark
                face_rowss = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in facess]).flatten())
                
                # Concate rows
                rowss = pose_rowss+face_rowss
                
    #             # Append class name 
    #             row.insert(0, class_name)
                
    #             # Export to CSV
    #             with open('coords.csv', mode='a', newline='') as f:
    #                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 csv_writer.writerow(row) 
    
                # Make Detections
                XSS = pd.DataFrame([rowss])
                body_language_classs = model.predict(XSS)[0]
                body_language_probss = model.predict_proba(XSS)[0]
                
                print(body_language_classs, body_language_probss)
                
                # Posture DataLog 
                markPosture(body_language_classs)
                
                # Grab ear coords
                coordss = tuple(np.multiply(
                                np.array(
                                    (resultss.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                     resultss.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                '''
                cv2.rectangle(imagess, 
                              (coordss[0], coordss[1]+5), 
                              (coordss[0]+len(body_language_classs)*20, coordss[1]-30), 
                              (245, 117, 16), -1)
                cv2.putText(imagess, body_language_classs, coordss, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(imagess, (0,0), (250, 60), (245, 117, 16), -1)
                '''
                
                #Display Class
                cv2.putText(img, 'CLASS'
                            , (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img, body_language_classs.split(' ')[0]
                            , (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(img, 'PROB'
                            , (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(img, str(round(body_language_probss[np.argmax(body_language_probss)],2))
                            , (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                
            except:
                pass
            
            #This code is the start of the Fatigue Detection
            faceCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
            
            faces = hog_face_detector(gray)
            for face in faces:
        
                face_landmarks = dlib_facelandmark(gray, face)
                leftEye = []
                rightEye = []
        
                for n in range(36,42):
                	x = face_landmarks.part(n).x
                	y = face_landmarks.part(n).y
                	leftEye.append((x,y))
                	next_point = n+1
                	if n == 41:
                		next_point = 36
                	x2 = face_landmarks.part(next_point).x
                	y2 = face_landmarks.part(next_point).y
                	cv2.line(img,(x,y),(x2,y2),(0,255,0),1)
        
                for n in range(42,48):
                	x = face_landmarks.part(n).x
                	y = face_landmarks.part(n).y
                	rightEye.append((x,y))
                	next_point = n+1
                	if n == 47:
                		next_point = 42
                	x2 = face_landmarks.part(next_point).x
                	y2 = face_landmarks.part(next_point).y
                	cv2.line(img,(x,y),(x2,y2),(0,255,0),1)
        
                left_ear = calculate_EAR(leftEye)
                right_ear = calculate_EAR(rightEye)
        
                EAR = (left_ear+right_ear)/2
                EAR = round(EAR,2)
                if EAR<0.26:
                    drowsiness = "Drowsy"
                else:
                    drowsiness = "Awake"        
             #This end code is the start of the Fatigue Detection
             
            #This is the start of the Face Match 
            #Iterate to all the images and compare it to the images on the camera
            for encodeFace, faceloc in zip(encodesCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis) #Getting the mimum value because having a mimum value is the closest
                print(matches[matchIndex])
                
                if faceDis[matchIndex] < 0.35:  #If the minimum value is less than 0.30 we will take the name else we will label it as unknown
                    name = classnames[matchIndex].upper()
                    print(name)
                    
                else:
                    name = 'UnknownPerson'
            #This is the end of the Face Match
            
            # Logging the results on the video panel                 
                y1, x1, y2, x2 = faceloc
                y1, x1, y2, x2 = y1*4, x1*4, y2*4, x2*4
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                # cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img, name, (x1-135, y2-6), font,1,(255,255,255), 2)
                cv2.putText(img, result['dominant_emotion'], (x1-135, y2+50), font, 1, (0,255,0),2)
                cv2.putText(img, drowsiness, (20,150), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
                            
                #Data Log the resut to CSV - In this code we use CSV to quickly check the results - for future reference, SQL or NOSQL is better
                markAttendance(name, behavior, drowsiness)
                
        except ValueError: #If no person is detected and an erro will be trown 
            print("No Person in the Camera")
            name = "NoPerson"
            behavior = "NoPerson" 
            drowsiness = "NoPerson"           
            markAttendance(name, behavior, drowsiness)
            
        cv2.imshow('Original Video', img)
                
        if cv2.waitKey(2) & 0xFF ==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
