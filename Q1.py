#imported the libraries
import mediapipe as mp
import cv2
import numpy as np
#mp_drawing is to use the predifined utilities of mediapipe and to draw interconnecting lines
mp_drawing=mp.solutions.drawing_utils
#this variable customises the interconnecting lines
mp_drawing_styles=mp.solutions.drawing_styles
#Contains the hand tracking model from mediapipe
mphands=mp.solutions.hands
#opens the camera
capture = cv2.VideoCapture(0)
#initialises the tracking model with default parameters
hands = mphands.Hands()
while True:
    #loop receives images from videocam and is stored in frame variable. the return variable will be 
    #true if image is received successfully from the videocamera
    ret, frame = capture.read()
    #the image in frame is latterally inverted and the colour is changed from open CV's BGR pattern to RGB
    img = cv2.cvtColor(cv2.flip(frame,1),cv2.COLOR_BGR2RGB)
    #stores the inverted image in results variable for hand detection model to process
    result = hands.process(img)
    #the image from processing ha its colour changed from open CV's BGR pattern to RGB   
    img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)
    #the if statements are used if any hands were detected if they were detected, landamarks are drawn on the
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,mphands.HAND_CONNECTIONS)
    cv2.imshow('Handtracker',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
