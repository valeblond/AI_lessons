# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:53:17 2021

@author: valer
"""

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands() # take only RGB images
mp_draw = mp.solutions.drawing_utils # draw the lines between the landmarks

prev_time = 0
curr_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the image to rgb
    results = hands.process(imgRGB) # process the frames
    # print(results.multi_hand_landmarks) # checking wether it's sth detected (hands)
    
    # Find hands
    if results.multi_hand_landmarks:
        # Loop for each hand. Extract the information of each hand.
        for hand_lms in results.multi_hand_landmarks:
            # Consider each landmark on the hand
            for lm_id, lm in enumerate(hand_lms.landmark):
                # print(lm_id, lm)
                hight, width, channels = img.shape
                x, y = int(lm.x*width), int(lm.y*hight) # transform ratio values of landmarks into pixels
                print(lm_id, x, y)
                # if lm_id == 0:
                cv2.circle( img, (x,y), 15, (255,0,255), cv2.FILLED )
                
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS) # draw landmarks - points & lines
    
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    
    cv2.putText( img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 )
        
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
















