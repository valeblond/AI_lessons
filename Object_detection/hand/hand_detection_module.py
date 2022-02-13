# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:53:17 2021

Hand detection

From here: https://www.youtube.com/watch?v=01sAkU_NvOY&list=PL6P8rMfgAhUIm4cUDzYp6RiOcR9rS4WkP&index=10&t=1220s&ab_channel=freeCodeCamp.org

@author: Valery Burau
"""

import cv2
import mediapipe as mp
import time


class Hand_Detector():
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_con = detection_con
        self.track_con = track_con
        

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.detection_con, self.track_con) # take only RGB images
        self.mp_draw = mp.solutions.drawing_utils # draw the lines between the landmarks

    def find_Hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the image to rgb
        self.results = self.hands.process(imgRGB) # process the frames
        # print(results.multi_hand_landmarks) # checking wether it's sth detected (hands)
        
        # Find hands
        if self.results.multi_hand_landmarks:
            # Loop for each hand. Extract the information of each hand.
            for hand_lms in self.results.multi_hand_landmarks:             
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, 
                                                self.mp_hands.HAND_CONNECTIONS) # draw landmarks - points & lines
    
        return img
    
    def find_Position(self, img, hand_nb=0, draw=True):
        lm_list = []
        
        # Find hands
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_nb] # choose a particular hand
            
            # Consider each landmark on the hand
            for lm_id, lm in enumerate(my_hand.landmark): 
                # print(lm_id, lm)
                hight, width, channels = img.shape
                x, y = int(lm.x*width), int(lm.y*hight) # transform ratio values of landmarks into pixels
                lm_list.append([lm_id, x, y])
                if draw:
                    cv2.circle( img, (x,y), 7, (255,0,0), cv2.FILLED )
            
        return lm_list
            

def main():
    prev_time = 0
    
    cap = cv2.VideoCapture(0) # read a live image from the camera nb. 0
    
    detector = Hand_Detector()
    
    while True:
        success, img = cap.read()
        img = detector.find_Hands(img)
        lm_list = detector.find_Position(img)
        
        if (len(lm_list) != 0):
            print(lm_list[4])

        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time
        
        cv2.putText( img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 ) # put a fps number at the image
            
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)





if __name__ == "__main__":
    main()














