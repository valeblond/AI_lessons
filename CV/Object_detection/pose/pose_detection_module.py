# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 20:11:20 2021

Pose estimation

@author: Valery Burau

From here: https://www.youtube.com/watch?v=01sAkU_NvOY&list=PL6P8rMfgAhUIm4cUDzYp6RiOcR9rS4WkP&index=10&t=1220s&ab_channel=freeCodeCamp.org
"""

import cv2
import mediapipe as mp
import time

class pose_detector():
    def __init__(self, mode=False, complexity=1, smooth_lm=2, enable_segm=False,
                 smooth_segm=True, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_lm = smooth_lm
        self.enable_segm = enable_segm
        self.smooth_segm = smooth_segm
        self.detection_con = detection_con
        self.track_con = track_con
        

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.complexity, self.smooth_lm, self.enable_segm,
                                       self.smooth_segm, self.detection_con, self.track_con) # take only RGB images
        self.mp_draw = mp.solutions.drawing_utils # draw the lines between the landmarks
   
    def Find_Pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the image to rgb
        self.results = self.pose.process(imgRGB) # process the frames
        
        # Find pose landmarks
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS) # draw landmarks - points & lines
    
        return img
  
    
    def Find_Position(self, img, hand_nb=0, draw=True):
        lm_list = []
        
        # Find pose landmarks
        if self.results.pose_landmarks:
            for lm_id, lm in enumerate(self.results.pose_landmarks.landmark):
                hight, width, channels = img.shape
                x, y = int(lm.x*width), int(lm.y*hight) # transform ratio values of landmarks into pixels
                lm_list.append([lm_id, x, y])
                
                if draw:
                    cv2.circle( img, (x,y), 10, (255,0,0), cv2.FILLED )
            
        return lm_list


def main():
    prev_time = 0
    
    cap = cv2.VideoCapture("data/video5.mp4") # read a video from the file location
    
    detector = pose_detector()
    
    while True:
        success, img = cap.read()
        img = detector.Find_Pose(img)
        lm_list = detector.Find_Position(img)
        
        if (len(lm_list) != 0):
            print(lm_list[4])

        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time
        
        cv2.putText( img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255,0,255), 3 ) # put a fps number at the image
            
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()



    

