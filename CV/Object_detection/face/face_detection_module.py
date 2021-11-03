# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:35:33 2021

Face detection

@author: Valery Burau

From here: https://www.youtube.com/watch?v=01sAkU_NvOY&list=PL6P8rMfgAhUIm4cUDzYp6RiOcR9rS4WkP&index=10&t=1220s&ab_channel=freeCodeCamp.org
"""

import cv2
import mediapipe as mp
import time


class Face_Detector():
    def __init__(self, detection_con=0.5, model_sel=0):
        self.detection_con = detection_con
        self.model_sel = model_sel

        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection(self.detection_con, self.model_sel) # take only RGB images
        self.mp_draw = mp.solutions.drawing_utils # draw the lines between the landmarks
        
    def find_Face(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the image to rgb
        self.results = self.face.process(imgRGB) # process the frames

        bound_boxes = []
        if self.results.detections:
            for face_id, detection in enumerate(self.results.detections):
                bound_box_C = detection.location_data.relative_bounding_box
                hight, width, channels = img.shape
                bound_box = int(bound_box_C.xmin * width), int(bound_box_C.ymin * hight), \
                            int(bound_box_C.width * width), int(bound_box_C.height * hight)
                bound_boxes.append([face_id, bound_box, detection.score])        
                    
                if draw:
                    img = self.fancy_Draw(img, bound_box)
                
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', 
                            (bound_box[0], bound_box[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0,255), 2)
                
        return img, bound_boxes

    def fancy_Draw(self, img, bbox, l=30, t=5):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        
        cv2.rectangle(img, bbox, (255, 0, 255), 1)
        # Top Left x, y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top Right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)        

        return img

        

def main():
    prev_time = 0
    
    cap = cv2.VideoCapture("data/video6.mp4") # read a video from the file location
    
    detector = Face_Detector()

    while True:
        success, img = cap.read()
        img, bound_boxes = detector.find_Face(img)

        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time
        
        cv2.putText( img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 ) # put a fps number at the image
            
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__  == "__main__":
    main()
