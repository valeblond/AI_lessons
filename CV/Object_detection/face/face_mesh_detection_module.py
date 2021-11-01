# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:43:42 2021

Face mesh detection

@author: valer

From here: https://www.youtube.com/watch?v=01sAkU_NvOY&list=PL6P8rMfgAhUIm4cUDzYp6RiOcR9rS4WkP&index=10&t=1220s&ab_channel=freeCodeCamp.org
"""

import cv2
import mediapipe as mp
import time


class face_mesh_detector():
    def __init__(self, static_mode=False, max_faces=1, ref_lms=False, detection_con=0.5, track_con=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.ref_lms = ref_lms
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.static_mode, self.max_faces, self.ref_lms,
                                                    self.detection_con, self.track_con) # take only RGB images
        self.mp_draw = mp.solutions.drawing_utils # draw the lines between the landmarks
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2) # set the parameters of lines and circles sizes
       
    def Find_Face_Mesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert the image to rgb
        self.results = self.face_mesh.process(self.imgRGB) # process the frames

        faces = []        
        if self.results.multi_face_landmarks:
            for lms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, lms, self.mp_face_mesh.FACEMESH_TESSELATION, self.draw_spec, self.draw_spec)
                
                face_lms = [] # list of coordinates for one specific face
                for lm_id, lm in enumerate(lms.landmark):
                    hight, width, channels = img.shape
                    x, y = int(lm.x*width), int(lm.y*hight) # transform normalised values of landmarks into pixels
                    face_lms.append([x,y])
            
                faces.append(face_lms)
                
        return img, faces


def main():
    prev_time = 0
    
    cap = cv2.VideoCapture("data/video6.mp4") # read a video from the file location
    
    detector = face_mesh_detector(max_faces=4)

    while True:
        success, img = cap.read()
        img, faces = detector.Find_Face_Mesh(img)

        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time
        
        cv2.putText( img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 ) # put a fps number at the image
            
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__  == "__main__":
    main()
