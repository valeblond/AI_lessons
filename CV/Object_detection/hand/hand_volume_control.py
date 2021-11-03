# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:03:41 2021

Volume control by hand

@author: valer
"""

import cv2
import time
import numpy as np
import math
import hand_detection_module as hdm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

WIDTH_CAM, HIGHT_CAM = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH_CAM) # 3 - for a width
cap.set(4, HIGHT_CAM) # 4 - for a hight

prev_time = 0

detector = hdm.Hand_Detector(detection_con=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
min_vol  = vol_range[0]
max_vol  = vol_range[1]
volume.SetMasterVolumeLevel(-20.0, None)

while True:
    success, img = cap.read()
    img = detector.find_Hands(img)
    lm_list = detector.find_Position(img, draw=False)
    
    if (len(lm_list) != 0):
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cntr_x, cntr_y = (x1 + x2)//2, (y1 + y2)//2
        
        cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cntr_x, cntr_y), 6, (125, 125, 0), cv2.FILLED)
        
        line_len = math.hypot(x2 - x1, y2 - y1) # count the distance between two points
        # print(line_len)
        
        # Hand range is from ~20 to 150
        # Volume range from the 'pycaw' library is from -65 to 0
        
        vol = np.interp(line_len, [20, 150], [min_vol, max_vol]) # change the range
        print(vol)
    
    
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    
    cv2.putText( img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3 )
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

