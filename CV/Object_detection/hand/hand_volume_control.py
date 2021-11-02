# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:03:41 2021

Volume control by hand

@author: valer
"""

import cv2
import time
import numpy as np

WIDTH_CAM, HIGHT_CAM = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH_CAM) # 3 - for a width
cap.set(4, HIGHT_CAM) # 4 - for a hight

while True:
    success, img = cap.read()
    
    
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

