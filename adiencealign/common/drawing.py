'''
Created on May 7, 2014

@author: eran
'''
import cv2
from adiencealign.cascade_detection.cascade_detector import CascadeResult
import numpy as np

def draw_rect(img, r, angle = 0, color=(255,255,255), thickness = 4, alpha = 0.5):
    '''
    accepts:
    1. a (x,y,dx,dy) list
    4. a [(x,y,dx,dy),score] list, as returned by cv2.CascadeClassifier.detectMultiScaleWithScores()
    5. a CascadeResult object
    '''
    if type(r) == CascadeResult:
        color = tuple(list(color) + [alpha])
        cv2.polylines(img, pts = [r.points_int], isClosed = True, color = color, thickness = thickness)
        return
    elif len(r)==4 or len(r)==2: # [x,y,dx,dy]
        if len(r)==2:
            if len(r[0]) == 4:
                r = r[0]
            else: 
                raise Exception("bad input to draw_rect...")
        pt1 = int(round(r[0])), int(round(r[1]))
        pt2 = int(round(r[0]+r[2])), int(round(r[1]+r[3]))
        color = tuple(list(color) + [alpha])
        cv2.rectangle(img, pt1, pt2, color, thickness = thickness)
    else:
        raise Exception("bad input to draw_rect...")
    return