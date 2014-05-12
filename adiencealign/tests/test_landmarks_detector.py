'''
Created on May 7, 2014

@author: eran
'''
import unittest
from adiencealign.landmarks_detection.landmarks_detector import detect_landmarks
import os
from adiencealign.common.landmarks import read_fidu
from adience.common.utils.fidu import draw_fidu
import cv2


class Test(unittest.TestCase):

    def testDetectLandmarks(self):
        input_file = os.path.abspath('./resources/landmarks/Fayssal_Mekdad_0002.jpg')
        fidu_file = input_file.replace('.jpg','.cfidu')
        if os.path.exists(fidu_file):
            os.remove(fidu_file)
            
        detect_landmarks(fname = input_file)
        fidu_score, yaw_angle, fidu_points = read_fidu(fidu_file)
        
        self.assertEqual(fidu_score, 252)
        self.assertEqual(yaw_angle, 0)
        
        img = cv2.imread(input_file)
        draw_fidu(img, fidu_points)
        cv2.imshow('landmarks',img)
        cv2.waitKey()
        
        
        if os.path.exists(fidu_file):
            os.remove(fidu_file)
            
        detect_landmarks(fname = input_file, max_size = 160*160)
        fidu_score, yaw_angle, fidu_points = read_fidu(fidu_file)
        
        print "new fidu score is", fidu_score
        self.assertGreater(fidu_score, 200)
        self.assertEqual(yaw_angle, 0)
        
        img = cv2.imread(input_file)
        draw_fidu(img, fidu_points)
        cv2.imshow('landmarks, detected on reduced image',img)
        cv2.waitKey()
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()