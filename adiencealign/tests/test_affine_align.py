'''
Created on May 7, 2014

@author: eran
'''
import unittest
from adiencealign.affine_alignment.affine_aligner import AffineAligner
import cv2
from adiencealign.common.landmarks import read_fidu, draw_fidu, unwarp_fidu
from adiencealign.landmarks_detection.landmarks_detector import detect_landmarks
import os

class Test(unittest.TestCase):

    def testAffineAlign(self):
        aligner = AffineAligner(fidu_model_file = '../resources/model_ang_0.txt')
        
        img_files = ['./resources/affine_align/Meryl_Streep_0013.jpg', './resources/affine_align/Fayssal_Mekdad_0002.jpg']
        for img_file in img_files:
            img_file = os.path.abspath(img_file)
            detect_landmarks(fname = img_file)
        
            fidu_file = img_file.replace('.jpg','.cfidu')
            img = cv2.imread(img_file)
            score, yaw_angle, fidu_points = read_fidu(fidu_file)
            aligned_img, R = aligner.align(img, fidu_points)
            
            aligned_img_cpy = aligned_img.copy()
            draw_fidu(img, fidu_points)
            fidu_points_in_aligned = unwarp_fidu(orig_fidu_points = fidu_points, unwarp_mat = R)
            draw_fidu(aligned_img_cpy, fidu_points_in_aligned, radius = 9, color = (255,0,0), thickness = 3)
            
            cv2.imshow('padded_face with landmarks', img)
            cv2.imshow('aligned_face', cv2.resize(aligned_img, (320,320), interpolation = cv2.INTER_CUBIC))
            cv2.imshow('aligned_face with landmarks', cv2.resize(aligned_img_cpy, (320,320), interpolation = cv2.INTER_CUBIC))
            cv2.waitKey()
            
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testAffineAlign']
    unittest.main()