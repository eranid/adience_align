'''
Created on May 8, 2014

@author: eran
'''
import unittest
import glob
from adiencealign.cascade_detection.cascade_face_finder import CascadeFaceFinder
from adiencealign.affine_alignment.affine_aligner import AffineAligner
import os
from adiencealign.landmarks_detection.landmarks_detector import detect_landmarks
from adiencealign.common.landmarks import read_fidu, unwarp_fidu, draw_fidu
import cv2
from adiencealign.pipeline.CascadeFaceAligner import CascadeFaceAligner


class Test(unittest.TestCase):


    def testPipeline(self):
        '''
        there is no assert here, just observe the outputs in tests/outputs/pipeline/
        '''
        input_folder = './resources/pipeline/'
        faces_folder = './outputs/pipeline/faces/'
        aligned_folder = './outputs/pipeline/aligned/'
        cascade_face_aligner = CascadeFaceAligner()
        
        # detect cascade           
        cascade_face_aligner.detect_faces(input_folder, faces_folder)
        cascade_face_aligner.align_faces(input_images = faces_folder,
                                         output_path = aligned_folder, 
                                         fidu_max_size = 200*200, 
                                         fidu_min_size = 50*50, 
                                         is_align = True, 
                                         is_draw_fidu = True, 
                                         delete_no_fidu = True)
                
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testPipeline']
    unittest.main()