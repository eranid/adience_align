'''
Created on May 7, 2014

@author: eran
'''
import unittest
from adiencealign.cascade_detection.cascade_face_finder import CascadeFaceFinder
import cv2
from adiencealign.common.drawing import draw_rect
from adiencealign.common.images import extract_box
import os
from adiencealign.cascade_detection.cascade_detector import CascadeResult


class Test(unittest.TestCase):

    def testDetectFaces(self):
        '''
        Go through two images, the first with 1 face, the second with 4 faces
        Assert that the detected faces are correct, and draw them.
        Also creates output images of the padded faces
        '''
        fnames = ['./resources/cascade/Fayssal_Mekdad_0002.jpg', './resources/cascade/family-home.png']
        
        expected_faces = [[],[]]
        
        expected_faces[0].append(CascadeResult(box_with_score = ([61,62,132,132], 344), cascade_type = 'haar', angle = 0.0))
        
        expected_faces[1].append(CascadeResult(box_with_score = ([327,101,119,119], 244), cascade_type = 'haar', angle = 0.0))
        expected_faces[1].append(CascadeResult(box_with_score = ([238,107,111,111], 135), cascade_type = 'lbp', angle = 0.0))
        expected_faces[1].append(CascadeResult(box_with_score = ([163,48,93,93], 51), cascade_type = 'lbp', angle = 0.0))
        expected_faces[1].append(CascadeResult(box_with_score = ([433,86,95,95], 92), cascade_type = 'lbp', angle = 0.0))
    
        for n_images, fname in enumerate(fnames):
            _, base_fname = os.path.split(fname)
            img = cv2.imread(fname)
            gray_img = cv2.imread(fname, 0)
            
            face_finder = CascadeFaceFinder(haar_file = '../resources/haarcascade_frontalface_default.xml',
                                            lbp_file = '../resources/lbpcascade_frontalface.xml')
            faces = face_finder.get_faces_list_in_photo(gray_img)
            
            img_to_draw_on = img.copy()
            
            for n_face, face in enumerate(faces):
                self.assertAlmostEqual(face.overlap(expected_faces[n_images][n_face]) / face.area, 1.00, 0.01)
                
                draw_rect(img_to_draw_on, face)
                
                padded_face, bounding_box_in_padded_face, _, _ = extract_box(img, face, padding_factor = 0.25)
                new_face_file = os.path.join('./outputs/cascade/1/', base_fname.split('.')[0] + '.face.%d.png' %n_face)
                cv2.imwrite(new_face_file, padded_face)
                padded_face_loaded = cv2.imread(new_face_file)
                draw_rect(padded_face_loaded, bounding_box_in_padded_face)
                cv2.imshow('face %d' %n_face, padded_face_loaded)
                cv2.waitKey()
                
            cv2.imshow('faces detected', img_to_draw_on)
            cv2.waitKey()
            
    def testDetectFacesAndCreateFiles(self):
        '''
        Go through two images, the first with 1 face, the second with 4 faces
        Assert that the detected faces are correct, and draw them.
        Also creates output images of the padded faces
        '''
        fnames = ['./resources/cascade/Fayssal_Mekdad_0002.jpg', './resources/cascade/family-home.png']
        
        expected_results = [['x,y,dx,dy,score,angle,type\n',
                             '61,62,132,132,344,0.0,haar'],
                            ['x,y,dx,dy,score,angle,type\n',
                             '327,101,121,121,154,0.0,lbp\n',
                             '237,106,113,113,139,0.0,lbp\n',
                             '164,49,91,91,49,0.0,lbp\n',
                             '434,86,94,94,95,0.0,lbp\n']
                            ]
        
        for n_image in range(len(fnames)):
            fname = fnames[n_image]
            expected_result = expected_results[n_image]
            
            _, base_fname = os.path.split(fname)
            
            face_finder = CascadeFaceFinder(haar_file = '../resources/haarcascade_frontalface_default.xml',
                                            lbp_file = '../resources/lbpcascade_frontalface.xml')
            faces_file = face_finder.create_faces_file(fname, is_overwrite = True, target_file = './outputs/cascade/2/' + base_fname + '.faces.txt')
            
            # get the sub images
            sub_images = face_finder.get_sub_images_from_file(original_image_file = fname, faces_file = faces_file)
            for n_face, sub_image in enumerate(sub_images):
                cv2.imshow('face_%d' %n_face, sub_image)
            cv2.waitKey()
            
            # create sub images files
            sub_images_file = face_finder.create_sub_images_from_file(original_image_file = fname, faces_file = faces_file, target_folder = None)
            with open(faces_file,'r') as fid:
                for i in range(len(expected_result)):
                    line = fid.readline()
                    self.assertEqual(line.strip(), expected_result[i].strip())
                    
                            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDetectFaces']
    unittest.main()