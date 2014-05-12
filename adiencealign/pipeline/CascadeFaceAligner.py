'''
Created on May 8, 2014

@author: eran
'''
from adiencealign.cascade_detection.cascade_face_finder import CascadeFaceFinder
from adiencealign.affine_alignment.affine_aligner import AffineAligner
import glob
import os
from adiencealign.landmarks_detection.landmarks_detector import detect_landmarks
from adiencealign.common.landmarks import read_fidu, unwarp_fidu, draw_fidu
import cv2

class CascadeFaceAligner(object):
    '''
    classdocs
    '''


    def __init__(self, haar_file = '../resources/haarcascade_frontalface_default.xml', 
                 lbp_file = '../resources/lbpcascade_frontalface.xml', 
                 fidu_model_file = '../resources/model_ang_0.txt',
                 fidu_exec_dir = '../resources/'):
        '''
        Constructor
        '''
        self.face_finder = CascadeFaceFinder(haar_file = haar_file,
                                            lbp_file = lbp_file)
        
        self.aligner = AffineAligner(fidu_model_file = fidu_model_file)
        self.fidu_exec_dir = fidu_exec_dir
        
        self.valid_angles = [-45,-30,-15,0,15,30,45]
        
    def detect_faces(self, input_folder, output_folder, mark_dones = True):
        '''
        mark_dones - if True, will create a hidden file, marking this file as done with a hidden file, starting with '.done.'
        '''        
        input_files1 = glob.glob(os.path.join(input_folder, '*.jpg'))
        input_files2 = glob.glob(os.path.join(input_folder, '*.png'))
        input_files = input_files1 + input_files2
        N = len(input_files)
        
        for n_file, input_file in enumerate(input_files):
            a,b = os.path.split(input_file)
            done_file = os.path.join(a, '.done.' + b.rsplit('.',1)[0])
            
            if os.path.exists(done_file):
                continue
            
            print "... processing", input_file

            target_faces_file = os.path.join( output_folder, os.path.split(input_file)[1].rsplit('.',1)[0] + '.faces.txt')
            
            faces_file = self.face_finder.create_faces_file( input_file, is_overwrite = False, target_file = target_faces_file )
            sub_images_files = self.face_finder.create_sub_images_from_file( original_image_file = input_file, 
                                                    faces_file = faces_file, 
                                                    target_folder = output_folder,
                                                    img_type = 'jpg')
            #touch
            open(done_file,'w').close()
            print "Detected on %d / %d files" %(n_file, N)
        
    def align_faces(self, input_images, output_path, fidu_max_size = None, fidu_min_size = None, is_align = True, is_draw_fidu = False, delete_no_fidu = False):
        '''
        input_images - can be either a folder (all *.jpgs in it) or a list of filenames
        
        , fidu_max_size = None, fidu_min_size = None):
        '''
        if type(input_images) == type(''):
            input_images = glob.glob(os.path.join(input_images, '*.jpg'))
        
        for input_image in input_images:
            detect_landmarks(fname = os.path.abspath(input_image),
                                 max_size = fidu_max_size,
                                 min_size = fidu_min_size,
                                 fidu_exec_dir = self.fidu_exec_dir)
            
            fidu_file = input_image.rsplit('.',1)[0] + '.cfidu'
            fidu_score, yaw_angle, fidu_points = read_fidu(fidu_file)
            
            if not (fidu_score is not None and yaw_angle in self.valid_angles):
                # skip face
                if delete_no_fidu:
                    os.remove(fidu_file)
                    os.remove(input_image)
                continue
            
            if is_align:
                # save the aligned image
                sub_img = cv2.imread(input_image)
                _, base_fname = os.path.split(input_image) 
                aligned_img, R = self.aligner.align(sub_img, fidu_points)
                aligned_img_file = os.path.join(output_path, base_fname.rsplit('.',1)[0] + '.aligned.png')
                cv2.imwrite(aligned_img_file, aligned_img)
                
                # save a copy of the aligned image, with the landmarks drawn
                if is_draw_fidu:
                    aligned_img_file = os.path.join(output_path, base_fname.rsplit('.',1)[0] + '.aligned.withpoints.png') 
                    fidu_points_in_aligned = unwarp_fidu(orig_fidu_points = fidu_points, unwarp_mat = R)
                    draw_fidu(aligned_img, fidu_points_in_aligned, radius = 9, color = (255,0,0), thickness = 3)
                    cv2.imwrite(aligned_img_file, aligned_img)
            
#             
#     def align_faces2(self, input_folder, output_folder, detect_landmarks = True, delete_no_fidu = True, is_align = True, is_draw_fidu = False, fidu_max_size = None, fidu_min_size = None):
#         '''
#         delete_no_fidu - deletes the .cfidu and sub_img if no fidu was found
#         is_draw_fidu - creates a copy of the aligned images with the original fiducial points on it
#         '''
#         input_files = glob.glob(os.path.join(input_folder, '*'))
#         for input_file in input_files:
#             print "... processing", input_file
#             target_faces_file = os.path.join( output_folder, os.path.split(input_file)[1].rsplit('.',1)[0] + '.faces.txt')
#             
#             faces_file = self.face_finder.create_faces_file( input_file, is_overwrite = False, target_file = target_faces_file )
#             sub_images_files = self.face_finder.create_sub_images_from_file( original_image_file = input_file, 
#                                                     faces_file = faces_file, 
#                                                     target_folder = output_folder,
#                                                     img_type = 'jpg')
#             
#             for sub_image_file in sub_images_files:
#                 detect_landmarks(fname = os.path.abspath(sub_image_file),
#                                  max_size = fidu_max_size,
#                                  min_size = fidu_min_size,
#                                  fidu_exec_dir = self.fidu_exec_dir)
#                 
#                 fidu_file = sub_image_file.rsplit('.',1)[0] + '.cfidu'
#                 fidu_score, yaw_angle, fidu_points = read_fidu(fidu_file)
#                 
#                 if not (fidu_score is not None and yaw_angle in self.valid_angles):
#                     # skip face
#                     os.remove(fidu_file)
#                     os.remove(sub_image_file)
#                     continue
#                 
#                 if is_align:
#                     # save the aligned image
#                     sub_img = cv2.imread(sub_image_file)
#                     aligned_img, R = self.aligner.align(sub_img, fidu_points)
#                     aligned_img_file = sub_image_file.rsplit('.',1)[0] + '.aligned.png'
#                     cv2.imwrite(aligned_img_file, aligned_img)
#                     
#                     # save a copy of the aligned image, with the landmarks drawn
#                     if is_draw_fidu:
#                         aligned_img_file = sub_image_file.rsplit('.',1)[0] + '.aligned.withpoints.png'
#                         fidu_points_in_aligned = unwarp_fidu(orig_fidu_points = fidu_points, unwarp_mat = R)
#                         draw_fidu(aligned_img, fidu_points_in_aligned, radius = 9, color = (255,0,0), thickness = 3)
#                         cv2.imwrite(aligned_img_file, aligned_img)
