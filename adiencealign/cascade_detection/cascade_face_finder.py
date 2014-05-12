'''
Created on May 7, 2014

@author: eran
'''
from adiencealign.common.images import extract_box
import glob
import os
import time
from adiencealign.cascade_detection.cascade_detector import CascadeDetector,\
    resolve_boxes, CascadeResult
import cv2
import csv
'''
Created on Dec 18, 2013

@author: eran
'''
'''
Created on Nov 26, 2013

@author: eran
'''

class CascadeFaceFinder(object):
    
    def __init__(self, 
                 min_size = 32, 
                 drawn_target_res = 360*360, 
                 hangles = [0, -22, 22], 
                 langles = [0,-45,-22,22,45], 
                 haar_file = 'haarcascade_frontalface_default.xml',
                 lbp_file = 'lbpcascade_frontalface.xml'):
        '''
        finder = CascadeFaceFinder(min_size = 32, drawn_target_res = 360*360, hangles = [0], langles = [0,-45,-22,22,45], parts_threshold = 0)
        
        finder.get_faces_in_folder(input_folder, output_dir, drawn_folder, is_small_drawn)
        
        or
        
        finder.get_faces_in_photo(full_file, output_dir, drawn_folder, is_small_drawn)
        '''
        self.min_size = (min_size,min_size)
        self.drawn_target_res = drawn_target_res
        self._hangles = hangles
        self._langles = langles
        self.recalc_detectors(haar_file, lbp_file)
        
#         self.funnel = FaceFunnel()
        
    @property
    def hangles(self):
        return self._hangles
    
    @hangles.setter
    def hangles(self,hangles):
        self._hangles = hangles
        self.recalc_detectors()

    @property
    def langles(self):
        return self._langles
    
    @langles.setter
    def langles(self,langles):
        self._langles = langles
        self.recalc_detectors()
        
    def recalc_detectors(self, haar_file, lbp_file):
        self.haar_dtct = CascadeDetector(cascade_file = haar_file,
                                      min_size = self.min_size,
                                      min_neighbors = 20,
                                      scale_factor = 1.03,
                                      cascade_type = 'haar',
                                      thr = 0.4,
                                      angles = self.hangles)
    
        self.lbp_dtct = CascadeDetector(cascade_file = lbp_file,
                                   min_size = self.min_size,
                                   min_neighbors = 15,
                                   scale_factor = 1.04,
                                   cascade_type = 'lbp',
                                   thr = 0.4,
                                   angles = self.langles)
        
        
    def get_faces_list_in_photo(self, img):
        if self.hangles:
                haar_faces = self.haar_dtct.detectWithAngles(img, resolve = True)
        else:
            haar_faces = []
        lbp_faces = self.lbp_dtct.detectWithAngles(img, resolve = True)
        faces = resolve_boxes({'haar':haar_faces, 'lbp':lbp_faces}, min_overlap = 0.6)
        
        return faces
    
    def create_faces_file(self, fname, is_overwrite = False, target_file = None):
        '''
        Runs facial detection on fname (say a.jpg, or a.png), and creates a results file (a.faces.txt)
        
        target_file - override, and specify a specific target file
        is_overwrite - allow overwriting an existing results file
        '''
        faces = self.get_faces_list_in_photo(cv2.imread(fname))
        results_file = fname.rsplit('.',1)[0] + '.faces.txt' if target_file is None else target_file
        
        if os.path.exists(results_file) and not is_overwrite:
            print "Warning, faces result file", results_file, "exists"
        else:
            with open(results_file,'w') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',')
                header = ['x', 'y','dx','dy', 'score', 'angle', 'type']
                csv_writer.writerow(header)
                for face in faces:
                    csv_writer.writerow([str(i) for i in [int(face.x), int(face.y), int(face.dx), int(face.dy), face.score, face.angle, face.cascade_type]])
        return results_file
    
    def get_sub_images_from_file(self,original_image_file, faces_file):
        '''
        extracts all the face sub-images from an image file, based on the results in a faces file
        
        returns - the list of face images (numpy arrays)
        '''
        img = cv2.imread(original_image_file)
        faces_reader = csv.reader(open(faces_file))
        faces_reader.next()  # discard the headings
        padded_face_images = []
        for line in faces_reader:
            x, y, dx, dy, score, angle, cascade_type = line
            [x,y,dx,dy,score, angle] = [int(float(i)) for i in [x,y,dx,dy,score, angle]]
            face = CascadeResult(([x,y,dx,dy], score), cascade_type, angle)
            padded_face, bounding_box_in_padded_face, _, _ = extract_box(img, face, padding_factor = 0.25)
            padded_face_images.append(padded_face)
        return padded_face_images
    
    def create_sub_images_from_file(self, original_image_file, faces_file, target_folder = None, img_type = 'png'):
        '''
        reads a faces file, created by "self.create_faces_file" and extracts padded faces from the original image
        The faces will be created in the same folder as the faces file, unless specified otherwise by "target_folder"
        
        returns - the list of face files (strings)
        '''
        target_folder = os.path.split(faces_file)[0] if target_folder is None else target_folder
        padded_face_images = self.get_sub_images_from_file(original_image_file, faces_file)
        
        base_image_name = os.path.split(faces_file)[1].split('.')[0]
        face_files = []
        for n_face, face_img in enumerate(padded_face_images):
            face_file = os.path.join(target_folder, base_image_name + '_face_%d.%s' %(n_face, img_type))
            cv2.imwrite( face_file , face_img )
            face_files.append(face_file)
        return face_files
        