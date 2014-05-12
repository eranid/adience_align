'''
Created on May 7, 2014

@author: eran
'''
import cv2
import pickle
import numpy as np
from shapely.geometry.polygon import Polygon
import math
from adiencealign.common.files import make_path
from adiencealign.common.images import pad_image_for_rotation
from adience.common.utils.utils import expand_path

class CascadeDetector(object):
    '''
    This is a haar cascade classifier capable of detecting in multiple angles
    '''
    def __init__(self, cascade_file = './resources/haarcascade_frontalface_default.xml', 
                 min_size = (10, 10),
                 min_neighbors = 20,
                 scale_factor = 1.04,
                 angles = [0],
                 thr = 0.4, 
                 cascade_type = 'haar'):
        '''
        cascade_type - is a string defining the type of cascade
        '''
        print expand_path('.')
        self.cascade_file = cascade_file.rsplit('/',1)[1]
        self._cascade_classifier = cv2.CascadeClassifier(cascade_file)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade_type = cascade_type
        self.angles = angles
        self.thr = thr
        
    def __str__(self):
        return ''.join([str(x) for x in ['cascade_file:',self.cascade_file,
                          ',scale_factor:',self.scale_factor,
                          ',min_neighbors:',self.min_neighbors,
                          ',min_neighbors:',self.min_neighbors,
                          ',cascade_type:',self.cascade_type
                          ]])
    
    def save_configuration(self, target_file):
        file_path = target_file.rsplit('/',1)[0]
        make_path(file_path)
        config = {'min_size':self.min_size, 'min_neighbours':self.min_neighbors, 'scale_factor':self.scale_factor, 'cascade_file':self.cascade_file}
        pickle.dump(obj=config, file = open(target_file,'w'), protocol = 2)
    
    @staticmethod
    def load_configuration(target_file):
        return pickle.load(open(target_file,'r'))
        
    def detectMultiScaleWithScores(self, img, scaleFactor = None, minNeighbors = None, minSize = None, flags = 4):
        scaleFactor = self.scale_factor if not scaleFactor else scaleFactor
        minNeighbors = self.min_neighbors if not minNeighbors else minNeighbors
        minSize = self.min_size if not minSize else minSize
        return self._cascade_classifier.detectMultiScaleWithScores(img, 
                                                                   scaleFactor = scaleFactor, 
                                                                   minNeighbors = minNeighbors, 
                                                                   minSize = minSize, 
                                                                   flags = flags)
        
    def detectWithAngles(self, img, angels = None, resolve = True, thr = None ):
        '''
        angles - a list of angles to test. If None, default to the value created at the constructor (which defaults to [0])
        resolve - a boolean flag, whether or not to cluster the boxes, and resolve cluster by highest score.
        thr - the maximum area covered with objects, before we break from the angles loop
        
        returns - a list of CascadeResult() objects
        '''
        
        if thr == None:
            thr = self.thr
            
        original_size = img.shape[0] * img.shape[0]
        if angels == None:
            angels = self.angles
            
        results = []
        total_area = 0
        for angle in angels:

            # the diagonal of the image is the diameter of the rotated image, so the big_image needs to bound this circle
            # by being that big

            big_image, x_shift, y_shift, diag, rot_center = pad_image_for_rotation(img)
            
            # find the rotation and the inverse rotation matrix, to allow translations between old and new coordinates and vice versa
            rot_mat = cv2.getRotationMatrix2D(rot_center, angle, scale = 1.0)
            inv_rot_mat = cv2.invertAffineTransform(rot_mat)
            
            # rotate the image by the desired angle
            rot_image = cv2.warpAffine(big_image, rot_mat, (big_image.shape[1],big_image.shape[0]), flags=cv2.INTER_CUBIC)
            faces = self.detectMultiScaleWithScores(rot_image, scaleFactor = 1.03, minNeighbors = 20, minSize = (15,15), flags = 4)
            for face in faces:
                xp = face[0][0]
                dx = face[0][2]
                yp = face[0][1]
                dy = face[0][3]
                score = face[1]
                dots = np.matrix([[xp,xp+dx,xp+dx,xp], [yp,yp,yp+dy,yp+dy], [1, 1, 1, 1]])
                # these are the original coordinates in the "big_image"
#                print dots
                originals_in_big = inv_rot_mat * dots
#                print originals_in_big
                shifter = np.matrix([[x_shift]*4, [y_shift]*4])
#                print shifter
                # these are the original coordinate in the original image
                originals = originals_in_big - shifter
#                print originals
                points = np.array(originals.transpose())
                x = points[0,0]
                y = points[0,1]
                box_with_score = ([x,y,dx,dy], score)
                
                cascade_result = CascadeResult.from_polygon_points(points, score, self.cascade_type)
#                print cascade_result
                
                results.append(cascade_result)
            
            #################
            # test and see, if we found enough objects, break out and don't waste our time
                total_area += cascade_result.area     
            
        if resolve:
            return resolve_angles(results, width = img.shape[1], height = img.shape[0])
        else:
            return results

class BoxInImage(object):
    def __init__(self, originals, dx, dy, score = None, angle = 0):
        self.originals = originals
        self.dx = dx
        self.dy = dy
        self.score = score
        self.angle = angle
    
    def __str__(self):
        return ",".join([str(x) for x in [self.originals, self.dx, self.dy, self.score, self.angle]])

def resolve_angles(list_of_results, width, height, thr = 0.3):
    '''
    we want to cluster the boxes into clusters, and then choose the best box in each cluster by score
        * thr - decides what the maximum distance is for a box to join a cluster, in the sense of how much of it's area is covered by the best box in the cluster
              note, that two squares, centered, with 45 degrees rotation, will overlap on 77% of their area (thr == 0.22)    
    '''
    clusters = []
    for box in list_of_results:
#        total_polygon = Polygon([(0,0), (width,0), (width,height), (0,height)])
#        if box.polygon.intersection(total_polygon).area < box.area:
#            # this means the box is outside the image somehow
#            continue
    
        area = box.area
        closest_cluster = None
        dist_to_closest_cluster = 1.0
        for n,cluster in enumerate(clusters):
            dist = 1.0
            for cluster_box in cluster:
                local_dist = 1.0 - box.overlap(cluster_box)/area
                dist = min(dist, local_dist)
            if dist < dist_to_closest_cluster:
                dist_to_closest_cluster = dist
                closest_cluster = n
        if closest_cluster == None or dist_to_closest_cluster > thr:
            # no good cluster was found, open a new cluster
            clusters.append([box])
        else:
            clusters[n].append(box)
    
    centroids = []
    for cluster in clusters:
        centroids.append(sorted(cluster,key=lambda x: x.score)[-1])
    
    return centroids
            
        
            
        
    
def resolve_boxes(dict_of_list_of_cascade_results, min_overlap = 0.7):
    '''
    Say you tried two different cascades to detect faces.
    enter a dictionary (the key is a string describing a cascade type) of detected objects
    This function returns a unified results list, where it resolves overlapping boxes, and chooses one of them.
    
    The bigger boxes are selected instead of smaller ones, whether they contain them, or enough of them, determined by min_overlap
    
    '''
    final_faces = []
    for cascade_str, faces in dict_of_list_of_cascade_results.iteritems():
        # go through each cascade type
        for face in faces:
            if type(face) == CascadeResult:
                new_res = face
            else:
                new_res = CascadeResult(face,cascade_type = cascade_str)
            to_add = True
            for old_index,old_res in enumerate(final_faces):
                ratio = new_res.area / old_res.area
                if ratio >1.0:
                    # new_box is bigger
                    if new_res.overlap(old_res)/old_res.area > min_overlap:
                        # the new box contains the old one, we want to replace it:
                        final_faces[old_index] = new_res
                        to_add = False
                        break
                if ratio <=1.0:
                    # the new_box is smaller
                    if new_res.overlap(old_res)/new_res.area > min_overlap:
                        # the old box contains the new one, we therefore dont need to add the new box:
                        to_add = False
                        break
            if to_add:
                # if there was no hit, this is a new face, we can add it
                final_faces.append(new_res)
    return final_faces

def most_centered_box( cascade_results, ( rows, cols ) ):
    best_err = 1e10
    for i, cascade in enumerate( cascade_results ):
        err = ( cascade.x + cascade.dx / 2 - cols / 2 ) ** 2 + ( cascade.y + cascade.dy / 2 - rows / 2 ) ** 2
        if err < best_err:
            index = i
    return cascade_results[ index ]
        
class CascadeResult(object):
    def __init__(self, box_with_score, cascade_type = None, angle = 0):
        self.x = box_with_score[0][0]
        self.y = box_with_score[0][1]
        self.dx = box_with_score[0][2]
        self.dy = box_with_score[0][3]
        self.score = box_with_score[1]
        self.cascade_type = cascade_type
        self.angle = angle
        
    @staticmethod
    def from_polygon_points(points, score, cascade_type = None):
        '''
        an alternative generator, allows giving the polygon points instead of [x,y,dx,dy]
        '''
        x = points[0,0]
        y = points[0,1]
        top = points[1,] - points[0,]
        left = points[3,] - points[0,]
        dx = math.sqrt(sum([i*i for i in top]))
        dy = math.sqrt(sum([i*i for i in left]))
        angle = math.atan(float(top[1])/top[0]) * 180 / math.pi if top[0] != 0 else (970 if top[1] >0 else -90)
        return CascadeResult(([x,y,dx,dy],score), cascade_type, angle)
        
    
    def __str__(self):
        return ''.join([str(x) for x in ['center:',self.center,
                                         ',\nx:',self.x,
                                         ',\ny:',self.y,
                                         ',\ndx:',self.dx,
                                         ',\ndy:',self.dy,
                                         ',\nscore:',self.score,
                                         ',\nangle:',self.angle,
                                         ',\ncascade_type:',self.cascade_type,
                                         ',\npoints_int:\n',self.points_int
                                         ]])
    
    @property
    def points(self):
        x = self.x
        y = self.y
        dx = self.dx
        dy = self.dy
        a = self.angle/180.0*math.pi
        dots = np.matrix([[x,y,1],[x+dx,y,1],[x+dx,y+dy,1],[x,y+dy,1]])
        dots = dots.transpose()
        rot_mat = cv2.getRotationMatrix2D((dots[0,0],dots[1,0]), -self.angle, scale = 1.0)
        points = rot_mat * dots
        points = points.transpose() 
        return points

    @property
    def center(self):
        return tuple(int(x) for x in (self.points.sum(0)/4.0).tolist()[0])
    
    @property
    def points_int(self):
        return self.points.astype(int)
        
    @property
    def score_with_type(self):
        if self.cascade_type:
            return self.cascade_type + ' ' + str(self.score)
        else:
            return str(self.score)
        
    @property
    def filename_encode(self):
        
        return '_'.join([str(x) for x in ['loct'] + self.cvformat_result[0] + ['ang', int(self.angle),self.cascade_type, self.score]]) 
    
    @property
    def cvformat_coords(self):
        if self.angle == 0:
            return [int(x) for x in [self.x, self.y, self.dx, self.dy]]
        else:
            raise Exception('cannot return [x,y,dx,dy] for a box with angle, use cvformat_result() instead')
        
    @property
    def cvformat_result(self):
        return ([int(x) for x in [self.x, self.y, self.dx, self.dy]], self.score, self.angle)
    
#    @property
#    def rot_matrix(self):
#        return array([[cos(math.radians(self.angle)), -sin(math.radians(self.angle))], 
#                   [sin(math.radians(self.angle)),  cos(math.radians(self.angle))]])

    @property
    def top_left(self):
        return tuple(self.points[0,].tolist()[0])
    
    @property
    def top_right(self):
        return tuple(self.points[1,].tolist()[0])

    @property
    def bottom_right(self):
        return tuple(self.points[2,].tolist()[0])

    @property
    def bottom_left(self):
        return tuple(self.points[3,].tolist()[0])
    
    @property
    def polygon(self):
        return Polygon([self.top_left, self.top_right, self.bottom_right, self.bottom_left])
        
    def overlap(self, otherRect):      
        return float(self.polygon.intersection(otherRect.polygon).area)
    
    @property
    def area(self):
        return float(self.polygon.area)
    
    def __gt__(self,b):
        return self.area>b.area
    def __ge__(self,b):
        return self.area>=b.area
    def __lt__(self,b):
        return self.area<b.area    
    def __le__(self,b):
        return self.area<=b.area
            
        