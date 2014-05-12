'''
Created on May 7, 2014

@author: eran
'''
from adiencealign.common.landmarks import fidu_transform, shift_vector,\
    WEIGHTS3

class AffineAligner(object):
    def __init__(self, fidu_model_file, ):
        
        self.shift = ( 0.25, 0.25 )
        fidu_model = [(int(x.split(',')[1]),int(x.split(',')[2])) for x in file(fidu_model_file,'r')]
        self.fidu_model = shift_vector(fidu_model, self.shift)
        self.WEIGHTS3 = WEIGHTS3
        
    def align(self, img, fidu_points):
        # create bs1 image
        funneled_img, R = fidu_transform(self.fidu_model, fidu_points, WEIGHTS3, img, self.shift)
        return funneled_img, R      