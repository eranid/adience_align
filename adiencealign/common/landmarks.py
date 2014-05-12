'''
Created on May 7, 2014

@author: eran
'''

import csv
import cv2
from numpy import linalg
import numpy as np


WEIGHTS3 = [11.1088851746,15.8721645013,12.3189439894,15.9467104922,13.9265119716,17.2447706133,11.4118267639,17.0728365324,12.7831886739,17.1908773151,9.6639887492,13.8443342456,8.76890470223,11.4441704453,7.52083144762,10.3245662427,6.35563072919,7.55739887985,6.42340544936,7.48786881875,10.8720924456,8.1349958353,12.3664410374,9.58137800608,6.29390307208,9.47697088783,8.49859202931,9.43946799727,7.92920023102,10.6126442536,10.2953809171,11.299323189,11.1181958685,12.9374719654,12.3764338392,14.7823225327,13.086272904,16.0571795811,15.079169884,17.5936174739,8.39112414861,7.68532826996,8.89386612449,8.70173923211,10.0826620269,8.70286074207,8.13121344224,9.80805203263,7.76044090777,9.2502084627,7.61334683331,10.4813589698,8.64831020289,11.0452512508,9.19528177019,13.0171747152,10.1204323102,14.0189765809,11.0232436734,14.7355286373,12.4881579947,15.4279914333,11.5785971474,16.7942051778,12.4916161829,17.57726411,14.3422306002,19.3015061859,16.3109851665,23.7227227093,17.7687071538,22.6848438204,14.9879312002,18.6763354368,12.927920123,17.7652660198,10.3584444834,15.5584775245,10.660322225,15.4351684107,11.6468441007,13.7962556973,12.9019472625,16.6407866045,13.1946878458,16.4137518526,9.86525395127,11.6687513083,10.4858060411,12.8407630953,9.24210197996,10.9728479778,9.37639005327,12.3418022852,12.2786533953,12.0629300205,14.8495857728,15.4667996708,14.7414922143,15.2761005039,8.5837102275,10.8010609515,6.55275411638,14.4240347981,10.4200283162,17.6888997346,11.4480670185,22.4669420211,13.1705102756,29.3073334802,16.9922725597,35.4031969543,18.7102372238,41.7466671473,21.7036998929,47.1495172267,24.4179633642,51.9023425203,26.6870471848,57.5921966087,7.71654443362,18.3796425232,9.84932443383,23.3915673615,15.7135746598,31.5768046636,18.159161567,39.0502675506,20.5154926286,44.6961338521,22.8541610324,50.9071591504,26.5569627651,54.4338495899,29.1062390164,61.5990210977]

def read_fidu(fidu_name):
    fidu_reader = csv.reader(open(fidu_name))
    try:
        line = fidu_reader.next()
        score, yaw_angle = line
        fidu_reader.next()
        fidu_points = [[int(float(field)) for field in row[-2:]] for row in fidu_reader]
        return int(score), int(yaw_angle), fidu_points
    except:
        if line[0] == 'nothing found':
            return -1000, None, None
        else:
            raise Exception('Corrupt Fidu File ' + fidu_name)
        
        
def draw_fidu(img, fidu_points, radius = 3, color = (255,0,0), thickness = 1, draw_numbers_color = None):
    '''
    set draw_numbers_color = (COLOR_BLUE), for example, to draw the point numbers
    '''
    for n,point in enumerate(fidu_points):
        cv2.circle(img, tuple([int(x) for x in point]),radius,color, thickness)
        if draw_numbers_color:
            cv2.putText( img, 
              '%d' %n,
             tuple([int(point[0])-4, int(point[1])-4]), 
             cv2.FONT_HERSHEY_COMPLEX, 
             0.35, 
             draw_numbers_color, 
             thickness = 1 )
            
            

def _compute_affine_transform_cvpy(refpoints, points, w = None): # Copied from the book
    if w == None:
        w = [1] * (len(points) * 2)
    assert(len(w) == 2*len(points))
    y = []
    for n, p in enumerate(refpoints):
        y += [p[0]/w[n*2], p[1]/w[n*2+1]]
    A = []
    for n, p in enumerate(points):
        A.extend([ [p[0]/w[n*2], p[1]/w[n*2], 0, 0, 1/w[n*2], 0], [0, 0, p[0]/w[n*2+1], p[1]/w[n*2+1], 0, 1/w[n*2+1]] ])
    
    lstsq = linalg.lstsq(A,y)
    h11, h12, h21, h22, dx, dy = lstsq[0]
    err = lstsq[1]

    R = np.array([[h11, h12, dx], [h21, h22, dy]])
    return R, err


def _compute_affine_transform_ocvlsq(refpoints, points, w = None):
    if w == None:
        w = [1] * (len(points) * 2)
    assert(len(w) == 2*len(points))
    y = []
    for n, p in enumerate(refpoints):
        y += [p[0]/w[n*2], p[1]/w[n*2+1]]
    A = []
    for n, p in enumerate(points):
        A.extend([ [p[0]/w[n*2], p[1]/w[n*2], 0, 0, 1/w[n*2], 0], [0, 0, p[0]/w[n*2+1], p[1]/w[n*2+1], 0, 1/w[n*2+1]] ])
    
    lstsq = cv2.solve(np.array(A), np.array(y), flags=cv2.DECOMP_SVD)
    h11, h12, h21, h22, dx, dy = lstsq[1]
    err = 0#lstsq[1]

    #R = np.array([[h11, h12, dx], [h21, h22, dy]])
    # The row above works too - but creates a redundant dimension
    R = np.array([[h11[0], h12[0], dx[0]], [h21[0], h22[0], dy[0]]])
    return R, err
            
def fidu_transform(fidu_model, fidu_points, weights, img, shift=(0.0,0.0), use_ocvlsq=False):
    FIDU_SIZE = 544
    
    SHIFT_Y = int(FIDU_SIZE * shift[0])
    SHIFT_X = int(FIDU_SIZE * shift[1])
    if not use_ocvlsq:
        R, err = _compute_affine_transform_cvpy(fidu_model, fidu_points, weights)
    else:
        R, err = _compute_affine_transform_ocvlsq(fidu_model, fidu_points, weights)

    funneled_img = cv2.warpAffine(img, R, (FIDU_SIZE + SHIFT_Y*2, FIDU_SIZE + SHIFT_X*2), flags=cv2.INTER_CUBIC)
    return funneled_img, R


def shift_vector(points, shift):
    FIDU_SIZE = 544
    
    SHIFT_Y = int(FIDU_SIZE * shift[0])
    SHIFT_X = int(FIDU_SIZE * shift[1])
    SHIFT = (SHIFT_X, SHIFT_Y)
    s_points = [(p[0] + SHIFT[0], p[1] + SHIFT[1]) for p in points]
    return s_points

def unwarp_fidu(orig_fidu_points, unwarp_mat):    
    points = np.array(orig_fidu_points).T
    points_h = np.r_[points, np.ones((1,points.shape[1]))]
    orig_fidu_points_in_aligned = np.dot(unwarp_mat,points_h).T.astype(np.int16)
    return orig_fidu_points_in_aligned 
