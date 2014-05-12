'''
Created on May 7, 2014

@author: eran
'''
import math
import numpy as np
import cv2



def pad_image_for_rotation(img):
    # we pad the image so, when we rotate it, it would never be clipped
    rot_y,rot_x = img.shape[:2]
    rot_x = int(rot_x / 2.0)
    rot_y = int(rot_y / 2.0)
    diag = int(math.sqrt(sum([math.pow(x,2) for x in img.shape])))
    diag = int(math.ceil(diag / 2.0) * 2.0) # make sure it is even
    if len(img.shape) == 3:
        big_image = np.zeros((diag, diag, 3), dtype=np.uint8)
    else:
        big_image = np.zeros((diag, diag), dtype=np.uint8)
    x_shift = int(diag/2-rot_x)
    y_shift = int(diag/2-rot_y)
    # the shift of the old image within big_image
    if len(img.shape) == 3:
        big_image[y_shift :y_shift+img.shape[0], x_shift:x_shift+img.shape[1], :] = img
    else:
        big_image[y_shift :y_shift+img.shape[0], x_shift:x_shift+img.shape[1]] = img
    
    # the rotation center is no the radius (half the old image diagonal)
    rot_center = diag/2, diag/2
    return big_image, x_shift, y_shift, diag, rot_center


def extract_rect(img, rect, factor = 0.2):
    (x,y,dx,dy) = rect
    new_x = max(0, int(x-dx*factor))
    new_y = max(0, int(y-dy*factor))
    new_dx = min(int(dx+2*factor*dx), img.shape[1] - new_x)
    new_dy = min(int(dy+2*factor*dy), img.shape[0] - new_y)
    Dx = x - new_x
    Dy = y - new_y
    #return [new_x, new_y, new_dx, new_dy]
    return img[new_y:new_y+new_dy,new_x:new_x+new_dx,:], Dx, Dy


def extract_box(img, box, padding_factor = 0.2):
    '''
    we can search for whatever we want in the rotated bordered image, 
    
    Any point found can be translated back to the original image by:
    1. adding the origins of the bordered area,
    2. rotating the point using the inverse rotation matrix    
    
    '''
    
    if box.angle != 0:
        
        b_w = max(img.shape)*2
        b_h = b_w
        dx_center = b_w / 2 - box.center[0]
        dy_center = b_h / 2 - box.center[1]
        new_img = np.zeros((b_w, b_h, 3), dtype = img.dtype)
        new_img[dy_center:(dy_center + img.shape[0]), dx_center:(dx_center + img.shape[1]), :] = img
        
        box_in_big_image = box.points + np.c_[np.ones((4,1)) * dx_center, np.ones((4,1)) * dy_center]

        rot_mat = cv2.getRotationMatrix2D((b_w/2, b_h/2), box.angle, scale = 1.0)
        inv_rot_mat = cv2.invertAffineTransform(rot_mat)
        rot_image = cv2.warpAffine(new_img, rot_mat, (new_img.shape[1],new_img.shape[0]), flags=cv2.INTER_CUBIC)
        box_UL_in_rotated = (rot_mat * np.matrix([box_in_big_image[0,0], box_in_big_image[0,1], 1]).transpose()).transpose().tolist()[0] 
        box_coords_in_rotated = np.matrix(np.c_[box_in_big_image, np.ones((4,1))]) * rot_mat.T
        box_coords_in_rotated = box_coords_in_rotated[0,:].tolist()[0] + [box.dx, box.dy]
    else:
        rot_mat = cv2.getRotationMatrix2D(box.center, box.angle, scale = 1.0)
        inv_rot_mat = cv2.invertAffineTransform(rot_mat)
        # for efficiency
        rot_image = img.copy()
        box_UL_in_rotated = (rot_mat * np.matrix([box.points[0,0], box.points[0,1], 1]).transpose()).transpose().tolist()[0] 
        box_coords_in_rotated = box_UL_in_rotated + [box.dx, box.dy]
    
    img_with_border, Dx, Dy = extract_rect(rot_image, box_coords_in_rotated, padding_factor)
    box_coords_in_bordered = [Dx, Dy] + [box.dx, box.dy]
    border_UL_in_rotated = [box_UL_in_rotated[0]-Dx, box_UL_in_rotated[1]-Dy]
    
    return img_with_border, box_coords_in_bordered, border_UL_in_rotated, inv_rot_mat