'''
Created on May 7, 2014

@author: eran
'''
import subprocess
import os

def detect_landmarks(fname, max_size = 400*400, min_size = 50*50, fidu_exec_dir = os.path.abspath('../resources/')):
    '''
    optional flags:
    max_size - If exceeds this pixel size, image is resized to that before landmark detection
    min_size - If below this pixel size, image is resized to that before landmark detection
    '''
    max_size = str(max_size) if max_size is not None else ''
    min_size = str(min_size) if min_size is not None else ''
    fidu_cmd = './FiducialFaceDetector.sh Face_small_146filters_-0.65thr.xml %s %s %s' %(fname, min_size, max_size)    
    print fidu_cmd
    x = subprocess.call(fidu_cmd, shell=True, cwd = fidu_exec_dir)
    return
