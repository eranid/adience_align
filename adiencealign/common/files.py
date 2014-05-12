'''
Created on May 7, 2014

@author: eran
'''
import os
import shutil
from os.path import expanduser

def make_path(path, delete_content_if_exists = False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif delete_content_if_exists:
        shutil.rmtree(path)
        
def expand_path(file_or_path):
    if file_or_path.startswith('~/'):
        home = expanduser("~")
        file_at = os.path.join(home,file_or_path[2:])
        return file_at
    else:
        return file_or_path        