#! /usr/bin/env python
from distutils.core import setup
from setuptools import find_packages
__version__ = 0.1

setup(
    name='adiencealign',
    version=__version__,
    url='http://www.openu.ac.il/home/hassner/Adience/index.html',
    packages = find_packages(),
    package_data={'': ['*.xml', '*.png', '*.jpg', 'FiducialFaceDetector*']},
    scripts=[ ],
    entry_points = {'console_scripts':[ ]},
    description='Adience Face alignment library',
    install_requires=[                      
                      'numpy',
                      'scipy',
                      'shapely'
                      ]
)

