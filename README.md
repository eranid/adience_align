adience_align
========

This project provides alignment tools for faces, to be used as a preprocessing step before computer vision tasks on face images.

Homepage for the project: http://www.openu.ac.il/home/hassner/Adience/


See the test for example usage.

Specificaly, the "pipeline" test, shows how to use the full process (just remember to change the location of the model files to where you stor the *.xml and other model files)

Installation
=========
in the root of the repository:

```
python setup.py sdist
sudo pip install dist/adience-<version_number>.tar.gz
```



CopyRight
=========
(contact: Eran Eidinger (eran@adience.com), Roee Enbar (roee.e@adience.com))

See the LICENSE.txt file (basically, an MIT license).


With any publication that uses this alignment code, or it's derivative, we kindly ask that you cite the paper:
E. Eidinger, R. Enbar, and T. Hassner, Age and Gender Estimation of Unfiltered Faces, Transactions on Information Forensics and Security (IEEE-TIFS), special issue on Face Recognition in the Wild

For more details, please see:
http://www.openu.ac.il/home/hassner/Adience/publications.html

Compilation notes
========
NOTE: The shared objects were compiled for linux 64bit on Ubuntu 13.10

We will release the source code for the shared object in the near future

