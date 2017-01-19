# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:34:30 2015

@author: teddy
"""

import numpy as np
import os
from PIL import Image

from alignment import NaiveDlib
from caffewrapper import Wrapper

im_dir = '/media/teddy/data/shandong_work_dir/shandong/negative/id/10163.jpg'
dlibmodel = '/home/teddy/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
caffemodel = '/home/teddy/deepface/models/deepface/deepface.caffemodel'
netdef = '/home/teddy/deepface/models/deepface/val.prototxt'

aligner = NaiveDlib(dlibmodel)
extractor = Wrapper(netdef,caffemodel)

img = Image.open(im_dir)
bb = aligner.getLargestFaceBoundingBox(img)
crop_img = aligner.prepocessImg('affine', 114, img, bb,offset = 0.3,gray = True)
fea = extractor.extract_batch([crop_img])
