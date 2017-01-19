# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:32:40 2015

@author: teddy
"""

#import dlib
#import numpy as np
#from PIL import Image
#from visualization import drawbox,drawpoint 
# 
#img = Image.open('/media/teddy/data/wuxi_work_dir/order_image/17_0.jpg')
#img1 = np.array(img)
#detector = dlib.get_frontal_face_detector()
#bb = detector(img1,1)[0]
#left = bb.left()
#right = bb.right()
#up = bb.top()
#down = bb.bottom()
#drawbox(img,(left,right,up,down))
#
#
#facePredictor = '/home/teddy/deepface/models/dlib/shape_predictor_68_face_landmarks.dat'
#predictor = dlib.shape_predictor(facePredictor)
#
#points = predictor(img1, bb)
#p = list(map(lambda p: (p.x, p.y), points.parts()))
#left_eye_l = p[36]
#left_eye_r = p[39]
#left_eye = (np.array(left_eye_l)+np.array(left_eye_r))/2
#
#right_eye_l = p[42]
#right_eye_r = p[45]
#right_eye = (np.array(right_eye_l)+np.array(right_eye_r))/2
#drawpoint(img,left_eye)
#drawpoint(img,right_eye)
#img.show()


#import os
#print os.path.dirname('/home/teddy')


from PIL import Image
from naive_dlib import NaiveDlib
from glob import glob
model = '/home/teddy/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
img_dir = '/media/teddy/data/wuxi_work_dir/order_image'
prepocessor = NaiveDlib(model)

                        
count = 0
im_name = glob(img_dir+'/*jpg')
for name in im_name:
    img = Image.open(name)
    bb = prepocessor.getLargestFaceBoundingBox(img)
    if bb is None:
        print 'fuck '+name
    else:
        crop_img = prepocessor.prepocessImg('affine', 128, img, bb,offset = 0.3,gray = True,
                        outputDebug=True,outputprefix = str(count))
    count+=1
