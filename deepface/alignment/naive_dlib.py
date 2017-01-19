import dlib,time
from dlib import rectangle
import numpy as np
import os
from canonical_face import crop_simi,crop_only
from visualization import drawbox,drawpoint

class NaiveDlib:

    def __init__(self, facePredictor = None):
        """Initialize the dlib-based alignment."""
        self.detector = dlib.get_frontal_face_detector()
        if facePredictor != None:
            self.predictor = dlib.shape_predictor(facePredictor)
        else:
            self.predictor = None 

    def getAllFaceBoundingBoxes(self, img):
        return self.detector(np.array(img), 1)

    def getLargestFaceBoundingBox(self, img):    #process only one face pertime
        img = img.resize((img.size[0]/2,img.size[1]/2))        
        faces = self.detector(np.array(img), 1)
        if len(faces) > 0:
            faces = max(faces, key=lambda rect: rect.width() * rect.height())
            left = faces.left()*2
            top = faces.top()*2
            right = faces.right()*2
            bottom = faces.bottom()*2
            faces = rectangle(left,top,right,bottom)
            return faces

        # if len(faces) > 0:
        #     return max(faces, key=lambda rect: rect.width() * rect.height())

    def align(self, img, bb):
        points = self.predictor(np.array(img), bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))

    def prepocessImg(self, method, size, img, bb,offset = 0.3,gray = True,
                      outputDebug=False,outputprefix = None):
        """
        the image is load by PIL image directly
        bb is rct object of dlib
        """
        if method == 'crop':
            crop_img = crop_only(img,bb.left(),bb.top(),bb.width(),bb.height(),offset,size)
        elif method == 'affine':
            if self.predictor == None:
                raise Exception("Error: method affine should initial with an facepredictor.")
            alignPoints = self.align(img, bb)
            (xs, ys) = zip(*alignPoints)
            (l, r, t, b) = (min(xs), max(xs), min(ys), max(ys))
            w,h = img.size
            if 0 <= l and r <= w and 0 <= t and b <= h:             
                left_eye_l = alignPoints[36]
                left_eye_r = alignPoints[39]
                left_eye = (np.array(left_eye_l)+np.array(left_eye_r))/2
                right_eye_l = alignPoints[42]
                right_eye_r = alignPoints[45]
                right_eye = (np.array(right_eye_l)+np.array(right_eye_r))/2
                crop_img = crop_simi(img,left_eye,right_eye,(offset,offset),(size,size))
            else:
                print("Warning: Unable to align and crop to the "
                  "face's bounding box.")
                return None
        else:
            raise Exception("Error: method affine should initial with an facepredictor.")
        if outputDebug:
            dirname = './aligndebug'
            if not os.path.exists(os.path.abspath(dirname)):
                os.mkdir(dirname)
            drawbox(img,(bb.left(),bb.right(),bb.top(),bb.bottom()))
            if method == 'affine':
                drawpoint(img,left_eye)
                drawpoint(img,right_eye)
            img.save('{}/{}_annotated.jpg'.format(dirname,outputprefix))
            crop_img.save('{}/{}_crop.jpg'.format(dirname,outputprefix))
        return crop_img
