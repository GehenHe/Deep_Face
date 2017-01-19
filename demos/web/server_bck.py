#!/usr/bin/env python2


import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))
import txaio
txaio.use_twisted()


from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.python import log
from twisted.internet import reactor

import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64
import pickle, time 
from glob import glob
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm



modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
deepfaceModelDir = os.path.join(modelDir, 'deepface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--pretrainModel', type=str, help="Path to caffe model.",
                    default=os.path.join(deepfaceModelDir, 'deepface.caffemodel'))
parser.add_argument('--mean', type=str, help="Path to mean file.",
                    default=os.path.join(deepfaceModelDir, 'mean.npy'))
parser.add_argument('--netdef', type=str, help="Path to the define of the net.",
                    default=os.path.join(deepfaceModelDir, 'val.prototxt'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=128)
parser.add_argument('--gpu', type=bool, default=True)
parser.add_argument('--gray', type=bool, default=True)
args = parser.parse_args()

from search_engine import Search_Engine
from deepface.alignment import NaiveDlib  # Depends on dlib.
from deepface import Wrapper

aligner = NaiveDlib(args.dlibFacePredictor)
mean_npy = np.load(args.mean)
extractor = Wrapper(args.netdef, args.pretrainModel, mean=mean_npy, gpu=args.gpu)


class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol(WebSocketServerProtocol):

    def __init__(self):
        self.images = {}
        self.training = True
        print 'init {0}'.format(self.training)
        self.people = []
        self.knn = None
        self.temp_save = {}
        self.thre = 1.0
        
    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = True

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "ALL_STATE":  #load all state training equal false
            print "type:ALL_STATE"
            self.loadState(msg['images'], msg['training'], msg['people'])
        elif msg['type'] == "NULL":
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            print "type:FRAME"
            self.processFrame(msg['dataURL'], msg['identity'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            print('type:training'.format(self.training))
            if not self.training:
                self.trainKNN()
        elif msg['type'] == "ADD_PERSON":
            self.people.append(msg['val'].encode('ascii', 'ignore'))
            print('type:add_person+{}'.format(msg['val']))
        elif msg['type'] == "UPDATE_IDENTITY":
            print "type:UPDATE_IDENTITY"
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                self.images[h].identity = msg['idx']
                if not self.training:
                    self.trainKNN()
            else:
                print("Image not found.")
        elif msg['type'] == "REMOVE_IMAGE":
            h = msg['hash'].encode('ascii', 'ignore')
            if h in self.images:
                del self.images[h]
                del self.temp_save[h]
                if not self.training:
                    self.trainKNN()
            else:
                print("Image not found.")
        elif msg['type'] == 'REQ_TSNE':
            print "type:TSNE"
            self.sendTSNE(msg['people'])
        elif msg['type'] == 'INIT':
            print "type:INIT"
            # img_dir = "/home/gehen/deepface_super/demos/web/ID/adams.jpg"
            # self.people.append("a".encode('ascii', 'ignore'))
            # self.Add_ID(img_dir,0)
            # self.Add_ID(img_dir,0)
            # self.Add_ID(img_dir,0)
            # self.Add_ID(img_dir,0)
            # self.trainKNN()
            # print "people {}".format(self.people)

            self.Load_ID("/home/gehen/deepface_super/demos/web/ID")
            if not self.training:
                self.trainKNN()

            

        elif msg['type'] == 'SAVE':
            print "type:SAVE"
            pickle.dump((self.temp_save,self.people),open('data.pkl','w')) 
            
            
        elif msg['type'] == 'LOAD':
            print "type:LOAD"
            temp_save,people=pickle.load(open('data.pkl','r'))
            for person in people:
                if person not in self.people:
                    self.people.append(person)
                    msg = {
                        "type": "NEW_PEOPLE",
                        "person": person,
                    }
                    self.sendMessage(json.dumps(msg))
            for msg in temp_save.values():
                if not self.images.has_key(msg['hash']):
                    self.temp_save[msg['hash']] = msg
                    self.images[msg['hash']] = Face(msg['representation'], msg['identity'])
                    self.sendMessage(json.dumps(msg))
            self.trainKNN()
        elif msg['type'] == 'SET_THRESH':
            self.thre = float(msg['val'])
            print('the new thre is {0:f}'.format(self.thre))
            self.trainKNN()
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def loadState(self, jsImages, training, jsPeople):
        self.training = training
        print 'loadstate {0}'.format(self.training)

        for jsImage in jsImages:
            h = jsImage['hash'].encode('ascii', 'ignore')
            self.images[h] = Face(np.array(jsImage['representation']),
                                  jsImage['identity'])

        for jsPerson in jsPeople:
            self.people.append(jsPerson.encode('ascii', 'ignore'))

        if not training:
            self.trainKNN()

    def getData(self):
        X = []
        y = []
        for img in self.images.values():
            X.append(img.rep)
            y.append(img.identity)

        numIdentities = len(set(y + [-1])) - 1
        if numIdentities == 0:
            return None
            
        X = np.vstack(X)
        y = np.array(y)
        return (X, y)

    def sendTSNE(self, people):
        d = self.getData()
        if d is None:
            return
        else:
            (X, y) = d

        X_pca = PCA(n_components=50).fit_transform(X, X)
        tsne = TSNE(n_components=2, init='random', random_state=0)
        X_r = tsne.fit_transform(X_pca)

        yVals = list(np.unique(y))
        colors = cm.rainbow(np.linspace(0, 1, len(yVals)))

        # print(yVals)

        plt.figure()
        for c, i in zip(colors, yVals):
            name = "Unknown" if i == -1 else people[i]
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=name)
            plt.legend()

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='jpg')
        imgdata.seek(0)

        content = 'data:image/png;base64,' + \
                  urllib.quote(base64.b64encode(imgdata.buf))
        msg = {
            "type": "TSNE_DATA",
            "content": content
        }
        plt.close()
        self.sendMessage(json.dumps(msg))

    def trainKNN(self):
        print("+ Training KNN on {} labeled images.".format(len(self.images)))
        d = self.getData()
        if d is None:
            self.knn = None
            return
        else:
            (X, y) = d
            numIdentities = len(set(y + [-1]))
            if numIdentities <= 1:
                return
            self.knn = Search_Engine(thre=self.thre,rej='mean')
            self.knn.fit(X,y)

    def processFrame(self, dataURL, identity):
        print "process frame, identity is {}".format(identity)
        time0 = time.time()
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)
        if not self.training:                       #train button on tle left is false
            annotatedFrame = np.copy(np.array(img))   # this if for show result in video,only untraining time
        identities = []
        
        bb = aligner.getLargestFaceBoundingBox(img)
        
        bbs = [bb] if bb is not None else []
        for bb in bbs:
            alignedFace =  aligner.prepocessImg('affine', 128, img, bb,offset = 0.3)

            if alignedFace is None:
                continue
            phash = str(imagehash.phash(alignedFace))
            if phash in self.images:
                identity = self.images[phash].identity  #same pic same id
            else:
                if args.gray:
                    in_face = alignedFace.convert('L')
                    in_face = np.array(in_face,dtype=np.float32)
                    in_face = in_face[:,:,np.newaxis]
                else:
                    in_face = np.array(alignedFace,dtype=np.float32)
                rep = extractor.extract_batch([in_face])[0]
                
                if self.training:        
                    self.images[phash] = Face(rep, identity)
                    alignedFace = np.array(alignedFace.resize((96,96)))[:,:,(2,1,0)]                    
                    content = [str(x) for x in alignedFace.flatten()]
                    msg = {
                        "type": "NEW_IMAGE",
                        "hash": phash,
                        "content": content,
                        "identity": identity,
                        "representation": rep.tolist()
                    }
                    self.temp_save[phash] = msg
                    self.sendMessage(json.dumps(msg))
                else:
                    if len(self.people) == 0:
                        identity = -1
                    elif len(self.people) == 1:
                        identity = 0
                    elif self.knn:
                        print len(rep)
                        identity,dis = self.knn.predict(rep)                      
                        # print('the nearest dis is {0:f}'.format(dis))
                    else:
                        print("no match")
                        identity = -1
                    if identity not in identities:
                        identities.append(identity)
                

            if not self.training:
                bl = (bb.left(), bb.bottom())
                tr = (bb.right(), bb.top())
                cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                              thickness=3)
                if identity == -1:
                    if len(self.people) == 1:
                        name = self.people[0]
                    else:
                        name = "Unknown"
                else:
                    name = self.people[identity]
                cv2.putText(annotatedFrame, name, (bb.left(), bb.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                            color=(152, 255, 204), thickness=2)

        if not self.training:
            msg = {
                "type": "IDENTITIES",
                "identities": identities
            }
            print "identies is {}".format(identities)
            self.sendMessage(json.dumps(msg))

            plt.figure()
            plt.imshow(annotatedFrame)
            plt.xticks([])
            plt.yticks([])

            imgdata = StringIO.StringIO()
            plt.savefig(imgdata, format='jpg')
            imgdata.seek(0)
            content = 'data:image/png;base64,' + \
                urllib.quote(base64.b64encode(imgdata.buf))
            msg = {
                "type": "ANNOTATED",
                "content": content
            }
            plt.close()
            self.sendMessage(json.dumps(msg))
        time1 = time.time()
        print 'extract took {:.4f}s'.format(time1-time0)

    def Add_ID(self, img_dir, identity):
        time0 = time.time()
        img = Image.open(img_dir)
        if not self.training:                       #train button on tle left is false
            annotatedFrame = np.copy(np.array(img))   # this if for show result in video,only untraining time
        identities = []
        
        bb = aligner.getLargestFaceBoundingBox(img)
        
        # bbs = [bb] if bb is not None else []
        # for bb in bbs:
        alignedFace =  aligner.prepocessImg('affine', 128, img, bb,offset = 0.3)

        # if alignedFace is None:
        #     continue
        phash = str(imagehash.phash(alignedFace))
        if phash in self.images:
            identity = self.images[phash].identity  #same pic same id
        else:
            if args.gray:
                in_face = alignedFace.convert('L')
                in_face = np.array(in_face,dtype=np.float32)
                in_face = in_face[:,:,np.newaxis]
            else:
                in_face = np.array(alignedFace,dtype=np.float32)
            rep = extractor.extract_batch([in_face])[0]
            
        #    # if self.training:        
            self.images[phash] = Face(rep, identity)

            alignedFace = np.array(alignedFace.resize((96,96)))[:,:,(2,1,0)]                    
            content = [str(x) for x in alignedFace.flatten()]
            msg = {
                "type": "NEW_IMAGE",
                "hash": phash,
                "content": content,
                "identity": identity,
                "representation": rep.tolist()
            }
            self.temp_save[phash] = msg
            self.sendMessage(json.dumps(msg))
            if identity not in identities:
                identities.append(identity)

        
                



        

    def Load_ID(self,ID_path):
        img_dir = glob(ID_path+'/*.jpg')
        for index in range(len(img_dir)):
            img_addr = img_dir[index]
            print "imd_addr : {}".format(img_addr)
            img_name = img_addr.split('/')[-1].split(".")[0]
            self.people.append(img_name.encode('ascii', 'ignore'))
            self.Add_ID(img_addr,index)
            print "index is {}".format(index)
        
        



if __name__ == '__main__':
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:9000", debug=False)
    factory.protocol = OpenFaceServerProtocol
    reactor.listenTCP(9000, factory)
    reactor.run()
