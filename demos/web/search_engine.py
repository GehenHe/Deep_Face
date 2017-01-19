# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:33:30 2016

@author: teddy
"""
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 

class Search_Engine(object):
    def __init__(self, n_neighbors=1, thre=0.3, rej= 'top' ,weights='uniform', 
                 algorithm='ball_tree', metric='euclidean'):
                     
        self.engin = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,
                                          algorithm=algorithm, metric=metric)
        self.thre = thre
        if rej not in {'top','mean'}:
            raise ValueError('No such rej mean')
        self.rej = rej
        self.n_neighbors = n_neighbors

    def fit(self,feas,labels):
        if feas.ndim == 2:
            fea_norm = np.linalg.norm(feas, axis=1)[:, np.newaxis]
        else:
            raise Exception('Wrong dimension')
        feas = feas/fea_norm
        self.engin.fit(feas,labels)
        
    def predict(self, query):
        if query.ndim == 1:
            query = query/np.linalg.norm(query)
        else:
            raise Exception('Wrong dimension')
        if self.rej == 'top':
            dis = self.engin.kneighbors(query,1)[0][0][0] 
        elif self.rej == 'mean':

            dis = np.mean(self.engin.kneighbors(query,self.n_neighbors/2+1)[0][0])
        if dis > self.thre:
            label = -1
        else:
            label = self.engin.predict(query)[0]
        return label,dis