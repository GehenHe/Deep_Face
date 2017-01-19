# -*- coding: utf-8 -*-
"""
Created on Wed May 11 22:35:39 2016

@author: teddy
"""

from search_engine import Search_Engine
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score 
from sklearn.cross_validation import train_test_split

data = load_iris()
feas =  data.data
labels = data.target

fea_train, fea_test, label_train, label_test = train_test_split(
       feas, labels, test_size=0.3, random_state=42)

engine = Search_Engine(thre=1,rej='top')
engine.fit(fea_train,label_train)
pre_labels = []
dis_t = []
for fea in fea_test:
    result = engine.predict(fea)
    pre_labels.append(result[0])
    dis_t.append(result[1])


score = accuracy_score(label_test,pre_labels)


    
