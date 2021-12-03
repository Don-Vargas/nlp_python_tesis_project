#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 22:10:15 2021

@author: hecvagu
"""


import pickle
import os
import pandas as pd
#from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np 
import disarray

    
#############################################################################
#DATA LOAD
#############################################################################
results = '/home/hecvagu/AnacondaProjects/MasterTesis/approach3/test_results/'
leyes = pd.read_csv("/home/hecvagu/AnacondaProjects/MasterTesis/todasLasLeyes.csv")
    
#############################################################################
#Count Vectorizer
#############################################################################

c_v_arr = os.listdir(results)
res = []
for i in c_v_arr:
    with open(results + i , 'rb') as f: 
        res.append(pickle.load(f))

with open(results + 'test_parameters.txt' , 'rb') as f:
    param = pickle.load(f)
    
    
#############################################################################
#METRICAS
#############################################################################
#cm confusion matrix
cm = res[1][1]
#clust
clust = res[1][0]
leyes['clust'] = clust[:-1]
#clust
kw_match = res[1][3]
leyes['kw_match'] = kw_match[:-1]
#clust
doc_clus_match = res[1][4]
leyes['doc_clus_match'] = doc_clus_match[:-1]
metricas = pd.DataFrame(cm).da.export_metrics()

con_matrix = pd.DataFrame(cm.ravel(), index=["tn", "fp", "fn", "tp"]).T

#############################################################################
#AGG
#############################################################################

articulo_similar = leyes[leyes.clust == clust.iloc[-1]]
articulo_similar.columns
articulo_similar = articulo_similar[['_c0', 
                                     'codigoEstado', 
                                     'articuloNumero', 
                                     'articuloOriginal',
                                     'clust', 
                                     'kw_match', 
                                     'doc_clus_match']]


kw_true = articulo_similar[articulo_similar.kw_match == True]
len(kw_true)
kw_false = articulo_similar[articulo_similar.kw_match == False]
len(kw_false)



