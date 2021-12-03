#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 21:08:29 2021

@author: hecvagu
"""


import pandas as pd
import pickle
import os

representation_technic = ["CountVectorizer"]
cluster_technic = ["agglomerative"]
scalers =  ['none']
casos = ["caso_test"]

#count_vectorizer
vocab_sizes = [400]

#ALL Clusters
numb_clusters_s = [26]
#agglomerative
affs = ['euclidean']
links = ['ward']

input_dictionary = {'representation': False,
                    'cluster': False,
                    'caso': False,
                    'scaler': False,
                    'resultados': False,
                    'vocab_size': False,
                    'vector_size': False,
                    'min_count': False,
                    'max_s_l': False,
                    'max_iter': False,
                    'numb_part': False,
                    'numb_features': False,
                    'numb_clusters': False,
                    'cov_type': False,
                    'numIterations': False,
                    'aff': False,
                    'link': False}

resultados = 0
parameters = []
hommie = '/home/hecvagu/AnacondaProjects/MasterTesis/approach3/'

representation = representation_technic[0]
input_dictionary['representation'] = representation

for cluster in cluster_technic:
    input_dictionary['cluster'] = cluster
    for vocab_size in vocab_sizes:
        input_dictionary['vocab_size'] = vocab_size      #c v
        for numb_clusters in numb_clusters_s:
            input_dictionary['numb_clusters'] = numb_clusters
############
            if cluster == "agglomerative":
                for link in links:
                    input_dictionary['link'] = link
                    for aff in affs:
                        if link == "ward" and aff != "euclidean":
                            continue
                        else:
                            input_dictionary['aff'] = aff                
                        for caso in casos:
                            input_dictionary['caso'] = caso
                            for scaler in scalers:
                                input_dictionary['scaler'] = scaler
                                input_dictionary['resultados'] = resultados    
                                parameters.append(input_dictionary.copy())                            
                                
                                file = open('/home/hecvagu/AnacondaProjects/MasterTesis/approach3/variables.txt', 'wb')
                                pickle.dump(input_dictionary, file)
                                file.close()
                                os.system('python /home/hecvagu/AnacondaProjects/MasterTesis/approach3/mainapp.py')
                                resultados = 1
                     
                    
    
parameters = pd.DataFrame(parameters)

file = open(hommie + 'test_results/' + 'test_parameters.txt', 'wb')
pickle.dump(parameters, file)
file.close()
        

