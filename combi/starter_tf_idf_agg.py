#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 20:35:01 2021

@author: hecvagu
"""

import pandas as pd
import pickle
import os

representation_technic = ["CountVectorizer","Word2Vec","TFIDF"]
cluster_technic = ["GMM","Kmeans","agglomerative"]
scalers =  ['minmax', 'standar', 'none']
casos = ["caso1", "caso2", "caso3"]

#tf_idf
numb_features_s = [200,400]

#ALL Clusters
numb_clusters_s = [20,25,30,35,40]
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

representation = representation_technic[2]
input_dictionary['representation'] = representation

cluster = cluster_technic[2]
input_dictionary['cluster'] = cluster

for numb_features in numb_features_s:
    input_dictionary['numb_features'] = numb_features    #tf idf
    for numb_clusters in numb_clusters_s:
        input_dictionary['numb_clusters'] = numb_clusters
        ##########################################################################################################
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
                        resultados = resultados + 1 

                    
parameters = pd.DataFrame(parameters)

file = open(hommie + 'results/' + 'tf_idf_agg_parameters.txt', 'wb')
pickle.dump(parameters, file)
file.close()
        
        

