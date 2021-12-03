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
scalers =  ['minmax', 'none']
casos = ["caso1", "caso2", "caso3"]

#word_2_vec
vector_sizes = [200,400]
min_counts = [1,2]
max_s_ls = [3,6]
max_iters = [100]
numb_parts = [5]

#ALL Clusters
numb_clusters_s = [20,25,30,35,40]
#g_m_m
cov_types = ['full', 'diag', 'spherical']


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

representation = representation_technic[1]
input_dictionary['representation'] = representation

cluster = cluster_technic[0]
input_dictionary['cluster'] = cluster



for vector_size in vector_sizes:
    input_dictionary['vector_size'] = vector_size        #w 2 v
    for min_count in min_counts:
        input_dictionary['min_count'] = min_count            #w 2 v
        for max_s_l in max_s_ls:
            input_dictionary['max_s_l'] = max_s_l                #w 2 v
            for max_iter in max_iters:
                input_dictionary['max_iter'] = max_iter              #w 2 v
                for numb_part in numb_parts:
                    input_dictionary['numb_part'] = numb_part            #w 2 v
                    for numb_clusters in numb_clusters_s:
                        input_dictionary['numb_clusters'] = numb_clusters
                        ##########################################################################################################
                        for cov_type in cov_types:
                            input_dictionary['cov_type'] = cov_type            
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

file = open(hommie + 'results/' + 'w_2_v_g_m_m_parameters.txt', 'wb')
pickle.dump(parameters, file)
file.close()
        
        


