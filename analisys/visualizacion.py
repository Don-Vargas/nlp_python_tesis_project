#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:33:09 2021

@author: hecvagu
"""

import pickle
import os
import pandas as pd
#from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np 
import disarray


def results_depth_1(model_params, numb_clusters_s):
    model_recall_list = []
    for nc in numb_clusters_s:
        model_c = model_params[model_params.numb_clusters == nc]
        tp = model_c.tp.sum()
        fn = model_c.fn.sum()
        model_recall_list.append(tp/(tp+fn))
    return(model_recall_list)


def model_plot(params_list, numb_clusters_s, titulus):
    for recall_list in params_list:
        if recall_list[1] == 0:
            labe = 'vocab size ' + str(recall_list[2])
            plt.plot(numb_clusters_s, recall_list[0], label=labe)
        elif recall_list[1] == 'all':
            labe = recall_list[2]
            plt.plot(numb_clusters_s, recall_list[0], label=labe)            
        else:
            labe = 'Scaler ' + recall_list[1] + '; vocab size ' + str(recall_list[2])
            plt.plot(numb_clusters_s, recall_list[0], label=labe)        
    plt.title(titulus)
    plt.ylabel('recall')
    plt.xlabel('Numero de clusers')
    plt.ylim([0, 0.62])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def tri_plot(model_params, numb_clusters_s, v_sizes, scalers, representacion, clust_model):
    if representacion == "c_v":
        v_size = "vocab_size"
    elif representacion == "tf_idf":
        v_size = "numb_features"
    else:
        v_size = "vector_size"
    
    ##########################################################################
    #model recall integrado 1 grafica
    ##########################################################################
    
    titulus = representacion + " " + clust_model + " integrado todo"
    model_recall_list = results_depth_1(model_params, numb_clusters_s)
    model_plot([(model_recall_list, 'all', 'all')], numb_clusters_s, titulus)
    
    ##########################################################################
    #model recall integrado por vocab size 3 graficas
    ##########################################################################
    vs_list = []
    for vs in v_sizes:
        model_vs = model_params[model_params[v_size] == vs]
        recall_list = results_depth_1(model_vs, numb_clusters_s)
        vs_list.append((recall_list, 0, vs))   
        
    titulus = representacion + " " + clust_model + " by vocab size"
    model_plot(vs_list, numb_clusters_s, titulus)
    
    ##########################################################################
    #model recall compuesto por vocab size y scaler 9 graficas
    ##########################################################################
    scl_vs_recall_list = []
    for scl in scalers:
        model_scl = model_params[model_params.scaler == scl]
        for vs in v_sizes:
            model_vs = model_scl[model_scl[v_size] == vs]
            recall_list = results_depth_1(model_vs, numb_clusters_s)  
            if (clust_model == "GMM") and (scl == 'standar') :
                continue
            else:          
                scl_vs_recall_list.append((recall_list, scl, vs))
                
    titulus = representacion + " " + clust_model + " Por scaler y vocab size"
    model_plot(scl_vs_recall_list, numb_clusters_s, titulus)
    
    return(model_recall_list)


def model_integrado_plots(representation_param, res, numb_clusters_s, v_sizes, scalers, representacion):
    #############################################################################
    #METRICAS
    #############################################################################
    #cm_s confusion matrixes
    cm_s = [i[1] for i in res]
    metricas = [pd.DataFrame(cm).da.export_metrics() for cm in cm_s]
    
    x = representation_param.copy()
    x["metricas"] = metricas
    x["confusion_matrix"] = cm_s
    
    con_matrix = pd.DataFrame([i[1].ravel() for i in res], columns=["tn", "fp", "fn", "tp"])
    
    x = pd.concat([x,con_matrix], axis=1)
    
    
    #############################################################################
    #AGG
    #############################################################################
    clust_model = "agglomerative"
    agg_params = x[x.cluster == clust_model]
    #agg_params = agg_params[['caso', 'scaler', 'resultados',
    #       'vocab_size',  'min_count', 'max_s_l', 'max_iter',
    #       'numb_part', 'numb_features', 'numb_clusters', 'aff', 'link', 
    #       'metricas', 'confusion_matrix', "tn", "fp", "fn", "tp"]]
     
    agg_params = agg_params.replace(False, np.nan).dropna(axis=1,how="all")
    
    agg_recall_list = tri_plot(agg_params, numb_clusters_s, v_sizes, scalers, representacion, clust_model)
    
    #############################################################################
    #GMNM
    #############################################################################
    clust_model = "GMM"
    gmm_params = x[x.cluster == clust_model]
    #gmm_params = gmm_params[['caso', 'scaler', 'resultados',
    #       'vocab_size', 'vector_size', 'min_count', 'max_s_l', 'max_iter',
    #       'numb_part', 'numb_features', 'numb_clusters', 'cov_type',
    #       'metricas', 'confusion_matrix', "tn", "fp", "fn", "tp"]]
     
    gmm_params = gmm_params.replace(False, np.nan).dropna(axis=1,how="all")
    
    gmm_recall_list = tri_plot(gmm_params, numb_clusters_s, v_sizes, scalers, representacion, clust_model)
    
    #############################################################################
    #KMN
    #############################################################################
    clust_model = "Kmeans"
    kmn_params = x[x.cluster == clust_model]
    #kmn_params = kmn_params[['caso', 'scaler', 'resultados',
    #       'vocab_size', 'vector_size', 'min_count', 'max_s_l', 'max_iter',
    #       'numb_part', 'numb_features', 'numb_clusters', 'numIterations', 
    #      'metricas', 'confusion_matrix', "tn", "fp", "fn", "tp"]]
     
    kmn_params = kmn_params.replace(False, np.nan).dropna(axis=1,how="all")
    
    knn_recall_list = tri_plot(kmn_params, numb_clusters_s, v_sizes, scalers, representacion, clust_model)
    
    
    #############################################################################
    #AGG GMM KMN
    #############################################################################
    
    general_list = [(agg_recall_list, 'all', 'agg'),
                    (gmm_recall_list, 'all', 'gmm'),
                    (knn_recall_list, 'all', 'kmn')]
    titulus = "General " + representacion + " AGG GMM KMN "
    model_plot(general_list, numb_clusters_s, titulus)
    return(agg_recall_list, gmm_recall_list, knn_recall_list, x)


    
#############################################################################
#DATA LOAD
#############################################################################
numb_clusters_s = [20,22,24,26,28,30,32,34,36]
scalers =  ['minmax', 'standar', 'none']
results = '/home/hecvagu/AnacondaProjects/MasterTesis/approach3/results/'
#leyes = pd.read_csv("/home/hecvagu/AnacondaProjects/MasterTesis/todasLasLeyes.csv")
    
#############################################################################
#Count Vectorizer
#############################################################################

vocab_sizes = [200, 400,800]
c_v_arr = os.listdir(results + 'c_v')
c_v_res = []
for i in c_v_arr:
    with open(results + 'c_v/' + i , 'rb') as f: 
        c_v_res.append(pickle.load(f))

with open(results + 'c_v_parameters.txt' , 'rb') as f:
    c_v_param = pickle.load(f)

c_v_agg, c_v_gmm, c_v_knn, x1 = model_integrado_plots(c_v_param, 
                                                  c_v_res, 
                                                  numb_clusters_s, 
                                                  vocab_sizes, 
                                                  scalers, 
                                                  "c_v")

    
#############################################################################
#TF IDF
#############################################################################

numb_features_s  = [200,400,800]
tf_idf_arr = os.listdir(results + 'tf_idf')
tf_idf_res = []
for i in tf_idf_arr:
    with open(results + 'tf_idf/' + i , 'rb') as f: 
        tf_idf_res.append(pickle.load(f))

with open(results + 'tf_idf_parameters.txt' , 'rb') as f:
    tf_idf_param = pickle.load(f)

tf_idf_agg, tf_idf_gmm, tf_idf_knn, x2 = model_integrado_plots(tf_idf_param, 
                                                  tf_idf_res, 
                                                  numb_clusters_s, 
                                                  numb_features_s, 
                                                  scalers, 
                                                  "tf_idf")


#############################################################################
#Word to Vec
#############################################################################

vector_size_s  = [200,400,800]
w_2_v_arr = os.listdir(results + 'w_2_v')
w_2_v_res = []
for i in w_2_v_arr:
    with open(results + 'w_2_v/' + i , 'rb') as f: 
        w_2_v_res.append(pickle.load(f))

with open(results + 'w_2_v_parameters.txt' , 'rb') as f:
    w_2_v_param = pickle.load(f)

w_2_v_agg, w_2_v_gmm, w_2_v_knn, x3 = model_integrado_plots(w_2_v_param, 
                                                           w_2_v_res, 
                                                           numb_clusters_s, 
                                                           vector_size_s, 
                                                           scalers, 
                                                           "w_2_v")


general_list = [(c_v_agg, 'all', 'c_v_agg'),
                (c_v_gmm, 'all', 'c_v_gmm'),
                (c_v_knn , 'all', 'c_v_kmn'),
                (tf_idf_agg, 'all', 'tf_idf_agg'),
                (tf_idf_gmm, 'all', 'tf_idf_gmm'),
                (tf_idf_knn , 'all', 'tf_idf_kmn'),
                (w_2_v_agg, 'all', 'w_2_v_agg'),
                (w_2_v_gmm, 'all', 'w_2_v_gmm'),
                (w_2_v_knn , 'all', 'w_2_v_kmn')]
titulus = "General All models "
model_plot(general_list, numb_clusters_s, titulus)









