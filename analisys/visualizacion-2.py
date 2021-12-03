#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:04:48 2021

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
    model_recall = []
    for nc in numb_clusters_s:
        model_c = model_params[model_params.numb_clusters == nc]
        tp = np.sum(model_c.tp)
        fn = np.sum(model_c.fn)
        model_recall.append([(tp/(tp+fn)),nc])
    return(pd.DataFrame(model_recall, columns=['recall_integrado','num_clusters']))

def recall_integrado(representacion, model_params, v_sizes, clust_model, scalers, numb_clusters_s):
    if representacion == "c_v":
        v_size = "vocab_size"
    elif representacion == "tf_idf":
        v_size = "numb_features"
    else:
        v_size = "vector_size"
        
        
    scl_vs_recall_list = []  
    
    for scl in scalers:
        model_scl = model_params[model_params.scaler == scl]
        for vs in v_sizes:
            model_vs = model_scl[model_scl[v_size] == vs]
            df = results_depth_1(model_vs, numb_clusters_s)
            df = df.set_index('num_clusters')
            df = df.T
            df['scaler'] = scl
            df['v_sizes'] = vs
            df['representacion'] = representacion
            df['clust_model'] = clust_model
            if (clust_model == "GMM") and (scl == 'standar') :
                continue
            else:
                scl_vs_recall_list.append(df)
        
    return(pd.concat(scl_vs_recall_list) )



def concentrado_recall_integrado(res, representation_param, v_sizes, representacion, scalers, numb_clusters_s):    
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
    agg_params = agg_params.replace(False, np.nan).dropna(axis=1,how="all")
    
    rec_integrado_agg = recall_integrado(representacion, agg_params, v_sizes, clust_model, scalers, numb_clusters_s)
            
    #############################################################################
    #GMNM
    #############################################################################
    clust_model = "GMM"
    gmm_params = x[x.cluster == clust_model]
    gmm_params = gmm_params.replace(False, np.nan).dropna(axis=1,how="all")
    rec_integrado_gmm = recall_integrado(representacion, gmm_params, v_sizes, clust_model, scalers, numb_clusters_s)
            
     
    #############################################################################
    #KMN
    #############################################################################
    clust_model = "Kmeans"       
    kmn_params = x[x.cluster == clust_model]
    kmn_params = kmn_params.replace(False, np.nan).dropna(axis=1,how="all")      
    rec_integrado_kmn = recall_integrado(representacion, kmn_params, v_sizes, clust_model, scalers, numb_clusters_s)
        
    return(pd.concat([rec_integrado_agg, rec_integrado_gmm, rec_integrado_kmn]))
    
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

vocab_size_s = [200, 400,800]
representacion = "c_v"

c_v_arr = os.listdir(results + representacion)
c_v_res = []
for i in c_v_arr:
    with open(results + representacion + '/' + i , 'rb') as f: 
        c_v_res.append(pickle.load(f))

with open(results + representacion + '_parameters.txt' , 'rb') as f:
    c_v_param = pickle.load(f)
    
#concentrado_recall_integrado(res, representation_param, v_sizes, representacion)

con_rec_int_cv = concentrado_recall_integrado(c_v_res, c_v_param, vocab_size_s, representacion, scalers, numb_clusters_s)  

#############################################################################
#TF IDF
#############################################################################

numb_features_s  = [200,400,800]
representacion = "tf_idf"

tf_idf_arr = os.listdir(results + representacion)
tf_idf_res = []
for i in tf_idf_arr:
    with open(results + representacion +'/' + i , 'rb') as f: 
        tf_idf_res.append(pickle.load(f))

with open(results + representacion + '_parameters.txt' , 'rb') as f:
    tf_idf_param = pickle.load(f)

con_rec_int_tfidf = concentrado_recall_integrado(tf_idf_res, tf_idf_param, numb_features_s, representacion, scalers, numb_clusters_s)


#############################################################################
#Word to Vec
#############################################################################

vector_size_s  = [200,400,800]
representacion = "w_2_v"

w_2_v_arr = os.listdir(results + representacion)
w_2_v_res = []
for i in w_2_v_arr:
    with open(results + representacion + '/' + i , 'rb') as f: 
        w_2_v_res.append(pickle.load(f))

with open(results + representacion + '_parameters.txt' , 'rb') as f:
    w_2_v_param = pickle.load(f)


con_rec_int_w2v = concentrado_recall_integrado(w_2_v_res, w_2_v_param, vector_size_s, representacion, scalers, numb_clusters_s)



con_rec_int = pd.concat([con_rec_int_cv, con_rec_int_tfidf, con_rec_int_w2v])
con_rec_int.reset_index(inplace=True, drop=True)

###########################################################################

con_rec_int['maximos'] = con_rec_int.iloc[:, 0:9].max(1)
con_rec_int.sort_values(by=['maximos'], ascending=False, inplace=True)

    
test = con_rec_int.dropna(0).head(15)
#test = con_rec_int.head(15)
#test.sort_values(by=['v_sizes'], ascending=False, inplace=True)
#test.sort_values(by=['clust_model'], ascending=False, inplace=True)
#test.sort_values(by=['scaler'], ascending=False, inplace=True)
test.sort_values(by=['representacion'], ascending=False, inplace=True)
#top = test

#top = test[test.v_sizes == 800]
#top = test[test.v_sizes == 400]
#top = test[test.v_sizes == 200]

#top = test[test.clust_model == 'GMM']
#top = test[test.clust_model == 'Kmeans']
#top = test[test.clust_model == 'agglomerative']

#top = test[test.scaler == 'none']
#top = test[test.scaler == 'minmax']
#top = test[test.scaler == 'standar']

#top = test[test.representacion == 'c_v']
#top = test[test.representacion == 'tf_idf']
top = test[test.representacion == 'w_2_v']

labs = top['scaler'] + ' ' + \
        top['v_sizes'].astype(str) + ' ' + \
        top['representacion'] + ' ' + \
        top['clust_model']
ax = top.iloc[:,0:9].T.plot()
ax.set_title("Mejores modelos")
ax.legend(labels = labs.tolist() , bbox_to_anchor=(1.0, 1.0))
#ax.ylabel('recall integrado')
#ax.xlabel('Numero de clusers')
ax.plot()


############################################################################
toptoptop = []
for i in numb_clusters_s:
    toptoptop.append(con_rec_int.nlargest(3,i).head(1))
    
toptoptop = pd.concat(toptoptop)

#toptoptop.iloc[:,0:9].T.plot()

ax = toptoptop.iloc[:,0:9].T.plot()
ax.set_title("Mejores modelos")
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.ylabel('recall integrado')
ax.xlabel('Numero de clusers')
ax.plot()

###########################################################################
toptoptop1 = []
for i in con_rec_int.representacion.unique():
    for j in con_rec_int.clust_model.unique():
        filtro = con_rec_int[(con_rec_int.representacion == i) & (con_rec_int.clust_model == j)]
        toptoptop1.append(filtro.loc[filtro.iloc[:, 0:9].max(1).idxmax()])
            
hand = toptoptop1['scaler'] + ' ' + toptoptop1['v_sizes'].astype(str)+ ' ' + toptoptop1['representacion']+ ' ' + toptoptop1['clust_model']
toptoptop1 = pd.DataFrame(toptoptop1)    
ax = toptoptop1.iloc[:,0:9].T.plot()
ax.set_title("Mejores modelos")
ax.legend(labels = hand.tolist() , bbox_to_anchor=(1.0, 1.0))
#ax.ylabel('recall integrado')
#ax.xlabel('Numero de clusers')
ax.plot()





















