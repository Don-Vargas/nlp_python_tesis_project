from analisys.loads import Loader
from analisys.preprocessings import Preprocessing
from analisys.representations import Representation
from analisys.clusters import Cluster
from analisys.evaluations import Evaluator

class Model():
    
    def __init__(self):
        print("Escogiendo el modelo")
        
    def representation_selector(self, documents, represent, **variables):
        
        if variables["representation"] == "CountVectorizer":
            representation_results = represent.count_vectorizer(documents, **variables)
        elif variables["representation"] == "Word2Vec":
            representation_results = represent.word_2_vec(documents, **variables)
        elif variables["representation"] == "TFIDF":
            representation_results = represent.tf_idf(documents, **variables)
        else:
            representation_results = 0
            print("La representación " + variables["representation"] + " no está disponible")
            
        return(representation_results)
    
    
    def cluster_selector(self, clust, sqlContext, representation_results,  **variables):
        
        if variables["cluster"] == "GMM":
            clu = clust.g_m_m(representation_results, sqlContext, **variables)
        elif variables["cluster"] == "Kmeans":
            clu = clust.k_means(representation_results, sqlContext, **variables)
        elif variables["cluster"] == "agglomerative":
            clu = clust.agglomerative(representation_results, sqlContext, **variables)
        else:
            clu = 0
            print("El cluster " + variables["cluster"] + " no está disponible")
            
        return(clu)
        
            
    
    def switcher(self, documentsPath, **variables):
        
        data_load = Loader()
        preprocesses = Preprocessing()
        represent = Representation()
        clust = Cluster()
        evaluate = Evaluator() 
        
        sc,sqlContext = data_load.context_loader()
        print("spark version: ", sc.version)
        
        documents = data_load.leyes_loader(sc,sqlContext, documentsPath)
        
        test_case = data_load.test_cases_loader(variables["caso"])
        documents = preprocesses.data_preprocess( test_case, documents, sc, sqlContext)
        
        representation_results = self.representation_selector(documents, represent, **variables)
        
        clu = self.cluster_selector(clust, sqlContext, representation_results, **variables)
        
        
        
        if (representation_results != 0) or (clu != 0):
            matrix, report, kw_match, clst_match = evaluate.test_case_evaluator(clu)
        else:
            pass
    
        return(clu.prediction, matrix, report, kw_match, clst_match)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
