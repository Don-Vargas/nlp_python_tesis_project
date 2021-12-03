from pyspark.ml.clustering import KMeans

from sklearn.mixture import GaussianMixture
#from pyspark.ml.clustering import PowerIterationClustering
from sklearn.cluster import AgglomerativeClustering
from pyspark.sql.functions import udf, col
from pyspark.sql.types import  DoubleType, IntegerType, ArrayType


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import Normalizer


class Cluster():
    
    def __init__(self):
        print("preparando el cluster")
        
    def spark_df_2_pandas(self, representation_results):
        if representation_results != 0 :
                            
            indices_udf = udf(lambda vector: vector.indices.tolist(), ArrayType(IntegerType()))
            values_udf = udf(lambda vector: vector.toArray().tolist(), ArrayType(DoubleType()))
            
            """
            # https://runawayhorse001.github.io/LearningApacheSpark/manipulation.html
            """
            predictionsPanDF = representation_results\
                .withColumn('indices', indices_udf(col('features')))\
                .withColumn('values', values_udf(col('features'))).toPandas()
                
            X = predictionsPanDF["values"]#.select('values')
            X = [i for i in X]
            
            
        else:
            predictionsPanDF = 0
            X = 0
            
        return(predictionsPanDF, X)
    
    

    def scaler_loader(self, documents, scaler):
        print("Cargando el método de normalización/escalación de los datos")
        
        def min_max_s(documents):
            scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
            scalerModel = scaler.fit(documents)
            scaledData = scalerModel.transform(documents)
            return (scaledData)
        
        def standar_s(documents):
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
            scalerModel = scaler.fit(documents)
            scaledData = scalerModel.transform(documents)
            return(scaledData)
        
        def taxicab_norm_s(documents):
            #For p = 1, we get the taxicab norm,[6] for p = 2, we get the Euclidean norm, and as p approaches ∞ the p-norm approaches the infinity norm or maximum norm: 
            normalizer = Normalizer(inputCol="features", outputCol="scaled_features", p=1.0)
            scaledData = normalizer.transform(documents)
            return(scaledData)
            
        def euclid_norm_s(documents):
            #For p = 1, we get the taxicab norm,[6] for p = 2, we get the Euclidean norm, and as p approaches ∞ the p-norm approaches the infinity norm or maximum norm: 
            normalizer = Normalizer(inputCol="features", outputCol="scaled_features", p=2.0)
            scaledData = normalizer.transform(documents)
            return(scaledData)
        
        def none_s(documents):
            documents = documents.withColumn("scaled_features", documents["features"])
            return(documents)
        
        def scaler_switcher(scaler, documents):
            switcher = {
            'minmax' : min_max_s(documents),
            'standar': standar_s(documents),
            'taxicab': taxicab_norm_s(documents),
            'euclid' : euclid_norm_s(documents),
            'none'   : none_s(documents)
            }
            return (switcher.get(scaler,"Oops! Invalid Option"))
        
        return(scaler_switcher(scaler, documents))
    
        
 ##############################################################################   
    def g_m_m(self, representation_results, sqlContext, **variables):
        if representation_results != 0 :
            if variables["representation"] != "Word2Vec":
                predictionsPanDF, X = self.spark_df_2_pandas(representation_results)
                
                
                X = sqlContext.createDataFrame([(i, Vectors.dense(j),) for i,j in enumerate(X)], ["id", "features"])
            else:
                X = representation_results
                predictionsPanDF = representation_results.toPandas()
            X = self.scaler_loader(X, variables["scaler"]).toPandas()
            X = [i.toArray() for i in X.scaled_features]
                
                                
            clustering = GaussianMixture(n_components = variables["numb_clusters"],
                                         covariance_type = variables["cov_type"]).fit(X)
            predictionsPanDF["prediction"] = clustering.predict(X)
            
            if variables["representation"] != "Word2Vec":
                predictionsPanDF.drop(["indices","values"],axis=1, inplace=True)
            
        else:
            predictionsPanDF = 0
        return(predictionsPanDF)
        
    
    
    
###############################################################################
    def k_means(self, representation_results, sqlContext, **variables):
        if representation_results != 0 :
            
            representation_results = self.scaler_loader(representation_results, variables["scaler"])
            
            kmeans = KMeans(featuresCol='scaled_features').setMaxIter(variables["numIterations"]).setK(variables["numb_clusters"])
            kmeans_model = kmeans.fit(representation_results)
            # Make predictions
            predictions = kmeans_model.transform(representation_results)
            
            predictionsPanDF = predictions.toPandas()
            
            
        else:
            predictionsPanDF = 0
        return(predictionsPanDF)

 #############################################################################   
    def agglomerative(self, representation_results, sqlContext, **variables):
        if representation_results != 0 :
            if variables["representation"] != "Word2Vec":
                predictionsPanDF, X = self.spark_df_2_pandas(representation_results)
                
                
                X = sqlContext.createDataFrame([(i, Vectors.dense(j),) for i,j in enumerate(X)], ["id", "features"])
            else:
                X = representation_results
                predictionsPanDF = representation_results.toPandas()
            X = self.scaler_loader(X, variables["scaler"]).toPandas()
            X = [i.toArray() for i in X.scaled_features]
            
            clustering = AgglomerativeClustering(n_clusters = variables["numb_clusters"],
                                                 affinity = variables["aff"],
                                                 linkage = variables["link"]).fit(X)
            predictionsPanDF["prediction"] = clustering.labels_
            
            if variables["representation"] != "Word2Vec":
                predictionsPanDF.drop(["indices","values"],axis=1, inplace=True)
            
        else:
            predictionsPanDF = 0
        return(predictionsPanDF)
    
