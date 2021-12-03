from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import HashingTF, IDF

class Representation():
    
    def __init__(self):
        print("preparando la representación")
    
        
 ##############################################################################
    def count_vectorizer(self, documents, **variables):
        print("representando la count_vectorizer")
        cv = CountVectorizer(inputCol = "text_splitted", 
                             outputCol = "features", 
                             vocabSize = variables["vocab_size"])
        model = cv.fit(documents)
        result = model.transform(documents)
        return(result)
 ##############################################################################

    def word_2_vec(self, documents, **variables):
        print("representando la  word_2_vec")
        word2Vec = Word2Vec(vectorSize = variables["vector_size"], 
                            minCount = variables["min_count"],
                            maxSentenceLength = variables["max_s_l"],
                            maxIter = variables["max_iter"], 
                            numPartitions = variables["numb_part"],
                            inputCol="text_splitted", 
                            outputCol="features")
        
        model = word2Vec.fit(documents)
        result = model.transform(documents)
        return(result)
 ##############################################################################

    def tf_idf(self, documents, **variables):
        print("representando la  tf_idf")
        hashingTF = HashingTF(inputCol = "text_splitted", 
                              outputCol = "TF", 
                              numFeatures = variables["numb_features"])
        
        featurizedData = hashingTF.transform(documents)
        # alternatively, CountVectorizer can also be used to get
        # term frequency vectors
        idf = IDF(inputCol="TF", 
                  outputCol="features")#, outputCol="TF-IDF")
        
        idfModel = idf.fit(featurizedData)
        result = idfModel.transform(featurizedData)
        return(result)
