#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:09:58 2021

@author: hecvagu
"""
from pyspark.sql.functions import split, lower, col, concat, array, lit, udf
from pyspark.sql import types as T
from pyspark.ml.feature import NGram

class Preprocessing():
    
    def __init__(self):
        print("preparando el pre procesamiento de los datos")
        
    
    def data_case_mix(self, test_case, documents, sc, sqlContext):
        case = ' '.join(test_case)
        vals = sc.parallelize([['9999','custom','1',case,case,case]])
        Nrow = sqlContext.createDataFrame(vals, documents.schema)
        print('Caso de prueba cargado')
        documents = documents.union(Nrow)
        return(documents)
    
    def clean_data(self, documents):
        documents = documents.withColumn("text_splitted", split(lower(col("articuloLimpio")), " "))
        print("Datos limpios")
        return(documents)
    
    def ngram_builder(self, documents):
        documents = NGram(n=2, inputCol="text_splitted", outputCol="ngrams2").transform(documents)
        documents = NGram(n=3, inputCol="text_splitted", outputCol="ngrams3").transform(documents)
        documents = documents.withColumn("text_splitted", concat(col("text_splitted"),col("ngrams2"),col("ngrams3")))
        print('Ngramas calculados')
        return(documents)
    
    def test_case_flag(self, documents, test_case):
        
        def containsAny(string, array):
            if len(string) == 0:
                return False
            else:
                return (any(word in string for word in array))
        
        contains_udf = udf(containsAny, T.BooleanType())
        documents =  documents.withColumn("keyword_match", contains_udf(col("text_splitted"), array([lit(i) for i in test_case])))
        return(documents)
    
    
    def data_preprocess(self, test_case, documents, sc, sqlContext):
        documents = self.data_case_mix(test_case, documents, sc, sqlContext)
        documents = documents.withColumn("text_splitted", split(lower(col("articuloLimpio")), " "))
        documents = self.clean_data(documents)
        documents = self.ngram_builder(documents)
        documents = self.test_case_flag(documents, test_case)
        return(documents)
    
    