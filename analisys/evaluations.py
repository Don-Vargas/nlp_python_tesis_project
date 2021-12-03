#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:16:11 2021

@author: hecvagu
"""

from sklearn.metrics import classification_report, confusion_matrix



class Evaluator():
    
    def __init__(self):
        print("preparando evaluacion de resultados")
        
    def test_case_evaluator(self, documents):
        print("Generando tabla de verdad")
        documents_cluster_match = documents[documents.codigoEstado == "custom"].prediction.item()
        documents["documents_cluster_match"] = documents.prediction == documents_cluster_match
        report = classification_report(documents.keyword_match, documents.documents_cluster_match)
        matrix = confusion_matrix(documents.keyword_match, documents.documents_cluster_match)
        return(matrix, report, documents.keyword_match, documents.documents_cluster_match)

        
        
