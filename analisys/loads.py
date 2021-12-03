#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:11:18 2021

@author: hecvagu
"""
from pyspark.sql import SQLContext
from pyspark import  SparkContext, SparkConf


class Loader():
    
    def __init__(self):
        print("preparando La carga de datos")
        
    def context_loader(self):
        print("Cargando el spark context")
        
        conf = SparkConf().setAppName("Tesis")
        conf = (conf.setMaster('local[*]')
                .set('spark.executor.memory', '4G')
                .set('spark.driver.memory', '45G')
                .set('spark.driver.maxResultSize', '10G'))
        sc = SparkContext.getOrCreate(conf=conf)
        
        """
        https://datascience.stackexchange.com/questions/8549/how-do-i-set-get-heap-size-for-spark-via-python-notebook
        """
        #sc=SparkContext.getOrCreate()
        sqlContext = SQLContext(sc)
        
        return(sc,sqlContext)
    
    def leyes_loader(self, sc,sqlContext, documentsPath):
        print("Cargando las leyes")
        #documentsPath = "/home/hecvagu/AnacondaProjects/MasterTesis/todasLasLeyes.csv"
        documents = sqlContext.read.format("csv").option("header", "true").load(documentsPath)
        return(documents)

    
    def test_cases_loader(self, test_case):
        print("Cargando los casos de prueba")
        def caso1():
            l = ['grupos originarios',
                'grupo étnico',
                'grupos étnicos',
                'indígenas',
                'pueblos indigenas',
                'pueblos tribales',
                'pueblos indios',
                'Poblaciones Indígenas y Tribales',
                'comunidades',
                'comunidades indígena',
                'comunidades',
                'pueblos originarios']
            return(l)

        def caso2():
            l = ['Libre determinación',
                'autonomía',
                'autoregulación',
                'propios sistemas normativos',
                'leyes',
                'prácticas tradicionales',
                'prácticas comunitarias']

            #'constitución de Oaxaca para buscarmás sinonimos)'
            return(l)

        def caso3():
            l = ['Acta Constitucional',
                 'Carta constitucional',
                 'Carta federal',
                 'Carta magna',
                 'Código fundamental',
                 'Código político',
                 'Código supremo',
                 'Ley fundamental',
                 'Ley suprema de toda la Unión',
                 'Norma fundamental',
                 'Norma primaria',
                 'Norma suprema',
                 'Texto fundamenta']
            return(l)

        def caso4():
            l = ['Derechos de la persona humana',
                 'Derechos del hombre',
                 'Derechos esenciales del hombre',
                 'Derechos implícitos',
                 'Derechos individuales',
                 'Derechos innatos',
                 'Derechos morales',
                 'Derechos naturales',
                 'Derechos originales',
                 'Derechos universales',
                 'Derechos fundamentales',
                 'Derechos subjetivos',
                 'Principio de dignidad humana',
                 'Principio pro persona',
                 'Tratados internacionales']
            return(l)
        
        def caso_test():
            l = ['sanciones',
                 'penas',
                 'actos u omisiones',
                 'faltas administrativas',
                 'Combate a la Corrupción']
            return(l)

        def test_case_switcher(test_case):
            switcher = {
            'caso1': caso1(),
            'caso2': caso2(),
            'caso3': caso3(),
            'caso4': caso4(),
            'caso_test': caso_test()
            }
            return (switcher.get(test_case,"Oops! Invalid Option"))
        
        return(test_case_switcher(test_case))
        
        
        
        # agregar caso de prueba
        # https://www.sitios.scjn.gob.mx/centrodedocumentacion/sites/default/files/tesauro_juridico_scjn/pdfs/00.%20Tesauro%20Juridico%20de%20la%20SCJN.pdf
        #derechos humanos
        #derechos constitucional
        
        
        
        
        
        
        
        
        
        
        
        
        