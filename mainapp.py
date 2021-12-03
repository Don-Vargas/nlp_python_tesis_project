#! home/hecvagu/anaconda3/bin/python
from analisys.models import Model
import pickle
    
    
def main():
    
    documentsPath = "/home/hecvagu/AnacondaProjects/MasterTesis/todasLasLeyes.csv"
    
    
    
    with open('/home/hecvagu/AnacondaProjects/MasterTesis/approach3/variables.txt', 'rb') as f:
        variables = pickle.load(f)
       
    
    print(variables["representation"], variables["cluster"], variables["scaler"], variables["cov_type"])
    modelo = Model()
    r = modelo.switcher(documentsPath, **variables)
    
    #file = open('/home/hecvagu/AnacondaProjects/MasterTesis/approach3/results/w_2_v/res_' + str(variables["resultados"]) + '.txt', 'wb')
    file = open('/home/hecvagu/AnacondaProjects/MasterTesis/approach3/test_results/test_res_' + str(variables["resultados"]) + '.txt', 'wb')
    pickle.dump(r, file)
    file.close()
    
    print("##################################")

if __name__ == "__main__":
    main()



"""
increse heap  memory
https://stackoverflow.com/questions/32336915/pyspark-java-lang-outofmemoryerror-java-heap-space
"""
