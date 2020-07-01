# -*- coding: utf-8 -*-
import cicflow

###apelare CICFLOW
inFile = "/home/matei/WORK/pcaps/captura_8Iunie_pcap.pcap"
outFile = "/home/matei/WORK/pcaps/"
#cicflow.Apel_CICFLOW(inFile, outFile)


import pandas as pd
import seaborn as sn
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import curatare_date
import echilibrare_date
import selectie_features
import algoritmi
import cicflow
    
    
def main(url):
    
    #url = "/home/matei/WORK/csv_cu_label/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    dataset = read_csv(url, encoding='iso-8859-1', skip_blank_lines=True, keep_default_na=False, 
                       dtype={"Flow ID": str, "Src IP": str, "Dst IP": str, "Timestamp": str, "Label": str})
    
    ################################prezentare date
    
    print("Dimensiune set de date: ")
    print(dataset.shape)   #cate randuri si coloane
    
    # vad tipul coloanelor
    for i in dataset.columns:   
        print("'" + i + "' - " + str(dataset[i].dtype))
    
    # class distribution
    print("\n------------------Distributie date dupa ' Label':")
    print(dataset.groupby(' Label').size())  #cate sunt din fiecare label final
    
    
    ##################CURATARE DATE 
    dataset_clean_poz = curatare_date.curatare(dataset)
     
    #################Conversie Label in clasificator multiclass numeric
    dataset_clean_poz = dataset_clean_poz.replace({' Label':{'BENIGN': 1, 'Web Attack - Brute Force': 2, 
                                                             'Web Attack - Sql Injection': 3, 'Web Attack - XSS': 4}})
    print("\nIlustrare clasificare multiclasa:\n")
    print(dataset_clean_poz.groupby(' Label').size())
    
    ################Conversie Label in clasificator binar
    dataset_binar = dataset_clean_poz.copy()
    #dataset_binar = dataset_binar.replace({' Label':{'BENIGN': 0, 'Web Attack - Brute Force': 1, 'Web Attack - Sql Injection': 1, 'Web Attack - XSS': 1}})
    dataset_binar = dataset_binar.replace({' Label':{1: 0, 2: 1, 3: 1, 4: 1}})
    print("\nIlustrare clasificare binara:\n")
    print(dataset_binar.groupby(' Label').size()) 
    
    #################Separare features de label
    X_all = dataset_clean_poz.loc[:, ~dataset_clean_poz.columns.isin(['index','Flow ID', ' Source IP',
                                                                      ' Destination IP', ' Timestamp',' Label'])]  #independent columns
    y_all = dataset_clean_poz.iloc[:,-1]    
    
    X_all_binar = dataset_binar.loc[:, ~dataset_binar.columns.isin(['index','Flow ID', ' Source IP', 
                                                                    ' Destination IP', ' Timestamp',' Label'])]  #independent columns
    y_all_binar = dataset_binar.iloc[:,-1]
    
    
    ################Normalizare || Standardizare
    
    #Normalizare
    normalized_X = preprocessing.normalize(X_all)
    df_normalized_X = pd.DataFrame(normalized_X, columns=X_all.columns)
    
    #Standardizare
    # standardized_X = preprocessing.scale(X_all)
    # df_standardized_X = pd.DataFrame(standardized_X)
    
    ####################BALANCING DATA
    
    #multi class
    #fac un dataframe nou cu toate feature-urile normalizate + labels
    df_y_all = pd.DataFrame(y_all);
    df_norX_Y = pd.concat([df_normalized_X.reset_index(drop=True), df_y_all.reset_index(drop=True)], axis=1, verify_integrity=True)  #am pus asa cu reset ca e prost Pandas si adauga randuri altfel
    #df_norX_Y = df_normalized_X.join(y_all) #strica labels
        
    oversampled = echilibrare_date.echilibrare(df_norX_Y, df_normalized_X, y_all)
    
    #binar
    df_y_all_binar = pd.DataFrame(y_all_binar);
    df_norX_Y_binar = pd.concat([df_normalized_X.reset_index(drop=True), df_y_all_binar.reset_index(drop=True)], axis=1, verify_integrity=True)  #am pus asa cu reset ca e prost Pandas si adauga randuri altfel
    
    oversampled_binar = echilibrare_date.echilibrare_binara(df_norX_Y_binar, df_normalized_X, y_all_binar)
    
    
    #########################SELECTARE FEATURES
    Xo = oversampled.iloc[:,0:80]
    Yo = oversampled.iloc[:,-1]
    
    Xo_binar = oversampled_binar.iloc[:,0:80]
    Yo_binar = oversampled_binar.iloc[:,-1]
    
    ####Universal Selection
    #selectie_features.universal_selection(Xo, Yo, X_all)
    #selectie_features.universal_selection_binar(Xo_binar, Yo_binar, X_all)
    
    ####Feature Importance
    #selectie_features.feature_importance(Xo, Yo, X_all)
    #selectie_features.feature_importances_binar(Xo_binar, Yo_binar, X_all)
    
    ####Correlation Matrix
    #selectie_features.cor_mat(df_norX_Y)
    #selectie_features.cor_mat_binar(df_norX_Y_binar)
    
    ####Rand Forest
    #selectie_features.random_for(Xo, Yo)
    #selectie_features.rand_for_binar(Xo_binar, Yo_binar, Xo)
    
    
    ################################ALGORITMI
    optiuneClass = input("\nAlegeti tipul de clasificare: \n1)Clasificare Multiclasa \n2)Clasificare Binara\n\n")
    
    
    if(optiuneClass == "1"):   
        #Multiclass
        #X_f1 = Xo.iloc[:,[0,1,16,17,23,24,25]]  #univeariate selection
        X_f1 = Xo.iloc[:, [24, 0, 20, 25, 19, 1, 23]]
        X_f1_train,X_f1_test,y_f1_train,y_f1_test = train_test_split(X_f1, Yo, test_size=0.30, random_state=20)
    
        optiune = input("\nAlegeti algoritmul dorit: \n1) Toate \n2) KNN \n3) Decision Tree \n4) Random Forest \n5) SVM \n\n")
        
        if(optiune == "1"):
            ###KNN
            modelKNN = algoritmi.knn(X_f1_train, y_f1_train, X_f1_test, y_f1_test)        
            
            ###Decision Tree
            modelTree = algoritmi.decision_tree(X_f1_train, y_f1_train, X_f1_test, y_f1_test)
    
            ###Random Forest
            modelRandF = algoritmi.random_f(X_f1_train, y_f1_train, X_f1_test, y_f1_test)
           
            ###SVM
            #modelSVM = algoritmi.SVM(X_f1_train, y_f1_train, X_f1_test, y_f1_test)
        elif(optiune == "2"):
            ###KNN
            modelKNN = algoritmi.knn(X_f1_train, y_f1_train, X_f1_test, y_f1_test) 
            #algoritmi.knn_cross(X_f1, Yo)
        elif(optiune == "3"):
            ###Decision Tree
            modelTree = algoritmi.decision_tree(X_f1_train, y_f1_train, X_f1_test, y_f1_test)
            #algoritmi.decision_tree_cross(X_f1, Yo)
        elif(optiune == "4"):
            ###Random Forest
            modelRandF = algoritmi.random_f(X_f1_train, y_f1_train, X_f1_test, y_f1_test)
            #algoritmi.random_f_cross(X_f1, Yo)
        elif(optiune == "5"):
            ###SVM
            modelSVM = algoritmi.SVM(X_f1_train, y_f1_train, X_f1_test, y_f1_test)
            #algoritmi.SVM_cross(X_f1, Yo)
    
    elif(optiuneClass == "2"):
        #binar
        X_f1_binar = Xo_binar.iloc[:, [24, 19, 23, 25, 27, 16]]
        X_f1_train_binar,X_f1_test_binar,y_f1_train_binar,y_f1_test_binar = train_test_split(X_f1_binar, Yo_binar, test_size=0.30, random_state=20)
    
        optiune = input("Alegeti algoritmul dorit: \n1) Toate \n2) KNN \n3) Decision Tree \n4) Random Forest \n5) SVM \n")
    
        if(optiune == "1"):
             ###KNN
            modelKNN_binar = algoritmi.knn_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar)
            
            ###Decision Tree
            modelTree_binar = algoritmi.decision_tree_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar)
            
            ###Random Forest
            modelRandF_binar = algoritmi.random_f_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar)
            
            ###SVM
            #modelSVM_binar = algoritmi.SVM_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar)
    
        elif(optiune == "2"):
            ###KNN
            modelKNN_binar = algoritmi.knn_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar)
            #algoritmi.knn_binar_cross(X_f1_binar, Yo_binar)
        elif(optiune == "3"):
            ###Decision Tree
            modelTree_binar = algoritmi.decision_tree_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar)
            #algoritmi.decision_tree_binar_cross(X_f1_binar, Yo_binar)
        elif(optiune == "4"):
            ###Random Forest
            modelRandF_binar = algoritmi.random_f_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar)       
            #algoritmi.random_f_binar_cross(X_f1_binar, Yo_binar)
        elif(optiune == "5"):
            ###SVM
            modelSVM_binar = algoritmi.SVM_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar)
            #algoritmi.SVM_binar_cross(X_f1_binar, Yo_binar)
        
    else:
        print("__finished.__")
    
    
    
    continua = input("Doriti sa testati pe alt set de date? (y/n)\n")
    if(continua == "y"):
        urlNou = input("Introduceti calea absoluta catre fisierul .pcap: ")
        
        cicflow.Apel_CICFLOW(urlNou, "/home/matei/WORK/cicflowCSV/")
        print("\n\nFisierul .pcap a fost precesat si a fost extras setul de date...\n")
        numeFis = urlNou[urlNou.rfind("/") + 1 : len(urlNou)-4] + "pcap_Flow.csv"
        urlNou = "/home/matei/WORK/cicflowCSV/" + numeFis
        print(urlNou)
        
        datasetNOU = read_csv(urlNou, encoding='iso-8859-1', skip_blank_lines=True, keep_default_na=False, dtype={"Flow ID": str, "Src IP": str, "Dst IP": str, "Timestamp": str, "Label": str})
        ################################prezentare date
    
        print("\nDimensiune set de date nou: ")
        print(datasetNOU.shape)   #cate randuri si coloane
        #vad tipul coloanelor
        for i in datasetNOU.columns:   
            print("'" + i + "' - " + str(datasetNOU[i].dtype))
    
        # class distribution
        print("\n------------------Distributie date noi dupa ' Label':")
        print(datasetNOU.groupby('Label').size())  #cate sunt din fiecare label final
    
        #datasetNOU_util = datasetNOU.copy().iloc[:, 1:80]
        #datasetNOU_util = datasetNOU.copy().loc[:, ~datasetNOU.columns.isin(['index','Flow ID', 'Source IP', 'Destination IP', 'Timestamp','Label'])]  #independent columns
        ##################CURATARE DATE 
        print("\ncuratare date...")
        datasetNOU_curat_all = curatare_date.curatare_test(datasetNOU)   
        datasetNOU_curat = datasetNOU_curat_all.copy().loc[:, ~datasetNOU_curat_all.columns.isin(['index','Flow ID', 'Src IP', 'Dst IP', 'Timestamp','Label'])]  #independent columns
        print("\nSet de date curatat: ")
        print(datasetNOU_curat.shape)
        
        
        if(optiuneClass == "1"):   
            #Multiclass
            X_nou = datasetNOU_curat.loc[:, ['Fwd IAT Std', 'Src Port', 'Flow IAT Max', 'Fwd IAT Max', 'Flow IAT Std', 'Dst Port', 'Fwd IAT Mean']]
            
            if(optiune == "1"):
                ###KNN
                predictieKNN = modelKNN.predict(X_nou)      
                df_out = datasetNOU_curat_all.assign(Label=predictieKNN)
                df_out = df_out.replace({'Label':{1: 'BENIGN', 2:'Web Attack - Brute Force', 3:'Web Attack - Sql Injection', 4:'Web Attack - XSS'}})
                print("\nSumar KNN: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_KNN_OUTTT.csv"          
                df_out.to_csv(fisOUT)
                
                ###Decision Tree
                predictieTree = modelTree.predict(X_nou)  
                df_out = datasetNOU_curat_all.assign(Label=predictieTree)
                df_out = df_out.replace({'Label':{1: 'BENIGN', 2:'Web Attack - Brute Force', 3:'Web Attack - Sql Injection', 4:'Web Attack - XSS'}})
                print("\nSumar Tree: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_Tree_OUTTT.csv"          
                df_out.to_csv(fisOUT)
                
                ###Random Forest
                predictieRandF = modelRandF.predict(X_nou)  
                df_out = datasetNOU_curat_all.assign(Label=predictieRandF)
                df_out = df_out.replace({'Label':{1: 'BENIGN', 2:'Web Attack - Brute Force', 3:'Web Attack - Sql Injection', 4:'Web Attack - XSS'}})
                print("\nSumar Rand Forest: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_RandF_OUTTT.csv"          
                df_out.to_csv(fisOUT)
                
                ###SVM
                #predictieSVM = modelSVM.predict(X_nou)  
                # df_out = datasetNOU_curat_all.assign(Label=predictieSVM)
                # df_out = df_out.replace({'Label':{1: 'BENIGN', 2:'Web Attack - Brute Force', 3:'Web Attack - Sql Injection', 4:'Web Attack - XSS'}})
                # print("\nSumar SVM: ")
                # print(df_out.groupby('Label').size())
                # indx = urlNou.index(".csv")
                # fisOUT = urlNou[0:indx] + "_SVM_OUTTT.csv"          
                # df_out.to_csv(fisOUT)
                
            elif(optiune == "2"):
                ###KNN
                predictieKNN = modelKNN.predict(X_nou) 
                df_out = datasetNOU_curat_all.assign(Label=predictieKNN)
                df_out = df_out.replace({'Label':{1: 'BENIGN', 2:'Web Attack - Brute Force', 3:'Web Attack - Sql Injection', 4:'Web Attack - XSS'}})
                print("\nSumar KNN: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_OUTTT.csv"          
                df_out.to_csv(fisOUT)
                
            elif(optiune == "3"):
                ###Decision Tree
                predictieTree = modelTree.predict(X_nou)
                df_out = datasetNOU_curat_all.assign(Label=predictieTree)
                df_out = df_out.replace({'Label':{1: 'BENIGN', 2:'Web Attack - Brute Force', 3:'Web Attack - Sql Injection', 4:'Web Attack - XSS'}})
                print("\nSumar Tree: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_Tree_OUTTT.csv"          
                df_out.to_csv(fisOUT)
                
            elif(optiune == "4"):
                ###Random Forest
                predictieRandF = modelRandF.predict(X_nou)
                df_out = datasetNOU_curat_all.assign(Label=predictieRandF)
                df_out = df_out.replace({'Label':{1: 'BENIGN', 2:'Web Attack - Brute Force', 3:'Web Attack - Sql Injection', 4:'Web Attack - XSS'}})
                print("\nSumar Rand Forest: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_RandF_OUTTT.csv"          
                df_out.to_csv(fisOUT)
                
            elif(optiune == "5"):
                ###SVM
                predictieSVM = modelSVM.predict(X_nou)
                df_out = datasetNOU_curat_all.assign(Label=predictieSVM)
                df_out = df_out.replace({'Label':{1: 'BENIGN', 2:'Web Attack - Brute Force', 3:'Web Attack - Sql Injection', 4:'Web Attack - XSS'}})
                print("\nSumar SVM: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_SVM_OUTTT.csv"          
                df_out.to_csv(fisOUT)
    
        elif(optiuneClass == "2"):
            #binar        
            X_nou = datasetNOU_curat.loc[:, ['Fwd IAT Std', 'Flow IAT Std', 'Fwd IAT Mean', 'Fwd IAT Max', 'Bwd IAT Tot', 'Flow Byts/s']]
            #alte featureuri 
            
            if(optiune == "1"):
                ###KNN
                predictieKNN_binar = modelKNN_binar.predict(X_nou)      
                df_out = datasetNOU_curat_all.assign(Label=predictieKNN_binar)
                df_out = df_out.replace({'Label':{0: 'BENIGN', 1:'MALIGN'}})
                print("\nSumar KNN: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_KNNbin_OUTTT.csv"          
                df_out.to_csv(fisOUT)
                
                ###Decision Tree
                predictieTree_binar = modelTree_binar.predict(X_nou)  
                df_out = datasetNOU_curat_all.assign(Label=predictieTree_binar)
                df_out = df_out.replace({'Label':{0: 'BENIGN', 1:'MALIGN'}})
                print("\nSumar Tree: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_Treebin_OUTTT.csv"          
                df_out.to_csv(fisOUT)
        
                ###Random Forest
                predictieRandF_binar = modelRandF_binar.predict(X_nou)
                df_out = datasetNOU_curat_all.assign(Label=predictieRandF_binar)
                df_out = df_out.replace({'Label':{0: 'BENIGN', 1:'MALIGN'}})
                print("\nSumar Rand Forest: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_RandFbin_OUTTT.csv"          
                df_out.to_csv(fisOUT)           
               
                ###SVM
                #predictieSVM_binar = modelSVM_binar.predict(X_nou)  
                # df_out = datasetNOU_curat_all.assign(Label=predictieSVM_binar)
                # df_out = df_out.replace({'Label':{0: 'BENIGN', 1:'MALIGN'}})
                # print("\nSumar SVM: ")
                # print(df_out.groupby('Label').size())
                # indx = urlNou.index(".csv")
                # fisOUT = urlNou[0:indx] + "_SVMbin_OUTTT.csv"          
                # df_out.to_csv(fisOUT)
                
            elif(optiune == "2"):
                ###KNN
                predictieKNN_binar = modelKNN_binar.predict(X_nou) 
                df_out = datasetNOU_curat_all.assign(Label=predictieKNN_binar)
                df_out = df_out.replace({'Label':{0: 'BENIGN', 1:'MALIGN'}})
                print("\nSumar KNN: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_KNNbin_OUTTT.csv"          
                df_out.to_csv(fisOUT)
                
            elif(optiune == "3"):
                ###Decision Tree
                predictieTree_binar = modelTree_binar.predict(X_nou)
                df_out = datasetNOU_curat_all.assign(Label=predictieTree_binar)
                df_out = df_out.replace({'Label':{0: 'BENIGN', 1:'MALIGN'}})
                print("\nSumar Tree: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_Treebin_OUTTT.csv"          
                df_out.to_csv(fisOUT)
                
            elif(optiune == "4"):
                ###Random Forest
                predictieRandF_binar = modelRandF_binar.predict(X_nou)
                df_out = datasetNOU_curat_all.assign(Label=predictieRandF_binar)
                df_out = df_out.replace({'Label':{0: 'BENIGN', 1:'MALIGN'}})
                print("\nSumar Rand Forest: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_RandFbin_OUTTT.csv"          
                df_out.to_csv(fisOUT) 
                
            elif(optiune == "5"):
                ###SVM
                predictieSVM_binar = modelSVM_binar.predict(X_nou)
                df_out = datasetNOU_curat_all.assign(Label=predictieSVM_binar)
                df_out = df_out.replace({'Label':{0: 'BENIGN', 1:'MALIGN'}})
                print("\nSumar SVM: ")
                print(df_out.groupby('Label').size())
                indx = urlNou.index(".csv")
                fisOUT = urlNou[0:indx] + "_SVMbin_OUTTT.csv"          
                df_out.to_csv(fisOUT)
        
            
    else:
        print("~finished~")    
        
        
    print("--GATA--")