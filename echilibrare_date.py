# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sn
import numpy as np
from matplotlib import pyplot 

def echilibrare(df_norX_Y, df_normalized_X, y_all):
    
    ###multiclass
    #plot cu distributia pe categorii
    count_classes = pd.value_counts(df_norX_Y[' Label'], sort = True)
    pyplot.figure()
    count_classes.plot(kind = 'bar', rot=0)
    pyplot.title("Distributia pe clase multiclass")
    #pyplot.xticks(range(2), LABELS)
    pyplot.xlabel("Clasa")
    pyplot.ylabel("Frecventa")
    #afisat si in consola distributia
    print("\n\nDistributia multiclasa inainte de echilibrare: ")
    print(df_norX_Y.groupby(' Label').size())  #cate sunt din fiecare label final   
    print(df_norX_Y.shape)
    print("\n --------------BALANCING DATA------------------------\n")
    print("Initial: ")
    print("Dimensiune set")
    print(df_normalized_X.shape)
    #print(y_all.shape)
    
    benign = df_norX_Y[df_norX_Y[' Label']==1]
    bruteForce = df_norX_Y[df_norX_Y[' Label']==2]
    sqlInj = df_norX_Y[df_norX_Y[' Label']==3]
    xss= df_norX_Y[df_norX_Y[' Label']==4]
    print("\nDimensiune clase")
    print(benign.shape, bruteForce.shape, sqlInj.shape, xss.shape)
    
    from sklearn.utils import resample
    
    bruteForce_up = resample(bruteForce,
                              replace=True, # sample with replacement
                              n_samples=int(0.4*len(benign)), # match number in majority class
                              random_state=20) # reproducible results
    
    sqlInj_up = resample(sqlInj,
                              replace=True, # sample with replacement
                              n_samples=int(0.4*len(benign)), # match number in majority class
                              random_state=20) # reproducible results
    
    xss_up = resample(xss,
                  replace=True, # sample with replacement
                  n_samples=int(0.4*len(benign)), # match number in majority class
                  random_state=20) # reproducible results 
    
    oversampled = pd.concat([benign, bruteForce_up, sqlInj_up, xss_up])
    pyplot.figure()
    print("\nDupa multiclasa dupa echilibrare: ")
    print(oversampled.groupby(' Label').size())  #cate sunt din fiecare label final
    #plot cu distributia pe categorii dupa oversample
    count_classes = pd.value_counts(oversampled[' Label'], sort = True)
    count_classes.plot(kind = 'bar', rot=0)
    pyplot.title("Distributia pe clase multiclass dupa oversample")
    #pyplot.xticks(range(2), LABELS)
    pyplot.xlabel("Clasa")
    pyplot.ylabel("Frecventa")
    
    return oversampled


def echilibrare_binara(df_norX_Y_binar, df_normalized_X, y_all_binar):
    
    ########clasificare binara
    #plot cu distributia pe categorii
    count_classes = pd.value_counts(df_norX_Y_binar[' Label'], sort = True)
    pyplot.figure()
    count_classes.plot(kind = 'bar', rot=0)
    pyplot.title("Distributia pe clase binar")
    #pyplot.xticks(range(2), LABELS)
    pyplot.xlabel("Clasa")
    pyplot.ylabel("Frecventa")
    #afisat si in consola distributia
    print("\n\nDistributia pe clase binara inainte de echilibrare:")
    print(df_norX_Y_binar.groupby(' Label').size())  #cate sunt din fiecare label final
    
    print(df_norX_Y_binar.shape)
    print("\n --------------BALANCING DATA------------------------\n")
    print("Initial: ")
    print("Dimensiune set")
    print(df_normalized_X.shape)
    #print(y_all_binar.shape)
    
    benign = df_norX_Y_binar[df_norX_Y_binar[' Label']==0]
    malign = df_norX_Y_binar[df_norX_Y_binar[' Label']!=0]
    print("\nDimensiune clase")
    print(benign.shape, malign.shape)
    
    from sklearn.utils import resample
    
    malign_up = resample(malign,
                              replace=True, # sample with replacement
                              n_samples=int(1*len(benign)), # match number in majority class
                              random_state=20) # reproducible results
    
    
    oversampled_binar = pd.concat([benign, malign_up])
    pyplot.figure()
    print("\nDistributia pe clase binara dupa echilibrare: ")
    print(oversampled_binar.groupby(' Label').size())  #cate sunt din fiecare label final
    #plot cu distributia pe categorii dupa oversample
    count_classes = pd.value_counts(oversampled_binar[' Label'], sort = True)
    count_classes.plot(kind = 'bar', rot=0)
    pyplot.title("Distributia pe clase dupa oversample")
    #pyplot.xticks(range(2), LABELS)
    pyplot.xlabel("Clasa")
    pyplot.ylabel("Frecventa")
    pyplot.show()
    return oversampled_binar