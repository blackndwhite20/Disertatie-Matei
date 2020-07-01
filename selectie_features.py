# -*- coding: utf-8 -*-
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def universal_selection(Xo, Yo, X_all):
    #multiclass
    # Universal Selection
    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(Xo, Yo)
    #fit = bestfeatures.fit(df_standardized_X, y_all)  #cu asta nu merge pt ca are valori negative
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_all.columns)
    #concat pt prezentare
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features
    
def universal_selection_binar(Xo_binar, Yo_binar, X_all):
    #binar
    #Universal Selection
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(Xo_binar, Yo_binar)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_all.columns)
    #concat pt prezentare
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features
    
def feature_importance(Xo, Yo, X_all):
    # Feature Importance - multiclass
    pyplot.figure()
    pyplot.clf()
     
    model = ExtraTreesClassifier()
    model.fit(Xo, Yo)
    print(model.feature_importances_) 
    #plot graph 4 vizualizare
    feat_importances = pd.Series(model.feature_importances_, index=X_all.columns)
    feat_importances.nlargest(10).plot(kind='bar')
    pyplot.title("Importanta Feature")
    pyplot.ylabel("Scor")
    pyplot.xlabel("Feature")
    pyplot.xticks(rotation=90)
    pyplot.show()
    print(feat_importances.nlargest(10))
    
def feature_importances_binar(Xo_binar, Yo_binar, X_all):
    # Feature Importance - binar
    pyplot.figure()
    pyplot.clf()
    
    model = ExtraTreesClassifier()
    model.fit(Xo_binar, Yo_binar)
    print(model.feature_importances_)
    #plot
    feat_importances = pd.Series(model.feature_importances_, index=X_all.columns)
    feat_importances.nlargest(10).plot(kind = 'bar')
    pyplot.title("Importanta Feature (binar)")
    pyplot.ylabel("Scor")
    pyplot.xlabel("Feature")
    pyplot.xticks(rotation=90)
    pyplot.show()
    print(feat_importances.nlargest(10));

def cor_mat(df_norX_Y):
    ###corelation matrix - multiclass
    cormat = df_norX_Y.corr();
    cor_target = abs(cormat[" Label"])
    print("\nFeature-uri relavante din Matricea de Corelare: ")
    relevant_features = cor_target[cor_target>0.10]
    #relevant_features.sort_values()
    print(relevant_features.sort_values(ascending=False))
    
def cor_mat_binar(df_norX_Y_binar):
    ###corelation matrix - binar
    cormat = df_norX_Y_binar.corr();
    cor_target = abs(cormat[" Label"])
    print("\nFeature-uri relavante din Matricea de Corelare: ")
    relevant_features = cor_target[cor_target>0.10]
    #relevant_features.sort_values()
    print(relevant_features.sort_values(ascending=False))
    
def random_for(Xo, Yo):
    X_train,X_test,y_train,y_test = train_test_split(Xo, Yo, test_size=0.30, random_state=20)
    clf = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    # Antrenez
    clf.fit(X_train, y_train)
    
    cont = 0
    for i in clf.feature_importances_:
        if i > 0.01:
            print(str(cont) + ": " + Xo.columns[cont] + " - " + str(i))
        cont=cont+1
    
def rand_for_binar(Xo_binar, Yo_binar, Xo):
    X_train,X_test,y_train,y_test = train_test_split(Xo_binar, Yo_binar, test_size=0.30, random_state=20)
    
    clf = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    # Antrenez
    clf.fit(X_train, y_train)
    
    cont = 0
    for i in clf.feature_importances_:
        if i > 0.01:
            print(str(cont) + ": " + Xo.columns[cont] + " - " + str(i))
        cont=cont+1
