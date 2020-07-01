# -*- coding: utf-8 -*-

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

url = "/home/matei/WORK/csv_cu_label/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
dataset = read_csv(url, encoding='iso-8859-1', skip_blank_lines=True, keep_default_na=False, dtype={"Flow ID": str, "Src IP": str, "Dst IP": str, "Timestamp": str, "Label": str})

################################prezentare date

print("Dimensiune set de date: ")
print(dataset.shape)   #cate randuri si coloane

#vad tipul coloanelor
# for i in dataset.columns:   
#    print("'" + i + "' - " + str(dataset[i].dtype))

# class distribution
print("\n------------------Distributie date dupa ' Label':")
print(dataset.groupby(' Label').size())  #cate sunt din fiecare label final


##################################curatare date
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df trebuie sa fie pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]      #.astype(np.float64)

dataset_clean = clean_dataset(dataset)
#coloana asta nush de ce era de tip Object asa ca am convertit-o in float64
dataset_clean['Flow Bytes/s'] = dataset_clean['Flow Bytes/s'].astype(np.float64)
#print(dataset_clean['Flow Bytes/s'].dtype)

############################pastrare doar a datelor pozitive
dataset_clean_poz = dataset_clean[(dataset_clean[[' Flow Duration', 'Flow Bytes/s', ' Flow IAT Min', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward']] >= 0).all(axis=1)]
print("\n------------------Distributie date pozitive dupa ' Label':")
print(dataset_clean_poz.groupby(' Label').size()) 

###########verificare daca pt coloanele de tip text, am valori relative reduse ca nr unic, ca sa vad
##########daca pot sa le inlocuiesc cu valori numerice generice (1,2,3...)
##dataset_clean_poz_numbers = dataset_clean_poz.loc[:, ~dataset_clean_poz.columns.isin(['index','Flow ID', ' Source IP', ' Destination IP', ' Timestamp',' Label'])]  #independent columns

# print("\nValori unice coloana 'Flow ID': ")
# print(dataset_clean_poz['Flow ID'].unique()) #prea multe
# cont = 0
# for j in dataset_clean_poz['Flow ID'].unique():
#      cont = cont + 1
# print("Nr valori unice: " + str(cont) + "\n")

# print("\nValori unice coloana ' Source IP': ")
# print(dataset_clean_poz[' Source IP'].unique()) #prea multe
# cont = 0
# for j in dataset_clean_poz[' Source IP'].unique():
#      cont = cont + 1
# print("Nr valori unice: " + str(cont) + "\n")

# print("\nValori unice coloana ' Destination IP': ")
# print(dataset_clean_poz[' Destination IP'].unique()) #prea multe
# cont = 0
# for j in dataset_clean_poz[' Destination IP'].unique():
#      cont = cont + 1
# print("Nr valori unice: " + str(cont) + "\n")


#################Conversie Label in clasificator multiclass numeric
dataset_clean_poz = dataset_clean_poz.replace({' Label':{'BENIGN': 1, 'Web Attack - Brute Force': 2, 'Web Attack - Sql Injection': 3, 'Web Attack - XSS': 4}})
print(dataset_clean_poz.groupby(' Label').size())

################Conversie Label in clasificator binar
dataset_binar = dataset_clean_poz.copy()
#dataset_binar = dataset_binar.replace({' Label':{'BENIGN': 0, 'Web Attack - Brute Force': 1, 'Web Attack - Sql Injection': 1, 'Web Attack - XSS': 1}})
dataset_binar = dataset_binar.replace({' Label':{1: 0, 2: 1, 3: 1, 4: 1}})
print(dataset_binar.groupby(' Label').size()) 

#################Separare features de label
X_all = dataset_clean_poz.loc[:, ~dataset_clean_poz.columns.isin(['index','Flow ID', ' Source IP', ' Destination IP', ' Timestamp',' Label'])]  #independent columns
y_all = dataset_clean_poz.iloc[:,-1]    

X_all_binar = dataset_binar.loc[:, ~dataset_binar.columns.isin(['index','Flow ID', ' Source IP', ' Destination IP', ' Timestamp',' Label'])]  #independent columns
y_all_binar = dataset_binar.iloc[:,-1]
################Normalizare || Standardizare

#Normalizare
normalized_X = preprocessing.normalize(X_all)
df_normalized_X = pd.DataFrame(normalized_X, columns=X_all.columns)


#Standardizare
# standardized_X = preprocessing.scale(X_all)
# df_standardized_X = pd.DataFrame(standardized_X)


############BALANCING DATA

#fac un dataframe nou cu toate feature-urile normalizate + labels
df_y_all = pd.DataFrame(y_all);
df_norX_Y = pd.concat([df_normalized_X.reset_index(drop=True), df_y_all.reset_index(drop=True)], axis=1, verify_integrity=True)  #am pus asa cu reset ca e prost Pandas si adauga randuri altfel
#df_norX_Y = df_normalized_X.join(y_all) #strica labels

df_y_all_binar = pd.DataFrame(y_all_binar);
df_norX_Y_binar = pd.concat([df_normalized_X.reset_index(drop=True), df_y_all_binar.reset_index(drop=True)], axis=1, verify_integrity=True)  #am pus asa cu reset ca e prost Pandas si adauga randuri altfel

###multiclass
#plot cu distributia pe categorii
count_classes = pd.value_counts(df_norX_Y[' Label'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
pyplot.title("Distributia pe clase multiclass")
#pyplot.xticks(range(2), LABELS)
pyplot.xlabel("Clasa")
pyplot.ylabel("Frecventa")
#afisat si in consola distributia
print(df_norX_Y.groupby(' Label').size())  #cate sunt din fiecare label final

print(df_norX_Y.shape)
print("\n --------------BALANCING DATA------------------------\n")
print("Initial: \n")
print(df_normalized_X.shape)
print(y_all.shape)

benign = df_norX_Y[df_norX_Y[' Label']==1]
bruteForce = df_norX_Y[df_norX_Y[' Label']==2]
sqlInj = df_norX_Y[df_norX_Y[' Label']==3]
xss= df_norX_Y[df_norX_Y[' Label']==4]
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
print("\nDupa oversampling: ")
print(oversampled.groupby(' Label').size())  #cate sunt din fiecare label final
#plot cu distributia pe categorii dupa oversample
count_classes = pd.value_counts(oversampled[' Label'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
pyplot.title("Distributia pe clase multiclass dupa oversample")
#pyplot.xticks(range(2), LABELS)
pyplot.xlabel("Clasa")
pyplot.ylabel("Frecventa")


########clasificare binara
#plot cu distributia pe categorii
count_classes = pd.value_counts(df_norX_Y_binar[' Label'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
pyplot.title("Distributia pe clase binar")
#pyplot.xticks(range(2), LABELS)
pyplot.xlabel("Clasa")
pyplot.ylabel("Frecventa")
#afisat si in consola distributia
print(df_norX_Y_binar.groupby(' Label').size())  #cate sunt din fiecare label final

print(df_norX_Y.shape)
print("\n --------------BALANCING DATA------------------------\n")
print("Initial: \n")
print(df_normalized_X.shape)
print(y_all_binar.shape)

benign = df_norX_Y_binar[df_norX_Y_binar[' Label']==0]
malign = df_norX_Y_binar[df_norX_Y_binar[' Label']!=0]

print(benign.shape, malign.shape)

from sklearn.utils import resample

malign_up = resample(malign,
                          replace=True, # sample with replacement
                          n_samples=int(1*len(benign)), # match number in majority class
                          random_state=20) # reproducible results


oversampled_binar = pd.concat([benign, malign_up])
print("\nDupa oversampling: ")
print(oversampled_binar.groupby(' Label').size())  #cate sunt din fiecare label final
#plot cu distributia pe categorii dupa oversample
count_classes = pd.value_counts(oversampled_binar[' Label'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
pyplot.title("Distributia pe clase dupa oversample")
#pyplot.xticks(range(2), LABELS)
pyplot.xlabel("Clasa")
pyplot.ylabel("Frecventa")


##############################SELECTIE FEATURES

Xo = oversampled.iloc[:,0:80]
Yo = oversampled.iloc[:,-1]

Xo_binar = oversampled_binar.iloc[:,0:80]
Yo_binar = oversampled_binar.iloc[:,-1]

# #multiclass
# # Universal Selection
# # apply SelectKBest class to extract top 10 best features
# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(Xo, Yo)
# #fit = bestfeatures.fit(df_standardized_X, y_all)  #cu asta nu merge pt ca are valori negative
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X_all.columns)
# #concat pt prezentare
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Feature','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features

# #binar
# #Universal Selection
# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(Xo_binar, Yo_binar)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X_all.columns)
# #concat pt prezentare
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Feature','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features

#----------------------------------

# Feature Importance - multiclass
pyplot.clf();
from sklearn.ensemble import ExtraTreesClassifier

# model = ExtraTreesClassifier()
# model.fit(Xo, Yo)
# print(model.feature_importances_) 
# #plot graph 4 vizualizare
# feat_importances = pd.Series(model.feature_importances_, index=X_all.columns)
# feat_importances.nlargest(10).plot(kind='bar')
# pyplot.title("Importanta Feature")
# pyplot.ylabel("Scor")
# pyplot.xlabel("Feature")
# pyplot.xticks(rotation=90)
# pyplot.show()
# print(feat_importances.nlargest(10));

# Feature Importance - binar
# pyplot.clf();

# model = ExtraTreesClassifier()
# model.fit(Xo_binar, Yo_binar)
# print(model.feature_importances_)
# #plot
# feat_importances = pd.Series(model.feature_importances_, index=X_all.columns)
# feat_importances.nlargest(10).plot(kind = 'bar')
# pyplot.title("Importanta Feature (binar)")
# pyplot.ylabel("Scor")
# pyplot.xlabel("Feature")
# pyplot.xticks(rotation=90)
# pyplot.show()
# print(feat_importances.nlargest(10));

#-------------------------------------------------

###corelation matrix - multiclass
# cormat = df_norX_Y.corr();
# cor_target = abs(cormat[" Label"])
# print("\nFeature-uri relavante din Matricea de Corelare: ")
# relevant_features = cor_target[cor_target>0.10]
# #relevant_features.sort_values()
# print(relevant_features.sort_values(ascending=False))

###corelation matrix - binar
# cormat = df_norX_Y_binar.corr();
# cor_target = abs(cormat[" Label"])
# print("\nFeature-uri relavante din Matricea de Corelare: ")
# relevant_features = cor_target[cor_target>0.10]
# #relevant_features.sort_values()
# print(relevant_features.sort_values(ascending=False))

#-----------------------------------

#cu random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# X_train,X_test,y_train,y_test = train_test_split(Xo, Yo, test_size=0.30, random_state=20)

# clf = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
# # Antrenez
# clf.fit(X_train, y_train)

# cont = 0
# for i in clf.feature_importances_:
#     if i > 0.01:
#         print(str(cont) + ": " + Xo.columns[cont] + " - " + str(i))
#     cont=cont+1
    
#si varianta pt binar
# X_train,X_test,y_train,y_test = train_test_split(Xo_binar, Yo_binar, test_size=0.30, random_state=20)

# clf = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
# # Antrenez
# clf.fit(X_train, y_train)

# cont = 0
# for i in clf.feature_importances_:
#     if i > 0.01:
#         print(str(cont) + ": " + Xo.columns[cont] + " - " + str(i))
#     cont=cont+1


###############################################
##################################
#construim X si Y doar pt cele mai relevante featureuri
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix


#Multiclass
#X_f1 = Xo.iloc[:,[0,1,16,17,23,24,25]]  #univeariate selection
X_f1 = Xo.iloc[:, [24, 0, 20, 25, 19, 1, 23]]
X_f1_train,X_f1_test,y_f1_train,y_f1_test = train_test_split(X_f1, Yo, test_size=0.30, random_state=20)

#binar
X_f1_binar = Xo_binar.iloc[:, [24, 19, 23, 25, 27, 16]]
X_f1_train_binar,X_f1_test_binar,y_f1_train_binar,y_f1_test_binar = train_test_split(X_f1_binar, Yo_binar, test_size=0.30, random_state=20)


# # KNN - multiclass
# pyplot.clf()
# model = KNeighborsClassifier();
# model.fit(X_f1_train, y_f1_train)
# print("\n\nPredictie cu KNeighboursClassifier: ")
# predictions = model.predict(X_f1_test)
# print("\nAccuracy:")
# print(accuracy_score(y_f1_test, predictions))
# print("\nMatricea de confuzie:")
# matrice = confusion_matrix(y_f1_test, predictions)
# print(matrice)
# pyplot.matshow(matrice)
# #--
# heatmap_df = pd.DataFrame(matrice, ['BENIGN', 'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS' ], ['BENIGN', 'Web Attack - Brute Force', 'Web Attack - Sql Injection', 'Web Attack - XSS' ])
# sn.set(font_scale=1)
# sn.heatmap(heatmap_df, annot=True)
# pyplot.show()

#KNN - binar
# pyplot.clf()
# model = KNeighborsClassifier();
# model.fit(X_f1_train_binar, y_f1_train_binar)
# print("\n\nPredictie cu KNeighboursClassifier: ")
# predictions = model.predict(X_f1_test_binar)
# print("\nAccuracy:")
# print(accuracy_score(y_f1_test_binar, predictions))
# print("\nMatricea de confuzie:")
# matrice = confusion_matrix(y_f1_test_binar, predictions)
# print(matrice)
# pyplot.matshow(matrice)
# #--
# heatmap_df = pd.DataFrame(matrice, ['BENIGN', 'MALIGN'], ['BENIGN', 'MALIGN'])
# sn.set(font_scale=1)
# sn.heatmap(heatmap_df, annot=True)
# pyplot.show()

#################Decision Tree
# from matplotlib import pyplot 
# tree = DecisionTreeClassifier(random_state=20)

##multiclass
# tree.fit(X_f1_train, y_f1_train)
# #print(f'Model Accuracy: {tree.score(X_f1_test, y_f1_test)}')
# y_pred = tree.predict(X_f1_test)
# print("Decision Tree Multiclass - Accuracy:", metrics.accuracy_score(y_f1_test, y_pred))
# matrice = confusion_matrix(y_f1_test, y_pred)
# print(matrice)
# #pyplot.matshow(matrice)
# #pyplot.plot(matrice)
# disp = plot_confusion_matrix(tree, X_f1_test, y_f1_test, values_format = 'd')
# #disp.ax_.set_title()
# pyplot.rcParams["axes.grid"] = False
# pyplot.ylabel("Real")
# pyplot.xlabel("Prezis")
# pyplot.title("Decision Tree - multiclass")

##binar
# tree.fit(X_f1_train_binar, y_f1_train_binar)
# #print(f'Model Accuracy: {tree.score(X_f1_test, y_f1_test)}')
# y_pred = tree.predict(X_f1_test_binar)
# print("Decision Tree Multiclass - Accuracy:", metrics.accuracy_score(y_f1_test_binar, y_pred))
# matrice = confusion_matrix(y_f1_test_binar, y_pred)
# print(matrice)
# #pyplot.matshow(matrice)
# #pyplot.plot(matrice)
# disp = plot_confusion_matrix(tree, X_f1_test_binar, y_f1_test_binar, values_format = 'd')
# #disp.ax_.set_title()
# pyplot.rcParams["axes.grid"] = False
# pyplot.ylabel("Real")
# pyplot.xlabel("Prezis")
# pyplot.title("Decision Tree - binar")


#####################Random Forest
# model = RandomForestClassifier(n_estimators=100,
#                                max_depth = 6,
#                                bootstrap = True,
#                                max_features = 'sqrt')
#####multiclass
# # Fit on training data
# model.fit(X_f1_train, y_f1_train)
# # Actual class predictions
# rf_predictions = model.predict(X_f1_test)
# # Probabilities for each class
# rf_probs = model.predict_proba(X_f1_test)[:, 1]
# from sklearn.metrics import roc_auc_score
# # Calculate roc auc
# # roc_value = roc_auc_score(y_f1_test, rf_probs)  #poate la binar

# # # Calculate the absolute errors
# # errors = abs(rf_predictions - y_f1_test)
# # # Calculate mean absolute percentage error (MAPE)
# # mape = 100 * (errors / y_f1_test)
# # # Calculate and display accuracy
# # accuracy = 100 - np.mean(mape)
# # print('Accuracy:', round(accuracy, 2), '%.')
# print("Rand Forest - Accuracy:", metrics.accuracy_score(y_f1_test, rf_predictions))
# matrice = confusion_matrix(y_f1_test, rf_predictions)
# print(matrice)
# #pyplot.matshow(matrice)
# #pyplot.plot(matrice)
# disp = plot_confusion_matrix(model, X_f1_test, y_f1_test, values_format = 'd')
# #disp.ax_.set_title()
# pyplot.rcParams["axes.grid"] = False
# pyplot.ylabel("Real")
# pyplot.xlabel("Prezis")
# pyplot.title("Random Forest - multiclass")


# #####binar
# # Fit on training data
# model.fit(X_f1_train_binar, y_f1_train_binar)
# # Actual class predictions
# rf_predictions = model.predict(X_f1_test_binar)
# # Probabilities for each class
# rf_probs = model.predict_proba(X_f1_test_binar)[:, 1]
# from sklearn.metrics import roc_auc_score
# # Calculate roc auc
# roc_value = roc_auc_score(y_f1_test_binar, rf_probs)  #poate la binar
# print("\n\nRand Forest - Accuracy:", metrics.accuracy_score(y_f1_test_binar, rf_predictions))
# matrice = confusion_matrix(y_f1_test_binar, rf_predictions)
# print(matrice)
# #pyplot.matshow(matrice)
# #pyplot.plot(matrice)
# disp = plot_confusion_matrix(model, X_f1_test_binar, y_f1_test_binar, values_format = 'd')
# #disp.ax_.set_title()
# pyplot.rcParams["axes.grid"] = False
# pyplot.ylabel("Real")
# pyplot.xlabel("Prezis")
# pyplot.title("Random Forest - binar")


######################SVM
#SVM=dureaza o vesnicie
#multiclass
# model = SVC(gamma='auto')
# model.fit(X_f1_train, y_f1_train)
# print("\nPredictie cu SVM: ")
# predictions = model.predict(X_f1_test)
# # Evaluate predictions
# print("Accuracy: ")
# print(accuracy_score(y_f1_test, predictions))
# print("\n Confussion Matrix: ")
# matrice = confusion_matrix(y_f1_test, predictions)
# print(matrice)
# disp = plot_confusion_matrix(model, X_f1_test, y_f1_test, values_format = 'd')
# pyplot.rcParams["axes.grid"] = False
# pyplot.ylabel("Real")
# pyplot.xlabel("Prezis")
# pyplot.title("SVM - multiclass")

#Binar
model = SVC(kernel='poly', gamma='auto')
model.fit(X_f1_train_binar, y_f1_train_binar)
print("\nPredictie cu SVM: ")
predictions = model.predict(X_f1_test_binar)
# Evaluate predictions
print("Accuracy: ")
print(accuracy_score(y_f1_test_binar, predictions))
print("\n Confussion Matrix: ")
matrice = confusion_matrix(y_f1_test_binar, predictions)
print(matrice)
disp = plot_confusion_matrix(model, X_f1_test_binar, y_f1_test_binar, values_format = 'd')
pyplot.rcParams["axes.grid"] = False
pyplot.ylabel("Real")
pyplot.xlabel("Prezis")
pyplot.title("SVM - binar")

print("__GATA__")
