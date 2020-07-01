# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sn
import numpy as np
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import StratifiedKFold
#from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score


def knn(X_f1_train, y_f1_train, X_f1_test, y_f1_test):
    # # KNN - multiclass
    pyplot.figure()
    pyplot.clf()
    #model = KNeighborsClassifier(n_neighbors = 20);
    model = KNeighborsClassifier(n_neighbors = 4);
    model.fit(X_f1_train, y_f1_train)
    print("\n\nPredictie cu KNeighboursClassifier: ")
    predictions = model.predict(X_f1_test)
    print("\nAccuracy:")
    print(accuracy_score(y_f1_test, predictions))
    print("\nMatricea de confuzie:")
    matrice = confusion_matrix(y_f1_test, predictions)
    print(matrice)
    # pyplot.matshow(matrice)
    # #--
    # heatmap_df = pd.DataFrame(matrice, ['BENIGN', 'Web Attack - Brute Force', 
    #                                     'Web Attack - Sql Injection', 'Web Attack - XSS' ],
    #                           ['BENIGN', 'Web Attack - Brute Force', 
    #                            'Web Attack - Sql Injection', 'Web Attack - XSS' ])
    # sn.set(font_scale=1)
    # sn.heatmap(heatmap_df, annot=True)
    # pyplot.show()
    disp = plot_confusion_matrix(model, X_f1_test, y_f1_test, values_format = 'd')
    pyplot.rcParams["axes.grid"] = False
    pyplot.ylabel("Real")
    pyplot.xlabel("Prezis")
    pyplot.title("KNN - multiclasa")
    pyplot.show() 
    return model

def knn_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar):
    #KNN - binar
    from matplotlib import pyplot 
    pyplot.figure()
    pyplot.clf()
    model = KNeighborsClassifier(n_neighbors = 20);
    model.fit(X_f1_train_binar, y_f1_train_binar)
    print("\n\nPredictie cu KNeighboursClassifier: ")
    predictions = model.predict(X_f1_test_binar)
    print("\nAccuracy:")
    print(accuracy_score(y_f1_test_binar, predictions))
    print("\nMatricea de confuzie:")
    matrice = confusion_matrix(y_f1_test_binar, predictions)
    print(matrice)
    # pyplot.matshow(matrice)
    # #--
    # heatmap_df = pd.DataFrame(matrice, ['BENIGN', 'MALIGN'], ['BENIGN', 'MALIGN'])
    # sn.set(font_scale=1)
    # sn.heatmap(heatmap_df, annot=True)
    disp = plot_confusion_matrix(model, X_f1_test_binar, y_f1_test_binar, values_format = 'd')
    pyplot.rcParams["axes.grid"] = False
    pyplot.ylabel("Real")
    pyplot.xlabel("Prezis")
    pyplot.title("KNN - binar")
    pyplot.show()   
    return model

def knn_binar_cross(Xall, Yall):   
    alg = KNeighborsClassifier(n_neighbors = 20)
    
    print("\nCross Validation....")
    kf = KFold(n_splits=3 , shuffle=True)
    #kf.split(Xall)    
    # Initializez lista cu accuracy
    accuracy_model = []
 
    for train_index, test_index in kf.split(Xall):
        # Split train-test
        X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
        #y_train, y_test = Yall[train_index], Yall[test_index]
        y_train, y_test = Yall.iloc[train_index], Yall.iloc[test_index]
        # Train the model
        model = alg.fit(X_train, y_train)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    #accuracy pt cele K teste 
    print(accuracy_model)
    
def knn_cross(Xall, Yall):   
    alg = KNeighborsClassifier(n_neighbors = 4)
    
    print("\nCross Validation....")
    kf = KFold(n_splits=3 , shuffle=True)
    #kf.split(Xall)    
    # Initializez lista cu accuracy
    accuracy_model = []
 
    for train_index, test_index in kf.split(Xall):
        # Split train-test
        X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
        #y_train, y_test = Yall[train_index], Yall[test_index]
        y_train, y_test = Yall.iloc[train_index], Yall.iloc[test_index]
        # Train the model
        model = alg.fit(X_train, y_train)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    #accuracy pt cele K teste 
    print(accuracy_model)    
    
def decision_tree(X_f1_train, y_f1_train, X_f1_test, y_f1_test):
    from matplotlib import pyplot 
    tree = DecisionTreeClassifier(random_state=20, criterion="gini", max_depth = 10)
    pyplot.figure()
    #multiclass
    tree.fit(X_f1_train, y_f1_train)
    print("\n\nPredictie cu arbore de decizie: ")
    #print(f'Model Accuracy: {tree.score(X_f1_test, y_f1_test)}')
    y_pred = tree.predict(X_f1_test)
    print("\nAccuracy:", metrics.accuracy_score(y_f1_test, y_pred))
    matrice = confusion_matrix(y_f1_test, y_pred)
    print("\nMatricea de confuzie:")
    print(matrice)
    #pyplot.matshow(matrice)
    #pyplot.plot(matrice)
    disp = plot_confusion_matrix(tree, X_f1_test, y_f1_test, values_format = 'd')
    #disp.ax_.set_title()
    pyplot.rcParams["axes.grid"] = False
    pyplot.ylabel("Real")
    pyplot.xlabel("Prezis")
    pyplot.title("Decision Tree - multiclass")
    pyplot.show()
    return tree
    
def decision_tree_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar):
    from matplotlib import pyplot 
    tree = DecisionTreeClassifier(random_state=20, criterion="gini", max_depth=2)
    pyplot.figure()
    #binar
    tree.fit(X_f1_train_binar, y_f1_train_binar)
    print("\n\nPredictie cu arbore de decizie: ")
    #print(f'Model Accuracy: {tree.score(X_f1_test, y_f1_test)}')
    y_pred = tree.predict(X_f1_test_binar)
    print("\nAccuracy:", metrics.accuracy_score(y_f1_test_binar, y_pred))
    matrice = confusion_matrix(y_f1_test_binar, y_pred)
    print("\nMatricea de confuzie:")
    print(matrice)
    #pyplot.matshow(matrice)
    #pyplot.plot(matrice)
    disp = plot_confusion_matrix(tree, X_f1_test_binar, y_f1_test_binar, values_format = 'd')
    #disp.ax_.set_title()
    pyplot.rcParams["axes.grid"] = False
    pyplot.ylabel("Real")
    pyplot.xlabel("Prezis")
    pyplot.title("Decision Tree - binar")
    pyplot.show()
    return tree

def decision_tree_binar_cross(Xall, Yall):   
    alg = DecisionTreeClassifier(random_state=20, criterion="gini", max_depth=2)
    
    print("\nCross Validation....")
    kf = KFold(n_splits=3 , shuffle=True)
    #kf.split(Xall)    
    # Initializez lista cu accuracy
    accuracy_model = []
 
    for train_index, test_index in kf.split(Xall):
        # Split train-test
        X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
        #y_train, y_test = Yall[train_index], Yall[test_index]
        y_train, y_test = Yall.iloc[train_index], Yall.iloc[test_index]
        # Train the model
        model = alg.fit(X_train, y_train)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    #accuracy pt cele K teste 
    print(accuracy_model)
    
def decision_tree_cross(Xall, Yall):   
    alg = DecisionTreeClassifier(random_state=20, criterion="gini", max_depth=10)
    
    print("\nCross Validation....")
    kf = KFold(n_splits=3 , shuffle=True)
    #kf.split(Xall)    
    # Initializez lista cu accuracy
    accuracy_model = []
 
    for train_index, test_index in kf.split(Xall):
        # Split train-test
        X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
        #y_train, y_test = Yall[train_index], Yall[test_index]
        y_train, y_test = Yall.iloc[train_index], Yall.iloc[test_index]
        # Train the model
        model = alg.fit(X_train, y_train)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    #accuracy pt cele K teste 
    print(accuracy_model)
    
def random_f(X_f1_train, y_f1_train, X_f1_test, y_f1_test):  
    pyplot.figure()
    model = RandomForestClassifier(n_estimators=100,
                                max_depth = 10,
                                bootstrap = True,
                                max_features = 'sqrt')
    ####multiclass
    # Fit on training data
    model.fit(X_f1_train, y_f1_train)
    print("\n\nPredictie cu Random Forest: ")
    # Actual class predictions
    rf_predictions = model.predict(X_f1_test)
    # Probabilities for each class
    rf_probs = model.predict_proba(X_f1_test)[:, 1]
    from sklearn.metrics import roc_auc_score

    print("\nAccuracy:", metrics.accuracy_score(y_f1_test, rf_predictions))
    matrice = confusion_matrix(y_f1_test, rf_predictions)
    print("\nMatricea de confuzie:")
    print(matrice)
    #pyplot.matshow(matrice)
    #pyplot.plot(matrice)
    disp = plot_confusion_matrix(model, X_f1_test, y_f1_test, values_format = 'd')
    #disp.ax_.set_title()
    pyplot.rcParams["axes.grid"] = False
    pyplot.ylabel("Real")
    pyplot.xlabel("Prezis")
    pyplot.title("Random Forest - multiclass")
    pyplot.show()
    return model
    
def random_f_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar):
    # #####binar
    pyplot.figure()
    model = RandomForestClassifier(n_estimators=100,
                                max_depth = 6,
                                bootstrap = True,
                                max_features = 'sqrt')
    # Fit on training data
    model.fit(X_f1_train_binar, y_f1_train_binar)
    print("\n\nPredictie cu Random Forest: ")
    # Actual class predictions
    rf_predictions = model.predict(X_f1_test_binar)
    
    # Probabilities for each class
    # rf_probs = model.predict_proba(X_f1_test_binar)[:, 1]
    # from sklearn.metrics import roc_auc_score
    # Calculate roc auc
    # roc_value = roc_auc_score(y_f1_test_binar, rf_probs)  #poate la binar
    print("\nAccuracy:", metrics.accuracy_score(y_f1_test_binar, rf_predictions))
    matrice = confusion_matrix(y_f1_test_binar, rf_predictions)
    print("\nMatricea de confuzie:")
    print(matrice)
    #pyplot.matshow(matrice)
    #pyplot.plot(matrice)
    disp = plot_confusion_matrix(model, X_f1_test_binar, y_f1_test_binar, values_format = 'd')
    #disp.ax_.set_title()
    pyplot.rcParams["axes.grid"] = False
    pyplot.ylabel("Real")
    pyplot.xlabel("Prezis")
    pyplot.title("Random Forest - binar")
    pyplot.show()
    return model

def random_f_binar_cross(Xall, Yall):   
    alg = RandomForestClassifier(n_estimators=100,
                                max_depth = 6,
                                bootstrap = True,
                                max_features = 'sqrt')
    
    print("\nCross Validation....")
    kf = KFold(n_splits=3 , shuffle=True)
    #kf.split(Xall)    
    # Initializez lista cu accuracy
    accuracy_model = []
 
    for train_index, test_index in kf.split(Xall):
        # Split train-test
        X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
        #y_train, y_test = Yall[train_index], Yall[test_index]
        y_train, y_test = Yall.iloc[train_index], Yall.iloc[test_index]
        # Train the model
        model = alg.fit(X_train, y_train)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    #accuracy pt cele K teste 
    print(accuracy_model)
    
def random_f_cross(Xall, Yall):   
    alg = RandomForestClassifier(n_estimators=100,
                                max_depth = 10,
                                bootstrap = True,
                                max_features = 'sqrt')
    
    print("\nCross Validation....")
    kf = KFold(n_splits=3 , shuffle=True)
    #kf.split(Xall)    
    # Initializez lista cu accuracy
    accuracy_model = []
 
    for train_index, test_index in kf.split(Xall):
        # Split train-test
        X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
        #y_train, y_test = Yall[train_index], Yall[test_index]
        y_train, y_test = Yall.iloc[train_index], Yall.iloc[test_index]
        # Train the model
        model = alg.fit(X_train, y_train)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    #accuracy pt cele K teste 
    print(accuracy_model)
    
def SVM(X_f1_train, y_f1_train, X_f1_test, y_f1_test):
    #SVM=dureaza o vesnicie
    #multiclass
    pyplot.figure()
    model = SVC(kernel='poly', gamma='auto')
    model.fit(X_f1_train, y_f1_train)
    print("\nPredictie cu SVM: ")
    predictions = model.predict(X_f1_test)
    # Evaluate predictions
    print("Accuracy: ")
    print(accuracy_score(y_f1_test, predictions))
    matrice = confusion_matrix(y_f1_test, predictions)
    print("\nMatricea de confuzie:")
    print(matrice)
    disp = plot_confusion_matrix(model, X_f1_test, y_f1_test, values_format = 'd')
    pyplot.rcParams["axes.grid"] = False
    pyplot.ylabel("Real")
    pyplot.xlabel("Prezis")
    pyplot.title("SVM - multiclass")
    pyplot.show()
    return model
    
def SVM_binar(X_f1_train_binar, y_f1_train_binar, X_f1_test_binar, y_f1_test_binar):
    #Binar
    model = SVC(kernel='poly', gamma='auto')
    model.fit(X_f1_train_binar, y_f1_train_binar)
    print("\nPredictie cu SVM: ")
    predictions = model.predict(X_f1_test_binar)
    # Evaluate predictions
    print("Accuracy: ")
    print(accuracy_score(y_f1_test_binar, predictions))
    print("\nMatricea de confuzie:")
    matrice = confusion_matrix(y_f1_test_binar, predictions)
    print(matrice)
    disp = plot_confusion_matrix(model, X_f1_test_binar, y_f1_test_binar, values_format = 'd')
    pyplot.rcParams["axes.grid"] = False
    pyplot.ylabel("Real")
    pyplot.xlabel("Prezis")
    pyplot.title("SVM - binar")
    pyplot.show()
    return model

def SVM_binar_cross(Xall, Yall):   
    alg = SVC(kernel='poly', gamma='auto')
    
    print("\nCross Validation....")
    kf = KFold(n_splits=3 , shuffle=True)
    #kf.split(Xall)    
    # Initializez lista cu accuracy
    accuracy_model = []
 
    for train_index, test_index in kf.split(Xall):
        # Split train-test
        X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
        #y_train, y_test = Yall[train_index], Yall[test_index]
        y_train, y_test = Yall.iloc[train_index], Yall.iloc[test_index]
        # Train the model
        model = alg.fit(X_train, y_train)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    #accuracy pt cele K teste 
    print(accuracy_model)
    
def SVM_cross(Xall, Yall):   
    alg = SVC(kernel='poly', gamma='auto')
    
    print("\nCross Validation....")
    kf = KFold(n_splits=3 , shuffle=True)
    #kf.split(Xall)    
    # Initializez lista cu accuracy
    accuracy_model = []
 
    for train_index, test_index in kf.split(Xall):
        # Split train-test
        X_train, X_test = Xall.iloc[train_index], Xall.iloc[test_index]
        #y_train, y_test = Yall[train_index], Yall[test_index]
        y_train, y_test = Yall.iloc[train_index], Yall.iloc[test_index]
        # Train the model
        model = alg.fit(X_train, y_train)
        # Append to accuracy_model the accuracy of the model
        accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

    #accuracy pt cele K teste 
    print(accuracy_model)