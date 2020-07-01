# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def curatare(dataset):
    ##################################curatare date
    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df trebuie sa fie pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep]      #.astype(np.float64)
    
    dataset_clean = clean_dataset(dataset)
    #se convertea aiurea
    dataset_clean['Flow Bytes/s'] = dataset_clean['Flow Bytes/s'].astype(np.float64)
    #print(dataset_clean['Flow Bytes/s'].dtype)
    
    ############################pastrare doar a datelor pozitive
    #dataset_clean_poz = dataset_clean[(dataset_clean[[' Flow Duration', 'Flow Bytes/s', ' Flow IAT Min', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward']] >= 0).all(axis=1)]
    
    dataset_clean_poz = dataset_clean[(dataset_clean[[' Flow Duration', 'Flow Bytes/s', ' Flow IAT Min']] >= 0).all(axis=1)]
    dataset_clean_poz = dataset_clean.replace({'Init_Win_bytes_forward':{-1: 0}})
    dataset_clean_poz = dataset_clean.replace({' Init_Win_bytes_backward':{-1: 0}})
    
    print("\n------------------Distributie date curatate dupa ' Label':")
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
    return dataset_clean_poz

def curatare_test(dataset):
    ##################################curatare date
    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df trebuie sa fie pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep]      #.astype(np.float64)
    
    dataset_clean = clean_dataset(dataset)
    #coloana asta nush de ce era de tip Object asa ca am convertit-o in float64
    dataset_clean['Flow Byts/s'] = dataset_clean['Flow Byts/s'].astype(np.float64)
    #print(dataset_clean['Flow Bytes/s'].dtype)
    
    ############################pastrare doar a datelor pozitive
    dataset_clean_poz = dataset_clean[(dataset_clean[['Flow Duration', 'Flow Byts/s', 'Flow IAT Min']] >= 0).all(axis=1)]
    dataset_clean_poz = dataset_clean.replace({'Init Fwd Win Byts':{-1: 0}})
    dataset_clean_poz = dataset_clean.replace({'Init Bwd Win Byts':{-1: 0}})
    print("\n------------------Distributie date curatate dupa 'Label':")
    print(dataset_clean_poz.groupby('Label').size()) 
    
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
    return dataset_clean_poz