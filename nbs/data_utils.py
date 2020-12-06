from pathlib import Path
from google.cloud import firestore
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import pandas as pd
import numpy as np
import math

def loadData(from_csv=False):
    file_name = "../data/main/BOVA_2020.pickle"
    my_file = Path(file_name)
    if from_csv == False and my_file.is_file():
        return pd.read_pickle(file_name)
    else:
        data = pd.read_csv("../data/COTAHIST_A2020.csv")
        stocks = ["BOVA11"]
        stocksDf = data.loc[data["CODNEG"].isin(stocks)] 
        
        ## Clear unused large dataset from memory
        del data
        
        stocksDf.set_index("DATPRG", inplace=True)
        stocksDf.index = pd.to_datetime(stocksDf.index)
        stocksDf.sort_index(inplace=True)
        #Dropping useless columns
        stocksDf.drop(columns=['TIPREG',
                       'CODBDI',
                       'TPMERC',
                       'NOMRES',
                       'ESPECI',
                       'PRAZOT',
                       'MODREF',
                       'PREEXE',
                       'INDOPC',
                       'DATVEN',
                       'FATCOT',
                       'PTOEXE',
                       'CODISI',
                       'TOTNEG',
                       'QUATOT',
                       'VOLTOT',
                       'DISMES',
                       'CODNEG'], inplace=True)
        stocksDf.to_pickle(file_name)
        return stocksDf    

def getTrainTestSets(df, perc=None, start_date='2020-05-01', end_date='2020-11-30', with_emotions=False, normalize=False):     
#     FEATURES = ['PREABE','PREMAX','PREMIN','PREMED','PREOFC','PREOFV','TOTNEG','QUATOT','VOLTOT']
    scaler = MinMaxScaler()
    if 'CODNEG' in df:
        df.drop(columns=['CODNEG'], inplace=True)
    if normalize == True:        
        df[['PREABE','PREMAX','PREMIN','PREMED','PREOFC','PREOFV', 'PREULT']] = scaler.fit_transform(df[['PREABE','PREMAX','PREMIN','PREMED','PREOFC','PREOFV', 'PREULT']])
    FEATURES = ['PREABE','PREMAX','PREMIN','PREMED','PREOFC','PREOFV', 'PREULT']
    if with_emotions:
        FEATURES.extend(['SUR','ANG','JOY','FEA','TRU','ANT','SAD','DIS'])
    Y = ['PREULT']
    df = df[start_date:end_date]
    cut = math.ceil(len(df) * perc)
    train_df_X = df[:cut]
    y_train_start = train_df_X.first_valid_index() + timedelta(days=1)
    y_train_end = train_df_X.last_valid_index() + timedelta(days=1)    
    train_df_y = df[y_train_start:y_train_end]  

    test_end = df.last_valid_index() - timedelta(days=1)    
    test_df_X = df[cut:-1]
    y_test_start = test_df_X.first_valid_index() + timedelta(days=1)
    y_test_end = end_date
    test_df_y = df[y_test_start: y_test_end]

    X_train, y_train = train_df_X[FEATURES], np.array(train_df_y[Y]).flatten()
    X_val, y_val = test_df_X[FEATURES], np.array(test_df_y[Y]).flatten()
    return X_train, y_train, X_val, y_val, scaler

def loadEmoData():
    file_name = "../data/main/B3_EMO.pickle"
    my_file = Path(file_name)
    if my_file.is_file():
        return pd.read_pickle(file_name)
    else:
        DB = firestore.Client.from_service_account_json("/Users/marcelcanhisares/Development/tcc/keys/mc-tcc1-2893283b8dce.json")
        collection = DB.collection('daily_emotions').stream()
        results = []
        for document in collection:
            results.append(document.to_dict())
        df = pd.DataFrame(data=results)
        df.set_index('date', inplace=True)
        df.to_pickle(file_name)
        return df