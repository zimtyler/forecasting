import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

def get_df():
    dir = os.getcwd()
    dfs = []
    for csv in os.listdir(dir):
        if csv.endswith('.csv'):
            df = pd.read_csv(csv)
            df['DATE'] = pd.to_datetime(df['DATE'])
            df.set_index('DATE', inplace=True, drop=True)
            dfs.append(df)
    
    return pd.concat(dfs, axis=1).dropna()

def get_best_ma_and_lags(df, ma=None):

    if ma == None:
        ma = [3, 6, 9, 12, 15, 18]
    lags = {}
    df = df.copy()
    for col in df.columns:
        lags[col] = []
        for i in range(132): # 11 years
            for s in ma:
                df_ = df[[col, 'VehicleSales']].copy()
                if col == 'M12MTVUSM227SFWA':
                    s = 0
                else:
                    df_[col] = df_[col].rolling(window=s).mean()
                if col != 'VehicleSales':
                    df_[col] = df_[col].shift(i)
                    df_.dropna(inplace=True)
                    cor = df_[[col, 'VehicleSales']].corr()
                    lags[col].append((i, s, np.array(cor).flatten()[1]))
                    
    best_lags = {}
    for k in lags.keys():
        l = sorted(lags[k], key=lambda x: abs(x[2]), reverse=True)[:1]
        if len(l) != 0:
            best_lags[k] = (l[0][0], l[0][1], l[0][2])
    
    return best_lags

def average_and_shift_data(df, best_lags):
    df = df.copy()
    for col in df.columns:
        if col != 'VehicleSales':
            window = best_lags[col][1]
            shift = best_lags[col][0]
            if window == 0:
                df[col] = df[col].shift(shift)
            else:
                df[col] = df[col].rolling(window=window).mean().shift(shift)
    
    return df

def return_x_y(df):
    y = np.array(df.iloc[:, -1]).reshape(-1, 1)
    X = df.iloc[:, :-1]
    xscaler = StandardScaler().fit(X)
    # yscaler = StandardScaler().fit(y)
    X_scaled = xscaler.transform(X)
    # y_scaled = yscaler.transform(y)
    return X_scaled, y, xscaler