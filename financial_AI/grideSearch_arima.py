import sys
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from datetime import datetime
# import pyodbc
from printdescribe import changepath

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pmdarima.arima import auto_arima


path1 = r"E:\Capstone\capstoneRepo"

with changepath(path1):
    df = pd.read_csv("finally.csv")

print(df.head())

df_train = df[:int(len(df)*0.8)]
df_test = df[int(len(df)*0.8):]

print(df_train.shape, df_test.shape)


model_search = auto_arima(df_train[["SPY_Last"]], 
                         X= df_train[["VIX_Last"]],
                         start_p=1, start_q=1, 
                         max_p=3, max_q=3, 
                          d=1, max_d=3,
                         trace=True, 
                         error_action='ignore', 
                         suppress_warnings=True, 
                         stepwise=True)

print(model_search.summary())