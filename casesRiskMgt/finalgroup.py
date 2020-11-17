import numpy as np
import pandas as pd
import pandas_datareader.wb as wb
import matplotlib.pyplot as plt

from printdescribe import print2, changepath

pth = r"D:\Wqu_FinEngr\Case_Studies_Risk_Mgt\GroupWork"

with changepath(pth):
    data_r = pd.read_excel("greece_quarterly_30Y_reduced_20201102.xlsx", sheet_name="Reduced")

print2(data_r.head())

data_r2 = data_r.iloc[2:,:].set_index("Name")
print2(data_r2.head(), data_r2.info())

data = data_r2.iloc[:, [0,10,12,27,9]]
