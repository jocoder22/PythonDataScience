# import required modules
#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tabulate

def print2(*args):
    for obj in args:
        print(obj, end="\n\n")

def describe2(x):
    print2(x.head(), x.shape, x.info())


# import excel sheets
path = "D:\Wqu_FinEngr\Portfolio Theory and Asset Pricing\GroupWork\GWP_PTAP_Data_2010.10.08.xlsx"
data = pd.read_excel(path, skiprows=1, sheet_name=[0,1,2])

# labels for data
names = ["XLE", "XLI", "S&P500"]

# combine the excel sheets
frames = [data[i].rename(columns={data[i].columns[1]:names[i]}).set_index("Date") for i in range(3)]
df2 = pd.concat(frames, axis=1, sort=False)

df = df2[["XLE", "XLI"]]


weight_XLE  = [round(i,1) for i in np.linspace(0,1,11)]