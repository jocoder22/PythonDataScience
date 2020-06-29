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


weight_XLE  = [round(i,1) for i in np.linspace(0,1, 11)]
weight_XLI = [round(i,1) for i in  np.linspace(1,0, 11)]

returnlist = []
vollist = []

def portfolioreturnVol(data, weight):
    assets_return = data.pct_change().dropna()
    portreturn = assets_return.dot(weight)
    port_com = (1 + portreturn).cumprod() 
    final_return = 1 - port_com[-1]
    
    #  annu_ = assets_return.cov() * np.sqrt(252)
    covariance = assets_return.cov()
    port_val = np.transpose(weight) @ covariance @ weight
    _ann_vol = np.sqrt(port_val) * np.sqrt(252)
    
    return final_return, _ann_vol
    
for i in range(len(weight_XLI)):
    weight = [weight_XLE[i], weight_XLI[i]]
    rt, vol = portfolioreturnVol(df, weight) 
    returnlist.append(rt)
    vollist.append(vol)
    
print2(returnlist, vollist)

plt.figure(figsize=[10,8])
plt.plot(vollist, returnlist)
plt.show()


# compute returns and volatility for S&P500

sp500 = df2[["S&P500"]]
sp500_ret = sp500.pct_change().dropna()

sp500_com = (1 + sp500_ret).cumprod() 
final_sp500_return = 1 - sp500_com[-1]

annul_sp500_vol = sp500_ret.std() * np.sqrt(252)
             
             
