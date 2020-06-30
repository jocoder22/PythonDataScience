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

# Define market variables
beta_xle = 1.07
beta_xli = 1.06
beta_bmk = 1.0
r_f = 0.0225
r_m = 0.09
std_market = 0.15

# compute CAPM expected returns
expected_ret_xle = r_f + beta_xle*(r_m - r_f)
expected_ret_xli = r_f + beta_xli*(r_m - r_f)
expected_ret_bmak = r_f + beta_bmk*(r_m - r_f)

expected_ret = np.array([expected_ret_xle, expected_ret_xli])

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

# compute s&p500 retruns
sp500_com = (1 + sp500_ret).cumprod() 
final_sp500_return = 1 - sp500_com.iloc[-1,:]

# compute s&p500 retruns annualized volatility
annul_sp500_vol = sp500_ret.std() * np.sqrt(252)

print2(sp500_com.tail(), final_sp500_return, sp500_com.iloc[-1,:][0])


def compute_portfolio_return(returns, weights):  
    return returns.dot(weights)

def compute_portfolio_vol(data, weight):    
    assets_return = data.pct_change().dropna()
    covariance = assets_return.cov()
    portfolio_val = np.transpose(weight) @ covariance @ weight
    annualized_volatility = np.sqrt(portfolio_val) * np.sqrt(252)
    
    return annualized_volatility
             
def compute_bmk_returnsVol(data):
    # compute benchmark returns using historic data
    bmk_returns = data.pct_change().dropna()
    sp500_com_ = (1 + bmk_returns).cumprod() 
    final_return = 1 - sp500_com_.iloc[-1,:][0]

    # calculate benchmark returns annualised volatility
    bmk_Vol = bmk_returns.std()[0]*np.sqrt(252)  

    return  final_return, bmk_Vol    


def compute_sharpe_ratio(returns, vol, riskfree_rate):
    return (returns - riskfree_rate) / vol

hist_return, vol = compute_bmk_returnsVol(sp500)

print2(expected_ret_bmak, hist_return, vol)


xle_weights2 = []
xli_weights2 = []
returnlist2 = []
vollist2 = []

for i in range(len(weight_XLI)):
    xle_weights2.append(weight_XLE[i])
    xli_weights2.append(weight_XLI[i])
    weights = [weight_XLE[i], weight_XLI[i]]

    ret = compute_portfolio_return(expected_ret, weights)
    vol = compute_portfolio_vol(df, weights)

    returnlist2.append(ret)
    vollist2.append(vol)

data2 = pd.DataFrame({'xle_weight':xle_weights2,
                       'xli_weight':xli_weights2,
                       'expected_return':returnlist2,
                       'volatility':vollist2})

print2(data2)