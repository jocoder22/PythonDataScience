# import required modules
#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from printdescribe import print2, describe2, changepath
import tabulate

# import excel sheets
path = r"D:\Wqu_FinEngr\Portfolio Theory and Asset Pricing\GroupWork"

with changepath(path):
    data = pd.read_excel("GWP_PTAP_Data_2010.10.08.xlsx", skiprows=1, sheet_name=[0,1,2])

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

# combine the CAPM expected returns in np array
expected_ret = np.array([expected_ret_xle, expected_ret_xli])

# combine the excel sheets
frames = [data[i].rename(columns={data[i].columns[1]:names[i]}).set_index("Date") for i in range(3)]
df2 = pd.concat(frames, axis=1, sort=False)

# subset for XLE, XLI
df = df2[["XLE", "XLI"]]

# assign the weights
weight_XLE  = [round(i,1) for i in np.linspace(0,1, 11)]
weight_XLI = [round(i,1) for i in  np.linspace(1,0, 11)]

# initiate empty list containers
returnlist = []
vollist = []

def portfolioreturnVol(data, weight):
    """The portfolioreturnVol function computes the portfolio expected returns
        and volatility.
        
        Inputs:
            data(dataframe): Historic assets close prices
            weights(float) : weights of assets in the portofolio
            
        Outputs:
            final_return: portfolio expected return on the last day
            _annualised_vol: Annualised portfolio volatility
    
    """
    # compute simple assets returns
    assets_return = data.pct_change().dropna()
    
    # compute portfolio returns
    portreturn = assets_return.dot(weight)
    
    # compute portfolio cumulative returns
    # extract the last day portfolio returns
    port_com = (1 + portreturn).cumprod() 
    final_return = 1 - port_com[-1]
    
    #  annu_ = assets_return.cov() * np.sqrt(252)
    # compute portfolio annualised volatility
    covariance = assets_return.cov()
    port_val = np.transpose(weight) @ covariance @ weight
    _annualised_vol = np.sqrt(port_val) * np.sqrt(252)
    
    return final_return, _annualised_vol

# loop through the weight combination
# calculate the portfolio expected returns and volatility
for i in range(len(weight_XLI)):
    weight = [weight_XLE[i], weight_XLI[i]]
    rt, vol = portfolioreturnVol(df, weight) 
    returnlist.append(rt)
    vollist.append(vol)
    
print2(returnlist, vollist)

# plot the efficient frontier
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
    """The compute_portfolio_return computes porfolio returns using
        CAPM expected return
        
        Inputs:
            returns(float): CAPM expected assets returns
            weights(float): assets weights in the portfolio
            
         Output:
            portfolio_returns: the portfolio returns
    
    """
    # compute portfolio returns
    portfolio_returns = returns.dot(weights)
    
    return portfolio_returns 

def compute_portfolio_vol(data, weight):
    """The compute_portfolio_vol function computes annaulized portfolio volatilty.
    
    Inputs:
        data(float): assets close prices
        weights(float): assets weights in the portfolio
        
    Output:
        annualized_volatility: the annualized portfolio volatility
    
    """   

    # compute assets returns 
    assets_return = data.pct_change().dropna()

    # computes assets covariance matrix
    covariance = assets_return.cov()

    # computes portfolio volatility
    portfolio_val = np.transpose(weight) @ covariance @ weight

    # computes annualized volatility
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
    """The compute_sharpe_ratio function computes sharpe ratio.
    
    Inputs:
        returns (float): portfolio returns
        volatility(float): portfolio volatililty
        riskfree_rate (float): the risk free rate
    
    Output:
        sharperatio: the calculated Sharpe ratio
    
    """   

    sharperatio = (returns - riskfree_rate) / vol
    
    return sharperatio


hist_return, vol_bmk = compute_bmk_returnsVol(sp500)

print2(expected_ret_bmak, hist_return, vol_bmk)


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

plt.figure(figsize=[10,8])
plt.scatter(vollist2, returnlist2)
plt.show()

data2 = pd.DataFrame({'xle_weight':xle_weights2,
                       'xli_weight':xli_weights2,
                       'expected_return':returnlist2,
                       'volatility':vollist2})

print2(data2)

selected_port = data2.loc[(data2['expected_return'] > 0.0943) & (data2['volatility'] < 0.168)]

# selected_port.reset_index(inplace=True)
print(tabulate.tabulate(selected_port, headers=selected_port.columns, tablefmt="fancy_grid", showindex="never"))

# compute sharpe ratio
port_sharpe_ratio = compute_sharpe_ratio(selected_port['expected_return'][4],selected_port['volatility'][4], r_f)
bmk_sharpe_ratio = compute_sharpe_ratio(expected_ret_bmak, vol_bmk, r_f)
data_sharpe = {"Porfolio Sharpe Ratio":[port_sharpe_ratio], "Benchmark Sharpe Ratio":[bmk_sharpe_ratio]}
print(tabulate.tabulate(data_sharpe, headers=data_sharpe.keys(), tablefmt="fancy_grid", showindex="never"))
print2(port_sharpe_ratio, bmk_sharpe_ratio)
describe2(df2)
print2(df2.shape)


def xprint(d):
    for arg in (d.head(), d.info(), d.shape, d. columns):
        print(arg, end='\n\n')

xprint(df2)

xx = df2.info()
print2(xx)
print2(pd.DataFrame(xx), xx)
