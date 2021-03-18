# mean-variance portfolio theorem
#!/usr/bin/env python
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import date, datetime
import matplotlib.pyplot as plt

# startdate = datetime(2015, 1, 1)
startdate = datetime(2018, 9, 12)
enddate = date.today()

folders = ['Apple', 'MicroSoft', 'Google', 'Netflix', 'Tesla', 'Amazon', 'Toyota', 'JPMorgan', 
            'Citigroup', 'Walmat', 'Target', "Fedex", "Ups", "Walgreens", "Disney", "Pfizer",
            "Cvs", "AT_T", "CocaCola", "Boeing", "SolarEdge", "AdvancedMicroDevices", "Twilio",
            "ExpWorld", "HomeDepot", "Ford", "PVH", "Twitter", "Salesforce", 
            "Alibaba", "NioElectricMotor", "BristolMyers"]

symbols = ['AAPL', 'MSFT', 'GOOGL', 'NFLX', 'TSLA', 'AMZN', 'TM', 'JPM', 'C', 'WMT', 'TGT', 'FDX',
            'UPS', 'WBA', 'DIS', 'PFE', 'CVS', 'T', 'KO', 'BA', 'SEDG', 'AMD', 'TWLO', 'EXPI', 'HD',
              'F', 'PVH', 'TWTR', 'CRM', 'BABA', 'NIO', 'BMY']

coldict = {symbols[i]:folders[i] for i in range(len(symbols))}

df = pdr.get_data_yahoo(symbols, startdate, enddate)['Adj Close']
portfolio = df.copy()
portfolio.head()
portfolio.info()


# calculate yearly return for each stock
stock_returns = portfolio.resample('Y').last().pct_change().mean()

# calculate annaulized or yearly volatility for each stock
# use use np.sqrt(250)
stock_annualized_vol = np.log(portfolio/portfolio.shift()).std().apply(lambda x: x*np.sqrt(250))

# compute the portfolio covariance
portfolio_cov_matrix = np.log(portfolio/portfolio.shift()).cov()
portfolio_cov_matrix.iloc[:6, :6]

# create a dataframe for returns(yearly) and Volatility(annualized) for each stock
assets_columns = ['Returns', 'Volatility']
assets = pd.DataFrame({assets_columns[0] : stock_returns, assets_columns[1]: stock_annualized_vol})
assets.iloc[:6, :6]



portfolio_returns = []
portfolio_volatility = [] 
portfolio_weights = [] 

number_assets = len(portfolio.columms)
combination_weights = 50000

for portt in range(combination_weights):
    weights_listR = np.random.random(number_assets)
    weights_listN = weights/np.sum(weights)
    portfolio_weights.append(weights_listN)
    returns = np.dot(weights_listN, stock_returns) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    portfolio_returns.append(returns)
    var = portfolio_cov_matrix.mul(weights_listN, axis=0).mul(weights_listN, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
    portfolio_volatility.append(ann_sd)


dict_ = {'Returns':portfolio_returns, 'Volatility':portfolio_volatility}

for i, ticker in enumerate(portfolio.columns.tolist()):
    dict_[ticker+'_weight'] = [wt[i] for wt in portfolio_weights]

Eff_portfolios  = pd.DataFrame(dict_)
Eff_portfolios.head()

Eff_portfolios.plot.scatter(x='Volatility', y='Returns', alpha=0.3, grid=True, figsize=[14,12])
plt.show()


# get the max return portfolio
max_ret_port = Eff_portfolios.iloc[Eff_portfolios['Returns'].idxmax()]

# get the min volatility portfolio
min_vol_port = Eff_portfolios.iloc[Eff_portfolios['Volatility'].idxmin()]


# diplay min volatilit and max return portfolios
plt.scatter(Eff_portfolios['Volatility'], Eff_portfolios['Returns'], figsize=[14,12])
plt.scatter(min_vol_port[1], min_vol_port[0], color='b', marker='*', s=300)
plt.scatter(max_ret_port[1], max_ret_port[0], color='b', marker='*', s=500)
