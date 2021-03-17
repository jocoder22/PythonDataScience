# mean-variance portfolio theorem
#!/usr/bin/env python
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import date, datetime

# startdate = datetime(2015, 1, 1)
startdate = datetime(2018, 9, 12)
enddate = date.today()

folders = ['Apple', 'MicroSoft', 'Google', 'Netflix', 'Tesla', 'Amazon', 'Toyota', 'JPMorgan', 
            'Citigroup', 'Walmat', 'Target', "Fedex", "Ups", "Walgreens", "Disney", "Pfizer",
            "Cvs", "AT_T", "CocaCola", "Boeing", "SolarEdge", "AdvancedMicroDevices", "Twilio",
            "ExpWorld", "HomeDepot", "Ford", "PVH", "Twitter", "Salesforce", 
            "Alibaba", "NioElectricMotor", "Apple" ,  "BristolMyers"]

symbols = ['AAPL', 'MSFT', 'GOOGL', 'NFLX', 'TSLA', 'AMZN', 'TM', 'JPM', 'C', 'WMT', 'TGT', 'FDX',
            'UPS', 'WBA', 'DIS', 'PFE', 'CVS', 'T', 'KO', 'BA', 'SEDG', 'AMD', 'TWLO', 'EXPI', 'HD',
              'F', 'PVH', 'TWTR', 'CRM', 'BABA', 'NIO', 'AAPL', 'BMY']

coldict = {symbols[i]:folders[i] for i in range(len(symbols))}

df = pdr.get_data_yahoo(symbols, startdate, enddate)['Adj Close']
portfolio = df.copy()
portfolio.head()
portfolio.info()

assets_columns = ['Returns', 'Volatility']
assets = pd.DataFrame(columns = assets_columns)


portfolio_volatility = [] 
portfolio_weights = [] 

number_assets = len(symbols)
combination_weights = 1000

# calculate yearly return for each stock
stock_returns = portfolio.resample('Y').last().pct_change().mean()

# calculate annaulized or yearly volatility for each stock
# use use np.sqrt(250)
stock_annualized_vol = np.log(portfolio/portfolio.shift()).std().apply(lambda x: x*np.sqrt(250))

# compute the portfolio covariance
portfolio_cov_matrix = np.log(portfolio/portfolio.shift()).cov()
