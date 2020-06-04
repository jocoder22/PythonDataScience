import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
import statsmodels.api as sm
from pandas.util.testing import assert_frame_equal

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

stocklist = ["C","JPM","MS", "GS"]
p_labels = ["Citibank", "J.P. Morgan", "Morgan Stanley", "Goldman Sachs"]

starttime = datetime.datetime(2000, 1, 1)
endtime = datetime.datetime(2019, 10, 1)

# get only the closing prices
portfolio = pdr.get_data_yahoo(stocklist, starttime, endtime)['Close']

# set the weights
weights = [0.25, 0.25, 0.25, 0.25]

# calculate percentage return and portfolio return
asset_returns = portfolio.pct_change()
returns = asset_returns.dot(weights)

# path = r"D:\PythonDataScience\risk_mgt\DRSFRMACBS.csv"
weblink = ("https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type="
        "line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars="
        "on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend="
        "yes&show_axis_titles=yes&show_tooltip=yes&id=DRSFRMACBS&scale=left&cosd=2000-01-01&coed="
        "2019-10-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw="
        "3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Quarterly%2C%20End%20of%20Period&fam="
        "avg&fgst=lin&fgsnd=2009-06-01&line_index=1&transformation=lin&vintage_date="
        "2020-05-29&revision_date=2020-05-29&nd=1991-01-01")

# df = pd.read_csv(weblink, parse_dates=True, index_col=0, names=["Mortage Deliquency Rate"])
mdr = pd.read_csv(weblink).set_index("DATE")
mdr.columns = ["Mortage Deliquency Rate"]
print2(mdr.loc["2005-03-31":], returns)

# Convert daily returns to quarterly average returns
returns_qmean = returns.resample('Q').mean().dropna()

print2(mdr.shape, returns_qmean.shape, returns_qmean.head(), mdr.head())

# Examine the beginning of the quarterly series
print2(returns_qmean.head())

# Now convert daily returns to weekly minimum returns
returns_w = returns.resample('W').min().dropna()

# Examine the beginning of the weekly series
print2(returns_w.head())

# Create a scatterplot between quarterly average returns and delinquency
plt.scatter(returns_qmean, mdr)
plt.axis([-0.007,0.006,0,14]) 
plt.show()

# Convert daily returns to quarterly minimium returns
returns_qmin = returns.resample('Q').min().dropna()

# Create a scatterplot between quarterly minimum returns and delinquency
plt.scatter(returns_qmin, mdr)
plt.axis([-0.125,0.006,0,14]) 
plt.xlabel("Quarterly Average Return")
plt.ylabel("Mortage Deliquemcy Rate (Percent)")
plt.show()

# Create a scatterplot between quarterly minimum returns and delinquency
plt.scatter(returns_qmin.loc['2005-01-01':'2010-12-31'], mdr.loc['2005-01-01':'2010-12-31'])
plt.axis([-0.125,0.006,0,14]) 
plt.xlabel("Quarterly Minimum Return")
plt.ylabel("Mortage Deliquemcy Rate (Percent)")
plt.show()

# Add a constant to the regression
Y = returns_qmean.values
X = mdr.values
X = sm.add_constant(mdr.values)

# Create the regression factor model and fit it to the data
results = sm.OLS(Y, X).fit()

# Print a summary of the results
print2(results.summary())

# Add a constant to the regression
Y = returns_qmin.values
X = mdr.values
X = sm.add_constant(mdr.values)

# Create the regression factor model and fit it to the data
results = sm.OLS(Y, X).fit()

# Print a summary of the results
print2(results.summary())

# using only periods of global recession
returns_qmean = returns_qmean.loc['2005-01-01':'2010-12-31']
returns_qmin = returns_qmin.loc['2005-01-01':'2010-12-31']
mdr = mdr.loc['2005-01-01':'2010-12-31']

# Add a constant to the regression
Y = returns_qmean.values
X = mdr.values
X = sm.add_constant(mdr.values)

# Create the regression factor model and fit it to the data
results = sm.OLS(Y, X).fit()



# Print a summary of the results
print2(results.summary())

# Add a constant to the regression
Y = returns_qmin.values
X = mdr.values
X = sm.add_constant(mdr.values)

# Create the regression factor model and fit it to the data
results = sm.OLS(Y, X).fit()



# Print a summary of the results
print2(results.summary())
