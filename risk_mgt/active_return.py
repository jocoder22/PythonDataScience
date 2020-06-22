#!/usr/bin/env python
# Import required modules for this CRT

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
# from statistics import stdev 


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")


# I'm using daily close prices
# SPDR S$p500 ETF (SPY)
SPY = "SPY"

# My two other ETFs are
# 1. Vanguard S$P500 ETF (VOO)
# 2. iShares Core S&P500 ETF (IVV)

etfs_tickers = ["^GSPC", "SPY", "VOO", "IVV"]

# using 2 years of data from January 01, 2018 to December 31, 2019
starttime = datetime.datetime(2018, 1, 1)
endtime = datetime.datetime(2019, 12, 31)

# get only the closing prices
etfs = pdr.get_data_yahoo(etfs_tickers, starttime, endtime)['Close']
etfs.columns = ["S&P500", "SPDR", "Vanguard", "iShares"]

print2(etfs.head())



etfs_return = etfs.pct_change().dropna()

etfs_return.fillna(0, inplace=True)
returns2 = round(etfs_return*100, 3)

print2(etfs_return, returns2)


eft_index = etfs_return["S&P500"]
etfs_activeR = etfs_return.sub([eft_index, eft_index,eft_index,eft_index], axis='columns')
etfs_activeR.drop("S&P500", axis=1, inplace=True)
etfs_activeR.columns = ["Active_SPDR", "Active_Vanguard", "Active_iShares"]


r_index = returns2["S&P500"]
r_activeR = returns2.sub([r_index, r_index,r_index, r_index], axis='columns')
r_activeR.drop("S&P500", axis=1, inplace=True)
r_activeR.columns = ["Active_SPDR", "Active_Vanguard", "Active_iShares"]


print2(etfs_activeR, r_activeR)

mean_return = etfs_return.mean()
_return = returns2.mean()


average_return = mean_return - etfs_return["S&P500"].mean()
average_return

cum_return = (1+etfs_return).cumprod() - 1


plt.figure(figsize=[14,6])
plt.plot(cum_return)
# cum_return.plot()
plt.legend(cum_return.columns)
plt.show()

tracking_e = etfs_activeR.std()
tracking_e_ = r_activeR.std()

print2(tracking_e, tracking_e_)


r_rbarSquared = (etfs_activeR - etfs_activeR.mean()) ** 2
ave_return = np.sqrt(r_rbarSquared.sum()/(etfs.shape[0] - 1))


m_rbarSquared = etfs_activeR ** 2
madj_return = np.sqrt(m_rbarSquared.sum()/etfs.shape[0])
print2(ave_return, madj_return)



tickers = ["^GSPC","XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

# get only the closing prices
spdr_funds = pdr.get_data_yahoo(tickers, starttime, endtime)['Close']
spdr_funds.columns = ["S&P500","XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
spdr_funds.head()

spdr_funds_R = spdr_funds.pct_change()
spdr_funds_R.fillna(0, inplace=True)
print2(spdr_funds_R.head())

cum_spdr = (1+spdr_funds_R).cumprod()
plt.figure(figsize=[14,6])
plt.plot(cum_spdr)
plt.legend(cum_spdr.columns)
plt.show()





spdr_index = spdr_funds_R["S&P500"]
spdr_funds = pd.DataFrame()

for col in spdr_funds_R.columns:
    spdr_f = spdr_funds_R[[col]].sub([spdr_index], axis='columns')
    spdr_funds = pd.concat([spdr_funds, spdr_f], axis=1, sort=True)

spdr_funds.drop("S&P500", axis=1, inplace=True)

print2(spdr_funds)
spdr_funds_ave = spdr_funds.mean()
spdr_funds_tracing = spdr_funds.std()
madj_tracing = np.sqrt((spdr_funds**2).sum() / spdr_funds.shape[0])
print2(spdr_funds_ave, spdr_funds_tracing, madj_tracing)



def activeReturn(funds, ref_index):
    """The activeReturn computes the active return
    
    Inputs:
        funds: selected funds
        ref_index:  the reference index    
    
    
    """
    
#     _index = funds[ref_index]
#     _activeR = funds.sub(funds.columns, axis='columns')
#     _activeR.drop(funds, axis=1, inplace=True)
#     _activeR.columns = [f"Active_{i}" for i in _activeR.columns]

    _activeR = funds.sub([ref_index], axis='columns')
    _activeR.columns = [f"Active_{funds.columns.tolist()[0]}"]
    
    return _activeR


active_mm = activeReturn(spdr_funds_R[["XLU"]], spdr_funds_R["S&P500"])
active_mm.head()