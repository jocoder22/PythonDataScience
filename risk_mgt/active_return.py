#!/usr/bin/env python
# Import required modules for this CRT

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
# from statistics import stdev 

from printdescribe import print2, describe2, changepath

# def print2(*args):
#     for arg in args:
#         print(arg, end="\n\n")


# I'm using daily close prices
# SPDR S$p500 ETF (SPY)
SPY = "SPY"

# My two other ETFs are
# 1. Vanguard S$P500 ETF (VOO)
# 2. iShares Core S&P500 ETF (IVV)

etfs_tickers = ["IVV", "SPY", "VOO", "^GSPC"]

# using 2 years of data from January 01, 2018 to December 31, 2019
starttime = datetime.datetime(2018, 1, 1)
endtime = datetime.datetime(2019, 12, 31)

# get only the closing prices
etfs = pdr.get_data_yahoo(etfs_tickers, starttime, endtime)['Close']
etfs.columns = ["iShares", "SPDR", "Vanguard",  "S&P500"]

print2(etfs.head())



etfs_return = etfs.pct_change().dropna()

# etfs_return.fillna(0, inplace=True)
returns2 = round(etfs_return*100, 3)

print2(etfs_return, returns2)


eft_index = etfs_return["S&P500"]
etfs_activeR = etfs_return.sub([eft_index, eft_index,eft_index,eft_index], axis='columns')
etfs_activeR.drop("S&P500", axis=1, inplace=True)
etfs_activeR.columns = ["Active_iShares", "Active_SPDR", "Active_Vanguard"]


r_index = returns2["S&P500"]
r_activeR = returns2.sub([r_index, r_index,r_index, r_index], axis='columns')
r_activeR.drop("S&P500", axis=1, inplace=True)
r_activeR.columns = ["Active_iShares", "Active_SPDR", "Active_Vanguard"]


print2(etfs_activeR, r_activeR)

mean_return = etfs_return.mean()
_return = returns2.mean()


mean_return_diff = mean_return - etfs_return["S&P500"].mean()
mean_return_diff

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
te = np.sqrt(r_rbarSquared.sum()/(r_rbarSquared.shape[0] - 1))


m_rbarSquared = etfs_activeR ** 2
mate = np.sqrt(m_rbarSquared.sum()/m_rbarSquared.shape[0])
print2(te, mate)



tickers = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "^GSPC"]

# get only the closing prices
spdr_funds = pdr.get_data_yahoo(tickers, starttime, endtime)['Close']
spdr_funds.columns = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "S&P500"]
spdr_funds.head()

spdr_funds_R = spdr_funds.pct_change().dropna()
# spdr_funds_R.fillna(0, inplace=True)
print2(spdr_funds_R.head())

cum_spdr = (1+spdr_funds_R).cumprod()
plt.figure(figsize=[14,6])
plt.plot(cum_spdr)
plt.legend(cum_spdr.columns)
plt.show()



spdr_index = spdr_funds_R["S&P500"]
spdr_funds_ar = pd.DataFrame()

for col in spdr_funds_R.columns[:-1]:
    spdr_f = spdr_funds_R[[col]].sub([spdr_index], axis='columns')
    spdr_funds_ar = pd.concat([spdr_funds_ar, spdr_f], axis=1, sort=True)

# spdr_funds.drop("S&P500", axis=1, inplace=True)

print2(spdr_funds_ar.head())
spdr_funds_ave = spdr_funds_ar.mean()
spdr_funds_te = spdr_funds_ar.std()
spdr_funds_mate = np.sqrt((spdr_funds_ar**2).sum() / spdr_funds_ar.shape[0])
print2(spdr_funds_ave, spdr_funds_te, spdr_funds_mate)



def activeReturn(etf, ref_index):
    """The activeReturn computes the active return
    
    Inputs:
        etf (Series) : selected ETF
        ref_index:  the reference stock market index 
        
     Output:
        _activeR (Series) : active returns
    
    """
    # computer daily returns
    etf_return = etf.pct_change().dropna()
    index_return = ref_index.pct_change().dropna()
    
    # computer active returns
    _activeR = etf_return.sub([index_return], axis='columns')
    _activeR.columns = [f"Active_{etf.columns[0]}"]
    
    return _activeR


active_mm = activeReturn(spdr_funds_R[["XLU"]], spdr_funds_R["S&P500"])
print(active_mm.head())

print()


# calculate mean adjusted tracking error
# loop through the selected SPY funds dataframe
for col in spdr_funds.columns[:-1]:
    # compute active returns for each
    active_ = activeReturn(spdr_funds[[col]], spdr_funds["S&P500"]) * 100
                                                                                                                                                                                                                            
    # compute mean adjusted tracking error for each                                                   
    mate_ = np.sqrt((active_ ** 2).sum()/active_.shape[0])
                                                             
    # print out the computed mean adjusted tracking error                                                    
    print(f'{col} mean adjusted tracking error: {round(mate_.values[0], 4)}')
    
                                                      
# plot the best tracker                                                         
plt.figure(figsize=[10,6])
plt.plot(cum_spdr[["XLY", "S&P500"]])
plt.legend(["XLY", "S&P500"])
plt.show()
                                                       
"""
DatetimeIndex(['1990-01-01', '1991-01-01', '1992-01-01', '1993-01-01',
               '1994-01-01', '1995-01-01', '1996-01-01', '1997-01-01',
               '1998-01-01', '1999-01-01', '2000-01-01', '2001-01-01',
               '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01',
               '2006-01-01'],
              dtype='datetime64[ns]', name='Date', freq=None)

 newplat = np.array([[100, 100, 100],
       [ 90,  93,  91],
       [104, 110, 104],
       [124, 136, 127],
       [161, 182, 167],
       [186, 216, 190],
       [204, 245, 206],
       [235, 291, 234],
       [258, 330, 260],
       [271, 359.7, 271],
       [339, 460, 346],
       [254, 355, 256],
       [216, 311, 221],
       [216, 321, 223],
       [238, 364, 243],
       [262, 413, 262],
       [275, 447, 273]], dtype=int64) 

[100  90 104 124 161 186 204 235 258 271 339 254 216 216 238 262 275]
[100  93 110 136 182 216 245 291 330 359.7 460 355 311 321 364 413 447]
[100  91 104 127 167 190 206 234 260 271 346 256 221 223 243 262 273]  
          
"""