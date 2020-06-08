#!/usr/bin/env python
# Import required modules
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr


# My first stock is Neflix
# I'm using daily close prices
stock_1 = "NFLX"

# using 2 years of data from January 01, 2018 to December 31, 2019
starttime = datetime.datetime(2018, 1, 1)
endtime = datetime.datetime(2019, 12, 31)

# get only the closing prices
neflix = pdr.get_data_yahoo(stock_1, starttime, endtime)['Close']


# Calculate log return
logReturn_neflix = np.log(neflix).diff().dropna()
logReturn_neflix.head()

# Calulate their correlation
# Create combine data using pandas dataframe
data = pd.DataFrame({"Neflix" : logReturn_neflix, "Tesla" :logReturn_tesla})
data.corr()

# define a function to show stocks movements
def stock_movt(data_input):
    """The stock_movt function will add a column to the dataset
        showing the movement of the stocks either 
        uu = both stocks moved upwards,
        ud = the first moved upwards and the second moved downwards,
        ud = the first moved downwards and the second moved upwards, or 
        dd = both stocks moved downwards

    Args: 
        data_input (DataFrame): the DataFrame with the stocks log return
        
    Returns: 
        DataFrame: The DataFrame with the movement column

    """
    
    # Make a copy of the input dataset
    data_out = data_input.copy()

    # Make categorise, u for up and d for down movements for each stock
    data_out['_movt0'] = np.where(data_input.iloc[:,0] > 0,"u","d")
    data_out['_movt1'] = np.where(data_input.iloc[:,1] > 0,"u","d")
    
    # Create direction to represent the combined movements of the stocks
    data_out['direction'] = data_out['_movt0'] .str.cat(data_out['_movt1'])
    
    # Remove the columns for each stock movement
    data_out.drop(columns=['_movt0', '_movt1'], inplace=True)
    
    
    # return the final dataset
    return data_out
    