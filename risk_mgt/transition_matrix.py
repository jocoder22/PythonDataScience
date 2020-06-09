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
    col = data_input.columns
    col1 = f"{col[0]}_movt"
    col2 = f"{col[1]}_movt"

    # Make categorise, u for up and d for down movements for each stock
    data_out[col1] = np.where(data_input.iloc[:,0] > 0,"u","d")
    data_out[col2] = np.where(data_input.iloc[:,1] > 0,"u","d")
    
    # Create direction to represent the combined movements of the stocks
    data_out["groupMovt"] = np.where(data_out[col1] == data_out[col2], "Together", "Apart")
    data_out['direction'] = data_out[col1] .str.cat(data_out[col2])
    
#     # Remove the columns for each stock movement
#     data_out.drop(columns=['_movt0', '_movt1'], inplace=True)
    
    
    # return the final dataset
    return data_out
    

# define a function to the returns transition matrix
def transitionMatrix(data_input2, colm):
    """The transitionMatrix function will return the transition matrix moving from
        present state of the stocks either 
        uu = both stocks moved upwards,
        ud = the first moved upwards and the second moved downwards,
        ud = the first moved downwards and the second moved upwards, or 
        dd = both stocks moved downwards
        
        to another state in one step, i.e from present day (today) to next day

    Args: 
        data_input2 (DataFrame): the DataFrame with the stoch present state
        colm: (string) column for transition matrix
        
    Returns: 
        DataFrame: The transition matrix dataframe

    """
    
    # Get the present state of the stocks movements
    present_state = data_input2[colm].values.tolist()
    
    # Create transition matrix, using crosstab lagged one day
    data_out2 = pd.crosstab(pd.Series(present_state[1:],name='Present_State'),
                pd.Series(present_state[:-1],name='Next_Movt'), normalize="index")
    data_out2['Total'] = data_out2.sum(axis=1)
    
    
    # return the transition matrix
    return data_out2

# Calculate normalized count values for taindata
traincount = trainMovt["direction"].value_counts(normalize=True).reset_index()
traincount.columns = ['index', "Train"]
traincount


# Calculate normalized count values for test data
testcount = testMovt["direction"].value_counts(normalize=True).reset_index()
testcount.columns = ['index', "Test"]
testcount

# merge the normalized count dataframe for comparison
countdata = traincount.merge(testcount, on="index")

