#!/usr/bin/env python
# Import required modules
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import pandas_datareader as pdr

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

# My first stock is Netflix
# I'm using daily close prices
stock_1 = "NFLX"

# using 2 years of data from January 01, 2018 to December 31, 2019
starttime = datetime.datetime(2018, 1, 1)
endtime = datetime.datetime(2019, 12, 31)

# get only the closing prices
netflix = pdr.get_data_yahoo(stock_1, starttime, endtime)['Close']

# Calculate log return
logReturn_netflix = np.log(netflix).diff().dropna()  # fastest
logReturn_netflix2 = np.log(netflix.diff()/netflix.shift() + 1).dropna() # faster
logReturn_netflix2 = np.log(netflix.pct_change() + 1).dropna()

print2(logReturn_netflix.head())

# calculate mean
netflixmean = logReturn_netflix.mean()

# calculate standard deviation
netflixstd = logReturn_netflix.std()

# calculate skewness
netflixskewness = logReturn_netflix.skew()

# calculate excess kurtosis
netflixkurtosis = logReturn_netflix.kurtosis()

print2(netflixmean, netflixstd, netflixskewness, netflixkurtosis)

# My second stock is Tesla
# I'm using daily close prices
stock_2 = "TSLA"

# using 2 years of data from January 01, 2018 to December 31, 2019
starttime = datetime.datetime(2018, 1, 1)
endtime = datetime.datetime(2019, 12, 31)

# get only the closing prices
tesla = pdr.get_data_yahoo(stock_2, starttime, endtime)['Close']

# Calculate log return
logReturn_tesla = np.log(tesla).diff().dropna()
print2(logReturn_tesla.head())

# calculate mean
teslamean = logReturn_tesla.mean()

# calculate standard deviation
teslastd = logReturn_tesla.std()

# calculate skewness
teslaskewness = logReturn_tesla.skew()

# calculate excess kurtosis
teslakurtosis = logReturn_tesla.kurtosis()

print2(teslamean, teslastd, teslaskewness, teslakurtosis)

# Calulate their correlation
# Create combine data using pandas dataframe
data = pd.DataFrame({"Netflix" : logReturn_netflix, "Tesla" :logReturn_tesla})
correlation = data.corr()
covariance = data.cov()

print2(correlation, covariance)

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
    data_out[col1] = np.where(data_out[col[0]] > 0,"u","d")
    data_out[col2] = np.where(data_out[col[1]] > 0,"u","d")
    
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


# spilt the data, train(80%) and test (20%)
train, test = np.split(data, [int(0.8 * len(data))])

# Split the data into train (80%) and test (20%) datasets
# size_t = int(len(data) * 0.8)
# train, test = data[0:size_t], data[size_t:]
# print2(train.shape, test.shape, train.head(), test.head())

# define the groups
trainMovt = stock_movt(train)
testMovt = stock_movt(test)

print2(train.head(), test.head(), trainMovt.head(), testMovt.head())

# define transition matrix on groupMovt
traingroupMovt = transitionMatrix(trainMovt, "groupMovt")
testgroupMovt = transitionMatrix(testMovt, "groupMovt")

# define transition matrix on direction
traindirectionMovt = transitionMatrix(trainMovt, "direction")
testdirectionMovt = transitionMatrix(testMovt, "direction")

print2(traingroupMovt, testgroupMovt, traindirectionMovt, testdirectionMovt)

# Calculate normalized count values for taindata
traincount = trainMovt["groupMovt"].value_counts(normalize=True).reset_index()
traincount.columns = ['index', "Train"]

# Calculate normalized count values for test data
testcount = testMovt["groupMovt"].value_counts(normalize=True).reset_index()
testcount.columns = ['index', "Test"]
print2(traincount, testcount)

# merge the normalized count dataframe for comparison
countdata = traincount.merge(testcount, on="index")
print("This is for Group Movement", countdata, sep ="\n", end="\n\n")

# Calculate normalized count values for taindata
traincount = trainMovt["direction"].value_counts(normalize=True).reset_index()
traincount.columns = ['index', "Train"]

# Calculate normalized count values for test data
testcount = testMovt["direction"].value_counts(normalize=True).reset_index()
testcount.columns = ['index', "Test"]
print2(traincount, testcount)

# merge the normalized count dataframe for comparison
countdata = traincount.merge(testcount, on="index")
print("This is for Direction Movement", countdata, sep ="\n", end="\n\n")

# kk = random.sample(range(10, 999), 50000)
# kk1 = random.sample(range(10, 999), 50000)
kk = np.random.randint(-3, 3, (5000, 1)).flatten()
kk1 = np.random.uniform(-3, 3, (5000, 1)).flatten()
