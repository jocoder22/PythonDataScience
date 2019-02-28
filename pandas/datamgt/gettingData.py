import os
import pandas as pd 
import matplotlib.pyplot as plt


print(os.getcwd())

os.chdir('C:/Users/.../Code/Section 1')
data = pd.read_csv('PopPyramids.csv')
data.head()


# Setting index columns;
data = pd.read_csv('PopPyramids.csv',
                   index_col=['Country', 'Year', 'Age'])


# drop column (Region) and sort data in place;
data.drop('Region', axis=1, inplace=True)
data.sort_index(inplace=True)
data.head()


# Subsetting for United States;
data.loc[('UnitedStates', 2013), :]
data.loc[('UnitedStates', 2013), :].head()


# Read excel file -- must install xlrd and openpyxl ;
data2 = pd.read_excel('PopPyramids.xlsx', sheet_name='Sheet1')
# sheet_name starts at zero - zero indexed;
data3 = pd.read_excel('PopPyramids.xlsx', sheet_name=0)


# Read html file  -- must install lxml;
data4 = pd.read_html('PopPyramids.html')  # return a list
print(data4)
data4[0].head()
type(data4)  # <class 'list'>

# form dataframe
data5 = pd.read_html('PopPyramids.html', 
                    attrs={'id': 'PopData'},
                    index_col=[1, 2, 3])[0]
type(data5)  # <class 'pandas.core.frame.DataFrame'>
data5.drop('Region', axis=1, inplace=True)
data5.sort_index(inplace=True)
data5.head()

