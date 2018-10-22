import os
import pandas as pd
import matplotlib.pyplot as plt
# from pandas import Series, DataFrame

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 2")


os.chdir("C:/Users/Jose/Documents/PythonDataScience1/Code/Code/Section 2")
"C:\Users\Jose\Documents\PythonDataScience1\Code\Code\Section 1"
# Read data1
data1 = pd.read_csv("AAPL-Close-15-17.csv", index_col="date")
data1.head()
plt.plot(data1)
plt.show()


# Read data2
data2 = pd.read_csv("AAPL-EPS-15-17.csv", index_col="Date")
data2.head()
plt.plot(data2)
plt.show()

# Merging datas - Inner, Outer, Right, Left
# Pandas use join() method
# Note the indexes - date - of both dataset are not of the same format
data1.index   # '2015-01-02', '2015-01-05', '2015-01-06'
data2.index   # '4/3/2017', '12/30/2016', '9/26/2016', '6/27/2016'


# convert to common datetime format
pd.to_datetime(data1.index)  # '2015-01-02', '2015-01-05', '2015-01-06',
pd.to_datetime(data2.index)  # '2017-04-03', '2016-12-30', '2016-09-26',

data1.index = pd.to_datetime(data1.index)
data2.index = pd.to_datetime(data2.index)

data1.head()
data2.head()

# Now do the joins
data1.join(data2, how="inner")
data1.join(data2, how="outer").head()
data1.join(data2, how="left").head()
data1.join(data2, how="right")


# Reshaping data;
# using Stacking and Unstacking


os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 1")
data3 = pd.read_csv("PopPyramids.csv",
                    index_col=["Country", "Year", "Age"])
data3.drop("Region", 1, inplace=True)
data3.head()

# list the columns names
data3.columns
data3.columns.tolist()

# Stacking
data3_stack = data3.stack()
data3_stack

# Unstacking
data3_stack.unstack()
data3_stack.unstack().head()

# Defining level of unstacking, using numeric index
data3_stack.unstack(level=0).head()  # on country
data3_stack.unstack(level=1).head()  # on year
data3_stack.unstack(level=2).head()  # on age
data3_stack.unstack(level=3).head()  # on rest of variables
data3_stack.unstack(level=4).head()  # Error! only 3 levels is possible

# Using list in levels, positional arguments
data3_stack.unstack(level=[0, 1]).head()
data3_stack.unstack(level=[2, 1]).head()


# Unstacking with column names
data3_stack.unstack(level="Age").head()
data3_stack.unstack(level=["Country", "Age"]).head()


# Reset Index
# this turns the index into regular columns
data4 = data3.reset_index()
data4.head()

data4.stack().head()
data4.stack().unstack().head()

# Melting and casting
pd.melt(data4)
data4.head()


# melting only the head of  the dataset
pd.melt(data4.head())

# melting while preserving the index columns
pd.melt(data4, id_vars=['Year', 'Age', 'Country'])
pd.melt(data4, id_vars=['Year', 'Age']).head()

# Form new datasets
data5 = pd.melt(data4.head())
data6 = pd.melt(data4, id_vars=['Year', 'Age', 'Country'])
data7 = pd.melt(data4, id_vars=['Year', 'Age'])

data5.head()
data6.head()
data7.head()


# Specify columns in the value column
pd.melt(data4.head(), value_vars=["Both Sexes Population",
                                  "Male Population",
                                  "Female Population"])


# specify both index columns and variables to melt
pd.melt(data4.head(),
        id_vars=["Year", "Age", "Country"],
        value_vars=["Both Sexes Population",
                    "Male Population",
                    "Female Population"])

pd.melt(data4.head(),
        value_vars=["Year", "Age", "Country",
                    "Both Sexes Population",
                    "Male Population",
                    "Female Population"])

pd.pivot_table(data6, values="value", index=["Year", "Age", "Country"],
               columns="variable")

data8 = pd.melt(data4.head(),
        value_vars=["Both Sexes Population",
                    "Male Population",
                    "Female Population"])

data8.head()

