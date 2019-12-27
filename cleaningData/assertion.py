import numpy as np 
import pandas as pd 

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  
sp = {"sep":"\n\n", "end":"\n\n"}

with open(r"D:\Wqu_Datascience\gapminder.csv") as file1:
    gapminder = pd.read_csv(file1, index_col=0)
    # g1800s = pd.read_csv('g1800s')


print2(gapminder.columns)
# Convert the year column to numeric
gapminder['year'] = pd.to_numeric(gapminder['Year'])
print2(gapminder.columns, gapminder.info(), gapminder.head())

# Test if country is of type object
assert gapminder.Country.dtypes == np.object

# Test if year is of type int64
assert gapminder.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder.life.dtypes == np.float64

# Assert that country does not contain any missing values
assert pd.notnull(gapminder.Country).all()

# Assert that year does not contain any missing values
assert pd.notnull(gapminder.Year).all()



with open(r"D:\PythonDataScience\pandas\people.csv") as file2:
    people = pd.read_csv(file2)
    
print2(people.head, people.columns, people.info(), people.tail())


# # Check whether the first column is 'Life expectancy'
# assert g1800s.columns[0] == 'Life expectancy'

# # Check whether the values in the row are valid
# assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# # Check that there is only one instance of each country
# assert g1800s['Life expectancy'].value_counts()[0] == 1

from pyspark import SparkContext as sc 
y = sc.textFile(name="read.md")