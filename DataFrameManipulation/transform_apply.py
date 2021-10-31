import numpy as np
import pandas as pd
from printdescribe import changepath


# Your job is to fill in missing 'age' values for passengers on the Titanic
# with the median age from their 'gender' and 'pclass'. To do this, you'll
# group by the 'sex' and 'pclass' columns and transform each group with a
# custom function to call .fillna() and impute the median value.
# Create a groupby object: by_sex_class

path = "E:\PythonDataScience\DataFrameManipulation"

# get the dataset
with changepath(path):
    titanic = pd.read_csv("titanic.csv", index_col=0)

# print(titanic.head())
print(titanic["age"].isna().sum())

by_sex_class = titanic.groupby(['sex', 'pclass'])

# Write a function that imputes median


def impute_median(series):
    return series.fillna(series.median())


# Impute age and assign to titanic['age']
titanic.age = by_sex_class.age.transform(impute_median)

# Print the output of titanic.tail(10)
# print(titanic.tail(10))
print(titanic["age"].isna().sum())


# In this exercise, you're going to analyze economic disparity within
# regions of the world using the Gapminder data set for 2010. To do this
# you'll define a function to compute the aggregate spread of per capita
# GDP in each region and the individual country's z-score of the regional
# per capita GDP. You'll then select three countries - United States, Great
#  Britain and China - to see a summary of the regional GDP and that
# country's z-score against the regional mean.

def disparity(gr):
    """
    
    
    """
    # Compute the spread of gr['gdp']: s
    spreadRegion = gr['gdp'].max() - gr['gdp'].min()

    regionMean = gr['gdp'].mean()

    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    zscoreCountry = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()

    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    ## 1. can use join
    gr = gr.join(pd.DataFrame({'z(gdp)': zscoreCountry, 
                                'regional spread(gdp)': spreadRegion,
                                'regional mean(gdp)': regionMean}, index=gr.index))


    ## 2. can use list unpacking                           
    # gr['z(gdp)'], gr['regional spread(gdp)'],  gr[''regional mean(gdp)'] =  [zscoreCountry,  spreadRegion, regionMean]

    ## 3. can use dataframe expansion
    # gr[['z(gdp)', 'regional spread(gdp)', 'regional mean(gdp)']]  =  pd.DataFrame([[zscoreCountry,  spreadRegion, regionMean]],
    #                                                                                 index=gr.index)

    ## 4. can use assign 
    # gr = gr.assign(zScore = zscoreCountry, Regionalspread = spreadRegion,
    #                             Regionalmean = regionMean)

    return gr



# get the dataset
with changepath(path):
    gapminder = pd.read_csv("gapminder.csv", index_col=0)

# select the 2010 dataset
gapminder_2010 = gapminder[gapminder["Year"] == 2010]

print(gapminder_2010.tail(), gapminder_2010.shape, sep="\n")

# Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby('region')

# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
query1 = "Country == 'United States' or  Country =='United Kingdom' or Country == 'China' or Country == 'Nigeria'" 
                                                                                                                        
print(reg_disp[["Country", "z(gdp)", "regional spread(gdp)", "regional mean(gdp)"]].query(query1))
# print(reg_disp[["Country", "zScore", "Regionalspread", "Regionalmean"]].query(query1))
                                                              


# In this exercise you'll take the Titanic data set and analyze survival rates from
# the 'C' deck, which contained the most passengers. To do this you'll group the
# dataset by 'sex' and then use the .apply() method on a provided user defined function
# which calculates the mean survival rates on the 'C' deck:


def c_deck_survival(gr):
    """
    
    
    """

    # select c deck passengers
    cdeck_passengers = gr['cabin'].str.startswith('C').fillna(False)

    # compute the mean survival of  c deck passengers
    return gr.loc[cdeck_passengers, 'survived'].mean()


# Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')

# Call by_sex.apply with the function c_deck_survival
c_surv_by_sex = by_sex.apply(c_deck_survival)

# Print the survival rates
print(c_surv_by_sex)


No_odd_squares = [num ** 2 for num in range(0, 10) if num % 2 == 0]
# No_odd_squares = [num ** 2 for num in [ p if p % 2 == 0 else 0  for p in range(0, 10)]]
print(No_odd_squares)