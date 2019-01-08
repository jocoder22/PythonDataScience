import pandas as import pd

# Your job is to fill in missing 'age' values for passengers on the Titanic
# with the median age from their 'gender' and 'pclass'. To do this, you'll
# group by the 'sex' and 'pclass' columns and transform each group with a
# custom function to call .fillna() and impute the median value.
# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex', 'pclass'])

# Write a function that imputes median


def impute_median(series):
    return series.fillna(series.median())


# Impute age and assign to titanic['age']
titanic.age = by_sex_class.age.transform(impute_median)

# Print the output of titanic.tail(10)
print(titanic.tail(10))


# In this exercise, you're going to analyze economic disparity within
# regions of the world using the Gapminder data set for 2010. To do this
#  you'll define a function to compute the aggregate spread of per capita
# GDP in each region and the individual country's z-score of the regional
# per capita GDP. You'll then select three countries - United States, Great
#  Britain and China - to see a summary of the regional GDP and that
# country's z-score against the regional mean.
def disparity(gr):
    # Compute the spread of gr['gdp']: s
    s = gr['gdp'].max() - gr['gdp'].min()
    # Compute the z-score of gr['gdp'] as (gr['gdp']-gr['gdp'].mean())/gr['gdp'].std(): z
    z = (gr['gdp'] - gr['gdp'].mean())/gr['gdp'].std()
    # Return a DataFrame with the inputs {'z(gdp)':z, 'regional spread(gdp)':s}
    return pd.DataFrame({'z(gdp)': z, 'regional spread(gdp)': s})


# Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby('region')

# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
print(reg_disp.loc[['United States', 'United Kingdom', 'China']])


# In this exercise you'll take the Titanic data set and analyze survival rates from
# the 'C' deck, which contained the most passengers. To do this you'll group the
# dataset by 'sex' and then use the .apply() method on a provided user defined function
# which calculates the mean survival rates on the 'C' deck:


def c_deck_survival(gr):

    c_passengers = gr['cabin'].str.startswith('C').fillna(False)

    return gr.loc[c_passengers, 'survived'].mean()


# Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')

# Call by_sex.apply with the function c_deck_survival
c_surv_by_sex = by_sex.apply(c_deck_survival)

# Print the survival rates
print(c_surv_by_sex)
