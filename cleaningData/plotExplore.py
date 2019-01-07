
#!/usr/bin/env python
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('dob_job_application_filings_subset.csv')
df_subset = df[:, 2:]

# data visualization
# Bar plots for discrete data counts 
# Histogram for continous data counts 
df.population.plot('hist')
df[df.population > 1000000]

# box plots
df.boxplot(columns='population', by='continent')

# scatter plots
# relationship b/w 2 numeric values

# Describe the column
df['Existing Zoning Sqft'].describe()

# Plot the histogram
df['Existing Zoning Sqft'].plot(kind='hist', rot=70, logx=True, logy=True)

# Display the histogram
plt.show()


# Create the boxplot
df.boxplot(column='initial_cost', by='Borough', rot=90)

# Display the plot
plt.show


# Create and display the first scatter plot
df.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()

# Create and display the second scatter plot
df_subset.plot(kind='scatter', x='initial_cost', y='total_est_fee', rot=70)
plt.show()



g1800s = pd.read_csv('g1800s.csv')
# Create the scatter plot
g1800s.plot(kind='scatter', x='1800', y='1899')

# Specify axis labels
plt.xlabel('Life Expectancy by Country in 1800')
plt.ylabel('Life Expectancy by Country in 1899')

# Specify axis limits
plt.xlim(20, 55)
plt.ylim(20, 55)

# Display the plot
plt.show()





gapminder = pd.read_csv('gapminder.csv')


# Create a histogram of the life_expectancy column using the .plot() method of gapminder. 
# Specify kind='hist'.
# Group gapminder by 'year' and aggregate 'life_expectancy' by the mean. To do this:
# Use the .groupby() method on gapminder with 'year' as the argument. Then select
#  'life_expectancy' and chain the .mean() method to it.
# Print the head and tail of gapminder_agg. This has been done for you.
# Create a line plot of average life expectancy per year by using 
# the .plot() method (without any arguments in plot) on gapminder_agg.
# Save gapminder and gapminder_agg to csv files called 'gapminder.csv' 
# and 'gapminder_agg.csv', respectively, using the .to_csv() method.

# Add first subplot
plt.subplot(2, 1, 1) 

# Create a histogram of life_expectancy
gapminder.life_expectancy.plot(kind='hist')

# Group gapminder: gapminder_agg
gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean()

# Print the head of gapminder_agg
print(gapminder_agg.head())

# Print the tail of gapminder_agg
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)

# Create a line plot of life expectancy per year
gapminder_agg.plot()

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder.to_csv('gapminder.csv')
gapminder_agg.to_csv('gapminder_agg.csv')




