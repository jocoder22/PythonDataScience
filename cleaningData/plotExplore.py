
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



