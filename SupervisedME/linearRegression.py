import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
plt.style.use('ggplot')


# Access the European dataset
url = 'https://assets.datacamp.com/production/course_1939/datasets/gm_2008_region.csv'

colname = ['population', 'fertility', 'HIV', 'CO2', 'BMI_male',
           'GDP', 'BMI_female', 'life', 'child_mortality', 'Region']

df = pd.read_csv(url, sep=',')


# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

print(type(X))
print(type(y))
# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))


# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

# Cells that are in green show positive correlation, 
# while cells that are in red show negative correlation
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.show()
