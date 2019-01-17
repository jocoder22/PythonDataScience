
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
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




# Create the regressor: reg
reg = LinearRegression()

X_fertility = df['fertility'].values
# Create the prediction space
prediction_space = np.linspace(
    min(X_fertility), max(X_fertility)).reshape(-1, 1)


X_fertility = X_fertility.reshape(-1, 1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X_fertility, y))  # 0.619244216774

# Plot regression line
plt.scatter(X_fertility, y, color='blue', marker='.')
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()




y = df['life'].values
# y = y.reshape(-1, 1)
X = df.drop(['life', 'Region'], axis=1).values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))  # R^2: 0.838046873142936
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))




# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(cv_scores.mean()))



