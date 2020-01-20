import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from scipy.optimize import minimize

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")
        
        
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

print2(diabetes_X, diabetes_y, diabetes_X[0].size)

def linearLoss(w):
    sse = 0
    
    for k in range(diabetes_y.size):
        y = diabetes_y[k]
        yhat = w@diabetes_X[k]
        
        sse += (y - yhat)**2
        
    return sse

# choose the initial values
x0 = np.linspace(-2, 2, diabetes_X[0].size)

# # Returns the w that makes my_loss(w) smallest
lossWeights = minimize(linearLoss, x0).x

# Fit linear regression using scipy LinearRegression
lr = linear_model.LinearRegression(fit_intercept=False).fit(diabetes_X, diabetes_y)
print2(lossWeights, lr.coef_)