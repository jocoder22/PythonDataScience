import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")
        
        
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

print2(diabetes_X, diabetes_y, diabetes_X[0].size)

# define the loss function for linear model
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



# Load the load_breast_cancer dataset

breast_X, breast_y = datasets.load_breast_cancer(return_X_y=True)

print2(breast_X, breast_y, breast_X[0].size)

# mydata = datasets.load_breast_cancer()
# print2(mydata.DESCR, mydata.data, mydata.target, mydata.target_names, mydata.feature_names)


# Define the logistic and hinge losses
def logisticfunct(modelOutput):
   return np.log(1+np.exp(-modelOutput))


def hingefunct(w):
   return np.maximum(0,1-modelOutput)


# Define the minimaxisation function
def classificationLoss(w):
    sse = 0
    
    for k in range(breast_y.size):
        yhat = w@breast_X[k]
        
        sse += logisticfunct(yhat * breast_y[k])
        
    return sse


# breast_X = breast_X[:, list(range(10))]
# choose the initial values
x0 = breast_X[0]

# # Returns the w that makes my_loss(w) smallest
classWeights = minimize(classificationLoss, x0).x


# Fit linear regression using scipy LinearRegression
llogic = linear_model.LogisticRegression(fit_intercept=False, max_iter = 10000).fit(breast_X, breast_y)
print2(classWeights, llogic.coef_, classWeights - llogic.coef_)