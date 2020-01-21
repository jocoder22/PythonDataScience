import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

def print2(*args):
    for arg in  args:
        print(arg, end="\n\n")
        
        
# Load the load_breast_cancer dataset
breast_X, breast_y = datasets.load_breast_cancer(return_X_y=True)

# print2(breast_XX[1])
# bb = [ 0,  1,  2,  3, 11, 12, 13, 21, 22, 23, 26, 27, 28]
# breast_X = np.delete(breast_XX, bb, 1)

# bb2 = [ 1,  2,  4,  6, 13, 14, 15]
# breast_X = np.delete(breast_X, bb2, 1)

# bb3 = [1, 2, 5]
# breast_X = np.delete(breast_X, bb3, 1)

# bb4 = [0, 3, 6]
# breast_X = np.delete(breast_X, bb4, 1)
print2(breast_X.shape, breast_X[1], breast_y.shape, breast_X[0].size)

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(
    breast_X, breast_y, test_size=0.5, random_state=42)


# Train and validaton errors initialized as empty list
trainError = []
testError = []
c =  [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]


# Loop over values of c
for c_values in c:
    # Create LogisticRegression object and fit
    logReg = LogisticRegression(C=c_values, solver='liblinear')
    logReg.fit(X_train, y_train)
    
    # Evaluate error rates and append to lists
    trainError.append( 1.0 - logReg.score(X_train, y_train) )
    testError.append( 1.0 - logReg.score(X_test, y_test) )
    
# Plot results
plt.semilogx(c, trainError, c, testError)
plt.legend(("train", "Test"))
plt.show()



# Specify L1 regularization
logReg2 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)

# Instantiate the GridSearchCV object and run the search
searcher = GridSearchCV(logReg2, {'C':[0.001, 0.01, 0.1, 1, 10]})
searcher.fit(X_train, y_train)
resultsKeys = sorted(searcher.cv_results_.keys())


# Report the best parameters
print("Best CV params", searcher.best_params_)

# Find the number of nonzero coefficients (selected features)
best_lr = searcher.best_estimator_
coefs = best_lr.coef_
nonzeroIndices = np.nonzero(coefs.flatten())
print("Total number of features:", coefs.size)
print("Number of selected features:", np.count_nonzero(coefs))
print2(f'The parameter keys: {resultsKeys}', f'The best estimator : {best_lr}', 
       f'The best coefficients: {coefs}', searcher.best_index_, nonzeroIndices, coefs.flatten())



mydata = datasets.load_breast_cancer()
print2(mydata.DESCR, mydata.feature_names)

for k in [14,15, 18, 19]:
    print(mydata.feature_names[k], end="\n")