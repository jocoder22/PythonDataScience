!pip install hyperopt

import numpy as np
import pandas as pd
import random


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from hyperopt import hp, fmin, tpe




from printdescribe import print2

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


data = pd.read_excel(path,header=1, index_col=0)
data = data.rename(columns={'default payment next month':"default"})

print2(data.head())
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], stratify=data.iloc[:,-1], test_size=0.3)
print2(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Set up space dictionary with specified hyperparameters
space = {'max_depth': hp.quniform('max_depth', 2, 10, 2),'learning_rate': hp.uniform('learning_rate', 0.001,0.9)}

# Set up objective function
def objective(params):
    params = {'max_depth': int(params["max_depth"]),'learning_rate': params["learning_rate"]}
    gbm_clf = GradientBoostingClassifier(n_estimators=100, **params) 
    best_score = cross_val_score(gbm_clf, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()
    loss = 1 - best_score
    writeout(best_score, params, iteration)
    return loss

  
# Run the algorithm
best = fmin(fn=objective,space=space, max_evals=20, rstate=np.random.RandomState(42), algo=tpe.suggest)
print2(best)
