!pip install TPOT printdescribe xlrd

from tpot import TPOTClassifier
from sklearn.utils import _safe_indexing


import numpy as np
import pandas as pd
import random
from itertools import product

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from printdescribe import print2

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


data = pd.read_excel(path,header=1, index_col=0)
data = data.rename(columns={'default payment next month':"default"})

print2(data.head())
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], stratify=data.iloc[:,-1], test_size=0.3)
print2(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Assign the values outlined to the inputs
number_generations = 3
population_size = 5
offspring_size = 10
scoring_function = "accuracy"

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
                          offspring_size=population_size, scoring=scoring_function,
                          verbosity=2, random_state=2, cv=2)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))
