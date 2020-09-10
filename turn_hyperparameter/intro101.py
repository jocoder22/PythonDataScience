import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt

from printdescribe import print2, describe2

# please install graphviz and pydotplus

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
data = pd.read_excel(path, header=1, index_col=0)

print2(data.head(), data.info(), data.shape)

data["default payment next month"].value_counts()

data = data.rename(columns={"default payment next month" : "default"})
print2(data.head())
prop = data["default"].value_counts(normalize=True)*100
print2(prop)


X_train, X_test, y_train, y_test = tts(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.3, 
                                       stratify=data.iloc[:,-1], random_state=42)

print2(X_train.shape)

model_lg = LogisticRegression(solver="lbfgs", max_iter=10000)
model_lg.fit(X_train, y_train)
print2(model_lg.coef_)


# var_names = list(data.iloc[:,:-1].columns)
var_names = list(X_train.columns)
zipper = list(zip(var_names, model_lg.coef_[0]))
print2(zipper)

coeffs = [list(x) for x in zipper]
coeffs
