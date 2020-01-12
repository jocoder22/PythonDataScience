import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pydotplus.graphviz import graph_from_dot_data
# plt.style.use('seaborn-whitegrid')
plt.style.use('dark_background')


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# load pickle file
mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
features = pd.read_pickle(os.path.join(mydir, "features.pkl"))
target= pd.read_pickle(os.path.join(mydir, "target.pkl"))

features.fillna(features.mean(),inplace=True)
target = target.loc[:, "Churn"]

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, stratify=target)

print2(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Initialize the model
logreg = LogisticRegression(penalty = "l1", C = 0.1, solver = "liblinear", random_state=1973)


model = logreg.fit(X_train, Y_train)

Y_predict_test = model.predict(X_test)
Y_predict_train = model.predict(X_train)

accuracyscoreTest = accuracy_score(Y_test, Y_predict_test)
accuracyscoreTrain = accuracy_score(Y_train, Y_predict_train)

print2("{} accuracy score : {:.2f}".format("Test", accuracyscoreTest),
       "{} accuracy score : {:.2f}".format("Train", accuracyscoreTrain))