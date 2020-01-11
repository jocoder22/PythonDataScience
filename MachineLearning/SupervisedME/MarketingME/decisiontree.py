import pandas as pd
import os 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# load pickle file
mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
features = pd.read_pickle(os.path.join(mydir, "features.pkl"))
target= pd.read_pickle(os.path.join(mydir, "target.pkl"))

features.fillna(features.mean(),inplace=True)
ddd = features.loc[:,["tenure",  "MonthlyCharges",  "TotalCharges"]].agg(["mean", "std"]).round()

print2(features.head(), target.head(), ddd, target.shape)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, stratify=target)

print2(X_train.info(), X_test.info(), Y_train.info(), Y_test.info())
# Initialize the model, with dep
dtree = DecisionTreeClassifier(max_depth=9)

model = dtree.fit(X_train, Y_train)

Y_predict_test = dtree.predict(X_test)
Y_predict_train = dtree.predict(X_train)

scoretest = accuracy_score(Y_test, Y_predict_test)
scoretrain = accuracy_score(Y_train, Y_predict_train)

print2("{} accuracy score : {:.2f}".format("Test", scoretest),
       "{} accuracy score : {:.2f}".format("Train", scoretrain))


