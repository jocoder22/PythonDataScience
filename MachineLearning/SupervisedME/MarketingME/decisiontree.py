import pandas as pd
import os 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# load pickle file
mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
features = pd.read_pickle(os.path.join(mydir, "features.pkl"))
target= pd.read_pickle(os.path.join(mydir, "target.pkl"))

ddd = features.loc[:,["tenure",  "MonthlyCharges",  "TotalCharges"]].agg(["mean", "std"]).round()

print2(features.head(), target.head(), ddd, target.shape)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, stratify=target)

# Initialize the model, with dep
dtree = DecisionTreeClassifier(max_depth=3)

model = dtree.fit(X_train, Y_train)

Y_predict = dtree.predict(X_test)

score = accurate_score(Y_test, Y_predict)


