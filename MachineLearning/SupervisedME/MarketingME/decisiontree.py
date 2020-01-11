import pandas as pd
import os 


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# load pickle file
mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
features = pd.read_pickle(os.path.join(mydir, "features.pkl"))
target= pd.read_pickle(os.path.join(mydir, "target.pkl"))

ddd = features.loc[:,["tenure",  "MonthlyCharges",  "TotalCharges"]].agg(["mean", "std"]).round()

print2(features.head(), target.head(), ddd)


