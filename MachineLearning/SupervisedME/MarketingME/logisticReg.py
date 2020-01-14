import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
# plt.style.use('seaborn-whitegrid')
plt.style.use('dark_background')


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# load pickle file
mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
features = pd.read_pickle(os.path.join(mydir, "features.pkl"))
target= pd.read_pickle(os.path.join(mydir, "target.pkl"))

print2(features.head())

features.fillna(features.mean(),inplace=True)
target = target.loc[:, "Churn"]

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, stratify=target)

print2(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Initialize the model
logreg = LogisticRegression(penalty = "l1", C = 0.05, solver = "liblinear", random_state=1973)


model = logreg.fit(X_train, Y_train)

Y_predict_test = model.predict(X_test)
Y_predict_train = model.predict(X_train)


accuracyscoreTest = accuracy_score(Y_test, Y_predict_test)
accuracyscoreTrain = accuracy_score(Y_train, Y_predict_train)

precisionscoreTest = precision_score(Y_test, Y_predict_test)
precisionscoreTrain = precision_score(Y_train, Y_predict_train)

recallscoresTest = recall_score(Y_test, Y_predict_test)
recallscoreTrain = recall_score(Y_train, Y_predict_train)

print2("{} accuracy score : {:.2f}".format("Test", accuracyscoreTest),
       "{} accuracy score : {:.2f}".format("Train", accuracyscoreTrain),
       "{} precision score : {:.2f}".format("Test", precisionscoreTest),
       "{} precision score : {:.2f}".format("Train", precisionscoreTrain),
       "{} recall score : {:.2f}".format("Test", recallscoresTest),
       "{} recall score : {:.2f}".format("Train", recallscoreTrain))



# find the best C
depth = np.linspace(1, 0.0005, 40).tolist()
# depth = np.arange(0.0025, 1, 0.005).tolist()
depthList = np.zeros((len(depth), 5))
depthList[:, 0] =  depth

print2(depth, depthList)

for indx in range(len(depth)):
    logreg = LogisticRegression(penalty = "l1", C = depth[indx], solver = "liblinear", random_state=1973)

    model = logreg.fit(X_train, Y_train)
    yhat = logreg.predict(X_test)
    
    depthList[indx, 1] = np.count_nonzero(model.coef_)
    depthList[indx, 2] = accuracy_score(Y_test, yhat)
    depthList[indx, 3] =  precision_score(Y_test, yhat)
    depthList[indx, 4] =  recall_score(Y_test, yhat)
    
    
colname = ["C_value", "NonZero_count", "AccuracyScore", "PrecisionScore", "RecallScore"]

AssessTable = pd.DataFrame(depthList, columns=colname)
AssessTable["R/P Ratio"] =  AssessTable.RecallScore / AssessTable.PrecisionScore 

print(AssessTable.head())

plt.plot(AssessTable["C_value"], AssessTable.loc[:, "AccuracyScore": "R/P Ratio"])
plt.xticks(AssessTable["C_value"], rotation=45)
plt.legend(labels=("NonZero_count", 'AccuracyScore', 'PrecisionScore','RecallScore' , 'R/P Ratio'), loc='upper right')
plt.grid()
plt.show()