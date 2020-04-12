import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pydotplus.graphviz import graph_from_dot_data
# plt.style.use('seaborn-whitegrid')
plt.style.use('dark_background')


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
        
# load pickle file
mydir = r"D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"
features = pd.read_pickle(os.path.join(mydir, "features.pkl"))
target= pd.read_pickle(os.path.join(mydir, "target.pkl"))

features.fillna(features.mean(),inplace=True)
ddd = features.loc[:,["tenure",  "MonthlyCharges",  "TotalCharges"]].agg(["mean", "std"]).round()

print2(features.head(), target.head(), ddd, target.shape)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, stratify=target)

# Initialize the model, with dep
dtree = DecisionTreeClassifier(max_depth=5, random_state=1973)

model = dtree.fit(X_train, Y_train)

Y_predict_test = dtree.predict(X_test)
Y_predict_train = dtree.predict(X_train)

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



graphtreeObject = export_graphviz(model, filled=True, rounded=True,
                class_names=["Churn", "Not Churn"],
                feature_names=features.columns,
                out_file=None)

graph = graph_from_dot_data(graphtreeObject)
graph.write_pdf(os.path.join(mydir, 'Marketing.pdf'))
graph.write_png(os.path.join(mydir, 'Marketing.png'))


# find the best depth
depth = list(range(2, 20))
depthList = np.zeros((len(depth), 4))
depthList[:, 0] =  depth

print2(depth, depthList)

for indx in range(len(depth)):
    dtree = DecisionTreeClassifier(max_depth=depth[indx])

    model = dtree.fit(X_train, Y_train)
    yhat = dtree.predict(X_test)
    
    depthList[indx, 1] = accuracy_score(Y_test, yhat)
    depthList[indx, 2] =  precision_score(Y_test, yhat)
    depthList[indx, 3] =  recall_score(Y_test, yhat)
    
    
colname = ["MaxDepth", "AccuracyScore", "PrecisionScore", "RecallScore"]

AssessTable = pd.DataFrame(depthList, columns=colname)
AssessTable["R/P Ratio"] =  AssessTable.RecallScore / AssessTable.PrecisionScore 

print(AssessTable.head())

plt.plot(AssessTable["MaxDepth"], AssessTable.loc[:, "AccuracyScore": "R/P Ratio"])
plt.xticks(AssessTable["MaxDepth"])
plt.legend(labels=('AccuracyScore', 'PrecisionScore','RecallScore' , 'R/P Ratio'), loc='upper right')
plt.grid()
plt.show()