#!/usr/bin/env python

import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split as splitit
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


iris_ = datasets.load_iris()
iris_features = iris_.data
iris_target = iris_.target
print(np.unique(iris_target))
print(iris_features[:5, :])


digits_ = datasets.load_digits()
digits_featrues = digits_.data
digits_target = digits_.target
print(np.unique(digits_target))
print(digits_featrues[:5, :])

# split the dataset for training and testing
Xi_train, Xi_test, yi_train, yi_test = splitit(iris_features, iris_target, test_size=0.3,
                                       random_state=1, stratify=iris_target)

Xd_train, Xd_test, yd_train, yd_test = splitit(digits_featrues, digits_target, test_size=0.3,
                                       random_state=1, stratify=digits_target)

# fit the Decision tree model
mytree1 = DecisionTreeClassifier(criterion='gini', max_depth=4, 
                            random_state=1)

mytree1.fit(Xi_train, yi_train)
yi_pred = mytree1.predict(Xi_test)
print('Accuracy: {:.3f}'.format(accuracy_score(yi_test, yi_pred)))
print('Accuracy: {:.2f}%'.format(accuracy_score(yi_test, yi_pred) * 100))
# Accuracy: 97.78%


mytree2 = DecisionTreeClassifier(criterion='gini', max_depth=30, 
                            random_state=1)
mytree2.fit(Xd_train, yd_train)
yd_pred = mytree2.predict(Xd_test)
print('Accuracy: {:.3f}'.format(accuracy_score(yd_test, yd_pred)))
print('Accuracy: {:.2f}%'.format(accuracy_score(yd_test, yd_pred) * 100))
# Accuracy: 85.74%

def runtree(xt, yt, xtest, ytest, n):
    scorelist = []
    for i in range(4, n):
        tree = DecisionTreeClassifier(criterion='gini', max_depth=i, 
                                random_state=1)
        tree.fit(xt, yt)
        y_pred = tree.predict(xtest)
        # print('Accuracy: {:.3f}'.format(accuracy_score(yd_test, yd_pred)))
        # print('Accuracy: {:.2f}%'.format(accuracy_score(yd_test, yd_pred) * 100))
        sss = accuracy_score(ytest, y_pred) * 100
        accu = round(sss, 2)
        scorelist.append(accu)
        # Accuracy: 97.78%
        # return scorelist
    print(scorelist)

# runtree(Xd_train, yd_train, Xd_test, yd_test, 50)


 
# fit a randomforest tree model
forest1 = RandomForestClassifier(criterion='gini', n_estimators=25,
                                 random_state=1, n_jobs=2)

forest1.fit(Xd_train, yd_train)
feature_influence = forest1.feature_importances_
feature_index = np.argsort(feature_influence)[::-1]
count = 0
newlist = []

for f in range(Xd_train.shape[1]):
    if feature_influence[feature_index[f]] >= 0.0100:
        count += 1
        print("%s)  %f" % (feature_index[f], 
                            feature_influence[feature_index[f]]))
    newlist.append(feature_influence[feature_index[f]])

np.cumsum(newlist)
print(count)



yf_pred = forest1.predict(Xd_test)
print('Accuracy: {:.3f}'.format(accuracy_score(yd_test, yf_pred)))
print('Accuracy: {:.2f}%'.format(accuracy_score(yd_test, yf_pred) * 100))
# Accuracy: 96.48%



# using sklearn SelectFromModel
## to select important features
feature_select = SelectFromModel(forest1, threshold=0.1, prefit=True)
features_selected = feature_select.transform(Xd_train)

