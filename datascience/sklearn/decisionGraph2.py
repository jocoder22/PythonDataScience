#!/usr/bin/env python

import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split as splitit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz


digits = datasets.load_digits()
X_digits = digits.data / digits.data.max()
y_digits = digits.target

print(X_digits.dtype)
print(y_digits.dtype)
n_samples = len(X_digits)

X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]
X_test = X_digits[:int((.1 * n_samples)+1)]
y_test = y_digits[:int((.1 * n_samples)+1)]

mytree3 = DecisionTreeClassifier(criterion='gini', max_depth=5, 
                            random_state=1)

mytree3.fit(X_train, y_train)

score = mytree3.score(X_test,y_test)
print(score)
# 0.6666


# fit a randomforest tree model
forest = RandomForestClassifier(criterion='gini', n_estimators=50,
                                 random_state=1, n_jobs=2)

forest.fit(X_train, y_train)

fscore = forest.score(X_test,y_test)
print(fscore)
# 1.0


# display the tree
dot_data = export_graphviz(mytree3, filled=True, rounded=True,
                class_names=digits.target_names.astype('U10'),
                out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf('tree2.pdf')
graph.write_png('tree2.png')



# Using different max_depth
digits_ = datasets.load_digits()
digits_features = digits_.data
digits_target = digits_.target
print(np.unique(digits_target))
print(digits_features[:5, :])

print(digits_.images)
print(digits_.target_names)
digits_.DESCR 


Xd_train = digits_features
yd_train = digits_target

# fit the Decision tree model
mytree2 = DecisionTreeClassifier(criterion='gini', max_depth=4, 
                            random_state=1)

mytree2.fit(Xd_train, yd_train)

# Graphing the tree structure
dot_data2 = export_graphviz(mytree2, filled=True, rounded=True,
                class_names=digits.target_names.astype('U10'),
                feature_names=None,
                out_file=None)

graph = graph_from_dot_data(dot_data2)
graph.write_pdf('tree3.pdf')
graph.write_png('tree3.png')


# Using different dataset
wine = datasets.load_wine()
wine_features = wine.data
wine_target = wine.target
print(np.unique(wine_target))
print(wine_features[:5, :])

print(wine.feature_names)
print(wine.target_names)
wine.DESCR 

Xw_train = wine_features
yw_train = wine_target

# fit the Decision tree model
mytree3 = DecisionTreeClassifier(criterion='gini', max_depth=4, 
                            random_state=1)

mytree3.fit(Xw_train, yw_train)


# Graphing the tree structure
dot_data3 = export_graphviz(mytree3, filled=True, rounded=True,
                class_names=wine.target_names,
                feature_names=wine.feature_names,
                out_file=None)

graph = graph_from_dot_data(dot_data3)
graph.write_pdf('tree4.pdf')
graph.write_png('tree4.png')