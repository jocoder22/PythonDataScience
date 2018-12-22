#!/usr/bin/env python

import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split as splitit
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz


print("######################################")

from sklearn import datasets, neighbors, linear_model

digits = datasets.load_digits()
X_digits = digits.data / digits.data.max()
y_digits = digits.target

print(X_digits.dtype)
print(y_digits.dtype)
n_samples = len(X_digits)

X_train = X_digits[:int(.9 * n_samples)]
y_train = y_digits[:int(.9 * n_samples)]

mytree3 = DecisionTreeClassifier(criterion='gini', max_depth=5, 
                            random_state=1)

mytree3.fit(X_train, y_train)


dot_data = export_graphviz(mytree3, filled=True, rounded=True,
                class_names=digits.target_names.astype('U10'),
                out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf('tree2.pdf')
graph.write_png('tree2.png')



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
dot_data = export_graphviz(mytree2, filled=True, rounded=True,
                class_names=digits.target_names.astype('U10'),
                feature_names=None,
                out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf('tree3.pdf')
graph.write_png('tree3.png')