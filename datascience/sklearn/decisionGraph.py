#!/usr/bin/env python

import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split as splitit
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz



iris_ = datasets.load_iris()
iris_features = iris_.data
iris_target = iris_.target
print(np.unique(iris_target))
print(iris_features[:5, :])

print(iris_.feature_names)
print(iris_.target_names)
iris_.DESCR 
print(iris_.filename)


# split the dataset for training and testing
Xi_train, Xi_test, yi_train, yi_test = splitit(iris_features, iris_target, test_size=0.3,
                                       random_state=1, stratify=iris_target)

# fit the Decision tree model
mytree1 = DecisionTreeClassifier(criterion='gini', max_depth=4, 
                            random_state=1)

mytree1.fit(Xi_train, yi_train)

# Graphing the tree structure
dot_data = export_graphviz(mytree1,
                class_names=iris_.target_names,
                feature_names=iris_.feature_names,
                out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf('tree.pdf')
graph.write_png('tree.png')