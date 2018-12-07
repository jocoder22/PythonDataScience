from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np


iris = load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, 
                                                test_size=0.50, random_state=4)

iclassifier = DecisionTreeClassifier(max_depth=2)
iclassifier.fit(xtrain, ytrain)
ypred = iclassifier.predict(xtest)
iris.target_names

# Performance metrics
c_metric = confusion_matrix(ytest, ypred)
print(c_metric)
print(classification_report(ytest, ypred,
                            target_names=iris.target_names))

# Classified wine dataset
wine = load_wine()
wtrain, wtest, wytrain, wytest = train_test_split(wine.data, wine.target,
                                                  test_size=0.50,
                                                  random_state=4)

ibclassifier = DecisionTreeClassifier(max_depth=2)
ibclassifier.fit(wtrain, wytrain)
wpred = ibclassifier.predict(wtest)
wine.target_names
list(wine.target_names)
wine.target[[20, 65, 123, 171]]

# Performance metrics
cb_metric = confusion_matrix(wytest, wpred)
print(cb_metric)
print(classification_report(wytest, wpred,
                            target_names=wine.target_names))


# Plots of metrics
# ## graph(1)
img = plt.matshow(c_metric, cmap=plt.cm.autumn)
plt.colorbar(img, fraction=0.045)
for x in range(c_metric.shape[0]):
    for y in range(c_metric.shape[1]):
        plt.text(x, y, "{:.02f}".format(c_metric[x, y]),
                 size=12, color='black', ha="center", va="center")
plt.show()


# ## graph(2)
img2 = plt.matshow(cb_metric, cmap=plt.cm.autumn)
plt.colorbar(img2, fraction=0.045)
for x in range(cb_metric.shape[0]):
    for y in range(cb_metric.shape[1]):
        plt.text(x, y, "{:.02f}".format(cb_metric[x, y]),
                 size=12, color='black', ha="center", va="center")
plt.show()