from sklearn.datasets import load_digits
from sklearn import svm

digits = load_digits()
print(digits.DESCR)
ddata = digits.data
dtarget = digits.target

ddata.shape

from sklearn import svm
h1 = svm.LinearSVC(C=1.0)
h2 = svm.SVC(kernel='rbf', degree=3, gamma=0.001, C=1.0)
h3 = svm.SVC(kernel='poly', degree=3, C=1.0)