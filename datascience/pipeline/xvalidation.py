from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

digits = load_digits()
print(digits.DESCR)
ddata = digits.data
dtarget = digits.target

ddata.shape

# svm classification
h1 = svm.LinearSVC(C=1.0, max_iter=1000000)
h2 = svm.SVC(kernel='rbf', degree=3, gamma=0.001, C=1.0)
h3 = svm.SVC(kernel='poly', degree=3, C=1.0)

# do cross validation
chosen_random_state = 1
cv_folds = 10  # Try 3, 5 or 20
eval_scoring = 'accuracy'  # Try also f1
workers = -1  # this will use all your CPU power


