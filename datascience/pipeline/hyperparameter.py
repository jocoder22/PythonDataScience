from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split

digits = load_digits()
print(digits.DESCR)
ddata = digits.data
dtarget = digits.target

ddata.shape