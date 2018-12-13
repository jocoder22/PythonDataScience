from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split

digits = load_digits()
print(digits.DESCR)
ddata, dtarget = digits.data,  digits.target
ddata.shape

# set up support vector machine algorithm
hs = svm.SVC()
hprob = svm.SVC(probability=True, random_state=1)