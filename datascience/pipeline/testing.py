from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split

digits = load_digits()
print(digits.DESCR)
ddata = digits.data
dtarget = digits.target

ddata.shape

# svm classification
h1 = svm.LinearSVC(C=1.0, max_iter=1000000)
h2 = svm.SVC(kernel='rbf', degree=3, gamma=0.001, C=1.0)
h3 = svm.SVC(kernel='poly', degree=3, C=1.0)

h1.fit(ddata, dtarget)
print(h1.score(ddata, dtarget))

# split dataset and refit
chosen_random_state = 1
Xtrain, Xtest, ytrain, ytest = train_test_split(ddata, dtarget, test_size=0.40, 
                                                random_state=chosen_random_state)
print ("(Xtrain shape {}, Xtest shape {},  ytrain shape {}, \
        ytest shape {})".format (Xtrain.shape, Xtest.shape, 
                            ytrain.shape, ytest.shape))