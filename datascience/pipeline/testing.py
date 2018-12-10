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
print ("(Xtrain shape {}, Xtest shape {},  \nytrain shape {}, \
        ytest shape {})".format (Xtrain.shape, Xtest.shape, 
                            ytrain.shape, ytest.shape))

h1.fit(Xtrain,ytrain)
print (h1.score(Xtest,ytest)) 

# Do validation test
chosen_random_state = 1
Xtrain, Xvalidation_test, ytrain, yvalidation_test = train_test_split(ddata, dtarget, 
                                    test_size=.40, random_state=chosen_random_state)
Xvalidation, Xtest, yvalidation, ytest = train_test_split(Xvalidation_test, yvalidation_test, 
                                test_size=.50, 
                                random_state=chosen_random_state)
print ("X train shape, {}, X validation shape {}, X test shape {}, \
        /ny train shape {}, y validation shape {}, y test shape {}/n".format
        (Xtrain.shape, Xvalidation.shape, Xtest.shape,  
        ytrain.shape, yvalidation.shape, ytest.shape))
for hypothesis in [h1, h2, h3]:
    hypothesis.fit(Xtrain,ytrain)
    print ("{} -> validation mean accuracy = {:.3f}".format(hypothesis,  
            hypothesis.score(Xvalidation,yvalidation)))

h2.fit(X_train,y_train)
print ("n{} -> test mean accuracy = {:.3f}".format(h2,   
       h2.score(Xtest,ytest)))

h3.fit(X_train,y_train)
print ("n{} -> test mean accuracy = {:.3f}".format(h3,   
       h3.score(Xtest,ytest)))